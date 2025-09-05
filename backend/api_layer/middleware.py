"""
Custom middleware for FastAPI application.

This module provides middleware for logging, error handling, authentication,
and request processing.
"""

import logging
import time
import traceback
from collections.abc import Callable

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        start_time = time.time()

        # Log incoming request
        logger.info(
            f"Incoming request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"-> {response.status_code} ({process_time:.3f}s)"
            )

            # Add processing time header
            response.headers["X-Process-Time"] = str(process_time)

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"-> Error: {str(e)} ({process_time:.3f}s)"
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for global error handling."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with error handling."""
        try:
            return await call_next(request)

        except HTTPException:
            # Re-raise HTTP exceptions (handled by FastAPI)
            raise

        except Exception as e:
            # Log unexpected errors
            logger.error(
                f"Unhandled exception in {request.method} {request.url.path}: "
                f"{str(e)}\n{traceback.format_exc()}"
            )

            # Return generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Internal server error",
                    "error_code": "INTERNAL_ERROR",
                    "details": {
                        "path": str(request.url.path),
                        "method": request.method,
                    },
                },
            )


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for authentication checking on protected routes.

    This middleware checks authentication status for routes that require
    authentication. It can be configured to require specific services
    (YNAB, Gmail, or both) to be authenticated.
    """

    def __init__(self, app, protected_paths: set[str] | None = None):
        """
        Initialize authentication middleware.

        Args:
            app: FastAPI application instance
            protected_paths: Set of path patterns that require authentication
        """
        super().__init__(app)

        # Default protected paths - routes that require authentication
        self.protected_paths = protected_paths or {
            "/api/transactions",
            "/api/email-search",
            "/api/ml",
            "/api/settings",
        }

        # Paths that are always allowed (no authentication required)
        self.public_paths = {
            "/",
            "/health",
            "/api/auth",
            "/docs",
            "/redoc",
            "/openapi.json",
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check authentication for protected routes."""
        path = request.url.path

        # Skip authentication for public paths
        if self._is_public_path(path):
            return await call_next(request)

        # Skip authentication for non-protected paths
        if not self._is_protected_path(path):
            return await call_next(request)

        # Check authentication for protected paths
        try:
            # Get auth manager from app state
            auth_manager = getattr(request.app.state, "auth_manager", None)

            if not auth_manager:
                logger.warning("Auth manager not available in middleware")
                return await call_next(request)  # Allow request to proceed

            # Check if any authentication is required based on the path
            required_auth = self._get_required_auth(path)

            if "ynab" in required_auth:
                ynab_authenticated = await auth_manager.is_ynab_authenticated()
                if not ynab_authenticated:
                    return JSONResponse(
                        status_code=401,
                        content={
                            "success": False,
                            "message": "YNAB authentication required",
                            "error_code": "YNAB_AUTH_REQUIRED",
                            "details": {"required_service": "ynab"},
                        },
                    )

            if "gmail" in required_auth:
                gmail_authenticated = await auth_manager.is_email_authenticated()
                if not gmail_authenticated:
                    return JSONResponse(
                        status_code=401,
                        content={
                            "success": False,
                            "message": "Gmail authentication required",
                            "error_code": "GMAIL_AUTH_REQUIRED",
                            "details": {"required_service": "gmail"},
                        },
                    )

            # Authentication passed, proceed with request
            return await call_next(request)

        except Exception as e:
            logger.error(f"Authentication middleware error: {e}")
            # On error, allow request to proceed (fail open)
            return await call_next(request)

    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (no authentication required)."""
        for public_path in self.public_paths:
            if path.startswith(public_path):
                return True
        return False

    def _is_protected_path(self, path: str) -> bool:
        """Check if path requires authentication."""
        for protected_path in self.protected_paths:
            if path.startswith(protected_path):
                return True
        return False

    def _get_required_auth(self, path: str) -> set[str]:
        """
        Determine which authentication services are required for a path.

        Returns a set containing 'ynab' and/or 'gmail' based on the path.
        """
        required = set()

        # Transactions require YNAB authentication
        if path.startswith("/api/transactions"):
            required.add("ynab")

        # Email search requires Gmail authentication
        if path.startswith("/api/email-search"):
            required.add("gmail")

        # ML endpoints may require YNAB for training data
        if path.startswith("/api/ml"):
            if "train" in path or "predict" in path:
                required.add("ynab")

        # Settings generally don't require specific auth, but user should be authenticated
        # We'll be lenient here and not require specific services

        return required


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Don't set X-Frame-Options for OAuth callback routes to allow popup communication
        if not request.url.path.endswith("/callback"):
            response.headers["X-Frame-Options"] = "DENY"

        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response


def setup_middleware(app):
    """Setup all middleware for the FastAPI application."""
    # Add middleware in reverse order (last added is executed first)

    # Security headers (outermost)
    app.add_middleware(SecurityHeadersMiddleware)

    # Authentication middleware (before error handling)
    app.add_middleware(AuthenticationMiddleware)

    # Error handling
    app.add_middleware(ErrorHandlingMiddleware)

    # Request logging (innermost)
    app.add_middleware(LoggingMiddleware)

    logger.info("Middleware setup completed")
