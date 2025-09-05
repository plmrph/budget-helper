"""
Custom exception handlers for FastAPI application.

This module provides structured error handling with consistent response formats.
"""

import logging
from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class AppException(Exception):
    """Base application exception with structured error information."""

    def __init__(
        self,
        message: str,
        error_code: str = None,
        details: dict[str, Any] = None,
        status_code: int = 400,
    ):
        self.message = message
        self.error_code = error_code or "APP_ERROR"
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)


class AuthenticationError(AppException):
    """Authentication-related errors."""

    def __init__(
        self, message: str = "Authentication failed", details: dict[str, Any] = None
    ):
        super().__init__(
            message=message, error_code="AUTH_ERROR", details=details, status_code=401
        )


class AuthorizationError(AppException):
    """Authorization-related errors."""

    def __init__(self, message: str = "Access denied", details: dict[str, Any] = None):
        super().__init__(
            message=message, error_code="AUTHZ_ERROR", details=details, status_code=403
        )


class ValidationError(AppException):
    """Data validation errors."""

    def __init__(
        self, message: str = "Validation failed", details: dict[str, Any] = None
    ):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details,
            status_code=422,
        )


class ExternalServiceError(AppException):
    """External service integration errors."""

    def __init__(
        self, service: str, message: str = None, details: dict[str, Any] = None
    ):
        message = message or f"{service} service error"
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={**(details or {}), "service": service},
            status_code=502,
        )


class StorageError(AppException):
    """Storage/database errors."""

    def __init__(
        self, message: str = "Storage operation failed", details: dict[str, Any] = None
    ):
        super().__init__(
            message=message,
            error_code="STORAGE_ERROR",
            details=details,
            status_code=500,
        )


class NotFoundError(AppException):
    """Resource not found errors."""

    def __init__(
        self, resource: str, resource_id: str = None, details: dict[str, Any] = None
    ):
        message = f"{resource} not found"
        if resource_id:
            message += f": {resource_id}"

        super().__init__(
            message=message,
            error_code="NOT_FOUND",
            details={
                **(details or {}),
                "resource": resource,
                "resource_id": resource_id,
            },
            status_code=404,
        )


def create_error_response(
    message: str,
    error_code: str = "ERROR",
    details: dict[str, Any] = None,
    status_code: int = 400,
) -> dict[str, Any]:
    """Create standardized error response."""
    return {
        "success": False,
        "message": message,
        "error_code": error_code,
        "details": details or {},
    }


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle custom application exceptions."""
    logger.warning(
        f"Application exception in {request.method} {request.url.path}: "
        f"{exc.error_code} - {exc.message}"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            message=exc.message, error_code=exc.error_code, details=exc.details
        ),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    logger.warning(
        f"HTTP exception in {request.method} {request.url.path}: "
        f"{exc.status_code} - {exc.detail}"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            message=str(exc.detail),
            error_code="HTTP_ERROR",
            details={"status_code": exc.status_code},
        ),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors."""
    logger.warning(
        f"Validation error in {request.method} {request.url.path}: "
        f"{len(exc.errors())} validation errors"
    )

    # Format validation errors
    validation_details = []
    for error in exc.errors():
        validation_details.append(
            {
                "field": ".".join(str(x) for x in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            }
        )

    return JSONResponse(
        status_code=422,
        content=create_error_response(
            message="Request validation failed",
            error_code="VALIDATION_ERROR",
            details={"validation_errors": validation_details},
        ),
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(
        f"Unhandled exception in {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {str(exc)}",
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content=create_error_response(
            message="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"path": str(request.url.path), "method": request.method},
        ),
    )


def setup_exception_handlers(app):
    """Setup all exception handlers for the FastAPI application."""

    # Custom application exceptions
    app.add_exception_handler(AppException, app_exception_handler)

    # FastAPI HTTP exceptions
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)

    # Request validation errors
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # General exception handler (catch-all)
    app.add_exception_handler(Exception, general_exception_handler)

    logger.info("Exception handlers setup completed")
