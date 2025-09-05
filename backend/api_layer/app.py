"""
FastAPI application setup and configuration.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from api_layer.routes import (
    auth,
    budgets,
    email,
    email_search,
    health,
    ml,
    root,
    settings,
    transactions,
)
from resource_layer.database_store_access.database import (
    shutdown_database,
    startup_database,
)

from .dependencies import cleanup_managers, initialize_managers
from .exceptions import setup_exception_handlers
from .middleware import setup_middleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    try:
        logger.info("Starting application...")

        # Initialize database
        await startup_database()
        logger.info("Database initialized")

        # Initialize managers
        await initialize_managers()
        logger.info("Managers initialized")

        logger.info("Application startup completed")

    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise

    yield

    # Shutdown
    try:
        logger.info("Shutting down application...")

        # Cleanup managers
        await cleanup_managers()
        logger.info("Managers cleaned up")

        # Shutdown database
        await shutdown_database()
        logger.info("Database shutdown")

        logger.info("Application shutdown completed")

    except Exception as e:
        logger.error(f"Application shutdown failed: {e}")


def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Budget Helper API",
        version="1.0.0",
        description="""
        Backend API for Budget Helper application.

        ## Features

        * **Authentication**: Manage YNAB and Gmail authentication
        * **Transactions**: CRUD operations for YNAB transactions
        * **Email Integration**: Search and link email receipts
        * **AI/ML**: Automated transaction categorization
        * **Settings**: Application configuration management
        * **History**: Track and undo transaction changes

        ## Authentication

        This API integrates with external services:
        - YNAB API using Personal Access Tokens
        - Gmail API using OAuth 2.0

        ## Data Models

        All monetary amounts are stored in milliunits (e.g., $1.00 = 1000).
        Transaction states include: cleared, uncleared, reconciled.
        """,
        routes=app.routes,
    )

    # Add custom info
    openapi_schema["info"]["contact"] = {
        "name": "Budget Helper",
        "url": "http://localhost:3000",
    }

    openapi_schema["info"]["license"] = {
        "name": "MIT",
    }

    # Add server information
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Development server"}
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    # Determine if we're in debug mode
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"

    app = FastAPI(
        title="Budget Helper API",
        description="Backend API for Budget Helper application",
        version="1.0.0",
        lifespan=lifespan,
        debug=debug_mode,
        docs_url="/docs" if debug_mode else None,
        redoc_url="/redoc" if debug_mode else None,
    )

    # Set custom OpenAPI schema
    app.openapi = custom_openapi

    # Setup exception handlers
    setup_exception_handlers(app)

    # Setup middleware (order matters - added in reverse execution order)
    setup_middleware(app)

    # Configure CORS
    # Note: In production, requests come through nginx proxy (same origin)
    # In development, we need CORS for direct frontend->backend communication
    allowed_origins = [
        "http://localhost",  # nginx proxy (main access point)
        "http://localhost:80",  # nginx proxy explicit port
        "http://localhost:3000",  # Frontend development (direct access)
        "http://frontend:3000",  # Docker frontend (internal)
    ]

    # Add production origins if not in debug mode
    if not debug_mode:
        production_origin = os.getenv("FRONTEND_URL")
        if production_origin:
            allowed_origins.append(production_origin)

    # In production through nginx, CORS might not be needed since it's same-origin
    # But we configure it for development and flexibility
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time"],
        # Allow preflight requests
        max_age=600,
    )

    # Include routers

    # Core routes
    app.include_router(health.router)
    app.include_router(root.router)

    # Feature routes (some are placeholders for future implementation)
    app.include_router(email.router)  # Currently implemented for testing
    app.include_router(auth.router)  # Placeholder - task 15
    app.include_router(transactions.router)  # Placeholder - task 16
    app.include_router(email_search.router)  # Placeholder - task 17
    app.include_router(ml.router)  # Placeholder - task 18
    app.include_router(settings.router)  # Placeholder - task 19
    app.include_router(budgets.router)  # Budget management

    logger.info(f"FastAPI application created (debug={debug_mode})")
    return app


# Create the app instance
app = create_app()
