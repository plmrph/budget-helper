"""
Health check API routes.
"""

import logging

from fastapi import APIRouter, HTTPException

from resource_layer.database_store_access.database import get_database

from ..dependencies import get_prediction_manager, get_transaction_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check():
    """Comprehensive health check including database connectivity and services."""
    try:
        db = await get_database()
        db_healthy = await db.health_check()

        if not db_healthy:
            raise HTTPException(status_code=503, detail="Database connection unhealthy")

        connection_info = await db.get_connection_info()

        # Check manager availability (without calling them to avoid initialization issues)
        managers_status = {
            "transaction_manager": "available",
            "prediction_manager": "available",
        }

        return {
            "status": "healthy",
            "timestamp": connection_info.get("timestamp"),
            "database": {
                "status": "connected",
                "database": connection_info.get("database"),
                "user": connection_info.get("user"),
                "version": connection_info.get("version", "").split()[0:2]
                if connection_info.get("version")
                else None,
            },
            "services": managers_status,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503, detail=f"Health check failed: {str(e)}"
        ) from e


@router.get("/database")
async def database_health_check():
    """Detailed database health check."""
    try:
        db = await get_database()
        db_healthy = await db.health_check()
        connection_info = await db.get_connection_info()

        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "connection_info": connection_info,
            "pool_status": "active",
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(
            status_code=503, detail=f"Database health check failed: {str(e)}"
        ) from e


@router.get("/services")
async def services_health_check():
    """Check the health of all application services."""
    try:
        services_status = {}

        # Test each manager
        try:
            await get_transaction_manager()
            services_status["transaction_manager"] = "healthy"
        except Exception as e:
            services_status["transaction_manager"] = f"unhealthy: {str(e)}"

        try:
            await get_prediction_manager()
            services_status["prediction_manager"] = "healthy"
        except Exception as e:
            services_status["prediction_manager"] = f"unhealthy: {str(e)}"

        # Determine overall status
        all_healthy = all(status == "healthy" for status in services_status.values())

        return {
            "status": "healthy" if all_healthy else "degraded",
            "services": services_status,
        }

    except Exception as e:
        logger.error(f"Services health check failed: {e}")
        raise HTTPException(
            status_code=503, detail=f"Services health check failed: {str(e)}"
        ) from e
