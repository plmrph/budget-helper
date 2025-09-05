"""
Dependency injection for FastAPI application.

This module provides dependency injection for the new service architecture,
ensuring proper initialization and lifecycle management.
"""

import logging

from fastapi import Depends, HTTPException

from business_layer.metadata_finding_engine import MetadataFindingEngine
from business_layer.ml_engine import MLEngine
from business_layer.prediction_manager import PredictionManager
from business_layer.transaction_manager import TransactionManager
from configs import ConfigService, get_config_service
from resource_layer.budgeting_platform_access.budgeting_platform_access import (
    BudgetingPlatformAccess,
)
from resource_layer.budgeting_platform_access.ynab_api_client import YnabApiClient
from resource_layer.database_store_access.database import get_database
from resource_layer.database_store_access.database_resource_access import (
    DatabaseResourceAccess,
)
from resource_layer.metadata_source_access.gmail_api_client import get_gmail_client
from resource_layer.metadata_source_access.metadata_source_access import (
    MetadataSourceAccess,
)

logger = logging.getLogger(__name__)

# Global service instances (initialized on startup)
# Following The Method: ResourceAccess → Engines → Managers → Utilities
_database_store_access: DatabaseResourceAccess | None = None
_budgeting_platform_access: BudgetingPlatformAccess | None = None
_metadata_source_access: MetadataSourceAccess | None = None
_ml_engine: MLEngine | None = None
_metadata_finding_engine: MetadataFindingEngine | None = None
_transaction_manager: TransactionManager | None = None
_prediction_manager: PredictionManager | None = None
_configs: ConfigService | None = None


async def initialize_managers():
    """Initialize all service instances during application startup."""
    global _database_store_access, _budgeting_platform_access, _metadata_source_access
    global _ml_engine, _metadata_finding_engine, _transaction_manager
    global _prediction_manager, _configs

    try:
        logger.info("Initializing services...")

        # Layer 4: Resources - Get database instance
        db = await get_database()

        # Initialize external API clients (gracefully handle missing credentials)
        try:
            ynab_client = YnabApiClient()
            logger.info("YNAB client initialized")
        except Exception as e:
            logger.warning(
                f"YNAB client initialization failed (will continue without YNAB): {e}"
            )
            ynab_client = None

        # Initialize Gmail client with config service
        # We need to initialize config service first for Gmail client
        db = await get_database()
        temp_database_store_access = DatabaseResourceAccess(database_resource=db)
        temp_config_service = get_config_service(
            database_store_access=temp_database_store_access
        )

        gmail_client = get_gmail_client(config_service=temp_config_service)
        logger.info("Gmail API client initialized with config service")

        # Layer 3: ResourceAccess - Initialize resource access services
        logger.info("Initializing ResourceAccess layer...")
        _database_store_access = DatabaseResourceAccess(database_resource=db)

        # Utilities: Initialize utility services first (needed by engines and managers)
        logger.info("Initializing Utility services...")
        _configs = get_config_service(database_store_access=_database_store_access)

        # Continue ResourceAccess initialization with config service
        _budgeting_platform_access = BudgetingPlatformAccess(
            ynab_api_client=ynab_client,
            database_store_access=_database_store_access,
            config_service=_configs,
        )
        _metadata_source_access = MetadataSourceAccess(
            gmail_api_client=gmail_client, database_store_access=_database_store_access
        )

        # Layer 2b: Engines - Initialize business activity engines
        logger.info("Initializing Engine layer...")
        model_storage_path = "/app/ml_models"
        _ml_engine = MLEngine(
            database_store_access=_database_store_access,
            model_storage_path=model_storage_path,
        )
        _metadata_finding_engine = MetadataFindingEngine(
            metadata_source_access=_metadata_source_access,
            database_store_access=_database_store_access,
            config_service=_configs,
        )

        # Layer 2a: Managers - Initialize business orchestration managers
        logger.info("Initializing Manager layer...")
        _transaction_manager = TransactionManager(
            metadata_finding_engine=_metadata_finding_engine,
            ml_engine=_ml_engine,
            budgeting_platform_access=_budgeting_platform_access,
            metadata_source_access=_metadata_source_access,
            database_store_access=_database_store_access,
            config_service=_configs,
        )
        _prediction_manager = PredictionManager(
            ml_engine=_ml_engine,
            database_store_access=_database_store_access,
            config_service=_configs,
        )

        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


async def cleanup_managers():
    """Cleanup service instances during application shutdown."""
    global _database_store_access, _budgeting_platform_access, _metadata_source_access
    global _ml_engine, _metadata_finding_engine, _transaction_manager
    global _prediction_manager, _configs

    try:
        logger.info("Cleaning up services...")

        # Cleanup in reverse order (top-down)
        # Managers first
        _transaction_manager = None
        _prediction_manager = None
        _configs = None

        # Then Engines
        _ml_engine = None
        _metadata_finding_engine = None

        # Then ResourceAccess
        _database_store_access = None
        _budgeting_platform_access = None
        _metadata_source_access = None

        # Utilities cleanup complete

        logger.info("Service cleanup completed")

    except Exception as e:
        logger.error(f"Failed to cleanup services: {e}")

async def get_transaction_manager() -> TransactionManager:
    """Dependency to get the TransactionManager instance."""
    if _transaction_manager is None:
        logger.error("TransactionManager not initialized")
        raise HTTPException(status_code=500, detail="Transaction service not available")
    return _transaction_manager


async def get_prediction_manager() -> PredictionManager:
    """Dependency to get the PredictionManager instance."""
    if _prediction_manager is None:
        logger.error("PredictionManager not initialized")
        raise HTTPException(status_code=500, detail="Prediction service not available")
    return _prediction_manager


async def get_configs() -> ConfigService:
    """Dependency to get the Configs service instance."""
    if _configs is None:
        logger.error("Configs service not initialized")
        raise HTTPException(
            status_code=500, detail="Configuration service not available"
        )
    return _configs


# Engine dependencies (for direct access when needed)
async def get_ml_engine() -> MLEngine:
    """Dependency to get the MLEngine instance."""
    if _ml_engine is None:
        logger.error("MLEngine not initialized")
        raise HTTPException(status_code=500, detail="ML engine not available")
    return _ml_engine


async def get_metadata_finding_engine() -> MetadataFindingEngine:
    """Dependency to get the MetadataFindingEngine instance."""
    if _metadata_finding_engine is None:
        logger.error("MetadataFindingEngine not initialized")
        raise HTTPException(
            status_code=500, detail="Metadata finding engine not available"
        )
    return _metadata_finding_engine


# ResourceAccess dependencies (for direct access when needed)
async def get_database_store_access() -> DatabaseResourceAccess:
    """Dependency to get the DatabaseResourceAccess instance."""
    if _database_store_access is None:
        logger.error("DatabaseResourceAccess not initialized")
        raise HTTPException(status_code=500, detail="Local store access not available")
    return _database_store_access


# Dependency aliases for easier use in routes
TransactionManagerDep = Depends(get_transaction_manager)
PredictionManagerDep = Depends(get_prediction_manager)
ConfigsDep = Depends(get_configs)

# Engine dependencies (use sparingly - prefer Manager dependencies)
MLEngineDep = Depends(get_ml_engine)
MetadataFindingEngineDep = Depends(get_metadata_finding_engine)


# ResourceAccess dependencies (use sparingly - prefer Manager dependencies)
async def get_budgeting_platform_access() -> BudgetingPlatformAccess:
    """Dependency to get the BudgetingPlatformAccess instance."""
    if _budgeting_platform_access is None:
        logger.error("BudgetingPlatformAccess not initialized")
        raise HTTPException(
            status_code=500, detail="Budgeting platform access not available"
        )
    return _budgeting_platform_access


async def get_metadata_source_access() -> MetadataSourceAccess:
    """Dependency to get the MetadataSourceAccess instance."""
    if _metadata_source_access is None:
        logger.error("MetadataSourceAccess not initialized")
        raise HTTPException(
            status_code=500, detail="Metadata source access not available"
        )
    return _metadata_source_access


DatabaseStoreAccessDep = Depends(get_database_store_access)
BudgetingPlatformAccessDep = Depends(get_budgeting_platform_access)
MetadataSourceAccessDep = Depends(get_metadata_source_access)
