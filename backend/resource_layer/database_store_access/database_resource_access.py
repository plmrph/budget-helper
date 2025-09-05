"""
DatabaseResourceAccess implementation.

This ResourceAccess service provides atomic business verbs for database operations,
implementing the DatabaseStoreAccess Thrift service interface.
Uses strategy pattern for different database backends.
"""

import logging

from thrift_gen.databasestoreaccess.DatabaseStoreAccess import (
    Iface as DatabaseStoreAccessIface,
)
from thrift_gen.databasestoreaccess.ttypes import Query, QueryResult
from thrift_gen.entities.ttypes import Entity, EntityType
from thrift_gen.exceptions.ttypes import (
    InternalException,
)

from .database_resource_strategy import (
    DatabaseResourceStrategy,
    PostgreSQLResourceStrategy,
)

logger = logging.getLogger(__name__)


class DatabaseResourceAccess(DatabaseStoreAccessIface):
    """
    DatabaseResourceAccess implements atomic business verbs for database operations.

    This ResourceAccess service converts business operations to database I/O operations
    while exposing atomic business verbs rather than CRUD operations.
    Uses strategy pattern to handle different database backends.
    """

    def __init__(self, database_resource=None, storage_type: str = "postgresql"):
        """
        Initialize DatabaseResourceAccess.

        Args:
            database_resource: Database resource access instance
            storage_type: Type of storage backend (default: postgresql)
        """
        self.db = database_resource
        self.storage_type = storage_type

        # Initialize database strategy
        self.strategy: DatabaseResourceStrategy = PostgreSQLResourceStrategy(
            database_client=database_resource
        )

        logger.info(f"DatabaseResourceAccess initialized with {storage_type} strategy")

    async def upsertEntities(self, entities: list[Entity]) -> list[Entity]:
        """
        Upsert entities using database strategy.

        Args:
            entities: List of entities to upsert

        Returns:
            List of upserted entities
        """
        try:
            logger.info(f"Upserting {len(entities)} entities")
            return await self.strategy.upsert_entities(entities)

        except Exception as e:
            logger.error(f"Error upserting entities: {e}")
            raise InternalException(f"Failed to upsert entities: {str(e)}") from e

    async def deleteEntities(
        self, entityType: EntityType, entityIds: list[str]
    ) -> list[str]:
        """
        Delete entities using database strategy.

        Args:
            entityType: Type of entities to delete
            entityIds: List of entity IDs to delete

        Returns:
            List of deleted entity IDs
        """
        try:
            logger.info(f"Deleting {len(entityIds)} entities of type {entityType}")
            return await self.strategy.delete_entities(entityType, entityIds)

        except Exception as e:
            logger.error(f"Error deleting entities: {e}")
            raise InternalException(f"Failed to delete entities: {str(e)}") from e

    async def getEntitiesById(
        self, entityType: EntityType, entityIds: list[str]
    ) -> list[Entity]:
        """
        Get entities by ID using database strategy.

        Args:
            entityType: Type of entities to get
            entityIds: List of entity IDs to get

        Returns:
            List of entities
        """
        try:
            logger.info(f"Getting {len(entityIds)} entities of type {entityType} by ID")
            return await self.strategy.get_entities_by_id(entityType, entityIds)

        except Exception as e:
            logger.error(f"Error getting entities by ID: {e}")
            raise InternalException(f"Failed to get entities by ID: {str(e)}") from e

    async def getEntities(self, query: Query) -> QueryResult:
        """
        Query entities using database strategy.

        Args:
            query: Query parameters

        Returns:
            Query result with entities
        """
        try:
            logger.info(f"Querying entities of type {query.entityType}")
            return await self.strategy.get_entities(query)

        except Exception as e:
            logger.error(f"Error querying entities: {e}")
            raise InternalException(f"Failed to query entities: {str(e)}") from e
