"""
DatabaseResourceStrategy interface and implementations.

This module defines the strategy interface for database resource access
and provides concrete implementations for each database type.
"""

import json
import logging
import traceback
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

# Removed Thrift serialization imports - using simple JSON instead
from configs import get_config_service
from thrift_gen.databasestoreaccess.ttypes import (
    Filter,
    FilterOperator,
    Query,
    QueryResult,
    SortDirection,
)
from thrift_gen.entities.ttypes import (
    Account,
    Budget,
    Category,
    ConfigItem,
    ConfigType,
    ConfigValue,
    Entity,
    EntityType,
    ExternalSystem,
    FileEntity,
    Metadata,
    MetadataValue,
    ModelCard,
    ModelType,
    Payee,
    TrainingStatus,
    Transaction,
)
from thrift_gen.exceptions.ttypes import (
    InternalException,
    ValidationException,
)

logger = logging.getLogger(__name__)


def thrift_to_json(thrift_object):
    """Convert a Thrift object to JSON string using simple dict serialization."""
    if thrift_object is None:
        return None
    try:
        # Use simple dict-based serialization instead of Thrift's JSON protocol
        if hasattr(thrift_object, "__dict__"):
            # Filter out None values and private attributes
            obj_dict = {
                k: v
                for k, v in thrift_object.__dict__.items()
                if v is not None and not k.startswith("_")
            }
            return json.dumps(obj_dict)
        return None
    except Exception as e:
        logger.error(f"Failed to serialize Thrift object {type(thrift_object)}: {e}")
        return None


def json_to_thrift(json_str, thrift_class):
    """Convert a JSON string back to a Thrift object using simple dict deserialization."""
    try:
        if not json_str:
            return None

        data = json.loads(json_str)
        if not isinstance(data, dict):
            return None

        # Create instance and set attributes
        instance = thrift_class()
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance
    except Exception as e:
        logger.error(f"Failed to deserialize JSON to {thrift_class}: {e}")
        return None


class DatabaseResourceStrategy(ABC):
    """Abstract interface for database resource strategies."""

    @abstractmethod
    async def upsert_entities(self, entities: list[Entity]) -> list[Entity]:
        """Upsert entities in database."""
        pass

    @abstractmethod
    async def delete_entities(
        self, entity_type: EntityType, entity_ids: list[str]
    ) -> list[str]:
        """Delete entities from database."""
        pass

    @abstractmethod
    async def get_entities_by_id(
        self, entity_type: EntityType, entity_ids: list[str]
    ) -> list[Entity]:
        """Get entities by ID from database."""
        pass

    @abstractmethod
    async def get_entities(self, query: Query) -> QueryResult:
        """Query entities from database."""
        pass


class PostgreSQLResourceStrategy(DatabaseResourceStrategy):
    """PostgreSQL implementation of DatabaseResourceStrategy."""

    def __init__(self, database_client=None):
        self.db_client = database_client
        self._table_mapping = {
            1: "transactions",  # EntityType.Transaction
            2: "metadata",  # EntityType.Metadata
            3: "config_items",  # EntityType.ConfigItem
            4: "model_info",  # EntityType.ModelCard
            5: "accounts",  # EntityType.Account
            6: "categories",  # EntityType.Category
            7: "payees",  # EntityType.Payee
            8: "budgets",  # EntityType.Budget
            9: "file_entities",  # EntityType.FileEntity
        }
        # Primary key mapping for tables that don't use 'id'
        self._primary_key_mapping = {
            3: "key",  # EntityType.ConfigItem
            4: "name",  # EntityType.ModelCard
            9: "path",  # EntityType.FileEntity
        }

    async def upsert_entities(self, entities: list[Entity]) -> list[Entity]:
        """Upsert entities in PostgreSQL."""
        try:
            if not self.db_client:
                raise InternalException("Database client not configured")

            upserted_entities = []

            for entity in entities:
                try:
                    upserted_entity = await self._upsert_single_entity(entity)
                    upserted_entities.append(upserted_entity)
                except Exception as e:
                    logger.error(f"Error upserting entity: {e}")
                    logger.error(f"Entity type: {type(entity)}")
                    logger.error(f"Entity data: {entity}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    raise InternalException(f"Failed to upsert entity: {str(e)}") from e

            logger.info(f"Upserted {len(upserted_entities)} entities")
            return upserted_entities

        except Exception as e:
            logger.error(f"Error in upsert_entities: {e}")
            raise InternalException(f"Database upsert failed: {str(e)}") from e

    async def delete_entities(
        self, entity_type: EntityType, entity_ids: list[str]
    ) -> list[str]:
        """Delete entities from PostgreSQL."""
        try:
            if not self.db_client:
                raise InternalException("Database client not configured")

            if not entity_ids:
                return []

            table_name = self._get_table_name(entity_type)
            primary_key = self._get_primary_key(entity_type)
            deleted_ids = []

            for entity_id in entity_ids:
                try:
                    query = f"DELETE FROM {table_name} WHERE {primary_key} = :id"
                    params = {"id": entity_id}

                    affected_rows = await self.db_client.execute_command(query, params)

                    if affected_rows > 0:
                        deleted_ids.append(entity_id)

                except Exception as e:
                    logger.warning(f"Error deleting entity {entity_id}: {e}")
                    continue

            logger.info(f"Deleted {len(deleted_ids)} entities of type {entity_type}")
            return deleted_ids

        except Exception as e:
            logger.error(f"Error in delete_entities: {e}")
            raise InternalException(f"Database delete failed: {str(e)}") from e

    async def get_entities_by_id(
        self, entity_type: EntityType, entity_ids: list[str]
    ) -> list[Entity]:
        """Get entities by ID from PostgreSQL."""
        try:
            if not self.db_client:
                raise InternalException("Database client not configured")

            if not entity_ids:
                return []

            table_name = self._get_table_name(entity_type)
            primary_key = self._get_primary_key(entity_type)
            entities = []

            # Build IN clause for multiple IDs
            placeholders = [f":id_{i}" for i in range(len(entity_ids))]
            query = f"SELECT * FROM {table_name} WHERE {primary_key} IN ({', '.join(placeholders)})"

            params = {f"id_{i}": entity_id for i, entity_id in enumerate(entity_ids)}

            result = await self.db_client.execute_query(query, params)

            for row in result:
                if entity_type == EntityType.Transaction:
                    # For transactions, fetch with metadata
                    entity = await self._fetch_transaction_with_metadata(str(row["id"]))
                else:
                    entity = await self._deserialize_entity(entity_type, row)
                entities.append(entity)

            logger.info(f"Retrieved {len(entities)} entities of type {entity_type}")
            return entities

        except Exception as e:
            logger.error(f"Error in get_entities_by_id: {e}")
            raise InternalException(f"Database get failed: {str(e)}") from e

    async def get_entities(self, query: Query) -> QueryResult:
        """Query entities from PostgreSQL."""
        try:
            if not self.db_client:
                raise InternalException("Database client not configured")

            table_name = self._get_table_name(query.entityType)

            # Build SQL query
            sql_query = f"SELECT * FROM {table_name}"
            params = {}
            where_clauses = []

            # Add filters
            if query.filters:
                for i, filter_item in enumerate(query.filters):
                    clause, filter_params = self._build_filter_clause(
                        filter_item, i, query.entityType
                    )
                    where_clauses.append(clause)
                    params.update(filter_params)

            if where_clauses:
                sql_query += f" WHERE {' AND '.join(where_clauses)}"

            # Add sorting
            if query.sort:
                order_clauses = []
                for sort_item in query.sort:
                    direction = (
                        "ASC" if sort_item.direction == SortDirection.ASC else "DESC"
                    )
                    order_clauses.append(f"{sort_item.fieldName} {direction}")
                sql_query += f" ORDER BY {', '.join(order_clauses)}"
            else:
                # Default ordering
                sql_query += " ORDER BY created_at DESC"

            # Add pagination
            if query.limit:
                sql_query += " LIMIT :limit"
                params["limit"] = query.limit

            if query.offset:
                sql_query += " OFFSET :offset"
                params["offset"] = query.offset

            # Execute query
            result = await self.db_client.execute_query(sql_query, params)

            # Convert results to entities
            entities = []
            for row in result:
                if query.entityType == EntityType.Transaction:
                    # For transactions, fetch with metadata
                    entity = await self._fetch_transaction_with_metadata(str(row["id"]))
                else:
                    entity = await self._deserialize_entity(query.entityType, row)
                entities.append(entity)

            # Get total count for pagination
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            if where_clauses:
                count_query += f" WHERE {' AND '.join(where_clauses)}"

            count_params = {
                k: v for k, v in params.items() if not k.startswith(("limit", "offset"))
            }
            count_result = await self.db_client.execute_query(count_query, count_params)
            total_count = count_result[0]["count"] if count_result else 0

            # Calculate pagination info
            page_size = query.limit or len(entities)
            page_number = (
                (query.offset // page_size) + 1 if query.offset and page_size > 0 else 1
            )

            return QueryResult(
                entities=entities,
                totalCount=total_count,
                pageNumber=page_number,
                pageSize=page_size,
            )

        except Exception as e:
            logger.error(f"Error in get_entities: {e}")
            raise InternalException(f"Database query failed: {str(e)}") from e

    def _get_table_name(self, entity_type: EntityType) -> str:
        """Get database table name for entity type."""
        table_name = self._table_mapping.get(entity_type)
        if table_name is None:
            raise InternalException(f"Unknown entity type: {entity_type}")
        return table_name

    def _get_primary_key(self, entity_type: EntityType) -> str:
        """Get primary key column name for entity type."""
        return self._primary_key_mapping.get(entity_type, "id")

    async def _upsert_single_entity(self, entity: Entity) -> Entity:
        """Upsert a single entity."""
        # Determine entity type and extract data
        entity_type, entity_data = self._extract_entity_data(entity)

        # Special handling for transactions with metadata
        if entity_type == EntityType.Transaction and entity_data.metadata:
            return await self._upsert_transaction_with_metadata(entity_data)

        table_name = self._get_table_name(entity_type)

        # Serialize entity data for database
        serialized_data = await self._serialize_entity_data(entity_data, entity_type)

        # Add timestamps (abstracted from business layer)
        now = datetime.utcnow()
        if "created_at" not in serialized_data:
            serialized_data["created_at"] = now
        serialized_data["updated_at"] = now

        # Build upsert query (PostgreSQL UPSERT)
        columns = list(serialized_data.keys())
        placeholders = [f":{col}" for col in columns]

        # Get primary key for conflict resolution
        primary_key = self._get_primary_key(entity_type)

        # Create conflict resolution for upsert
        update_clauses = [
            f"{col} = EXCLUDED.{col}" for col in columns if col != primary_key
        ]

        query = f"""
            INSERT INTO {table_name} ({", ".join(columns)})
            VALUES ({", ".join(placeholders)})
            ON CONFLICT ({primary_key}) DO UPDATE SET
            {", ".join(update_clauses)}
            RETURNING *
        """

        result = await self.db_client.execute_query(query, serialized_data)

        if result:
            return await self._deserialize_entity(entity_type, result[0])
        else:
            raise InternalException("Failed to upsert entity")

    async def _upsert_transaction_with_metadata(
        self, transaction: Transaction
    ) -> Entity:
        """Upsert a transaction and its metadata separately."""
        # Extract metadata and create a transaction without metadata
        metadata_list = transaction.metadata
        transaction_without_metadata = Transaction(
            id=transaction.id,
            date=transaction.date,
            amount=transaction.amount,
            approved=transaction.approved,
            platformType=transaction.platformType,
            payeeId=transaction.payeeId,
            categoryId=transaction.categoryId,
            accountId=transaction.accountId,
            budgetId=transaction.budgetId,
            memo=transaction.memo,
            metadata=None,  # Remove metadata for transaction upsert
        )

        # Upsert the transaction first (without metadata)
        transaction_entity = Entity(transaction=transaction_without_metadata)
        await self._upsert_transaction_only(transaction_entity)

        # Upsert each metadata item separately (upsert will handle existing ones)
        if metadata_list:
            for metadata_item in metadata_list:
                # Add transaction_id to metadata
                metadata_with_transaction_id = Metadata(
                    id=metadata_item.id,
                    type=metadata_item.type,
                    value=metadata_item.value,
                    sourceSystem=metadata_item.sourceSystem,
                    description=metadata_item.description,
                )
                metadata_entity = Entity(metadata=metadata_with_transaction_id)
                await self._upsert_metadata_with_transaction_id(
                    metadata_entity, transaction.id
                )

        # Fetch the complete transaction with metadata for return
        return await self._fetch_transaction_with_metadata(transaction.id)

    async def _upsert_transaction_only(self, entity: Entity) -> Entity:
        """Upsert just the transaction part without metadata."""
        entity_type, entity_data = self._extract_entity_data(entity)
        table_name = self._get_table_name(entity_type)

        # Serialize entity data for database
        serialized_data = await self._serialize_entity_data(entity_data, entity_type)

        # Add timestamps
        now = datetime.utcnow()
        if "created_at" not in serialized_data:
            serialized_data["created_at"] = now
        serialized_data["updated_at"] = now

        # Build upsert query
        columns = list(serialized_data.keys())
        placeholders = [f":{col}" for col in columns]
        primary_key = self._get_primary_key(entity_type)
        update_clauses = [
            f"{col} = EXCLUDED.{col}" for col in columns if col != primary_key
        ]

        query = f"""
            INSERT INTO {table_name} ({", ".join(columns)})
            VALUES ({", ".join(placeholders)})
            ON CONFLICT ({primary_key}) DO UPDATE SET
            {", ".join(update_clauses)}
            RETURNING *
        """

        result = await self.db_client.execute_query(query, serialized_data)
        if result:
            return await self._deserialize_entity(entity_type, result[0])
        else:
            raise InternalException("Failed to upsert transaction")

    async def _upsert_metadata_with_transaction_id(
        self, metadata_entity: Entity, transaction_id: str
    ):
        """Upsert metadata with transaction_id."""
        entity_type, entity_data = self._extract_entity_data(metadata_entity)

        # Generate ID if not provided
        metadata_id = entity_data.id if entity_data.id else str(uuid.uuid4())

        # Convert Thrift enum to database enum string
        from thrift_gen.entities.ttypes import MetadataType

        metadata_type_str = "Email"  # Default
        if entity_data.type == MetadataType.Email:
            metadata_type_str = "Email"
        elif entity_data.type == MetadataType.Prediction:
            metadata_type_str = "Prediction"

        # Serialize metadata using manual approach for better reliability
        value_json = thrift_to_json(entity_data.value)

        # Manual serialization for ExternalSystem union to avoid Thrift serialization issues
        source_system_json = None
        if entity_data.sourceSystem:
            if (
                hasattr(entity_data.sourceSystem, "emailPlatformType")
                and entity_data.sourceSystem.emailPlatformType is not None
            ):
                source_system_json = json.dumps(
                    {"emailPlatformType": entity_data.sourceSystem.emailPlatformType}
                )
            elif (
                hasattr(entity_data.sourceSystem, "budgetingPlatformType")
                and entity_data.sourceSystem.budgetingPlatformType is not None
            ):
                source_system_json = json.dumps(
                    {
                        "budgetingPlatformType": entity_data.sourceSystem.budgetingPlatformType
                    }
                )
            elif (
                hasattr(entity_data.sourceSystem, "modelType")
                and entity_data.sourceSystem.modelType is not None
            ):
                source_system_json = json.dumps(
                    {"modelType": entity_data.sourceSystem.modelType}
                )
            else:
                # Fallback to Thrift serialization
                source_system_json = thrift_to_json(entity_data.sourceSystem)

        serialized_data = {
            "id": metadata_id,
            "transaction_id": transaction_id,
            "type": metadata_type_str,
            "value": value_json,
            "source_system": source_system_json,
            "description": entity_data.description,
        }

        # Add timestamps
        now = datetime.utcnow()
        if "created_at" not in serialized_data:
            serialized_data["created_at"] = now
        serialized_data["updated_at"] = now

        # Build upsert query
        columns = list(serialized_data.keys())
        placeholders = [f":{col}" for col in columns]

        query = f"""
            INSERT INTO metadata ({", ".join(columns)})
            VALUES ({", ".join(placeholders)})
            ON CONFLICT (id) DO UPDATE SET
            transaction_id = EXCLUDED.transaction_id,
            type = EXCLUDED.type,
            value = EXCLUDED.value,
            source_system = EXCLUDED.source_system,
            description = EXCLUDED.description,
            updated_at = EXCLUDED.updated_at
        """

        await self.db_client.execute_query(query, serialized_data)

    # _clear_transaction_metadata method removed - upserts handle metadata properly

    async def _fetch_transaction_with_metadata(self, transaction_id: str) -> Entity:
        """Fetch a transaction with its metadata."""
        # Fetch transaction
        transaction_query = "SELECT * FROM transactions WHERE id = :transaction_id"
        transaction_result = await self.db_client.execute_query(
            transaction_query, {"transaction_id": transaction_id}
        )

        if not transaction_result:
            raise InternalException(
                f"Transaction {transaction_id} not found after upsert"
            )

        # Fetch metadata
        metadata_query = "SELECT * FROM metadata WHERE transaction_id = :transaction_id"
        metadata_results = await self.db_client.execute_query(
            metadata_query, {"transaction_id": transaction_id}
        )

        # Deserialize transaction
        transaction_data = transaction_result[0]
        transaction = self._deserialize_transaction_data(transaction_data)

        # Deserialize metadata
        metadata_list = []
        if metadata_results:
            for metadata_row in metadata_results:
                metadata_obj = self._deserialize_metadata_data(metadata_row)
                metadata_list.append(metadata_obj)

        # Add metadata to transaction
        transaction.metadata = metadata_list

        return Entity(transaction=transaction)

    def _deserialize_transaction_data(self, row_data: dict) -> Transaction:
        """Deserialize transaction data from database row."""
        # Convert database row to Transaction object
        return Transaction(
            id=str(row_data["id"]),
            date=row_data["date"].isoformat() if row_data["date"] else None,
            amount=row_data["amount"],
            approved=row_data["approved"],
            platformType=row_data["platform_type"],
            payeeId=str(row_data["payee_id"]) if row_data["payee_id"] else None,
            categoryId=str(row_data["category_id"])
            if row_data["category_id"]
            else None,
            accountId=str(row_data["account_id"]) if row_data["account_id"] else None,
            budgetId=str(row_data["budget_id"]) if row_data["budget_id"] else None,
            memo=row_data["memo"],
            metadata=[],  # Will be populated separately
        )

    def _deserialize_metadata_data(self, row_data: dict) -> Metadata:
        """Deserialize metadata data from database row."""
        # Convert database enum string back to Thrift enum
        from thrift_gen.entities.ttypes import MetadataType

        metadata_type = MetadataType.Email  # Default
        if row_data["type"] == "Email":
            metadata_type = MetadataType.Email
        elif row_data["type"] == "Prediction":
            metadata_type = MetadataType.Prediction

        # Deserialize Thrift objects from JSON
        value = (
            json_to_thrift(row_data["value"], MetadataValue)
            if row_data["value"]
            else None
        )

        # Manual deserialization for ExternalSystem union
        source_system = None
        if row_data["source_system"]:
            try:
                source_data = json.loads(row_data["source_system"])
                if "emailPlatformType" in source_data:
                    source_system = ExternalSystem(
                        emailPlatformType=source_data["emailPlatformType"]
                    )
                elif "budgetingPlatformType" in source_data:
                    source_system = ExternalSystem(
                        budgetingPlatformType=source_data["budgetingPlatformType"]
                    )
                elif "modelType" in source_data:
                    source_system = ExternalSystem(modelType=source_data["modelType"])
                else:
                    # Fallback to Thrift deserialization
                    source_system = json_to_thrift(
                        row_data["source_system"], ExternalSystem
                    )
            except Exception as e:
                logger.warning(f"Failed to deserialize source_system: {e}")
                source_system = None

        return Metadata(
            id=row_data["id"],
            type=metadata_type,
            value=value,
            sourceSystem=source_system,
            description=row_data["description"],
        )

    def _extract_entity_data(self, entity: Entity) -> tuple:
        """Extract entity type and data from Entity union."""
        if entity.transaction:
            return EntityType.Transaction, entity.transaction
        elif entity.metadata:
            return EntityType.Metadata, entity.metadata
        elif entity.configItem:
            return EntityType.ConfigItem, entity.configItem
        elif entity.modelCard:
            return EntityType.ModelCard, entity.modelCard
        elif entity.account:
            return EntityType.Account, entity.account
        elif entity.category:
            return EntityType.Category, entity.category
        elif entity.payee:
            return EntityType.Payee, entity.payee
        elif entity.budget:
            return EntityType.Budget, entity.budget
        elif entity.file:
            return EntityType.FileEntity, entity.file
        else:
            raise ValidationException("Unknown entity type in union")

    async def _serialize_entity_data(
        self, entity_data, entity_type: EntityType
    ) -> dict[str, Any]:
        """Serialize entity data for database storage."""
        # Convert entity to dictionary
        if hasattr(entity_data, "__dict__"):
            data = entity_data.__dict__.copy()
        else:
            # Handle Thrift objects
            data = {}
            for attr_name in dir(entity_data):
                if not attr_name.startswith("_"):
                    attr_value = getattr(entity_data, attr_name)
                    if not callable(attr_value):
                        data[attr_name] = attr_value

        # Map Thrift field names to database column names
        field_mapping = self._get_field_mapping(entity_type)
        mapped_data = {}
        for thrift_field, db_column in field_mapping.items():
            if thrift_field in data:
                mapped_data[db_column] = data[thrift_field]

        # Handle special data types and enum conversions
        for key, value in mapped_data.items():
            if key == "metadata" and isinstance(value, list):
                # Special handling for metadata - it's a list of Thrift Metadata objects
                if value:
                    # Serialize each Metadata object using Thrift serialization
                    serialized_metadata = [
                        thrift_to_json(metadata_obj) for metadata_obj in value
                    ]
                    mapped_data[key] = json.dumps(serialized_metadata)
                else:
                    mapped_data[key] = None
            elif isinstance(value, list) and value:
                # Regular list, serialize as JSON
                mapped_data[key] = json.dumps(value)
            elif isinstance(value, list):
                # Empty list
                mapped_data[key] = None
            elif isinstance(value, dict):
                # Serialize dicts as JSON
                mapped_data[key] = json.dumps(value) if value else None
            elif hasattr(value, "read") and hasattr(value, "write"):
                # It's a Thrift object, use Thrift serialization
                mapped_data[key] = thrift_to_json(value) if value else None
            elif hasattr(value, "__dict__"):
                # Serialize other nested objects as JSON
                mapped_data[key] = json.dumps(value.__dict__) if value else None
            elif key == "platform_type" and isinstance(value, int):
                # Convert Thrift enum integer to database enum string
                from thrift_gen.entities.ttypes import BudgetingPlatformType

                if value == BudgetingPlatformType.YNAB:
                    mapped_data[key] = "YNAB"
                else:
                    mapped_data[key] = "YNAB"  # Default fallback
            elif key == "date" and isinstance(value, str):
                # Convert ISO date string to date object
                from datetime import datetime

                try:
                    if "T" in value:
                        # Full datetime string, extract date part
                        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                        mapped_data[key] = dt.date()
                    else:
                        # Just date string
                        mapped_data[key] = datetime.fromisoformat(value).date()
                except ValueError:
                    # If parsing fails, keep as string and let database handle it
                    mapped_data[key] = value

        # Special handling for ConfigItem
        if entity_type == EntityType.ConfigItem:
            if "value" in mapped_data:
                config_value = mapped_data["value"]
                if hasattr(config_value, "__dict__"):
                    # Convert ConfigValue union to JSON
                    mapped_data["value"] = json.dumps(config_value.__dict__)

            # Convert ConfigType enum to string for database
            if "type" in mapped_data:
                config_type = mapped_data["type"]
                if isinstance(config_type, int):
                    type_name = ConfigType._VALUES_TO_NAMES.get(config_type)
                    if type_name is None:
                        logger.warning(
                            f"Unknown ConfigType value: {config_type}, defaulting to System"
                        )
                        type_name = "System"
                    mapped_data["type"] = type_name

        # Special handling for ModelCard
        elif entity_type == EntityType.ModelCard:
            # Convert ModelType enum to string for database
            if "model_type" in mapped_data:
                model_type = mapped_data["model_type"]
                if isinstance(model_type, int):
                    # Get default from ConfigService
                    config_service = get_config_service()
                    default_model_type = await config_service.getDefaultModelType()

                    if model_type == ModelType.PXBlendSC:
                        mapped_data["model_type"] = "PXBlendSC"
                    else:
                        # Use enum name for default
                        mapped_data["model_type"] = default_model_type

            # Convert TrainingStatus enum to string for database
            if "status" in mapped_data:
                status = mapped_data["status"]
                if isinstance(status, int):
                    status_names = {
                        TrainingStatus.Scheduled: "Scheduled",
                        TrainingStatus.Pending: "Pending",
                        TrainingStatus.Success: "Success",
                        TrainingStatus.Fail: "Fail",
                    }
                    mapped_data["status"] = status_names.get(status, "Pending")

        return mapped_data

    def _get_field_mapping(self, entity_type: EntityType) -> dict[str, str]:
        """Get mapping from Thrift field names to database column names."""
        if entity_type == EntityType.Transaction:
            return {
                "id": "id",
                "date": "date",
                "amount": "amount",
                "approved": "approved",
                "platformType": "platform_type",
                "payeeId": "payee_id",  # Store actual payee ID
                "categoryId": "category_id",
                "accountId": "account_id",
                "budgetId": "budget_id",
                "memo": "memo",
            }
        elif entity_type == EntityType.Account:
            return {
                "id": "id",
                "name": "name",
                "type": "type",
                "platformType": "platform_type",
                "institution": "institution",
                "currency": "currency",
                "balance": "balance",
                "status": "status",
                "budgetId": "budget_id",
            }
        elif entity_type == EntityType.Budget:
            return {
                "id": "id",
                "name": "name",
                "currency": "currency",
                "platformType": "platform_type",
                "totalAmount": "total_amount",
                "startDate": "start_date",
                "endDate": "end_date",
            }
        elif entity_type == EntityType.Category:
            return {
                "id": "id",
                "name": "name",
                "platformType": "platform_type",
                "description": "description",
                "isIncomeCategory": "is_income_category",
                "budgetId": "budget_id",
            }
        elif entity_type == EntityType.Payee:
            return {
                "id": "id",
                "name": "name",
                "platformType": "platform_type",
                "description": "description",
                "budgetId": "budget_id",
            }
        elif entity_type == EntityType.Metadata:
            return {
                "id": "id",
                "transactionId": "transaction_id",
                "type": "type",
                "value": "value",
                "sourceSystem": "source_system",
                "description": "description",
            }
        elif entity_type == EntityType.ConfigItem:
            return {
                "key": "key",
                "type": "type",
                "value": "value",
                "description": "description",
            }
        elif entity_type == EntityType.ModelCard:
            return {
                "name": "name",
                "modelType": "model_type",
                "version": "version",
                "description": "description",
                "status": "status",
                "trainedDate": "trained_date",
                "performanceMetrics": "performance_metrics",
            }
        else:
            # Default mapping - use field names as-is
            return {}

    async def _deserialize_entity(
        self, entity_type: EntityType, row_data: dict[str, Any]
    ) -> Entity:
        """Deserialize database row to Entity."""
        # Map database column names back to Thrift field names
        field_mapping = self._get_field_mapping(entity_type)
        reverse_mapping = {
            db_col: thrift_field for thrift_field, db_col in field_mapping.items()
        }

        # Handle JSON fields and filter out database-specific fields
        processed_data = {}
        for key, value in row_data.items():
            # Skip database-specific timestamp fields (abstracted from business layer)
            if key in ["created_at", "updated_at"]:
                continue

            # Map database column to Thrift field name
            thrift_field = reverse_mapping.get(key, key)

            if isinstance(value, str) and key == "metadata":
                # Special handling for metadata - deserialize list of Thrift Metadata objects
                try:
                    metadata_json_list = json.loads(value)
                    if isinstance(metadata_json_list, list):
                        metadata_objects = []
                        for metadata_json in metadata_json_list:
                            if isinstance(metadata_json, str):
                                # It's a JSON string of a Thrift object
                                metadata_obj = json_to_thrift(metadata_json, Metadata)
                                metadata_objects.append(metadata_obj)
                        processed_data[thrift_field] = metadata_objects
                    else:
                        processed_data[thrift_field] = []
                except (json.JSONDecodeError, TypeError):
                    processed_data[thrift_field] = []
            elif isinstance(value, str) and key in [
                "performanceMetrics",
                "value",
            ]:
                try:
                    processed_data[thrift_field] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    processed_data[thrift_field] = value
            else:
                # Convert UUID objects to strings for Thrift compatibility
                if hasattr(value, "__class__") and "UUID" in str(type(value)):
                    processed_data[thrift_field] = str(value)
                else:
                    processed_data[thrift_field] = value

        # Special handling for ConfigItem
        if entity_type == EntityType.ConfigItem and "value" in processed_data:
            value_data = processed_data["value"]
            if isinstance(value_data, dict):
                # Reconstruct ConfigValue from dict
                config_value = ConfigValue(**value_data)
                processed_data["value"] = config_value

        # Create entity based on type, filtering out fields that don't exist in Thrift structs
        if entity_type == EntityType.Transaction:
            # Filter to only include fields that Transaction accepts (no metadata)
            valid_fields = {
                "id",
                "date",
                "amount",
                "approved",
                "platformType",
                "payeeId",
                "categoryId",
                "accountId",
                "budgetId",
                "memo",
            }
            filtered_data = {
                k: v for k, v in processed_data.items() if k in valid_fields
            }
            transaction = Transaction(**filtered_data)
            return Entity(transaction=transaction)
        elif entity_type == EntityType.Budget:
            # Filter to only include fields that Budget accepts
            valid_fields = {
                "id",
                "name",
                "currency",
                "platformType",
                "totalAmount",
                "startDate",
                "endDate",
            }
            filtered_data = {
                k: v for k, v in processed_data.items() if k in valid_fields
            }
            budget = Budget(**filtered_data)
            return Entity(budget=budget)
        elif entity_type == EntityType.Account:
            # Filter to only include fields that Account accepts
            valid_fields = {
                "id",
                "name",
                "type",
                "platformType",
                "institution",
                "currency",
                "balance",
                "status",
                "budgetId",
            }
            filtered_data = {
                k: v for k, v in processed_data.items() if k in valid_fields
            }
            account = Account(**filtered_data)
            return Entity(account=account)
        elif entity_type == EntityType.Category:
            # Filter to only include fields that Category accepts
            valid_fields = {
                "id",
                "name",
                "platformType",
                "description",
                "isIncomeCategory",
                "budgetId",
            }
            filtered_data = {
                k: v for k, v in processed_data.items() if k in valid_fields
            }
            category = Category(**filtered_data)
            return Entity(category=category)
        elif entity_type == EntityType.Payee:
            # Filter to only include fields that Payee accepts
            valid_fields = {"id", "name", "platformType", "description", "budgetId"}
            filtered_data = {
                k: v for k, v in processed_data.items() if k in valid_fields
            }
            payee = Payee(**filtered_data)
            return Entity(payee=payee)
        elif entity_type == EntityType.Metadata:
            metadata = Metadata(**processed_data)
            return Entity(metadata=metadata)
        elif entity_type == EntityType.ConfigItem:
            config = ConfigItem(**processed_data)
            return Entity(configItem=config)
        elif entity_type == EntityType.ModelCard:
            # Convert string enums back to Thrift enums for ModelCard
            if "modelType" in processed_data and isinstance(
                processed_data["modelType"], str
            ):
                # Get default from ConfigService
                config_service = get_config_service()
                default_model_type = await config_service.getDefaultModelType()

                # Use enum value comparison instead of hardcoded string
                model_type_name = processed_data["modelType"]
                if model_type_name == ModelType._VALUES_TO_NAMES.get(
                    ModelType.PXBlendSC
                ):
                    processed_data["modelType"] = ModelType.PXBlendSC
                else:
                    processed_data["modelType"] = default_model_type

            if "status" in processed_data and isinstance(processed_data["status"], str):
                from thrift_gen.entities.ttypes import TrainingStatus

                status_map = {
                    "Scheduled": TrainingStatus.Scheduled,
                    "Pending": TrainingStatus.Pending,
                    "Success": TrainingStatus.Success,
                    "Fail": TrainingStatus.Fail,
                }
                processed_data["status"] = status_map.get(
                    processed_data["status"], TrainingStatus.Pending
                )

            # Filter to only include fields that ModelCard accepts
            valid_fields = {
                "modelType",
                "name",
                "version",
                "description",
                "status",
                "trainedDate",
                "performanceMetrics",
            }
            filtered_data = {
                k: v for k, v in processed_data.items() if k in valid_fields
            }
            model_info = ModelCard(**filtered_data)
            return Entity(modelCard=model_info)
        elif entity_type == EntityType.FileEntity:
            file_entity = FileEntity(**processed_data)
            return Entity(file=file_entity)
        else:
            raise InternalException(f"Unknown entity type: {entity_type}")

    def get_entity_timestamps(
        self, entity_type: EntityType, entity_id: str
    ) -> dict[str, Any]:
        """Get timestamp information for an entity (created_at, updated_at)."""
        try:
            table_name = self._get_table_name(entity_type)
            query = f"SELECT created_at, updated_at FROM {table_name} WHERE id = :id"
            params = {"id": entity_id}

            result = self.db_client.execute_query(query, params)

            if result:
                return {
                    "created_at": result[0]["created_at"],
                    "updated_at": result[0]["updated_at"],
                }
            else:
                return {}

        except Exception as e:
            logger.error(f"Error getting timestamps for {entity_type} {entity_id}: {e}")
            return {}

    def _build_filter_clause(
        self, filter_item: Filter, index: int, entity_type: EntityType = None
    ) -> tuple:
        """Build SQL WHERE clause from filter."""
        field_name = filter_item.fieldName
        operator = filter_item.operator
        value = filter_item.value

        # Map Thrift field name to database column name if entity type is provided
        if entity_type:
            field_mapping = self._get_field_mapping(entity_type)
            db_column_name = field_mapping.get(field_name, field_name)
        else:
            # Fallback to using field name as-is
            db_column_name = field_name

        param_name = f"filter_{index}"
        params = {}

        # Extract value based on type
        if value.stringValue is not None:
            filter_value = value.stringValue
            # Special handling for date fields - convert string to date object
            if db_column_name == "date" and isinstance(filter_value, str):
                try:
                    from datetime import datetime

                    filter_value = datetime.fromisoformat(filter_value).date()
                except ValueError:
                    # If parsing fails, keep as string and let database handle it
                    pass
        elif value.intValue is not None:
            filter_value = value.intValue
        elif value.doubleValue is not None:
            filter_value = value.doubleValue
        elif value.boolValue is not None:
            filter_value = value.boolValue
        elif value.timestampValue is not None:
            filter_value = value.timestampValue
        else:
            raise ValidationException(
                f"No value provided in filter for field {field_name}"
            )

        # Build clause based on operator
        if operator == FilterOperator.EQ:
            clause = f"{db_column_name} = :{param_name}"
            params[param_name] = filter_value
        elif operator == FilterOperator.NEQ:
            clause = f"{db_column_name} != :{param_name}"
            params[param_name] = filter_value
        elif operator == FilterOperator.GT:
            clause = f"{db_column_name} > :{param_name}"
            params[param_name] = filter_value
        elif operator == FilterOperator.GTE:
            clause = f"{db_column_name} >= :{param_name}"
            params[param_name] = filter_value
        elif operator == FilterOperator.LT:
            clause = f"{db_column_name} < :{param_name}"
            params[param_name] = filter_value
        elif operator == FilterOperator.LTE:
            clause = f"{db_column_name} <= :{param_name}"
            params[param_name] = filter_value
        elif operator == FilterOperator.LIKE:
            clause = f"{db_column_name} ILIKE :{param_name}"
            params[param_name] = f"%{filter_value}%"
        elif operator == FilterOperator.IN:
            # Handle IN operator (value should be a list)
            if isinstance(filter_value, list):
                placeholders = [f":{param_name}_{i}" for i in range(len(filter_value))]
                clause = f"{db_column_name} IN ({', '.join(placeholders)})"
                for i, val in enumerate(filter_value):
                    params[f"{param_name}_{i}"] = val
            else:
                clause = f"{db_column_name} = :{param_name}"
                params[param_name] = filter_value
        else:
            raise ValidationException(f"Unsupported filter operator: {operator}")

        return clause, params
