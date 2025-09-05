"""
History Handler implementation for tracking and managing entity changes.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from uuid import uuid4

from models.history import HistoryEntry
from resource_layer.database_store_access.database_resource_access import (
    DatabaseResourceAccess,
)
from thrift_gen.entities.ttypes import EntityType, Transaction

logger = logging.getLogger(__name__)


class HistoryHandler(ABC):
    """Abstract base class for history tracking and management."""

    @abstractmethod
    async def record_change(
        self,
        entity_type: EntityType,
        entity_id: str,
        field_name: str,
        old_value: Any,
        new_value: Any,
        user_action: str,
    ) -> str:
        """Record a change in history."""
        pass

    @abstractmethod
    async def get_history(
        self,
        entity_type: EntityType | None = None,
        entity_id: str | None = None,
        limit: int = 50,
    ) -> list[HistoryEntry]:
        """Get change history with optional filters."""
        pass

    @abstractmethod
    async def undo_change(self, history_id: str) -> bool:
        """Undo a specific change."""
        pass

    @abstractmethod
    async def clear_history(
        self,
        entity_type: EntityType | None = None,
        older_than: datetime | None = None,
    ) -> int:
        """Clear history with optional filters."""
        pass


class DatabaseHistoryHandler(HistoryHandler):
    """Database-backed implementation of history tracking."""

    def __init__(self, db: DatabaseResourceAccess, ynab_resource=None):
        """
        Initialize the database history handler.

        Args:
            db: Database resource access instance
            ynab_resource: Optional YNAB resource access for undo sync
        """
        self.db = db
        self.ynab_resource = ynab_resource

    async def record_change(
        self,
        entity_type: EntityType,
        entity_id: str,
        field_name: str,
        old_value: Any,
        new_value: Any,
        user_action: str,
    ) -> str:
        """
        Record a change in history.

        Args:
            entity_type: Type of entity that was changed
            entity_id: ID of the entity that was changed
            field_name: Name of the field that was changed
            old_value: Previous value of the field
            new_value: New value of the field
            user_action: Description of the user action that caused the change

        Returns:
            ID of the created history entry
        """
        try:
            history_id = str(uuid4())

            # Convert values to JSON for storage
            old_value_json = json.dumps(old_value) if old_value is not None else None
            new_value_json = json.dumps(new_value) if new_value is not None else None

            # Determine if this change can be undone
            can_undo = self._can_undo_change(entity_type, field_name)

            command = """
                INSERT INTO history_entries
                (id, entity_type, entity_id, field_name, old_value, new_value, timestamp, user_action, can_undo)
                VALUES (:id, :entity_type, :entity_id, :field_name, :old_value, :new_value, :timestamp, :user_action, :can_undo)
            """

            params = {
                "id": history_id,
                "entity_type": entity_type.value,
                "entity_id": entity_id,
                "field_name": field_name,
                "old_value": old_value_json,
                "new_value": new_value_json,
                "timestamp": datetime.utcnow(),
                "user_action": user_action,
                "can_undo": can_undo,
            }

            await self.db.execute_command(command, params)

            logger.info(
                f"Recorded history change: {entity_type.value}/{entity_id}/{field_name}"
            )
            return history_id

        except Exception as e:
            logger.error(f"Failed to record history change: {e}")
            raise

    async def get_history(
        self,
        entity_type: EntityType | None = None,
        entity_id: str | None = None,
        limit: int = 50,
    ) -> list[HistoryEntry]:
        """
        Get change history with optional filters.

        Args:
            entity_type: Optional filter by entity type
            entity_id: Optional filter by entity ID
            limit: Maximum number of entries to return

        Returns:
            List of history entries
        """
        try:
            # Build query with optional filters
            where_conditions = []
            params = {"limit": limit}

            if entity_type:
                where_conditions.append("entity_type = :entity_type")
                params["entity_type"] = entity_type.value

            if entity_id:
                where_conditions.append("entity_id = :entity_id")
                params["entity_id"] = entity_id

            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)

            query = f"""
                SELECT id, entity_type, entity_id, field_name, old_value, new_value,
                       timestamp, user_action, can_undo, undone, undo_timestamp
                FROM history_entries
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT :limit
            """

            rows = await self.db.execute_query(query, params)

            # Convert rows to HistoryEntry objects
            history_entries = []
            for row in rows:
                # Parse JSON values
                old_value = json.loads(row["old_value"]) if row["old_value"] else None
                new_value = json.loads(row["new_value"]) if row["new_value"] else None

                entry = HistoryEntry(
                    id=str(row["id"]),
                    entity_type=EntityType(row["entity_type"]),
                    entity_id=row["entity_id"],
                    field_name=row["field_name"],
                    old_value=old_value,
                    new_value=new_value,
                    timestamp=row["timestamp"],
                    user_action=row["user_action"],
                    can_undo=row["can_undo"],
                    undone=row["undone"],
                    undo_timestamp=row["undo_timestamp"],
                )
                history_entries.append(entry)

            logger.info(f"Retrieved {len(history_entries)} history entries")
            return history_entries

        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            raise

    async def undo_change(self, history_id: str) -> bool:
        """
        Undo a specific change.

        Args:
            history_id: ID of the history entry to undo

        Returns:
            True if undo was successful, False otherwise
        """
        try:
            # Get the history entry
            query = """
                SELECT id, entity_type, entity_id, field_name, old_value, new_value,
                       timestamp, user_action, can_undo, undone, undo_timestamp
                FROM history_entries
                WHERE id = :history_id
            """

            rows = await self.db.execute_query(query, {"history_id": history_id})

            if not rows:
                logger.warning(f"History entry not found: {history_id}")
                return False

            row = rows[0]

            # Check if change can be undone
            if not row["can_undo"]:
                logger.warning(f"History entry cannot be undone: {history_id}")
                return False

            # Check if already undone
            if row["undone"]:
                logger.warning(f"History entry already undone: {history_id}")
                return False

            # Parse the old value
            old_value = json.loads(row["old_value"]) if row["old_value"] else None
            entity_type = EntityType(row["entity_type"])
            entity_id = row["entity_id"]
            field_name = row["field_name"]

            # Perform the undo operation
            success = await self._perform_undo(
                entity_type, entity_id, field_name, old_value
            )

            if success:
                # Mark the history entry as undone
                update_command = """
                    UPDATE history_entries
                    SET undone = true, undo_timestamp = :undo_timestamp
                    WHERE id = :history_id
                """

                await self.db.execute_command(
                    update_command,
                    {"history_id": history_id, "undo_timestamp": datetime.utcnow()},
                )

                # Record the undo action as a new history entry
                await self.record_change(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    field_name=field_name,
                    old_value=json.loads(row["new_value"])
                    if row["new_value"]
                    else None,
                    new_value=old_value,
                    user_action=f"Undo: {row['user_action']}",
                )

                logger.info(f"Successfully undid history change: {history_id}")
                return True
            else:
                logger.error(f"Failed to perform undo operation for: {history_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to undo change: {e}")
            return False

    async def clear_history(
        self,
        entity_type: EntityType | None = None,
        older_than: datetime | None = None,
    ) -> int:
        """
        Clear history with optional filters.

        Args:
            entity_type: Optional filter by entity type
            older_than: Optional filter by timestamp

        Returns:
            Number of entries cleared
        """
        try:
            # Build delete query with optional filters
            where_conditions = []
            params = {}

            if entity_type:
                where_conditions.append("entity_type = :entity_type")
                params["entity_type"] = entity_type.value

            if older_than:
                where_conditions.append("timestamp < :older_than")
                params["older_than"] = older_than

            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)

            command = f"DELETE FROM history_entries {where_clause}"

            affected_rows = await self.db.execute_command(command, params)

            logger.info(f"Cleared {affected_rows} history entries")
            return affected_rows

        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            raise

    def _can_undo_change(self, entity_type: EntityType, field_name: str) -> bool:
        """
        Determine if a change can be undone.

        Args:
            entity_type: Type of entity
            field_name: Name of the field that was changed

        Returns:
            True if the change can be undone, False otherwise
        """
        # Define which changes can be undone
        undoable_changes = {
            EntityType.TRANSACTION: {
                "category_id",
                "payee_name",
                "memo",
                "approved",
                "cleared",
            },
            EntityType.SETTINGS: {"email_search", "email_automation", "display", "ai"},
        }

        entity_undoable = undoable_changes.get(entity_type, set())
        return field_name in entity_undoable

    async def _perform_undo(
        self, entity_type: EntityType, entity_id: str, field_name: str, old_value: Any
    ) -> bool:
        """
        Perform the actual undo operation.

        Args:
            entity_type: Type of entity
            entity_id: ID of the entity
            field_name: Name of the field to revert
            old_value: Value to revert to

        Returns:
            True if undo was successful, False otherwise
        """
        try:
            if entity_type == EntityType.TRANSACTION:
                return await self._undo_transaction_change(
                    entity_id, field_name, old_value
                )
            elif entity_type == EntityType.SETTINGS:
                return await self._undo_settings_change(
                    entity_id, field_name, old_value
                )
            else:
                logger.warning(f"Undo not implemented for entity type: {entity_type}")
                return False

        except Exception as e:
            logger.error(f"Failed to perform undo: {e}")
            return False

    async def _undo_transaction_change(
        self, transaction_id: str, field_name: str, old_value: Any
    ) -> bool:
        """
        Undo a transaction change.

        Args:
            transaction_id: ID of the transaction
            field_name: Name of the field to revert
            old_value: Value to revert to

        Returns:
            True if undo was successful, False otherwise
        """
        try:
            # Update the transaction in the database
            command = f"""
                UPDATE transactions
                SET {field_name} = :old_value, updated_at = :updated_at
                WHERE id = :transaction_id
            """

            params = {
                "old_value": old_value,
                "updated_at": datetime.utcnow(),
                "transaction_id": transaction_id,
            }

            affected_rows = await self.db.execute_command(command, params)

            if affected_rows == 0:
                logger.warning(f"Transaction not found for undo: {transaction_id}")
                return False

            # If YNAB resource is available, sync the change
            if self.ynab_resource:
                try:
                    # Get the updated transaction
                    query = """
                        SELECT id, account_id, category_id, payee_name, memo, amount,
                               date, approved, cleared, deleted
                        FROM transactions
                        WHERE id = :transaction_id
                    """

                    rows = await self.db.execute_query(
                        query, {"transaction_id": transaction_id}
                    )

                    if rows:
                        row = rows[0]
                        # Create transaction object for YNAB sync
                        Transaction(
                            id=row["id"],
                            account_id=row["account_id"],
                            category_id=row["category_id"],
                            payee_name=row["payee_name"],
                            memo=row["memo"],
                            amount=row["amount"],
                            date=row["date"],
                            approved=row["approved"],
                            cleared=row["cleared"],
                            deleted=row["deleted"],
                        )

                        # Sync with YNAB (assuming we have budget_id available)
                        # Note: In a real implementation, we'd need to get the budget_id
                        # For now, we'll skip YNAB sync and just log
                        logger.info(
                            f"Would sync transaction {transaction_id} with YNAB"
                        )

                except Exception as e:
                    logger.warning(f"Failed to sync undo with YNAB: {e}")
                    # Don't fail the undo if YNAB sync fails

            return True

        except Exception as e:
            logger.error(f"Failed to undo transaction change: {e}")
            return False

    async def _undo_settings_change(
        self, settings_id: str, field_name: str, old_value: Any
    ) -> bool:
        """
        Undo a settings change.

        Args:
            settings_id: ID of the settings record (usually 'app_settings')
            field_name: Name of the field to revert
            old_value: Value to revert to

        Returns:
            True if undo was successful, False otherwise
        """
        try:
            # Get the most recent settings record (since settings_id is usually 'app_settings')
            query = "SELECT id, settings_data FROM app_settings ORDER BY updated_at DESC LIMIT 1"
            rows = await self.db.execute_query(query)

            if not rows:
                logger.warning("No settings found for undo")
                return False

            actual_settings_id = rows[0]["id"]
            current_settings_json = rows[0]["settings_data"]

            # Parse the JSON settings data
            if isinstance(current_settings_json, str):
                current_settings = json.loads(current_settings_json)
            else:
                current_settings = current_settings_json

            # Update the specific field
            current_settings[field_name] = old_value

            # Save back to database
            command = """
                UPDATE app_settings
                SET settings_data = :settings_data, updated_at = :updated_at
                WHERE id = :settings_id
            """

            params = {
                "settings_data": json.dumps(current_settings),
                "updated_at": datetime.utcnow(),
                "settings_id": actual_settings_id,
            }

            affected_rows = await self.db.execute_command(command, params)

            if affected_rows > 0:
                logger.info(
                    f"Successfully undid settings change for field: {field_name}"
                )
                return True
            else:
                logger.warning(
                    f"No settings record updated for undo: {actual_settings_id}"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to undo settings change: {e}")
            return False
