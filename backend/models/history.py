"""
History tracking data model.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from thrift_gen.entities.ttypes import EntityType


class HistoryEntry(BaseModel):
    """History entry model for tracking changes to entities."""

    id: str = Field(..., description="Unique history entry identifier")
    entity_type: EntityType = Field(..., description="Type of entity that was changed")
    entity_id: str = Field(..., description="ID of the entity that was changed")
    field_name: str = Field(..., description="Name of the field that was changed")
    old_value: Any = Field(..., description="Previous value of the field")
    new_value: Any = Field(..., description="New value of the field")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the change occurred"
    )
    user_action: str = Field(
        ..., description="Description of the user action that caused the change"
    )
    can_undo: bool = Field(
        default=True, description="Whether this change can be undone"
    )
    undone: bool = Field(
        default=False, description="Whether this change has been undone"
    )
    undo_timestamp: datetime | None = Field(
        None, description="When this change was undone"
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}
        json_schema_extra = {
            "example": {
                "id": "history_123",
                "entity_type": "transaction",
                "entity_id": "trans_456",
                "field_name": "category_id",
                "old_value": "category_old",
                "new_value": "category_new",
                "timestamp": "2024-01-15T10:30:00Z",
                "user_action": "Manual category change",
                "can_undo": True,
                "undone": False,
                "undo_timestamp": None,
            }
        }
