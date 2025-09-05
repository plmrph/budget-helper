"""
PayeeLocation model for YNAB payee locations.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class PayeeLocation(BaseModel):
    """Model for YNAB payee locations."""

    id: UUID = Field(..., description="Unique identifier for the payee location")
    payee_id: UUID = Field(..., description="Payee this location belongs to")
    latitude: float | None = Field(None, description="Latitude coordinate")
    longitude: float | None = Field(None, description="Longitude coordinate")
    deleted: bool | None = Field(
        None, description="Whether the payee location is deleted"
    )
    created_at: datetime | None = Field(None, description="When the record was created")
    updated_at: datetime | None = Field(
        None, description="When the record was last updated"
    )

    class Config:
        """Pydantic configuration."""

        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}
