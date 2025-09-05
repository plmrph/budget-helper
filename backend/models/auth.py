"""
Authentication models for the YNAB application.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AuthToken(BaseModel):
    """Authentication token model."""

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat() if v else None}}

    service: str = Field(..., description="Service name (ynab, gmail)")
    token_type: str = Field(
        ...,
        description="Token type (access_token, refresh_token, personal_access_token)",
    )
    token_value: str = Field(..., description="The actual token value")
    expires_at: datetime | None = Field(None, description="Token expiration time")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Token creation time"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Token last update time"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional token metadata"
    )


class YnabAuthData(BaseModel):
    """YNAB authentication data."""

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat() if v else None}}

    personal_access_token: str = Field(..., description="YNAB personal access token")
    is_authenticated: bool = Field(default=False, description="Authentication status")
    user_id: str | None = Field(None, description="YNAB user ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EmailAuthData(BaseModel):
    """Email service authentication data."""

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat() if v else None}}

    access_token: str = Field(..., description="Email service access token")
    refresh_token: str | None = Field(None, description="Email service refresh token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_at: datetime | None = Field(None, description="Token expiration time")
    scope: str | None = Field(None, description="Granted scopes")
    is_authenticated: bool = Field(default=False, description="Authentication status")
    user_email: str | None = Field(None, description="User's email address")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AuthStatus(BaseModel):
    """Authentication status for a service."""

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat() if v else None}}

    service: str = Field(..., description="Service name")
    is_authenticated: bool = Field(..., description="Authentication status")
    expires_at: datetime | None = Field(None, description="Token expiration time")
    user_info: dict[str, Any] = Field(
        default_factory=dict, description="User information"
    )
    error: str | None = Field(None, description="Last authentication error")


class AuthError(Exception):
    """Authentication error exception."""

    def __init__(self, message: str, service: str, error_code: str | None = None):
        self.message = message
        self.service = service
        self.error_code = error_code
        super().__init__(self.message)
