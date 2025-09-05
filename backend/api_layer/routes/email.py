"""
Email API routes for testing email resource access.
"""

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.email import Email
from resource_layer.metadata_source_access.email_resource_interface import (
    EmailAuthConfig,
)

router = APIRouter(prefix="/api/email", tags=["email"])


class EmailSearchRequest(BaseModel):
    """Request model for email search."""

    query: str = "receipt"
    days_back: int = 7
    max_results: int = 20


class AuthUrlResponse(BaseModel):
    """Response model for auth URL."""

    auth_url: str


class EmailTestResponse(BaseModel):
    """Response model for email tests."""

    message: str
    authenticated: bool
    client_type: str


@router.get("/test", response_model=EmailTestResponse)
async def test_email_client():
    """Test email client instantiation and basic functionality."""
    # Temporarily disabled due to Gmail client dependency issues
    return EmailTestResponse(
        message="Email client temporarily disabled - dependencies need fixing",
        authenticated=False,
        client_type="GmailApiClient",
    )


def get_gmail_auth_config() -> EmailAuthConfig:
    """Get Gmail auth configuration from environment variables."""
    client_id = os.getenv("GMAIL_CLIENT_ID", "your-client-id")
    client_secret = os.getenv("GMAIL_CLIENT_SECRET", "your-client-secret")
    redirect_uri = os.getenv(
        "GMAIL_REDIRECT_URI", "http://localhost:8000/api/email/callback"
    )

    return EmailAuthConfig(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
    )


@router.get("/auth-url", response_model=AuthUrlResponse)
async def get_auth_url():
    """Get OAuth authorization URL for Gmail."""
    # Temporarily disabled due to Gmail client dependency issues
    raise HTTPException(
        status_code=503,
        detail="Gmail integration temporarily disabled - dependencies need fixing",
    )


@router.post("/callback")
async def oauth_callback(auth_code: str):
    """Handle OAuth callback."""
    # Temporarily disabled due to Gmail client dependency issues
    raise HTTPException(
        status_code=503,
        detail="Gmail integration temporarily disabled - dependencies need fixing",
    )


@router.post("/search", response_model=list[Email])
async def search_emails(request: EmailSearchRequest):
    """Search for emails (requires authentication)."""
    # Temporarily disabled due to Gmail client dependency issues
    raise HTTPException(
        status_code=503,
        detail="Gmail integration temporarily disabled - dependencies need fixing",
    )


@router.get("/status")
async def get_email_status():
    """Get current email authentication status."""
    # Temporarily disabled due to Gmail client dependency issues
    return {
        "authenticated": False,
        "client_type": "GmailApiClient",
        "message": "Gmail integration temporarily disabled - dependencies need fixing",
    }
