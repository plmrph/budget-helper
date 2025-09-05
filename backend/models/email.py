"""
Email data model.
"""

from datetime import datetime

from pydantic import BaseModel, Field, validator


class Email(BaseModel):
    """Email model representing an email message."""

    id: str = Field(..., description="Unique email identifier")
    thread_id: str = Field(..., description="Email thread identifier")
    subject: str = Field(..., description="Email subject")
    sender: str = Field(..., description="Email sender address")
    date: datetime = Field(..., description="Email date")
    snippet: str = Field(default="", description="Email snippet/preview")
    body_text: str = Field(default="", description="Email body as plain text")
    body_html: str = Field(default="", description="Email body as HTML")
    url: str = Field(default="", description="URL to view the email")
    labels: list[str] = Field(default_factory=list, description="Email labels/tags")
    matched_terms: str = Field(
        default="", description="Search terms that matched this email"
    )
    properties: dict | None = Field(default=None, description="Optional map of primitive properties (e.g., relevance)")

    @validator("sender")
    def validate_sender_email(cls, v):
        """Flexible sender validation - allows display names and email addresses."""
        if not v or not v.strip():
            return "Unknown"
        return v.strip()

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}
        json_schema_extra = {
            "example": {
                "id": "email_123",
                "thread_id": "thread_456",
                "subject": "Your receipt from Grocery Store",
                "sender": "receipts@grocerystore.com",
                "date": "2024-01-15T10:30:00Z",
                "snippet": "Thank you for your purchase...",
                "body_text": "Thank you for your purchase of $50.00 at Grocery Store...",
                "body_html": "<html><body>Thank you for your purchase...</body></html>",
                "url": "https://mail.google.com/mail/u/0/#inbox/email_123",
                "labels": ["INBOX", "IMPORTANT"],
            }
        }
