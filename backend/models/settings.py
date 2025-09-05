"""
Application settings data model.
"""

from typing import Any

from pydantic import BaseModel, Field, validator


class EmailSearchSettings(BaseModel):
    """Email search configuration settings."""

    days_before: int = Field(
        default=3, description="Days before transaction date to search"
    )
    days_after: int = Field(
        default=3, description="Days after transaction date to search"
    )
    max_results: int = Field(
        default=20, description="Maximum number of email results to return"
    )
    include_amount: bool = Field(
        default=True, description="Include transaction amount in search"
    )
    include_payee: bool = Field(
        default=True, description="Include payee name in search"
    )
    additional_keywords: list[str] = Field(
        default_factory=list, description="Additional keywords to include in search"
    )
    custom_search_string: str | None = Field(
        None, description="Custom search string template"
    )

    @validator("days_before", "days_after")
    def validate_days(cls, v):
        """Validate days are positive."""
        if v < 0:
            raise ValueError("Days must be non-negative")
        if v > 30:
            raise ValueError("Days cannot exceed 30")
        return v

    @validator("max_results")
    def validate_max_results(cls, v):
        """Validate max results is reasonable."""
        if v < 1:
            raise ValueError("Max results must be at least 1")
        if v > 100:
            raise ValueError("Max results cannot exceed 100")
        return v


class EmailAutomationSettings(BaseModel):
    """Email automation configuration settings."""

    auto_add_url_single_match: bool = Field(
        default=False, description="Automatically add URL when only one email matches"
    )
    append_url_by_default: bool = Field(
        default=True, description="Append URL to memo by default"
    )
    auto_search_on_transaction_create: bool = Field(
        default=False,
        description="Automatically search for emails when transaction is created",
    )
    auto_search_on_transaction_update: bool = Field(
        default=False,
        description="Automatically search for emails when transaction is updated",
    )


class DisplaySettings(BaseModel):
    """Display configuration settings."""

    theme: str = Field(default="light", description="UI theme")
    items_per_page: int = Field(
        default=50, description="Number of items to show per page"
    )
    show_approved: bool = Field(
        default=False, description="Show approved transactions by default"
    )
    default_columns: list[str] = Field(
        default_factory=list, description="Default columns to show in transaction table"
    )
    auto_refresh_on_startup: bool = Field(
        default=True, description="Auto refresh data on application startup"
    )
    date_format: str = Field(default="YYYY-MM-DD", description="Date format preference")
    currency_display: str = Field(
        default="symbol", description="Currency display preference"
    )

    @validator("theme")
    def validate_theme(cls, v):
        """Validate theme is one of allowed values."""
        allowed_themes = ["light", "dark", "system"]
        if v not in allowed_themes:
            raise ValueError(f"Theme must be one of: {', '.join(allowed_themes)}")
        return v

    @validator("items_per_page")
    def validate_items_per_page(cls, v):
        """Validate items per page is reasonable."""
        if v < 10:
            raise ValueError("Items per page must be at least 10")
        if v > 200:
            raise ValueError("Items per page cannot exceed 200")
        return v


class AISettings(BaseModel):
    """AI configuration settings."""

    auto_categorize: bool = Field(
        default=False, description="Automatically categorize transactions using AI"
    )
    training_months: int = Field(
        default=6, description="Number of months of data to use for training"
    )
    active_model: str | None = Field(
        None, description="Name of the currently active model"
    )
    confidence_threshold: float = Field(
        default=0.8, description="Minimum confidence threshold for auto-categorization"
    )
    retrain_frequency_days: int = Field(
        default=30, description="How often to retrain models in days"
    )

    @validator("training_months")
    def validate_training_months(cls, v):
        """Validate training months is reasonable."""
        if v < 1:
            raise ValueError("Training months must be at least 1")
        if v > 24:
            raise ValueError("Training months cannot exceed 24")
        return v

    @validator("confidence_threshold")
    def validate_confidence_threshold(cls, v):
        """Validate confidence threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        return v

    @validator("retrain_frequency_days")
    def validate_retrain_frequency(cls, v):
        """Validate retrain frequency is reasonable."""
        if v < 1:
            raise ValueError("Retrain frequency must be at least 1 day")
        if v > 365:
            raise ValueError("Retrain frequency cannot exceed 365 days")
        return v


class AppSettings(BaseModel):
    """Application settings container."""

    email_search: EmailSearchSettings | None = Field(
        default=None, description="Email search settings"
    )
    email_automation: EmailAutomationSettings | None = Field(
        default=None, description="Email automation settings"
    )
    display: DisplaySettings | None = Field(
        default=None, description="Display settings"
    )
    ai: AISettings | None = Field(default=None, description="AI settings")

    # Auth data fields (stored as raw dicts to avoid circular imports)
    ynab_auth: dict[str, Any] | None = Field(
        default=None, description="YNAB authentication data"
    )
    email_auth: dict[str, Any] | None = Field(
        default=None, description="Email service authentication data"
    )
    auth_data: dict[str, Any] | None = Field(
        default=None, description="Unified authentication data"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "email_search": {
                    "days_before": 3,
                    "days_after": 3,
                    "max_results": 20,
                    "include_amount": True,
                    "include_payee": True,
                    "additional_keywords": ["receipt", "invoice"],
                    "custom_search_string": None,
                },
                "email_automation": {
                    "auto_add_url_single_match": False,
                    "append_url_by_default": True,
                    "auto_search_on_transaction_create": False,
                    "auto_search_on_transaction_update": False,
                },
                "display": {
                    "theme": "light",
                    "items_per_page": 50,
                    "show_approved": False,
                    "default_columns": ["date", "payee", "category", "amount"],
                    "auto_refresh_on_startup": True,
                    "date_format": "YYYY-MM-DD",
                    "currency_display": "symbol",
                },
                "ai": {
                    "auto_categorize": False,
                    "training_months": 6,
                    "active_model": None,
                    "confidence_threshold": 0.8,
                    "retrain_frequency_days": 30,
                },
            }
        }
