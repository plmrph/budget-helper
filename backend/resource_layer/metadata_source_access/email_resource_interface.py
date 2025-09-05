"""
Email Resource Access interface - abstract base class for email operations.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from models.email import Email


class EmailSearchParams:
    """Parameters for email search operations."""

    def __init__(
        self,
        query: str,
        start_date: datetime,
        end_date: datetime,
        max_results: int = 20,
    ):
        self.query = query
        self.start_date = start_date
        self.end_date = end_date
        self.max_results = max_results


class EmailAuthConfig:
    """Configuration for email authentication."""

    def __init__(
        self, client_id: str, client_secret: str, redirect_uri: str, scopes: list[str]
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scopes = scopes


class EmailResourceAccess(ABC):
    """Abstract base class for email resource access."""

    @abstractmethod
    async def validate_credentials(self, credentials: Any) -> bool:
        """Validate if credentials work with email API."""
        pass

    @abstractmethod
    async def search_emails(
        self, credentials: Any, params: EmailSearchParams
    ) -> list[Email]:
        """Search for emails based on parameters."""
        pass
