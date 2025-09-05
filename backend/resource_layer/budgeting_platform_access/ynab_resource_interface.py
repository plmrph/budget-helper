"""
YNAB Resource Access interface - abstract base class for YNAB API operations.
"""

from abc import ABC, abstractmethod
from datetime import datetime

from thrift_gen.entities.ttypes import Account, Budget, Category, Payee, Transaction


class YnabResourceAccess(ABC):
    """Abstract base class for YNAB resource access."""

    @abstractmethod
    async def validate_token(self, token: str) -> bool:
        """Validate if token works with YNAB API."""
        pass

    @abstractmethod
    async def get_budgets(self, token: str) -> list[Budget]:
        """Get all budgets for authenticated user."""
        pass

    @abstractmethod
    async def get_accounts(self, token: str, budget_id: str) -> list[Account]:
        """Get all accounts for a budget."""
        pass

    @abstractmethod
    async def get_transactions(
        self, token: str, budget_id: str, since_date: datetime | None = None
    ) -> list[Transaction]:
        """Get transactions for a budget."""
        pass

    @abstractmethod
    async def get_categories(self, token: str, budget_id: str) -> list[Category]:
        """Get categories for a budget."""
        pass

    @abstractmethod
    async def get_payees(self, token: str, budget_id: str) -> list[Payee]:
        """Get payees for a budget."""
        pass

    @abstractmethod
    async def update_transaction(
        self, token: str, budget_id: str, transaction: Transaction
    ) -> Transaction:
        """Update a single transaction - unified method for all field updates."""
        pass
