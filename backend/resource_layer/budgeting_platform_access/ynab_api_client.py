"""
YNAB API client implementation.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

import httpx

from thrift_gen.entities.ttypes import (
    Account,
    Budget,
    BudgetingPlatformType,
    Category,
    Payee,
    Transaction,
)

from .ynab_resource_interface import YnabResourceAccess

logger = logging.getLogger(__name__)


class YnabApiError(Exception):
    """Exception raised for YNAB API errors."""

    def __init__(self, message: str, status_code: int = None, error_code: str = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)


class YnabApiClient(YnabResourceAccess):
    """YNAB API client implementation."""

    def __init__(self, base_url: str = "https://api.youneedabudget.com/v1"):
        self.base_url = base_url
        self._client: httpx.AsyncClient | None = None
        self._rate_limit_remaining = 200  # YNAB default rate limit
        self._rate_limit_reset_time: datetime | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def _make_request(
        self, token: str, method: str, endpoint: str, **kwargs
    ) -> dict[str, Any]:
        """Make authenticated request to YNAB API with rate limiting."""
        if not token:
            raise YnabApiError("Token is required for API requests.")

        # Check rate limiting
        if self._rate_limit_remaining <= 1:
            if (
                self._rate_limit_reset_time
                and datetime.now() < self._rate_limit_reset_time
            ):
                wait_time = (
                    self._rate_limit_reset_time - datetime.now()
                ).total_seconds()
                await asyncio.sleep(wait_time)

        client = await self._get_client()
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {token}"

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = await client.request(method, url, headers=headers, **kwargs)

            # Update rate limiting info from headers
            if "X-Rate-Limit" in response.headers:
                self._rate_limit_remaining = int(
                    response.headers.get("X-Rate-Limit", 200)
                )

            if response.status_code == 429:  # Rate limited
                retry_after = int(response.headers.get("Retry-After", 60))
                await asyncio.sleep(retry_after)
                return await self._make_request(token, method, endpoint, **kwargs)

            if response.status_code >= 400:
                error_data = response.json() if response.content else {}
                error_message = error_data.get("error", {}).get(
                    "detail", f"HTTP {response.status_code}"
                )
                raise YnabApiError(
                    message=error_message,
                    status_code=response.status_code,
                    error_code=error_data.get("error", {}).get("id"),
                )

            return response.json()

        except httpx.RequestError as e:
            raise YnabApiError(f"Request failed: {str(e)}") from e

    async def validate_token(self, token: str) -> bool:
        """Validate if token works with YNAB API."""
        try:
            # Test token by getting user info
            await self._make_request(token, "GET", "/user")
            return True
        except YnabApiError:
            return False

    async def get_budgets(self, token: str) -> list[Budget]:
        """Get all budgets for authenticated user."""
        logger.info(f"Making YNAB API request with token length: {len(token)}")
        response = await self._make_request(token, "GET", "/budgets")
        budgets_data = response.get("data", {}).get("budgets", [])

        budgets = []
        for budget_data in budgets_data:
            # Extract currency from currency_format if available
            currency = "USD"  # Default
            if "currency_format" in budget_data:
                currency = budget_data["currency_format"].get("iso_code", "USD")

            budget = Budget(
                id=budget_data["id"],
                name=budget_data["name"],
                currency=currency,
                platformType=BudgetingPlatformType.YNAB,
            )
            budgets.append(budget)

        return budgets

    async def get_accounts(self, token: str, budget_id: str) -> list[Account]:
        """Get all accounts for a budget."""
        response = await self._make_request(
            token, "GET", f"/budgets/{budget_id}/accounts"
        )
        accounts_data = response.get("data", {}).get("accounts", [])

        accounts = []
        for account_data in accounts_data:
            # Create a simple object with YNAB data
            class SimpleAccount:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            account = SimpleAccount(
                id=account_data["id"],
                name=account_data["name"],
                type=account_data["type"],
                on_budget=account_data.get("on_budget", True),
                closed=account_data.get("closed", False),
                note=account_data.get("note", ""),
                balance=account_data.get("balance", 0),
                cleared_balance=account_data.get("cleared_balance", 0),
                uncleared_balance=account_data.get("uncleared_balance", 0),
            )
            accounts.append(account)

        return accounts

    async def get_transactions(
        self, token: str, budget_id: str, since_date: datetime | None = None
    ) -> list[Transaction]:
        """Get transactions for a budget."""
        endpoint = f"/budgets/{budget_id}/transactions"
        params = {}

        if since_date:
            params["since_date"] = since_date.strftime("%Y-%m-%d")

        response = await self._make_request(token, "GET", endpoint, params=params)
        transactions_data = response.get("data", {}).get("transactions", [])

        transactions = []
        for trans_data in transactions_data:
            # Parse date string to datetime
            date_str = trans_data["date"]
            transaction_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))

            # Create a simple transaction object with the data we have
            class SimpleTransaction:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            transaction = SimpleTransaction(
                id=trans_data["id"],
                account_id=trans_data["account_id"],
                category_id=trans_data.get("category_id"),
                payee_id=trans_data.get("payee_id"),  # Store actual payee ID
                payee_name=trans_data.get("payee_name"),  # Keep name for reference
                account_name=trans_data.get("account_name"),  # Store account name
                category_name=trans_data.get("category_name"),  # Store category name
                memo=trans_data.get("memo"),
                amount=trans_data["amount"],  # YNAB API uses 'amount' not 'milliunits'
                date=transaction_date,
                approved=trans_data.get("approved", False),
                deleted=trans_data.get("deleted", False),
                cleared=trans_data.get("cleared", "uncleared"),
            )
            transactions.append(transaction)

        return transactions

    async def get_categories(self, token: str, budget_id: str) -> list[Category]:
        """Get categories for a budget."""
        response = await self._make_request(
            token, "GET", f"/budgets/{budget_id}/categories"
        )
        category_groups = response.get("data", {}).get("category_groups", [])

        categories = []
        for group in category_groups:
            group_name = group["name"]
            group_id = group["id"]

            for category_data in group.get("categories", []):
                # Create a simple object with YNAB data
                class SimpleCategory:
                    def __init__(self, **kwargs):
                        for k, v in kwargs.items():
                            setattr(self, k, v)

                category = SimpleCategory(
                    id=category_data["id"],
                    name=category_data["name"],
                    category_group_id=group_id,
                    category_group_name=group_name,
                    budgeted=category_data.get("budgeted", 0),
                    activity=category_data.get("activity", 0),
                    balance=category_data.get("balance", 0),
                    goal_type=category_data.get("goal_type", ""),
                    goal_creation_month=category_data.get("goal_creation_month", ""),
                    goal_target=category_data.get("goal_target", 0),
                    goal_target_month=category_data.get("goal_target_month", ""),
                    goal_percentage_complete=category_data.get(
                        "goal_percentage_complete", 0
                    ),
                    deleted=category_data.get("deleted", False),
                )
                categories.append(category)

        return categories

    async def get_payees(self, token: str, budget_id: str) -> list[Payee]:
        """Get payees for a budget."""
        response = await self._make_request(
            token, "GET", f"/budgets/{budget_id}/payees"
        )
        payees_data = response.get("data", {}).get("payees", [])

        payees = []
        for payee_data in payees_data:
            # Create a simple object with YNAB data
            class SimplePayee:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            payee = SimplePayee(
                id=payee_data["id"],
                name=payee_data["name"],
                transfer_account_id=payee_data.get("transfer_account_id"),
                deleted=payee_data.get("deleted", False),
            )
            payees.append(payee)

        return payees

    async def update_transaction(
        self, token: str, budget_id: str, transaction: Transaction
    ) -> Transaction:
        """Update a single transaction - unified method for all field updates."""
        # Prepare update data - only include fields that can be updated
        update_data = {
            "transaction": {
                "id": transaction.id,
                "account_id": transaction.account_id,
                "payee_name": transaction.payee_name,
                "category_id": transaction.category_id,
                "memo": transaction.memo,
                # YNAB uses 'amount' (milliunits)
                "amount": transaction.amount,
                "date": transaction.date.strftime("%Y-%m-%d"),
                "approved": transaction.approved,
                "cleared": transaction.cleared,
            }
        }

        # Remove None values
        update_data["transaction"] = {
            k: v for k, v in update_data["transaction"].items() if v is not None
        }

        response = await self._make_request(
            token,
            "PATCH",
            f"/budgets/{budget_id}/transactions/{transaction.id}",
            json=update_data,
        )

        # Parse the updated transaction from response
        trans_data = response.get("data", {}).get("transaction", {})
        date_str = trans_data["date"]
        transaction_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))

        # Create a simple transaction object with the data we have
        class SimpleTransaction:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        updated_transaction = SimpleTransaction(
            id=trans_data["id"],
            account_id=trans_data["account_id"],
            category_id=trans_data.get("category_id"),
            payee_name=trans_data.get("payee_name"),
            memo=trans_data.get("memo"),
            amount=trans_data["amount"],  # YNAB API uses 'amount' not 'milliunits'
            date=transaction_date,
            approved=trans_data.get("approved", False),
            deleted=trans_data.get("deleted", False),
            cleared=trans_data.get("cleared", "uncleared"),
        )

        return updated_transaction

    async def update_transactions(
        self, token: str, budget_id: str, transactions: list[Transaction]
    ) -> bool:
        """Update multiple transactions for a budget using YNAB batch API."""
        try:
            # Prepare the batch request payload
            # Accept either dicts in YNAB shape or Transaction objects
            validated_transactions: list[dict] = []
            for t in transactions:
                if isinstance(t, dict):
                    # Already in YNAB shape from resource strategy
                    # Whitelist supported fields
                    allowed_keys = {
                        "id",
                        "account_id",
                        "payee_id",
                        "payee_name",
                        "category_id",
                        "memo",
                        "amount",
                        "date",
                        "approved",
                        "cleared",
                        "flag_color",
                    }
                    txn = {
                        k: v
                        for k, v in t.items()
                        if k in allowed_keys and v is not None
                    }
                elif isinstance(t, Transaction):
                    # Map Thrift Transaction to YNAB shape
                    txn = {
                        "id": t.id,
                        "account_id": t.accountId,
                        "payee_id": t.payeeId,
                        "category_id": t.categoryId,
                        "memo": t.memo,
                        "amount": t.amount,
                        "date": t.date,
                        "approved": t.approved,
                    }
                    txn = {k: v for k, v in txn.items() if v is not None}
                else:
                    raise TypeError(f"Invalid transaction type: {type(t)}")

                if "id" not in txn or not txn["id"]:
                    raise ValueError("Each transaction update requires an 'id'")
                validated_transactions.append(txn)

            for txn in validated_transactions:
                logger.debug(f"Batch update txn: {txn}")

            transactions_payload = validated_transactions

            # Make the batch API call
            response = await self._make_request(
                token,
                "PATCH",
                f"/budgets/{budget_id}/transactions",
                json={"transactions": transactions_payload},
            )

            # Log the result
            updated_transactions = response.get("data", {}).get("transactions", [])
            logger.info(
                f"Successfully updated {len(updated_transactions)}/{len(transactions)} transactions."
            )
            return len(updated_transactions) == len(transactions)

        except YnabApiError as e:
            logger.error(f"YNAB API error during batch transaction update: {e.message}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during batch transaction update: {e}")
            return False
