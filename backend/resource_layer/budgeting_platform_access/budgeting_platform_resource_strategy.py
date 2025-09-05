"""
BudgetingPlatformResourceStrategy interface and implementations.

This module defines the strategy interface for budgeting platform resources
and provides concrete implementations for each platform type.
"""

import logging
from abc import ABC, abstractmethod

from configs import ConfigKeys
from thrift_gen.entities.ttypes import (
    Account,
    Budget,
    BudgetingPlatformType,
    Category,
    Payee,
    Transaction,
)
from thrift_gen.exceptions.ttypes import (
    ConflictException,
    RemoteServiceException,
    UnauthorizedException,
)

logger = logging.getLogger(__name__)


class BudgetingPlatformResourceStrategy(ABC):
    """Abstract interface for budgeting platform resource strategies."""

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate to the budgeting platform."""
        pass

    @abstractmethod
    async def get_accounts(self) -> list[Account]:
        """Get accounts from the platform."""
        pass

    @abstractmethod
    async def get_categories(self) -> list[Category]:
        """Get categories from the platform."""
        pass

    @abstractmethod
    async def get_payees(self) -> list[Payee]:
        """Get payees from the platform."""
        pass

    @abstractmethod
    async def get_budgets(self) -> list[Budget]:
        """Get budgets from the platform."""
        pass

    @abstractmethod
    async def get_transactions(
        self,
        is_pending: bool | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Transaction]:
        """Get transactions from the platform."""
        pass

    @abstractmethod
    async def update_transactions(self, transactions: list[Transaction]) -> bool:
        """Update transactions on the platform."""
        pass


class YNABStrategy(BudgetingPlatformResourceStrategy):
    """YNAB implementation of BudgetingPlatformResourceStrategy."""

    def __init__(self, ynab_api_client=None, config_service=None):
        self.ynab_client = ynab_api_client
        self.config_service = config_service
        self._token = None  # Store the authentication token

    async def authenticate(self) -> bool:
        """Authenticate to YNAB."""
        try:
            if not self.ynab_client:
                logger.warning("YNAB client not configured")
                return False

            # Check if we have a stored token in configs
            if self.config_service:
                try:
                    token = await self.config_service.getConfigValue(
                        ConfigKeys.BUDGET_YNAB_AUTH_CONFIG
                    )
                    if token and token.strip():
                        self._token = token.strip()
                        logger.info(
                            f"YNAB token found in configuration (length: {len(self._token)}, starts with: {self._token[:10]}...)"
                        )

                        # Validate token format - YNAB tokens are typically 64 characters long
                        if len(self._token) < 20:
                            logger.warning(
                                f"YNAB token seems too short: {len(self._token)} characters"
                            )
                            return False

                        return True
                    else:
                        logger.info(
                            "YNAB token not found in configuration - YNAB features will be unavailable until token is provided"
                        )
                        return False
                except Exception as config_error:
                    logger.warning(
                        f"Error retrieving YNAB token from config: {config_error}"
                    )
                    return False
            else:
                logger.warning("Config service not available for YNAB authentication")
                return False

        except Exception as e:
            logger.warning(f"YNAB authentication error: {e}")
            return False

    async def get_accounts(self) -> list[Account]:
        """Get accounts from YNAB."""
        try:
            if not await self.authenticate():
                logger.warning(
                    "YNAB authentication not available - returning empty account list"
                )
                return []

            if not self.ynab_client:
                logger.warning(
                    "YNAB client not available - returning empty account list"
                )
                return []

            # Get the selected budget ID from configuration
            selected_budget_id = None
            if self.config_service:
                selected_budget_id = await self.config_service.getConfigValue(
                    ConfigKeys.SELECTED_BUDGET_ID
                )

            if not selected_budget_id:
                logger.warning("No budget ID configured - returning empty account list")
                return []

            # Get accounts from YNAB API
            ynab_accounts = await self.ynab_client.get_accounts(
                self._token, selected_budget_id
            )

            # Convert YNAB accounts to our Account objects
            accounts = []
            for ynab_account in ynab_accounts:
                account = self._convert_ynab_account_from_api(
                    ynab_account, selected_budget_id
                )
                accounts.append(account)

            logger.info(f"Retrieved {len(accounts)} accounts from YNAB")
            return accounts

        except (UnauthorizedException, RemoteServiceException):
            raise
        except Exception as e:
            logger.error(f"Error getting YNAB accounts: {e}")
            raise RemoteServiceException(f"YNAB API error: {str(e)}") from e

    async def get_categories(self) -> list[Category]:
        """Get categories from YNAB."""
        try:
            if not await self.authenticate():
                logger.warning(
                    "YNAB authentication not available - returning empty category list"
                )
                return []

            if not self.ynab_client:
                logger.warning(
                    "YNAB client not available - returning empty category list"
                )
                return []

            selected_budget_id = None
            if self.config_service:
                selected_budget_id = await self.config_service.getConfigValue(
                    ConfigKeys.SELECTED_BUDGET_ID
                )

            if not selected_budget_id:
                logger.warning(
                    "No selected budget ID found - returning empty category list"
                )
                return []

            # Get categories from YNAB API
            ynab_categories = await self.ynab_client.get_categories(
                self._token, selected_budget_id
            )

            # Convert YNAB categories to Category objects (include all categories, even deleted ones)
            categories = []
            for ynab_category in ynab_categories:
                category = self._convert_ynab_category_from_api(
                    ynab_category, selected_budget_id
                )
                categories.append(category)

            logger.info(f"Retrieved {len(categories)} categories from YNAB")
            return categories

        except (UnauthorizedException, RemoteServiceException):
            raise
        except Exception as e:
            logger.error(f"Error getting YNAB categories: {e}")
            raise RemoteServiceException(f"YNAB API error: {str(e)}") from e

    async def get_payees(self) -> list[Payee]:
        """Get payees from YNAB."""
        try:
            if not await self.authenticate():
                logger.warning(
                    "YNAB authentication not available - returning empty payee list"
                )
                return []

            if not self.ynab_client:
                logger.warning("YNAB client not available - returning empty payee list")
                return []

            # Get the selected budget ID from configuration
            selected_budget_id = None
            if self.config_service:
                selected_budget_id = await self.config_service.getConfigValue(
                    ConfigKeys.SELECTED_BUDGET_ID
                )

            if not selected_budget_id:
                logger.warning("No budget ID configured - returning empty payee list")
                return []

            # Get payees from YNAB API
            ynab_payees = await self.ynab_client.get_payees(
                self._token, selected_budget_id
            )

            # Convert YNAB payees to Payee objects
            payees = []
            for ynab_payee in ynab_payees:
                payee = self._convert_ynab_payee_from_api(
                    ynab_payee, selected_budget_id
                )
                payees.append(payee)

            logger.info(f"Retrieved {len(payees)} payees from YNAB")
            return payees

        except (UnauthorizedException, RemoteServiceException):
            raise
        except Exception as e:
            logger.error(f"Error getting YNAB payees: {e}")
            raise RemoteServiceException(f"YNAB API error: {str(e)}") from e

    async def get_budgets(self) -> list[Budget]:
        """Get budgets from YNAB."""
        try:
            if not await self.authenticate():
                logger.warning(
                    "YNAB authentication not available - returning empty budget list"
                )
                return []

            if not self.ynab_client:
                logger.warning(
                    "YNAB client not available - returning empty budget list"
                )
                return []

            # Get budgets from YNAB API (already returns Budget objects)
            budgets = await self.ynab_client.get_budgets(self._token)

            logger.info(f"Retrieved {len(budgets)} budgets from YNAB")
            return budgets

        except (UnauthorizedException, RemoteServiceException):
            raise
        except Exception as e:
            logger.error(f"Error getting YNAB budgets: {e}")
            raise RemoteServiceException(f"YNAB API error: {str(e)}") from e

    async def get_transactions(
        self,
        is_pending: bool | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Transaction]:
        """Get transactions from YNAB."""
        try:
            if not await self.authenticate():
                logger.warning(
                    "YNAB authentication not available - returning empty transaction list"
                )
                return []

            if not self.ynab_client:
                logger.warning(
                    "YNAB client not available - returning empty transaction list"
                )
                return []

            # Get the selected budget ID from configuration
            selected_budget_id = None
            if self.config_service:
                selected_budget_id = await self.config_service.getConfigValue(
                    ConfigKeys.SELECTED_BUDGET_ID
                )

            if not selected_budget_id:
                logger.warning(
                    "No budget ID configured - returning empty transaction list"
                )
                return []

            logger.info(f"Using configured budget ID: {selected_budget_id}")

            # Parse start_date if provided
            since_date = None
            if start_date:
                from datetime import datetime

                try:
                    since_date = datetime.fromisoformat(
                        start_date.replace("Z", "+00:00")
                    )
                except ValueError:
                    logger.warning(f"Invalid start_date format: {start_date}")

            # Get transactions from YNAB API
            ynab_transactions = await self.ynab_client.get_transactions(
                token=self._token, budget_id=selected_budget_id, since_date=since_date
            )

            # Convert YNAB transactions to our Transaction objects
            transactions = []
            for ynab_transaction in ynab_transactions:
                # Skip deleted transactions unless specifically requested
                if ynab_transaction.deleted:
                    continue

                # Filter by pending status if specified
                if is_pending is not None:
                    # In YNAB, pending transactions are typically unapproved
                    transaction_is_pending = not ynab_transaction.approved
                    if transaction_is_pending != is_pending:
                        continue

                # Skip YNAB "Split" transactions by name
                category_name = getattr(ynab_transaction, "category_name", None)
                if (
                    isinstance(category_name, str)
                    and category_name.strip().lower() == "split"
                ):
                    logger.info(
                        f"Skipping split transaction from YNAB: {getattr(ynab_transaction, 'id', '<unknown>')} (category_name=Split)"
                    )
                    continue

                transaction = self._convert_ynab_transaction_from_api(
                    ynab_transaction, selected_budget_id
                )
                transactions.append(transaction)

            logger.info(f"Retrieved {len(transactions)} transactions from YNAB")
            return transactions

        except (UnauthorizedException, RemoteServiceException):
            raise
        except Exception as e:
            logger.error(f"Error getting YNAB transactions: {e}")
            raise RemoteServiceException(f"YNAB API error: {str(e)}") from e

    async def update_transactions(self, transactions: list[Transaction]) -> bool:
        """Update transactions on YNAB."""
        try:
            if not await self.authenticate():
                logger.warning(
                    "YNAB authentication not available - cannot update transactions"
                )
                return False

            if not self._token:
                logger.error("Authentication token is missing. Cannot proceed.")
                return False

            # Retrieve the budget ID from configuration
            if not self.config_service:
                logger.error(
                    "Config service is unavailable. Cannot retrieve budget ID."
                )
                return False

            selected_budget_id = await self.config_service.getConfigValue(
                ConfigKeys.SELECTED_BUDGET_ID
            )

            if not selected_budget_id:
                logger.error(
                    "No selected budget ID found. Aborting transaction update."
                )
                return False

            # Convert Transaction objects to YNAB format
            ynab_transactions = [
                self._convert_to_ynab_transaction(transaction)
                for transaction in transactions
            ]

            # Make the API call to update transactions
            success = await self.ynab_client.update_transactions(
                token=self._token,
                budget_id=selected_budget_id,
                transactions=ynab_transactions,
            )

            # Log the result
            if success:
                logger.info(
                    f"Successfully updated {len(transactions)} transactions on YNAB."
                )
            else:
                logger.warning(
                    f"Failed to update transactions on YNAB. Total attempted: {len(transactions)}"
                )

            return success

        except UnauthorizedException:
            logger.error("Unauthorized access to YNAB API.")
            raise
        except RemoteServiceException as e:
            logger.error(f"Remote service error while updating transactions: {e}")
            raise
        except ConflictException as e:
            logger.error(f"Conflict error while updating transactions: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating YNAB transactions: {e}")
            raise RemoteServiceException(f"YNAB API error: {str(e)}") from e

    def _convert_ynab_account_from_api(self, ynab_account, budget_id: str) -> Account:
        """Convert YNAB API account object to our Account object."""
        return Account(
            id=ynab_account.id,
            name=ynab_account.name,
            type=ynab_account.type,
            platformType=BudgetingPlatformType.YNAB,
            institution=None,  # YNAB API doesn't provide institution info
            currency="USD",  # Default currency
            balance=ynab_account.balance / 1000.0
            if ynab_account.balance
            else 0.0,  # Convert from milliunits
            status="closed" if getattr(ynab_account, "closed", False) else "open",
            budgetId=budget_id,
        )

    def _convert_ynab_category_from_api(
        self, ynab_category, budget_id: str
    ) -> Category:
        """Convert YNAB API category object to our Category object."""
        return Category(
            id=ynab_category.id,
            name=f"{ynab_category.category_group_name}: {ynab_category.name}",
            platformType=BudgetingPlatformType.YNAB,
            description=None,  # YNAB categories don't have descriptions
            isIncomeCategory=ynab_category.category_group_name == "Income",
            budgetId=budget_id,
        )

    def _convert_ynab_payee_from_api(self, ynab_payee, budget_id: str) -> Payee:
        """Convert YNAB API payee object to our Payee object."""
        return Payee(
            id=ynab_payee.id,
            name=ynab_payee.name,
            platformType=BudgetingPlatformType.YNAB,
            description=None,  # YNAB doesn't have payee descriptions
            budgetId=budget_id,
        )

    def _convert_ynab_budget(self, ynab_budget: dict) -> Budget:
        """Convert YNAB budget to Budget object."""
        return Budget(
            id=ynab_budget.get("id", ""),
            name=ynab_budget.get("name", ""),
            currency=ynab_budget.get("currency_format", {}).get("iso_code", "USD"),
            platformType=BudgetingPlatformType.YNAB,
            totalAmount=None,  # YNAB doesn't have a single total amount
            startDate=ynab_budget.get("first_month"),
            endDate=ynab_budget.get("last_month"),
        )

    def _convert_ynab_transaction(self, ynab_transaction: dict) -> Transaction:
        """Convert YNAB transaction dict to Transaction object."""
        return Transaction(
            id=ynab_transaction.get("id", ""),
            date=ynab_transaction.get("date", ""),
            amount=ynab_transaction.get("amount", 0),  # Already in milliunits
            approved=ynab_transaction.get("approved", False),
            platformType=BudgetingPlatformType.YNAB,
            payeeId=ynab_transaction.get("payee_id"),
            categoryId=ynab_transaction.get("category_id"),
            accountId=ynab_transaction.get("account_id"),
            budgetId=ynab_transaction.get("budget_id"),
            memo=ynab_transaction.get("memo"),
            metadata=[],  # Will be populated separately
        )

    def _convert_ynab_transaction_from_api(
        self, ynab_transaction, budget_id: str
    ) -> Transaction:
        """Convert YNAB API client transaction object to our Transaction object."""
        # Convert datetime to ISO string if needed
        date_str = ynab_transaction.date
        if hasattr(ynab_transaction.date, "isoformat"):
            date_str = ynab_transaction.date.isoformat()
        elif hasattr(ynab_transaction.date, "strftime"):
            date_str = ynab_transaction.date.strftime("%Y-%m-%d")

        # Handle null category_id (set to None instead of empty string)
        category_id = ynab_transaction.category_id
        if not category_id or category_id == "":
            category_id = None

        # Validate and clean transaction ID - YNAB sometimes returns invalid UUIDs
        transaction_id = ynab_transaction.id
        if transaction_id and len(transaction_id) > 36:
            # If ID is too long, it might have extra data appended (like dates)
            # Check for underscore and trim at that point
            if "_" in transaction_id:
                underscore_pos = transaction_id.find("_")
                transaction_id = transaction_id[:underscore_pos]
                logger.warning(
                    f"Transaction ID had extra data, trimmed at underscore: {ynab_transaction.id} -> {transaction_id}"
                )
            else:
                # This is an error scenario - log it as such
                logger.error(
                    f"Transaction ID is too long and has no underscore to trim: {transaction_id} (length: {len(transaction_id)})"
                )
                raise ValueError(f"Invalid transaction ID format: {transaction_id}")

        return Transaction(
            id=transaction_id,
            date=date_str,
            amount=ynab_transaction.amount,  # Already in milliunits
            approved=ynab_transaction.approved,
            platformType=BudgetingPlatformType.YNAB,
            payeeId=ynab_transaction.payee_id,  # Store actual payee ID
            categoryId=category_id,
            accountId=ynab_transaction.account_id,
            budgetId=budget_id,  # Add budget ID
            memo=ynab_transaction.memo,
            metadata=[],  # Will be populated separately
        )

    def _convert_to_ynab_transaction(self, transaction: Transaction) -> dict:
        """Convert Transaction object to YNAB format."""
        return {
            "id": transaction.id,
            "date": transaction.date,
            "amount": transaction.amount,  # Already in milliunits
            "approved": transaction.approved,
            "payee_id": transaction.payeeId,
            "category_id": transaction.categoryId,
            "account_id": transaction.accountId,
            "memo": transaction.memo,
        }
