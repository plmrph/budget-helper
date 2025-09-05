"""
BudgetingPlatformAccess implementation.

This ResourceAccess service provides atomic business verbs for budgeting platform operations,
implementing the BudgetingPlatformAccess Thrift service interface.
Uses strategy pattern for different budgeting platform types.
"""

import logging

from thrift_gen.budgetingplatformaccess.BudgetingPlatformAccess import (
    Iface as BudgetingPlatformAccessIface,
)
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
    InternalException,
    NotFoundException,
    RemoteServiceException,
    UnauthorizedException,
)

from .budgeting_platform_resource_strategy import (
    BudgetingPlatformResourceStrategy,
    YNABStrategy,
)

logger = logging.getLogger(__name__)


class BudgetingPlatformAccess(BudgetingPlatformAccessIface):
    """
    BudgetingPlatformAccess implements atomic business verbs for budgeting platform operations.

    This ResourceAccess service converts business operations to external API calls
    while exposing atomic business verbs rather than raw API operations.
    Uses strategy pattern to handle different budgeting platform types.
    """

    def __init__(
        self, ynab_api_client=None, database_store_access=None, config_service=None
    ):
        """
        Initialize BudgetingPlatformAccess.

        Args:
            ynab_api_client: YNAB API client instance
            database_store_access: Database store access instance for config
            config_service: Config service instance (optional, will create if not provided)
        """
        self.ynab_client = ynab_api_client
        from configs import get_config_service

        self.config_service = config_service or get_config_service(
            database_store_access
        )
        self.strategies: dict[
            BudgetingPlatformType, BudgetingPlatformResourceStrategy
        ] = {}

        # Initialize strategies
        self._initialize_strategies()

        logger.info("BudgetingPlatformAccess initialized with strategy pattern")

    def _initialize_strategies(self):
        """Initialize budgeting platform strategies."""
        try:
            # Initialize YNAB strategy
            self.strategies[BudgetingPlatformType.YNAB] = YNABStrategy(
                ynab_api_client=self.ynab_client, config_service=self.config_service
            )

            logger.info("Budgeting platform strategies initialized")

        except Exception as e:
            logger.error(f"Error initializing strategies: {e}")
            raise InternalException(f"Failed to initialize strategies: {str(e)}") from e

    async def authenticate(self) -> bool:
        """
        Authenticate to all configured budgeting platforms.

        Returns:
            True if at least one platform authenticated successfully
        """
        try:
            logger.info("Authenticating to budgeting platforms")

            auth_results = []

            for platform_type, strategy in self.strategies.items():
                try:
                    result = await strategy.authenticate()
                    auth_results.append(result)
                    logger.info(
                        f"Authentication for {platform_type}: {'success' if result else 'failed'}"
                    )
                except Exception as e:
                    logger.warning(f"Authentication error for {platform_type}: {e}")
                    auth_results.append(False)

            # Return True if at least one authentication succeeded
            success = any(auth_results)
            logger.info(
                f"Overall authentication result: {'success' if success else 'failed'}"
            )

            return success

        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            raise InternalException(f"Authentication failed: {str(e)}") from e

    async def getAccounts(
        self, platform: BudgetingPlatformType | None = None
    ) -> list[Account]:
        """
        Get accounts from budgeting platform using strategy pattern.

        Args:
            platform: Budgeting platform type (defaults to YNAB)

        Returns:
            List of accounts

        Raises:
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
            NotFoundException: If not found
            RemoteServiceException: If remote service error occurs
        """
        try:
            platform_type = (
                platform or await self.config_service.getDefaultBudgetPlatform()
            )
            logger.info(f"Getting accounts from {platform_type}")

            # Get strategy for platform type
            strategy = self.strategies.get(platform_type)
            if not strategy:
                raise InternalException(
                    f"No strategy available for platform: {platform_type}"
                )

            # Execute using strategy
            return await strategy.get_accounts()

        except (UnauthorizedException, NotFoundException, RemoteServiceException):
            raise
        except Exception as e:
            logger.error(f"Error getting accounts: {e}")
            raise InternalException(f"Failed to get accounts: {str(e)}") from e

    async def getCategories(
        self, platform: BudgetingPlatformType | None = None
    ) -> list[Category]:
        """
        Get categories from budgeting platform using strategy pattern.

        Args:
            platform: Budgeting platform type (defaults to YNAB)

        Returns:
            List of categories

        Raises:
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
            NotFoundException: If not found
            RemoteServiceException: If remote service error occurs
        """
        try:
            platform_type = (
                platform or await self.config_service.getDefaultBudgetPlatform()
            )
            logger.info(f"Getting categories from {platform_type}")

            # Get strategy for platform type
            strategy = self.strategies.get(platform_type)
            if not strategy:
                raise InternalException(
                    f"No strategy available for platform: {platform_type}"
                )

            # Execute using strategy
            return await strategy.get_categories()

        except (UnauthorizedException, NotFoundException, RemoteServiceException):
            raise
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            raise InternalException(f"Failed to get categories: {str(e)}") from e

    async def getPayees(
        self, platform: BudgetingPlatformType | None = None
    ) -> list[Payee]:
        """
        Get payees from budgeting platform using strategy pattern.

        Args:
            platform: Budgeting platform type (defaults to YNAB)

        Returns:
            List of payees

        Raises:
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
            NotFoundException: If not found
            RemoteServiceException: If remote service error occurs
        """
        try:
            platform_type = (
                platform or await self.config_service.getDefaultBudgetPlatform()
            )
            logger.info(f"Getting payees from {platform_type}")

            # Get strategy for platform type
            strategy = self.strategies.get(platform_type)
            if not strategy:
                raise InternalException(
                    f"No strategy available for platform: {platform_type}"
                )

            # Execute using strategy
            return await strategy.get_payees()

        except (UnauthorizedException, NotFoundException, RemoteServiceException):
            raise
        except Exception as e:
            logger.error(f"Error getting payees: {e}")
            raise InternalException(f"Failed to get payees: {str(e)}") from e

    async def getBudgets(
        self, platform: BudgetingPlatformType | None = None
    ) -> list[Budget]:
        """
        Get budgets from budgeting platform using strategy pattern.

        Args:
            platform: Budgeting platform type (defaults to YNAB)

        Returns:
            List of budgets

        Raises:
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
            NotFoundException: If not found
            RemoteServiceException: If remote service error occurs
        """
        try:
            platform_type = (
                platform or await self.config_service.getDefaultBudgetPlatform()
            )
            logger.info(f"Getting budgets from {platform_type}")

            # Get strategy for platform type
            strategy = self.strategies.get(platform_type)
            if not strategy:
                raise InternalException(
                    f"No strategy available for platform: {platform_type}"
                )

            # Execute using strategy
            return await strategy.get_budgets()

        except (UnauthorizedException, NotFoundException, RemoteServiceException):
            raise
        except Exception as e:
            logger.error(f"Error getting budgets: {e}")
            raise InternalException(f"Failed to get budgets: {str(e)}") from e

    async def getTransactions(
        self,
        platform: BudgetingPlatformType | None = None,
        isPending: bool | None = None,
        startDate: str | None = None,
        endDate: str | None = None,
    ) -> list[Transaction]:
        """
        Get transactions from budgeting platform using strategy pattern.

        Args:
            platform: Budgeting platform type (defaults to YNAB)
            isPending: Filter by pending status
            startDate: Start date filter (ISO format)
            endDate: End date filter (ISO format)

        Returns:
            List of transactions

        Raises:
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
            NotFoundException: If not found
            RemoteServiceException: If remote service error occurs
        """
        try:
            platform_type = (
                platform or await self.config_service.getDefaultBudgetPlatform()
            )
            logger.info(f"Getting transactions from {platform_type}")

            # Get strategy for platform type
            strategy = self.strategies.get(platform_type)
            if not strategy:
                raise InternalException(
                    f"No strategy available for platform: {platform_type}"
                )

            # Execute using strategy
            return await strategy.get_transactions(isPending, startDate, endDate)

        except (UnauthorizedException, NotFoundException, RemoteServiceException):
            raise
        except Exception as e:
            logger.error(f"Error getting transactions: {e}")
            raise InternalException(f"Failed to get transactions: {str(e)}") from e

    async def updateTransactions(
        self,
        transactions: list[Transaction],
        platform: BudgetingPlatformType | None = None,
    ) -> bool:
        """
        Update transactions on budgeting platform using strategy pattern.

        Args:
            transactions: List of transactions to update
            platform: Budgeting platform type (defaults to YNAB)

        Returns:
            True if update successful

        Raises:
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
            ConflictException: If conflict occurs
            RemoteServiceException: If remote service error occurs
        """
        try:
            platform_type = (
                platform or await self.config_service.getDefaultBudgetPlatform()
            )
            logger.info(f"Updating {len(transactions)} transactions on {platform_type}")

            # Get strategy for platform type
            strategy = self.strategies.get(platform_type)
            if not strategy:
                raise InternalException(
                    f"No strategy available for platform: {platform_type}"
                )

            # Execute using strategy
            return await strategy.update_transactions(transactions)

        except (UnauthorizedException, ConflictException, RemoteServiceException):
            raise
        except Exception as e:
            logger.error(f"Error updating transactions: {e}")
            raise InternalException(f"Failed to update transactions: {str(e)}") from e
