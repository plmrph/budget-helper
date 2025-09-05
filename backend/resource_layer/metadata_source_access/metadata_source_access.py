"""
MetadataSourceAccess implementation.

This ResourceAccess service provides atomic business verbs for metadata source operations,
implementing the MetadataSourceAccess Thrift service interface.
Uses strategy pattern for different metadata source types.
"""

import logging

from configs import ConfigService
from thrift_gen.entities.ttypes import Metadata, MetadataType
from thrift_gen.exceptions.ttypes import (
    InternalException,
)
from thrift_gen.metadatasourceaccess.MetadataSourceAccess import (
    Iface as MetadataSourceAccessIface,
)
from thrift_gen.metadatasourceaccess.ttypes import (
    MetadataSourceQuery,
    MetadataSourceType,
)

from .email_platform_strategy import GmailStrategy
from .metadata_source_access_strategy import (
    EmailMetadataSourceAccess,
    MetadataSourceAccessStrategy,
    PredictionMetadataSourceAccess,
)

logger = logging.getLogger(__name__)


class MetadataSourceAccess(MetadataSourceAccessIface):
    """
    MetadataSourceAccess implements atomic business verbs for metadata source operations.

    This ResourceAccess service converts business operations to external metadata source calls
    while exposing atomic business verbs rather than raw API operations.
    Uses strategy pattern to handle different metadata source types.
    """

    def __init__(
        self, gmail_api_client=None, ml_engine=None, database_store_access=None
    ):
        """
        Initialize MetadataSourceAccess.

        Args:
            gmail_api_client: Gmail API client instance
            ml_engine: ML engine instance for predictions
            database_store_access: Database store access instance for config
        """
        self.gmail_client = gmail_api_client
        self.ml_engine = ml_engine
        self.config_service = ConfigService(database_store_access)
        self.strategies: dict[MetadataType, MetadataSourceAccessStrategy] = {}

        # Initialize strategies
        self._initialize_strategies()

        logger.info("MetadataSourceAccess initialized with strategy pattern")

    def _initialize_strategies(self):
        """Initialize metadata source strategies."""
        try:
            # Initialize email strategy with Gmail platform strategy
            gmail_strategy = GmailStrategy(gmail_api_client=self.gmail_client)
            self.strategies[MetadataType.Email] = EmailMetadataSourceAccess(
                email_platform_strategy=gmail_strategy
            )

            # Initialize prediction strategy
            self.strategies[MetadataType.Prediction] = PredictionMetadataSourceAccess(
                ml_engine=self.ml_engine
            )

            logger.info("Metadata source strategies initialized")

        except Exception as e:
            logger.error(f"Error initializing strategies: {e}")
            raise InternalException(f"Failed to initialize strategies: {str(e)}") from e

    async def authenticate(self) -> bool:
        """
        Authenticate to all configured metadata sources.

        Returns:
            True if at least one source authenticated successfully
        """
        try:
            logger.info("Authenticating to metadata sources")

            auth_results = []

            for metadata_type, strategy in self.strategies.items():
                try:
                    result = await strategy.authenticate()
                    auth_results.append(result)
                    logger.info(
                        f"Authentication for {metadata_type}: {'success' if result else 'failed'}"
                    )
                except Exception as e:
                    logger.warning(f"Authentication error for {metadata_type}: {e}")
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

    async def getMetadata(self, queries: list[MetadataSourceQuery]) -> list[Metadata]:
        """
        Get metadata from various sources using strategy pattern.

        Args:
            queries: List of metadata source queries

        Returns:
            List of metadata items

        Raises:
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
            NotFoundException: If not found
            RemoteServiceException: If remote service error occurs
        """
        try:
            logger.info(f"Getting metadata from {len(queries)} queries")

            all_metadata = []

            for query in queries:
                try:
                    metadata_items = await self._execute_single_query(query)
                    all_metadata.extend(metadata_items)
                except Exception as e:
                    logger.warning(f"Error executing metadata query: {e}")
                    continue

            logger.info(f"Retrieved {len(all_metadata)} metadata items")
            return all_metadata

        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
            raise InternalException(f"Failed to get metadata: {str(e)}") from e

    async def _execute_single_query(self, query: MetadataSourceQuery) -> list[Metadata]:
        """
        Execute a single metadata source query using appropriate strategy.

        Args:
            query: Metadata source query

        Returns:
            List of metadata items
        """
        try:
            # Map MetadataSourceType to MetadataType
            metadata_type = None
            if query.sourceType == MetadataSourceType.Email:
                metadata_type = MetadataType.Email

            if not metadata_type:
                logger.warning(
                    f"No metadata type mapping for source type: {query.sourceType}"
                )
                return []

            # Get strategy for metadata type
            strategy = self.strategies.get(metadata_type)
            if not strategy:
                logger.warning(
                    f"No strategy available for metadata type: {metadata_type}"
                )
                return []

            # Execute query using strategy
            return await strategy.get_metadata(query)

        except Exception as e:
            logger.error(f"Error executing metadata query: {e}")
            return []
