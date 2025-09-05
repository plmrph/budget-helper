"""
MetadataSourceAccess strategy interface and implementations.

This module defines the strategy interface for metadata source access
and provides concrete implementations for each MetadataType.
"""

import logging
from abc import ABC, abstractmethod

from thrift_gen.entities.ttypes import Metadata
from thrift_gen.exceptions.ttypes import (
    InternalException,
    RemoteServiceException,
    UnauthorizedException,
)
from thrift_gen.metadatasourceaccess.ttypes import MetadataSourceQuery

logger = logging.getLogger(__name__)


class MetadataSourceAccessStrategy(ABC):
    """Abstract interface for metadata source access strategies."""

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate to the metadata source."""
        pass

    @abstractmethod
    async def get_metadata(self, query: MetadataSourceQuery) -> list[Metadata]:
        """Get metadata from the source."""
        pass


class EmailMetadataSourceAccess(MetadataSourceAccessStrategy):
    """Email implementation of MetadataSourceAccessStrategy."""

    def __init__(self, email_platform_strategy=None):
        self.email_platform_strategy = email_platform_strategy

    async def authenticate(self) -> bool:
        """Authenticate to email platform."""
        try:
            if not self.email_platform_strategy:
                logger.warning("Email platform strategy not configured")
                return False

            return await self.email_platform_strategy.authenticate()

        except Exception as e:
            logger.error(f"Email authentication error: {e}")
            raise RemoteServiceException(
                f"Email authentication failed: {str(e)}"
            ) from e

    async def get_metadata(self, query: MetadataSourceQuery) -> list[Metadata]:
        """Get email metadata."""
        try:
            if not self.email_platform_strategy:
                logger.warning("Email platform strategy not configured")
                return []

            # Ensure authentication
            if not await self.authenticate():
                raise UnauthorizedException("Email authentication required")

            # Delegate to email platform strategy
            return await self.email_platform_strategy.get_metadata(query)

        except (UnauthorizedException, RemoteServiceException):
            raise
        except Exception as e:
            logger.error(f"Error getting email metadata: {e}")
            raise RemoteServiceException(f"Email API error: {str(e)}") from e


class PredictionMetadataSourceAccess(MetadataSourceAccessStrategy):
    """Prediction implementation of MetadataSourceAccessStrategy."""

    def __init__(self, ml_engine=None):
        self.ml_engine = ml_engine

    async def authenticate(self) -> bool:
        """Authenticate to prediction service (always true for local ML)."""
        return True

    async def get_metadata(self, query: MetadataSourceQuery) -> list[Metadata]:
        """Get prediction metadata."""
        try:
            if not self.ml_engine:
                logger.warning("ML engine not configured")
                return []

            # For predictions, we would typically need transaction data
            # This is a placeholder implementation
            logger.info("Getting prediction metadata")

            # This would involve calling the ML engine to get predictions
            # and converting them to Metadata objects
            predictions = []

            return predictions

        except Exception as e:
            logger.error(f"Error getting prediction metadata: {e}")
            raise InternalException(f"Prediction error: {str(e)}") from e
