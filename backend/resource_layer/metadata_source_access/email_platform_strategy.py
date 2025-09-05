"""
EmailPlatformStrategy interface and implementations.

This module defines the strategy interface for email platform resources
and provides concrete implementations for each email platform type.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from resource_layer.metadata_source_access.email_resource_interface import (
    EmailSearchParams,
)
from thrift_gen.entities.ttypes import (
    EmailPlatformType,
    ExternalSystem,
    Metadata,
    MetadataType,
    MetadataValue,
)
from thrift_gen.exceptions.ttypes import (
    InternalException,
    RemoteServiceException,
    UnauthorizedException,
)
from thrift_gen.metadatasourceaccess.ttypes import MetadataSourceQuery

logger = logging.getLogger(__name__)


class EmailPlatformStrategy(ABC):
    """Abstract interface for email platform strategies."""

    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate to the email platform."""
        pass

    @abstractmethod
    async def get_metadata(self, query: MetadataSourceQuery) -> list[Metadata]:
        """Get metadata from the email platform."""
        pass


class GmailStrategy(EmailPlatformStrategy):
    """Gmail implementation of EmailPlatformStrategy."""

    def __init__(self, gmail_api_client=None):
        self.gmail_client = gmail_api_client

    async def authenticate(self) -> bool:
        """Authenticate to Gmail."""
        try:
            if not self.gmail_client:
                logger.warning(
                    "Gmail client not configured - Gmail features will be unavailable"
                )
                return False

            # Check if already authenticated using validate_credentials
            is_authenticated = await self.gmail_client.validate_credentials()
            if is_authenticated:
                logger.info("Gmail authentication successful")
                return True
            else:
                logger.info(
                    "Gmail authentication failed - no valid credentials available"
                )
                return False

        except Exception as e:
            logger.warning(f"Gmail authentication error: {e}")
            return False

    async def get_metadata(self, query: MetadataSourceQuery) -> list[Metadata]:
        """Get email metadata from Gmail."""
        try:
            if not self.gmail_client:
                logger.warning(
                    "Gmail client not configured - returning empty metadata list"
                )
                return []

            # Ensure authentication
            if not await self.authenticate():
                logger.warning(
                    "Gmail authentication not available - returning empty metadata list"
                )
                return []

            # Build search parameters using EmailSearchParams
            search_params_obj = EmailSearchParams(
                query=query.searchPhrase or "",
                start_date=datetime.fromtimestamp(query.startTime)
                if query.startTime
                else datetime.now() - timedelta(days=7),
                end_date=datetime.fromtimestamp(query.endTime)
                if query.endTime
                else datetime.now(),
                max_results=query.limit or 10,
            )

            # Search emails via Gmail API with retry logic
            gmail_emails = await self._search_with_retry(
                search_params_obj, query.searchPhrase
            )

            # Convert Gmail emails to Metadata objects
            metadata_items = []
            for email in gmail_emails:
                try:
                    # Convert Email object to dict for metadata storage
                    email_dict = {
                        "id": email.id,
                        "subject": email.subject,
                        "sender": email.sender,
                        "date": email.date.isoformat()
                        if hasattr(email.date, "isoformat")
                        else str(email.date),
                        "snippet": email.snippet,
                        "body": email.body_text,
                        "body_html": getattr(email, "body_html", ""),
                        "url": getattr(email, "url", ""),
                    }

                    metadata = self._convert_email_to_metadata(
                        email_dict, query.searchPhrase
                    )
                    metadata_items.append(metadata)
                except Exception as e:
                    logger.warning(f"Error converting email to metadata: {e}")
                    continue

            return metadata_items

        except (UnauthorizedException, RemoteServiceException):
            raise
        except Exception as e:
            logger.error(f"Error getting Gmail metadata: {e}")
            raise RemoteServiceException(f"Gmail API error: {str(e)}") from e

    def _convert_email_to_metadata(
        self, email_data: dict, search_phrase: str = None
    ) -> Metadata:
        """Convert Gmail email data to Metadata object."""
        try:
            # Create email content string
            email_content = self._format_email_content(email_data)

            # Create metadata value
            metadata_value = MetadataValue(stringValue=email_content)

            # Create external system reference
            external_system = ExternalSystem(emailPlatformType=EmailPlatformType.Gmail)

            # Create description with search phrase if available
            subject = email_data.get("subject", "No subject")
            sender = email_data.get("sender", "Unknown sender")
            date = email_data.get("date", "Unknown date")
            description = f"Email from {sender} - {subject} ({date})"

            if search_phrase:
                description += f" | Matched: {search_phrase}"

            # Create metadata object
            metadata = Metadata(
                type=MetadataType.Email,
                value=metadata_value,
                sourceSystem=external_system,
                description=description,
            )

            return metadata

        except Exception as e:
            logger.error(f"Error converting email to metadata: {e}")
            raise InternalException(f"Failed to convert email: {str(e)}") from e

    async def _search_with_retry(self, search_params_obj, search_phrase: str) -> list:
        """Search emails with retry logic for network issues."""
        max_retries = 2  # Keep at 2 for faster completion
        base_delay = 0.5  # Reduced to 0.5 second delay

        for attempt in range(max_retries):
            try:
                # Reduced timeout to fail faster and allow retries
                timeout = 15.0  # Reduced from 20 to 15 seconds

                gmail_emails = await asyncio.wait_for(
                    self.gmail_client.search_emails(search_params_obj), timeout=timeout
                )
                return gmail_emails

            except TimeoutError:
                if attempt == max_retries - 1:
                    logger.warning(
                        f"Gmail search timed out for query: {search_phrase} after {max_retries} attempts"
                    )
                    return []
                else:
                    delay = base_delay * (1.2**attempt)  # Gentler exponential backoff
                    logger.warning(
                        f"Gmail search timed out for query: {search_phrase}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)

            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a network/SSL error that might be retryable
                if any(
                    keyword in error_str
                    for keyword in [
                        "ssl",
                        "connection",
                        "timeout",
                        "network",
                        "read operation timed out",
                    ]
                ):
                    if attempt == max_retries - 1:
                        logger.warning(
                            f"Gmail search failed for query: {search_phrase} after {max_retries} attempts: {e}"
                        )
                        return []
                    else:
                        delay = base_delay * (
                            1.2**attempt
                        )  # Gentler exponential backoff
                        logger.warning(
                            f"Gmail search failed for query: {search_phrase}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        await asyncio.sleep(delay)
                else:
                    # Non-retryable error, log and return empty instead of raising
                    logger.warning(
                        f"Non-retryable Gmail search error for query: {search_phrase}: {e}"
                    )
                    return []

        return []

    def _format_email_content(self, email_data: dict) -> str:
        """Format email data as JSON string for storage."""
        try:
            # Store the email data as JSON for easy parsing later
            return json.dumps(email_data)
        except (TypeError, ValueError) as e:
            logger.error(f"Error formatting email content as JSON: {e}")
            # Fallback to basic text format
            try:
                content_parts = []

                # Add basic email info
                if email_data.get("subject"):
                    content_parts.append(f"Subject: {email_data['subject']}")

                if email_data.get("sender"):
                    content_parts.append(f"From: {email_data['sender']}")

                if email_data.get("date"):
                    content_parts.append(f"Date: {email_data['date']}")

                # Add email body content
                if email_data.get("body_text"):
                    content_parts.append(f"Body: {email_data['body_text']}")
                elif email_data.get("snippet"):
                    content_parts.append(f"Snippet: {email_data['snippet']}")

                # Add labels if available
                if email_data.get("labels"):
                    labels_str = ", ".join(email_data["labels"])
                    content_parts.append(f"Labels: {labels_str}")

                return "\n".join(content_parts)
            except Exception as fallback_e:
                logger.warning(f"Error formatting email content as text: {fallback_e}")
                return str(email_data)  # Final fallback to string representation
