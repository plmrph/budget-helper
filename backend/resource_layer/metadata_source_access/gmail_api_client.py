"""
Gmail API client implementation for email resource access.
"""

import asyncio
import base64
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from configs import ConfigKeys, ConfigService
from models.email import Email
from resource_layer.metadata_source_access.email_resource_interface import (
    EmailResourceAccess,
    EmailSearchParams,
)

logger = logging.getLogger(__name__)


class GmailApiClient(EmailResourceAccess):
    """Gmail API client implementation."""

    _instance = None
    _initialized = False

    def __new__(cls, config_service: ConfigService = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_service: ConfigService = None):
        if not self._initialized:
            self.executor = ThreadPoolExecutor(max_workers=2)
            self.service = None
            self.config_service = config_service
            self.oauth_config = None
            self.credentials = None
            self._credentials_validated = False  # Cache validation state
            self._last_validation_time = None
            self._concurrency_semaphore = asyncio.Semaphore(2)
            self._service_lock = threading.Lock()
            self._initialized = True
        elif config_service and not self.config_service:
            # Update config service if not set
            self.config_service = config_service

    async def _load_oauth_config(self):
        """Load OAuth configuration from database."""
        if not self.config_service:
            logger.warning("Config service not available for Gmail client")
            return None

        try:
            config_value = await self.config_service.getConfigValue(
                ConfigKeys.EMAIL_GMAIL_AUTH_CONFIG
            )

            if not config_value:
                logger.info("No Gmail OAuth configuration found in database")
                return None

            # Parse the JSON configuration
            oauth_config = json.loads(config_value)
            logger.info("Gmail OAuth configuration loaded from database")
            return oauth_config

        except Exception as e:
            logger.error(f"Error loading Gmail OAuth config: {e}")
            return None

    async def _load_oauth_tokens(self):
        """Load OAuth tokens from database if they exist."""
        if not self.config_service:
            logger.warning("Config service not available for Gmail client")
            return None

        try:
            tokens_value = await self.config_service.getConfigValue(
                ConfigKeys.EMAIL_GMAIL_TOKENS
            )

            if not tokens_value:
                logger.info("No Gmail OAuth tokens found in database")
                return None

            # Parse the JSON tokens
            oauth_tokens = json.loads(tokens_value)
            logger.info("Gmail OAuth tokens loaded from database")
            return oauth_tokens

        except Exception as e:
            logger.error(f"Error loading Gmail OAuth tokens: {e}")
            return None

    async def validate_credentials(self) -> bool:
        """Validate if service is working with current credentials."""
        try:
            # Check if we already validated recently (cache for 5 minutes)
            import time

            current_time = time.time()
            if (
                self._credentials_validated
                and self._last_validation_time
                and current_time - self._last_validation_time < 300
            ):  # 5 minutes
                return True

            # Load OAuth configuration
            if not self.oauth_config:
                self.oauth_config = await self._load_oauth_config()

            if not self.oauth_config:
                logger.info("No Gmail OAuth configuration available")
                self._credentials_validated = False
                return False

            # Load OAuth tokens
            oauth_tokens = await self._load_oauth_tokens()
            if not oauth_tokens:
                logger.info("No Gmail OAuth tokens available - authentication required")
                self._credentials_validated = False
                return False

            # Initialize service with tokens
            if not self.service:
                self.credentials = Credentials(
                    token=oauth_tokens.get("access_token"),
                    refresh_token=oauth_tokens.get("refresh_token"),
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=self.oauth_config.get("client_id"),
                    client_secret=self.oauth_config.get("client_secret"),
                    scopes=self.oauth_config.get("scopes", []),
                )
                self.service = self._build_service_sync(self.credentials)

            # Test credentials by making a simple API call
            loop = asyncio.get_event_loop()

            def _profile_call():
                # Serialize calls into the google service to avoid thread-safety issues
                with self._service_lock:
                    return self.service.users().getProfile(userId="me").execute()

            await loop.run_in_executor(self.executor, _profile_call)

            # Cache successful validation
            self._credentials_validated = True
            self._last_validation_time = current_time
            return True

        except Exception as e:
            logger.warning(f"Gmail credentials validation failed: {e}")
            self._credentials_validated = False
            return False

    async def search_emails(self, params: EmailSearchParams) -> list[Email]:
        """Search for emails based on parameters."""
        try:
            async with self._concurrency_semaphore:
                # Validate credentials first
                if not await self.validate_credentials():
                    logger.warning("Gmail credentials not valid - cannot search emails")
                    return []

                if self.service is None:
                    logger.warning(
                        "Gmail service not initialized - cannot search emails"
                    )
                    return []

                # Build Gmail search query
                query = await self._build_search_query(params)
                logger.info(f"Gmail search query: {query}")

                # Execute search in thread pool to avoid blocking with timeout
                loop = asyncio.get_event_loop()
                try:
                    messages = await asyncio.wait_for(
                        loop.run_in_executor(
                            self.executor,
                            self._search_messages,
                            self.service,
                            query,
                            params.max_results,
                        ),
                        timeout=12.0,  # Reduced to 12 seconds to fail faster
                    )
                except TimeoutError:
                    logger.warning("Gmail search timed out after 12 seconds")
                    raise Exception(
                        "Gmail search timed out. Please try again."
                    ) from None

                # Get full message details
                emails = []
                for message in messages:
                    email = await loop.run_in_executor(
                        self.executor,
                        self._get_message_details,
                        self.service,
                        message["id"],
                    )
                    if email:
                        emails.append(email)

                logger.info(f"Gmail search returned {len(emails)} emails")
                return emails

        except HttpError as e:
            if e.resp.status == 429:  # Rate limit exceeded
                logger.error("Gmail API rate limit exceeded")
                raise Exception("Rate limit exceeded. Please try again later.") from e
            elif e.resp.status == 401:  # Unauthorized
                logger.error("Gmail API authentication failed")
                raise Exception(
                    "Gmail authentication required. Please re-authenticate."
                ) from e
            else:
                logger.error(f"Gmail API error: {e}")
                raise Exception(f"Gmail API error: {e}") from e
        except Exception as e:
            logger.error(f"Gmail email search error: {e}")
            raise

    async def _build_search_query(self, params: EmailSearchParams) -> str:
        """Build Gmail search query from parameters."""
        query_parts = []

        # Add base query
        if params.query:
            query_parts.append(params.query)

        # Add date range
        start_date_str = params.start_date.strftime("%Y/%m/%d")
        end_date_str = params.end_date.strftime("%Y/%m/%d")
        query_parts.append(f"after:{start_date_str}")
        query_parts.append(f"before:{end_date_str}")

        return " ".join(query_parts)

    def _search_messages(
        self, service, query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Search for messages (synchronous) with retry logic."""
        max_retries = 2
        base_delay = 0.5  # Reduced delay

        for attempt in range(max_retries):
            try:
                logger.info(f"Executing Gmail API search with query: {query}")
                with self._service_lock:
                    result = (
                        service.users()
                        .messages()
                        .list(userId="me", q=query, maxResults=max_results)
                        .execute()
                    )

                messages = result.get("messages", [])
                logger.info(f"Gmail API returned {len(messages)} messages")
                return messages

            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a retryable network/SSL error
                if attempt < max_retries - 1 and any(
                    keyword in error_str
                    for keyword in [
                        "ssl",
                        "connection",
                        "timeout",
                        "network",
                        "read operation timed out",
                    ]
                ):
                    delay = base_delay * (1.5**attempt)  # Gentler backoff
                    logger.warning(
                        f"Gmail API search failed, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    import time

                    time.sleep(delay)
                else:
                    logger.warning(f"Message search error (returning empty): {e}")
                    return []  # Return empty instead of raising

        return []

    def _get_message_details(self, service, message_id: str) -> Email | None:
        """Get full message details (synchronous) with retry logic."""
        max_retries = 2
        base_delay = 0.3  # Reduced delay

        for attempt in range(max_retries):
            try:
                logger.debug(f"Getting details for message: {message_id}")
                with self._service_lock:
                    message = (
                        service.users()
                        .messages()
                        .get(userId="me", id=message_id, format="full")
                        .execute()
                    )

                return self._parse_message(message)

            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a retryable network/SSL error
                if attempt < max_retries - 1 and any(
                    keyword in error_str
                    for keyword in [
                        "ssl",
                        "connection",
                        "timeout",
                        "network",
                        "read operation timed out",
                    ]
                ):
                    delay = base_delay * (1.3**attempt)  # Gentler backoff
                    logger.warning(
                        f"Gmail message details failed for {message_id}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    import time

                    time.sleep(delay)
                else:
                    logger.warning(
                        f"Message details error for {message_id} (skipping): {e}"
                    )
                    return None  # Return None for individual message failures, don't crash the whole search

        return None

    def _parse_message(self, message: dict[str, Any]) -> Email:
        """Parse Gmail message into Email model."""
        headers = {h["name"]: h["value"] for h in message["payload"].get("headers", [])}

        # Extract basic info
        email_id = message["id"]
        thread_id = message["threadId"]
        subject = headers.get("Subject", "")
        sender_raw = headers.get("From", "")
        date_str = headers.get("Date", "")

        # Parse sender to extract just the email address or name
        sender = self._parse_sender(sender_raw)

        # Parse date
        try:
            # Gmail date format can vary, try common formats
            date = parsedate_to_datetime(date_str)
        except Exception:
            date = datetime.now()

        # Extract body
        body_text, body_html = self._extract_body(message["payload"])

        # Create Gmail URL
        url = f"https://mail.google.com/mail/u/0/#inbox/{email_id}"

        # Extract labels
        labels = message.get("labelIds", [])

        # Get snippet
        snippet = message.get("snippet", "")

        return Email(
            id=email_id,
            thread_id=thread_id,
            subject=subject,
            sender=sender,
            date=date,
            snippet=snippet,
            body_text=body_text,
            body_html=body_html,
            url=url,
            labels=labels,
        )

    def _parse_sender(self, sender_raw: str) -> str:
        """Parse sender field to extract readable name or email."""
        if not sender_raw:
            return "Unknown"

        # Handle formats like "Name <email@domain.com>" or just "email@domain.com"
        # Try to extract name from "Name <email>" format
        name_match = re.match(r"^([^<]+)<([^>]+)>$", sender_raw.strip())
        if name_match:
            name = name_match.group(1).strip().strip('"')
            if name:
                return name
            # Fall back to email if name is empty
            return name_match.group(2).strip()

        # If it's just an email address, return it
        if "@" in sender_raw:
            return sender_raw.strip()

        # Otherwise return as-is
        return sender_raw.strip() or "Unknown"

    def _extract_body(self, payload: dict[str, Any]) -> tuple[str, str]:
        """Extract text and HTML body from message payload."""
        body_text = ""
        body_html = ""

        def extract_from_part(part):
            nonlocal body_text, body_html

            mime_type = part.get("mimeType", "")

            if mime_type == "text/plain":
                data = part.get("body", {}).get("data", "")
                if data:
                    body_text = base64.urlsafe_b64decode(data).decode(
                        "utf-8", errors="ignore"
                    )
            elif mime_type == "text/html":
                data = part.get("body", {}).get("data", "")
                if data:
                    body_html = base64.urlsafe_b64decode(data).decode(
                        "utf-8", errors="ignore"
                    )
            elif mime_type.startswith("multipart/"):
                for subpart in part.get("parts", []):
                    extract_from_part(subpart)

        extract_from_part(payload)
        return body_text, body_html

    async def _refresh_credentials(self):
        """Refresh credentials (async wrapper)."""
        loop = asyncio.get_event_loop()

        def _do_refresh():
            # Ensure only one refresh happens at a time against shared creds/transport
            with self._service_lock:
                self.credentials.refresh(Request())

        await loop.run_in_executor(self.executor, _do_refresh)

    def _build_service_sync(self, credentials: Credentials):
        """Build Gmail service (synchronous)."""
        # Disable discovery cache to avoid file-system/cache races in containers
        return build("gmail", "v1", credentials=credentials, cache_discovery=False)


# Singleton instance getter
def get_gmail_client(config_service: ConfigService = None) -> GmailApiClient:
    """Get the singleton Gmail client instance."""
    return GmailApiClient(config_service=config_service)
