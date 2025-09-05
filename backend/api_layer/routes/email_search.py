"""
Email search API routes.

This module provides endpoints for email search functionality:
API -> TransactionManager -> MetadataFindingEngine -> MetadataSourceAccess
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field

from api_layer.dependencies import get_transaction_manager
from configs import ConfigDefaults, ConfigKeys
from models.email import Email
from thrift_gen.entities.ttypes import (
    EmailPlatformType,
    ExternalSystem,
    Metadata,
    MetadataType,
    MetadataValue,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/email-search", tags=["email-search"])
# Simple per-transaction lock registry to avoid duplicate attachments from races
_transaction_locks: dict[str, asyncio.Lock] = {}


def _get_transaction_lock(transaction_id: str) -> asyncio.Lock:
    lock = _transaction_locks.get(transaction_id)
    if lock is None:
        # setdefault ensures only one is kept
        lock = _transaction_locks.setdefault(transaction_id, asyncio.Lock())
    return lock


def _parse_email_date(date_str) -> datetime:
    """Parse email date string to datetime object."""
    if isinstance(date_str, datetime):
        return date_str
    if isinstance(date_str, str):
        try:
            # Try parsing ISO format first
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            # Fallback to current time if parsing fails
            return datetime.utcnow()
    return datetime.utcnow()


# Request/Response Models


class EmailSearchResponse(BaseModel):
    """Response model for email search results."""

    emails: list[Email] = Field(..., description="Found emails")
    total_found: int = Field(..., description="Total number of emails found")


class EmailAttachRequest(BaseModel):
    """Request model for attaching email to transaction."""

    email_id: str = Field(..., description="Email ID to attach")
    email_subject: str = Field(..., description="Email subject")
    email_sender: str = Field(..., description="Email sender")
    email_date: str = Field(..., description="Email date")
    email_snippet: str = Field("", description="Email snippet")
    email_body_text: str = Field("", description="Email body text")
    email_body_html: str = Field("", description="Email body HTML")
    email_url: str = Field("", description="Email URL")


class EmailAttachResponse(BaseModel):
    """Response model for email attachment."""

    success: bool = Field(..., description="Whether attachment was successful")
    transaction_id: str = Field(..., description="Transaction ID")
    message: str = Field(..., description="Success or error message")


class EmailDetachRequest(BaseModel):
    """Request model for detaching email from transaction."""

    email_id: str = Field(..., description="Email ID to detach")


# API Endpoints


@router.get("/{transaction_id}/search", response_model=EmailSearchResponse)
async def search_emails_for_transaction(
    transaction_id: str = Path(...),
    q: str | None = Query(None),
    transaction_manager=Depends(get_transaction_manager),
):
    """
    Search for emails related to a specific transaction.

    This endpoint uses the proper architecture: API -> TransactionManager -> MetadataFindingEngine -> MetadataSourceAccess

    Args:
        transaction_id: Transaction ID to search emails for
        q: Optional custom search query to override automatic term extraction
    """
    try:
        logger.info(f"Searching emails for transaction: {transaction_id}")
        if q:
            logger.info(f"Using custom search query: {q}")

        # Use TransactionManager to find metadata
        metadata_list = await transaction_manager.findTransactionMetadata(
            transaction_id, q
        )

        # Convert metadata to Email objects for frontend compatibility
        emails = []
        for metadata in metadata_list:
            if (
                metadata.type == MetadataType.Email
                and metadata.value
                and metadata.value.stringValue
            ):
                try:
                    email_data = json.loads(metadata.value.stringValue)

                    # Extract matched terms from metadata description
                    matched_terms = ""
                    if metadata.description and "| Matched: " in metadata.description:
                        matched_terms = metadata.description.split("| Matched: ", 1)[1]

                    email = Email(
                        id=email_data.get("id", "unknown"),
                        thread_id=email_data.get("thread_id", "unknown"),
                        subject=email_data.get("subject", "No Subject"),
                        sender=email_data.get("sender", "Unknown Sender"),
                        date=_parse_email_date(
                            email_data.get("date", datetime.utcnow().isoformat())
                        ),
                        snippet=email_data.get("snippet", ""),
                        body_text=email_data.get("body_text", ""),
                        body_html=email_data.get("body_html", ""),
                        url=email_data.get(
                            "url",
                            f"https://mail.google.com/mail/u/0/#inbox/{email_data.get('id', '')}",
                        ),
                        matched_terms=matched_terms,
                    )
                    emails.append(email)
                except Exception as e:
                    logger.warning(f"Error parsing email metadata: {e}")

        logger.info(
            f"Found {len(emails)} email candidates for transaction {transaction_id}"
        )

        return EmailSearchResponse(
            emails=emails,
            total_found=len(emails),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Email search failed for transaction {transaction_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Email search failed: {str(e)}"
        ) from e


@router.post("/{transaction_id}/attach-email", response_model=EmailAttachResponse)
async def attach_email_to_transaction(
    transaction_id: str = Path(...),
    request: EmailAttachRequest = ...,
    transaction_manager=Depends(get_transaction_manager),
):
    """Attach an email to a transaction using provided email details."""
    try:
        # Serialize attach operations per transaction to avoid races
        async with _get_transaction_lock(transaction_id):
            # Prevent duplicate attachments by checking existing metadata for the same email ID
            try:
                existing = await transaction_manager.getTransaction(transaction_id)
                existing_ids = set()
                if existing and getattr(existing, "metadata", None):
                    for m in existing.metadata:
                        try:
                            if (
                                m.type == MetadataType.Email
                                and m.value
                                and m.value.stringValue
                            ):
                                data = json.loads(m.value.stringValue)
                                eid = data.get("id")
                                if eid:
                                    existing_ids.add(eid)
                        except Exception:
                            continue
                if request.email_id in existing_ids:
                    logger.info(
                        f"Email {request.email_id} already attached to transaction {transaction_id}, skipping duplicate attach"
                    )
                    return EmailAttachResponse(
                        success=True,
                        transaction_id=transaction_id,
                        message="Email already attached",
                    )
            except Exception as e:
                logger.warning(
                    f"Failed duplicate-check before attach for transaction {transaction_id}: {e}"
                )

            # Create MetadataValue with email information from request
            email_info = {
                "id": request.email_id,
                "subject": request.email_subject,
                "sender": request.email_sender,
                "date": request.email_date,
                "snippet": request.email_snippet,
                "body_text": request.email_body_text,
                "body_html": request.email_body_html,
                "url": request.email_url
                or f"https://mail.google.com/mail/u/0/#inbox/{request.email_id}",
                "attached": True,
                "platform": "gmail",
                "attached_at": datetime.utcnow().isoformat(),
            }
            metadata_value = MetadataValue(stringValue=json.dumps(email_info))

            # Create ExternalSystem for Gmail
            source_system = ExternalSystem(emailPlatformType=EmailPlatformType.Gmail)

            # Create the Metadata object
            email_metadata = Metadata(
                id=str(uuid.uuid4()),
                type=MetadataType.Email,
                value=metadata_value,
                sourceSystem=source_system,
                description=f"Attached email: {request.email_subject}",
            )

            # Use TransactionManager to attach metadata
            await transaction_manager.attachTransactionMetadata(
                transaction_id, email_metadata
            )

        logger.info(
            f"Email {request.email_id} successfully attached to transaction {transaction_id}"
        )

        return EmailAttachResponse(
            success=True,
            transaction_id=transaction_id,
            message="Email attached successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to attach email to transaction: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to attach email: {str(e)}"
        ) from e


@router.post("/{transaction_id}/detach-email", response_model=EmailAttachResponse)
async def detach_email_from_transaction(
    transaction_id: str = Path(...),
    request: EmailDetachRequest = ...,
    transaction_manager=Depends(get_transaction_manager),
):
    """Detach an email from a transaction."""
    try:
        # Get the transaction to find the metadata ID
        transaction = await transaction_manager.getTransaction(transaction_id)
        if not transaction or not transaction.metadata:
            raise HTTPException(
                status_code=404,
                detail=f"No email attachments found for transaction {transaction_id}",
            )

        # Find the metadata ID for the email to detach
        metadata_id_to_detach = None
        for metadata in transaction.metadata:
            if (
                metadata.type == MetadataType.Email
                and metadata.value
                and metadata.value.stringValue
            ):
                try:
                    email_data = json.loads(metadata.value.stringValue)
                    if email_data.get("id") == request.email_id:
                        metadata_id_to_detach = metadata.id
                        break
                except Exception as e:
                    logger.warning(f"Error parsing email metadata: {e}")

        if not metadata_id_to_detach:
            raise HTTPException(
                status_code=404,
                detail=f"Email {request.email_id} not found attached to transaction {transaction_id}",
            )

        # Use TransactionManager to detach metadata
        await transaction_manager.detachTransactionMetadata(
            transaction_id, metadata_id_to_detach
        )

        logger.info(
            f"Email {request.email_id} successfully detached from transaction {transaction_id}"
        )

        return EmailAttachResponse(
            success=True,
            transaction_id=transaction_id,
            message="Email detached successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to detach email from transaction: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to detach email: {str(e)}"
        ) from e


class BulkEmailSearchRequest(BaseModel):
    """Request model for bulk email search."""

    transaction_ids: list[str] = Field(
        ..., description="List of transaction IDs to search emails for"
    )


class BulkEmailSearchResponse(BaseModel):
    """Response model for bulk email search results."""

    results: dict[str, dict] = Field(..., description="Results per transaction ID")
    total_processed: int = Field(..., description="Total transactions processed")
    total_attached: int = Field(..., description="Total emails auto-attached")


async def _search_single_transaction(transaction_manager, transaction_id: str) -> dict:
    """
    Search emails for a single transaction (async task).

    Returns a dict with the search result for this transaction.
    """
    try:
        # Get the transaction first
        transaction = await transaction_manager.getTransaction(transaction_id)
        if not transaction:
            return {
                "status": "error",
                "error": "Transaction not found",
                "emails": [],
                "attached": False,
            }

        # Search for emails using the existing logic with timeout
        try:
            metadata_list = await asyncio.wait_for(
                transaction_manager.findTransactionMetadata(
                    transaction_id,
                    None,  # Let it use automatic term extraction
                ),
                timeout=20.0,  # 20 second timeout per transaction
            )
        except TimeoutError:
            logger.warning(f"Email search timed out for transaction {transaction_id}")
            return {
                "status": "error",
                "error": "Timeout",
                "emails": [],
                "attached": False,
            }

        # Convert metadata to Email objects
        emails = []
        for metadata in metadata_list:
            if (
                metadata.type == MetadataType.Email
                and metadata.value
                and metadata.value.stringValue
            ):
                try:
                    email_data = json.loads(metadata.value.stringValue)

                    # Extract matched terms from metadata description
                    matched_terms = ""
                    if metadata.description and "| Matched: " in metadata.description:
                        matched_terms = metadata.description.split("| Matched: ", 1)[1]

                    email = Email(
                        id=email_data.get("id", "unknown"),
                        thread_id=email_data.get("thread_id", "unknown"),
                        subject=email_data.get("subject", "No Subject"),
                        sender=email_data.get("sender", "Unknown Sender"),
                        date=_parse_email_date(
                            email_data.get("date", datetime.utcnow().isoformat())
                        ),
                        snippet=email_data.get("snippet", ""),
                        body_text=email_data.get("body_text", ""),
                        body_html=email_data.get("body_html", ""),
                        url=email_data.get(
                            "url",
                            f"https://mail.google.com/mail/u/0/#inbox/{email_data.get('id', '')}",
                        ),
                        matched_terms=matched_terms,
                    )
                    emails.append(email)
                except Exception as e:
                    logger.warning(f"Error parsing email metadata: {e}")

        # Determine attached status and optionally auto-attach when exactly 1 result
        attached = False
        if len(emails) == 1:
            try:
                email_to_attach = emails[0]

                # Check existing attachments first using current transaction snapshot
                existing_attached_ids = set()
                try:
                    if transaction and getattr(transaction, "metadata", None):
                        for m in transaction.metadata:
                            if (
                                m.type == MetadataType.Email
                                and m.value
                                and m.value.stringValue
                            ):
                                try:
                                    data = json.loads(m.value.stringValue)
                                    eid = data.get("id")
                                    if eid:
                                        existing_attached_ids.add(eid)
                                except Exception:
                                    continue
                except Exception as e:
                    logger.warning(
                        f"Failed to inspect existing attachments for {transaction_id}: {e}"
                    )

                if email_to_attach.id in existing_attached_ids:
                    # Already attached; reflect status and skip
                    attached = True
                else:
                    # Respect config for auto-attach
                    try:
                        auto_attach_enabled = (
                            await transaction_manager.config_service.getConfigValue(
                                ConfigKeys.EMAIL_BULK_AUTO_ATTACH_SINGLE_RESULT,
                                ConfigDefaults.BULK_AUTO_ATTACH_SINGLE_RESULT,
                            )
                        )
                    except Exception as ce:
                        logger.warning(
                            f"Failed to read auto-attach config, using default: {ce}"
                        )
                        auto_attach_enabled = (
                            ConfigDefaults.BULK_AUTO_ATTACH_SINGLE_RESULT
                        )

                    if auto_attach_enabled:
                        # Serialize auto-attach to avoid duplicate attachments due to races
                        async with _get_transaction_lock(transaction_id):
                            # Refresh transaction metadata under lock to re-check duplicates
                            try:
                                transaction = await transaction_manager.getTransaction(
                                    transaction_id
                                )
                            except Exception:
                                pass

                            # Rebuild existing IDs under lock
                            existing_attached_ids = set()
                            try:
                                if transaction and getattr(
                                    transaction, "metadata", None
                                ):
                                    for m in transaction.metadata:
                                        if (
                                            m.type == MetadataType.Email
                                            and m.value
                                            and m.value.stringValue
                                        ):
                                            try:
                                                data = json.loads(m.value.stringValue)
                                                eid = data.get("id")
                                                if eid:
                                                    existing_attached_ids.add(eid)
                                            except Exception:
                                                continue
                            except Exception as e:
                                logger.warning(
                                    f"Failed to inspect existing attachments (under lock) for {transaction_id}: {e}"
                                )

                            if email_to_attach.id in existing_attached_ids:
                                attached = True
                                logger.info(
                                    f"Email {email_to_attach.id} already attached to transaction {transaction_id}, skipping"
                                )
                            else:
                                # Create MetadataValue with email information
                                email_info = {
                                    "id": email_to_attach.id,
                                    "subject": email_to_attach.subject,
                                    "sender": email_to_attach.sender,
                                    "date": email_to_attach.date.isoformat()
                                    if isinstance(email_to_attach.date, datetime)
                                    else email_to_attach.date,
                                    "snippet": email_to_attach.snippet,
                                    "body_text": email_to_attach.body_text,
                                    "body_html": email_to_attach.body_html,
                                    "url": email_to_attach.url,
                                    "attached": True,
                                    "platform": "gmail",
                                    "attached_at": datetime.utcnow().isoformat(),
                                }
                                metadata_value = MetadataValue(
                                    stringValue=json.dumps(email_info)
                                )

                                # Create ExternalSystem for Gmail
                                source_system = ExternalSystem(
                                    emailPlatformType=EmailPlatformType.Gmail
                                )

                                # Create the Metadata object
                                email_metadata = Metadata(
                                    id=str(uuid.uuid4()),
                                    type=MetadataType.Email,
                                    value=metadata_value,
                                    sourceSystem=source_system,
                                    description=f"Auto-attached email: {email_to_attach.subject}",
                                )

                                # Attach the email
                                await transaction_manager.attachTransactionMetadata(
                                    transaction_id, email_metadata
                                )

                                attached = True
                                logger.info(
                                    f"Auto-attached email {email_to_attach.id} to transaction {transaction_id}"
                                )
                    else:
                        # Auto-attach disabled; do not attach
                        attached = False
            except Exception as e:
                logger.warning(
                    f"Failed processing auto-attach logic for transaction {transaction_id}: {e}"
                )

        # Return results
        return {
            "status": "success",
            "emails": [
                {
                    "id": email.id,
                    "subject": email.subject,
                    "sender": email.sender,
                    "date": email.date,
                    "snippet": email.snippet,
                    "matched_terms": email.matched_terms,
                }
                for email in emails
            ],
            "email_count": len(emails),
            "attached": attached,
        }
    except Exception as e:
        logger.error(f"Error processing transaction {transaction_id}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "emails": [],
            "attached": False,
        }


@router.post("/bulk-search", response_model=BulkEmailSearchResponse)
async def bulk_search_emails_for_transactions(
    request: BulkEmailSearchRequest,
    transaction_manager=Depends(get_transaction_manager),
):
    """
    Bulk search for emails related to multiple transactions.

    This endpoint processes multiple transactions in parallel using asyncio tasks.
    Auto-attaches emails when there's exactly one result and the setting is enabled.
    """
    try:
        # Dedupe transaction IDs to avoid double-processing the same transaction
        transaction_ids = list(dict.fromkeys(request.transaction_ids))
        logger.info(f"Bulk searching emails for {len(transaction_ids)} transactions")

        # Create tasks for parallel processing
        tasks = [
            _search_single_transaction(transaction_manager, transaction_id)
            for transaction_id in transaction_ids
        ]

        # Run all searches in parallel with a reasonable timeout
        # Reduced timeout since individual searches now fail faster
        try:
            task_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=45.0,  # Reduced to 45 seconds for the entire batch
            )
        except TimeoutError:
            logger.warning(
                f"Bulk search timed out for batch of {len(transaction_ids)} transactions"
            )
            # Return partial results for transactions that might have completed
            task_results = [
                {"status": "error", "error": "Timeout", "emails": [], "attached": False}
                for _ in transaction_ids
            ]

        # Process results
        results = {}
        total_attached = 0

        for i, result in enumerate(task_results):
            transaction_id = transaction_ids[i]

            if isinstance(result, Exception):
                logger.error(f"Task failed for transaction {transaction_id}: {result}")
                results[transaction_id] = {
                    "status": "error",
                    "error": str(result),
                    "emails": [],
                    "attached": False,
                }
            else:
                results[transaction_id] = result
                if result.get("attached", False):
                    total_attached += 1

        logger.info(
            f"Bulk search completed: {len(transaction_ids)} processed, {total_attached} auto-attached"
        )

        return BulkEmailSearchResponse(
            results=results,
            total_processed=len(transaction_ids),
            total_attached=total_attached,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk email search failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Bulk email search failed: {str(e)}"
        ) from e
