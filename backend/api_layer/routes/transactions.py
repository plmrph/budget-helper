"""
Transaction API routes.

This module provides endpoints for transaction management including
CRUD operations, filtering, pagination, batch updates, and history tracking.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field, field_validator

from api_layer.dependencies import TransactionManagerDep
from business_layer.transaction_manager import TransactionManager
from configs import ConfigKeys, FieldNames
from thrift_gen.databasestoreaccess.ttypes import (
    Filter,
    FilterOperator,
    FilterValue,
    Sort,
    SortDirection,
)
from thrift_gen.databasestoreaccess.ttypes import (
    Query as ThriftQuery,
)

# Import generated Thrift types
from thrift_gen.entities.ttypes import Entity, EntityType
from thrift_gen.exceptions.ttypes import (
    ConflictException,
    InternalException,
    NotFoundException,
    UnauthorizedException,
    ValidationException,
)
from thrift_gen.transactionmanager.ttypes import TransactionEdit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/transactions", tags=["transactions"])


# Request/Response Models (keeping Pydantic for API layer during transition)


class TransactionUpdateRequest(BaseModel):
    """Request model for updating a transaction."""

    category_id: str | None = Field(
        None, description="Category ID for this transaction"
    )
    payee_name: str | None = Field(None, description="Name of the payee")
    memo: str | None = Field(None, description="Transaction memo/description")
    approved: bool | None = Field(None, description="Whether transaction is approved")
    cleared: str | None = Field(
        None, description="Cleared status: cleared, uncleared, reconciled"
    )

    @field_validator("cleared")
    def validate_cleared_status(cls, v):
        """Validate cleared status is one of the allowed values."""
        if v is not None:
            allowed_values = ["cleared", "uncleared", "reconciled"]
            if v not in allowed_values:
                raise ValueError(
                    f"Cleared status must be one of: {', '.join(allowed_values)}"
                )
        return v


class BatchTransactionUpdate(BaseModel):
    """Single transaction update in a batch operation."""

    transaction_id: str = Field(..., description="ID of the transaction to update")
    updates: TransactionUpdateRequest = Field(..., description="Updates to apply")


class BatchTransactionRequest(BaseModel):
    """Request model for batch transaction updates."""

    updates: list[BatchTransactionUpdate] = Field(
        ..., description="List of transaction updates"
    )
    user_action: str = Field(
        default="Batch update transactions",
        description="Description of the batch operation",
    )


class TransactionResponse(BaseModel):
    """Standard transaction response."""

    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    data: dict[str, Any] | None = Field(None, description="Transaction data")


class TransactionListResponse(BaseModel):
    """Response model for transaction list."""

    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    data: list[dict[str, Any]] = Field(..., description="List of transactions")
    pagination: dict[str, Any] = Field(..., description="Pagination information")


class BatchTransactionResponse(BaseModel):
    """Response model for batch transaction operations."""

    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    data: list[dict[str, Any]] = Field(..., description="Updated transactions")
    errors: list[dict[str, Any]] = Field(
        default_factory=list, description="Any errors that occurred"
    )


class TransactionHistoryResponse(BaseModel):
    """Response model for transaction history."""

    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    data: list[dict[str, Any]] = Field(..., description="Transaction history entries")


class DiffChange(BaseModel):
    """Represents a single field change in a transaction diff."""

    field: str
    from_value: Any | None
    to_value: Any | None


class TransactionDiffItem(BaseModel):
    """Represents a diff for a single transaction between local and remote."""

    transaction_id: str
    action: str = Field(..., description="add | update | delete | unchanged")
    local: dict[str, Any] | None = None
    remote: dict[str, Any] | None = None
    changes: list[DiffChange] = Field(default_factory=list)


class DiffPreviewResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any]

    # data: { summary: {...}, diffs: TransactionDiffItem[] }


class DiffApplyRequest(BaseModel):
    """Selection of diffs to apply by action group."""

    add: list[str] = Field(default_factory=list)
    update: list[str] = Field(default_factory=list)
    delete: list[str] = Field(default_factory=list)


class UnifiedPreviewResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any]


class UnifiedApplyRequest(BaseModel):
    plan: list[dict[str, Any]] = Field(
        default_factory=list, description="[{id, action: 'left'|'right'}]"
    )


class ResetTrackingResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, Any] | None = None


def thrift_to_dict(thrift_obj):
    """Convert a Thrift object to a dictionary for JSON serialization."""
    if thrift_obj is None:
        return None

    # Handle primitive types
    if isinstance(thrift_obj, str | int | float | bool):
        return thrift_obj

    # Handle lists
    if isinstance(thrift_obj, list):
        return [thrift_to_dict(item) for item in thrift_obj]

    # Handle dict/map types (Thrift maps are represented as Python dict)
    if isinstance(thrift_obj, dict):
        return {k: thrift_to_dict(v) for k, v in thrift_obj.items()}

    # Handle Thrift objects
    if hasattr(thrift_obj, "__dict__"):
        result = {}
        # Use thrift_spec if available for better field detection
        if hasattr(thrift_obj, "thrift_spec") and thrift_obj.thrift_spec:
            for field_spec in thrift_obj.thrift_spec:
                if field_spec is None:
                    continue
                field_id, field_type, field_name = field_spec[:3]
                if hasattr(thrift_obj, field_name):
                    attr_value = getattr(thrift_obj, field_name)
                    if attr_value is not None:
                        result[field_name] = thrift_to_dict(attr_value)
        else:
            # Fallback to __dict__ inspection
            for attr_name, attr_value in thrift_obj.__dict__.items():
                if not attr_name.startswith("_") and attr_value is not None:
                    result[attr_name] = thrift_to_dict(attr_value)
        return result

    # Return as-is for other types
    return thrift_obj


# Simple response helpers - no Pydantic models needed


def success_response(message: str, data: Any = None) -> dict[str, Any]:
    """Create a success response."""
    response = {"success": True, "message": message}
    if data is not None:
        response["data"] = data
    return response


def error_response(message: str, status_code: int = 400) -> dict[str, Any]:
    """Create an error response."""
    return {"success": False, "message": message, "status_code": status_code}


# Transaction CRUD Endpoints


@router.get("/")
async def list_transactions(
    # Filtering parameters
    account_id: str | None = Query(None),
    category_id: str | None = Query(None),
    payee_name: str | None = Query(None),
    approved: bool | None = Query(None),
    cleared: str | None = Query(None),
    deleted: bool | None = Query(False),
    has_email: bool | None = Query(None),
    has_ai_category: bool | None = Query(None),
    # Date range filtering
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    # Amount filtering
    amount_min: int | None = Query(None),
    amount_max: int | None = Query(None),
    # Pagination parameters
    limit: int | None = Query(None),
    offset: int = Query(0),
    # Sorting
    sort_by: str = Query("date"),
    sort_order: str = Query("desc"),
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """
    List transactions with filtering, pagination, and sorting.

    This endpoint supports comprehensive filtering by various transaction fields,
    date ranges, amounts, and metadata. Results are paginated and sortable.
    """
    try:
        logger.info(
            f"Listing transactions with filters: account_id={account_id}, limit={limit}, offset={offset}"
        )

        # If limit not provided, fall back to configured default
        if limit is None:
            try:
                from api_layer.dependencies import get_configs

                config_service = await get_configs()
                cfg_default = await config_service.getConfigValue(
                    ConfigKeys.SYSTEM_MAX_TRANSACTIONS_TO_LOAD, 500
                )
                # Ensure int and within sane bounds
                try:
                    parsed = int(cfg_default)
                except Exception:
                    parsed = 500
                limit = max(1, min(parsed, 10000))
            except Exception as e:
                logger.warning(f"Failed to load default limit from settings: {e}")
                limit = 50

        # Build filters dictionary
        filters = {}

        if account_id:
            filters["account_id"] = account_id
        if category_id:
            filters["category_id"] = category_id
        if payee_name:
            filters["payee_name_like"] = payee_name  # Partial match
        if approved is not None:
            filters["approved"] = approved
        if cleared:
            filters["cleared"] = cleared
        if deleted is not None:
            filters["deleted"] = deleted
        if has_email is not None:
            if has_email:
                filters["selected_email_not_null"] = True
            else:
                filters["selected_email_null"] = True
        if has_ai_category is not None:
            if has_ai_category:
                filters["ai_category_not_null"] = True
            else:
                filters["ai_category_null"] = True

        # Date range filters
        if date_from:
            filters["date_gte"] = date_from
        if date_to:
            filters["date_lte"] = date_to

        # Amount range filters
        if amount_min is not None:
            filters["amount_gte"] = amount_min
        if amount_max is not None:
            filters["amount_lte"] = amount_max

        # Add sorting to filters
        if sort_by in [
            "date",
            "amount",
            "payee_name",
            "approved",
            "created_at",
            "updated_at",
        ]:
            filters["sort_by"] = sort_by
            filters["sort_order"] = (
                sort_order.lower() if sort_order.lower() in ["asc", "desc"] else "desc"
            )

        # Get selected budget ID from config service
        selected_budget_id = None
        try:
            from api_layer.dependencies import get_configs
            from thrift_gen.entities.ttypes import ConfigType

            config_service = await get_configs()
            configs = await config_service.getConfigs(
                ConfigType.System, "system.selected_budget_id"
            )
            if configs:
                config_item = configs[0]
                if config_item.value and config_item.value.stringValue:
                    selected_budget_id = config_item.value.stringValue
                    logger.info(
                        f"Filtering transactions by selected budget: {selected_budget_id}"
                    )
        except Exception as e:
            logger.warning(f"Could not get selected budget ID: {e}")

        # Build all filters for transaction query
        transaction_filters = []

        # Add budget filter if selected
        if selected_budget_id:
            budget_filter_value = FilterValue(stringValue=selected_budget_id)
            budget_filter = Filter(
                fieldName=FieldNames.BUDGET_ID,
                operator=FilterOperator.EQ,
                value=budget_filter_value,
            )
            transaction_filters.append(budget_filter)

        # Add account filter
        if account_id:
            account_filter_value = FilterValue(stringValue=account_id)
            account_filter = Filter(
                fieldName=FieldNames.ACCOUNT_ID,
                operator=FilterOperator.EQ,
                value=account_filter_value,
            )
            transaction_filters.append(account_filter)

        # Add category filter
        if category_id:
            category_filter_value = FilterValue(stringValue=category_id)
            category_filter = Filter(
                fieldName=FieldNames.CATEGORY_ID,
                operator=FilterOperator.EQ,
                value=category_filter_value,
            )
            transaction_filters.append(category_filter)

        # Add payee name filter (using LIKE for partial match)
        if payee_name:
            payee_filter_value = FilterValue(stringValue=f"%{payee_name}%")
            payee_filter = Filter(
                fieldName=FieldNames.PAYEE_ID,
                operator=FilterOperator.LIKE,
                value=payee_filter_value,
            )
            transaction_filters.append(payee_filter)

        # Add approved filter
        if approved is not None:
            approved_filter_value = FilterValue(boolValue=approved)
            approved_filter = Filter(
                fieldName=FieldNames.APPROVED,
                operator=FilterOperator.EQ,
                value=approved_filter_value,
            )
            transaction_filters.append(approved_filter)

        # Build sorting for database query

        sort_items = []
        if sort_by in ["date", "amount", "approved"]:
            # Map API sort field to database column
            db_sort_field = sort_by
            if sort_by == "date":
                db_sort_field = "date"
            elif sort_by == "amount":
                db_sort_field = "amount"
            elif sort_by == "approved":
                db_sort_field = "approved"

            sort_direction = (
                SortDirection.DESC
                if sort_order.lower() == "desc"
                else SortDirection.ASC
            )
            sort_item = Sort(fieldName=db_sort_field, direction=sort_direction)
            sort_items.append(sort_item)

        # Create query with filters, sorting, and pagination at database level
        query = ThriftQuery(
            entityType=EntityType.Transaction,
            filters=transaction_filters,
            sort=sort_items if sort_items else None,
            limit=limit,
            offset=offset,
        )

        # Get transactions with database-level filtering, sorting, and pagination
        result = await transaction_manager.database_store_access.getEntities(query)

        transactions = []
        for entity in result.entities:
            if entity.transaction:
                transactions.append(entity.transaction)

        # Get total count from query result
        total_count = result.totalCount

        # Convert Thrift objects to dictionaries for JSON response
        transaction_dicts = [thrift_to_dict(t) for t in transactions]

        # Calculate pagination info
        has_next = (offset + limit) < total_count
        has_prev = offset > 0
        total_pages = (total_count + limit - 1) // limit  # Ceiling division
        current_page = (offset // limit) + 1

        pagination = {
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "current_page": current_page,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_prev": has_prev,
        }

        logger.info(
            f"Retrieved {len(transactions)} transactions (total: {total_count})"
        )

        return {
            "success": True,
            "message": f"Retrieved {len(transactions)} transactions",
            "data": transaction_dicts,
            "pagination": pagination,
        }

    except (
        NotFoundException,
        ValidationException,
        ConflictException,
        InternalException,
        UnauthorizedException,
    ) as e:
        logger.error(f"Service error listing transactions: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to list transactions: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error listing transactions: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while listing transactions"
        ) from e


@router.get("/{transaction_id}")
async def get_transaction(
    transaction_id: str = Path(...),
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """
    Get a single transaction by ID.

    This endpoint retrieves a specific transaction including all its metadata,
    email attachments, and AI predictions.
    """
    try:
        logger.info(f"Getting transaction: {transaction_id}")

        transaction = await transaction_manager.getTransaction(transaction_id)

        logger.info(f"Retrieved transaction: {transaction_id}")

        return success_response(
            "Transaction retrieved successfully", thrift_to_dict(transaction)
        )

    except NotFoundException:
        logger.warning(f"Transaction not found: {transaction_id}")
        raise HTTPException(
            status_code=404, detail=f"Transaction {transaction_id} not found"
        ) from None
    except (ValidationException, InternalException, UnauthorizedException) as e:
        logger.error(f"Service error getting transaction {transaction_id}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to get transaction: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error getting transaction {transaction_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while getting transaction"
        ) from e


@router.put("/{transaction_id}")
async def update_transaction(
    transaction_id: str = Path(...),
    request: TransactionUpdateRequest = ...,
    user_action: str = Query("Update transaction"),
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """
    Update a single transaction.

    This endpoint updates specific fields of a transaction while preserving
    other fields. Changes are tracked in the history system.
    """
    try:
        logger.info(f"Updating transaction: {transaction_id}")

        # Create TransactionEdit from request
        transaction_edit = TransactionEdit(
            transactionId=transaction_id,
            categoryId=request.category_id,
            approved=request.approved,
            memo=request.memo,
            # Note: metadata updates would need to be handled separately
        )

        # Update through TransactionManager
        updated_transactions = await transaction_manager.updateTransactions(
            [transaction_edit]
        )

        if not updated_transactions:
            raise HTTPException(status_code=500, detail="Failed to update transaction")

        result = updated_transactions[0]

        logger.info(f"Updated transaction: {transaction_id}")

        return success_response(
            message="Transaction updated successfully", data=thrift_to_dict(result)
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except (
        NotFoundException,
        ValidationException,
        ConflictException,
        InternalException,
        UnauthorizedException,
    ) as e:
        logger.error(f"Service error updating transaction {transaction_id}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to update transaction: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error updating transaction {transaction_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while updating transaction"
        ) from e


@router.delete("/{transaction_id}")
async def delete_transaction(
    transaction_id: str = Path(...),
    user_action: str = Query("Delete transaction"),
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """
    Delete a transaction.

    NOTE: This endpoint is not yet implemented in the new TransactionManager architecture.
    The TransactionManager Thrift interface doesn't include delete operations.
    """
    logger.warning(
        f"Delete transaction endpoint called but not implemented: {transaction_id}"
    )
    raise HTTPException(
        status_code=501,
        detail="Transaction deletion not implemented in current architecture",
    )


# Batch Operations


@router.post("/batch")
async def batch_update_transactions(
    request: BatchTransactionRequest,
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """
    Batch update multiple transactions.

    This endpoint allows updating multiple transactions in a single request.
    All updates are performed in a transaction to ensure consistency.
    """
    try:
        logger.info(f"Batch updating {len(request.updates)} transactions")

        if not request.updates:
            raise HTTPException(
                status_code=400, detail="No updates provided in batch request"
            )

        if len(request.updates) > 100:
            raise HTTPException(
                status_code=400,
                detail="Batch size too large. Maximum 100 transactions per batch.",
            )

        # Convert batch request to TransactionEdit objects
        transaction_edits = []
        errors = []

        for update_item in request.updates:
            try:
                # Create TransactionEdit from request
                transaction_edit = TransactionEdit(
                    transactionId=update_item.transaction_id,
                    categoryId=update_item.updates.category_id,
                    approved=update_item.updates.approved,
                    memo=update_item.updates.memo,
                    # Note: metadata updates would need to be handled separately
                )
                transaction_edits.append(transaction_edit)

            except Exception as e:
                logger.error(
                    f"Error preparing update for transaction {update_item.transaction_id}: {e}"
                )
                errors.append(
                    {"transaction_id": update_item.transaction_id, "error": str(e)}
                )

        # Execute batch update through TransactionManager
        updated_transactions = []
        if transaction_edits:
            try:
                updated_transactions = await transaction_manager.updateTransactions(
                    transaction_edits
                )

            except Exception as e:
                logger.error(f"Batch operation failed: {e}")
                # If batch fails, add errors for all transactions
                for edit in transaction_edits:
                    errors.append(
                        {
                            "transaction_id": edit.transactionId,
                            "error": f"Batch operation failed: {str(e)}",
                        }
                    )

        success_count = len(updated_transactions)
        error_count = len(errors)

        logger.info(
            f"Batch update completed: {success_count} successful, {error_count} errors"
        )

        return {
            "success": error_count == 0,
            "message": f"Batch update completed: {success_count} successful, {error_count} errors",
            "data": [thrift_to_dict(t) for t in updated_transactions],
            "errors": errors,
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except (
        NotFoundException,
        ValidationException,
        ConflictException,
        InternalException,
        UnauthorizedException,
    ) as e:
        logger.error(f"Service error in batch update: {e}")
        raise HTTPException(
            status_code=400, detail=f"Batch update failed: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error in batch update: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during batch update"
        ) from e


# History Endpoints


@router.get("/{transaction_id}/history")
async def get_transaction_history(
    transaction_id: str = Path(...),
    limit: int = Query(50),
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """
    Get change history for a specific transaction.

    NOTE: This endpoint is not yet implemented in the new TransactionManager architecture.
    The TransactionManager Thrift interface doesn't include history operations.
    """
    logger.warning(
        f"Transaction history endpoint called but not implemented: {transaction_id}"
    )
    raise HTTPException(
        status_code=501,
        detail="Transaction history not implemented in current architecture",
    )


# Sync Endpoints


class SyncRequest(BaseModel):
    """Request model for sync operations."""

    budgetPlatform: str | None = Field("YNAB", description="Budget platform type")
    fromDate: str | None = Field(
        None, description="Override start date (YYYY-MM-DD) for sync preview scope"
    )


def _normalize_txn_for_diff(txn: dict[str, Any]) -> dict[str, Any]:
    """Pick stable fields for comparison to reduce noise in diffs."""
    if not txn:
        return {}
    keys = [
        "id",
        "date",
        "amount",
        "approved",
        "payeeId",
        "categoryId",
        "accountId",
        "budgetId",
        "memo",
    ]

    # Normalize date-like values to date-only (YYYY-MM-DD) to avoid false diffs
    def _normalize_date(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        from datetime import datetime

        try:
            # Handle ISO strings with or without timezone
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.date().isoformat()
        except Exception:
            pass
        try:
            # Already a date-only string
            dt = datetime.strptime(value, "%Y-%m-%d")
            return dt.date().isoformat()
        except Exception:
            return value

    norm = {k: txn.get(k) for k in keys}
    if "date" in norm and norm["date"] is not None:
        norm["date"] = _normalize_date(norm["date"])
    return norm


def _compute_transaction_diff(
    local_list: list[dict[str, Any]], remote_list: list[dict[str, Any]]
):
    """Compute add/update/delete diffs between local and remote lists of transactions.

    Returns: (diffs, summary)
    """
    local_map = {t.get("id"): _normalize_txn_for_diff(t) for t in (local_list or [])}
    remote_map = {t.get("id"): _normalize_txn_for_diff(t) for t in (remote_list or [])}

    diffs: list[TransactionDiffItem] = []

    # Added (present in remote, missing locally)
    for tid, r in remote_map.items():
        if tid not in local_map:
            diffs.append(
                TransactionDiffItem(
                    transaction_id=tid or "",
                    action="add",
                    local=None,
                    remote=r,
                    changes=[
                        DiffChange(field=k, from_value=None, to_value=v)
                        for k, v in r.items()
                        if k != "id"
                    ],
                )
            )

    # Deleted (present locally, missing in remote)
    for tid, local_txn in local_map.items():
        if tid not in remote_map:
            diffs.append(
                TransactionDiffItem(
                    transaction_id=tid or "",
                    action="delete",
                    local=local_txn,
                    remote=None,
                    changes=[
                        DiffChange(field=k, from_value=v, to_value=None)
                        for k, v in local_txn.items()
                        if k != "id"
                    ],
                )
            )

    # Updated (in both but fields differ)
    for tid, local_txn in local_map.items():
        if tid in remote_map:
            remote_txn = remote_map[tid]
            field_changes: list[DiffChange] = []
            for k in set(local_txn.keys()) | set(remote_txn.keys()):
                if k == "id":
                    continue
                if local_txn.get(k) != remote_txn.get(k):
                    field_changes.append(
                        DiffChange(
                            field=k,
                            from_value=local_txn.get(k),
                            to_value=remote_txn.get(k),
                        )
                    )
            if field_changes:
                diffs.append(
                    TransactionDiffItem(
                        transaction_id=tid or "",
                        action="update",
                        local=local_txn,
                        remote=remote_txn,
                        changes=field_changes,
                    )
                )

    summary = {
        "add": sum(1 for d in diffs if d.action == "add"),
        "update": sum(1 for d in diffs if d.action == "update"),
        "delete": sum(1 for d in diffs if d.action == "delete"),
        "total": len(diffs),
    }
    return diffs, summary


def _to_unified_rows(diffs: list[TransactionDiffItem]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not diffs:
        return rows
    for d in diffs:
        # Convert local/remote dicts into compact side strings
        def side_text(obj: dict[str, Any] | None) -> str:
            if not obj:
                return ""
            parts = []
            # Prefer human names over IDs when available
            if obj.get("date") is not None:
                parts.append(f"date={obj.get('date')}")
            if obj.get("amount") is not None:
                parts.append(f"amount={obj.get('amount')}")
            # Names
            if obj.get("payee") is not None:
                parts.append(f"payee={obj.get('payee')}")
            elif obj.get("payeeId") is not None:
                parts.append(f"payeeId={obj.get('payeeId')}")
            if obj.get("category") is not None:
                parts.append(f"category={obj.get('category')}")
            elif obj.get("categoryId") is not None:
                parts.append(f"categoryId={obj.get('categoryId')}")
            if obj.get("account") is not None:
                parts.append(f"account={obj.get('account')}")
            elif obj.get("accountId") is not None:
                parts.append(f"accountId={obj.get('accountId')}")
            if obj.get("memo") is not None:
                parts.append(f"memo={obj.get('memo')}")
            return ", ".join(parts)

        def pick_details(obj: dict[str, Any] | None) -> dict[str, Any]:
            if not obj:
                return {}
            return {
                "date": obj.get("date"),
                "amount": obj.get("amount"),
                "payee": obj.get("payee") or obj.get("payeeId"),
                "category": obj.get("category") or obj.get("categoryId"),
                "account": obj.get("account") or obj.get("accountId"),
                "memo": obj.get("memo"),
            }

        status = "update"
        if d.action == "add":
            status = "add"
        elif d.action == "delete":
            status = "delete"
        elif d.action == "update":
            status = "update"
        else:
            status = "same"

        rows.append(
            {
                "id": d.transaction_id,
                "status": status,
                "left": side_text(d.local),
                "right": side_text(d.remote),
                "leftDetails": pick_details(d.local),
                "rightDetails": pick_details(d.remote),
            }
        )
    return rows


@router.post("/sync/in/preview")
async def preview_sync_in(
    request: SyncRequest,
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """Preview importing from YNAB by computing diffs without applying changes."""
    try:
        from thrift_gen.entities.ttypes import BudgetingPlatformType

        platform_type = (
            BudgetingPlatformType.YNAB
            if (request.budgetPlatform or "YNAB") == "YNAB"
            else None
        )

        # Remote transactions from platform
        remote_txns = (
            await transaction_manager.budgeting_platform_access.getTransactions(
                platform_type
            )
        )

        # Local transactions from DB
        local_txns = await transaction_manager.getAllTransactions()

        # Convert to dicts
        remote_dicts = [thrift_to_dict(t) for t in remote_txns]
        local_dicts = [thrift_to_dict(t) for t in local_txns]
        try:
            logger.info(
                f"unified_sync_preview: initial sizes local={len(local_dicts)} remote={len(remote_dicts)}"
            )
        except Exception:
            pass

        diffs, summary = _compute_transaction_diff(local_dicts, remote_dicts)

        return DiffPreviewResponse(
            success=True,
            message="Preview computed",
            data={
                "summary": summary,
                "diffs": [d.model_dump() for d in diffs],
            },
        )
    except Exception as e:
        logger.error(f"Error previewing sync in: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute preview") from e


@router.post("/sync/in/apply")
async def apply_sync_in(
    request: DiffApplyRequest,
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """Apply selected import changes from preview to local DB only."""
    try:
        # Get remote and local again to compute exact changes to apply

        platform = await transaction_manager.config_service.getDefaultBudgetPlatform()
        remote_txns = (
            await transaction_manager.budgeting_platform_access.getTransactions(
                platform
            )
        )

        local_txns = await transaction_manager.getAllTransactions()
        remote_map = {t.id: t for t in remote_txns}
        local_ids = {t.id for t in local_txns}

        upserts = []
        deletes: list[str] = []

        # Adds and Updates -> upsert remote version into local
        for tid in set(request.add or []) | set(request.update or []):
            if tid in remote_map:
                upserts.append(remote_map[tid])

        # Deletes -> remove local if exists
        for tid in request.delete or []:
            if tid in local_ids:
                deletes.append(tid)

        # Perform DB operations
        if deletes:
            try:
                await transaction_manager.database_store_access.deleteEntities(
                    EntityType.Transaction, deletes
                )
            except Exception as e:
                logger.warning(f"DeleteEntities not supported or failed: {e}")

        if upserts:
            entities = [Entity(transaction=t) for t in upserts]
            await transaction_manager.database_store_access.upsertEntities(entities)

        return success_response(
            message=f"Applied import: upserts={len(upserts)}, deletes={len(deletes)}",
            data={"upserts": len(upserts), "deletes": len(deletes)},
        )
    except Exception as e:
        logger.error(f"Error applying sync in: {e}")
        raise HTTPException(status_code=500, detail="Failed to apply changes") from e


@router.post("/sync/in")
async def sync_transactions_from_ynab(
    request: SyncRequest,
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """
    Sync transactions from YNAB (import).

    This endpoint imports transactions from the connected YNAB budget
    into the local database.
    """
    try:
        logger.info(f"Starting sync from YNAB (platform: {request.budgetPlatform})")

        # Import BudgetingPlatformType enum
        from thrift_gen.entities.ttypes import BudgetingPlatformType

        # Convert string to enum value
        platform_type = None
        if request.budgetPlatform == "YNAB":
            platform_type = BudgetingPlatformType.YNAB

        # Call TransactionManager sync method
        sync_result = await transaction_manager.syncTransactionsIn(platform_type)

        # Convert Thrift result to dict
        result_dict = thrift_to_dict(sync_result)

        logger.info(f"Sync from YNAB completed: {sync_result.batchStatus}")

        return success_response(
            message=f"Sync from YNAB completed with status: {sync_result.batchStatus}",
            data=result_dict,
        )

    except (InternalException, UnauthorizedException) as e:
        logger.error(f"Service error syncing from YNAB: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to sync from YNAB: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error syncing from YNAB: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while syncing from YNAB"
        ) from e


@router.post("/sync/out")
async def sync_transactions_to_ynab(
    request: SyncRequest,
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """
    Sync transactions to YNAB (export).

    This endpoint exports local transaction changes back to the
    connected YNAB budget.
    """
    try:
        logger.info(f"Starting sync to YNAB (platform: {request.budgetPlatform})")

        # Import BudgetingPlatformType enum
        from thrift_gen.entities.ttypes import BudgetingPlatformType

        # Convert string to enum value
        platform_type = None
        if request.budgetPlatform == "YNAB":
            platform_type = BudgetingPlatformType.YNAB

        # Call TransactionManager sync method
        sync_result = await transaction_manager.syncTransactionsOut(platform_type)

        # Convert Thrift result to dict
        result_dict = thrift_to_dict(sync_result)

        logger.info(f"Sync to YNAB completed: {sync_result.batchStatus}")

        return success_response(
            message=f"Sync to YNAB completed with status: {sync_result.batchStatus}",
            data=result_dict,
        )

    except (InternalException, UnauthorizedException) as e:
        logger.error(f"Service error syncing to YNAB: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to sync to YNAB: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error syncing to YNAB: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while syncing to YNAB"
        ) from e


@router.post("/sync/out/preview")
async def preview_sync_out(
    request: SyncRequest,
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """Preview syncing to YNAB by computing diffs from local to remote without applying."""
    try:
        from thrift_gen.entities.ttypes import BudgetingPlatformType

        platform_type = (
            BudgetingPlatformType.YNAB
            if (request.budgetPlatform or "YNAB") == "YNAB"
            else None
        )

        remote_txns = (
            await transaction_manager.budgeting_platform_access.getTransactions(
                platform_type
            )
        )

        local_txns = await transaction_manager.getAllTransactions()

        remote_dicts = [thrift_to_dict(t) for t in remote_txns]
        local_dicts = [thrift_to_dict(t) for t in local_txns]

        diffs, summary = _compute_transaction_diff(remote_dicts, local_dicts)

        return DiffPreviewResponse(
            success=True,
            message="Preview computed",
            data={"summary": summary, "diffs": [d.model_dump() for d in diffs]},
        )
    except Exception as e:
        logger.error(f"Error previewing sync out: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute preview") from e


@router.post("/sync/out/apply")
async def apply_sync_out(
    request: DiffApplyRequest,
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """Apply selected changes to YNAB from local data (export)."""
    try:
        # Determine which local transactions to send
        target_ids = set(
            (request.add or []) + (request.update or []) + (request.delete or [])
        )
        if not target_ids:
            return success_response("No changes selected", data={"updated": 0})

        # Load selected local transactions
        all_local = await transaction_manager.getAllTransactions()

        id_map = {t.id: t for t in all_local}
        to_send = [id_map[tid] for tid in target_ids if tid in id_map]

        platform = await transaction_manager.config_service.getDefaultBudgetPlatform()
        success = (
            await transaction_manager.budgeting_platform_access.updateTransactions(
                to_send, platform
            )
        )
        if success:
            return success_response(
                message=f"Exported {len(to_send)} transactions to YNAB",
                data={"updated": len(to_send)},
            )
        else:
            raise HTTPException(status_code=502, detail="YNAB update failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error applying sync out: {e}")
        raise HTTPException(status_code=500, detail="Failed to export changes") from e


@router.post("/sync/preview")
async def unified_sync_preview(
    request: SyncRequest | None = None,
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """Unified preview: produce two-column rows showing local vs remote differences."""
    try:
        from thrift_gen.entities.ttypes import BudgetingPlatformType

        platform_type = (
            BudgetingPlatformType.YNAB
            if ((request and request.budgetPlatform) or "YNAB") == "YNAB"
            else None
        )

        # Optimization: only fetch remote since min(earliest locally edited baseline, last local transaction date)
        last_local_edit: str | None = None
        try:
            last_local_edit = await transaction_manager.config_service.getConfigValue(
                ConfigKeys.SYNC_LAST_LOCAL_EDIT_TIME, None
            )
        except Exception as e:
            logger.warning(f"Could not read last local edit time: {e}")

        # If not set, initialize tracking to now to avoid fetching entire remote history
        if not last_local_edit:
            try:
                from datetime import datetime

                from thrift_gen.entities.ttypes import (
                    ConfigItem,
                    ConfigType,
                    ConfigValue,
                )

                now_iso = datetime.now(UTC).isoformat()
                await transaction_manager.config_service.updateConfigs(
                    [
                        ConfigItem(
                            key=ConfigKeys.SYNC_LAST_LOCAL_EDIT_TIME,
                            type=ConfigType.System,
                            value=ConfigValue(stringValue=now_iso),
                            description="Initialized last local edit time",
                        )
                    ]
                )
                last_local_edit = now_iso
            except Exception as e:
                logger.warning(f"Failed to initialize last local edit time: {e}")

        # Load local transactions first to compute the latest local transaction date
        local_txns = await transaction_manager.getAllTransactions()

        # Also read edited IDs early so we can ensure we include their remote counterparts
        edited_ids: list[str] = []
        try:
            edited_ids = (
                await transaction_manager.config_service.getConfigValue(
                    ConfigKeys.SYNC_LOCAL_EDITED_IDS, []
                )
                or []
            )
        except Exception as e:
            logger.warning(f"Could not read edited IDs: {e}")

        # Helper to parse various date strings (YYYY-MM-DD or ISO datetime)
        from datetime import datetime

        def _parse_dt(s: str | None):
            if not s:
                return None
            # Try ISO first, normalize to UTC if naive
            try:
                dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return dt
            except Exception:
                pass
            # Fallback to date-only
            try:
                return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)
            except Exception:
                return None

        # Compute latest local transaction date
        local_dates = [_parse_dt(getattr(t, "date", None)) for t in (local_txns or [])]
        local_dates = [d for d in local_dates if d is not None]
        latest_local_txn_dt = max(local_dates) if local_dates else None

        # Compute baseline (earliest edited) date
        baseline_dt = (
            _parse_dt(last_local_edit) if isinstance(last_local_edit, str) else None
        )

        # If caller provided fromDate, prefer it; otherwise compute a reasonable since_dt
        use_custom_date = bool(request and getattr(request, "fromDate", None))
        if use_custom_date:
            since_dt = _parse_dt(request.fromDate)
            since_str = (
                since_dt.date().isoformat() if since_dt else str(request.fromDate)
            )
        else:
            # since_date: avoid huge pulls but don't miss older edited items
            # Strategy:
            # - If we have a baseline (earliest locally edited date), respect it (even if older than cutoff)
            # - Otherwise, use latest_local_txn_dt - buffer_days, but not earlier than cutoff
            from datetime import timedelta

            lookback_days = 90
            buffer_days = 14
            cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
            if baseline_dt:
                if latest_local_txn_dt:
                    since_dt = min(
                        baseline_dt, latest_local_txn_dt - timedelta(days=buffer_days)
                    )
                else:
                    since_dt = baseline_dt
            else:
                if latest_local_txn_dt:
                    since_dt = max(
                        latest_local_txn_dt - timedelta(days=buffer_days), cutoff
                    )
                else:
                    # Default to cutoff window if we have no signal
                    since_dt = cutoff

        # If we have edited IDs, ensure since_dt goes back to at least the earliest edited local date
        if edited_ids:
            try:
                from datetime import datetime

                def _parse_local_dt(t):
                    v = getattr(t, "date", None)
                    if not v:
                        return None
                    try:
                        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
                    except Exception:
                        try:
                            return datetime.strptime(str(v), "%Y-%m-%d").replace(
                                tzinfo=UTC
                            )
                        except Exception:
                            return None

                edited_dates = [
                    _parse_local_dt(t)
                    for t in local_txns
                    if getattr(t, "id", None) in set(edited_ids)
                ]
                edited_dates = [d for d in edited_dates if d is not None]
                if edited_dates:
                    earliest_edited_dt = min(edited_dates)
                    if earliest_edited_dt and (
                        since_dt is None or earliest_edited_dt < since_dt
                    ):
                        since_dt = earliest_edited_dt
            except Exception as e:
                logger.warning(f"Failed to widen since_dt for edited IDs: {e}")

        if not use_custom_date:
            since_str = since_dt.date().isoformat() if since_dt else None
        try:
            logger.info(
                f"unified_sync_preview: since={since_str}, baseline={baseline_dt.date().isoformat() if baseline_dt else None}, "
                f"latest_local={(latest_local_txn_dt.date().isoformat() if latest_local_txn_dt else None)}, edited_ids_count={len(edited_ids)}"
            )
        except Exception:
            pass

        remote_txns = (
            await transaction_manager.budgeting_platform_access.getTransactions(
                platform_type, None, since_str, None
            )
        )

        remote_dicts = [thrift_to_dict(t) for t in remote_txns]
        local_dicts = [thrift_to_dict(t) for t in local_txns]

        # If user supplied fromDate, filter local by that date and bypass the 'interesting IDs' trimming
        if use_custom_date and since_str:

            def _date_of(d):
                v = d.get("date")
                if isinstance(v, str):
                    return v.split("T")[0]
                return None

            local_dicts = [
                t for t in local_dicts if (_date_of(t) or "0000-00-00") >= since_str
            ]
        # Otherwise, if we have a last edit time, reduce local set to relevant edited IDs to trim comparisons
        elif last_local_edit or edited_ids:
            remote_ids = {t.get("id") for t in remote_dicts if t.get("id")}
            interesting_ids = remote_ids | set(edited_ids)
            if interesting_ids:
                local_dicts = [t for t in local_dicts if t.get("id") in interesting_ids]
            else:
                # Fallback: include local transactions from the last 30 days so we don't miss nearby items
                try:
                    from datetime import datetime, timedelta

                    cutoff_local = (
                        (datetime.now(UTC) - timedelta(days=30)).date().isoformat()
                    )

                    def _date_of(d):
                        v = d.get("date")
                        if isinstance(v, str):
                            return v.split("T")[0]
                        return None

                    local_dicts = [
                        t
                        for t in local_dicts
                        if (_date_of(t) or "0000-00-00") >= cutoff_local
                    ]
                except Exception:
                    pass

        try:
            logger.info(
                f"unified_sync_preview: filtered sizes local={len(local_dicts)} remote={len(remote_dicts)}"
            )
        except Exception:
            pass

        diffs, summary = _compute_transaction_diff(local_dicts, remote_dicts)
        # Drop unchanged diffs to reduce noise
        diffs = [d for d in diffs if d.action in ("add", "update", "delete")]

        # Build ID->name maps for enrichment
        payee_names: dict[str, str] = {}
        category_names: dict[str, str] = {}
        account_names: dict[str, str] = {}
        try:
            from thrift_gen.entities.ttypes import EntityType as ET

            # Payees
            payee_result = await transaction_manager.database_store_access.getEntities(
                ThriftQuery(entityType=ET.Payee)
            )
            for entity in payee_result.entities:
                if entity.payee and entity.payee.id and entity.payee.name:
                    payee_names[entity.payee.id] = entity.payee.name
            # Categories
            category_result = (
                await transaction_manager.database_store_access.getEntities(
                    ThriftQuery(entityType=ET.Category)
                )
            )
            for entity in category_result.entities:
                if entity.category and entity.category.id and entity.category.name:
                    category_names[entity.category.id] = entity.category.name
            # Accounts
            account_result = (
                await transaction_manager.database_store_access.getEntities(
                    ThriftQuery(entityType=ET.Account)
                )
            )
            for entity in account_result.entities:
                if entity.account and entity.account.id and entity.account.name:
                    account_names[entity.account.id] = entity.account.name
        except Exception as e:
            logger.warning(f"Failed to build name maps: {e}")

        def enrich(obj: dict[str, Any] | None) -> dict[str, Any] | None:
            if not obj:
                return obj
            o = dict(obj)
            if o.get("payeeId") and o["payeeId"] in payee_names:
                o["payee"] = payee_names[o["payeeId"]]
            if o.get("categoryId") and o["categoryId"] in category_names:
                o["category"] = category_names[o["categoryId"]]
            if o.get("accountId") and o["accountId"] in account_names:
                o["account"] = account_names[o["accountId"]]
            return o

        # Enrich before building rows (adds human-readable names)
        for d in diffs:
            d.local = enrich(d.local)
            d.remote = enrich(d.remote)

        rows = _to_unified_rows(diffs)

        return UnifiedPreviewResponse(
            success=True,
            message="Preview computed",
            data={"summary": summary, "items": rows},
        )
    except Exception as e:
        logger.error(f"Error unified preview: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute preview") from e


@router.post("/sync/reset-tracking")
async def reset_sync_tracking(
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """Reset sync tracking: set last local edit time to now and clear edited IDs."""
    try:
        from datetime import datetime

        from thrift_gen.entities.ttypes import ConfigItem, ConfigType, ConfigValue

        now_iso = datetime.now(UTC).isoformat()
        updates = [
            ConfigItem(
                key=ConfigKeys.SYNC_LAST_LOCAL_EDIT_TIME,
                type=ConfigType.System,
                value=ConfigValue(stringValue=now_iso),
                description="Last ISO time a local transaction was edited",
            ),
            ConfigItem(
                key=ConfigKeys.SYNC_LOCAL_EDITED_IDS,
                type=ConfigType.System,
                value=ConfigValue(stringList=[]),
                description="Locally edited transaction IDs awaiting sync",
            ),
        ]
        await transaction_manager.config_service.updateConfigs(updates)
        return ResetTrackingResponse(
            success=True,
            message="Sync tracking reset",
            data={"last_local_edit": now_iso},
        )
    except Exception as e:
        logger.error(f"Error resetting sync tracking: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset tracking") from e


@router.post("/sync/apply")
async def unified_sync_apply(
    request: UnifiedApplyRequest,
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """Apply a mixed plan where each item chooses:
    - left: pull remote → local (right → left)
    - right: push local → remote (left → right)
    """
    try:
        # Load local and remote maps
        platform = await transaction_manager.config_service.getDefaultBudgetPlatform()
        remote_txns = (
            await transaction_manager.budgeting_platform_access.getTransactions(
                platform
            )
        )

        local_txns = await transaction_manager.getAllTransactions()

        remote_map = {t.id: t for t in remote_txns}
        local_map = {t.id: t for t in local_txns}

        to_upsert_local: list = []
        to_update_remote: list = []
        pushed_ids: list[str] = []

        for step in request.plan or []:
            tid = step.get("id")
            action = (step.get("action") or "skip").lower()
            if action == "left":
                # Pull remote into local (upsert local)
                t = remote_map.get(tid)
                if t is not None:
                    to_upsert_local.append(t)
            elif action == "right":
                # Push local to remote (update remote)
                t = local_map.get(tid)
                if t is not None:
                    to_update_remote.append(t)
                    pushed_ids.append(tid)
            else:
                continue

        # Apply local upserts
        if to_upsert_local:
            await transaction_manager.database_store_access.upsertEntities(
                [Entity(transaction=t) for t in to_upsert_local]
            )

        # Apply remote updates
        updated_remote = 0
        if to_update_remote:
            success = (
                await transaction_manager.budgeting_platform_access.updateTransactions(
                    to_update_remote, platform
                )
            )
            updated_remote = len(to_update_remote) if success else 0
            # If push succeeded, clear those IDs from edited set
            if success and pushed_ids:
                try:
                    existing_ids = (
                        await transaction_manager.config_service.getConfigValue(
                            ConfigKeys.SYNC_LOCAL_EDITED_IDS, []
                        )
                        or []
                    )
                    remaining = [i for i in existing_ids if i not in set(pushed_ids)]
                    from thrift_gen.entities.ttypes import (
                        ConfigItem,
                        ConfigType,
                        ConfigValue,
                    )

                    await transaction_manager.config_service.updateConfigs(
                        [
                            ConfigItem(
                                key=ConfigKeys.SYNC_LOCAL_EDITED_IDS,
                                type=ConfigType.System,
                                value=ConfigValue(stringList=remaining),
                                description="Locally edited transaction IDs awaiting sync",
                            )
                        ]
                    )
                except Exception as e:
                    logger.warning(f"Failed to clear edited IDs after push: {e}")

        return success_response(
            message="Unified apply completed",
            data={
                "upserted_local": len(to_upsert_local),
                "updated_remote": updated_remote,
            },
        )
    except Exception as e:
        logger.error(f"Error unified apply: {e}")
        raise HTTPException(status_code=500, detail="Failed to apply plan") from e
