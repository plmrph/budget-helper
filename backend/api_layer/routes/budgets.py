"""
Budgets API routes - Budget management endpoints.

This module provides endpoints for budget management and selection.
"""

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api_layer.dependencies import ConfigsDep, TransactionManagerDep
from business_layer.transaction_manager import TransactionManager
from configs import ConfigKeys, ConfigService

# Import generated Thrift types
from thrift_gen.entities.ttypes import (
    BudgetingPlatformType,
    ConfigItem,
    ConfigType,
    ConfigValue,
    EntityType,
)
from thrift_gen.exceptions.ttypes import (
    InternalException,
    NotFoundException,
    UnauthorizedException,
    ValidationException,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/budgets", tags=["budgets"])


class BudgetSelectionRequest(BaseModel):
    """Request model for selecting a budget."""

    budget_id: str


def success_response(message: str, data=None):
    """Create a success response."""
    response = {"success": True, "message": message}
    if data is not None:
        response["data"] = data
    return response


def error_response(message: str, error_code: str = "BUDGET_ERROR"):
    """Create an error response."""
    return {"success": False, "message": message, "error_code": error_code}


@router.get("/info")
async def get_budget_info(
    budget_ids: list[str] | None = Query(
        None, description="List of budget IDs to retrieve (defaults to default budget)"
    ),
    entity_types: list[str] | None = Query(
        None, description="List of entity types to include (Account, Payee, Category)"
    ),
    refresh_data: bool = Query(
        False, description="Whether to refresh data from YNAB before returning"
    ),
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """
    Get budget information including categories, payees, and accounts.

    This endpoint provides comprehensive budget data for frontend caching and ID-to-name mapping.
    Uses TransactionManager.getBudgetsInfo.
    """
    try:
        logger.info(
            f"Getting budget info for budgetIds: {budget_ids}, entityTypes: {entity_types}"
        )

        # Convert string entity types to EntityType enum values
        parsed_entity_types = None
        if entity_types:
            parsed_entity_types = []
            valid_types = {
                "Account": EntityType.Account,
                "Payee": EntityType.Payee,
                "Category": EntityType.Category,
            }
            for entity_type_str in entity_types:
                if entity_type_str in valid_types:
                    parsed_entity_types.append(valid_types[entity_type_str])
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid entity type '{entity_type_str}'. Valid types: Account, Payee, Category",
                    )

        # Call TransactionManager.getBudgetsInfo
        result = await transaction_manager.getBudgetsInfo(
            budgetIds=budget_ids,
            entityTypes=parsed_entity_types,
            refreshData=refresh_data,
        )

        # Convert Thrift result to JSON-serializable format
        response_data = {"budgets": [], "categories": [], "payees": [], "accounts": []}

        # Convert budgets
        if result.budgets:
            for budget in result.budgets:
                # Convert platformType integer to string using enum mapping
                platform_type_str = None
                if budget.platformType is not None:
                    platform_type_str = BudgetingPlatformType._VALUES_TO_NAMES.get(
                        budget.platformType, "UNKNOWN"
                    )

                budget_dict = {
                    "id": budget.id,
                    "name": budget.name,
                    "currency": budget.currency,
                    "platform_type": platform_type_str,
                    "total_amount": budget.totalAmount,
                    "start_date": budget.startDate,
                    "end_date": budget.endDate,
                }
                response_data["budgets"].append(budget_dict)

        # Convert categories
        if result.categories:
            for category in result.categories:
                category_dict = {
                    "id": category.id,
                    "name": category.name,
                    "description": category.description,
                    "is_income_category": category.isIncomeCategory,
                    "budget_id": category.budgetId,
                    "platform_type": BudgetingPlatformType._VALUES_TO_NAMES.get(
                        category.platformType, "UNKNOWN"
                    )
                    if category.platformType is not None
                    else None,
                }
                response_data["categories"].append(category_dict)

        # Convert payees
        if result.payees:
            for payee in result.payees:
                payee_dict = {
                    "id": payee.id,
                    "name": payee.name,
                    "description": payee.description,
                    "budget_id": payee.budgetId,
                    "platform_type": BudgetingPlatformType._VALUES_TO_NAMES.get(
                        payee.platformType, "UNKNOWN"
                    )
                    if payee.platformType is not None
                    else None,
                }
                response_data["payees"].append(payee_dict)

        # Convert accounts
        if result.accounts:
            for account in result.accounts:
                account_dict = {
                    "id": account.id,
                    "name": account.name,
                    "type": account.type,
                    "institution": account.institution,
                    "currency": account.currency,
                    "balance": account.balance,
                    "status": account.status,
                    "budget_id": account.budgetId,
                    "platform_type": BudgetingPlatformType._VALUES_TO_NAMES.get(
                        account.platformType, "UNKNOWN"
                    )
                    if account.platformType is not None
                    else None,
                }
                response_data["accounts"].append(account_dict)

        logger.info(
            f"Budget info retrieved successfully: {len(response_data['budgets'])} budgets, "
            f"{len(response_data['categories'])} categories, "
            f"{len(response_data['payees'])} payees, "
            f"{len(response_data['accounts'])} accounts"
        )

        return response_data

    except (NotFoundException, ValidationException, UnauthorizedException) as e:
        logger.error(f"Service error getting budget info: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to get budget info: {str(e)}"
        ) from e
    except InternalException as e:
        logger.error(f"Internal service error getting budget info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error getting budget info: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while getting budget info"
        ) from e


@router.get("/")
async def get_budgets(
    transaction_manager: TransactionManager = TransactionManagerDep,
):
    """
    Get all available budgets from the connected budgeting platform.

    Returns a list of budgets that can be selected for transaction syncing.
    UPDATED: Now uses TransactionManager instead of direct BudgetingPlatformAccess.
    """
    try:
        logger.info("Getting available budgets")

        result = await transaction_manager.getBudgetsInfo(
            budgetIds=None,
            entityTypes=None,
            refreshData=False,
        )

        # Convert budgets to dictionary format
        budget_list = []
        if result.budgets:
            for budget in result.budgets:
                # Convert platformType integer to string using enum mapping
                platform_type_str = "YNAB"  # Default fallback
                if budget.platformType is not None:
                    platform_type_str = BudgetingPlatformType._VALUES_TO_NAMES.get(
                        budget.platformType, "YNAB"
                    )

                budget_dict = {
                    "id": budget.id,
                    "name": budget.name,
                    "currency": budget.currency,
                    "platform_type": platform_type_str,
                    "total_amount": budget.totalAmount,
                    "start_date": budget.startDate,
                    "end_date": budget.endDate,
                }
                budget_list.append(budget_dict)

        logger.info(f"Retrieved {len(budget_list)} budgets")

        return success_response(f"Retrieved {len(budget_list)} budgets", budget_list)

    except NotFoundException:
        # No budgets found is not an error for the frontend — return empty list
        logger.info("No budgets found")
        return success_response("Retrieved 0 budgets", [])
    except (ValidationException, UnauthorizedException) as e:
        logger.error(f"Service error getting budgets: {e}")
        return error_response(f"Failed to get budgets: {str(e)}", "SERVICE_ERROR")
    except InternalException as e:
        logger.error(f"Internal service error getting budgets: {e}")
        return error_response(f"Internal server error: {str(e)}", "INTERNAL_ERROR")
    except Exception as e:
        logger.error(f"Unexpected error getting budgets: {e}")
        return error_response(
            "Internal server error while getting budgets", "INTERNAL_ERROR"
        )


@router.get("/selected")
async def get_selected_budget(configs: ConfigService = ConfigsDep):
    """
    Get the currently selected budget ID.

    Returns the budget ID that is currently selected for transaction syncing.
    """
    try:
        logger.info("Getting selected budget")

        # Get the selected budget ID from configuration
        selected_budget_id = await configs.getConfigValue(ConfigKeys.SELECTED_BUDGET_ID)

        if selected_budget_id:
            return success_response(
                "Retrieved selected budget", {"selected_budget_id": selected_budget_id}
            )
        else:
            return success_response("No budget selected", {"selected_budget_id": None})

    except Exception as e:
        logger.error(f"Error getting selected budget: {e}")
        return error_response("Failed to get selected budget", "CONFIG_ERROR")


@router.post("/select")
async def select_budget(
    request: BudgetSelectionRequest, configs: ConfigService = ConfigsDep
):
    """
    Select a budget for transaction syncing.

    Sets the specified budget as the active budget for all transaction operations.
    """
    try:
        logger.info(f"Selecting budget: {request.budget_id}")

        # Create config item for selected budget
        config_value = ConfigValue(stringValue=request.budget_id)
        config_item = ConfigItem(
            key=ConfigKeys.SELECTED_BUDGET_ID,
            value=config_value,
            type=ConfigType.System,
            description="Currently selected budget ID for transaction syncing",
        )

        # Update the configuration
        updated_configs = await configs.updateConfigs([config_item])

        if updated_configs:
            logger.info(f"Successfully selected budget: {request.budget_id}")
            return success_response(
                f"Budget {request.budget_id} selected successfully",
                {"selected_budget_id": request.budget_id},
            )
        else:
            return error_response(
                "Failed to update budget selection", "CONFIG_UPDATE_ERROR"
            )

    except Exception as e:
        logger.error(f"Error selecting budget: {e}")
        return error_response("Failed to select budget", "CONFIG_ERROR")
