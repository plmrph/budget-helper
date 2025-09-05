"""
Authentication API routes.

This module provides endpoints for YNAB and Gmail authentication.
"""

import json
import logging
import secrets
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from api_layer.dependencies import BudgetingPlatformAccessDep, MetadataSourceAccessDep, TransactionManagerDep
from configs import ConfigKeys, get_config_service
from resource_layer.budgeting_platform_access.budgeting_platform_access import (
    BudgetingPlatformAccess,
)
from resource_layer.budgeting_platform_access.ynab_api_client import YnabApiClient

# TODO fix this violation later.
from resource_layer.metadata_source_access.gmail_api_client import GmailApiClient
from resource_layer.metadata_source_access.metadata_source_access import (
    MetadataSourceAccess,
)
from thrift_gen.entities.ttypes import ConfigItem, ConfigType, ConfigValue
from thrift_gen.exceptions.ttypes import (
    InternalException,
    RemoteServiceException,
    UnauthorizedException,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])


# Request/Response Models


class AuthResponse(BaseModel):
    """Standard authentication response."""

    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Response message")
    data: dict[str, Any] = Field(
        default_factory=dict, description="Additional response data"
    )


# YNAB Authentication Endpoints


class YnabConnectRequest(BaseModel):
    """YNAB connection request model."""

    personal_access_token: str = Field(..., description="YNAB personal access token")


@router.post("/ynab/connect", response_model=AuthResponse)
async def connect_ynab(
    request: YnabConnectRequest,
    budgeting_platform_access: BudgetingPlatformAccess = BudgetingPlatformAccessDep,
    transaction_manager = TransactionManagerDep,
):
    """
    Connect to YNAB using provided personal access token.

    This endpoint accepts a YNAB personal access token, validates it,
    and stores it in configuration for future use.
    """
    try:
        logger.info("Attempting YNAB authentication with provided token")

        token = request.personal_access_token.strip()

        if not token:
            raise HTTPException(status_code=400, detail="YNAB token is required")

        # Basic token format validation (YNAB tokens are typically long alphanumeric strings)
        if len(token) < 20:
            raise HTTPException(status_code=400, detail="Invalid YNAB token format")

        # Validate token with YNAB API before storing
        logger.info("Validating YNAB token with API")
        try:
            ynab_client = YnabApiClient()
            is_valid = await ynab_client.validate_token(token)

            if not is_valid:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid YNAB token - authentication failed with YNAB API",
                )

            logger.info("YNAB token validated successfully")

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"YNAB token validation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"YNAB token validation failed: {str(e)}"
            ) from e

        # Store token in database via Configs service
        config_service = get_config_service()
        if config_service:
            config_item = ConfigItem(
                key=ConfigKeys.BUDGET_YNAB_AUTH_CONFIG,
                type=ConfigType.ExternalSystem,
                value=ConfigValue(stringValue=token),
                description="YNAB personal access token for authentication",
            )
            await config_service.updateConfigs([config_item])
            logger.info("YNAB token stored in database successfully")

            if transaction_manager:
                logger.info("Triggering initial budget metadata sync after YNAB connect")
                # Request budgets info with refreshData=True to force sync from YNAB
                await transaction_manager.getBudgetsInfo(
                    budgetIds=None, entityTypes=None, refreshData=True
                )
                logger.info("Initial budget metadata sync completed")
        else:
            logger.error("Config service not available")
            raise HTTPException(
                status_code=500, detail="Configuration service unavailable"
            )

        return AuthResponse(
            success=True,
            message="Successfully connected to YNAB",
            data={"service": "ynab", "token_stored": True},
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except UnauthorizedException as e:
        logger.error(f"YNAB authentication unauthorized: {e}")
        raise HTTPException(
            status_code=401, detail="YNAB authentication failed - unauthorized"
        ) from e
    except RemoteServiceException as e:
        logger.error(f"YNAB remote service error: {e}")
        raise HTTPException(status_code=502, detail="YNAB service unavailable") from e
    except InternalException as e:
        logger.error(f"YNAB internal error: {e}")
        raise HTTPException(
            status_code=500, detail="Internal error during YNAB authentication"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error during YNAB authentication: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during YNAB authentication"
        ) from e


@router.post("/ynab/disconnect", response_model=AuthResponse)
async def disconnect_ynab():
    """
    Disconnect from YNAB.

    Disconnection is handled by clearing configuration data through the Configs service.
    """
    try:
        logger.info("YNAB disconnection requested")

        # Clear stored token from configs
        config_service = get_config_service()
        if config_service:
            config_item = ConfigItem(
                key=ConfigKeys.BUDGET_YNAB_AUTH_CONFIG,
                type=ConfigType.ExternalSystem,
                value=ConfigValue(stringValue=""),  # Clear the token
                description="YNAB personal access token for authentication",
            )
            await config_service.updateConfigs([config_item])
            logger.info("YNAB token cleared from database")

        return AuthResponse(
            success=True,
            message="Successfully disconnected from YNAB",
            data={"service": "ynab"},
        )

    except Exception as e:
        logger.error(f"Error during YNAB disconnection: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during YNAB disconnection"
        ) from e


@router.get("/ynab/status")
async def get_ynab_status(
    budgeting_platform_access: BudgetingPlatformAccess = BudgetingPlatformAccessDep,
):
    """
    Get YNAB authentication status.

    This endpoint returns the current authentication status for YNAB
    by attempting to authenticate with the BudgetingPlatformAccess service.
    """
    try:
        logger.debug("Checking YNAB authentication status")

        # Check if we have a stored token in configs
        config_service = get_config_service()
        is_authenticated = False
        if config_service:
            token = await config_service.getConfigValue(
                ConfigKeys.BUDGET_YNAB_AUTH_CONFIG
            )
            is_authenticated = token is not None and token.strip() != ""

        return {
            "success": True,
            "data": {"status": {"is_authenticated": is_authenticated}},
            "service": "ynab",
        }

    except Exception as e:
        logger.error(f"Error checking YNAB status: {e}")
        return {
            "success": True,
            "data": {"status": {"is_authenticated": False}},
            "service": "ynab",
            "error": str(e),
        }


# Gmail Authentication Endpoints


class GmailConnectRequest(BaseModel):
    """Gmail connection request model."""

    client_id: str = Field(..., description="Gmail OAuth client ID")
    client_secret: str = Field(..., description="Gmail OAuth client secret")
    redirect_uri: str = Field(..., description="OAuth redirect URI")
    scopes: list[str] = Field(..., description="OAuth scopes")


@router.post("/gmail/connect", response_model=AuthResponse)
async def connect_gmail(
    request: GmailConnectRequest,
    metadata_source_access: MetadataSourceAccess = MetadataSourceAccessDep,
):
    """
    Connect to Gmail using provided OAuth configuration.

    This endpoint accepts Gmail OAuth configuration, validates it,
    stores it, and returns the OAuth authorization URL for the user to visit.
    """
    try:
        logger.info("Attempting Gmail OAuth setup with provided configuration")

        if not request.client_id or not request.client_secret:
            return AuthResponse(
                success=False,
                message="Gmail OAuth client ID and secret are required",
                data={"service": "gmail"},
            )

        # Store OAuth config in database via Configs service
        config_service = get_config_service()
        if config_service:
            oauth_config = {
                "client_id": request.client_id,
                "client_secret": request.client_secret,
                "redirect_uri": request.redirect_uri,
                "scopes": request.scopes,
            }
            config_item = ConfigItem(
                key=ConfigKeys.EMAIL_GMAIL_AUTH_CONFIG,
                type=ConfigType.ExternalSystem,
                value=ConfigValue(
                    stringValue=json.dumps(oauth_config)
                ),  # Store as JSON string
                description="Gmail OAuth configuration for authentication",
            )
            await config_service.updateConfigs([config_item])
            logger.info("Gmail OAuth configuration stored in database successfully")
        else:
            logger.error("Config service not available")
            raise HTTPException(
                status_code=500, detail="Configuration service unavailable"
            )

        # Generate OAuth authorization URL
        # Generate and store state parameter for security
        state = secrets.token_urlsafe(32)
        state_config_item = ConfigItem(
            key=ConfigKeys.EMAIL_GMAIL_OAUTH_STATE,
            type=ConfigType.ExternalSystem,
            value=ConfigValue(stringValue=state),
            description="Gmail OAuth state parameter for security",
        )
        await config_service.updateConfigs([state_config_item])

        # Build authorization URL
        auth_params = {
            "client_id": request.client_id,
            "redirect_uri": request.redirect_uri,
            "scope": " ".join(request.scopes),
            "response_type": "code",
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }

        auth_url = (
            f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(auth_params)}"
        )

        logger.info("Gmail OAuth authorization URL generated successfully")

        return AuthResponse(
            success=True,
            message="Gmail OAuth configuration stored. Please visit the authorization URL.",
            data={"service": "gmail", "config_stored": True, "auth_url": auth_url},
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except UnauthorizedException as e:
        logger.error(f"Gmail authentication unauthorized: {e}")
        raise HTTPException(
            status_code=401, detail="Gmail authentication failed - unauthorized"
        ) from e
    except RemoteServiceException as e:
        logger.error(f"Gmail remote service error: {e}")
        raise HTTPException(status_code=502, detail="Gmail service unavailable") from e
    except InternalException as e:
        logger.error(f"Gmail internal error: {e}")
        raise HTTPException(
            status_code=500, detail="Internal error during Gmail authentication"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error during Gmail authentication: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during Gmail authentication"
        ) from e


@router.get("/gmail/callback")
async def gmail_oauth_callback(
    code: str = Query(...),
    state: str = Query(...),
):
    """
    Handle Gmail OAuth callback from Google.

    This endpoint receives the authorization code from Google's OAuth flow
    and exchanges it for access and refresh tokens.
    """
    try:
        logger.info("Processing Gmail OAuth callback")

        config_service = get_config_service()
        if not config_service:
            logger.error("Config service not available")
            raise HTTPException(
                status_code=500, detail="Configuration service unavailable"
            )

        # Verify state parameter
        stored_state = await config_service.getConfigValue(
            ConfigKeys.EMAIL_GMAIL_OAUTH_STATE
        )
        if not stored_state or stored_state != state:
            logger.error("Invalid or missing OAuth state parameter")
            raise HTTPException(status_code=400, detail="Invalid OAuth state parameter")

        # Get OAuth configuration
        oauth_config_str = await config_service.getConfigValue(
            ConfigKeys.EMAIL_GMAIL_AUTH_CONFIG
        )
        if not oauth_config_str:
            logger.error("Gmail OAuth configuration not found")
            raise HTTPException(
                status_code=400, detail="Gmail OAuth configuration not found"
            )

        oauth_config = json.loads(oauth_config_str)

        # Exchange authorization code for tokens
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "client_id": oauth_config["client_id"],
            "client_secret": oauth_config["client_secret"],
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": oauth_config["redirect_uri"],
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, data=token_data)

        if response.status_code != 200:
            logger.error(f"Token exchange failed: {response.text}")
            raise HTTPException(
                status_code=400,
                detail="Failed to exchange authorization code for tokens",
            )

        tokens = response.json()

        # Store tokens in configuration
        token_config_item = ConfigItem(
            key=ConfigKeys.EMAIL_GMAIL_TOKENS,
            type=ConfigType.ExternalSystem,
            value=ConfigValue(stringValue=json.dumps(tokens)),
            description="Gmail OAuth access and refresh tokens",
        )
        await config_service.updateConfigs([token_config_item])

        # Clear the state parameter
        clear_state_item = ConfigItem(
            key=ConfigKeys.EMAIL_GMAIL_OAUTH_STATE,
            type=ConfigType.ExternalSystem,
            value=ConfigValue(stringValue=""),
            description="Gmail OAuth state parameter for security",
        )
        await config_service.updateConfigs([clear_state_item])

        logger.info("Gmail OAuth tokens stored successfully")

        # Return a simple HTML page that closes the popup/redirects
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Gmail Connected</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                .success { color: #28a745; }
            </style>
        </head>
        <body>
            <h1 class="success">Gmail Connected Successfully!</h1>
            <p>You can now close this window and return to the application.</p>
            <script>
                // Try to close the popup window
                if (window.opener) {
                    window.opener.postMessage({type: 'gmail_auth_success'}, '*');
                    window.close();
                } else {
                    // If not a popup, redirect to the main app
                    setTimeout(() => {
                        window.location.href = '/';
                    }, 2000);
                }
            </script>
        </body>
        </html>
        """

        return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during Gmail OAuth callback: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/gmail/auth-url")
async def get_gmail_auth_url():
    """
    Get Gmail OAuth authorization URL.

    OAuth flow should be handled by the MetadataSourceAccess service with configuration from Configs service.
    """
    try:
        logger.info("Gmail OAuth URL generation requested")

        # TODO: Implement OAuth URL generation through MetadataSourceAccess
        # This should use configuration from Configs service

        return {
            "message": "Gmail OAuth URL generation to be implemented via MetadataSourceAccess service"
        }

    except Exception as e:
        logger.error(f"Unexpected error generating Gmail OAuth URL: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error generating OAuth URL"
        ) from e


@router.post("/gmail/complete")
async def complete_gmail_auth():
    """
    Complete Gmail OAuth flow.

    OAuth completion should be handled by the MetadataSourceAccess service.
    """
    try:
        logger.info("Gmail OAuth completion requested")

        # TODO: Implement OAuth completion through MetadataSourceAccess

        return {
            "message": "Gmail OAuth completion to be implemented via MetadataSourceAccess service"
        }

    except Exception as e:
        logger.error(f"Unexpected error during Gmail OAuth completion: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during Gmail OAuth completion",
        ) from e


@router.post("/gmail/disconnect", response_model=AuthResponse)
async def disconnect_gmail():
    """
    Disconnect from Gmail.

    Disconnection is handled by clearing configuration data through the Configs service.
    """
    try:
        logger.info("Gmail disconnection requested")

        # Clear stored OAuth config and tokens from configs
        config_service = get_config_service()
        if config_service:
            # Clear OAuth config
            config_item = ConfigItem(
                key=ConfigKeys.EMAIL_GMAIL_AUTH_CONFIG,
                type=ConfigType.ExternalSystem,
                value=ConfigValue(stringValue=""),  # Clear the config
                description="Gmail OAuth configuration for authentication",
            )
            # Clear OAuth tokens
            tokens_item = ConfigItem(
                key=ConfigKeys.EMAIL_GMAIL_TOKENS,
                type=ConfigType.ExternalSystem,
                value=ConfigValue(stringValue=""),  # Clear the tokens
                description="Gmail OAuth access and refresh tokens",
            )
            # Clear OAuth state
            state_item = ConfigItem(
                key=ConfigKeys.EMAIL_GMAIL_OAUTH_STATE,
                type=ConfigType.ExternalSystem,
                value=ConfigValue(stringValue=""),  # Clear the state
                description="Gmail OAuth state parameter for security",
            )
            await config_service.updateConfigs([config_item, tokens_item, state_item])
            logger.info("Gmail OAuth config, tokens, and state cleared from database")

            # TODO fix this: Invalidate GmailApiClient credential cache
            gmail_client = GmailApiClient(config_service)
            gmail_client._credentials_validated = False
            gmail_client._last_validation_time = None

        return AuthResponse(
            success=True,
            message="Successfully disconnected from Gmail",
            data={"service": "gmail"},
        )

    except Exception as e:
        logger.error(f"Error during Gmail disconnection: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during Gmail disconnection"
        ) from e


@router.get("/gmail/status")
async def get_gmail_status():
    """
    Get Gmail authentication status by actually testing the credentials.
    """
    try:
        logger.debug("Checking Gmail authentication status")

        config_service = get_config_service()
        gmail_client = GmailApiClient(config_service)

        # Test if credentials are valid
        is_authenticated = await gmail_client.validate_credentials()

        return {
            "success": True,
            "data": {"status": {"is_authenticated": is_authenticated}},
            "service": "gmail",
        }

    except Exception as e:
        logger.error(f"Error checking Gmail status: {e}")
        return {
            "success": True,
            "data": {"status": {"is_authenticated": False}},
            "service": "gmail",
            "error": str(e),
        }


# Combined Status Endpoints


@router.get("/status")
async def get_all_auth_status():
    """
    Get authentication status for all services by calling individual status endpoints.
    """
    try:
        logger.debug("Checking authentication status for all services")

        # Call individual status endpoints
        ynab_status = await get_ynab_status()
        gmail_status = await get_gmail_status()

        # Extract authentication status from responses
        ynab_authenticated = (
            ynab_status.get("data", {}).get("status", {}).get("is_authenticated", False)
        )
        gmail_authenticated = (
            gmail_status.get("data", {})
            .get("status", {})
            .get("is_authenticated", False)
        )

        statuses = {
            "ynab": {"service": "ynab", "is_authenticated": ynab_authenticated},
            "gmail": {"service": "gmail", "is_authenticated": gmail_authenticated},
        }

        return {"success": True, "statuses": statuses}

    except Exception as e:
        logger.error(f"Error checking all auth statuses: {e}")
        # Return empty statuses on error
        return {
            "success": False,
            "statuses": {
                "ynab": {"service": "ynab", "is_authenticated": False, "error": str(e)},
                "gmail": {
                    "service": "gmail",
                    "is_authenticated": False,
                    "error": str(e),
                },
            },
        }
