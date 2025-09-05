"""
Settings API routes

This module provides endpoints for application settings management using only
the methods defined in the ConfigService Thrift service interface.
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from api_layer.dependencies import ConfigsDep
from configs import ConfigService

# Import generated Thrift types
from thrift_gen.entities.ttypes import ConfigItem, ConfigType, ConfigValue
from thrift_gen.exceptions.ttypes import (
    InternalException,
    NotFoundException,
    UnauthorizedException,
    ValidationException,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])


def thrift_to_dict(thrift_obj):
    """Convert a Thrift object to a dictionary for JSON serialization."""
    if thrift_obj is None:
        return None

    result = {}
    # Use __dict__ if available, otherwise iterate through known attributes
    if hasattr(thrift_obj, "__dict__"):
        for attr_name, attr_value in thrift_obj.__dict__.items():
            if not attr_name.startswith("_") and attr_value is not None:
                if hasattr(attr_value, "__dict__"):
                    # Nested Thrift object
                    result[attr_name] = thrift_to_dict(attr_value)
                elif isinstance(attr_value, list):
                    # List of potentially Thrift objects
                    result[attr_name] = [
                        thrift_to_dict(item) if hasattr(item, "__dict__") else item
                        for item in attr_value
                    ]
                else:
                    result[attr_name] = attr_value
    else:
        # Fallback for objects without __dict__
        for attr_name in ["key", "type", "value", "description"]:
            if hasattr(thrift_obj, attr_name):
                attr_value = getattr(thrift_obj, attr_name)
                if attr_value is not None:
                    if hasattr(attr_value, "__dict__"):
                        result[attr_name] = thrift_to_dict(attr_value)
                    else:
                        result[attr_name] = attr_value

    return result


def success_response(message: str, data: Any = None) -> dict[str, Any]:
    """Create a success response."""
    response = {"success": True, "message": message}
    if data is not None:
        response["data"] = data
    return response


# Request Models


class ConfigUpdateRequest(BaseModel):
    """Request model for updating configuration items."""

    configs: list[dict[str, Any]] = Field(
        ..., description="List of configuration items to update"
    )


class ConfigResetRequest(BaseModel):
    """Request model for resetting configurations."""

    config_type: str | None = Field(
        None, description="Type of configs to reset (if None, resets all)"
    )


# Settings Endpoints


@router.get("/")
async def get_settings(
    config_type: str | None = Query(None),
    key: str | None = Query(None),
    configs: ConfigService = ConfigsDep,
):
    """
    Get application settings.

    Returns the current application settings, optionally filtered by type or key.
    """
    try:
        logger.info(f"Getting settings (type={config_type}, key={key})")

        # Convert string to ConfigType enum if provided
        config_type_enum = None
        if config_type:
            try:
                config_type_map = {
                    "system": ConfigType.System,
                    "email": ConfigType.Email,
                    "ai": ConfigType.AI,
                    "display": ConfigType.Display,
                    "external_system": ConfigType.ExternalSystem,
                }
                config_type_enum = config_type_map.get(config_type.lower())
                if config_type_enum is None:
                    logger.warning(f"Invalid config type: {config_type}")
            except Exception:
                logger.warning(f"Invalid config type: {config_type}")

        # Get configuration items
        config_items = await configs.getConfigs(config_type_enum, key)

        # Convert to dictionary format
        settings_data = {}
        for config_item in config_items:
            config_dict = thrift_to_dict(config_item)
            if config_dict:
                settings_data[config_item.key] = config_dict

        logger.info(f"Retrieved {len(config_items)} configuration items")

        return success_response(
            f"Retrieved {len(config_items)} configuration items",
            {"settings": settings_data, "retrieved_at": datetime.utcnow().isoformat()},
        )

    except (InternalException, UnauthorizedException) as e:
        logger.error(f"Service error getting settings: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve settings: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error getting settings: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while retrieving settings"
        ) from e


@router.put("/")
async def update_settings(
    request: ConfigUpdateRequest, configs: ConfigService = ConfigsDep
):
    """
    Update application settings.

    Updates the application settings with the provided configuration items.
    """
    try:
        logger.info(f"Updating {len(request.configs)} configuration items")

        # Convert request to ConfigItem objects
        config_items = []
        for config_data in request.configs:
            # Create ConfigValue based on the value type
            value = config_data.get("value")
            config_value = ConfigValue()

            if isinstance(
                value, bool
            ):  # Check bool first since bool is a subclass of int in Python
                config_value.boolValue = value
            elif isinstance(value, str):
                config_value.stringValue = value
            elif isinstance(value, int):
                config_value.intValue = value
            elif isinstance(value, float):
                config_value.doubleValue = value
            else:
                config_value.stringValue = str(value) if value is not None else ""

            # Create ConfigItem
            config_item = ConfigItem(
                key=config_data.get("key"),
                value=config_value,
                type=ConfigType.System,  # Default type
                description=config_data.get("description", ""),
            )
            config_items.append(config_item)

        # Update configurations through ConfigService
        updated_configs = await configs.updateConfigs(config_items)

        logger.info(f"Successfully updated {len(updated_configs)} configuration items")

        return success_response(
            f"Updated {len(updated_configs)} configuration items",
            {
                "updated_configs": [
                    thrift_to_dict(config) for config in updated_configs
                ],
                "updated_at": datetime.utcnow().isoformat(),
            },
        )

    except (
        ValidationException,
        NotFoundException,
        InternalException,
        UnauthorizedException,
    ) as e:
        logger.error(f"Service error updating settings: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to update settings: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error updating settings: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while updating settings"
        ) from e


@router.post("/reset")
async def reset_settings(
    request: ConfigResetRequest, configs: ConfigService = ConfigsDep
):
    """
    Reset application settings to defaults.

    Resets configuration items to their default values, optionally filtered by type.
    """
    try:
        logger.info(f"Resetting settings (type={request.config_type})")

        # Convert string to ConfigType enum if provided
        config_type_enum = None
        if request.config_type:
            try:
                config_type_map = {
                    "system": ConfigType.System,
                    "email": ConfigType.Email,
                    "ai": ConfigType.AI,
                    "display": ConfigType.Display,
                    "external_system": ConfigType.ExternalSystem,
                }
                config_type_enum = config_type_map.get(request.config_type.lower())
                if config_type_enum is None:
                    logger.warning(f"Invalid config type: {request.config_type}")
            except Exception:
                logger.warning(f"Invalid config type: {request.config_type}")

        # Reset configurations through ConfigService
        success = await configs.resetConfigs(config_type_enum)

        if success:
            logger.info("Successfully reset configurations")
            return success_response("Configurations reset to defaults successfully")
        else:
            raise HTTPException(
                status_code=500, detail="Failed to reset configurations"
            )

    except (InternalException, UnauthorizedException) as e:
        logger.error(f"Service error resetting settings: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to reset settings: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error resetting settings: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while resetting settings"
        ) from e


@router.get("/export")
async def export_settings(configs: ConfigService = ConfigsDep):
    """
    Export all application settings.

    Returns all configuration items in a format suitable for backup or transfer.
    """
    try:
        logger.info("Exporting all settings")

        # Get all configuration items
        config_items = await configs.getConfigs()

        # Convert to exportable format with metadata
        export_data = {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "app_name": "Budget Helper",
            "export_info": {
                "total_settings": len(config_items),
                "export_format": "full",
                "compatible_versions": ["1.0"],
            },
            "settings": {},
        }

        # Group settings by type for better organization
        settings_by_type = {}
        for config_item in config_items:
            config_dict = thrift_to_dict(config_item)
            if config_dict:
                export_data["settings"][config_item.key] = config_dict

                # Also group by type for metadata
                config_type = config_dict.get("type", "Unknown")
                if config_type not in settings_by_type:
                    settings_by_type[config_type] = 0
                settings_by_type[config_type] += 1

        export_data["export_info"]["settings_by_type"] = settings_by_type

        logger.info(f"Exported {len(config_items)} configuration items")

        return success_response(
            f"Exported {len(config_items)} configuration items", export_data
        )

    except (InternalException, UnauthorizedException) as e:
        logger.error(f"Service error exporting settings: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to export settings: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error exporting settings: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while exporting settings"
        ) from e


class ConfigImportRequest(BaseModel):
    """Request model for importing configuration items."""

    settings: dict[str, Any] = Field(..., description="Settings data to import")
    overwrite_existing: bool = Field(
        True, description="Whether to overwrite existing settings"
    )


@router.post("/import")
async def import_settings(
    request: ConfigImportRequest, configs: ConfigService = ConfigsDep
):
    """
    Import application settings.

    Imports configuration items from exported data. Only accepts files that match
    the exact export schema format to prevent malformed configurations.
    """
    try:
        logger.info(f"Importing settings (overwrite={request.overwrite_existing})")

        import_data = request.settings

        # Validate that this is a proper export format
        if not isinstance(import_data, dict):
            raise HTTPException(
                status_code=400,
                detail="Invalid import format. Expected JSON object.",
            )

        # Check for required export format fields
        required_fields = ["version", "exported_at", "app_name", "settings"]
        missing_fields = [
            field for field in required_fields if field not in import_data
        ]

        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid export format. Missing required fields: {', '.join(missing_fields)}. "
                f"Please use a file exported from this application.",
            )

        # Validate version compatibility
        version = import_data.get("version")
        compatible_versions = ["1.0"]
        if version not in compatible_versions:
            raise HTTPException(
                status_code=400,
                detail=f"Incompatible export version '{version}'. "
                f"Supported versions: {', '.join(compatible_versions)}",
            )

        # Validate app name matches
        app_name = import_data.get("app_name")
        expected_app_name = "Budget Helper"
        if app_name != expected_app_name:
            logger.warning(
                f"App name mismatch: expected '{expected_app_name}', got '{app_name}'"
            )

        # Extract settings data
        settings_data = import_data["settings"]
        if not isinstance(settings_data, dict):
            raise HTTPException(
                status_code=400,
                detail="Invalid settings format. Expected settings to be an object.",
            )

        if not settings_data:
            raise HTTPException(
                status_code=400, detail="No settings data found in import file"
            )

        logger.info(
            f"Importing from export format (version: {version}, "
            f"exported: {import_data.get('exported_at', 'unknown')}, "
            f"settings count: {len(settings_data)})"
        )

        # Convert to ConfigItem objects with strict validation
        config_items = []
        validation_errors = []

        for key, config_data in settings_data.items():
            try:
                # Validate config item structure
                if not isinstance(config_data, dict):
                    validation_errors.append(
                        f"Setting '{key}': Expected object, got {type(config_data).__name__}"
                    )
                    continue

                # Check required fields for each config item
                required_config_fields = ["key", "type", "value", "description"]
                missing_config_fields = [
                    field
                    for field in required_config_fields
                    if field not in config_data
                ]

                if missing_config_fields:
                    validation_errors.append(
                        f"Setting '{key}': Missing fields {missing_config_fields}"
                    )
                    continue

                # Validate key matches
                if config_data["key"] != key:
                    validation_errors.append(f"Setting '{key}': Key mismatch in data")
                    continue

                # Validate value structure
                value_data = config_data["value"]
                if not isinstance(value_data, dict):
                    validation_errors.append(
                        f"Setting '{key}': Value must be an object"
                    )
                    continue

                # Create ConfigValue with strict validation
                config_value = ConfigValue()
                value_set = False

                # Only accept known ConfigValue fields
                valid_value_fields = [
                    "stringValue",
                    "intValue",
                    "doubleValue",
                    "boolValue",
                    "stringList",
                    "stringMap",
                ]
                value_fields_present = [
                    field for field in valid_value_fields if field in value_data
                ]

                if len(value_fields_present) != 1:
                    validation_errors.append(
                        f"Setting '{key}': Value must have exactly one value field, found: {value_fields_present}"
                    )
                    continue

                # Set the appropriate value type
                if "stringValue" in value_data:
                    if not isinstance(value_data["stringValue"], str):
                        validation_errors.append(
                            f"Setting '{key}': stringValue must be a string"
                        )
                        continue
                    config_value.stringValue = value_data["stringValue"]
                    value_set = True
                elif "intValue" in value_data:
                    if not isinstance(value_data["intValue"], int):
                        validation_errors.append(
                            f"Setting '{key}': intValue must be an integer"
                        )
                        continue
                    config_value.intValue = value_data["intValue"]
                    value_set = True
                elif "doubleValue" in value_data:
                    if not isinstance(value_data["doubleValue"], int | float):
                        validation_errors.append(
                            f"Setting '{key}': doubleValue must be a number"
                        )
                        continue
                    config_value.doubleValue = float(value_data["doubleValue"])
                    value_set = True
                elif "boolValue" in value_data:
                    if not isinstance(value_data["boolValue"], bool):
                        validation_errors.append(
                            f"Setting '{key}': boolValue must be a boolean"
                        )
                        continue
                    config_value.boolValue = value_data["boolValue"]
                    value_set = True
                elif "stringList" in value_data:
                    if not isinstance(value_data["stringList"], list) or not all(
                        isinstance(item, str) for item in value_data["stringList"]
                    ):
                        validation_errors.append(
                            f"Setting '{key}': stringList must be a list of strings"
                        )
                        continue
                    config_value.stringList = value_data["stringList"]
                    value_set = True
                elif "stringMap" in value_data:
                    if not isinstance(value_data["stringMap"], dict) or not all(
                        isinstance(k, str) and isinstance(v, str)
                        for k, v in value_data["stringMap"].items()
                    ):
                        validation_errors.append(
                            f"Setting '{key}': stringMap must be a dict of string to string"
                        )
                        continue
                    config_value.stringMap = value_data["stringMap"]
                    value_set = True

                if not value_set:
                    validation_errors.append(
                        f"Setting '{key}': No valid value field found"
                    )
                    continue

                # Validate and convert config type
                config_type_str = config_data["type"]
                if not isinstance(config_type_str, str):
                    validation_errors.append(f"Setting '{key}': type must be a string")
                    continue

                config_type_map = {
                    "System": ConfigType.System,
                    "Email": ConfigType.Email,
                    "AI": ConfigType.AI,
                    "Display": ConfigType.Display,
                    "ExternalSystem": ConfigType.ExternalSystem,
                }

                config_type_enum = config_type_map.get(config_type_str)
                if config_type_enum is None:
                    validation_errors.append(
                        f"Setting '{key}': Invalid type '{config_type_str}'. Valid types: {list(config_type_map.keys())}"
                    )
                    continue

                # Validate description
                description = config_data["description"]
                if not isinstance(description, str):
                    validation_errors.append(
                        f"Setting '{key}': description must be a string"
                    )
                    continue

                # Create ConfigItem
                config_item = ConfigItem(
                    key=key,
                    value=config_value,
                    type=config_type_enum,
                    description=description,
                )
                config_items.append(config_item)

            except Exception as item_error:
                validation_errors.append(
                    f"Setting '{key}': Unexpected error - {str(item_error)}"
                )
                continue

        # If there are validation errors, reject the entire import
        if validation_errors:
            error_summary = (
                f"Import validation failed with {len(validation_errors)} errors:\n"
                + "\n".join(validation_errors[:10])
            )
            if len(validation_errors) > 10:
                error_summary += f"\n... and {len(validation_errors) - 10} more errors"

            logger.error(f"Import validation failed: {validation_errors}")
            raise HTTPException(
                status_code=400,
                detail=error_summary,
            )

        if not config_items:
            raise HTTPException(
                status_code=400,
                detail="No valid configuration items found in import data",
            )

        # Import configurations through ConfigService
        updated_configs = await configs.updateConfigs(config_items)

        import_summary = {
            "total_processed": len(settings_data),
            "successfully_imported": len(updated_configs),
            "validation_errors": 0,
            "source_version": version,
            "source_exported_at": import_data.get("exported_at"),
        }

        logger.info(f"Import completed successfully: {import_summary}")

        return success_response(
            f"Successfully imported {len(updated_configs)} configuration items",
            {
                "imported_configs": [
                    thrift_to_dict(config) for config in updated_configs
                ],
                "import_summary": import_summary,
                "imported_at": datetime.utcnow().isoformat(),
            },
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except (
        ValidationException,
        NotFoundException,
        InternalException,
        UnauthorizedException,
    ) as e:
        logger.error(f"Service error importing settings: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to import settings: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error importing settings: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while importing settings"
        ) from e
