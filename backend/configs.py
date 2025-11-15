"""
This utility service handles configuration management across the system,
implementing the Configs Thrift service interface.
"""

import json
import logging

from thrift_gen.databasestoreaccess.ttypes import (
    Filter,
    FilterOperator,
    FilterValue,
    Query,
)
from thrift_gen.entities.ttypes import (
    BudgetingPlatformType,
    ConfigItem,
    ConfigType,
    ConfigValue,
    EmailPlatformType,
    Entity,
    EntityType,
    ModelType,
)
from thrift_gen.exceptions.ttypes import InternalException, ValidationException

logger = logging.getLogger(__name__)


# Configuration key constants
class ConfigKeys:
    # System configuration keys
    APP_NAME = "app.name"
    APP_VERSION = "app.version"
    SYSTEM_DEFAULT_BUDGET_PLATFORM = "system.default_budget_platform"
    SYSTEM_DEFAULT_EMAIL = "system.default_email"
    SYSTEM_DEFAULT_MODEL_TYPE = "system.default_model_type"
    SYSTEM_DEFAULT_MODEL_NAME = "system.default_model_name"
    SYSTEM_MAX_TRANSACTIONS_TO_LOAD = "system.max_transactions_to_load"

    # Email configuration keys
    EMAIL_SEARCH_DAYS_BUFFER = "email.search_days_buffer"

    # Metadata configuration keys
    METADATA_SEARCH_DAYS_BUFFER = "metadata.search_days_buffer"
    EMAIL_CUSTOM_SEARCH_STRING = "email.custom_search_string"
    EMAIL_INCLUDE_PAYEE_BY_DEFAULT = "email.include_payee_by_default"
    EMAIL_INCLUDE_AMOUNT_BY_DEFAULT = "email.include_amount_by_default"
    EMAIL_APPEND_URL_TO_MEMO = "email.append_url_to_memo"
    EMAIL_BULK_AUTO_ATTACH_SINGLE_RESULT = "email.bulk_auto_attach_single_result"

    # Display configuration keys
    DISPLAY_DEFAULT_SORT_ORDER = "display.default_sort_order"
    DISPLAY_TRANSACTIONS_PER_PAGE = "display.transactions_per_page"
    DISPLAY_VISIBLE_COLUMNS = "display.visible_columns"
    DISPLAY_THEME = "display.theme"

    # External system configuration keys
    EXTERNAL_AUTO_SYNC_ON_STARTUP = "external.auto_sync_on_startup"
    SELECTED_BUDGET_ID = "system.selected_budget_id"
    # Sync performance keys
    SYNC_LAST_LOCAL_EDIT_TIME = "sync.last_local_edit_time"
    SYNC_LOCAL_EDITED_IDS = "sync.local_edited_ids"

    # AI configuration keys
    AI_TRAINING_DATA_MONTHS = "ai.training_data_months"
    AI_TRAINING_TIME_MINUTES = "ai.training_time_minutes"

    # PXBlendSC-RF ML Strategy configuration
    PXBLENDSC_CONFIG = "ai.pxblendsc_config"

    # Authentication configuration keys
    # YNAB authentication
    BUDGET_YNAB_AUTH_CONFIG = "budget.ynab.auth_config"

    # Gmail authentication
    EMAIL_GMAIL_AUTH_CONFIG = "email.gmail.auth_config"
    EMAIL_GMAIL_OAUTH_STATE = "email.gmail.oauth_state"
    EMAIL_GMAIL_TOKENS = "email.gmail.tokens"


# Database field name constants
class FieldNames:
    ID = "id"
    TYPE = "type"
    KEY = "key"
    NAME = "name"
    DATE = "date"
    APPROVED = "approved"
    MODEL_TYPE = "model_type"
    STATUS = "status"
    BUDGET_ID = "budget_id"
    ACCOUNT_ID = "account_id"
    CATEGORY_ID = "category_id"
    PAYEE_ID = "payee_id"


# Default value constants
class ConfigDefaults:
    # System defaults
    APP_NAME = "Budget Helper"
    APP_VERSION = "1.0.0"
    BUDGET_PLATFORM = "YNAB"
    EMAIL_PROVIDER = "GMAIL"
    MODEL_TYPE = "PXBlendSC"
    MAX_TRANSACTIONS_TO_LOAD = 500

    # Email defaults
    SEARCH_DAYS_BUFFER = 3

    # Metadata defaults
    METADATA_SEARCH_DAYS_BUFFER = 3
    CUSTOM_SEARCH_STRING = ""
    INCLUDE_PAYEE_BY_DEFAULT = True
    INCLUDE_AMOUNT_BY_DEFAULT = True
    APPEND_URL_TO_MEMO = True
    BULK_AUTO_ATTACH_SINGLE_RESULT = True

    # Display defaults
    SORT_ORDER = "desc_by_date"
    TRANSACTIONS_PER_PAGE = 25
    VISIBLE_COLUMNS = "all"
    THEME = "system"

    # External system defaults
    AUTO_SYNC_ON_STARTUP = True
    # Sync defaults
    LAST_LOCAL_EDIT_TIME = None
    LOCAL_EDITED_IDS: list[str] = []

    # AI defaults
    TRAINING_DATA_MONTHS = 6
    TRAINING_TIME_MINUTES = 15

    # PXBlendSC-RF default configuration
    PXBLENDSC_CONFIG = {
        "columns": {
            "LABEL_COL": "category_name",
            "TEXT_COLS": ["payee_name", "memo", "account_name"],
            "NUM_COLS": ["amount"],
            "DATE_COL": "date",
        },
        "random_state": 42,
        "cv": {
            "n_folds": 3,
            "force_folds": True,
        },  # Force 3 folds. Slower and drops more rare categories but better performance.
        "parallel": {"n_jobs_cv": None, "threads_per_fold": None},
        "models": {
            "use_lgbm": True,
            "use_svm_blend": True,
            "use_recency_frequency": True,
            "svm_calibration": "sigmoid",
            "lgbm_weight": 0.4,
            "svm_weight": 0.2,
            "recency_freq_weight": 0.4,
            # Adaptive training strategies for small classes
            "adaptive_splitting": True,  # Use adaptive train/test split for better small class performance
            "final_retraining": True,  # Retrain final model on all data (training + test) after evaluation
            "recency_freq_params": {
                "recency_weight": 0.6,  # How much to weight recent vs frequent
                "frequency_weight": 0.4,
                "min_frequency": 3,  # Ignore one-off categories
                "lookback_window": 50,  # Consider last 50 transactions per payee
                "recency_window": 5,  # Weight last 5 transactions heavily
            },
            "lgbm_params": {
                "learning_rate": 0.03,
                "n_estimators": 750,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 20.0,
                "reg_alpha": 7.0,
                "min_child_samples": 30,
                "num_leaves": 20,
                "max_depth": 5,
                "force_row_wise": True,
                "verbosity": -1,
                "early_stopping_rounds": 50,
            },
        },
        "features": {
            "tfidf_word_max_features": 5000,
            "tfidf_char_max_features": 2500,
            "hashed_cross": {
                "n_features": 4096,
                "payee_min_count": 3,
                "quantiles": [0.15, 0.3, 0.5, 0.7, 0.85],
                "sign_bins": "two_sided",
                "emit_unaries": True,
                "emit_pairs": True,
                "emit_triples": True,
            },
        },
        "sampler": {"ros_cap_percentile": 60},
        "priors": {
            "vendor": {"min_count": 3, "hard_override_share": 0.85, "beta": 2.0},
            "recency": {"k_last": 8, "beta": 1.5},
            "backoff": {
                "global": 0.40,
            },
        },
        "thresholds": {
            "mode": "learn",
            "global_grid": [0.15, 0.2, 0.25, 0.3, 0.35],  # Lower thresholds
            "tail_support_max": 8,
            "tail_f1_max": 0.25,
            "tail_grid": [
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.375,
                0.4,
                0.425,
                0.45,
                0.475,
                0.5,
                0.525,
                0.55,
                0.575,
                0.6,
                0.625,
                0.65,
                0.675,
                0.7,
            ],
        },
        "alias": {"generic_noise": True, "custom_map_path": None},
    }


# Global config service instance
_config_service_instance = None


def get_config_service(database_store_access=None):
    """Get or create the global config service instance."""
    global _config_service_instance
    if _config_service_instance is None and database_store_access is not None:
        _config_service_instance = ConfigService(database_store_access)
    return _config_service_instance


class ConfigService:
    """
    ConfigService handles system configuration management.

    This is a utility service that can be called by any layer
    to manage application configuration settings.

    Public methods correspond exactly to the Configs Thrift interface.
    All other methods are private (prefixed with _).
    """

    def __init__(self, database_store_access):
        """
        Initialize ConfigService.

        Args:
            database_store_access: DatabaseStoreAccess service for data persistence
        """
        self.database_store_access = database_store_access
        self._initialized = False
        self._initializing = False  # Flag to prevent recursion during initialization
        self._config_cache = {}  # Cache for config values
        self._cache_timestamp = None
        self._cache_ttl = 30  # Cache for 30 seconds
        logger.info("ConfigService initialized")

    async def _ensure_initialized(self):
        """Ensure configs are initialized with defaults if needed."""
        if not self._initialized and not self._initializing:
            self._initializing = True
            try:
                await self._initialize_configs()
                self._initialized = True
            finally:
                self._initializing = False

    async def updateConfigs(self, configs: list[ConfigItem]) -> list[ConfigItem]:
        """
        Update configuration items.

        Args:
            configs: List of configuration items to update

        Returns:
            List of updated configuration items

        Raises:
            ValidationException: If config data is invalid
            NotFoundException: If config item doesn't exist
            InternalException: If internal error occurs
            UnauthorizedException: If user not authorized
        """
        try:
            # Only ensure initialization if we're not currently initializing (prevent recursion)
            if not self._initializing:
                await self._ensure_initialized()
            logger.info(f"Updating {len(configs)} configuration items")

            # Validate configuration items
            for config in configs:
                if not config.key:
                    raise ValidationException("Configuration key cannot be empty")
                if not config.value:
                    raise ValidationException("Configuration value cannot be empty")

            # Convert ConfigItems to Entity objects for database storage
            entities = []
            for config in configs:
                entity = Entity(configItem=config)
                entities.append(entity)

            # Update configurations through DatabaseStoreAccess
            updated_entities = await self.database_store_access.upsertEntities(entities)

            # Extract ConfigItems from updated entities
            updated_configs = []
            for entity in updated_entities:
                if entity.configItem:
                    updated_configs.append(entity.configItem)
                    # Clear cache for updated config
                    if entity.configItem.key in self._config_cache:
                        del self._config_cache[entity.configItem.key]
                    logger.debug(f"Updated config: {entity.configItem.key}")

            logger.info(f"Successfully updated {len(updated_configs)} configurations")
            return updated_configs

        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Failed to update configurations: {e}")
            raise InternalException(f"Configuration update failed: {str(e)}") from e

    async def getConfigs(
        self, type: ConfigType | None = None, key: str | None = None
    ) -> list[ConfigItem]:
        """
        Retrieve configuration items.

        Args:
            type: Optional config type filter
            key: Optional config key filter

        Returns:
            List of configuration items matching the filters

        Raises:
            InternalException: If internal error occurs
            UnauthorizedException: If user not authorized
        """
        try:
            # Only ensure initialization if we're not currently initializing (prevent recursion)
            if not self._initializing:
                await self._ensure_initialized()

            # Check cache first for single key lookups
            if key is not None and type is None:
                cached_value = self._get_cached_config(key)
                if cached_value is not None:
                    return [cached_value]

            logger.info(f"Retrieving configurations (type={type}, key={key})")

            # Build query for configurations
            query = Query(entityType=EntityType.ConfigItem)

            # Add filters if provided
            if type is not None or key is not None:
                query.filters = []
                if type is not None:
                    # Convert ConfigType enum to database string value using Thrift's built-in mapping
                    type_value = ConfigType._VALUES_TO_NAMES.get(type)
                    if type_value is None:
                        logger.warning(
                            f"Unknown ConfigType value: {type}, defaulting to System"
                        )
                        type_value = "System"
                    type_filter = Filter(
                        fieldName=FieldNames.TYPE,
                        operator=FilterOperator.EQ,
                        value=FilterValue(stringValue=type_value),
                    )
                    query.filters.append(type_filter)

                if key is not None:
                    key_filter = Filter(
                        fieldName=FieldNames.KEY,
                        operator=FilterOperator.EQ,
                        value=FilterValue(stringValue=key),
                    )
                    query.filters.append(key_filter)

            # Query configurations through DatabaseStoreAccess
            result = await self.database_store_access.getEntities(query)

            # Extract ConfigItems from entities
            configs = []
            for entity in result.entities:
                if entity.configItem:
                    configs.append(entity.configItem)
                    # Cache single key lookups
                    if key is not None and type is None:
                        self._cache_config(entity.configItem)

            logger.info(f"Retrieved {len(configs)} configurations")
            return configs

        except Exception as e:
            logger.error(f"Failed to retrieve configurations: {e}")
            raise InternalException(f"Configuration retrieval failed: {str(e)}") from e

    async def resetConfigs(self, type: ConfigType | None = None) -> bool:
        """
        Reset configuration items to defaults.

        Args:
            type: Optional config type to reset (if None, resets all)

        Returns:
            True if reset was successful

        Raises:
            InternalException: If internal error occurs
            UnauthorizedException: If user not authorized
        """
        try:
            # Only ensure initialization if we're not currently initializing (prevent recursion)
            if not self._initializing:
                await self._ensure_initialized()
            logger.info(f"Resetting configurations (type={type})")

            # Get existing configurations to delete
            existing_configs = await self.getConfigs(type=type)

            if existing_configs:
                # Delete existing configurations
                config_ids = [config.key for config in existing_configs]
                await self.database_store_access.deleteEntities(
                    EntityType.ConfigItem, config_ids
                )
                logger.info(f"Deleted {len(config_ids)} existing configurations")

            # Create default configurations
            default_configs = self._get_default_configs(type)
            if default_configs:
                await self.updateConfigs(default_configs)
                logger.info(f"Created {len(default_configs)} default configurations")

            logger.info("Successfully reset configurations")
            return True

        except Exception as e:
            logger.error(f"Failed to reset configurations: {e}")
            raise InternalException(f"Configuration reset failed: {str(e)}") from e

    async def getConfigValue(self, key: str, default_value=None):
        """
        Get a single configuration value by key.

        Args:
            key: Configuration key
            default_value: Default value if config not found

        Returns:
            Configuration value or default
        """
        try:
            configs = await self.getConfigs(key=key)
            if configs:
                config = configs[0]
                # Extract the actual value from the ConfigValue union
                return self._extract_config_value(config.value)
            return default_value
        except Exception as e:
            logger.warning(f"Failed to get config value for {key}: {e}")
            return default_value

    async def getDefaultBudgetPlatform(self):
        """Get the default budget platform type."""
        platform_str = await self.getConfigValue(
            ConfigKeys.SYSTEM_DEFAULT_BUDGET_PLATFORM, ConfigDefaults.BUDGET_PLATFORM
        )
        return getattr(BudgetingPlatformType, platform_str, BudgetingPlatformType.YNAB)

    async def getDefaultEmailProvider(self):
        """Get the default email provider type."""
        provider_str = await self.getConfigValue(
            ConfigKeys.SYSTEM_DEFAULT_EMAIL, ConfigDefaults.EMAIL_PROVIDER
        )
        return getattr(EmailPlatformType, provider_str, EmailPlatformType.Gmail)

    async def getDefaultModelType(self):
        """Get the default ML model type."""
        model_str = await self.getConfigValue(
            ConfigKeys.SYSTEM_DEFAULT_MODEL_TYPE, ConfigDefaults.MODEL_TYPE
        )
        return getattr(ModelType, model_str, ModelType.PXBlendSC)

    async def getDefaultModelName(self):
        """Get the default ML model name."""
        return await self.getConfigValue(ConfigKeys.SYSTEM_DEFAULT_MODEL_NAME, None)

    async def setDefaultModelName(self, model_name: str):
        """Set the default ML model name."""
        config_item = ConfigItem(
            key=ConfigKeys.SYSTEM_DEFAULT_MODEL_NAME,
            type=ConfigType.System,
            value=ConfigValue(stringValue=model_name),
            description="Default ML model name for predictions",
        )
        await self.updateConfigs([config_item])

    async def getPXBlendSCConfig(self):
        """Get the PXBlendSC-RF configuration as a dictionary."""
        # Return the config from code defaults for easier development iteration
        # return ConfigDefaults.PXBLENDSC_CONFIG
        return json.loads(await self.getConfigValue(ConfigKeys.PXBLENDSC_CONFIG, ConfigDefaults.PXBLENDSC_CONFIG))

    def _extract_config_value(self, config_value: ConfigValue):
        """
        Extract the actual value from a ConfigValue union.

        Args:
            config_value: ConfigValue union object

        Returns:
            The actual value (string, int, bool, etc.) or None if no value is set
        """
        # Check each possible value type in priority order
        # Simple types first, then complex types
        value_attrs = [
            "stringValue",  # Most common
            "intValue",  # Numbers
            "doubleValue",  # Floating point
            "boolValue",  # Booleans
            "stringList",  # Lists
            "stringMap",  # Maps/dictionaries
        ]

        for attr in value_attrs:
            value = getattr(config_value, attr, None)
            if value is not None:
                return value

        return None

    def _get_cached_config(self, key: str) -> ConfigItem | None:
        """Get config from cache if valid."""
        import time

        if not self._cache_timestamp:
            return None

        # Check if cache is expired
        if time.time() - self._cache_timestamp > self._cache_ttl:
            self._config_cache.clear()
            self._cache_timestamp = None
            return None

        return self._config_cache.get(key)

    def _cache_config(self, config_item: ConfigItem):
        """Cache a config item."""
        import time

        if not self._cache_timestamp:
            self._cache_timestamp = time.time()

        self._config_cache[config_item.key] = config_item

    async def _initialize_configs(self):
        """Initialize configs with defaults if none exist."""
        try:
            # Check if any configs exist by directly querying database (avoid recursion)
            query = Query(entityType=EntityType.ConfigItem)
            result = await self.database_store_access.getEntities(query)

            if not result.entities:
                logger.info("No configurations found, initializing with defaults")
                # Create default configurations without calling resetConfigs (which would cause recursion)
                default_configs = self._get_default_configs()
                if default_configs:
                    # Convert ConfigItems to Entity objects for database storage
                    entities = []
                    for config in default_configs:
                        entity = Entity(configItem=config)
                        entities.append(entity)

                    # Insert default configurations directly
                    await self.database_store_access.upsertEntities(entities)
                    logger.info(
                        f"Created {len(default_configs)} default configurations"
                    )
        except Exception as e:
            logger.error(f"Failed to initialize configs: {e}")

    def _get_default_configs(self, type: ConfigType | None = None) -> list[ConfigItem]:
        """
        Get default configuration items.

        Args:
            type: Optional config type filter

        Returns:
            List of default configuration items
        """
        defaults = []

        # System defaults
        if type is None or type == ConfigType.System:
            defaults.extend(
                [
                    ConfigItem(
                        key=ConfigKeys.APP_NAME,
                        type=ConfigType.System,
                        value=ConfigValue(stringValue=ConfigDefaults.APP_NAME),
                        description="Application name",
                    ),
                    ConfigItem(
                        key=ConfigKeys.APP_VERSION,
                        type=ConfigType.System,
                        value=ConfigValue(stringValue=ConfigDefaults.APP_VERSION),
                        description="Application version",
                    ),
                    ConfigItem(
                        key=ConfigKeys.SYSTEM_DEFAULT_BUDGET_PLATFORM,
                        type=ConfigType.System,
                        value=ConfigValue(stringValue=ConfigDefaults.BUDGET_PLATFORM),
                        description="Default budget platform",
                    ),
                    ConfigItem(
                        key=ConfigKeys.SYSTEM_DEFAULT_EMAIL,
                        type=ConfigType.System,
                        value=ConfigValue(stringValue=ConfigDefaults.EMAIL_PROVIDER),
                        description="Default email provider",
                    ),
                    ConfigItem(
                        key=ConfigKeys.SYSTEM_DEFAULT_MODEL_TYPE,
                        type=ConfigType.System,
                        value=ConfigValue(stringValue=ConfigDefaults.MODEL_TYPE),
                        description="Default ML model type",
                    ),
                    ConfigItem(
                        key=ConfigKeys.SYSTEM_MAX_TRANSACTIONS_TO_LOAD,
                        type=ConfigType.System,
                        value=ConfigValue(
                            intValue=ConfigDefaults.MAX_TRANSACTIONS_TO_LOAD
                        ),
                        description="Maximum number of transactions to load per request on Transactions page",
                    ),
                ]
            )

        # Email defaults
        if type is None or type == ConfigType.Email:
            defaults.extend(
                [
                    ConfigItem(
                        key=ConfigKeys.EMAIL_SEARCH_DAYS_BUFFER,
                        type=ConfigType.Email,
                        value=ConfigValue(intValue=ConfigDefaults.SEARCH_DAYS_BUFFER),
                        description="Number of days before and after transaction date to search for emails",
                    ),
                    ConfigItem(
                        key=ConfigKeys.EMAIL_CUSTOM_SEARCH_STRING,
                        type=ConfigType.Email,
                        value=ConfigValue(
                            stringValue=ConfigDefaults.CUSTOM_SEARCH_STRING
                        ),
                        description="Custom search string with variables: {payee}, {amount}, {date}",
                    ),
                    ConfigItem(
                        key=ConfigKeys.EMAIL_INCLUDE_PAYEE_BY_DEFAULT,
                        type=ConfigType.Email,
                        value=ConfigValue(
                            boolValue=ConfigDefaults.INCLUDE_PAYEE_BY_DEFAULT
                        ),
                        description="Include payee in email search by default",
                    ),
                    ConfigItem(
                        key=ConfigKeys.EMAIL_INCLUDE_AMOUNT_BY_DEFAULT,
                        type=ConfigType.Email,
                        value=ConfigValue(
                            boolValue=ConfigDefaults.INCLUDE_AMOUNT_BY_DEFAULT
                        ),
                        description="Include amount in email search by default",
                    ),
                    ConfigItem(
                        key=ConfigKeys.EMAIL_APPEND_URL_TO_MEMO,
                        type=ConfigType.Email,
                        value=ConfigValue(boolValue=ConfigDefaults.APPEND_URL_TO_MEMO),
                        description="Append email URL to memo when only 1 email is found",
                    ),
                    ConfigItem(
                        key=ConfigKeys.EMAIL_BULK_AUTO_ATTACH_SINGLE_RESULT,
                        type=ConfigType.Email,
                        value=ConfigValue(
                            boolValue=ConfigDefaults.BULK_AUTO_ATTACH_SINGLE_RESULT
                        ),
                        description="When bulk searching, auto-attach email when there's only 1 result",
                    ),
                ]
            )

        # Display defaults
        if type is None or type == ConfigType.Display:
            defaults.extend(
                [
                    ConfigItem(
                        key=ConfigKeys.DISPLAY_DEFAULT_SORT_ORDER,
                        type=ConfigType.Display,
                        value=ConfigValue(stringValue=ConfigDefaults.SORT_ORDER),
                        description="Default sort order for transactions",
                    ),
                    ConfigItem(
                        key=ConfigKeys.DISPLAY_TRANSACTIONS_PER_PAGE,
                        type=ConfigType.Display,
                        value=ConfigValue(
                            intValue=ConfigDefaults.TRANSACTIONS_PER_PAGE
                        ),
                        description="Number of transactions to display per page",
                    ),
                    ConfigItem(
                        key=ConfigKeys.DISPLAY_VISIBLE_COLUMNS,
                        type=ConfigType.Display,
                        value=ConfigValue(stringValue=ConfigDefaults.VISIBLE_COLUMNS),
                        description="Which columns to display in the transaction grid",
                    ),
                    ConfigItem(
                        key=ConfigKeys.DISPLAY_THEME,
                        type=ConfigType.Display,
                        value=ConfigValue(stringValue=ConfigDefaults.THEME),
                        description="UI theme (light, dark, system)",
                    ),
                ]
            )

        # Metadata defaults
        if type is None or type == ConfigType.System:
            defaults.extend(
                [
                    ConfigItem(
                        key=ConfigKeys.METADATA_SEARCH_DAYS_BUFFER,
                        type=ConfigType.System,
                        value=ConfigValue(
                            intValue=ConfigDefaults.METADATA_SEARCH_DAYS_BUFFER
                        ),
                        description="Default number of days before and after transaction date to search for metadata",
                    )
                ]
            )

        # External System defaults
        if type is None or type == ConfigType.ExternalSystem:
            defaults.extend(
                [
                    ConfigItem(
                        key=ConfigKeys.EXTERNAL_AUTO_SYNC_ON_STARTUP,
                        type=ConfigType.ExternalSystem,
                        value=ConfigValue(
                            boolValue=ConfigDefaults.AUTO_SYNC_ON_STARTUP
                        ),
                        description="Automatically sync external systems on startup",
                    )
                ]
            )

        # AI defaults
        if type is None or type == ConfigType.AI:
            defaults.extend(
                [
                    ConfigItem(
                        key=ConfigKeys.AI_TRAINING_DATA_MONTHS,
                        type=ConfigType.AI,
                        value=ConfigValue(intValue=ConfigDefaults.TRAINING_DATA_MONTHS),
                        description="Number of months of data to use for training",
                    ),
                    ConfigItem(
                        key=ConfigKeys.AI_TRAINING_TIME_MINUTES,
                        type=ConfigType.AI,
                        value=ConfigValue(
                            intValue=ConfigDefaults.TRAINING_TIME_MINUTES
                        ),
                        description="Maximum training time in minutes",
                    ),
                    ConfigItem(
                        key=ConfigKeys.PXBLENDSC_CONFIG,
                        type=ConfigType.AI,
                        value=ConfigValue(
                            stringValue=json.dumps(ConfigDefaults.PXBLENDSC_CONFIG)
                        ),
                        description="PXBlendSC-RF ML strategy configuration parameters",
                    ),
                ]
            )

        return defaults
