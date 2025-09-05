"""
TransactionManager implementation.

This Manager orchestrates transaction-related use cases and workflows,
implementing the TransactionManager Thrift service interface.
"""

import logging
from datetime import UTC

from configs import ConfigKeys, FieldNames
from thrift_gen.databasestoreaccess.ttypes import (
    Filter,
    FilterOperator,
    FilterValue,
    Query,
)
from thrift_gen.entities.ttypes import (
    BudgetingPlatformType,
    Entity,
    EntityType,
    Metadata,
    SyncStatus,
    Transaction,
    Transactions,
)
from thrift_gen.exceptions.ttypes import (
    ConflictException,
    InternalException,
    NotFoundException,
    UnauthorizedException,
    ValidationException,
)
from thrift_gen.metadatafindingengine.ttypes import MetadataFilter
from thrift_gen.transactionmanager.ttypes import (
    BudgetsInfoResult,
    SyncResult,
    SyncResultItem,
    TransactionEdit,
)

logger = logging.getLogger(__name__)


class TransactionManager:
    """
    TransactionManager implements transaction workflow orchestration.

    This Manager coordinates between Engines and ResourceAccess services
    to implement complete transaction-related use cases.

    Public methods correspond exactly to the TransactionManager Thrift interface.
    All other methods are private (prefixed with _).
    """

    def __init__(
        self,
        metadata_finding_engine=None,
        ml_engine=None,
        budgeting_platform_access=None,
        metadata_source_access=None,
        database_store_access=None,
        config_service=None,
    ):
        """
        Initialize TransactionManager with required services.

        Args:
            metadata_finding_engine: MetadataFindingEngine instance
            ml_engine: MLEngine instance
            budgeting_platform_access: BudgetingPlatformAccess instance
            metadata_source_access: MetadataSourceAccess instance
            database_store_access: DatabaseStoreAccess instance
            config_service: ConfigService instance
        """
        self.metadata_finding_engine = metadata_finding_engine
        self.ml_engine = ml_engine
        self.budgeting_platform_access = budgeting_platform_access
        self.metadata_source_access = metadata_source_access
        self.database_store_access = database_store_access
        self.config_service = config_service

        # Control auto sync behavior (can be toggled via config later)
        self.auto_sync_on_update = False
        logger.info("TransactionManager initialized (auto_sync_on_update=False)")

    async def getBudgetsInfo(
        self,
        budgetIds: list[str] | None = None,
        entityTypes: list[EntityType] | None = None,
        refreshData: bool = False,
    ) -> BudgetsInfoResult:
        """
        Get budget information including categories, payees, and accounts.

        Args:
            budgetIds: Optional list of budget IDs to retrieve (defaults to default budget)
            entityTypes: Optional list of entity types to include (Account, Payee, Category)
            refreshData: Whether to refresh data from YNAB before returning (defaults to False)

        Returns:
            BudgetsInfoResult containing budgets and requested entity data

        Raises:
            NotFoundException: If budget not found
            ValidationException: If validation fails
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
        """
        try:
            logger.info(
                f"Getting budget info for budgetIds: {budgetIds}, entityTypes: {entityTypes}, refreshData: {refreshData}"
            )

            # Refresh data from YNAB if requested
            if refreshData:
                logger.info("Refreshing budget metadata from YNAB...")
                platform = await self.config_service.getDefaultBudgetPlatform()
                await self._syncReferenceData(platform)
                logger.info("Budget metadata refresh completed")

            # Get budgets from database (use default if none specified)
            if not budgetIds:
                # Get all budgets from database and use first as default
                budget_query = Query(entityType=EntityType.Budget)
                budget_result = await self.database_store_access.getEntities(
                    budget_query
                )

                budgets = []
                for entity in budget_result.entities:
                    if entity.budget:
                        budgets.append(entity.budget)

                if not budgets:
                    raise NotFoundException("No budgets found in database")
                # Return all budgets when no specific budgetIds are requested
                logger.info(f"Found {len(budgets)} budgets in database")
            else:
                # Get specific budgets from database by ID
                budget_entities = await self.database_store_access.getEntitiesById(
                    EntityType.Budget, budgetIds
                )

                budgets = []
                found_ids = set()
                for entity in budget_entities:
                    if entity.budget:
                        budgets.append(entity.budget)
                        found_ids.add(entity.budget.id)

                # Check if all requested budgets were found
                missing_ids = set(budgetIds) - found_ids
                if missing_ids:
                    raise NotFoundException(
                        f"Budgets not found in database: {list(missing_ids)}"
                    )
                logger.info(f"Found {len(budgets)} requested budgets from database")

            result = BudgetsInfoResult(budgets=budgets)

            # Get additional entities from database if requested
            if entityTypes:
                if EntityType.Category in entityTypes:
                    logger.info("Fetching categories from database")
                    category_query = Query(entityType=EntityType.Category)
                    category_result = await self.database_store_access.getEntities(
                        category_query
                    )
                    result.categories = []
                    for entity in category_result.entities:
                        if entity.category:
                            result.categories.append(entity.category)

                if EntityType.Payee in entityTypes:
                    logger.info("Fetching payees from database")
                    payee_query = Query(entityType=EntityType.Payee)
                    payee_result = await self.database_store_access.getEntities(
                        payee_query
                    )
                    result.payees = []
                    for entity in payee_result.entities:
                        if entity.payee:
                            result.payees.append(entity.payee)

                if EntityType.Account in entityTypes:
                    logger.info("Fetching accounts from database")
                    account_query = Query(entityType=EntityType.Account)
                    account_result = await self.database_store_access.getEntities(
                        account_query
                    )
                    result.accounts = []
                    for entity in account_result.entities:
                        if entity.account:
                            result.accounts.append(entity.account)

            logger.info(
                f"Budget info retrieved successfully from database: {len(result.budgets)} budgets, "
                f"{len(result.categories or [])} categories, "
                f"{len(result.payees or [])} payees, "
                f"{len(result.accounts or [])} accounts"
            )

            return result

        except (NotFoundException, ValidationException, UnauthorizedException):
            raise
        except Exception as e:
            logger.error(f"Error getting budget info: {e}")
            raise InternalException(f"Failed to get budget info: {str(e)}") from e

    async def getTransaction(self, transactionId: str) -> Transaction:
        """
        Get a single transaction by ID.

        Args:
            transactionId: Transaction ID to retrieve

        Returns:
            Transaction entity

        Raises:
            NotFoundException: If transaction not found
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
        """
        try:
            logger.info(f"Getting transaction: {transactionId}")

            # Use DatabaseStoreAccess to get transaction

            # Create query for transaction by ID
            filter_value = FilterValue(stringValue=transactionId)
            id_filter = Filter(
                fieldName=FieldNames.ID, operator=FilterOperator.EQ, value=filter_value
            )

            query = Query(
                entityType=EntityType.Transaction, filters=[id_filter], limit=1
            )

            result = await self.database_store_access.getEntities(query)

            if not result.entities:
                raise NotFoundException(f"Transaction {transactionId} not found")

            # Extract transaction from Entity union
            entity = result.entities[0]
            if not entity.transaction:
                raise InternalException("Invalid entity type returned")

            return entity.transaction

        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Error getting transaction {transactionId}: {e}")
            raise InternalException(f"Failed to get transaction: {str(e)}") from e

    async def getTransactions(
        self,
        transactionIds: list[str] | None = None,
        filters: list[Filter] | None = None,
    ) -> list[Transaction]:
        """
        Get multiple transactions by IDs or filters.

        Args:
            transactionIds: Optional list of transaction IDs to retrieve
            filters: Optional list of filters to apply

        Returns:
            List of transaction entities

        Raises:
            NotFoundException: If any transaction not found (when using IDs)
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
        """
        try:
            # Build filters list
            query_filters = list(filters) if filters else []

            # Add ID filter if transaction IDs are provided
            if transactionIds:
                logger.info(
                    f"Getting {len(transactionIds)} transactions by IDs with additional filters"
                )
                id_filter_value = FilterValue(stringListValue=transactionIds)
                id_filter = Filter(
                    fieldName=FieldNames.ID,
                    operator=FilterOperator.IN,
                    value=id_filter_value,
                )
                query_filters.append(id_filter)
            else:
                logger.info("Getting transactions with filters")

            # Create single query with all filters
            query = Query(entityType=EntityType.Transaction, filters=query_filters)

            result = await self.database_store_access.getEntities(query)

            transactions = []
            found_ids = set()

            for entity in result.entities:
                if entity.transaction:
                    transactions.append(entity.transaction)
                    found_ids.add(entity.transaction.id)

            # Check if all requested transactions were found (only when specific IDs requested)
            if transactionIds:
                missing_ids = set(transactionIds) - found_ids
                if missing_ids:
                    raise NotFoundException(
                        f"Transactions not found: {', '.join(missing_ids)}"
                    )

            logger.info(f"Retrieved {len(transactions)} transactions")
            return transactions

        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Error getting transactions: {e}")
            raise InternalException(f"Failed to get transactions: {str(e)}") from e

    async def updateTransactions(
        self, transactionEdits: list[TransactionEdit]
    ) -> list[Transaction]:
        """
        Update multiple transactions with automatic YNAB sync.

        Args:
            transactionEdits: List of transaction edits to apply

        Returns:
            List of updated transactions

        Raises:
            NotFoundException: If transaction not found
            ValidationException: If validation fails
            ConflictException: If conflict occurs
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
        """
        try:
            logger.info(f"Updating {len(transactionEdits)} transactions")

            updated_transactions = []
            # Track all touched transactions for reliable edit tracking even if upserts return empty
            touched_ids: set[str] = set()
            touched_dates: list = []  # populated from existing transaction dates

            for edit in transactionEdits:
                # Get existing transaction
                existing_transaction = await self.getTransaction(edit.transactionId)

                # Apply edits to create updated transaction
                logger.info(
                    f"Updating transaction {edit.transactionId}: existing metadata count = {len(existing_transaction.metadata or [])}, new metadata count = {len(edit.metadata or [])}"
                )

                # Record touched ID and date for edit tracking
                try:
                    touched_ids.add(existing_transaction.id)
                    # Parse existing transaction date; prefer ISO then date-only
                    from datetime import datetime

                    def _parse_dt_local(s: str) -> datetime | None:
                        try:
                            return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
                        except Exception:
                            try:
                                return datetime.strptime(str(s), "%Y-%m-%d").replace(
                                    tzinfo=UTC
                                )
                            except Exception:
                                return None

                    if getattr(existing_transaction, "date", None):
                        dt = _parse_dt_local(existing_transaction.date)
                        if dt is not None:
                            touched_dates.append(dt)
                except Exception:
                    # Non-fatal: continue without early tracking
                    pass

                updated_transaction = Transaction(
                    id=existing_transaction.id,
                    date=existing_transaction.date,
                    amount=existing_transaction.amount,
                    approved=edit.approved
                    if edit.approved is not None
                    else existing_transaction.approved,
                    platformType=existing_transaction.platformType,
                    payeeId=existing_transaction.payeeId,
                    categoryId=edit.categoryId
                    if edit.categoryId is not None
                    else existing_transaction.categoryId,
                    accountId=existing_transaction.accountId,
                    budgetId=existing_transaction.budgetId,
                    memo=edit.memo
                    if edit.memo is not None
                    else existing_transaction.memo,
                    metadata=edit.metadata
                    if edit.metadata is not None
                    else existing_transaction.metadata,
                )

                logger.info(
                    f"Updated transaction metadata count = {len(updated_transaction.metadata or [])}"
                )

                # Update via DatabaseStoreAccess
                entity = Entity(transaction=updated_transaction)

                result = await self.database_store_access.upsertEntities([entity])
                if result and result[0].transaction:
                    updated_transactions.append(result[0].transaction)
                else:
                    raise InternalException(
                        f"Failed to update transaction {edit.transactionId}"
                    )

            # Trigger automatic sync only if enabled
            if self.auto_sync_on_update:
                await self._syncUpdatedTransactions(updated_transactions)
            else:
                # Track local edits for optimized unified preview
                try:
                    from datetime import datetime

                    from thrift_gen.entities.ttypes import (
                        ConfigItem,
                        ConfigType,
                        ConfigValue,
                    )

                    # If DB returned updated transactions, include their IDs/dates too
                    try:
                        for _t in updated_transactions or []:
                            if getattr(_t, "id", None):
                                touched_ids.add(_t.id)
                            if getattr(_t, "date", None):
                                # Parse date from returned transaction as well
                                def _parse_dt_iso(s: str) -> datetime | None:
                                    try:
                                        return datetime.fromisoformat(
                                            str(s).replace("Z", "+00:00")
                                        )
                                    except Exception:
                                        try:
                                            return datetime.strptime(
                                                str(s), "%Y-%m-%d"
                                            ).replace(tzinfo=UTC)
                                        except Exception:
                                            return None

                                dt2 = _parse_dt_iso(_t.date)
                                if dt2 is not None:
                                    touched_dates.append(dt2)
                    except Exception:
                        pass

                    # Determine earliest touched date; fallback to now if none parsed
                    min_edit_dt = (
                        min(touched_dates) if touched_dates else datetime.now(UTC)
                    )

                    # Merge with existing baseline: move baseline earlier if needed, never later
                    existing_baseline = await self.config_service.getConfigValue(
                        ConfigKeys.SYNC_LAST_LOCAL_EDIT_TIME, None
                    )

                    def _parse_dt_any(s: str | None) -> datetime | None:
                        if not s:
                            return None
                        try:
                            return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
                        except Exception:
                            try:
                                return datetime.strptime(str(s), "%Y-%m-%d").replace(
                                    tzinfo=UTC
                                )
                            except Exception:
                                return None

                    existing_dt = (
                        _parse_dt_any(existing_baseline)
                        if isinstance(existing_baseline, str)
                        else None
                    )
                    new_baseline_dt = (
                        min(existing_dt, min_edit_dt) if existing_dt else min_edit_dt
                    )

                    # Store baseline as date-only ISO (YYYY-MM-DD) for stable YNAB since_date
                    baseline_str = new_baseline_dt.date().isoformat()

                    # Append IDs to edited set (set-union)
                    edited_ids_list = list(touched_ids)
                    existing_ids = (
                        await self.config_service.getConfigValue(
                            ConfigKeys.SYNC_LOCAL_EDITED_IDS, []
                        )
                        or []
                    )
                    merged = list({*existing_ids, *edited_ids_list})

                    updates = [
                        ConfigItem(
                            key=ConfigKeys.SYNC_LAST_LOCAL_EDIT_TIME,
                            type=ConfigType.System,
                            value=ConfigValue(stringValue=baseline_str),
                            description="Earliest date of locally edited transactions requiring comparison",
                        ),
                        ConfigItem(
                            key=ConfigKeys.SYNC_LOCAL_EDITED_IDS,
                            type=ConfigType.System,
                            value=ConfigValue(stringList=merged),
                            description="Locally edited transaction IDs awaiting sync",
                        ),
                    ]
                    await self.config_service.updateConfigs(updates)
                    try:
                        logger.info(
                            f"Edit tracking updated: baseline={baseline_str}, edited_ids_count={len(merged)}"
                        )
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(f"Failed to update local edit tracking config: {e}")

            return updated_transactions

        except (NotFoundException, ValidationException, ConflictException):
            raise
        except Exception as e:
            logger.error(f"Error updating transactions: {e}")
            raise InternalException(f"Failed to update transactions: {str(e)}") from e

    async def attachTransactionMetadata(
        self, transactionId: str, metadata: Metadata
    ) -> Transaction:
        """
        Attach metadata to a transaction.

        Args:
            transactionId: Transaction ID
            metadata: Metadata to attach

        Returns:
            Updated transaction

        Raises:
            NotFoundException: If transaction not found
            ValidationException: If validation fails
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
        """
        try:
            logger.info(f"Attaching metadata to transaction: {transactionId}")

            # Get existing transaction
            transaction = await self.getTransaction(transactionId)

            # Add metadata to transaction
            if transaction.metadata is None:
                transaction.metadata = []

            transaction.metadata.append(metadata)

            # Update transaction
            edit = TransactionEdit(
                transactionId=transactionId, metadata=transaction.metadata
            )

            updated_transactions = await self.updateTransactions([edit])
            return updated_transactions[0]

        except (NotFoundException, ValidationException):
            raise
        except Exception as e:
            logger.error(
                f"Error attaching metadata to transaction {transactionId}: {e}"
            )
            raise InternalException(f"Failed to attach metadata: {str(e)}") from e

    async def detachTransactionMetadata(
        self, transactionId: str, metadataId: str
    ) -> Transaction:
        """
        Detach metadata from a transaction.

        Args:
            transactionId: Transaction ID
            metadataId: Metadata ID to detach

        Returns:
            Updated transaction

        Raises:
            NotFoundException: If transaction or metadata not found
            ValidationException: If validation fails
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
        """
        try:
            logger.info(
                f"Detaching metadata {metadataId} from transaction: {transactionId}"
            )

            # Get existing transaction
            transaction = await self.getTransaction(transactionId)

            if not transaction.metadata:
                raise NotFoundException(
                    f"No metadata found for transaction {transactionId}"
                )

            # Remove metadata with matching ID
            original_count = len(transaction.metadata)
            transaction.metadata = [
                m for m in transaction.metadata if m.id != metadataId
            ]

            if len(transaction.metadata) == original_count:
                raise NotFoundException(
                    f"Metadata {metadataId} not found in transaction {transactionId}"
                )

            await self.database_store_access.deleteEntities(
                EntityType.Metadata, [metadataId]
            )

            # Update transaction
            edit = TransactionEdit(
                transactionId=transactionId, metadata=transaction.metadata
            )

            updated_transactions = await self.updateTransactions([edit])
            return updated_transactions[0]

        except (NotFoundException, ValidationException):
            raise
        except Exception as e:
            logger.error(
                f"Error detaching metadata from transaction {transactionId}: {e}"
            )
            raise InternalException(f"Failed to detach metadata: {str(e)}") from e

    async def findTransactionMetadata(
        self, transactionId: str, customSearchQuery: str | None = None
    ) -> list[Metadata]:
        """
        Find metadata candidates for a transaction using MetadataFindingEngine.

        Args:
            transactionId: Transaction ID to find metadata for
            customSearchQuery: Optional custom search query to override automatic term extraction

        Returns:
            List of metadata candidates

        Raises:
            NotFoundException: If transaction not found
            ValidationException: If validation fails
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
        """
        try:
            logger.info(f"Finding metadata for transaction: {transactionId}")

            # Get the transaction
            transaction = await self.getTransaction(transactionId)

            transactions = Transactions(transactions=[transaction])

            # Create filter with custom search query if provided
            metadata_filter = None
            if customSearchQuery:
                metadata_filter = MetadataFilter(searchPhrase=customSearchQuery)
                logger.info(f"Using custom search query: {customSearchQuery}")

            candidates = await self.metadata_finding_engine.getMetadataCandidates(
                transactions, metadata_filter
            )

            # Extract metadata from candidates
            metadata_list = []
            for candidate in candidates:
                if candidate.metadata:
                    metadata_list.append(candidate.metadata)

            logger.info(
                f"Found {len(metadata_list)} metadata candidates for transaction {transactionId}"
            )
            return metadata_list

        except (NotFoundException, ValidationException):
            raise
        except Exception as e:
            logger.error(f"Error finding metadata for transaction {transactionId}: {e}")
            raise InternalException(f"Failed to find metadata: {str(e)}") from e

    async def getPendingTransactions(self) -> list[Transaction]:
        """
        Get all pending (unapproved) transactions.

        Returns:
            List of pending transactions

        Raises:
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
        """
        try:
            logger.info("Getting pending transactions")

            # Query for unapproved transactions

            filter_value = FilterValue(boolValue=False)
            approved_filter = Filter(
                fieldName=FieldNames.APPROVED,
                operator=FilterOperator.EQ,
                value=filter_value,
            )

            query = Query(entityType=EntityType.Transaction, filters=[approved_filter])

            result = await self.database_store_access.getEntities(query)

            transactions = []
            for entity in result.entities:
                if entity.transaction:
                    transactions.append(entity.transaction)

            return transactions

        except Exception as e:
            logger.error(f"Error getting pending transactions: {e}")
            raise InternalException(
                f"Failed to get pending transactions: {str(e)}"
            ) from e

    async def getAllTransactions(self) -> list[Transaction]:
        """
        Get all transactions.

        Returns:
            List of all transactions

        Raises:
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
        """
        try:
            logger.info("Getting all transactions")

            # Query for all transactions
            query = Query(entityType=EntityType.Transaction)
            result = await self.database_store_access.getEntities(query)

            transactions = []
            for entity in result.entities:
                if entity.transaction:
                    transactions.append(entity.transaction)

            return transactions

        except Exception as e:
            logger.error(f"Error getting all transactions: {e}")
            raise InternalException(f"Failed to get all transactions: {str(e)}") from e

    async def _syncReferenceData(self, platform: BudgetingPlatformType) -> None:
        """
        Sync reference data (budgets, accounts, categories, payees) to ensure foreign key constraints.

        Args:
            platform: Budgeting platform type

        Raises:
            InternalException: If sync fails
            UnauthorizedException: If not authorized
            RemoteServiceException: If remote service error occurs
        """
        logger.info("Syncing reference data to ensure foreign key constraints...")

        try:
            # Sync budgets first
            logger.info("Syncing budgets...")
            budgets = await self.budgeting_platform_access.getBudgets(platform)
            if budgets:
                budget_entities = [Entity(budget=budget) for budget in budgets]
                await self.database_store_access.upsertEntities(budget_entities)
                logger.info(f"Synced {len(budgets)} budgets")

            # Sync accounts
            logger.info("Syncing accounts...")
            accounts = await self.budgeting_platform_access.getAccounts(platform)
            if accounts:
                account_entities = [Entity(account=account) for account in accounts]
                await self.database_store_access.upsertEntities(account_entities)
                logger.info(f"Synced {len(accounts)} accounts")

            # Sync categories
            logger.info("Syncing categories...")
            categories = await self.budgeting_platform_access.getCategories(platform)
            if categories:
                category_entities = [
                    Entity(category=category) for category in categories
                ]
                await self.database_store_access.upsertEntities(category_entities)
                logger.info(f"Synced {len(categories)} categories")

                # Log category IDs for debugging
                category_ids = [cat.id for cat in categories]
                logger.info(
                    f"Category IDs synced: {category_ids[:10]}{'...' if len(category_ids) > 10 else ''}"
                )

            # Sync payees
            logger.info("Syncing payees...")
            payees = await self.budgeting_platform_access.getPayees(platform)
            if payees:
                payee_entities = [Entity(payee=payee) for payee in payees]
                await self.database_store_access.upsertEntities(payee_entities)
                logger.info(f"Synced {len(payees)} payees")

            logger.info("Reference data sync completed successfully")

        except Exception as e:
            logger.error(f"Error syncing reference data: {e}")
            raise InternalException(f"Failed to sync reference data: {str(e)}") from e


    async def syncTransactionsIn(
        self, budgetPlatform: BudgetingPlatformType | None = None
    ) -> SyncResult:
        """
        Sync transactions from external budgeting platform.

        Args:
            budgetPlatform: Optional platform type (defaults to YNAB)

        Returns:
            Sync result with status information

        Raises:
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
            RemoteServiceException: If remote service error occurs
        """
        try:
            platform = (
                budgetPlatform or await self.config_service.getDefaultBudgetPlatform()
            )
            logger.info(f"Syncing transactions in from {platform}")

            # First, sync reference data to ensure foreign key constraints are satisfied
            await self._syncReferenceData(platform)

            # Now sync transactions
            logger.info("Syncing transactions...")
            transactions = await self.budgeting_platform_access.getTransactions(
                platform
            )
            logger.info(f"Retrieved {len(transactions)} transactions from platform")

            # Get valid category IDs to validate transactions (refresh after split handling)
            category_query = Query(entityType=EntityType.Category)
            category_result = await self.database_store_access.getEntities(
                category_query
            )
            valid_category_ids = {
                entity.category.id
                for entity in category_result.entities
                if entity.category
            }
            logger.info(
                f"Found {len(valid_category_ids)} valid category IDs in database"
            )

            # Clean up transactions with invalid category IDs
            cleaned_transactions = []
            invalid_category_count = 0

            for transaction in transactions:
                if (
                    transaction.categoryId
                    and transaction.categoryId not in valid_category_ids
                ):
                    logger.warning(
                        f"Transaction {transaction.id} has invalid category ID: {transaction.categoryId}, setting to None"
                    )
                    transaction.categoryId = None
                    invalid_category_count += 1
                cleaned_transactions.append(transaction)

            if invalid_category_count > 0:
                logger.warning(
                    f"Found {invalid_category_count} transactions with invalid category IDs, set to None"
                )

            # Store transactions locally
            entities = []
            for transaction in cleaned_transactions:
                entities.append(Entity(transaction=transaction))

            stored_entities = await self.database_store_access.upsertEntities(entities)

            # Create sync result
            results = []
            for i, entity in enumerate(stored_entities):
                if entity.transaction:
                    results.append(
                        SyncResultItem(
                            transactionId=entity.transaction.id,
                            status=SyncStatus.Success,
                            errorMessage=None,
                        )
                    )
                else:
                    results.append(
                        SyncResultItem(
                            transactionId=cleaned_transactions[i].id
                            if i < len(cleaned_transactions)
                            else "unknown",
                            status=SyncStatus.Fail,
                            errorMessage="Failed to store transaction",
                        )
                    )

            batch_status = (
                SyncStatus.Success
                if all(r.status == SyncStatus.Success for r in results)
                else SyncStatus.Partial
            )

            logger.info(
                f"Transaction sync completed: {len(results)} transactions processed"
            )
            return SyncResult(results=results, batchStatus=batch_status)

        except Exception as e:
            logger.error(f"Error syncing transactions in: {e}")
            raise InternalException(f"Failed to sync transactions: {str(e)}") from e

    async def syncTransactionsOut(
        self, budgetPlatform: BudgetingPlatformType | None = None
    ) -> SyncResult:
        """
        Sync transactions to external budgeting platform.

        Args:
            budgetPlatform: Optional platform type (defaults to YNAB)

        Returns:
            Sync result with status information

        Raises:
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
            RemoteServiceException: If remote service error occurs
        """
        try:
            platform = (
                budgetPlatform or await self.config_service.getDefaultBudgetPlatform()
            )
            logger.info(f"Syncing transactions out to {platform}")

            # Get local transactions that need syncing
            transactions = await self.getAllTransactions()

            # Update transactions on budgeting platform
            success = await self.budgeting_platform_access.updateTransactions(
                transactions, platform
            )

            # Create sync result
            results = []
            status = SyncStatus.Success if success else SyncStatus.Fail

            for transaction in transactions:
                results.append(
                    SyncResultItem(
                        transactionId=transaction.id,
                        status=status,
                        errorMessage=None
                        if success
                        else "Failed to update on platform",
                    )
                )

            return SyncResult(results=results, batchStatus=status)

        except Exception as e:
            logger.error(f"Error syncing transactions out: {e}")
            raise InternalException(f"Failed to sync transactions: {str(e)}") from e

    async def _syncUpdatedTransactions(self, transactions: list[Transaction]) -> None:
        """
        Sync updated transactions to YNAB using existing sync-out logic.

        This private method triggers automatic sync to YNAB after successful local updates.
        Errors in sync operations do not fail the local update - they are logged as warnings.

        Args:
            transactions: List of updated transactions to sync

        Note:
            This method does not raise exceptions for sync failures to ensure
            local updates are not rolled back due to external sync issues.
        """
        try:
            if not transactions:
                logger.info("No transactions to sync")
                return

            logger.info(f"Syncing {len(transactions)} updated transactions to YNAB")

            # Get default platform for sync
            platform = await self.config_service.getDefaultBudgetPlatform()

            # Use existing BudgetingPlatformAccess for sync
            success = await self.budgeting_platform_access.updateTransactions(
                transactions, platform
            )

            if success:
                logger.info(
                    f"Successfully synced {len(transactions)} transactions to YNAB"
                )
            else:
                logger.warning(
                    f"YNAB sync failed for {len(transactions)} transactions (local update succeeded)"
                )

        except Exception as e:
            # Log sync errors but don't propagate them - local update was successful
            logger.warning(
                f"YNAB sync error for {len(transactions)} transactions (local update succeeded): {e}"
            )
