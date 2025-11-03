"""
MLEngine implementation.

This Engine performs atomic ML business activities including model training,
prediction generation, and model management operations using a strategy pattern.
"""

import asyncio
import csv
import json
import logging
import os
import pickle
import shutil
import uuid
from datetime import datetime

from configs import FieldNames, get_config_service
from thrift_gen.databasestoreaccess.ttypes import (
    Filter,
    FilterOperator,
    FilterValue,
    Query,
)
from thrift_gen.entities.ttypes import (
    Entity,
    EntityType,
    ModelCard,
    ModelType,
    TrainingStatus,
)
from thrift_gen.exceptions.ttypes import (
    InternalException,
    NotFoundException,
    ValidationException,
)
from thrift_gen.mlengine.ttypes import (
    DatasetInfo,
    DatasetPreparationRequest,
    DatasetPreparationResult,
    ModelPredictionBatchRequest,
    ModelPredictionResult,
    ModelTrainingRequest,
    ModelTrainingResult,
    PredictionBatchResult,
)

# Import base strategy and implementations
from .ml_strategy_base import MLModelStrategy
from .pxblendsc_strategy import PXBlendSCStrategy

logger = logging.getLogger(__name__)


class MLEngine:
    """
    MLEngine implements atomic ML business activities using strategy pattern.

    This Engine handles model training, prediction generation, and model
    lifecycle management as reusable atomic operations.

    Public methods correspond exactly to the MLEngine Thrift interface.
    All other methods are private (prefixed with _).
    """

    def __init__(self, database_store_access=None, model_storage_path="ml_models"):
        """
        Initialize MLEngine.

        Args:
            database_store_access: DatabaseStoreAccess instance for metadata storage
            model_storage_path: Path to store trained models
        """
        self.database_store_access = database_store_access
        self.model_storage_path = model_storage_path
        self._loaded_model = None
        self._loaded_model_name = None

        # Initialize config service with database access
        if database_store_access:
            get_config_service(database_store_access)

        # Strategy registry
        self._strategies = {ModelType.PXBlendSC: PXBlendSCStrategy()}

        # Ensure model storage directory exists
        os.makedirs(model_storage_path, exist_ok=True)

        logger.info(f"MLEngine initialized with storage path: {model_storage_path}")

        # Clean up any orphaned training models on startup
        if database_store_access:
            asyncio.create_task(self._cleanup_orphaned_training_models())

    async def getModels(self, modelType: ModelType | None = None) -> list[ModelCard]:
        """
        Get available models.

        Args:
            modelType: Optional filter by model type

        Returns:
            List of model information

        Raises:
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
            NotFoundException: If no models found
        """
        try:
            logger.info(f"Getting models (type filter: {modelType})")

            # Query models from local storage

            filters = []
            if modelType is not None:
                type_filter = Filter(
                    fieldName=FieldNames.MODEL_TYPE,
                    operator=FilterOperator.EQ,
                    value=FilterValue(intValue=modelType),
                )
                filters.append(type_filter)

            query = Query(
                entityType=EntityType.ModelCard, filters=filters if filters else None
            )

            result = await self.database_store_access.getEntities(query)

            models = []
            for entity in result.entities:
                if entity.modelCard:
                    models.append(entity.modelCard)

            return models

        except Exception as e:
            logger.error(f"Error getting models: {e}")
            raise InternalException(f"Failed to get models: {str(e)}") from e

    async def deleteModels(self, modelCards: list[ModelCard]) -> bool:
        """
        Delete models.

        Args:
            modelCards: List of models to delete

        Returns:
            True if all deletions successful

        Raises:
            NotFoundException: If model not found
            ConflictException: If model cannot be deleted
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
        """
        try:
            logger.info(f"Deleting {len(modelCards)} models")

            success = True

            for model_info in modelCards:
                try:
                    # Delete model file
                    model_path = os.path.join(
                        self.model_storage_path, f"{model_info.name}.pkl"
                    )
                    if os.path.exists(model_path):
                        # Load model to check if it's PXBlendSC-RF and has bundle directory
                        try:
                            with open(model_path, "rb") as f:
                                model_data = pickle.load(f)

                            # Delete PXBlendSC-RF bundle directory if it exists
                            if (
                                model_data.get("strategy")
                                == ModelType._VALUES_TO_NAMES.get(
                                    ModelType.PXBlendSC
                                ).lower()
                                and "bundle_path" in model_data
                                and os.path.exists(model_data["bundle_path"])
                            ):
                                shutil.rmtree(
                                    model_data["bundle_path"], ignore_errors=True
                                )
                        except Exception as e:
                            logger.warning(
                                f"Could not clean up PXBlendSC-RF-RF files for {model_info.name}: {e}"
                            )

                        os.remove(model_path)

                    # Delete from local storage
                    await self.database_store_access.deleteEntities(
                        EntityType.ModelCard, [model_info.name]
                    )

                except Exception as e:
                    logger.error(f"Error deleting model {model_info.name}: {e}")
                    success = False

            return success

        except Exception as e:
            logger.error(f"Error deleting models: {e}")
            raise InternalException(f"Failed to delete models: {str(e)}") from e

    async def prepareDatasets(
        self, request: DatasetPreparationRequest
    ) -> DatasetPreparationResult:
        """
        Prepare datasets by splitting transactions into training and test sets.

        Args:
            request: Dataset preparation request with transactions and split parameters

        Returns:
            Dataset preparation result with file locations and statistics

        Raises:
            ValidationException: If validation fails
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
        """
        try:
            logger.info("Preparing datasets for ML training")

            # Validate request
            if not request.transactions or not request.transactions.transactions:
                raise ValidationException(
                    "No transactions provided for dataset preparation"
                )

            transactions = request.transactions.transactions
            test_split_ratio = request.testSplitRatio or 0.2
            min_samples_per_category = request.minSamplesPerCategory or 1

            # Filter transactions that have categories
            categorized_transactions = [
                t for t in transactions if t.categoryId and t.categoryId.strip()
            ]

            if not categorized_transactions:
                raise ValidationException(
                    "No categorized transactions found for dataset preparation"
                )

            # Group by category for stratified split
            category_groups = {}
            for transaction in categorized_transactions:
                category = transaction.categoryId
                if category not in category_groups:
                    category_groups[category] = []
                category_groups[category].append(transaction)

            # Get configuration to determine splitting strategy
            config_service = get_config_service()
            use_adaptive_splitting = True  # Default to adaptive splitting

            if config_service:
                try:
                    cfg = await config_service.getPXBlendSCConfig()
                    use_adaptive_splitting = cfg.get("models", {}).get(
                        "adaptive_splitting", True
                    )
                    logger.debug(f"Using adaptive splitting: {use_adaptive_splitting}")
                except Exception as e:
                    logger.warning(
                        f"Could not get adaptive splitting config: {e}, using default (True)"
                    )

            # Perform stratified split - adaptive or standard based on configuration
            training_data = []
            test_data = []

            def adaptive_split_count(
                category_size: int, base_ratio: float = 0.2
            ) -> int:
                """Calculate test count using adaptive strategy for better small class learning."""
                if not use_adaptive_splitting:
                    # Use standard splitting
                    return max(1, int(category_size * base_ratio))

                # Use adaptive splitting strategy
                if category_size <= 10:
                    # Very small classes: use only 10% for test, 90% for training
                    test_count = max(1, int(category_size * 0.1))
                elif category_size <= 50:
                    # Small classes: use 15% for test, 85% for training
                    test_count = max(2, int(category_size * 0.15))
                else:
                    # Large classes: use standard 20% split
                    test_count = int(category_size * base_ratio)

                return min(test_count, category_size)

            for category, transactions_in_category in category_groups.items():
                if len(transactions_in_category) < min_samples_per_category:
                    logger.warning(
                        f"Category {category} has only {len(transactions_in_category)} samples, skipping"
                    )
                    continue

                # Calculate split to maximize training data for small classes (if enabled)
                test_count = adaptive_split_count(
                    len(transactions_in_category), test_split_ratio
                )
                train_count = len(transactions_in_category) - test_count

                split_strategy = "adaptive" if use_adaptive_splitting else "standard"
                logger.debug(
                    f"Category {category} ({split_strategy}): {len(transactions_in_category)} samples → "
                    f"{train_count} train ({train_count / len(transactions_in_category):.1%}) / "
                    f"{test_count} test ({test_count / len(transactions_in_category):.1%})"
                )

                # Split the data
                test_data.extend(transactions_in_category[:test_count])
                training_data.extend(transactions_in_category[test_count:])

            if not training_data or not test_data:
                raise ValidationException("Insufficient data for train/test split")

            # Check for recent duplicate datasets to prevent rapid duplicate creation
            training_count = len(training_data)
            test_count = len(test_data)

            try:
                existing_datasets = await self.getDatasets()
                current_time = datetime.utcnow()

                for existing in existing_datasets:
                    # Check if there's a recent dataset with same sample counts
                    if (
                        existing.trainingSamples == training_count
                        and existing.testSamples == test_count
                    ):
                        # Check if dataset was created recently (within last 30 seconds)
                        try:
                            dataset_dir = os.path.dirname(existing.trainingDataLocation)
                            metadata_file = os.path.join(dataset_dir, "metadata.json")
                            if os.path.exists(metadata_file):
                                with open(metadata_file, encoding="utf-8") as f:
                                    metadata = json.load(f)
                                    created_at_str = metadata.get("created_at")
                                    if created_at_str:
                                        created_at = datetime.fromisoformat(
                                            created_at_str.replace("Z", "+00:00")
                                        )
                                        time_diff = (
                                            current_time - created_at
                                        ).total_seconds()
                                        if time_diff < 30:  # Within 30 seconds
                                            logger.info(
                                                f"Found recent duplicate dataset {existing.datasetId}, returning existing instead of creating new"
                                            )
                                            return DatasetPreparationResult(
                                                datasetInfo=existing
                                            )
                        except Exception as e:
                            logger.warning(
                                f"Could not check creation time for dataset {existing.datasetId}: {e}"
                            )
                            continue

            except Exception as e:
                logger.warning(f"Could not check for duplicate datasets: {e}")
                # Continue with creation if check fails

            # Create dataset files
            dataset_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            dataset_name = (
                request.datasetName or f"dataset_{timestamp}_{dataset_id[:8]}"
            )

            # Create dataset directory
            dataset_dir = f"ml_datasets/{dataset_name}"
            os.makedirs(dataset_dir, exist_ok=True)

            training_file = f"{dataset_dir}/training_data.csv"
            test_file = f"{dataset_dir}/test_data.csv"

            # Collect unique IDs for name lookup
            all_transactions = training_data + test_data
            category_ids = {t.categoryId for t in all_transactions if t.categoryId}
            payee_ids = {t.payeeId for t in all_transactions if t.payeeId}
            account_ids = {t.accountId for t in all_transactions if t.accountId}

            # Fetch category and payee names
            category_name_lookup = {}
            if category_ids:
                category_entities = await self.database_store_access.getEntitiesById(
                    EntityType.Category, list(category_ids)
                )
                for entity in category_entities:
                    if entity.category:
                        category_name_lookup[entity.category.id] = entity.category.name

            payee_name_lookup = {}
            if payee_ids:
                payee_entities = await self.database_store_access.getEntitiesById(
                    EntityType.Payee, list(payee_ids)
                )
                for entity in payee_entities:
                    if entity.payee:
                        payee_name_lookup[entity.payee.id] = entity.payee.name

            account_name_lookup = {}
            if account_ids:
                account_entities = await self.database_store_access.getEntitiesById(
                    EntityType.Account, list(account_ids)
                )
                for entity in account_entities:
                    if entity.account:
                        account_name_lookup[entity.account.id] = entity.account

            # Write training data CSV
            with open(training_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "date",
                        "payee_name",
                        "memo",
                        "amount",
                        "account_name",
                        "category_name",
                    ]
                )
                for transaction in training_data:
                    writer.writerow(
                        [
                            transaction.date or "",
                            payee_name_lookup.get(
                                transaction.payeeId, transaction.payeeId or ""
                            ),
                            transaction.memo or "",
                            transaction.amount or 0,
                            account_name_lookup.get(
                                transaction.accountId, transaction.accountId or ""
                            ),
                            category_name_lookup.get(
                                transaction.categoryId, transaction.categoryId or ""
                            ),
                        ]
                    )

            # Write test data CSV
            with open(test_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "date",
                        "payee_name",
                        "memo",
                        "amount",
                        "account_name",
                        "category_name",
                    ]
                )
                for transaction in test_data:
                    writer.writerow(
                        [
                            transaction.date or "",
                            payee_name_lookup.get(
                                transaction.payeeId, transaction.payeeId or ""
                            ),
                            transaction.memo or "",
                            transaction.amount or 0,
                            account_name_lookup.get(
                                transaction.accountId, transaction.accountId or ""
                            ),
                            category_name_lookup.get(
                                transaction.categoryId, transaction.categoryId or ""
                            ),
                        ]
                    )

            # Calculate statistics
            category_stats = {}
            for category, transactions_in_category in category_groups.items():
                if len(transactions_in_category) >= min_samples_per_category:
                    training_count = len(
                        [t for t in training_data if t.categoryId == category]
                    )
                    test_count = len([t for t in test_data if t.categoryId == category])
                    category_stats[category] = training_count + test_count

            # Calculate date range
            dates = [t.date for t in categorized_transactions if t.date]
            date_from = min(dates) if dates else None
            date_to = max(dates) if dates else None

            # Create dataset info
            dataset_info = DatasetInfo(
                datasetId=dataset_id,
                datasetName=dataset_name,
                trainingDataLocation=training_file,
                testDataLocation=test_file,
                trainingSamples=len(training_data),
                testSamples=len(test_data),
                categories=len(category_stats),
                categoryBreakdown=category_stats,
                dateFrom=date_from,
                dateTo=date_to,
            )

            # Save metadata.json file for API compatibility
            metadata = {
                "id": dataset_id,
                "dataset_name": dataset_name,
                "training_file": training_file,
                "test_file": test_file,
                "training_samples": len(training_data),
                "test_samples": len(test_data),
                "categories": len(category_stats),
                "category_breakdown": category_stats,
                "date_from": date_from,
                "date_to": date_to,
                "created_at": datetime.utcnow().isoformat(),
                "budget_id": request.budgetId if hasattr(request, "budgetId") else None,
            }

            metadata_file = f"{dataset_dir}/metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            result = DatasetPreparationResult(datasetInfo=dataset_info)

            logger.info(
                f"Dataset prepared: {len(training_data)} training, {len(test_data)} test samples, "
                f"{len(category_stats)} categories"
            )

            return result

        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Error preparing datasets: {e}")
            raise InternalException(f"Failed to prepare datasets: {str(e)}") from e

    async def trainModels(
        self,
        trainingRequests: list[ModelTrainingRequest],
    ) -> list[ModelTrainingResult]:
        """
        Train models.

        Args:
            trainingRequests: List of training requests

        Returns:
            List of training results

        Raises:
            ValidationException: If validation fails
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
        """
        try:
            logger.info(f"Training {len(trainingRequests)} models")

            results = []

            for request in trainingRequests:
                try:
                    result = await self._train_single_model(request)
                    results.append(result)
                except asyncio.CancelledError:
                    # Re-raise cancellation to propagate up
                    raise
                except Exception as e:
                    logger.error(f"Error training model {request.modelCard.name}: {e}")
                    results.append(
                        ModelTrainingResult(
                            modelCard=request.modelCard,
                            status=TrainingStatus.Fail,
                            errorMessage=str(e),
                        )
                    )

            return results

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise InternalException(f"Failed to train models: {str(e)}") from e

    async def getPredictions(
        self, predictionRequests: list[ModelPredictionBatchRequest]
    ) -> list[ModelPredictionResult]:
        """
        Get predictions for batch requests.

        Args:
            predictionRequests: List of prediction requests

        Returns:
            List of prediction results

        Raises:
            ValidationException: If validation fails
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
            RemoteServiceException: If remote service error occurs
        """
        try:
            logger.info(f"Getting predictions for {len(predictionRequests)} requests")

            results = []

            for request in predictionRequests:
                try:
                    result = await self._predict_single_batch(request)
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Error predicting for model {request.modelCard.name}: {e}"
                    )
                    results.append(
                        ModelPredictionResult(
                            modelCard=request.modelCard,
                            result=PredictionBatchResult(
                                categoricalPredictionResults={}
                            ),
                            errorMessage=str(e),
                        )
                    )

            return results

        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            raise InternalException(f"Failed to get predictions: {str(e)}") from e

    async def getDatasets(
        self, datasetIds: list[str] | None = None, budgetId: str | None = None
    ) -> list[DatasetInfo]:
        """
        Get datasets, optionally filtered by IDs or budget.

        Args:
            datasetIds: Optional list of specific dataset IDs to retrieve
            budgetId: Optional budget ID filter

        Returns:
            List of dataset information

        Raises:
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
            NotFoundException: If specific datasets not found
        """
        try:
            logger.info(f"Getting datasets (IDs: {datasetIds}, budget: {budgetId})")

            datasets = []
            datasets_dir = "ml_datasets"

            if not os.path.exists(datasets_dir):
                return datasets

            # If specific dataset IDs requested, look for those
            if datasetIds:
                found_ids = set()
                for dataset_name in os.listdir(datasets_dir):
                    dataset_path = os.path.join(datasets_dir, dataset_name)
                    metadata_file = os.path.join(dataset_path, "metadata.json")

                    if os.path.isdir(dataset_path) and os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, encoding="utf-8") as f:
                                metadata = json.load(f)

                            dataset_id = metadata.get("id")
                            if dataset_id in datasetIds:
                                # Filter by budget if specified
                                # Include datasets with budget_id=null as "global" datasets
                                if budgetId:
                                    dataset_budget_id = metadata.get("budget_id")
                                    if (
                                        dataset_budget_id is not None
                                        and dataset_budget_id != budgetId
                                    ):
                                        continue

                                dataset_info = self._metadata_to_dataset_info(metadata)
                                if dataset_info:
                                    datasets.append(dataset_info)
                                    found_ids.add(dataset_id)

                        except Exception as e:
                            logger.warning(
                                f"Error reading dataset metadata {dataset_name}: {e}"
                            )
                            continue

                # Check if all requested IDs were found
                missing_ids = set(datasetIds) - found_ids
                if missing_ids:
                    from thrift_gen.exceptions.ttypes import NotFoundException

                    raise NotFoundException(f"Datasets not found: {list(missing_ids)}")

            else:
                # List all datasets
                for dataset_name in os.listdir(datasets_dir):
                    dataset_path = os.path.join(datasets_dir, dataset_name)
                    metadata_file = os.path.join(dataset_path, "metadata.json")

                    if os.path.isdir(dataset_path) and os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, encoding="utf-8") as f:
                                metadata = json.load(f)

                            # Filter by budget if specified
                            # Include datasets with budget_id=null as "global" datasets
                            if budgetId:
                                dataset_budget_id = metadata.get("budget_id")
                                if (
                                    dataset_budget_id is not None
                                    and dataset_budget_id != budgetId
                                ):
                                    continue

                            dataset_info = self._metadata_to_dataset_info(metadata)
                            if dataset_info:
                                datasets.append(dataset_info)

                        except Exception as e:
                            logger.warning(
                                f"Error reading dataset metadata {dataset_name}: {e}"
                            )
                            continue

            # Sort by creation date (newest first)
            datasets.sort(key=lambda x: x.dateFrom or "", reverse=True)

            logger.info(f"Found {len(datasets)} datasets")
            return datasets

        except Exception as e:
            logger.error(f"Error getting datasets: {e}")
            raise InternalException(f"Failed to get datasets: {str(e)}") from e

    async def deleteDatasets(self, datasetIds: list[str]) -> list[bool]:
        """
        Delete datasets.

        Args:
            datasetIds: List of dataset IDs to delete

        Returns:
            List of boolean results indicating success for each dataset

        Raises:
            NotFoundException: If datasets not found
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
        """
        try:
            logger.info(f"Deleting {len(datasetIds)} datasets")

            results = []
            datasets_dir = "ml_datasets"

            if not os.path.exists(datasets_dir):
                # All datasets not found
                results = [False] * len(datasetIds)
                raise NotFoundException(f"Datasets not found: {datasetIds}")

            for dataset_id in datasetIds:
                deleted = False

                for dataset_name in os.listdir(datasets_dir):
                    dataset_path = os.path.join(datasets_dir, dataset_name)
                    metadata_file = os.path.join(dataset_path, "metadata.json")

                    if os.path.isdir(dataset_path) and os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, encoding="utf-8") as f:
                                import json

                                metadata = json.load(f)

                            if metadata.get("id") == dataset_id:
                                # Delete the entire dataset directory
                                shutil.rmtree(dataset_path)
                                deleted = True
                                logger.info(f"Deleted dataset: {dataset_id}")
                                break

                        except Exception as e:
                            logger.warning(
                                f"Error checking dataset {dataset_name}: {e}"
                            )
                            continue

                results.append(deleted)

            # Check if any datasets were not found
            if not any(results):
                raise NotFoundException(f"Datasets not found: {datasetIds}")

            return results

        except Exception as e:
            logger.error(f"Error deleting datasets: {e}")
            raise InternalException(f"Failed to delete datasets: {str(e)}") from e

    async def cancelTraining(self, modelCard: ModelCard) -> bool:
        """
        Cancel training for a model.

        Args:
            modelCard: Model card of the model to cancel training for

        Returns:
            True if cancellation successful

        Raises:
            NotFoundException: If model not found
            ConflictException: If model is not currently training
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
        """
        try:
            logger.info(f"Canceling training for model: {modelCard.name}")

            # Check if model files exist
            model_path = os.path.join(self.model_storage_path, f"{modelCard.name}.pkl")

            # For models currently training, we need to:
            # 1. Stop any ongoing training process (if possible)
            # 2. Clean up any partial model files
            # 3. Remove any training artifacts

            # Clean up model file if it exists
            if os.path.exists(model_path):
                try:
                    # Load model to check if it's PXBlendSC-RF and has bundle directory
                    with open(model_path, "rb") as f:
                        model_data = pickle.load(f)

                    # Delete PXBlendSC-RF bundle directory if it exists
                    if (
                        model_data.get("strategy")
                        == ModelType._VALUES_TO_NAMES.get(ModelType.PXBlendSC).lower()
                        and "bundle_path" in model_data
                        and os.path.exists(model_data["bundle_path"])
                    ):
                        shutil.rmtree(model_data["bundle_path"], ignore_errors=True)
                except Exception as e:
                    logger.warning(
                        f"Could not clean up PXBlendSC-RF files for {modelCard.name}: {e}"
                    )

                os.remove(model_path)

            logger.info(f"Successfully canceled training for model: {modelCard.name}")
            return True

        except Exception as e:
            logger.error(f"Error canceling training for model {modelCard.name}: {e}")
            raise InternalException(f"Failed to cancel training: {str(e)}") from e

    async def _train_single_model(
        self, request: ModelTrainingRequest
    ) -> ModelTrainingResult:
        """
        Train a single model using the appropriate strategy.

        Args:
            request: Training request

        Returns:
            Training result
        """
        try:
            model_name = request.modelCard.name
            model_type = request.modelCard.modelType
            logger.info(f"Training model: {model_name} (type: {model_type})")

            # Validate request
            if not model_name:
                raise ValidationException("Model name is required")

            if not request.trainingDataLocation:
                raise ValidationException("Training data location is required")

            # Get appropriate strategy
            strategy = self._strategies.get(model_type)
            if not strategy:
                raise ValidationException(
                    f"No strategy available for model type: {model_type}"
                )

            # Load training data
            training_data = await self._load_training_data(request.trainingDataLocation)

            if not training_data:
                raise ValidationException("No training data found")

            # Train model using strategy
            model = await strategy.train_model(request, training_data)

            # Save model to disk
            model_path = os.path.join(self.model_storage_path, f"{model_name}.pkl")

            # Construct test data path from training data location
            training_dir = os.path.dirname(request.trainingDataLocation)
            test_data_path = os.path.join(training_dir, "test_data.csv")

            if os.path.exists(test_data_path):
                logger.info(f"Evaluating model on test data: {test_data_path}")
                test_metrics = await strategy.evaluate_model(model, test_data_path)

                # Merge test metrics into model performance metrics
                if test_metrics and "performance_metrics" in model:
                    model["performance_metrics"].update(test_metrics)
                    logger.info(
                        f"Added test metrics to model: {list(test_metrics.keys())}"
                    )
            else:
                logger.info(f"No test data found at {test_data_path}")

            # For PXBlendSC-RF models, we need to handle the bundle path
            if (
                model.get("strategy")
                == ModelType._VALUES_TO_NAMES.get(ModelType.PXBlendSC).lower()
                and "bundle_path" in model
            ):
                # Move the temporary bundle directory to permanent storage
                temp_path = model["bundle_path"]
                permanent_path = os.path.join(
                    self.model_storage_path, f"{model_name}_bundle"
                )

                if os.path.exists(permanent_path):
                    shutil.rmtree(permanent_path)

                if os.path.exists(temp_path):
                    shutil.move(temp_path, permanent_path)
                    model["bundle_path"] = permanent_path
                else:
                    logger.error(f"Temporary bundle path not found: {temp_path}")
                    raise InternalException(f"Bundle path not found: {temp_path}")

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Create model info with actual performance metrics
            performance_metrics = model.get("performance_metrics", {})
            model_info = ModelCard(
                modelType=model_type,
                name=model_name,
                version=request.modelCard.version or "1.0",
                description=request.modelCard.description,
                status=TrainingStatus.Success,
                trainedDate=datetime.utcnow().isoformat(),
                performanceMetrics={
                    # Cross-validation metrics
                    "cv_macro_f1": str(performance_metrics.get("cv_macro_f1", 0.0)),
                    "cv_balanced_accuracy": str(
                        performance_metrics.get("cv_balanced_accuracy", 0.0)
                    ),
                    "cv_accuracy": str(performance_metrics.get("cv_accuracy", 0.0)),
                    "cv_abstain_rate": str(
                        performance_metrics.get("cv_abstain_rate", 0.0)
                    ),
                    # Training metrics (on full training dataset)
                    "train_macro_f1": str(
                        performance_metrics.get("train_macro_f1", 0.0)
                    ),
                    "train_balanced_accuracy": str(
                        performance_metrics.get("train_balanced_accuracy", 0.0)
                    ),
                    "train_accuracy": str(
                        performance_metrics.get("train_accuracy", 0.0)
                    ),
                    "train_abstain_rate": str(
                        performance_metrics.get("train_abstain_rate", 0.0)
                    ),
                    # Final retraining metrics (if available)
                    "final_macro_f1": str(
                        performance_metrics.get("final_macro_f1", 0.0)
                    ),
                    "final_balanced_accuracy": str(
                        performance_metrics.get("final_balanced_accuracy", 0.0)
                    ),
                    "final_accuracy": str(
                        performance_metrics.get("final_accuracy", 0.0)
                    ),
                    "final_abstain_rate": str(
                        performance_metrics.get("final_abstain_rate", 0.0)
                    ),
                    # Test set metrics (if available)
                    "test_macro_f1": str(performance_metrics.get("test_macro_f1", 0.0)),
                    "test_balanced_accuracy": str(
                        performance_metrics.get("test_balanced_accuracy", 0.0)
                    ),
                    "test_accuracy": str(performance_metrics.get("test_accuracy", 0.0)),
                    "test_abstain_rate": str(
                        performance_metrics.get("test_abstain_rate", 0.0)
                    ),
                    "test_rows_evaluated": str(
                        performance_metrics.get("test_rows_evaluated", 0)
                    ),
                    "test_rows_skipped_unseen": str(
                        performance_metrics.get("test_rows_skipped_unseen", 0)
                    ),
                    # Model metadata
                    "training_samples": str(model.get("training_samples", 0)),
                    "feature_count": str(len(model.get("feature_columns", []))),
                    "n_classes": str(model.get("n_classes", 0)),
                    "has_lgbm": str(model.get("has_lgbm", False)),
                    "has_svm": str(model.get("has_svm", False)),
                },
            )

            # Store model info
            entity = Entity(modelCard=model_info)
            await self.database_store_access.upsertEntities([entity])

            return ModelTrainingResult(
                modelCard=model_info, status=TrainingStatus.Success, errorMessage=None
            )

        except asyncio.CancelledError:
            # Training was canceled - clean up any partial files
            logger.info(f"Training canceled for model: {request.modelCard.name}")
            model_path = os.path.join(
                self.model_storage_path, f"{request.modelCard.name}.pkl"
            )
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    logger.info(f"Cleaned up partial model file: {model_path}")
                except Exception as e:
                    logger.warning(f"Could not clean up model file {model_path}: {e}")
            raise
        except (ValidationException, InternalException):
            raise
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise InternalException(f"Model training failed: {str(e)}") from e

    async def _predict_single_batch(
        self, request: ModelPredictionBatchRequest
    ) -> ModelPredictionResult:
        """
        Generate predictions for a single batch request using the appropriate strategy.

        Args:
            request: Prediction request

        Returns:
            Prediction result
        """
        try:
            model_name = request.modelCard.name
            model_type = request.modelCard.modelType
            logger.info(f"Predicting with model: {model_name} (type: {model_type})")

            # Get appropriate strategy
            strategy = self._strategies.get(model_type)
            if not strategy:
                raise ValidationException(
                    f"No strategy available for model type: {model_type}"
                )

            # Load model if not already loaded
            if self._loaded_model_name != model_name:
                await self._load_model(model_name)

            if not self._loaded_model:
                raise InternalException(f"Model {model_name} not available")

            # Extract input data
            if not request.input or not request.input.primitiveBatchInput:
                raise ValidationException("Primitive batch input is required")

            batch_input = request.input.primitiveBatchInput

            # Generate predictions for each row using strategy
            predictions = {}

            for row_id, row_data in batch_input.items():
                try:
                    category_predictions = await strategy.predict(
                        self._loaded_model, [row_data]
                    )
                    predictions[row_id] = category_predictions

                except Exception as e:
                    logger.warning(f"Error predicting for row {row_id}: {e}")
                    predictions[row_id] = []

            result = PredictionBatchResult(categoricalPredictionResults=predictions)

            return ModelPredictionResult(
                modelCard=request.modelCard, result=result, errorMessage=None
            )

        except (ValidationException, InternalException):
            raise
        except Exception as e:
            logger.error(f"Error predicting batch: {e}")
            raise InternalException(f"Batch prediction failed: {str(e)}") from e

    async def _load_training_data(self, data_location: str) -> list[list]:
        """
        Load training data from location.

        Args:
            data_location: Path or identifier for training data

        Returns:
            Training data as list of rows
        """
        try:
            # Check if it's a file path
            if os.path.exists(data_location):
                training_data = []
                with open(data_location, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    # Skip header if present
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 5:  # Ensure we have all required columns
                            training_data.append(row)
                return training_data
            else:
                logger.warning(f"Training data file not found: {data_location}")
                return []

        except Exception as e:
            logger.error(f"Error loading training data from {data_location}: {e}")
            return []

    async def _load_model(self, model_name: str) -> bool:
        """
        Load a model into memory.

        Args:
            model_name: Name of model to load

        Returns:
            True if loaded successfully
        """
        try:
            model_path = os.path.join(self.model_storage_path, f"{model_name}.pkl")

            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                return False

            with open(model_path, "rb") as f:
                self._loaded_model = pickle.load(f)
                self._loaded_model_name = model_name

            logger.info(f"Loaded model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False

    def _get_strategy(self, model_type: ModelType) -> MLModelStrategy:
        """Get the appropriate strategy for a model type."""
        strategy = self._strategies.get(model_type)
        if not strategy:
            raise ValidationException(
                f"No strategy available for model type: {model_type}"
            )
        return strategy

    def _metadata_to_dataset_info(self, metadata: dict) -> DatasetInfo | None:
        """
        Convert metadata dictionary to DatasetInfo object.

        Args:
            metadata: Dataset metadata dictionary

        Returns:
            DatasetInfo object or None if conversion fails
        """
        try:
            # Check if files still exist
            training_file = metadata.get("training_file", "")
            test_file = metadata.get("test_file", "")
            training_exists = os.path.exists(training_file) if training_file else False
            test_exists = os.path.exists(test_file) if test_file else False

            if not (training_exists and test_exists):
                logger.warning(f"Dataset files missing for {metadata.get('id')}")
                return None

            return DatasetInfo(
                datasetId=metadata.get("id", ""),
                datasetName=metadata.get("dataset_name", ""),
                trainingDataLocation=training_file,
                testDataLocation=test_file,
                trainingSamples=metadata.get("training_samples", 0),
                testSamples=metadata.get("test_samples", 0),
                categories=metadata.get("categories", 0),
                categoryBreakdown=metadata.get("category_breakdown", {}),
                dateFrom=metadata.get("date_from"),
                dateTo=metadata.get("date_to"),
            )

        except Exception as e:
            logger.warning(f"Error converting metadata to DatasetInfo: {e}")
            return None

    async def _cleanup_orphaned_training_models(self):
        """
        Clean up models that are stuck in Pending status after container restart.

        This handles the case where training was interrupted by container restart,
        leaving models in Pending status with no active training process.
        """
        try:
            logger.info("Checking for orphaned training models...")

            # Get all models with Pending status
            pending_name = TrainingStatus._VALUES_TO_NAMES.get(TrainingStatus.Pending, "Pending")
            filters = [
                Filter(
                    fieldName=FieldNames.STATUS,
                    operator=FilterOperator.EQ,
                    value=FilterValue(stringValue=pending_name),
                )
            ]

            query = Query(entityType=EntityType.ModelCard, filters=filters)
            result = await self.database_store_access.getEntities(query)

            orphaned_models = []
            for entity in result.entities:
                if entity.modelCard:
                    orphaned_models.append(entity.modelCard)

            if orphaned_models:
                logger.warning(
                    f"Found {len(orphaned_models)} orphaned training models, marking as failed"
                )

                for model in orphaned_models:
                    # Update model status to Failed
                    updated_model = ModelCard(
                        modelType=model.modelType,
                        name=model.name,
                        version=model.version,
                        description=f"{model.description} (Training interrupted by restart)",
                        status=TrainingStatus.Fail,
                        trainedDate=model.trainedDate,
                        performanceMetrics=model.performanceMetrics,
                    )

                    # Store the updated model
                    entity = Entity(modelCard=updated_model)
                    await self.database_store_access.storeEntity(entity)

                    logger.info(f"Marked orphaned model '{model.name}' as failed")
            else:
                logger.info("No orphaned training models found")

        except Exception as e:
            logger.error(f"Error during orphaned model cleanup: {e}")
            # Don't raise - this is a background cleanup task
