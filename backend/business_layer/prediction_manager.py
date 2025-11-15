"""
PredictionManager implementation.

This Manager orchestrates ML prediction workflows and model management,
implementing the PredictionManager Thrift service interface.
"""

import asyncio
import csv
import logging
import os
from datetime import datetime

from configs import FieldNames
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
    PrimitiveValue,
    TrainingStatus,
    Transactions,
)
from thrift_gen.exceptions.ttypes import (
    ConflictException,
    InternalException,
    NotFoundException,
    RemoteServiceException,
    ValidationException,
)
from thrift_gen.mlengine.ttypes import (
    DatasetPreparationRequest,
    DatasetPreparationResult,
    ModelPredictionBatchRequest,
    ModelPredictionResult,
    ModelTrainingRequest,
    ModelTrainingResult,
    PredictionInput,
)
from thrift_gen.predictionmanager.ttypes import (
    PredictionRequest,
    PredictionResult,
    TrainingDataPreparationRequest,
)

logger = logging.getLogger(__name__)


class PredictionManager:
    """
    PredictionManager implements ML prediction workflow orchestration.

    This Manager coordinates between MLEngine and DatabaseStoreAccess
    to implement ML-related use cases.
    """

    def __init__(self, ml_engine=None, database_store_access=None, config_service=None):
        """
        Initialize PredictionManager with required services.

        Args:
            ml_engine: MLEngine instance
            database_store_access: DatabaseStoreAccess instance
            config_service: ConfigService instance
        """
        self.ml_engine = ml_engine
        self.database_store_access = database_store_access
        self.config_service = config_service

        # Task management for training operations
        self._current_training_task = None
        self._current_training_model = None

        logger.info("PredictionManager initialized")

    async def getModels(self, modelType: ModelType | None = None) -> list[ModelCard]:
        """
        Get available ML models with enhanced filtering and sorting.

        Args:
            modelType: Optional filter by model type

        Returns:
            List of model information sorted by version (newest first)

        Raises:
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
            NotFoundException: If no models found
        """
        try:
            logger.info(f"Getting models (type filter: {modelType})")

            # Get models from MLEngine
            models = await self.ml_engine.getModels(modelType)

            if not models:
                raise NotFoundException("No models found")

            # Enhanced sorting and filtering
            filtered_models = []

            for model in models:
                # Additional validation of model data
                if self._is_valid_model_card(model):
                    filtered_models.append(model)
                else:
                    logger.warning(f"Skipping invalid model: {model.name}")

            # Sort by version (newest first), then by name
            sorted_models = sorted(
                filtered_models,
                key=lambda m: (float(m.version or "0.0"), m.name or ""),
                reverse=True,
            )

            logger.info(f"Retrieved {len(sorted_models)} valid models")
            return sorted_models

        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            raise InternalException(f"Failed to get models: {str(e)}") from e

    async def prepareTrainingData(
        self, request: TrainingDataPreparationRequest
    ) -> DatasetPreparationResult:
        """
        Prepare training data by fetching transactions and delegating to MLEngine for splitting.

        Args:
            request: Training data preparation parameters

        Returns:
            Dataset preparation result with file locations and statistics

        Raises:
            ValidationException: If validation fails
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
            NotFoundException: If no data found
        """
        try:
            logger.info(
                f"Preparing training data (budget: {request.budgetId}, months: {request.monthsBack})"
            )

            # Build query filters for transactions
            filters = []

            # Filter by budget if specified
            if request.budgetId:
                budget_filter = Filter(
                    fieldName=FieldNames.BUDGET_ID,
                    operator=FilterOperator.EQ,
                    value=FilterValue(stringValue=request.budgetId),
                )
                filters.append(budget_filter)

            # Add date filtering if months_back is specified
            if request.monthsBack and request.monthsBack > 0:
                from datetime import datetime

                today = datetime.now().date()
                year = today.year
                month = today.month - request.monthsBack

                # Handle year rollover
                while month <= 0:
                    month += 12
                    year -= 1

                cutoff_date = datetime(year, month, 1).date()

                date_filter = Filter(
                    fieldName=FieldNames.DATE,
                    operator=FilterOperator.GTE,
                    value=FilterValue(stringValue=cutoff_date.isoformat()),
                )
                filters.append(date_filter)

            # Only approved transactions with categories
            approved_filter = Filter(
                fieldName=FieldNames.APPROVED,
                operator=FilterOperator.EQ,
                value=FilterValue(boolValue=True),
            )
            filters.append(approved_filter)

            # Query for transactions
            query = Query(entityType=EntityType.Transaction, filters=filters)
            result = await self.database_store_access.getEntities(query)

            if not result.entities:
                raise NotFoundException(
                    "No transactions found for training data preparation"
                )

            # Extract transactions and filter for categorized ones
            transactions = []
            for entity in result.entities:
                if entity.transaction and entity.transaction.categoryId:
                    transactions.append(entity.transaction)

            if not transactions:
                raise NotFoundException(
                    "No categorized transactions found for training"
                )

            # Create Transactions object for MLEngine
            transactions_obj = Transactions(transactions=transactions)

            # Create dataset preparation request for MLEngine
            dataset_request = DatasetPreparationRequest(
                transactions=transactions_obj,
                testSplitRatio=request.testSplitRatio or 0.2,
                minSamplesPerCategory=request.minSamplesPerCategory or 1,
            )

            # Delegate to MLEngine for dataset preparation and splitting
            result = await self.ml_engine.prepareDatasets(dataset_request)

            logger.info(
                f"Training data prepared: {result.datasetInfo.trainingSamples} training, "
                f"{result.datasetInfo.testSamples} test samples"
            )

            return result

        except (ValidationException, NotFoundException):
            raise
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise InternalException(f"Failed to prepare training data: {str(e)}") from e

    def _is_valid_model_card(self, model: ModelCard) -> bool:
        """
        Validate that a model card has required fields.

        Args:
            model: Model card to validate

        Returns:
            True if model card is valid
        """
        if not model:
            return False

        # Check required fields
        if not model.name or len(model.name.strip()) == 0:
            return False

        if not model.modelType:
            return False

        # Check version format
        if model.version:
            try:
                float(model.version)
            except ValueError:
                logger.warning(
                    f"Invalid version format for model {model.name}: {model.version}"
                )
                return False

        return True

    async def trainModel(self, params: ModelTrainingRequest) -> ModelTrainingResult:
        """
        Train a new ML model with comprehensive data preparation and validation.

        Args:
            params: Model training parameters

        Returns:
            Training result with model information

        Raises:
            ValidationException: If validation fails
            ConflictException: If training already in progress
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
        """
        try:
            logger.info(f"Training model: {params.modelCard.name}")

            # Check if training is already in progress
            if (
                self._current_training_task is not None
                and not self._current_training_task.done()
            ):
                raise ConflictException(
                    f"Training already in progress for model '{self._current_training_model}'. "
                    "Please wait for it to complete or cancel it before starting a new training."
                )

            # Enhanced validation of training parameters
            validation_errors = await self._validate_training_request(params)
            if validation_errors:
                raise ValidationException(
                    f"Training validation failed: {'; '.join(validation_errors)}"
                )

            if (
                not params.trainingDataLocation
                or not params.trainingDataLocation.strip()
            ):
                raise ValidationException("Training data location must be provided")

            logger.info(f"Using training data location: {params.trainingDataLocation}")
            training_data_location = params.trainingDataLocation

            # Update params with training data location
            enhanced_params = ModelTrainingRequest(
                modelCard=params.modelCard,
                trainingDataLocation=training_data_location,
                parameters=params.parameters,
            )

            # Check for model versioning - increment version if model exists
            enhanced_params.modelCard = await self._handle_model_versioning(
                params.modelCard
            )

            # Set current training model name
            self._current_training_model = enhanced_params.modelCard.name

            # CREATE MODEL RECORD UPFRONT with Pending status
            # This ensures the model appears in the UI immediately and persists across refreshes
            pending_model = ModelCard(
                modelType=enhanced_params.modelCard.modelType,
                name=enhanced_params.modelCard.name,
                version=enhanced_params.modelCard.version,
                description=enhanced_params.modelCard.description,
                status=TrainingStatus.Pending,  # Set to Pending initially
                trainedDate=datetime.utcnow().isoformat(),
                performanceMetrics=None,  # Will be updated after training
            )

            # Store the pending model in database
            pending_entity = Entity(modelCard=pending_model)
            await self.database_store_access.upsertEntities([pending_entity])
            logger.info(f"Created pending model record: {pending_model.name}")

            try:
                # Create training task and store reference for cancellation
                self._current_training_task = asyncio.create_task(
                    self.ml_engine.trainModels([enhanced_params]),
                    name=f"training_{enhanced_params.modelCard.name}",
                )

                # Start training via MLEngine
                training_results = await self._current_training_task

                if not training_results or len(training_results) == 0:
                    # Update model status to failed
                    failed_model = ModelCard(
                        modelType=pending_model.modelType,
                        name=pending_model.name,
                        version=pending_model.version,
                        description=pending_model.description,
                        status=TrainingStatus.Fail,
                        trainedDate=pending_model.trainedDate,
                        performanceMetrics=None,
                    )
                    failed_entity = Entity(modelCard=failed_model)
                    await self.database_store_access.upsertEntities([failed_entity])
                    raise InternalException("Training failed - no results returned")

                result = training_results[0]

                # Update model status based on training result
                if result.status == TrainingStatus.Success:
                    # Enhanced model storage with performance tracking
                    await self._store_trained_model_with_metrics(result)
                else:
                    # Update model status to failed
                    failed_model = ModelCard(
                        modelType=pending_model.modelType,
                        name=pending_model.name,
                        version=pending_model.version,
                        description=pending_model.description,
                        status=TrainingStatus.Fail,
                        trainedDate=pending_model.trainedDate,
                        performanceMetrics=None,
                    )
                    failed_entity = Entity(modelCard=failed_model)
                    await self.database_store_access.upsertEntities([failed_entity])

                return result

            except asyncio.CancelledError:
                # Training was canceled - clean up and delete model record
                logger.info(f"Training was canceled for model {pending_model.name}")
                await self.database_store_access.deleteEntities(
                    EntityType.ModelCard, [pending_model.name]
                )
                raise ConflictException(
                    f"Training was canceled for model {pending_model.name}"
                ) from None

            except Exception as training_error:
                # Update model status to failed if training throws an exception
                logger.error(
                    f"Training failed for model {pending_model.name}: {training_error}"
                )
                failed_model = ModelCard(
                    modelType=pending_model.modelType,
                    name=pending_model.name,
                    version=pending_model.version,
                    description=pending_model.description,
                    status=TrainingStatus.Fail,
                    trainedDate=pending_model.trainedDate,
                    performanceMetrics=None,
                )
                failed_entity = Entity(modelCard=failed_model)
                await self.database_store_access.upsertEntities([failed_entity])
                raise

            finally:
                # Clear the training task and model
                self._current_training_task = None
                self._current_training_model = None

        except (ValidationException, InternalException, ConflictException):
            raise
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise InternalException(f"Failed to train model: {str(e)}") from e

    async def getPredictions(self, request: PredictionRequest) -> PredictionResult:
        """
        Get predictions for transactions with enhanced validation and error handling.

        Args:
            request: Prediction request with transactions and optional model

        Returns:
            Prediction results

        Raises:
            ValidationException: If validation fails
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
            RemoteServiceException: If remote service error occurs
        """
        try:
            logger.info("Getting predictions for transactions")

            # Enhanced request validation
            validation_errors = await self._validate_prediction_request(request)
            if validation_errors:
                raise ValidationException(
                    f"Prediction validation failed: {'; '.join(validation_errors)}"
                )

            # Prepare prediction input with enhanced data processing
            primitive_batch = await self._prepare_prediction_input(request.transactions)

            if not primitive_batch:
                raise ValidationException("No valid transactions found for prediction")

            # Create prediction request for MLEngine
            prediction_input = PredictionInput(primitiveBatchInput=primitive_batch)

            # Select and validate model
            model_info = await self._select_prediction_model(request.modelCard)

            ml_request = ModelPredictionBatchRequest(
                modelCard=model_info, input=prediction_input
            )

            # Get predictions from MLEngine with retry logic
            prediction_results = await self._get_predictions_with_retry(ml_request)

            if not prediction_results or len(prediction_results) == 0:
                raise InternalException("Prediction failed - no results returned")

            ml_result = prediction_results[0]

            # Process and validate prediction results
            processed_results = await self._process_prediction_results(ml_result)

            return processed_results

        except (ValidationException, InternalException, RemoteServiceException):
            raise
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            raise InternalException(f"Failed to get predictions: {str(e)}") from e

    async def deleteModel(self, modelCard: ModelCard) -> bool:
        """
        Delete a trained model.

        Args:
            modelCard: Model information to delete

        Returns:
            True if deletion successful

        Raises:
            NotFoundException: If model not found
            ConflictException: If model cannot be deleted
            InternalException: If internal error occurs
            UnauthorizedException: If not authorized
        """
        try:
            logger.info(f"Deleting model: {modelCard.name}")

            # Delete from MLEngine
            success = await self.ml_engine.deleteModels([modelCard])

            if not success:
                raise InternalException("Failed to delete model from ML engine")

            # Delete from local storage

            # Find model in storage by name
            filter_value = FilterValue(stringValue=modelCard.name)
            name_filter = Filter(
                fieldName=FieldNames.NAME,
                operator=FilterOperator.EQ,
                value=filter_value,
            )

            query = Query(entityType=EntityType.ModelCard, filters=[name_filter])

            result = await self.database_store_access.getEntities(query)

            if result.entities:
                # Delete found models
                entity_ids = []
                for entity in result.entities:
                    if entity.modelCard:
                        entity_ids.append(entity.modelCard.name)  # Using name as ID

                if entity_ids:
                    await self.database_store_access.deleteEntities(
                        EntityType.ModelCard, entity_ids
                    )

            # Check if the deleted model was the default model and clear it if so
            if self.config_service:
                try:
                    current_default = await self.config_service.getDefaultModelName()
                    if current_default == modelCard.name:
                        logger.info(
                            f"Clearing default model configuration as '{modelCard.name}' was deleted"
                        )
                        # Clear the default model by setting it to None/empty
                        await self.config_service.setDefaultModelName("")
                except Exception as config_error:
                    # Log the error but don't fail the deletion
                    logger.warning(
                        f"Failed to clear default model configuration: {config_error}"
                    )

            return True

        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            raise InternalException(f"Failed to delete model: {str(e)}") from e

    async def cancelTraining(self, modelCard: ModelCard) -> bool:
        """
        Cancel training for a model and delete the model record.

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

            # Check if this model is currently training
            if (
                self._current_training_task is None
                or self._current_training_task.done()
                or self._current_training_model != modelCard.name
            ):
                raise ConflictException(
                    f"Model '{modelCard.name}' is not currently training"
                )

            # Cancel the training task using asyncio
            canceled = self._current_training_task.cancel(
                f"Training canceled for model {modelCard.name}"
            )

            if canceled:
                logger.info(
                    f"Successfully canceled asyncio task for model {modelCard.name}"
                )
            else:
                logger.warning(
                    f"Task for model {modelCard.name} was already done or could not be canceled"
                )

            # Also cancel training in MLEngine for cleanup
            try:
                await self.ml_engine.cancelTraining(modelCard)
            except Exception as e:
                logger.warning(f"Error cancelling in ML engine: {e}")
                # Don't fail the cancellation if ML engine cleanup fails

            # Delete the model record completely
            await self.database_store_access.deleteEntities(
                EntityType.ModelCard, [modelCard.name]
            )

            logger.info(
                f"Successfully canceled training and deleted model: {modelCard.name}"
            )
            return True

        except (NotFoundException, ConflictException):
            raise
        except Exception as e:
            logger.error(f"Error canceling training for model {modelCard.name}: {e}")
            raise InternalException(f"Failed to cancel training: {str(e)}") from e

    async def _validate_training_request(
        self, params: ModelTrainingRequest
    ) -> list[str]:
        """
        Validate training request parameters.

        Args:
            params: Training request to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate model card
        if not params.modelCard:
            errors.append("Model card is required")
        else:
            if not params.modelCard.name or len(params.modelCard.name.strip()) == 0:
                errors.append("Model name is required and cannot be empty")

            if len(params.modelCard.name) > 100:
                errors.append("Model name cannot exceed 100 characters")

            if not params.modelCard.modelType:
                errors.append("Model type is required")

        # Validate training data location
        if (
            not params.trainingDataLocation
            or len(params.trainingDataLocation.strip()) == 0
        ):
            errors.append("Training data location is required")

        # Validate training parameters
        if params.parameters:
            # Check for required parameters based on model type
            if params.modelCard and params.modelCard.modelType == ModelType.PXBlendSC:
                if "max_categories" in params.parameters:
                    try:
                        max_cats = int(params.parameters["max_categories"])
                        if max_cats <= 0:
                            errors.append("max_categories must be positive")
                    except ValueError:
                        errors.append("max_categories must be a valid integer")

        return errors

    async def _prepare_training_data(self, params: ModelTrainingRequest) -> str:
        """
        Prepare and validate training data for model training.

        Args:
            params: Training request parameters

        Returns:
            Location of prepared training data

        Raises:
            ValidationException: If data preparation fails
            InternalException: If internal error occurs
        """
        try:
            logger.info("Preparing training data")

            # Add filters for better performance
            filters = []

            # Only get approved transactions with categories
            approved_filter = Filter(
                fieldName=FieldNames.APPROVED,
                operator=FilterOperator.EQ,
                value=FilterValue(boolValue=True),
            )
            filters.append(approved_filter)

            # Get transactions from database for training with filters
            query = Query(
                entityType=EntityType.Transaction,
                filters=filters,
                limit=5000,  # Reduced limit for faster processing
            )

            result = await self.database_store_access.getEntities(query)

            if not result.entities:
                raise ValidationException("No transaction data available for training")

            # Extract and validate training data more efficiently
            training_records = []
            categories_seen = set()
            payee_ids_seen = set()
            account_ids_seen = set()
            category_ids_seen = set()

            for entity in result.entities:
                if entity.transaction:
                    transaction = entity.transaction

                    # Skip transactions without required fields
                    if not transaction.categoryId or not transaction.payeeId:
                        continue

                    # Skip if category already has enough samples (for faster processing)
                    if (
                        len(categories_seen) > 20
                        and transaction.categoryId in categories_seen
                    ):
                        continue

                    # Collect IDs for name lookup
                    category_ids_seen.add(transaction.categoryId)
                    payee_ids_seen.add(transaction.payeeId)
                    account_ids_seen.add(transaction.accountId)

                    # Create training record with IDs for now
                    record = {
                        "date": transaction.date or "",
                        "payee_id": transaction.payeeId or "",
                        "memo": transaction.memo or "",
                        "amount": transaction.amount or 0,
                        "account_id": transaction.accountId or "",
                        "category_id": transaction.categoryId,
                    }

                    training_records.append(record)
                    categories_seen.add(transaction.categoryId)

                    # Early exit if we have enough data
                    if len(training_records) >= 1000:
                        break

            # Fetch category, payee, and account names using centralized helper
            logger.info("Fetching category, payee, and account names for training data")

            (
                payee_name_lookup,
                account_name_lookup,
                category_name_lookup,
            ) = await self._resolve_entity_names_bulk(
                payee_ids=payee_ids_seen,
                account_ids=account_ids_seen,
                category_ids=category_ids_seen,
            )

            # Update training records with actual names
            for record in training_records:
                record["payee"] = payee_name_lookup.get(
                    record["payee_id"], record["payee_id"]
                )
                record["category"] = category_name_lookup.get(
                    record["category_id"], record["category_id"]
                )
                record["account_name"] = account_name_lookup.get(
                    record["account_id"], record["account_id"]
                )

                # Convert amount to dollars
                record["amount"] = self._convert_amount_to_dollars(record["amount"])

                # Remove the ID fields as we now have names
                del record["payee_id"]
                del record["category_id"]
                del record["account_id"]

            # Validate training data quality
            if len(training_records) < 10:
                raise ValidationException(
                    "Insufficient training data - need at least 10 transactions"
                )

            if len(categories_seen) < 2:
                raise ValidationException(
                    "Insufficient category diversity - need at least 2 different categories"
                )

            logger.info(
                f"Prepared {len(training_records)} training records with {len(categories_seen)} categories"
            )

            # Create actual CSV file for training
            # Create training data directory if it doesn't exist
            training_dir = "ml_datasets/training"
            os.makedirs(training_dir, exist_ok=True)

            # Create unique filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"training_data_{params.modelCard.name}_{timestamp}.csv"
            file_path = os.path.join(training_dir, filename)

            # Write CSV file
            with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                # Write header
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

                # Write data rows
                for record in training_records:
                    writer.writerow(
                        [
                            record["date"],
                            record["payee"],
                            record["memo"],
                            record["amount"],
                            record["account_name"],
                            record["category"],
                        ]
                    )

            logger.info(f"Created training data file: {file_path}")
            return file_path

        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise InternalException(f"Failed to prepare training data: {str(e)}") from e

    async def _handle_model_versioning(self, model_card: ModelCard) -> ModelCard:
        """
        Handle model versioning by checking for existing models and incrementing version.

        Args:
            model_card: Original model card

        Returns:
            Model card with appropriate version
        """
        try:
            # Check if model with same name already exists
            existing_models = await self.getModels()

            existing_versions = []
            for model in existing_models:
                if model.name == model_card.name:
                    try:
                        version_num = float(model.version or "1.0")
                        existing_versions.append(version_num)
                    except ValueError:
                        # Handle non-numeric versions
                        existing_versions.append(1.0)

            # Determine next version
            if existing_versions:
                next_version = max(existing_versions) + 0.1
                new_version = f"{next_version:.1f}"
            else:
                new_version = "1.0"

            # Create new model card with updated version
            versioned_card = ModelCard(
                modelType=model_card.modelType,
                name=model_card.name,
                version=new_version,
                description=model_card.description
                or f"Version {new_version} of {model_card.name}",
                status=model_card.status,
                trainedDate=model_card.trainedDate,
                performanceMetrics=model_card.performanceMetrics,
            )

            logger.info(f"Model versioning: {model_card.name} -> version {new_version}")
            return versioned_card

        except Exception as e:
            logger.warning(
                f"Error handling model versioning: {e}, using original version"
            )
            return model_card

    async def _store_trained_model_with_metrics(self, result: ModelTrainingResult):
        """
        Store trained model with enhanced performance tracking.

        Args:
            result: Training result to store
        """
        try:
            # Enhance model card with additional metadata
            enhanced_model = ModelCard(
                modelType=result.modelCard.modelType,
                name=result.modelCard.name,
                version=result.modelCard.version,
                description=result.modelCard.description,
                status=result.status,
                trainedDate=result.modelCard.trainedDate,
                performanceMetrics=result.modelCard.performanceMetrics or {},
            )

            # Add training metadata
            if not enhanced_model.performanceMetrics:
                enhanced_model.performanceMetrics = {}

            enhanced_model.performanceMetrics.update(
                {
                    "training_completed_at": enhanced_model.trainedDate or "",
                    "training_status": "success" if result.status == 3 else "failed",
                    "model_version": enhanced_model.version or "1.0",
                }
            )

            # Store enhanced model
            entity = Entity(modelCard=enhanced_model)
            await self.database_store_access.upsertEntities([entity])

            logger.info(
                f"Stored trained model with metrics: {enhanced_model.name} v{enhanced_model.version}"
            )

        except Exception as e:
            logger.error(f"Error storing trained model: {e}")
            # Don't raise exception here as training was successful

    async def _validate_prediction_request(
        self, request: PredictionRequest
    ) -> list[str]:
        """
        Validate prediction request parameters.

        Args:
            request: Prediction request to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not request.transactions:
            errors.append("Transactions are required for prediction")
            return errors

        # Validate transaction data
        if request.transactions.transactions:
            if len(request.transactions.transactions) == 0:
                errors.append("At least one transaction is required")
            elif len(request.transactions.transactions) > 1000:
                errors.append("Too many transactions - maximum 1000 per request")
        elif request.transactions.transactionIds:
            if len(request.transactions.transactionIds) == 0:
                errors.append("At least one transaction ID is required")
            elif len(request.transactions.transactionIds) > 1000:
                errors.append("Too many transaction IDs - maximum 1000 per request")
        else:
            errors.append("Either transactions or transaction IDs must be provided")

        # Validate model card if provided
        if request.modelCard:
            if not request.modelCard.name:
                errors.append("Model name is required when model card is provided")

        return errors

    async def _prepare_prediction_input(self, transactions: Transactions) -> dict:
        """
        Prepare prediction input from transactions with proper data resolution.

        Args:
            transactions: Transactions to prepare for prediction

        Returns:
            Primitive batch data for prediction with resolved names and formatted amounts
        """
        primitive_batch = {}
        prediction_transactions = []

        # Check if we have transaction IDs first, if so fetch the transactions
        if transactions.transactionIds:
            # Fetch transactions by IDs
            fetched_transactions = []
            for transaction_id in transactions.transactionIds:
                try:
                    filter_value = FilterValue(stringValue=transaction_id)
                    id_filter = Filter(
                        fieldName=FieldNames.ID,
                        operator=FilterOperator.EQ,
                        value=filter_value,
                    )

                    query = Query(
                        entityType=EntityType.Transaction, filters=[id_filter], limit=1
                    )

                    result = await self.database_store_access.getEntities(query)

                    if result.entities and result.entities[0].transaction:
                        fetched_transactions.append(result.entities[0].transaction)

                except Exception as e:
                    logger.warning(f"Error fetching transaction {transaction_id}: {e}")
                    continue

            prediction_transactions = fetched_transactions

        prediction_transactions = prediction_transactions or (
            transactions.transactions if transactions.transactions else []
        )

        # Process transactions into primitive batch with proper data resolution
        if prediction_transactions:
            for i, transaction in enumerate(prediction_transactions):
                if transaction and self._is_valid_transaction_for_prediction(
                    transaction
                ):
                    # Resolve payee name from payeeId
                    payee_name = await self._resolve_payee_name(transaction.payeeId)

                    # Resolve account name from accountId
                    account_name = await self._resolve_account_name(
                        transaction.accountId
                    )

                    # Convert amount from millicents to dollars
                    amount_dollars = self._convert_amount_to_dollars(transaction.amount)

                    row = [
                        PrimitiveValue(stringValue=transaction.date or ""),
                        PrimitiveValue(stringValue=payee_name or ""),
                        PrimitiveValue(stringValue=transaction.memo or ""),
                        PrimitiveValue(doubleValue=amount_dollars),
                        PrimitiveValue(stringValue=account_name or ""),
                    ]
                    primitive_batch[i] = row

        return primitive_batch

    def _is_valid_transaction_for_prediction(self, transaction) -> bool:
        """
        Check if a transaction has sufficient data for prediction.

        Args:
            transaction: Transaction to validate

        Returns:
            True if transaction is valid for prediction
        """
        if not transaction:
            return False

        # Must have at least payee or memo for meaningful prediction
        has_payee = bool(transaction.payeeId and len(transaction.payeeId.strip()) > 0)
        has_memo = bool(transaction.memo and len(transaction.memo.strip()) > 0)
        has_amount = transaction.amount is not None and transaction.amount != 0

        return (has_payee or has_memo) and has_amount

    async def _select_prediction_model(
        self, requested_model: ModelCard | None
    ) -> ModelCard:
        """
        Select and validate the model to use for prediction.

        Args:
            requested_model: Optional specific model requested

        Returns:
            Model card to use for prediction

        Raises:
            ValidationException: If requested model is invalid
            InternalException: If no models available
        """
        if requested_model:
            # Validate requested model exists
            existing_models = await self.ml_engine.getModels()

            for model in existing_models:
                if model.name == requested_model.name:
                    logger.info(f"Using requested model: {model.name} v{model.version}")
                    return model

            raise ValidationException(
                f"Requested model '{requested_model.name}' not found"
            )

        else:
            # Get default model - first check for configured default, then fall back to most recent
            models = await self.ml_engine.getModels()

            if not models:
                raise InternalException("No models available for prediction")

            # Check if there's a configured default model
            default_model_name = await self.config_service.getDefaultModelName()
            if default_model_name and default_model_name.strip():
                # Look for the configured default model
                for model in models:
                    if (
                        model.name == default_model_name
                        and model.status == TrainingStatus.Success
                    ):
                        logger.info(
                            f"Using configured default model: {model.name} v{model.version}"
                        )
                        return model

                # If configured default model not found or not ready, log warning and fall back
                logger.warning(
                    f"Configured default model '{default_model_name}' not found or not ready, "
                    "falling back to most recent model"
                )

            # Fall back to most recent successful training
            completed_models = [m for m in models if m.status == TrainingStatus.Success]
            if not completed_models:
                raise InternalException("No completed models available for prediction")

            # Sort by version (descending) to get most recent
            sorted_models = sorted(
                completed_models, key=lambda m: float(m.version or "0.0"), reverse=True
            )

            selected_model = sorted_models[0]
            logger.info(
                f"Using fallback default model: {selected_model.name} v{selected_model.version}"
            )
            return selected_model

    async def _get_predictions_with_retry(
        self, ml_request: ModelPredictionBatchRequest, max_retries: int = 2
    ) -> list[ModelPredictionResult]:
        """
        Get predictions with retry logic for resilience.

        Args:
            ml_request: ML prediction request
            max_retries: Maximum number of retry attempts

        Returns:
            List of prediction results

        Raises:
            RemoteServiceException: If all retries fail
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                logger.info(
                    f"Getting predictions (attempt {attempt + 1}/{max_retries + 1})"
                )

                prediction_results = await self.ml_engine.getPredictions([ml_request])

                if prediction_results:
                    return prediction_results
                else:
                    last_error = "No results returned from ML engine"

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Prediction attempt {attempt + 1} failed: {e}")

                if attempt < max_retries:
                    # Brief delay before retry
                    await asyncio.sleep(0.5)

        raise RemoteServiceException(
            f"Prediction failed after {max_retries + 1} attempts: {last_error}"
        )

    async def _process_prediction_results(
        self, ml_result: ModelPredictionResult
    ) -> PredictionResult:
        """
        Process and validate ML prediction results.

        Args:
            ml_result: Raw ML prediction result

        Returns:
            Processed prediction result
        """
        try:
            # Extract and validate categorical predictions
            if ml_result.result and ml_result.result.categoricalPredictionResults:
                results = ml_result.result.categoricalPredictionResults

                # Validate prediction quality
                total_predictions = len(results)
                valid_predictions = 0

                for _row_id, predictions in results.items():
                    if predictions and len(predictions) > 0:
                        # Check if predictions have reasonable confidence scores
                        for pred in predictions:
                            if pred.confidence >= 0.0 and pred.confidence <= 1.0:
                                valid_predictions += 1
                                break

                prediction_quality = (
                    valid_predictions / total_predictions
                    if total_predictions > 0
                    else 0.0
                )

                logger.info(
                    f"Prediction quality: {prediction_quality:.2%} ({valid_predictions}/{total_predictions})"
                )

                return PredictionResult(
                    results=results, errorMessage=ml_result.errorMessage
                )
            else:
                return PredictionResult(
                    results={},
                    errorMessage=ml_result.errorMessage or "No predictions generated",
                )

        except Exception as e:
            logger.error(f"Error processing prediction results: {e}")
            return PredictionResult(
                results={}, errorMessage=f"Error processing results: {str(e)}"
            )

    # Centralized data resolution helper methods

    async def _resolve_entity_names_bulk(
        self, payee_ids: set = None, account_ids: set = None, category_ids: set = None
    ) -> tuple[dict, dict, dict]:
        """
        Resolve entity IDs to names in bulk for efficient lookups.

        Args:
            payee_ids: Set of payee IDs to resolve
            account_ids: Set of account IDs to resolve
            category_ids: Set of category IDs to resolve

        Returns:
            Tuple of (payee_lookup, account_lookup, category_lookup) dictionaries
        """
        payee_lookup = {}
        account_lookup = {}
        category_lookup = {}

        # Get payee names
        if payee_ids:
            try:
                payee_entities = await self.database_store_access.getEntitiesById(
                    EntityType.Payee, list(payee_ids)
                )
                for entity in payee_entities:
                    if entity.payee:
                        payee_lookup[entity.payee.id] = entity.payee.name
            except Exception as e:
                logger.warning(f"Error fetching payee names: {e}")

        # Get account names
        if account_ids:
            try:
                account_entities = await self.database_store_access.getEntitiesById(
                    EntityType.Account, list(account_ids)
                )
                for entity in account_entities:
                    if entity.account:
                        account_lookup[entity.account.id] = entity.account.name
            except Exception as e:
                logger.warning(f"Error fetching account names: {e}")

        # Get category names
        if category_ids:
            try:
                category_entities = await self.database_store_access.getEntitiesById(
                    EntityType.Category, list(category_ids)
                )
                for entity in category_entities:
                    if entity.category:
                        category_lookup[entity.category.id] = entity.category.name
            except Exception as e:
                logger.warning(f"Error fetching category names: {e}")

        return payee_lookup, account_lookup, category_lookup

    async def _resolve_payee_name(self, payee_id: str) -> str:
        """
        Resolve a single payee ID to name.

        Args:
            payee_id: Payee ID to resolve

        Returns:
            Payee name or original ID if not found
        """
        if not payee_id or not payee_id.strip():
            return ""

        try:
            payee_lookup, _, _ = await self._resolve_entity_names_bulk(
                payee_ids={payee_id}
            )
            return payee_lookup.get(payee_id, payee_id)
        except Exception as e:
            logger.warning(f"Error resolving payee name for {payee_id}: {e}")
            return payee_id

    async def _resolve_account_name(self, account_id: str) -> str:
        """
        Resolve a single account ID to name.

        Args:
            account_id: Account ID to resolve

        Returns:
            Account name or original ID if not found
        """
        if not account_id or not account_id.strip():
            return ""

        try:
            _, account_lookup, _ = await self._resolve_entity_names_bulk(
                account_ids={account_id}
            )
            return account_lookup.get(account_id, account_id)
        except Exception as e:
            logger.warning(f"Error resolving account name for {account_id}: {e}")
            return account_id

    async def _resolve_category_name(self, category_id: str) -> str:
        """
        Resolve a single category ID to name.

        Args:
            category_id: Category ID to resolve

        Returns:
            Category name or original ID if not found
        """
        if not category_id or not category_id.strip():
            return ""

        try:
            _, _, category_lookup = await self._resolve_entity_names_bulk(
                category_ids={category_id}
            )
            return category_lookup.get(category_id, category_id)
        except Exception as e:
            logger.warning(f"Error resolving category name for {category_id}: {e}")
            return category_id

    def _convert_amount_to_dollars(self, amount: int | None) -> float:
        """
        Convert amount from millicents to dollars.

        Args:
            amount: Amount in millicents (or whatever unit is stored)

        Returns:
            Amount in dollars as float
        """
        if amount is None:
            return 0.0

        # Based on your debug output showing -30670.0 for -$30.67,
        # it looks like amounts are stored in millicents (1000x dollars)
        return float(amount) / 1000.0
