"""
AI/ML API routes - Simplified version.

This module provides endpoints for machine learning functionality using only
the methods defined in the PredictionManager Thrift service interface.
"""

import asyncio
import logging
import os
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from api_layer.dependencies import ConfigsDep, MLEngineDep, PredictionManagerDep
from business_layer.ml_engine import MLEngine
from business_layer.prediction_manager import PredictionManager
from configs import ConfigService, get_config_service
from thrift_gen.entities.ttypes import (
    ModelCard,
    ModelType,
    TrainingStatus,
    Transactions,
)
from thrift_gen.exceptions.ttypes import (
    ConflictException,
    InternalException,
    NotFoundException,
    RemoteServiceException,
    UnauthorizedException,
    ValidationException,
)
from thrift_gen.mlengine.ttypes import ModelTrainingRequest
from thrift_gen.predictionmanager.ttypes import (
    PredictionRequest,
    TrainingDataPreparationRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["machine-learning"])

# In-memory training status tracking
# In production, this should be stored in Redis or database
training_status_store = {}


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

    # Handle dictionaries
    if isinstance(thrift_obj, dict):
        return {key: thrift_to_dict(value) for key, value in thrift_obj.items()}

    # Handle Thrift objects
    if hasattr(thrift_obj, "__dict__"):
        result = {}
        # Get the actual attributes from the Thrift object
        if hasattr(thrift_obj, "__slots__"):
            # Use __slots__ if available
            for attr_name in thrift_obj.__slots__:
                if hasattr(thrift_obj, attr_name):
                    attr_value = getattr(thrift_obj, attr_name)
                    if attr_value is not None:
                        result[attr_name] = thrift_to_dict(attr_value)
        else:
            # Fallback to __dict__
            for attr_name, attr_value in thrift_obj.__dict__.items():
                if not attr_name.startswith("_") and attr_value is not None:
                    result[attr_name] = thrift_to_dict(attr_value)
        return result

    # For other types, try to convert to string
    try:
        return str(thrift_obj)
    except Exception:
        return None


def success_response(message: str, data: Any = None) -> dict[str, Any]:
    """Create a success response."""
    response = {"success": True, "message": message}
    if data is not None:
        response["data"] = data
    return response


def error_response(message: str, status_code: int = 400) -> dict[str, Any]:
    """Create an error response."""
    return {"success": False, "message": message, "status_code": status_code}


# Request/Response Models (keeping Pydantic for API layer)


class TrainingRequest(BaseModel):
    """Request model for training a new model."""

    ml_model_name: str = Field(..., description="Name for the new model")
    ml_model_type: str = Field(
        default="CATEGORICAL", description="Type of model to train"
    )
    training_data_location: str = Field(
        ..., description="Path to training data CSV file"
    )
    training_params: dict[str, Any] = Field(
        default_factory=dict, description="Training parameters"
    )


class TrainingDataRequest(BaseModel):
    """Request model for preparing training data."""

    budget_id: str | None = Field(None, description="Budget ID to filter transactions")
    months_back: int = Field(
        default=12, description="Number of months of data to include"
    )
    test_split_ratio: float = Field(
        default=0.2, description="Ratio of data to use for testing (0.0-1.0)"
    )
    min_samples_per_category: int = Field(
        default=1, description="Minimum samples per category in test set"
    )


class PredictionRequestModel(BaseModel):
    """Request model for getting predictions."""

    transaction_ids: list[str] = Field(
        ..., description="List of transaction IDs to predict"
    )
    ml_model_name: str | None = Field(
        None, description="Specific model to use for predictions"
    )


# Model Management Endpoints


@router.get("/models")
async def list_models(
    ml_model_type: str | None = Query(None),
    prediction_manager: PredictionManager = PredictionManagerDep,
):
    """
    List available ML models.

    This endpoint returns all available models, optionally filtered by type.
    """
    try:
        logger.info(f"Listing models (type filter: {ml_model_type})")

        # Convert string to ModelType enum if provided
        model_type_enum = None
        if ml_model_type:
            try:
                # This would need to be mapped to the actual ModelType enum values
                model_type_enum = ModelType._VALUES_TO_NAMES.get(ml_model_type.upper())
            except Exception:
                logger.warning(f"Invalid model type: {ml_model_type}")

        # Get models from PredictionManager
        try:
            models = await prediction_manager.getModels(model_type_enum)
        except NotFoundException:
            # Return empty list instead of error when no models found
            models = []

        logger.info(f"Retrieved {len(models)} models")

        return success_response(
            f"Retrieved {len(models)} models",
            [thrift_to_dict(model) for model in models],
        )

    except (InternalException, UnauthorizedException) as e:
        logger.error(f"Service error listing models: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to list models: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error listing models: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while listing models"
        ) from e


@router.delete("/models/{model_name}")
async def delete_model(
    model_name: str,
    prediction_manager: PredictionManager = PredictionManagerDep,
    config_service: ConfigService = ConfigsDep,
):
    """
    Delete a trained model.

    This endpoint removes a model from the system.
    """
    try:
        logger.info(f"Deleting model: {model_name}")

        # Look up the existing model to get its type
        existing_models = await prediction_manager.getModels()
        model_info = next(
            (model for model in existing_models if model.name == model_name),
            None,
        )

        if not model_info:
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not found"
            )

        # Delete model through PredictionManager
        success = await prediction_manager.deleteModel(model_info)

        if success:
            logger.info(f"Successfully deleted model: {model_name}")
            return success_response(f"Model {model_name} deleted successfully")
        else:
            raise HTTPException(status_code=500, detail="Failed to delete model")

    except HTTPException:
        # Re-raise HTTPExceptions (like 404) without modification
        raise
    except (
        NotFoundException,
        ConflictException,
        InternalException,
        UnauthorizedException,
    ) as e:
        logger.error(f"Service error deleting model {model_name}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to delete model: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(
            f"Unexpected error deleting model {model_name}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while deleting model: {str(e)}",
        ) from e


@router.post("/models/{model_name}/set-default")
async def set_default_model(
    model_name: str,
    prediction_manager: PredictionManager = PredictionManagerDep,
    config_service: ConfigService = ConfigsDep,
):
    """
    Set a model as the default for future predictions.

    This endpoint sets the specified model as the default model that will be used
    for predictions when no specific model is requested.
    """
    try:
        logger.info(f"Setting default model: {model_name}")

        # Verify the model exists
        existing_models = await prediction_manager.getModels()
        model_info = next(
            (model for model in existing_models if model.name == model_name),
            None,
        )

        if not model_info:
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not found"
            )

        # Verify the model is trained and ready
        if model_info.status != TrainingStatus.Success:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' is not ready for use. Status: {TrainingStatus._VALUES_TO_NAMES.get(model_info.status, 'Unknown')}",
            )

        # Set as default model in configuration
        await config_service.setDefaultModelName(model_name)

        logger.info(f"Successfully set default model: {model_name}")
        return success_response(f"Model '{model_name}' set as default successfully")

    except HTTPException:
        # Re-raise HTTPExceptions (like 404, 400) without modification
        raise
    except (
        NotFoundException,
        ConflictException,
        InternalException,
        UnauthorizedException,
    ) as e:
        logger.error(f"Service error setting default model {model_name}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to set default model: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(
            f"Unexpected error setting default model {model_name}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while setting default model: {str(e)}",
        ) from e


# Training Endpoints


async def _background_train_model(
    training_request: ModelTrainingRequest,
    prediction_manager: PredictionManager,
    model_name: str,
):
    """Background task for model training."""
    try:
        # Update status to in progress
        training_status_store[model_name] = {
            "status": "training",
            "message": "Training in progress...",
            "progress": 0.1,
            "started_at": time.time(),
        }

        logger.info(f"Background training started for model: {model_name}")

        result = await prediction_manager.trainModel(training_request)

        # Update status based on result
        if result.status == TrainingStatus.Success:
            training_status_store[model_name] = {
                "status": "completed",
                "message": "Training completed successfully",
                "progress": 1.0,
                "result": thrift_to_dict(result),
                "completed_at": time.time(),
            }
            logger.info(
                f"Background training completed successfully for model: {model_name}"
            )
        else:
            training_status_store[model_name] = {
                "status": "failed",
                "message": result.errorMessage or "Training failed",
                "progress": 0.0,
                "error": result.errorMessage,
                "completed_at": time.time(),
            }
            logger.error(f"Background training failed for model: {model_name}")

    except ConflictException:
        # Training was canceled
        logger.info(f"Training was canceled for model: {model_name}")
        training_status_store[model_name] = {
            "status": "canceled",
            "message": "Training was canceled by user",
            "progress": 0.0,
            "canceled_at": time.time(),
        }
    except Exception as e:
        logger.error(f"Background training error for model {model_name}: {e}")
        training_status_store[model_name] = {
            "status": "failed",
            "message": f"Training failed: {str(e)}",
            "progress": 0.0,
            "error": str(e),
            "completed_at": time.time(),
        }


@router.post("/train")
async def train_model(
    request: TrainingRequest,
    prediction_manager: PredictionManager = PredictionManagerDep,
    config_service: ConfigService = ConfigsDep,
):
    """
    Start training a new ML model asynchronously.

    This endpoint starts training a new model in the background and returns immediately.
    Use the /train/status/{model_name} endpoint to check training progress.
    """
    logger.info(f"Starting async training for model: {request.ml_model_name}")

    # Check if training is already in progress for this specific model
    if request.ml_model_name in training_status_store:
        current_status = training_status_store[request.ml_model_name]
        if current_status["status"] in ["starting", "training"]:
            raise HTTPException(
                status_code=409,  # Conflict
                detail=f"Training already in progress for model '{request.ml_model_name}'",
            )

    try:
        # Create ModelCard and ModelTrainingRequest
        # Map string model type to enum, get default from ConfigService
        config_service = config_service or get_config_service()
        model_type = await config_service.getDefaultModelType()
        if hasattr(request, "ml_model_type") and request.ml_model_type:
            # Use enum name comparison instead of hardcoded string
            if request.ml_model_type.upper() == ModelType._VALUES_TO_NAMES.get(
                ModelType.PXBlendSC
            ):
                model_type = ModelType.PXBlendSC

        model_card = ModelCard(
            modelType=model_type,
            name=request.ml_model_name,
            version="1.0",
            description=f"Model trained from {request.training_data_location}",
            status=TrainingStatus.Pending,
        )

        # Convert training parameters to string map as required by Thrift
        string_params = {}
        if request.training_params:
            logger.info(f"Original training params: {request.training_params}")
            for key, value in request.training_params.items():
                string_params[key] = str(value)
            logger.info(f"Converted string params: {string_params}")

        training_request = ModelTrainingRequest(
            modelCard=model_card,
            trainingDataLocation=request.training_data_location,
            parameters=string_params,
        )

        # Initialize training status
        training_status_store[request.ml_model_name] = {
            "status": "starting",
            "message": "Training is starting...",
            "progress": 0.0,
            "started_at": time.time(),
        }

        # Start background training as a task
        asyncio.create_task(
            _background_train_model(
                training_request,
                prediction_manager,
                request.ml_model_name,
            )
        )

        logger.info(f"Async training queued for model: {request.ml_model_name}")

        return success_response(
            f"Training started for model {request.ml_model_name}",
            {
                "status": "starting",
                "message": "Training has been queued and will start shortly",
                "model_name": request.ml_model_name,
            },
        )

    except (
        ValidationException,
        InternalException,
        UnauthorizedException,
        ConflictException,
    ) as e:
        # Clean up training status
        if request.ml_model_name in training_status_store:
            del training_status_store[request.ml_model_name]
        logger.error(f"Service error training model {request.ml_model_name}: {e}")

        # Map ConflictException to 409 status code
        status_code = 409 if isinstance(e, ConflictException) else 400
        raise HTTPException(
            status_code=status_code, detail=f"Failed to train model: {str(e)}"
        ) from e
    except Exception as e:
        # Clean up training status
        if request.ml_model_name in training_status_store:
            del training_status_store[request.ml_model_name]
        logger.error(f"Unexpected error training model {request.ml_model_name}: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while training model"
        ) from e


@router.get("/train/status")
async def get_current_training_status():
    """
    Get the current training status (if any training is in progress).

    Returns information about the currently training model, or null if no training is active.
    """
    try:
        # Find any model currently training
        current_training = None
        for model_name, status_info in training_status_store.items():
            if status_info["status"] in ["starting", "training"]:
                current_training = model_name
                break

        if current_training is None:
            return success_response(
                "No training currently in progress",
                {"current_training": None, "training_active": False},
            )

        # Get status of currently training model
        status_info = training_status_store[current_training].copy()

        # Calculate elapsed time
        if status_info["status"] in ["starting", "training"]:
            elapsed = time.time() - status_info["started_at"]
            status_info["elapsed_seconds"] = round(elapsed, 1)

        return success_response(
            f"Training in progress for model: {current_training}",
            {
                "current_training": current_training,
                "training_active": True,
                "status": status_info,
            },
        )

    except Exception as e:
        logger.error(f"Error getting current training status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting training status",
        ) from e


@router.get("/train/status/{model_name}")
async def get_training_status(model_name: str):
    """
    Get the training status for a specific model.

    Returns the current status, progress, and any error messages.
    """
    try:
        if model_name not in training_status_store:
            raise HTTPException(
                status_code=404,
                detail=f"No training status found for model: {model_name}",
            )

        status_info = training_status_store[model_name].copy()

        # Calculate elapsed time if training is in progress
        if status_info["status"] in ["starting", "training"]:
            elapsed = time.time() - status_info["started_at"]
            status_info["elapsed_seconds"] = round(elapsed, 1)
        elif "completed_at" in status_info and "started_at" in status_info:
            elapsed = status_info["completed_at"] - status_info["started_at"]
            status_info["elapsed_seconds"] = round(elapsed, 1)

        return success_response(f"Training status for {model_name}", status_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training status for {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting training status",
        ) from e


@router.post("/train/cancel/{model_name}")
async def cancel_training(
    model_name: str,
    prediction_manager: PredictionManager = PredictionManagerDep,
):
    """
    Cancel training for a specific model and delete the model record.

    This endpoint cancels ongoing training and removes the model completely.
    """
    try:
        logger.info(f"Canceling training for model: {model_name}")

        # Create a ModelCard for the cancellation request
        # We need to look up the existing model to get its details
        try:
            existing_models = await prediction_manager.getModels()
            model_card = next(
                (model for model in existing_models if model.name == model_name),
                None,
            )

            if not model_card:
                # If model not found in database, create a minimal ModelCard for cancellation
                model_card = ModelCard(
                    name=model_name,
                    modelType=ModelType.PXBlendSC,  # Default type
                    version="1.0",
                    status=TrainingStatus.Pending,
                )
        except Exception as e:
            logger.warning(f"Could not look up model {model_name}: {e}")
            # Create a minimal ModelCard for cancellation
            model_card = ModelCard(
                name=model_name,
                modelType=ModelType.PXBlendSC,  # Default type
                version="1.0",
                status=TrainingStatus.Pending,
            )

        # Cancel training through PredictionManager
        success = await prediction_manager.cancelTraining(model_card)

        if success:
            # Update training status to canceled immediately for UI responsiveness
            training_status_store[model_name] = {
                "status": "canceled",
                "message": "Training was canceled by user",
                "progress": 0.0,
                "canceled_at": time.time(),
            }

            logger.info(f"Successfully canceled training for model: {model_name}")
            return success_response(
                f"Training canceled and model {model_name} deleted successfully"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to cancel training",
            )

    except (NotFoundException, ConflictException) as e:
        # Map Thrift exceptions to HTTP status codes
        if isinstance(e, NotFoundException):
            status_code = 404
        elif isinstance(e, ConflictException):
            status_code = 409
        else:
            status_code = 400

        raise HTTPException(
            status_code=status_code,
            detail=f"Failed to cancel training: {str(e)}",
        ) from e
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error canceling training for {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while canceling training",
        ) from e


# Prediction Endpoints


@router.post("/predict")
async def get_predictions(
    request: PredictionRequestModel,
    prediction_manager: PredictionManager = PredictionManagerDep,
    config_service: ConfigService = ConfigsDep,
):
    """
    Get predictions for transactions.

    This endpoint returns category predictions for the specified transactions.
    """
    try:
        logger.info(
            f"Getting predictions for {len(request.transaction_ids)} transactions"
        )

        # Create Transactions object with transaction IDs
        transactions = Transactions(transactionIds=request.transaction_ids)

        # Look up the existing model if model name is specified
        model_info = None
        if request.ml_model_name:
            existing_models = await prediction_manager.getModels()
            model_info = next(
                (
                    model
                    for model in existing_models
                    if model.name == request.ml_model_name
                ),
                None,
            )

            if not model_info:
                raise HTTPException(
                    status_code=404, detail=f"Model '{request.ml_model_name}' not found"
                )

        # Create PredictionRequest
        prediction_request = PredictionRequest(
            transactions=transactions, modelCard=model_info
        )

        # Get predictions through PredictionManager
        result = await prediction_manager.getPredictions(prediction_request)

        logger.info(
            f"Retrieved predictions for {len(request.transaction_ids)} transactions"
        )

        # Transform the result to match frontend expectations
        predictions = []
        if result.results:
            for i, transaction_id in enumerate(request.transaction_ids):
                transaction_predictions = []

                # Try both string and integer keys (ML engine uses integer keys)
                raw_predictions = None
                if str(transaction_id) in result.results:
                    raw_predictions = result.results[str(transaction_id)]
                elif i in result.results:
                    raw_predictions = result.results[i]
                elif str(i) in result.results:
                    raw_predictions = result.results[str(i)]

                if raw_predictions:
                    for pred in raw_predictions:
                        transaction_predictions.append(
                            {
                                "categoryId": pred.predictedCategory,
                                "confidence": pred.confidence,
                            }
                        )

                predictions.append(
                    {
                        "transactionId": transaction_id,
                        "predictions": transaction_predictions,
                    }
                )

        return success_response(
            f"Retrieved predictions for {len(request.transaction_ids)} transactions",
            {"predictions": predictions},
        )

    except (
        ValidationException,
        InternalException,
        UnauthorizedException,
        RemoteServiceException,
    ) as e:
        logger.error(f"Service error getting predictions: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to get predictions: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error getting predictions: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error while getting predictions"
        ) from e


# Training Data Preparation Endpoints


@router.get("/training-data/stats")
async def get_training_data_stats(
    budget_id: str | None = Query(None),
    months_back: int = Query(12),
    prediction_manager: PredictionManager = PredictionManagerDep,
):
    """
    Get training data statistics for analysis.

    This endpoint provides statistics about available training data
    to help users understand data quality before training.
    """
    try:
        logger.info(
            f"Getting training data stats (budget: {budget_id}, months: {months_back})"
        )

        thrift_request = TrainingDataPreparationRequest(
            budgetId=budget_id,
            monthsBack=months_back,
            testSplitRatio=0.2,  # Standard split for analysis
            minSamplesPerCategory=1,
        )

        result = await prediction_manager.prepareTrainingData(thrift_request)

        dataset_info = result.datasetInfo

        # Calculate suitability metrics based on modern ML requirements
        # Our PXBlendSC-RF strategy uses cross-validation, multiple models, and sophisticated features
        total_samples = dataset_info.trainingSamples
        num_categories = dataset_info.categories
        avg_samples_per_category = (
            total_samples / num_categories if num_categories > 0 else 0
        )

        # Check for category distribution issues
        category_warnings = []
        if avg_samples_per_category < 5:
            category_warnings.append(
                f"Low average samples per category ({avg_samples_per_category:.1f})"
            )

        # Check for severely imbalanced categories if breakdown is available
        if dataset_info.categoryBreakdown and (
            category_counts := list(dataset_info.categoryBreakdown.values())
        ):
            min_category_samples = min(category_counts)
            max_category_samples = max(category_counts)

            if min_category_samples < 3:
                category_warnings.append(
                    f"Some categories have very few samples (min: {min_category_samples})"
                )

            if (
                min_category_samples > 0
                and max_category_samples / min_category_samples > 20
            ):
                category_warnings.append("Highly imbalanced categories detected")

        excellent_data = (
            total_samples >= 500
            and num_categories >= 15
            and avg_samples_per_category >= 20
        )

        good_data = (
            total_samples >= 200
            and num_categories >= 8
            and avg_samples_per_category >= 10
        )

        minimal_data = (
            total_samples >= 100
            and num_categories >= 5
            and avg_samples_per_category >= 5
        )

        # Build concise recommendation
        if excellent_data:
            recommendation = "Excellent data quality - ideal for robust model training"
        elif good_data:
            recommendation = (
                "Good data quality - sufficient for reliable model training"
            )
        elif minimal_data:
            recommendation = (
                "Minimal data quality - training possible but more data recommended"
            )
        else:
            recommendation = "Insufficient data - need at least 100+ transactions across 5+ categories"

        sufficient_data = minimal_data  # Maintain backward compatibility

        stats = {
            "categorized_transactions": dataset_info.trainingSamples
            + dataset_info.testSamples,
            "unique_categories": dataset_info.categories,
            "category_breakdown": dataset_info.categoryBreakdown or {},
            "suitability": {
                "sufficient_data": sufficient_data,
                "recommendation": recommendation,
                "warnings": category_warnings,  # Add warnings as separate array
            },
        }

        logger.info(
            f"Training data stats: {stats['categorized_transactions']} transactions, "
            f"{stats['unique_categories']} categories"
        )

        return success_response(
            f"Training data stats: {stats['categorized_transactions']} transactions, "
            f"{stats['unique_categories']} categories",
            stats,
        )

    except (ValidationException, NotFoundException) as e:
        logger.error(f"Service error getting training data stats: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to get training data stats: {str(e)}"
        ) from e
    except (InternalException, UnauthorizedException) as e:
        logger.error(f"Service error getting training data stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get training data stats: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error getting training data stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while getting training data stats",
        ) from e


@router.post("/training-data/prepare")
async def prepare_training_data(
    request: TrainingDataRequest,
    prediction_manager: PredictionManager = PredictionManagerDep,
):
    """
    Prepare training data from transactions.

    This endpoint delegates to PredictionManager to orchestrate data preparation.
    """
    try:
        logger.info(
            f"Preparing training data (budget: {request.budget_id}, months: {request.months_back})"
        )

        # Create Thrift request
        thrift_request = TrainingDataPreparationRequest(
            budgetId=request.budget_id,
            monthsBack=request.months_back,
            testSplitRatio=request.test_split_ratio,
            minSamplesPerCategory=request.min_samples_per_category,
        )

        result = await prediction_manager.prepareTrainingData(thrift_request)

        # Convert result to API response format
        dataset_info = result.datasetInfo
        stats = {
            "dataset_id": dataset_info.datasetId,
            "dataset_name": dataset_info.datasetName,
            "training_samples": dataset_info.trainingSamples,
            "test_samples": dataset_info.testSamples,
            "categories": dataset_info.categories,
            "category_breakdown": dataset_info.categoryBreakdown or {},
            "training_file": dataset_info.trainingDataLocation,
            "test_file": dataset_info.testDataLocation,
            "date_from": dataset_info.dateFrom,
            "date_to": dataset_info.dateTo,
            "auto_continue": False,  # Always require user review before training
        }

        logger.info(
            f"Training data prepared: {dataset_info.trainingSamples} training, "
            f"{dataset_info.testSamples} test samples"
        )

        return success_response(
            f"Training data prepared: {dataset_info.trainingSamples} training, "
            f"{dataset_info.testSamples} test samples",
            stats,
        )

    except (ValidationException, NotFoundException) as e:
        logger.error(f"Service error preparing training data: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to prepare training data: {str(e)}"
        ) from e
    except (InternalException, UnauthorizedException) as e:
        logger.error(f"Service error preparing training data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to prepare training data: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error preparing training data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while preparing training data",
        ) from e


@router.get("/datasets")
async def list_datasets(
    budget_id: str | None = Query(None),
    ml_engine: MLEngine = MLEngineDep,
):
    """
    List available training datasets.

    This endpoint returns all stored training datasets with their metadata.
    """
    try:
        logger.info(f"Listing datasets (budget filter: {budget_id})")

        # Get datasets from MLEngine
        datasets = await ml_engine.getDatasets(budgetId=budget_id)

        # Convert to API response format
        dataset_list = []
        for dataset in datasets:
            # Try to get created_at from metadata file
            created_at = None
            try:
                dataset_dir = os.path.dirname(dataset.trainingDataLocation)
                metadata_file = os.path.join(dataset_dir, "metadata.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, encoding="utf-8") as f:
                        import json

                        metadata = json.load(f)
                        created_at = metadata.get("created_at")
            except Exception as e:
                logger.warning(
                    f"Could not read metadata for dataset {dataset.datasetId}: {e}"
                )

            dataset_dict = {
                "id": dataset.datasetId,
                "name": dataset.datasetName,  # Frontend expects 'name', not 'dataset_name'
                "dataset_name": dataset.datasetName,  # Keep both for compatibility
                "training_file": dataset.trainingDataLocation,
                "test_file": dataset.testDataLocation,
                "training_samples": dataset.trainingSamples,
                "test_samples": dataset.testSamples,
                "total_transactions": dataset.trainingSamples
                + dataset.testSamples,  # Frontend expects this
                "categories": dataset.categories,
                "category_breakdown": dataset.categoryBreakdown or {},
                "date_from": dataset.dateFrom,
                "date_to": dataset.dateTo,
                "files_exist": True,  # MLEngine only returns datasets with existing files
                "created_at": created_at,
            }
            dataset_list.append(dataset_dict)

        logger.info(f"Found {len(dataset_list)} datasets")
        return success_response(f"Found {len(dataset_list)} datasets", dataset_list)

    except (InternalException, UnauthorizedException) as e:
        logger.error(f"Service error listing datasets: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list datasets: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list datasets: {str(e)}"
        ) from e


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    ml_engine: MLEngine = MLEngineDep,
):
    """
    Delete a training dataset.

    This endpoint removes a dataset and its associated files.
    """
    try:
        logger.info(f"Deleting dataset: {dataset_id}")

        # Delete dataset through MLEngine
        results = await ml_engine.deleteDatasets([dataset_id])

        if results and results[0]:
            return success_response(f"Dataset {dataset_id} deleted successfully")
        else:
            raise HTTPException(
                status_code=404, detail=f"Dataset {dataset_id} not found"
            )

    except NotFoundException as e:
        logger.error(f"Dataset not found: {dataset_id}")
        raise HTTPException(
            status_code=404, detail=f"Dataset {dataset_id} not found"
        ) from e
    except (InternalException, UnauthorizedException) as e:
        logger.error(f"Service error deleting dataset {dataset_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete dataset: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Error deleting dataset {dataset_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete dataset: {str(e)}"
        ) from e


@router.get("/datasets/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    ml_engine: MLEngine = MLEngineDep,
):
    """
    Get details of a specific training dataset.

    This endpoint returns detailed information about a dataset.
    """
    try:
        logger.info(f"Getting dataset: {dataset_id}")

        # Get specific dataset from MLEngine
        datasets = await ml_engine.getDatasets(datasetIds=[dataset_id])

        if not datasets:
            raise HTTPException(
                status_code=404, detail=f"Dataset {dataset_id} not found"
            )

        dataset = datasets[0]

        # Try to get created_at from metadata file
        created_at = None
        try:
            dataset_dir = os.path.dirname(dataset.trainingDataLocation)
            metadata_file = os.path.join(dataset_dir, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, encoding="utf-8") as f:
                    import json

                    metadata = json.load(f)
                    created_at = metadata.get("created_at")
        except Exception as e:
            logger.warning(
                f"Could not read metadata for dataset {dataset.datasetId}: {e}"
            )

        # Convert to API response format with additional file info
        dataset_dict = {
            "id": dataset.datasetId,
            "name": dataset.datasetName,  # Frontend expects 'name', not 'dataset_name'
            "dataset_name": dataset.datasetName,
            "training_file": dataset.trainingDataLocation,
            "test_file": dataset.testDataLocation,
            "training_samples": dataset.trainingSamples,
            "test_samples": dataset.testSamples,
            "total_transactions": dataset.trainingSamples + dataset.testSamples,
            "categories": dataset.categories,
            "category_breakdown": dataset.categoryBreakdown or {},
            "date_from": dataset.dateFrom,
            "date_to": dataset.dateTo,
            "files_exist": True,  # MLEngine only returns datasets with existing files
            "created_at": created_at,
        }

        # Add file sizes if they exist
        if os.path.exists(dataset.trainingDataLocation):
            dataset_dict["training_file_size"] = os.path.getsize(
                dataset.trainingDataLocation
            )
        if os.path.exists(dataset.testDataLocation):
            dataset_dict["test_file_size"] = os.path.getsize(dataset.testDataLocation)

        return success_response("Dataset found", dataset_dict)

    except NotFoundException as e:
        logger.error(f"Dataset not found: {dataset_id}")
        raise HTTPException(
            status_code=404, detail=f"Dataset {dataset_id} not found"
        ) from e
    except (InternalException, UnauthorizedException) as e:
        logger.error(f"Service error getting dataset {dataset_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get dataset: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Error getting dataset {dataset_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get dataset: {str(e)}"
        ) from e
