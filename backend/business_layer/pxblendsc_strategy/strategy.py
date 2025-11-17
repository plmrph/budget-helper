"""Main strategy class for PXBlendSC-RF."""

import asyncio
import logging
import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

from configs import ConfigDefaults, get_config_service
from thrift_gen.entities.ttypes import ModelType
from thrift_gen.exceptions.ttypes import InternalException, ValidationException
from thrift_gen.mlengine.ttypes import CategoricalPredictionResult

from ..ml_strategy_base import MLModelStrategy
from .bundle import ModelBundle
from .training import train_model_bundle
from .training_utils import calculate_metrics
from .utils import convert_numpy_types

logger = logging.getLogger(__name__)


class PXBlendSCStrategy(MLModelStrategy):
    """PXBlendSC-RF ML strategy implementation."""

    def get_model_type(self) -> ModelType:
        return ModelType.PXBlendSC

    def _get_default_config(self) -> dict:
        """Get default PXBlendSC-RF configuration from ConfigDefaults."""
        return ConfigDefaults.PXBLENDSC_CONFIG.copy()

    async def train_model(self, request, training_data: list) -> dict:
        """Train a PXBlendSC-RF model."""
        logger.info(f"Training PXBlendSC-RF model: {request.modelCard.name}")

        if not training_data:
            raise ValidationException("No training data provided")

        # Get training data location
        training_data_location = request.trainingDataLocation
        if not os.path.exists(training_data_location):
            raise ValidationException(
                f"Training data file not found: {training_data_location}"
            )

        # Load training data
        train_df = pd.read_csv(training_data_location)
        logger.info(
            f"Loaded training data: {len(train_df)} rows, columns: {list(train_df.columns)}"
        )

        # Get configuration
        cfg = await self._get_configuration()
        
        # Check for LightGBM availability
        try:
            from lightgbm import LGBMClassifier
            HAS_LGBM = True
        except ImportError:
            HAS_LGBM = False
        
        cfg["models"]["use_lgbm"] = cfg["models"]["use_lgbm"] and HAS_LGBM

        # Override with request parameters
        self._apply_request_parameters(cfg, request.parameters)

        try:
            # Train the model bundle
            bundle, metrics = await asyncio.to_thread(train_model_bundle, cfg, train_df)
        except asyncio.CancelledError:
            logger.info(f"Training cancelled for model: {request.modelCard.name}")
            raise
        except Exception as e:
            # Add more detailed error logging
            logger.error(
                f"PXBlendSC-RF training error details: {type(e).__name__}: {str(e)}"
            )
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise InternalException(f"PXBlendSC-RF training failed: {str(e)}") from e

        # Save model
        temp_dir = tempfile.mkdtemp(prefix="pxblendsc_")
        bundle.save(temp_dir)

        # Create model metadata
        model = self._create_model_metadata(
            request, bundle, metrics, train_df, temp_dir
        )

        logger.info(
            f"PXBlendSC-RF training completed:\n"
            f"  Validation: F1={metrics['cv_macro_f1']:.4f}, "
            f"Acc={metrics['cv_accuracy']:.4f}, "
            f"Abstain={metrics['cv_abstain_rate']:.2%}"
        )

        return model

    async def predict(
        self, model: dict, input_data: list
    ) -> list[CategoricalPredictionResult]:
        """Generate predictions using PXBlendSC-RF model."""
        bundle_path = model.get("bundle_path")

        if not bundle_path or not os.path.exists(bundle_path):
            raise InternalException("PXBlendSC-RF model not found or path invalid")

        try:
            # Load the model bundle
            bundle = ModelBundle.load(bundle_path)

            # Convert input data to the expected format
            data = self._convert_input_data(input_data)

            if not data:
                return []

            # Get predictions with top 3 results
            predictions = bundle.predict(data, top_k=3)

            # Convert to Thrift format
            return self._convert_predictions_to_thrift(predictions)

        except Exception as e:
            logger.error(f"PXBlendSC-RF prediction error: {e}")
            raise InternalException(f"Prediction failed: {str(e)}") from e

    async def evaluate_model(self, model: dict, test_data_path: str) -> dict:
        """Evaluate a trained PXBlendSC-RF model on test data."""
        try:
            bundle_path = model.get("bundle_path")
            if not bundle_path or not os.path.exists(bundle_path):
                logger.warning(f"Model bundle not found at {bundle_path}")
                return {}

            if not os.path.exists(test_data_path):
                logger.warning(f"Test data not found at {test_data_path}")
                return {}

            # Load model bundle and test data
            bundle = ModelBundle.load(bundle_path)
            test_df = pd.read_csv(test_data_path)
            logger.info(f"Evaluating model on {len(test_df)} test samples")

            # Check for required column
            if "category_name" not in test_df.columns:
                logger.warning("No 'category_name' column found in test data")
                return {}

            # Convert test data to expected format and get predictions
            test_data = self._convert_test_data_for_evaluation(test_df)
            if not test_data:
                logger.warning("No valid test data after conversion")
                return {}

            # Get predictions
            predictions = bundle.predict(test_data)

            # Extract labels, predictions, and abstention flags
            y_true_labels = test_df["category_name"].values
            y_pred_labels = [pred["label"] for pred in predictions]
            abstained = np.array([pred.get("abstained", False) for pred in predictions])
            confidences = np.array([pred["confidence"] for pred in predictions])

            # Convert string labels to indices for metrics calculation
            # Use the model's label encoder to ensure consistency
            try:
                y_true = bundle.label_encoder.transform(y_true_labels)
                y_pred = bundle.label_encoder.transform(y_pred_labels)
            except ValueError as e:
                logger.warning(f"Label encoding issue during evaluation: {e}")
                # Some test labels might not be in training data
                # Filter to only labels that exist in both
                valid_mask = np.isin(y_true_labels, bundle.classes) & np.isin(
                    y_pred_labels, bundle.classes
                )
                if valid_mask.sum() == 0:
                    logger.warning("No valid labels found for evaluation")
                    return {}

                y_true_labels = y_true_labels[valid_mask]
                y_pred_labels = np.array(y_pred_labels)[valid_mask]
                abstained = abstained[valid_mask]
                confidences = confidences[valid_mask]

                y_true = bundle.label_encoder.transform(y_true_labels)
                y_pred = bundle.label_encoder.transform(y_pred_labels)

            test_metrics = calculate_metrics(y_true, y_pred, abstained, "test_")

            # Add additional test-specific metrics
            test_metrics.update(
                {
                    "test_avg_confidence": float(np.mean(confidences)),
                    "test_samples": int(len(test_df)),
                    "test_non_abstained_samples": int((~abstained).sum()),
                }
            )

            logger.info(
                f"Test evaluation completed: "
                f"Acc={test_metrics['test_accuracy']:.4f}, "
                f"F1={test_metrics['test_macro_f1']:.4f}, "
                f"Abstain={test_metrics['test_abstain_rate']:.2%}"
            )

            return test_metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}

    # Helper methods

    async def _get_configuration(self) -> dict:
        """Get configuration from service or defaults."""
        config_service = get_config_service()
        if config_service:
            try:
                cfg = await config_service.getPXBlendSCConfig()
                logger.info("Using PXBlendSC-RF configuration from Configs service")
                return cfg
            except Exception as e:
                logger.warning(
                    f"Failed to get PXBlendSC-RF config from service: {e}, using defaults"
                )

        logger.warning("Config service not available, using default configuration")
        return self._get_default_config()

    def _apply_request_parameters(self, cfg: dict, params: dict):
        """Apply request parameters to configuration."""
        if not params:
            return

        if "time_limit" in params:
            time_limit = int(params.get("time_limit", 600))
            if time_limit < 300:  # Less than 5 minutes
                cfg["models"]["lgbm_params"]["n_estimators"] = 50
                cfg["cv"]["n_folds"] = 2
            elif time_limit > 1800:  # More than 30 minutes
                cfg["models"]["lgbm_params"]["n_estimators"] = 200
                cfg["cv"]["n_folds"] = 5

    def _create_model_metadata(
        self, request, bundle, metrics, train_df, temp_dir
    ) -> dict:
        """Create model metadata dictionary."""
        metadata = {
            "name": request.modelCard.name,
            "type": ModelType._VALUES_TO_NAMES.get(ModelType.PXBlendSC).lower(),
            "bundle_path": temp_dir,
            "categories": bundle.classes,
            "feature_columns": [
                col for col in train_df.columns if col != "category_name"
            ],
            "training_samples": len(train_df),
            "trained_at": datetime.utcnow().isoformat(),
            "strategy": ModelType._VALUES_TO_NAMES.get(ModelType.PXBlendSC).lower(),
            "performance_metrics": {
                **{f"{k}": v for k, v in metrics.items()},
            },
            "training_params": bundle.cfg,
            "n_classes": len(bundle.classes),
            "has_lgbm": bundle.lgbm is not None,
            "has_svm": bundle.svm is not None,
        }
        # Convert any numpy types to avoid JSON serialization errors
        return convert_numpy_types(metadata)

    def _convert_input_data(self, input_data: list) -> list[dict]:
        """Convert Thrift input data to DataFrame format."""
        data = []
        for row_data in input_data:
            row_dict = {}
            # Extract values from Thrift FilterValue objects
            for i, value in enumerate(row_data):
                if i == 0:  # date
                    row_dict["date"] = (
                        value.stringValue
                        if hasattr(value, "stringValue") and value.stringValue
                        else ""
                    )
                elif i == 1:  # payee_name
                    row_dict["payee_name"] = (
                        value.stringValue
                        if hasattr(value, "stringValue") and value.stringValue
                        else ""
                    )
                elif i == 2:  # memo
                    row_dict["memo"] = (
                        value.stringValue
                        if hasattr(value, "stringValue") and value.stringValue
                        else ""
                    )
                elif i == 3:  # amount
                    if hasattr(value, "doubleValue") and value.doubleValue is not None:
                        row_dict["amount"] = float(value.doubleValue)
                    elif hasattr(value, "intValue") and value.intValue is not None:
                        row_dict["amount"] = float(value.intValue)
                    else:
                        row_dict["amount"] = 0.0
                elif i == 4:  # account_name
                    row_dict["account_name"] = (
                        value.stringValue
                        if hasattr(value, "stringValue") and value.stringValue
                        else ""
                    )
            data.append(row_dict)
        return data

    def _convert_test_data_for_evaluation(self, test_df: pd.DataFrame) -> list[dict]:
        """Convert test DataFrame to format expected by model prediction."""
        data = []
        for _, row in test_df.iterrows():
            row_dict = {
                "date": str(row.get("date", "")),
                "payee_name": str(row.get("payee_name", "")),
                "memo": str(row.get("memo", "")),
                "amount": float(row.get("amount", 0.0)),
                "account_name": str(row.get("account_name", "")),
            }
            data.append(row_dict)
        return data

    def _convert_predictions_to_thrift(
        self, predictions: list
    ) -> list[CategoricalPredictionResult]:
        """Convert predictions to Thrift format."""
        results = []
        for pred in predictions:
            # Handle abstention - still return top_k if available
            if pred.get("abstained", False):
                if "top_k" in pred and pred["top_k"]:
                    # Return top predictions even when abstaining, but with reduced confidence
                    for top_pred in pred["top_k"]:
                        results.append(
                            CategoricalPredictionResult(
                                predictedCategory=top_pred["label"],
                                confidence=min(
                                    top_pred["p"], 0.3
                                ),  # Cap abstained predictions at 0.3
                            )
                        )
                else:
                    results.append(
                        CategoricalPredictionResult(
                            predictedCategory=pred["label"],
                            confidence=0.1,  # Low confidence for abstained predictions
                        )
                    )
            else:
                # If we have top_k results, return all of them
                if "top_k" in pred and pred["top_k"]:
                    for top_pred in pred["top_k"]:
                        results.append(
                            CategoricalPredictionResult(
                                predictedCategory=top_pred["label"],
                                confidence=top_pred["p"],
                            )
                        )
                else:
                    # Fallback to single prediction
                    results.append(
                        CategoricalPredictionResult(
                            predictedCategory=pred["label"],
                            confidence=pred["confidence"],
                        )
                    )
        return results
