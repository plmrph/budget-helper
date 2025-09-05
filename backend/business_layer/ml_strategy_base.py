"""
Base ML strategy interface.

This module defines the abstract base class for ML model strategies
to avoid circular import issues.
"""

from abc import ABC, abstractmethod

from thrift_gen.entities.ttypes import ModelType
from thrift_gen.mlengine.ttypes import (
    CategoricalPredictionResult,
    ModelTrainingRequest,
)


class MLModelStrategy(ABC):
    """Abstract base class for ML model strategies."""

    @abstractmethod
    async def train_model(
        self,
        request: ModelTrainingRequest,
        training_data: list,
    ) -> dict:
        """Train a model with the given data."""
        pass

    @abstractmethod
    async def predict(
        self, model: dict, input_data: list
    ) -> list[CategoricalPredictionResult]:
        """Generate predictions using the trained model."""
        pass

    @abstractmethod
    def get_model_type(self) -> ModelType:
        """Get the model type this strategy handles."""
        pass

    @abstractmethod
    async def evaluate_model(self, model: dict, test_data_path: str) -> dict:
        """
        Evaluate a trained model on test data.

        Args:
            model: The trained model dictionary
            test_data_path: Path to the test data file

        Returns:
            Dictionary of evaluation metrics
        """
        # Default implementation returns empty metrics
        # Strategies can override this to provide actual evaluation
        pass
