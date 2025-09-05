"""
Machine Learning related data models.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, validator


class ModelMetadata(BaseModel):
    """Model metadata for ML models."""

    ml_model_config: dict[str, Any] = {
        "protected_namespaces": (),
        "json_encoders": {datetime: lambda v: v.isoformat()},
        "json_schema_extra": {
            "example": {
                "name": "transaction_categorizer_v1",
                "created_at": "2024-01-15T10:30:00Z",
                "path": "/models/transaction_categorizer_v1.pkl",
                "performance_metrics": {
                    "accuracy": 0.85,
                    "precision": 0.83,
                    "recall": 0.82,
                    "f1_score": 0.825,
                },
                "transaction_count": 1500,
                "category_count": 25,
                "feature_count": 10,
                "ml_model_type": "classification",
                "version": "1.0",
            }
        },
    }

    name: str = Field(..., description="Model name")
    created_at: datetime = Field(..., description="When the model was created")
    path: str = Field(..., description="Path to the model file")
    performance_metrics: dict[str, Any] = Field(
        ..., description="Model performance metrics"
    )
    transaction_count: int | None = Field(
        None, description="Number of transactions used for training"
    )
    category_count: int | None = Field(
        None, description="Number of categories in the model"
    )
    feature_count: int | None = Field(
        None, description="Number of features in the model"
    )
    ml_model_type: str = Field(default="classification", description="Type of ML model")
    version: str = Field(default="1.0", description="Model version")

    @validator("performance_metrics")
    def validate_performance_metrics(cls, v):
        """Validate that performance metrics contain expected keys."""
        expected_keys = ["accuracy", "precision", "recall", "f1_score"]
        for key in expected_keys:
            if key not in v:
                raise ValueError(f"Performance metrics must contain {key}")
            if not isinstance(v[key], int | float):
                raise ValueError(f"{key} must be a number")
            if not 0 <= v[key] <= 1:
                raise ValueError(f"{key} must be between 0 and 1")
        return v


class Prediction(BaseModel):
    """Prediction result for transaction categorization."""

    ml_model_config: dict[str, Any] = {
        "protected_namespaces": (),
        "json_encoders": {datetime: lambda v: v.isoformat()},
        "json_schema_extra": {
            "example": {
                "transaction_id": "trans_123",
                "category_id": "category_456",
                "confidence": 0.92,
                "ml_model_name": "transaction_categorizer_v1",
                "created_at": "2024-01-15T10:30:00Z",
            }
        },
    }

    transaction_id: str = Field(
        ..., description="ID of the transaction being predicted"
    )
    category_id: str = Field(..., description="Predicted category ID")
    confidence: float = Field(..., description="Confidence score for the prediction")
    ml_model_name: str = Field(
        ..., description="Name of the model that made the prediction"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the prediction was made"
    )

    @validator("confidence")
    def validate_confidence(cls, v):
        """Validate confidence is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v
