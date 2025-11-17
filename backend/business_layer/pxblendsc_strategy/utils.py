"""Utility functions for PXBlendSC-RF strategy."""

import asyncio
import logging
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _check_cancellation():
    """Check if the current task has been cancelled."""
    try:
        current_task = asyncio.current_task()
        if current_task and current_task.cancelled():
            raise asyncio.CancelledError("Training cancelled")
    except RuntimeError:
        # Not running in an async context, continue normally
        pass


def label_prefixes(classes: list[str]) -> list[str]:
    """
    Extract hierarchical prefixes from class labels.

    For categories in format "Category: Subcategory", extracts "Category".
    For categories without colon, uses entire category name.

    Examples:
        "Monthly Spending: Groceries" -> "Monthly Spending"
        "Transportation: Gas" -> "Transportation"
        "Income" -> "Income"

    Args:
        classes: List of category names

    Returns:
        List of extracted prefixes
    """
    prefixes = []
    for cls in classes:
        cls_str = str(cls)
        if ":" in cls_str:
            prefix = cls_str.split(":", 1)[0].strip()
        else:
            prefix = cls_str.strip()
        prefixes.append(prefix)
    return prefixes


def pad_proba(model, proba: np.ndarray, n_classes: int) -> np.ndarray:
    """Pad probability matrix to full class space."""
    if proba.shape[1] == n_classes:
        return proba

    # Create full probability matrix
    full_proba = np.zeros((proba.shape[0], n_classes))

    # Get classes that the model was trained on
    if hasattr(model, "classes_"):
        model_classes = model.classes_
        for i, cls in enumerate(model_classes):
            if cls < n_classes:
                full_proba[:, cls] = proba[:, i]
    else:
        # Fallback: assume first classes
        min_cols = min(proba.shape[1], n_classes)
        full_proba[:, :min_cols] = proba[:, :min_cols]

    return full_proba


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        return [convert_numpy_types(item) for item in obj]
    return obj


def prepare_dataframe_columns(
    df: pd.DataFrame, text_cols: list[str], date_col: str, num_col: str
) -> pd.DataFrame:
    """Apply common data hygiene to DataFrame columns."""
    df = df.copy()

    # Ensure text columns exist and are strings
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # Ensure date column exists
    if date_col not in df.columns:
        df[date_col] = None

    # Ensure numeric column exists
    if num_col not in df.columns:
        df[num_col] = 0.0

    return df


def capped_ros_strategy(y: np.ndarray, cap_percentile: float) -> dict[int, int]:
    """Create improved ROS strategy with better class balancing."""
    cnt = Counter(y.tolist())
    total_samples = len(y)
    n_classes = len(cnt)

    # Calculate target samples per class
    if cap_percentile > 50:  # Conservative approach
        base_target = int(np.median(list(cnt.values())))
        cap = max(
            base_target, int(np.percentile(list(cnt.values()), cap_percentile * 0.8))
        )
    else:
        cap = int(np.percentile(list(cnt.values()), cap_percentile))

    # Minimum samples per class to avoid extreme overfitting
    min_samples = max(3, int(total_samples / (n_classes * 10)))

    strategy = {}
    for c, n in cnt.items():
        # Convert numpy types to native Python types
        c = int(c)
        n = int(n)

        if n < min_samples:
            # For very small classes, sample conservatively
            target = max(n, min(min_samples * 2, cap // 2))
            strategy[c] = target
        elif n < cap:
            # Conservative upsampling for underrepresented classes
            target = min(cap, int(n * 1.5))  # Max 1.5x original size (rounded down)
            strategy[c] = max(n, target)
        else:
            # Keep original size for well-represented classes
            strategy[c] = n

    return strategy
