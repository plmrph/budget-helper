"""Model classes for PXBlendSC-RF strategy."""

import logging
from collections import Counter

import numpy as np
import pandas as pd

from .utils import _check_cancellation

logger = logging.getLogger(__name__)


class RecencyFrequencyModel:
    """
    Model that predicts based on payee categorization history.

    Combines frequency (how often a payee goes to each category) with
    recency (recent categorizations matter more) to make predictions.
    """

    def __init__(
        self,
        recency_weight=0.7,
        frequency_weight=0.3,
        min_frequency=2,
        lookback_window=50,
        recency_window=5,
    ):
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.min_frequency = min_frequency
        self.lookback_window = lookback_window
        self.recency_window = recency_window
        self.payee_patterns = {}
        self.n_classes = 0

    def fit(self, X, y, dates=None):
        """
        Learn payee categorization patterns from training data.

        Args:
            X: DataFrame with PAYEE_NORM column
            y: Category labels (encoded)
            dates: Optional dates for recency weighting
        """
        self.n_classes = len(np.unique(y))
        self.payee_patterns = {}

        # If no dates provided, use sequential ordering
        if dates is None:
            dates = list(range(len(y)))

        # Build payee history
        for i, (payee, category, date) in enumerate(
            zip(X["PAYEE_NORM"], y, dates, strict=False)
        ):
            if pd.isna(payee) or payee == "":
                continue

            if payee not in self.payee_patterns:
                self.payee_patterns[payee] = []

            self.payee_patterns[payee].append((int(category), date, i))

        # Sort each payee's history by date and keep only recent transactions
        for payee in self.payee_patterns:
            history = sorted(self.payee_patterns[payee], key=lambda x: x[1])
            self.payee_patterns[payee] = history[-self.lookback_window :]

        logger.info(
            f"RecencyFrequency model: learned patterns for {len(self.payee_patterns)} payees"
        )
        return self

    def predict_proba(self, X):
        """
        Predict category probabilities based on payee history.

        Args:
            X: DataFrame with PAYEE_NORM column

        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        probas = []

        for payee in X["PAYEE_NORM"]:
            proba = self._get_payee_proba(payee)
            probas.append(proba)

        return np.array(probas)

    def _get_payee_proba(self, payee):
        """Get probability distribution for a specific payee."""
        if pd.isna(payee) or payee == "" or payee not in self.payee_patterns:
            # No history - return uniform distribution
            return np.ones(self.n_classes) / self.n_classes

        history = self.payee_patterns[payee]
        if len(history) == 0:
            return np.ones(self.n_classes) / self.n_classes

        # Calculate frequency scores
        categories = [cat for cat, date, idx in history]
        category_counts = Counter(categories)

        # Calculate recency scores (recent transactions matter more)
        recent_categories = [cat for cat, date, idx in history[-self.recency_window :]]
        recent_counts = Counter(recent_categories)

        # Build probability distribution
        scores = np.zeros(self.n_classes)

        for category, count in category_counts.items():
            if count >= self.min_frequency and category < self.n_classes:
                # Frequency component
                frequency_score = count / len(history)

                # Recency component
                recency_score = recent_counts.get(category, 0) / len(recent_categories)

                # Combined score
                combined_score = (
                    self.frequency_weight * frequency_score
                    + self.recency_weight * recency_score
                )

                scores[category] = combined_score

        # Normalize to probabilities
        if scores.sum() > 0:
            scores = scores / scores.sum()
        else:
            # Fallback to uniform if no valid patterns
            scores = np.ones(self.n_classes) / self.n_classes

        return scores


class CancellationCallback:
    """LightGBM callback that checks for asyncio task cancellation."""

    def __init__(self):
        self.check_interval = 10  # Check every 10 iterations

    def __call__(self, env):
        """Called by LightGBM during training."""
        if env.iteration % self.check_interval == 0:
            _check_cancellation()
        return False  # Don't stop training unless cancelled


class CancellableEstimator:
    """Wrapper for sklearn estimators that supports cancellation checks."""

    def __init__(self, estimator, check_interval=50):
        self.estimator = estimator
        self.check_interval = check_interval

    def fit(self, X, y, **kwargs):
        """Fit the estimator with periodic cancellation checks."""
        _check_cancellation()
        self.estimator.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        """Predict using the fitted estimator."""
        return self.estimator.predict(X)

    def predict_proba(self, X):
        """Predict probabilities using the fitted estimator."""
        return self.estimator.predict_proba(X)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped estimator."""
        if name == "estimator" or not hasattr(self, "estimator"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )
        return getattr(self.estimator, name)
