"""
PXBlendSC ML Strategy implementation.

This strategy implements a ML pipeline combining:
- LightGBM and SVM models with blending
- Prefix-based classification
- Payee and account memory priors
- Payee+amount pattern matching
- Adaptive thresholds for abstention
- Enhanced feature engineering (text, amount, date, cross-features)
"""

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from configs import ConfigDefaults, get_config_service
from thrift_gen.entities.ttypes import ModelType
from thrift_gen.exceptions.ttypes import InternalException, ValidationException
from thrift_gen.mlengine.ttypes import CategoricalPredictionResult

from .ml_strategy_base import MLModelStrategy

try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

GENERIC_NOISE_PATTERNS = [
    r"\bpos\b",
    r"\bach\b",
    r"\beft\b",
    r"\bxfer\b",
    r"\bp2p\b",
    r"\bdebit\b",
    r"\bcredit\b",
    r"\bvenmo\b",
    r"\bzelle\b",
    r"\bpaypal\b",
    r"\bstripe\b",
    r"\bsquare\b",
    r"\bcash\s*app\b",
    r"\bauth\b",
    r"\bcapture\b",
    r"\btrx\b",
    r"\btxn\b",
    r"\btransacti?on?\b",
    r"\border\s*id\b",
    r"\bref\w*\b",
    r"\bconf\w*\b",
    r"\bwww\.[a-z0-9\.\-_/]+\b",
    r"\bhttps?://[a-z0-9\.\-_/]+\b",
]

GENERIC_NUM_LOCATION_PATTERNS = [
    r"[#*]{0,2}\d{3,}",
    r"\bno\.\s*\d+\b",
    r"\bstore\s*\d+\b",
    r"\blocation\s*\d+\b",
    r"\bunit\s*\d+\b",
    r"\bapt\s*\d+\b",
    r"\b\d{1,5}\s+[a-z]+\b",
    r"\b[a-z]+,\s*[a-z]{2}\b",
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


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
    """Extract prefixes from class labels (first 3 characters)."""
    return [str(cls)[:3] if len(str(cls)) >= 3 else str(cls) for cls in classes]


def make_generic_normalizer(use_generic_noise: bool, alias_map: dict):
    """Create a generic text normalizer (returns a picklable class instance)."""
    return GenericNormalizer(use_generic_noise, alias_map)


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
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


# =============================================================================
# DATA PREPARATION UTILITIES
# =============================================================================


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
            # Moderate upsampling for underrepresented classes
            target = min(cap, n * 3)  # Max 3x original size
            strategy[c] = max(n, target)
        else:
            # Keep original size for well-represented classes
            strategy[c] = n

    return strategy


# =============================================================================
# CANCELLATION SUPPORT CLASSES
# =============================================================================


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


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================


class GenericNormalizer:
    """Picklable text normalizer for payee names."""

    def __init__(self, use_generic_noise: bool = True, alias_map: dict = None):
        self.use_generic_noise = use_generic_noise
        self.alias_map = alias_map or {}

        # Compile regex patterns
        if use_generic_noise:
            noise_patterns = list(GENERIC_NOISE_PATTERNS) + list(
                GENERIC_NUM_LOCATION_PATTERNS
            )
            self.compiled_patterns = [
                re.compile(pat, re.IGNORECASE) for pat in noise_patterns
            ]
        else:
            self.compiled_patterns = []

    def __call__(self, s: str) -> str:
        """Normalize a string."""
        if not isinstance(s, str):
            return ""

        s = s.lower()

        # Apply noise pattern removal
        if self.use_generic_noise:
            for rx in self.compiled_patterns:
                s = rx.sub(" ", s)

        # Remove special characters
        s = re.sub(r"[\|\[\]\(\)_~`^=:;@<>\"" "']", " ", s)
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()

        # Apply alias replacements
        for pat, repl in self.alias_map.items():
            try:
                s = re.sub(pat, repl, s)
            except re.error:
                if s == pat:
                    s = repl

        return s


# =============================================================================
# FEATURE ENGINEERING CLASSES
# =============================================================================


class AddCombinedText(BaseEstimator, TransformerMixin):
    """Combine text columns (payee, memo, account, etc.) and normalize them."""

    def __init__(self, text_cols: list[str], normalizer):
        self.text_cols = text_cols or []
        self.normalizer = normalizer

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        # Ensure text columns exist
        for c in self.text_cols:
            if c not in X.columns:
                X[c] = ""

        payee_raw = (
            X[self.text_cols[0]].fillna("").astype(str)
            if self.text_cols
            else pd.Series([""] * len(X))
        )
        memo_raw = (
            X[self.text_cols[1]].fillna("").astype(str)
            if len(self.text_cols) > 1
            else pd.Series([""] * len(X))
        )
        account_raw = (
            X[self.text_cols[2]].fillna("").astype(str)
            if len(self.text_cols) > 2
            else pd.Series([""] * len(X))
        )

        X["PAYEE_NORM"] = payee_raw.map(self.normalizer)
        X["ACCOUNT_NORM"] = account_raw.map(self.normalizer)
        X["TEXT_ALL"] = (payee_raw + " " + memo_raw + " " + account_raw).map(
            self.normalizer
        )
        return X


class AmountFeaturizer(BaseEstimator, TransformerMixin):
    """Extract features from amount column with enhanced amount binning and per-payee percentiles."""

    def __init__(self, amount_cols: list[str]):
        self.amount_cols = amount_cols
        self.payee_amount_percentiles = {}
        self.global_amount_bins = None
        self.amount_percentiles = [0.25, 0.5, 0.75, 0.9]
        self.min_samples_for_payee_bins = 5

    def fit(self, X, y=None):
        # Learn per-payee amount distributions
        if (
            self.amount_cols
            and self.amount_cols[0] in X.columns
            and "PAYEE_NORM" in X.columns
        ):
            amount_col = self.amount_cols[0]
            amounts = np.abs(X[amount_col].fillna(0.0))

            # Create global bins as fallback
            self.global_amount_bins = (
                np.percentile(amounts[amounts > 0], [25, 50, 75, 90])
                if (amounts > 0).sum() > 0
                else np.array([10, 50, 100, 500])
            )

            # Create per-payee bins
            for payee in X["PAYEE_NORM"].unique():
                payee_amounts = amounts[X["PAYEE_NORM"] == payee]
                payee_amounts = payee_amounts[payee_amounts > 0]

                if len(payee_amounts) >= self.min_samples_for_payee_bins:
                    self.payee_amount_percentiles[payee] = np.percentile(
                        payee_amounts, [25, 50, 75, 90]
                    )

        return self

    def transform(self, X):
        features = []
        for col in self.amount_cols:
            if col in X.columns:
                amounts = X[col].fillna(0.0)
                abs_amounts = np.abs(amounts)

                # Basic amount features
                amount_features = [
                    amounts.values,  # raw amount (with sign)
                    abs_amounts.values,  # absolute amount
                    np.log1p(abs_amounts).values,  # log amount
                    (amounts > 0).astype(int).values,  # is_positive
                    (amounts == 0).astype(int).values,  # is_zero
                ]

                # Add per-payee percentile features
                if "PAYEE_NORM" in X.columns:
                    payee_features = self._get_payee_percentile_features(X, abs_amounts)
                    amount_features.extend(payee_features)

                features.extend(amount_features)

        return np.column_stack(features) if features else np.zeros((len(X), 1))

    def _get_payee_percentile_features(self, X, abs_amounts):
        """Generate per-payee percentile features."""
        n_samples = len(X)
        payee_features = []

        for i in range(n_samples):
            payee = X["PAYEE_NORM"].iloc[i] if "PAYEE_NORM" in X.columns else ""
            amount = (
                abs_amounts.iloc[i] if hasattr(abs_amounts, "iloc") else abs_amounts[i]
            )

            # Get percentiles for this payee (or global fallback)
            if payee in self.payee_amount_percentiles:
                percentiles = self.payee_amount_percentiles[payee]
            else:
                percentiles = self.global_amount_bins

            # Create percentile indicator features
            features_for_sample = [
                float(amount <= percentiles[0]),  # below_25th
                float(amount <= percentiles[1]),  # below_median
                float(amount <= percentiles[2]),  # below_75th
                float(amount <= percentiles[3]),  # below_90th
                float(amount > percentiles[3]),  # above_90th
            ]

            payee_features.append(features_for_sample)

        # Convert to numpy array and transpose to get features as columns
        payee_features_array = np.array(payee_features)
        return [
            payee_features_array[:, i] for i in range(payee_features_array.shape[1])
        ]


class DateFeaturizer(BaseEstimator, TransformerMixin):
    """Extract enhanced features from date column."""

    def __init__(self, date_col: str):
        self.date_col = date_col

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        if self.date_col not in X.columns:
            return np.zeros((len(X), 18))

        dates = pd.to_datetime(X[self.date_col], errors="coerce")
        current_year = datetime.now().year

        features = np.column_stack(
            [
                # Basic date features
                dates.dt.dayofweek.fillna(0).values,
                dates.dt.day.fillna(1).values,
                dates.dt.month.fillna(1).values,
                dates.dt.year.fillna(current_year).values % 100,
                # Enhanced date features
                dates.dt.quarter.fillna(1).values,
                (dates.dt.dayofweek >= 5).astype(int),  # is_weekend
                (dates.dt.day <= 7).astype(int),  # first_week_of_month
                (dates.dt.day >= 25).astype(int),  # last_week_of_month
                (dates.dt.month.isin([12, 1, 2])).astype(int),  # winter_months
                (dates.dt.month.isin([6, 7, 8])).astype(int),  # summer_months
                (dates.dt.month.isin([11, 12])).astype(int),  # holiday_season
                # Individual day features
                (dates.dt.dayofweek == 0).astype(int),  # is_monday
                (dates.dt.dayofweek == 1).astype(int),  # is_tuesday
                (dates.dt.dayofweek == 2).astype(int),  # is_wednesday
                (dates.dt.dayofweek == 3).astype(int),  # is_thursday
                (dates.dt.dayofweek == 4).astype(int),  # is_friday
                (dates.dt.dayofweek == 5).astype(int),  # is_saturday
                (dates.dt.dayofweek == 6).astype(int),  # is_sunday
            ]
        )
        return features


class CrossFeaturizer(BaseEstimator, TransformerMixin):
    """Create enhanced cross features between payee, amount, and date."""

    def __init__(
        self,
        payee_col: str,
        amount_col: str,
        date_col: str,
        payee_min_count: int = 5,
        quantiles: tuple = (0.25, 0.5, 0.75),
        sign_bins: str = "two_sided",
        emit_unaries: bool = True,
        emit_pairs: bool = True,
        emit_triples: bool = True,
    ):
        self.payee_col = payee_col
        self.amount_col = amount_col
        self.date_col = date_col
        self.payee_min_count = payee_min_count
        self.quantiles = quantiles
        self.sign_bins = sign_bins
        self.emit_unaries = emit_unaries
        self.emit_pairs = emit_pairs
        self.emit_triples = emit_triples
        self.payee_counts = {}
        self.amount_bins = None
        self.payee_amount_stats = {}

    def fit(self, X, y=None):
        # Count payees
        if self.payee_col in X.columns:
            self.payee_counts = X[self.payee_col].value_counts().to_dict()

        # Create amount bins
        if self.amount_col in X.columns:
            amounts = np.abs(X[self.amount_col].fillna(0.0))
            self.amount_bins = np.quantile(amounts[amounts > 0], list(self.quantiles))

            # Learn payee-specific amount patterns
            if self.payee_col in X.columns:
                for payee in X[self.payee_col].unique():
                    payee_amounts = amounts[X[self.payee_col] == payee]
                    if len(payee_amounts) >= 3:
                        self.payee_amount_stats[payee] = {
                            "mean": payee_amounts.mean(),
                            "std": payee_amounts.std(),
                            "median": payee_amounts.median(),
                        }

        return self

    def transform(self, X):
        features = []
        common_payees = {
            p for p, c in self.payee_counts.items() if c >= self.payee_min_count
        }

        for _idx, row in X.iterrows():
            row_features = []
            payee = row.get(self.payee_col, "")
            amount = abs(float(row.get(self.amount_col, 0.0)))
            date_str = str(row.get(self.date_col, ""))

            # Unary features
            if self.emit_unaries:
                row_features.extend(
                    [
                        f"payee:{payee}" if payee in common_payees else "payee:OTHER",
                        f"amount_bin:{self._get_amount_bin(amount)}",
                        f"month:{pd.to_datetime(date_str, errors='coerce').month if date_str else 'UNK'}",
                    ]
                )

            # Enhanced pair features
            if self.emit_pairs and payee in common_payees:
                row_features.extend(
                    [
                        f"payee_amount:{payee}:{self._get_amount_bin(amount)}",
                        f"payee_month:{payee}:{pd.to_datetime(date_str, errors='coerce').month if date_str else 'UNK'}",
                    ]
                )

            # Triple features for general recurring patterns
            if self.emit_triples and payee in common_payees:
                month = (
                    pd.to_datetime(date_str, errors="coerce").month
                    if date_str
                    else "UNK"
                )
                row_features.append(
                    f"payee_amount_month:{payee}:{self._get_amount_bin(amount)}:{month}"
                )

            features.append(row_features)

        return features

    def _get_amount_bin(self, amount):
        """Get amount bin for given amount."""
        if self.amount_bins is None or len(self.amount_bins) == 0:
            return "UNKNOWN"

        for i, threshold in enumerate(self.amount_bins):
            if amount <= threshold:
                return f"BIN_{i}"
        return f"BIN_{len(self.amount_bins)}"


# =============================================================================
# FEATURE PIPELINE CONSTRUCTION
# =============================================================================


def build_feature_pipeline(
    cfg: dict[str, Any], learned_params: dict = None
) -> Pipeline:
    """Build feature pipeline with optional pre-computed learned parameters."""
    MAXF_WORD = int(cfg["features"]["tfidf_word_max_features"])
    MAXF_CHAR = int(cfg["features"]["tfidf_char_max_features"])
    HC = cfg["features"]["hashed_cross"]

    # Create normalizer from config
    alias_map = {}
    use_generic_noise = bool(cfg["alias"].get("generic_noise", True))
    normalizer = make_generic_normalizer(use_generic_noise, alias_map)

    # TfidfVectorizers with optional pre-computed vocabularies
    word_tfidf_params = {
        "lowercase": True,
        "strip_accents": "unicode",
        "analyzer": "word",
        "ngram_range": (1, 2),
        "min_df": 1,
        "sublinear_tf": True,
        "max_features": MAXF_WORD,
    }
    char_tfidf_params = {
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
        "min_df": 1,
        "max_features": MAXF_CHAR,
    }

    if learned_params:
        word_tfidf_params["vocabulary"] = learned_params["word_vocab"]
        char_tfidf_params["vocabulary"] = learned_params["char_vocab"]

    word_tfidf = TfidfVectorizer(**word_tfidf_params)
    char_tfidf = TfidfVectorizer(**char_tfidf_params)

    # AmountFeaturizer with optional pre-computed parameters
    amount_featurizer = AmountFeaturizer(cfg["columns"]["NUM_COLS"])
    amount_scaler = StandardScaler(with_mean=False)

    if learned_params:
        amount_featurizer.global_amount_bins = learned_params["amount_global_bins"]
        amount_featurizer.payee_amount_percentiles = learned_params[
            "amount_payee_percentiles"
        ]

        if learned_params["amount_scaler_mean"] is not None:
            amount_scaler.mean_ = learned_params["amount_scaler_mean"]
            amount_scaler.scale_ = learned_params["amount_scaler_scale"]
            amount_scaler.n_features_in_ = len(learned_params["amount_scaler_mean"])

    num_pipe = Pipeline(
        [
            ("amount_fe", amount_featurizer),
            ("sc", amount_scaler),
        ]
    )

    # DateFeaturizer and scaler
    date_featurizer = DateFeaturizer(cfg["columns"]["DATE_COL"])
    date_scaler = StandardScaler(with_mean=False)

    if learned_params and learned_params["date_scaler_mean"] is not None:
        date_scaler.mean_ = learned_params["date_scaler_mean"]
        date_scaler.scale_ = learned_params["date_scaler_scale"]
        date_scaler.n_features_in_ = len(learned_params["date_scaler_mean"])

    date_pipe = Pipeline(
        [
            ("date_fe", date_featurizer),
            ("sc", date_scaler),
        ]
    )

    # CrossFeaturizer with optional pre-computed parameters
    cross_featurizer = CrossFeaturizer(
        payee_col="PAYEE_NORM",
        amount_col=cfg["columns"]["NUM_COLS"][0],
        date_col=cfg["columns"]["DATE_COL"],
        payee_min_count=int(HC["payee_min_count"]),
        quantiles=tuple(float(x) for x in HC["quantiles"]),
        sign_bins=HC.get("sign_bins", "two_sided"),
        emit_unaries=bool(HC.get("emit_unaries", True)),
        emit_pairs=bool(HC.get("emit_pairs", True)),
        emit_triples=bool(HC.get("emit_triples", True)),
    )

    if learned_params:
        cross_featurizer.payee_counts = learned_params["cross_payee_counts"]
        cross_featurizer.amount_bins = learned_params["cross_amount_bins"]
        cross_featurizer.payee_amount_stats = learned_params["cross_payee_amount_stats"]

    cross_pipe = Pipeline(
        [
            ("cross", cross_featurizer),
            (
                "hash",
                FeatureHasher(
                    n_features=int(HC["n_features"]),
                    input_type="string",
                    alternate_sign=False,
                ),
            ),
        ]
    )

    col_tfm = ColumnTransformer(
        transformers=[
            ("tfidf_word", word_tfidf, "TEXT_ALL"),
            ("tfidf_char", char_tfidf, "TEXT_ALL"),
            ("num", num_pipe, [cfg["columns"]["NUM_COLS"][0]]),
            ("date", date_pipe, [cfg["columns"]["DATE_COL"]]),
            (
                "cross_hash",
                cross_pipe,
                [
                    "PAYEE_NORM",
                    "ACCOUNT_NORM",
                    cfg["columns"]["NUM_COLS"][0],
                    cfg["columns"]["DATE_COL"],
                ],
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    feature_pipe = Pipeline(
        [
            ("add_text", AddCombinedText(cfg["columns"]["TEXT_COLS"], normalizer)),
            ("features", col_tfm),
        ]
    )
    return feature_pipe


# =============================================================================
# MEMORY BUILDING FUNCTIONS
# =============================================================================


def build_payee_memory(
    X_norm: pd.DataFrame, y: np.ndarray, n_min: int = 3
) -> dict[str, tuple[int, float, int]]:
    """Build payee memory for priors with enhanced pattern detection."""
    memory = {}

    if "PAYEE_NORM" not in X_norm.columns:
        return memory

    for payee in X_norm["PAYEE_NORM"].unique():
        if pd.isna(payee) or payee == "":
            continue

        mask = X_norm["PAYEE_NORM"] == payee
        payee_labels = y[mask]
        if len(payee_labels) < n_min:
            continue

        counts = Counter(payee_labels)
        label_idx, count = counts.most_common(1)[0]
        share = count / len(payee_labels)

        # Convert numpy types to native Python types for JSON serialization
        memory[payee] = (int(label_idx), float(share), int(len(payee_labels)))

        if share == 1.0:
            logger.info(
                f"Perfect pattern detected: '{payee}' -> class {label_idx} "
                f"(100% over {len(payee_labels)} samples)"
            )

    logger.info(
        f"Built payee memory: {len(memory)} payees, "
        f"{sum(1 for _, (_, share, total) in memory.items() if share == 1.0)} perfect patterns"
    )
    return memory


def build_payee_amount_patterns(
    X_norm: pd.DataFrame, y: np.ndarray, amount_col: str, n_min: int = 2
) -> dict[str, tuple[int, float, int]]:
    """Build payee+amount combination patterns for ultra-precise matching."""
    memory = {}

    if "PAYEE_NORM" not in X_norm.columns or amount_col not in X_norm.columns:
        return memory

    # Create payee+amount combinations
    payee_amount_combos = (
        X_norm["PAYEE_NORM"].astype(str) + "|" + X_norm[amount_col].round(2).astype(str)
    )

    for combo in payee_amount_combos.unique():
        if pd.isna(combo) or combo == "":
            continue

        mask = payee_amount_combos == combo
        combo_labels = y[mask]
        if len(combo_labels) < n_min:
            continue

        counts = Counter(combo_labels)
        label_idx, count = counts.most_common(1)[0]
        share = count / len(combo_labels)

        # Convert numpy types to native Python types for JSON serialization
        memory[combo] = (int(label_idx), float(share), int(len(combo_labels)))

        if share >= 0.8:  # Strong pattern
            payee, amount = combo.split("|", 1)
            logger.info(
                f"Strong payee+amount pattern: '{payee}' + ${amount} -> class {label_idx} "
                f"({share:.1%} over {len(combo_labels)} samples)"
            )

    logger.info(f"Built payee+amount memory: {len(memory)} combinations")
    return memory


def build_account_memory(
    X_norm: pd.DataFrame, y: np.ndarray, n_min: int = 3
) -> dict[str, tuple[int, float, int]]:
    """Build account memory for priors."""
    memory = {}

    if "ACCOUNT_NORM" not in X_norm.columns:
        return memory

    for acct in X_norm["ACCOUNT_NORM"].unique():
        if pd.isna(acct) or acct == "":
            continue

        mask = X_norm["ACCOUNT_NORM"] == acct
        acct_labels = y[mask]

        if len(acct_labels) >= n_min:
            counts = Counter(acct_labels)
            most_common = counts.most_common(1)[0]
            label_idx, count = most_common
            share = count / len(acct_labels)
            # Convert numpy types to native Python types for JSON serialization
            memory[acct] = (int(label_idx), float(share), int(len(acct_labels)))

    return memory


# =============================================================================
# PROBABILITY CALCULATION AND THRESHOLD FUNCTIONS
# =============================================================================


def apply_prefix_and_priors(
    proba: np.ndarray,
    pref_str: np.ndarray,
    X_norm: pd.DataFrame,
    prefix_to_ids: dict[str, np.ndarray],
    mem_payee: dict,
    mem_account: dict | None = None,
    mem_payee_amount: dict | None = None,
    amount_col: str = "amount",
) -> np.ndarray:
    """Apply prefix constraints and prior knowledge with strong pattern detection."""
    proba_adj = proba.copy()

    # Apply prefix constraints
    for i, prefix in enumerate(pref_str):
        if prefix not in prefix_to_ids:
            continue
        allowed_ids = prefix_to_ids[prefix]
        mask = np.ones(proba_adj.shape[1], dtype=bool)
        mask[allowed_ids] = False
        proba_adj[i, mask] = 0.0
        # Renormalize
        row_sum = proba_adj[i].sum()
        if row_sum > 0:
            proba_adj[i] /= row_sum

    # Apply payee+amount patterns FIRST (highest priority)
    if (
        mem_payee_amount
        and "PAYEE_NORM" in X_norm.columns
        and amount_col in X_norm.columns
    ):
        for i in range(len(X_norm)):
            payee = X_norm["PAYEE_NORM"].iloc[i]
            amount = X_norm[amount_col].iloc[i]
            combo = f"{payee}|{round(float(amount), 2)}"
            if combo not in mem_payee_amount:
                continue
            label_idx, share, count = mem_payee_amount[combo]
            if share < 0.9 or count < 3:
                continue
            proba_adj[i] = 0.0
            proba_adj[i, label_idx] = 1.0

    # Apply payee memory with strong priors for perfect patterns
    if "PAYEE_NORM" in X_norm.columns:
        for i, payee in enumerate(X_norm["PAYEE_NORM"]):
            if payee not in mem_payee:
                continue
            label_idx, share, count = mem_payee[payee]
            if share == 1.0 and count >= 5:  # Perfect pattern with sufficient support
                proba_adj[i] = 0.0
                proba_adj[i, label_idx] = 1.0
            elif share >= 0.8 and count >= 3:  # Strong pattern
                boost = min(0.7, share * 0.5)  # Cap boost at 0.7
                proba_adj[i, label_idx] = min(1.0, proba_adj[i, label_idx] + boost)
                # Renormalize
                proba_adj[i] /= proba_adj[i].sum()

    # Apply account memory (similar pattern for accounts)
    if mem_account and "ACCOUNT_NORM" in X_norm.columns:
        for i, acct in enumerate(X_norm["ACCOUNT_NORM"]):
            if acct not in mem_account:
                continue
            label_idx, share, count = mem_account[acct]
            if share < 0.8 or count < 3:
                continue
            boost = min(0.3, share * 0.2)  # Smaller boost for accounts
            proba_adj[i, label_idx] = min(1.0, proba_adj[i, label_idx] + boost)
            # Renormalize
            proba_adj[i] /= proba_adj[i].sum()

    return proba_adj


def apply_thresholds(
    proba: np.ndarray,
    pref_str: np.ndarray,
    thr_global: float,
    thr_per_prefix: dict[str, float],
    thr_per_class: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply thresholds for abstention."""
    y_hat = proba.argmax(axis=1)
    max_proba = proba.max(axis=1)
    abstain = np.zeros(len(proba), dtype=bool)

    # Global threshold
    abstain |= max_proba < thr_global

    # Per-prefix thresholds
    for i, prefix in enumerate(pref_str):
        if prefix in thr_per_prefix:
            abstain[i] |= max_proba[i] < thr_per_prefix[prefix]

    # Per-class thresholds for tail classes
    if thr_per_class is not None:
        for i in range(len(y_hat)):
            class_idx = y_hat[i]
            if class_idx < len(thr_per_class) and not np.isnan(
                thr_per_class[class_idx]
            ):
                abstain[i] |= max_proba[i] < thr_per_class[class_idx]

    return y_hat, abstain


def pick_best_threshold(
    y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray
) -> tuple[float, float]:
    """Pick best global threshold based on F1 score."""
    best_f1 = 0.0
    best_threshold = thresholds[0]

    for threshold in thresholds:
        y_pred = proba.argmax(axis=1)
        mask = proba.max(axis=1) >= threshold

        if mask.sum() > 0:
            f1 = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    return float(best_threshold), float(best_f1)


def pick_per_prefix_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    pref_str: np.ndarray,
    thresholds: np.ndarray,
    init: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Pick per-prefix thresholds."""
    prefix_thresholds = init.copy()
    prefix_scores = {}

    for prefix in np.unique(pref_str):
        mask = pref_str == prefix
        if mask.sum() < 10:
            continue

        prefix_y_true = y_true[mask]
        prefix_proba = proba[mask]

        best_f1 = 0.0
        best_threshold = thresholds[0]

        for threshold in thresholds:
            pred_mask = prefix_proba.max(axis=1) >= threshold
            if pred_mask.sum() > 0:
                y_pred = prefix_proba.argmax(axis=1)
                f1 = f1_score(
                    prefix_y_true[pred_mask],
                    y_pred[pred_mask],
                    average="macro",
                    zero_division=0,
                )
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

        prefix_thresholds[prefix] = float(best_threshold)
        prefix_scores[prefix] = float(best_f1)

    return prefix_thresholds, prefix_scores


def pick_tail_class_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_classes: int,
    tail_classes: set,
    thresholds: np.ndarray,
) -> np.ndarray | None:
    """Pick thresholds for tail classes."""
    if not tail_classes:
        return None

    class_thresholds = np.full(n_classes, 0.0)

    for class_idx in tail_classes:
        if class_idx >= n_classes:
            continue

        class_mask = y_true == class_idx
        if class_mask.sum() < 5:
            continue

        class_proba = proba[class_mask, class_idx]

        best_f1 = 0.0
        best_threshold = thresholds[0]

        for threshold in thresholds:
            pred_mask = class_proba >= threshold
            if pred_mask.sum() > 0:
                y_true_class = np.ones(pred_mask.sum())
                y_pred_class = np.ones(pred_mask.sum())
                f1 = f1_score(
                    y_true_class, y_pred_class, average="macro", zero_division=0
                )
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

        class_thresholds[class_idx] = float(best_threshold)

    return class_thresholds


# =============================================================================
# MODEL BUNDLE CLASS
# =============================================================================


class ModelBundle:
    """Complete model bundle for PXBlendSC strategy."""

    def __init__(
        self,
        *,
        cfg,
        feature_pipe,
        lgbm_model,
        svm_model,
        lgbm_weight,
        prefix_clf,
        prefix_label_encoder,
        label_encoder,
        thresholds,
        prefix_to_ids,
        classes,
        mem_payee,
        mem_account=None,
        mem_payee_amount=None,
    ):
        self.cfg = cfg
        self.feature_pipe = feature_pipe
        self.lgbm = lgbm_model
        self.svm = svm_model
        self.lgbm_weight = float(lgbm_weight)
        self.prefix_clf = prefix_clf
        self.prefix_le = prefix_label_encoder
        self.label_encoder = label_encoder
        self.thresholds = thresholds
        self.prefix_to_ids = {k: list(v) for k, v in prefix_to_ids.items()}
        self.classes = list(classes)
        self.mem_payee = mem_payee
        self.mem_account = mem_account or {}
        self.mem_payee_amount = mem_payee_amount or {}

    def save(self, folder: str):
        """Save model bundle to disk."""
        os.makedirs(folder, exist_ok=True)

        # Save sklearn components
        dump(self.feature_pipe, os.path.join(folder, "feature_pipe.joblib"))
        dump(self.prefix_clf, os.path.join(folder, "prefix_clf.joblib"))
        dump(self.prefix_le, os.path.join(folder, "prefix_label_encoder.joblib"))
        dump(self.label_encoder, os.path.join(folder, "label_encoder.joblib"))

        if self.lgbm is not None:
            dump(self.lgbm, os.path.join(folder, "lgbm.joblib"))
        if self.svm is not None:
            dump(self.svm, os.path.join(folder, "svm.joblib"))

        # Save artifact with full configuration
        artifact = {
            "version": "px-blend-sc:1",
            "cfg": self.cfg,
            "label_encoder_classes": self.label_encoder.classes_.tolist(),
            "feature_pipe": "feature_pipe.joblib",
            "prefix_clf": "prefix_clf.joblib",
            "lgbm": "lgbm.joblib" if self.lgbm is not None else None,
            "svm": "svm.joblib" if self.svm is not None else None,
            "mem_payee": self.mem_payee,
            "mem_account": self.mem_account,
            "mem_payee_amount": self.mem_payee_amount,
            "thresholds": {
                "global": float(self.thresholds["global"]),
                "per_prefix": {
                    k: float(v) for k, v in self.thresholds["per_prefix"].items()
                },
                "per_class": (
                    np.where(
                        np.isnan(self.thresholds["per_class_tail_only"]),
                        None,
                        self.thresholds["per_class_tail_only"],
                    ).tolist()
                    if self.thresholds.get("per_class_tail_only") is not None
                    else None
                ),
            },
            "classes": self.classes,
            "lgbm_weight": self.lgbm_weight,
            "prefix_to_ids": {
                k: list(map(int, v)) for k, v in self.prefix_to_ids.items()
            },
        }

        # Convert any remaining numpy types to avoid JSON serialization errors
        artifact = convert_numpy_types(artifact)

        with open(os.path.join(folder, "artifact.json"), "w") as f:
            json.dump(artifact, f, indent=2)

    @staticmethod
    def load(folder: str):
        """Load model bundle from disk."""
        with open(os.path.join(folder, "artifact.json")) as f:
            artifact = json.load(f)

        # Load components
        feature_pipe = load(os.path.join(folder, artifact["feature_pipe"]))
        prefix_clf = load(os.path.join(folder, artifact["prefix_clf"]))
        prefix_le = load(os.path.join(folder, "prefix_label_encoder.joblib"))
        label_encoder = load(os.path.join(folder, "label_encoder.joblib"))

        lgbm_model = None
        if artifact["lgbm"] is not None:
            lgbm_model = load(os.path.join(folder, artifact["lgbm"]))

        svm_model = None
        if artifact["svm"] is not None:
            svm_model = load(os.path.join(folder, artifact["svm"]))

        return ModelBundle(
            cfg=artifact["cfg"],
            feature_pipe=feature_pipe,
            lgbm_model=lgbm_model,
            svm_model=svm_model,
            lgbm_weight=artifact["lgbm_weight"],
            prefix_clf=prefix_clf,
            prefix_label_encoder=prefix_le,
            label_encoder=label_encoder,
            thresholds=artifact["thresholds"],
            prefix_to_ids=artifact["prefix_to_ids"],
            classes=artifact["classes"],
            mem_payee=artifact.get(
                "mem_payee", artifact.get("mem_vendor", {})
            ),  # Backward compatibility
            mem_account=artifact.get("mem_account", {}),
            mem_payee_amount=artifact.get("mem_payee_amount", {}),
        )

    def predict(
        self, raw_rows: list[dict], return_proba: bool = False, top_k: int = 1
    ) -> list[dict]:
        """Generate predictions."""
        df = pd.DataFrame(raw_rows).copy()

        # Apply data hygiene
        text_cols = self.cfg["columns"]["TEXT_COLS"]
        num_col = self.cfg["columns"]["NUM_COLS"][0]
        date_col = self.cfg["columns"]["DATE_COL"]

        df = prepare_dataframe_columns(df, text_cols, date_col, num_col)

        # Extract features and get predictions
        Xfeats = self.feature_pipe.transform(df)
        pref_idx = self.prefix_clf.predict(Xfeats)
        pref_str = self.prefix_le.inverse_transform(pref_idx)

        # Get base probabilities and blend
        proba = self._get_blended_probabilities(Xfeats)

        # Apply prefix and priors
        x_norm = self.feature_pipe.named_steps["add_text"].transform(df.copy())
        proba = apply_prefix_and_priors(
            proba,
            pref_str,
            x_norm,
            self.prefix_to_ids,
            self.mem_payee,
            self.mem_account,
            self.mem_payee_amount,
            amount_col=num_col,
        )

        # Apply thresholds
        thr_glob = float(self.thresholds["global"])
        thr_pp = self.thresholds["per_prefix"]
        thr_pc = (
            np.array(self.thresholds["per_class_tail_only"], dtype=float)
            if self.thresholds.get("per_class_tail_only") is not None
            else None
        )

        y_hat, abst = apply_thresholds(proba, pref_str, thr_glob, thr_pp, thr_pc)

        # Format results
        results = []
        for i in range(len(df)):
            result = {
                "label": self.classes[y_hat[i]],
                "confidence": float(proba[i, y_hat[i]]),
                "abstained": bool(abst[i]),
            }

            if top_k > 1:
                top_indices = np.argsort(proba[i])[-top_k:][::-1]
                result["top_k"] = [
                    {"label": self.classes[idx], "p": float(proba[i, idx])}
                    for idx in top_indices
                ]

            results.append(result)

        if return_proba:
            return results, proba
        return results

    def _get_blended_probabilities(self, Xfeats):
        """Get blended probabilities from models."""
        n_classes = len(self.classes)

        # Base probabilities
        proba_l = self.lgbm.predict_proba(Xfeats) if self.lgbm is not None else None
        proba_s = self.svm.predict_proba(Xfeats) if self.svm is not None else None

        # Align to full class space
        if proba_l is not None:
            proba_l = pad_proba(self.lgbm, proba_l, n_classes)
        if proba_s is not None:
            proba_s = pad_proba(self.svm, proba_s, n_classes)

        # Blend models
        if proba_l is not None and proba_s is not None:
            proba = self.lgbm_weight * proba_l + (1.0 - self.lgbm_weight) * proba_s
        else:
            proba = proba_l if proba_l is not None else proba_s

        return proba


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================


def determine_cv_folds(raw_labels: pd.Series, n_folds: int) -> int:
    """Determine optimal CV folds based on class distribution."""
    class_counts = Counter(raw_labels.astype(str))
    min_class_size = min(class_counts.values())

    # Adaptive fold selection: ensure each class has enough samples for stratified CV
    if min_class_size >= n_folds:
        logger.info(
            f"Using configured {n_folds} folds (min class size: {min_class_size})"
        )
        return n_folds
    elif min_class_size >= 3:
        logger.info(
            f"Reduced to 3 folds due to small classes (min class size: {min_class_size})"
        )
        return 3
    elif min_class_size >= 2:
        logger.info(
            f"Reduced to 2 folds due to very small classes (min class size: {min_class_size})"
        )
        return 2
    else:
        # Classes with only 1 sample - filter them out
        logger.warning(
            f"Removing classes with only 1 sample (min class size: {min_class_size})"
        )
        raise InternalException(
            "Classes with single samples must be filtered before CV"
        )


def filter_single_sample_classes(df: pd.DataFrame, label_col: str):
    """Filter out classes with only one sample."""
    raw_labels = df[label_col].astype(str)
    class_counts = Counter(raw_labels)

    # Log initial class distribution
    min_count = min(class_counts.values()) if class_counts else 0
    logger.info(
        f"Initial class distribution: {len(class_counts)} classes, min count: {min_count}"
    )

    valid_classes = [cls for cls, count in class_counts.items() if count >= 2]
    removed_classes = [cls for cls, count in class_counts.items() if count < 2]

    if removed_classes:
        logger.info(
            f"Filtering out {len(removed_classes)} classes with single samples: {removed_classes[:10]}..."
        )
        df_filtered = df[raw_labels.isin(valid_classes)].copy()
        logger.info(
            f"After filtering: {len(valid_classes)} classes remaining, min count: 2"
        )
        return df_filtered

    logger.info("No single-sample classes found, no filtering needed")
    return df


def filter_classes_for_forced_folds(df: pd.DataFrame, label_col: str, n_folds: int):
    """Filter out classes that have fewer samples than required for forced folds."""
    raw_labels = df[label_col].astype(str)
    class_counts = Counter(raw_labels)

    # Log initial class distribution
    min_count = min(class_counts.values()) if class_counts else 0
    logger.info(
        f"Initial class distribution: {len(class_counts)} classes, min count: {min_count}"
    )

    # For stratified CV, each class needs at least n_folds samples
    valid_classes = [cls for cls, count in class_counts.items() if count >= n_folds]
    removed_classes = [cls for cls, count in class_counts.items() if count < n_folds]

    if removed_classes:
        logger.info(
            f"Force folds enabled: filtering out {len(removed_classes)} classes with < {n_folds} samples: {removed_classes[:10]}..."
        )
        df_filtered = df[raw_labels.isin(valid_classes)].copy()
        new_min_count = (
            min([count for cls, count in class_counts.items() if cls in valid_classes])
            if valid_classes
            else 0
        )
        logger.info(
            f"After filtering for {n_folds} folds: {len(valid_classes)} classes remaining, min count: {new_min_count}"
        )
        return df_filtered

    logger.info(
        f"All classes have >= {n_folds} samples, no filtering needed for forced folds"
    )
    return df


def perform_cross_validation(
    df: pd.DataFrame,
    y: np.ndarray,
    cfg: dict,
    feature_pipe,
    prefix_le,
    adjusted_n_folds: int,
):
    """Perform cross-validation and return validation predictions."""
    skf = StratifiedKFold(
        n_splits=adjusted_n_folds, shuffle=True, random_state=int(cfg["random_state"])
    )

    logger.info("Using sequential processing for optimal performance on small datasets")

    # Pre-compute features once for speed (datasets always < 10K)
    logger.info(
        f"Dataset ({len(df)} rows) - pre-computing features for optimal performance"
    )
    X_precomputed = None

    # Pre-compute features once (with timing)
    try:
        start_time = time.time()
        X_precomputed = feature_pipe.fit_transform(df)
        feature_time = time.time() - start_time
        logger.info(
            f"Pre-computed features shape: {X_precomputed.shape} in {feature_time:.2f}s"
        )
    except Exception as e:
        logger.warning(f"Failed to pre-compute features: {e}")
        X_precomputed = None

    def _fold(tr_idx, va_idx):
        _check_cancellation()

        # Get data splits
        df_tr, df_va = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Use pre-computed features if available, otherwise build fresh pipeline
        if X_precomputed is not None:
            # Just slice the pre-computed features
            X_tr, X_va = X_precomputed[tr_idx], X_precomputed[va_idx]
        else:
            # Fallback: build fresh pipeline for this fold (slower)
            logger.warning("Building fresh pipeline for fold - this will be slower")
            fold_pipe = build_feature_pipeline(cfg)
            X_tr = fold_pipe.fit_transform(df_tr, y_tr)
            X_va = fold_pipe.transform(df_va)

        # Resampling
        sampler = RandomOverSampler(
            random_state=int(cfg["random_state"]),
            sampling_strategy=capped_ros_strategy(
                y_tr, cfg["sampler"]["ros_cap_percentile"]
            ),
        )
        X_tr_rs, y_tr_rs = sampler.fit_resample(X_tr, y_tr)

        # Train models - use thread limiting only for larger datasets
        proba_l = proba_s = None

        if cfg["models"]["use_lgbm"] and HAS_LGBM:
            # Optimize LightGBM settings for small datasets (always < 10K)
            lgbm_params = cfg["models"]["lgbm_params"].copy()
            lgbm_params["n_estimators"] = min(25, lgbm_params.get("n_estimators", 100))
            lgbm_params["max_depth"] = min(3, lgbm_params.get("max_depth", 6))
            lgbm_params["num_leaves"] = min(15, lgbm_params.get("num_leaves", 31))
            lgbm_params["min_data_in_leaf"] = max(
                1, min(5, lgbm_params.get("min_data_in_leaf", 20))
            )
            logger.info(
                f"Optimized LightGBM params for dataset: n_estimators={lgbm_params['n_estimators']}"
            )

            lgbm = LGBMClassifier(
                objective="multiclass",
                random_state=int(cfg["random_state"]),
                n_jobs=-1,  # Use all available threads
                **lgbm_params,
            )
            lgbm.set_params(num_class=len(np.unique(y)))
            lgbm.fit(X_tr_rs, y_tr_rs)
            proba_l = lgbm.predict_proba(X_va)

        if cfg["models"]["use_svm_blend"]:
            max_folds = int(cfg["cv"]["n_folds"])
            try:
                calibration_folds = determine_cv_folds(
                    pd.Series(y_tr_rs), min(max_folds, adjusted_n_folds)
                )

                svm_base = CancellableEstimator(
                    SGDClassifier(
                        loss="log_loss",
                        alpha=1e-4,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=int(cfg["random_state"]),
                    )
                )
                svm = CalibratedClassifierCV(
                    estimator=svm_base,
                    method=cfg["models"]["svm_calibration"],
                    cv=calibration_folds,
                )
                svm.fit(X_tr_rs, y_tr_rs)
                proba_s = svm.predict_proba(X_va)

            except (InternalException, ValueError) as e:
                logger.warning(f"SVM training failed: {e}")
                logger.info(
                    "Skipping SVM blending for this fold due to training issues"
                )
                proba_s = None

        # Blend probabilities
        lgbm_weight = float(cfg["models"]["lgbm_weight"])
        n_classes = len(np.unique(y))

        if proba_l is not None:
            proba_l = pad_proba(lgbm, proba_l, n_classes)
        if proba_s is not None:
            proba_s = pad_proba(svm, proba_s, n_classes)

        if proba_l is not None and proba_s is not None:
            proba_va = lgbm_weight * proba_l + (1.0 - lgbm_weight) * proba_s
        else:
            proba_va = proba_l if proba_l is not None else proba_s

        # Prefix predictions
        y_prefix_tr = prefix_le.fit_transform(label_prefixes(y_tr.astype(str)))
        prefix_clf = LogisticRegression(
            multi_class="ovr",
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=int(cfg["random_state"]),
        )
        prefix_clf.fit(X_tr, y_prefix_tr)
        pref_pred_va = prefix_le.inverse_transform(prefix_clf.predict(X_va))

        return y_va, proba_va, pref_pred_va

    # Run cross-validation sequentially (datasets always < 10K)
    logger.info("Running CV sequentially for optimal performance")
    backend = "sequential"
    prefer = "none"
    jobs = []

    for fold_idx, (tr, va) in enumerate(skf.split(df, y)):
        fold_start = time.time()
        logger.info(f"Starting fold {fold_idx + 1}/{adjusted_n_folds}")
        result = _fold(tr, va)
        fold_time = time.time() - fold_start
        logger.info(f"Completed fold {fold_idx + 1} in {fold_time:.2f}s")
        jobs.append(result)

    # Combine results
    val_true = np.concatenate([j[0] for j in jobs])
    val_proba = np.vstack([j[1] for j in jobs])
    val_pref = np.array(sum([list(j[2]) for j in jobs], []))

    # Return performance info
    perf_info = {
        "precompute_features": X_precomputed is not None,
        "backend": backend,
        "prefer": prefer,
    }

    return val_true, val_proba, val_pref, perf_info


def learn_thresholds(val_true, val_proba, val_pref, cfg, n_classes):
    """Learn optimal thresholds from validation data."""
    TH = cfg["thresholds"]
    THR_GRID = np.asarray(TH["global_grid"], dtype=float)

    # Global threshold
    thr_global, _ = pick_best_threshold(val_true, val_proba, THR_GRID)

    # Dynamic per-prefix thresholds
    unique_prefixes = np.unique(val_pref)
    dynamic_per_prefix_init = {prefix: float(thr_global) for prefix in unique_prefixes}

    logger.info(
        f"Generated dynamic per-prefix init thresholds for {len(unique_prefixes)} prefixes"
    )

    thr_per_prefix, _ = pick_per_prefix_thresholds(
        val_true,
        val_proba,
        val_pref,
        THR_GRID,
        init=dynamic_per_prefix_init,
    )

    # Tail class thresholds
    counts = np.bincount(val_true, minlength=n_classes)
    y_arg_tmp = val_proba.argmax(axis=1)
    mask_tmp = val_proba.max(axis=1) >= thr_global
    f1_per_class = np.zeros(n_classes)

    for c in range(n_classes):
        sel = mask_tmp & ((val_true == c) | (y_arg_tmp == c))
        if not np.any(sel):
            continue
        y_true_c = (val_true[sel] == c).astype(int)
        y_pred_c = (y_arg_tmp[sel] == c).astype(int)
        tp = ((y_pred_c == 1) & (y_true_c == 1)).sum()
        fp = ((y_pred_c == 1) & (y_true_c == 0)).sum()
        fn = ((y_pred_c == 0) & (y_true_c == 1)).sum()
        denom = 2 * tp + fp + fn
        f1_per_class[c] = (2 * tp) / denom if denom > 0 else 0.0

    TAIL_MAX_SUP = int(TH["tail_support_max"])
    TAIL_MAX_F1 = float(TH["tail_f1_max"])
    TAIL_GRID = np.asarray(TH["tail_grid"], dtype=float)

    tail_classes = set(np.where(counts <= TAIL_MAX_SUP)[0].tolist()) | set(
        np.where(f1_per_class <= TAIL_MAX_F1)[0].tolist()
    )

    thr_per_class = pick_tail_class_thresholds(
        val_true, val_proba, n_classes, tail_classes, TAIL_GRID
    )

    return thr_global, thr_per_prefix, thr_per_class


def train_final_models(df, y, cfg, feature_pipe, prefix_le, adjusted_n_folds=3):
    """Train final models on all data."""
    # No threadpool limits needed for small datasets (< 10K)
    X_all_ft = feature_pipe.fit_transform(df, y)

    sampler_all = RandomOverSampler(
        random_state=int(cfg["random_state"]),
        sampling_strategy=capped_ros_strategy(y, cfg["sampler"]["ros_cap_percentile"]),
    )
    X_all, y_all = sampler_all.fit_resample(X_all_ft, y)

    # Prefix classifier
    y_prefix_all = prefix_le.fit_transform(label_prefixes(y.astype(str)))
    prefix_clf = LogisticRegression(
        multi_class="ovr",
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=int(cfg["random_state"]),
    )
    prefix_clf.fit(X_all_ft, y_prefix_all)

    # Final models
    lgbm_full = svm_full = None

    if cfg["models"]["use_lgbm"] and HAS_LGBM:
        # Optimize LightGBM settings for small datasets (always < 10K)
        lgbm_params = cfg["models"]["lgbm_params"].copy()
        lgbm_params["n_estimators"] = min(25, lgbm_params.get("n_estimators", 100))
        lgbm_params["max_depth"] = min(3, lgbm_params.get("max_depth", 6))
        lgbm_params["num_leaves"] = min(15, lgbm_params.get("num_leaves", 31))
        lgbm_params["min_data_in_leaf"] = max(
            1, min(5, lgbm_params.get("min_data_in_leaf", 20))
        )

        lgbm_full = LGBMClassifier(
            objective="multiclass",
            random_state=int(cfg["random_state"]),
            n_jobs=-1,  # Use all threads for small datasets
            **lgbm_params,
        )
        lgbm_full.set_params(num_class=len(np.unique(y)))
        lgbm_full.fit(X_all, y_all)

        if cfg["models"]["use_svm_blend"]:
            # Determine appropriate CV folds for calibration based on resampled data
            max_folds = int(cfg["cv"]["n_folds"])
            calibration_folds = determine_cv_folds(
                pd.Series(y_all), min(max_folds, adjusted_n_folds)
            )

            svm_base = CancellableEstimator(
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=int(cfg["random_state"]),
                )
            )
            svm_full = CalibratedClassifierCV(
                estimator=svm_base,
                method=cfg["models"]["svm_calibration"],
                cv=calibration_folds,
            )
            svm_full.fit(X_all, y_all)

    return lgbm_full, svm_full, prefix_clf


def calculate_metrics(y_true, y_hat, abstain_mask, metric_prefix=""):
    """Calculate standard metrics for predictions."""
    if (~abstain_mask).any():
        macro_f1 = f1_score(
            y_true[~abstain_mask],
            y_hat[~abstain_mask],
            average="macro",
            zero_division=0,
        )
        bal_acc = balanced_accuracy_score(y_true[~abstain_mask], y_hat[~abstain_mask])
        acc = accuracy_score(y_true[~abstain_mask], y_hat[~abstain_mask])
    else:
        macro_f1 = bal_acc = acc = 0.0

    abstain_rate = float(abstain_mask.mean())

    return {
        f"{metric_prefix}macro_f1": float(macro_f1),
        f"{metric_prefix}balanced_accuracy": float(bal_acc),
        f"{metric_prefix}accuracy": float(acc),
        f"{metric_prefix}abstain_rate": abstain_rate,
    }


def build_all_memories(df, y, feature_pipe, cfg, date_col, num_col):
    """Build all memory types using existing build functions."""
    train_norm = feature_pipe.named_steps["add_text"].transform(df.copy())

    return {
        "payee": build_payee_memory(
            train_norm, y, n_min=int(cfg["priors"]["vendor"]["min_count"])
        ),
        "account": build_account_memory(
            train_norm, y, n_min=int(cfg["priors"]["vendor"]["min_count"])
        ),
        "payee_amount": build_payee_amount_patterns(train_norm, y, num_col, n_min=3),
    }


def _get_model_predictions(lgbm_model, svm_model, X_features, lgbm_weight, n_classes):
    """Get blended predictions from trained models."""
    proba_l = lgbm_model.predict_proba(X_features) if lgbm_model is not None else None
    proba_s = svm_model.predict_proba(X_features) if svm_model is not None else None

    # Align to full class space
    if proba_l is not None:
        proba_l = pad_proba(lgbm_model, proba_l, n_classes)
    if proba_s is not None:
        proba_s = pad_proba(svm_model, proba_s, n_classes)

    # Blend models
    if proba_l is not None and proba_s is not None:
        return lgbm_weight * proba_l + (1.0 - lgbm_weight) * proba_s
    else:
        return proba_l if proba_l is not None else proba_s


def _perform_final_retraining(
    df,
    y,
    cfg,
    feature_pipe,
    prefix_le,
    lgbm_full,
    svm_full,
    prefix_clf,
    memories,
    thr_global,
    thr_per_prefix,
    thr_per_class,
    n_classes,
    adjusted_n_folds,
    lgbm_weight,
    random_state,
    date_col,
    num_col,
):
    """Perform final retraining with train/test split and return metrics."""
    # Calculate appropriate test size based on dataset and number of classes
    total_samples = len(df)

    # Ensure we have at least 2 samples per class in test set, but not more than 20% of data
    min_test_samples = max(
        n_classes * 2, 10
    )  # At least 2 per class or 10 samples minimum
    max_test_samples = int(total_samples * 0.20)  # Maximum 20% of data

    # Choose test size that ensures stratification works
    if total_samples <= min_test_samples * 2:
        test_samples = min_test_samples
    else:
        test_samples = min(min_test_samples, max_test_samples)

    test_size = test_samples / total_samples

    logger.info(
        f"Using {test_samples} samples ({test_size:.1%}) for final test evaluation"
    )

    # Reserve a test set for final model evaluation
    df_train_final, df_test_final, y_train_final, y_test_final = train_test_split(
        df, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Reserved {len(df_test_final)} samples for final test evaluation")

    # Retrain final models using existing train_final_models function
    lgbm_final, svm_final, prefix_clf_final = train_final_models(
        df_train_final, y_train_final, cfg, feature_pipe, prefix_le, adjusted_n_folds
    )

    # Rebuild memories using existing build_all_memories function
    updated_memories = build_all_memories(
        df_train_final, y_train_final, feature_pipe, cfg, date_col, num_col
    )

    # Evaluate final model on reserved test set
    logger.info("Evaluating final retrained model on reserved test set...")

    # Get test predictions using existing helper function
    X_test_final = feature_pipe.transform(df_test_final)
    final_pref_idx = prefix_clf_final.predict(X_test_final)
    final_pref = prefix_le.inverse_transform(final_pref_idx)
    final_proba = _get_model_predictions(
        lgbm_final, svm_final, X_test_final, lgbm_weight, n_classes
    )

    # Apply thresholds to get final predictions
    y_hat_final, abst_final = apply_thresholds(
        final_proba, final_pref, thr_global, thr_per_prefix, thr_per_class
    )

    # Calculate final test metrics
    final_test_metrics = calculate_metrics(
        y_test_final, y_hat_final, abst_final, "final_"
    )

    logger.info(f"Final retraining completed on {len(y_train_final)} samples")
    logger.info(
        f"Final model test performance: F1={final_test_metrics['final_macro_f1']:.4f}, "
        f"Acc={final_test_metrics['final_accuracy']:.4f}, "
        f"Abstain={final_test_metrics['final_abstain_rate']:.2%}"
    )

    return {
        **final_test_metrics,
        "updated_memories": updated_memories,  # Return the updated memories
        "updated_models": {  # Return the updated models for bundle update
            "lgbm": lgbm_final,
            "svm": svm_final,
            "prefix_clf": prefix_clf_final,
        },
        "final_retraining": {
            "enabled": True,
            "final_training_samples": int(len(y_train_final)),
            "test_samples": int(len(y_test_final)),
            "test_size_percent": f"{test_size:.1%}",
            "note": f"Final model trained on {len(y_train_final)} samples, evaluated on {len(y_test_final)} test samples",
        },
    }


def train_model_bundle(
    cfg: dict[str, Any], train_df: pd.DataFrame
) -> tuple[ModelBundle, dict[str, float]]:
    """Train complete PXBlendSC model bundle."""
    # Setup
    RANDOM_STATE = int(cfg["random_state"])
    N_FOLDS = int(cfg["cv"]["n_folds"])
    N_CPUS = os.cpu_count() or 4
    LABEL_COL = cfg["columns"]["LABEL_COL"]
    TEXT_COLS = cfg["columns"]["TEXT_COLS"]
    NUM_COL = cfg["columns"]["NUM_COLS"][0]
    DATE_COL = cfg["columns"]["DATE_COL"]
    LGBM_WEIGHT = float(cfg["models"]["lgbm_weight"])

    # Data preparation
    df = prepare_dataframe_columns(train_df.copy(), TEXT_COLS, DATE_COL, NUM_COL)

    # Check if force_folds is enabled
    force_folds = cfg["cv"].get("force_folds", False)
    logger.info(f"Force folds configuration: {force_folds} (from config: {cfg['cv']})")

    if force_folds:
        # When force_folds is enabled, filter classes to ensure each has enough samples for stratified CV
        df = filter_classes_for_forced_folds(df, LABEL_COL, N_FOLDS)
        # Use the configured number of folds
        adjusted_n_folds = N_FOLDS
        logger.info(f"Force folds enabled: using {N_FOLDS} folds with filtered data")
    else:
        # Original behavior: filter single-sample classes and adapt folds based on data
        df = filter_single_sample_classes(df, LABEL_COL)
        # Determine CV folds based on class distribution
        adjusted_n_folds = determine_cv_folds(df[LABEL_COL], N_FOLDS)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL].astype(str))
    classes = list(le.classes_)
    n_classes = len(classes)

    # Processing is always sequential for optimal performance on small datasets

    # Prefixes
    prefix_of = np.array(label_prefixes(classes))
    prefix_to_ids = {
        p: np.where(prefix_of == p)[0].tolist() for p in sorted(set(prefix_of))
    }

    # Features
    feature_pipe = build_feature_pipeline(cfg)
    prefix_le = LabelEncoder()

    logger.info(f"Final CV setup: {adjusted_n_folds} folds, {n_classes} classes")

    # Cross-validation
    val_true, val_proba, val_pref, perf_info = perform_cross_validation(
        df, y, cfg, feature_pipe, prefix_le, adjusted_n_folds
    )

    # Learn thresholds
    thr_global, thr_per_prefix, thr_per_class = learn_thresholds(
        val_true, val_proba, val_pref, cfg, n_classes
    )

    # CV metrics
    y_hat_cv, abst_cv = apply_thresholds(
        val_proba, val_pref, thr_global, thr_per_prefix, thr_per_class
    )
    cv_metrics = calculate_metrics(val_true, y_hat_cv, abst_cv, "cv_")

    # Final fit on all data
    lgbm_full, svm_full, prefix_clf = train_final_models(
        df, y, cfg, feature_pipe, prefix_le, adjusted_n_folds
    )

    # Build all memory types
    train_norm = feature_pipe.named_steps["add_text"].transform(df.copy())
    memories = {
        "payee": build_payee_memory(
            train_norm, y, n_min=int(cfg["priors"]["vendor"]["min_count"])
        ),
        "account": build_account_memory(
            train_norm, y, n_min=int(cfg["priors"]["vendor"]["min_count"])
        ),
        "payee_amount": build_payee_amount_patterns(train_norm, y, NUM_COL, n_min=3),
    }

    # Calculate training metrics on full final model
    logger.info("Calculating training metrics on full dataset...")
    X_full_pred = feature_pipe.transform(df)
    pref_pred_full = prefix_le.inverse_transform(prefix_clf.predict(X_full_pred))

    # Generate training predictions from final models
    train_proba = _get_model_predictions(
        lgbm_full, svm_full, X_full_pred, LGBM_WEIGHT, len(classes)
    )

    # Apply prefix and priors to training predictions
    train_proba = apply_prefix_and_priors(
        train_proba,
        pref_pred_full,
        train_norm,
        prefix_to_ids,
        memories["payee"],
        memories["account"],
        memories["payee_amount"],
        amount_col=NUM_COL,
    )

    # Apply thresholds to training data
    y_hat_train, abst_train = apply_thresholds(
        train_proba, pref_pred_full, thr_global, thr_per_prefix, thr_per_class
    )

    # Calculate training metrics
    train_metrics = calculate_metrics(y, y_hat_train, abst_train, "train_")

    # Calculate total dataset metrics (training + validation combined)
    total_samples = len(val_true) + len(y)
    total_true = np.concatenate([val_true, y])
    total_proba = np.vstack([val_proba, train_proba])
    total_pref = np.concatenate([val_pref, pref_pred_full])

    y_hat_total, abst_total = apply_thresholds(
        total_proba, total_pref, thr_global, thr_per_prefix, thr_per_class
    )

    total_metrics = calculate_metrics(total_true, y_hat_total, abst_total, "total_")

    # Optional: Final retraining on all data (if enabled in config)
    final_metrics = {}
    FINAL_RETRAIN = bool(cfg["models"].get("final_retraining", False))
    if FINAL_RETRAIN:
        logger.info("Performing final retraining with test set evaluation...")
        final_metrics = _perform_final_retraining(
            df,
            y,
            cfg,
            feature_pipe,
            prefix_le,
            lgbm_full,
            svm_full,
            prefix_clf,
            memories,
            thr_global,
            thr_per_prefix,
            thr_per_class,
            n_classes,
            adjusted_n_folds,
            LGBM_WEIGHT,
            RANDOM_STATE,
            DATE_COL,
            NUM_COL,
        )

        # Update models and memories with final retrained versions
        if "updated_models" in final_metrics:
            lgbm_full = final_metrics["updated_models"]["lgbm"]
            svm_full = final_metrics["updated_models"]["svm"]
            prefix_clf = final_metrics["updated_models"]["prefix_clf"]

        if "updated_memories" in final_metrics:
            memories = final_metrics["updated_memories"]
    else:
        final_metrics = {
            "final_retraining": {
                "enabled": False,
                "note": "Final model uses only original training data",
            }
        }

    # Create bundle
    thresholds = {
        "global": float(thr_global),
        "per_prefix": {k: float(v) for k, v in thr_per_prefix.items()},
        "per_class_tail_only": thr_per_class.tolist()
        if thr_per_class is not None
        else [],
    }

    bundle = ModelBundle(
        cfg=cfg,
        feature_pipe=feature_pipe,
        lgbm_model=lgbm_full,
        svm_model=svm_full,
        lgbm_weight=LGBM_WEIGHT,
        prefix_clf=prefix_clf,
        prefix_label_encoder=prefix_le,
        label_encoder=le,
        thresholds=thresholds,
        prefix_to_ids=prefix_to_ids,
        classes=classes,
        mem_payee=memories["payee"],
        mem_account=memories["account"],
        mem_payee_amount=memories["payee_amount"],
    )

    # Combine all metrics
    metrics = {
        # Cross-validation (validation) metrics
        **cv_metrics,
        # Training metrics on full dataset
        **train_metrics,
        # Total dataset metrics (both training and validation)
        **total_metrics,
        # Final retraining metrics (if available)
        **final_metrics,
        # Dataset composition
        "dataset_info": {
            "total_samples": int(total_samples),
            "training_samples": int(len(y)),
            "validation_samples": int(len(val_true)),
            "n_classes": int(n_classes),
            "cv_folds": int(adjusted_n_folds),
        },
        "thresholds": {
            "global": float(thr_global),
            "per_prefix": {k: float(v) for k, v in thr_per_prefix.items()},
            "per_class_tail_only": {
                int(i): float(thr_per_class[i])
                for i in np.where(~np.isnan(thr_per_class))[0]
            }
            if thr_per_class is not None
            else {},
        },
        "performance": perf_info,
    }

    return bundle, metrics


# =============================================================================
# MAIN STRATEGY CLASS
# =============================================================================


class PXBlendSCStrategy(MLModelStrategy):
    """PXBlendSC ML strategy implementation."""

    def get_model_type(self) -> ModelType:
        return ModelType.PXBlendSC

    def _get_default_config(self) -> dict:
        """Get default PXBlendSC configuration from ConfigDefaults."""
        return ConfigDefaults.PXBLENDSC_CONFIG.copy()

    async def train_model(self, request, training_data: list) -> dict:
        """Train a PXBlendSC model."""
        logger.info(f"Training PXBlendSC model: {request.modelCard.name}")

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
                f"PXBlendSC training error details: {type(e).__name__}: {str(e)}"
            )
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise InternalException(f"PXBlendSC training failed: {str(e)}") from e

        # Save model
        temp_dir = tempfile.mkdtemp(prefix="pxblendsc_")
        bundle.save(temp_dir)

        # Create model metadata
        model = self._create_model_metadata(
            request, bundle, metrics, train_df, temp_dir
        )

        logger.info(
            f"PXBlendSC training completed:\n"
            f"  Validation: F1={metrics['cv_macro_f1']:.4f}, "
            f"Acc={metrics['cv_accuracy']:.4f}, "
            f"Abstain={metrics['cv_abstain_rate']:.2%}"
        )

        return model

    async def predict(
        self, model: dict, input_data: list
    ) -> list[CategoricalPredictionResult]:
        """Generate predictions using PXBlendSC model."""
        bundle_path = model.get("bundle_path")

        if not bundle_path or not os.path.exists(bundle_path):
            raise InternalException("PXBlendSC model not found or path invalid")

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
            logger.error(f"PXBlendSC prediction error: {e}")
            raise InternalException(f"Prediction failed: {str(e)}") from e

    async def evaluate_model(self, model: dict, test_data_path: str) -> dict:
        """Evaluate a trained PXBlendSC model on test data."""
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
                logger.info("Using PXBlendSC configuration from Configs service")
                return cfg
            except Exception as e:
                logger.warning(
                    f"Failed to get PXBlendSC config from service: {e}, using defaults"
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
