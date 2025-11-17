"""Feature engineering classes for PXBlendSC-RF strategy."""

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
