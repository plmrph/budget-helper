"""Memory building functions for payee and account priors."""

import logging
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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
