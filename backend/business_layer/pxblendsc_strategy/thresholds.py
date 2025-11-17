"""Threshold calculation and application for PXBlendSC-RF strategy."""

import logging

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def apply_prefix_and_priors(
    proba: np.ndarray,
    pref_str: np.ndarray,
    X_norm,
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
    """
    Pick optimal thresholds for tail classes based on actual performance.

    For each tail class, finds the threshold that maximizes F1 score
    by comparing actual predictions vs ground truth at different
    confidence levels.

    Args:
        y_true: Ground truth labels
        proba: Predicted probabilities (n_samples, n_classes)
        n_classes: Total number of classes
        tail_classes: Set of class indices considered as tail classes
        thresholds: Array of threshold values to test

    Returns:
        Array of optimal thresholds per class, or None if no tail classes
    """
    if not tail_classes:
        return None

    class_thresholds = np.full(n_classes, np.nan)

    # Get predicted classes for all samples
    y_pred_all = proba.argmax(axis=1)

    logger.debug(f"Learning thresholds for {len(tail_classes)} tail classes")

    for class_idx in tail_classes:
        if class_idx >= n_classes:
            continue

        # Check if we have enough samples of this class in ground truth
        class_mask = y_true == class_idx
        if class_mask.sum() < 5:
            logger.debug(
                f"Class {class_idx}: insufficient samples ({class_mask.sum()}), skipping"
            )
            continue

        best_f1 = 0.0
        best_threshold = thresholds[0]

        for threshold in thresholds:
            # Apply threshold: only predict this class if confidence >= threshold
            confident_mask = proba[:, class_idx] >= threshold

            # Create binary predictions: 1 if we predict this class with confidence, 0 otherwise
            y_pred_binary = (y_pred_all == class_idx) & confident_mask
            y_true_binary = y_true == class_idx

            # Calculate F1 for this class at this threshold
            if y_pred_binary.sum() > 0 or y_true_binary.sum() > 0:
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

        class_thresholds[class_idx] = float(best_threshold)
        logger.debug(
            f"Class {class_idx}: threshold={best_threshold:.3f}, F1={best_f1:.3f}"
        )

    # Log summary statistics
    valid_thresholds = class_thresholds[~np.isnan(class_thresholds)]
    if len(valid_thresholds) > 0:
        logger.info(f"Learned thresholds for {len(valid_thresholds)} tail classes")
        logger.info(
            f"Threshold range: {valid_thresholds.min():.3f} - {valid_thresholds.max():.3f}"
        )
        logger.info(f"Mean threshold: {valid_thresholds.mean():.3f}")
    else:
        logger.warning("No valid thresholds learned for tail classes")

    return class_thresholds
