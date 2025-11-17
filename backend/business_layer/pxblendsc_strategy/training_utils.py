"""Training utility functions for PXBlendSC-RF strategy."""

import logging
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from thrift_gen.exceptions.ttypes import InternalException

logger = logging.getLogger(__name__)


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


def learn_thresholds(val_true, val_proba, val_pref, cfg, n_classes):
    """Learn optimal thresholds from validation data."""
    from .thresholds import (
        pick_best_threshold,
        pick_per_prefix_thresholds,
        pick_tail_class_thresholds,
    )

    logger.info(f"Learning thresholds from {len(val_true)} validation samples")

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

    # Log tail class identification details
    logger.info(f"Identified {len(tail_classes)} tail classes for threshold learning")
    logger.debug(f"Tail class criteria: support <= {TAIL_MAX_SUP}, F1 <= {TAIL_MAX_F1}")

    if tail_classes:
        logger.debug("Tail classes identified:")
        for class_idx in sorted(tail_classes):
            support = counts[class_idx] if class_idx < len(counts) else 0
            f1 = f1_per_class[class_idx] if class_idx < len(f1_per_class) else 0.0
            logger.debug(f"  Class {class_idx}: support={support}, F1={f1:.3f}")

    # Calculate baseline abstention rate before tail thresholds
    baseline_abstentions = (val_proba.max(axis=1) < thr_global).sum()
    baseline_abstention_rate = baseline_abstentions / len(val_true)
    logger.info(
        f"Baseline abstention rate (global threshold): {baseline_abstention_rate:.1%}"
    )

    thr_per_class = pick_tail_class_thresholds(
        val_true, val_proba, n_classes, tail_classes, TAIL_GRID
    )

    # Log impact of tail class thresholds on abstention rate
    if thr_per_class is not None:
        # Simulate abstention rate with tail thresholds
        y_pred_with_tail = val_proba.argmax(axis=1)
        abstain_mask = val_proba.max(axis=1) < thr_global

        # Apply per-class thresholds
        for class_idx in range(n_classes):
            if not np.isnan(thr_per_class[class_idx]):
                class_predictions = y_pred_with_tail == class_idx
                low_confidence = val_proba[:, class_idx] < thr_per_class[class_idx]
                abstain_mask |= class_predictions & low_confidence

        tail_abstentions = abstain_mask.sum()
        tail_abstention_rate = tail_abstentions / len(val_true)
        additional_abstentions = tail_abstentions - baseline_abstentions

        logger.info(f"Abstention rate with tail thresholds: {tail_abstention_rate:.1%}")
        logger.info(
            f"Additional abstentions from tail thresholds: {additional_abstentions} ({additional_abstentions / len(val_true):.1%})"
        )
    else:
        logger.info("No tail class thresholds learned")

    return thr_global, thr_per_prefix, thr_per_class
