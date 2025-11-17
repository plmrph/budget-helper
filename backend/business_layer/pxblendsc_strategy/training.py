"""Training functions for PXBlendSC-RF strategy."""

import asyncio
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from thrift_gen.exceptions.ttypes import InternalException

from .bundle import ModelBundle
from .memory import build_account_memory, build_payee_amount_patterns, build_payee_memory
from .models import CancellableEstimator, CancellationCallback, RecencyFrequencyModel
from .pipeline import build_feature_pipeline
from .thresholds import apply_prefix_and_priors, apply_thresholds
from .training_utils import (
    calculate_metrics,
    determine_cv_folds,
    filter_classes_for_forced_folds,
    filter_single_sample_classes,
    learn_thresholds,
)
from .utils import (
    _check_cancellation,
    capped_ros_strategy,
    label_prefixes,
    pad_proba,
    prepare_dataframe_columns,
)

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

logger = logging.getLogger(__name__)


def perform_cross_validation(
    df: pd.DataFrame,
    y: np.ndarray,
    cfg: dict,
    prefix_le,
    adjusted_n_folds: int,
):
    """Perform cross-validation and return validation predictions."""
    skf = StratifiedKFold(
        n_splits=adjusted_n_folds, shuffle=True, random_state=int(cfg["random_state"])
    )

    logger.info("Using sequential processing for optimal performance on small datasets")
    logger.info(f"Dataset ({len(df)} rows) - building feature pipeline per fold")

    def _fold(tr_idx, va_idx):
        _check_cancellation()

        # Get data splits
        df_tr, df_va = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Build feature pipeline on training fold ONLY
        fold_feature_pipe = build_feature_pipeline(cfg)
        X_tr = fold_feature_pipe.fit_transform(df_tr, y_tr)
        X_va = fold_feature_pipe.transform(df_va)  # Transform only

        # Resampling
        sampler = RandomOverSampler(
            random_state=int(cfg["random_state"]),
            sampling_strategy=capped_ros_strategy(
                y_tr, cfg["sampler"]["ros_cap_percentile"]
            ),
        )
        X_tr_rs, y_tr_rs = sampler.fit_resample(X_tr, y_tr)

        # Train models with CONSISTENT parameters from config
        proba_l = proba_s = None

        if cfg["models"]["use_lgbm"] and HAS_LGBM:
            lgbm = LGBMClassifier(
                objective="multiclass",
                random_state=int(cfg["random_state"]),
                n_jobs=-1,
                **cfg["models"]["lgbm_params"],  # Use config params directly
            )
            lgbm.set_params(num_class=len(np.unique(y)))
            lgbm.fit(
                X_tr_rs,
                y_tr_rs,
                eval_set=[(X_va, y_va)],
                eval_metric="multi_logloss",
                callbacks=[CancellationCallback()],
            )
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
                        alpha=1e-2,
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

        # Train recency-frequency model
        proba_rf = None
        if cfg["models"]["use_recency_frequency"]:
            try:
                rf_params = cfg["models"]["recency_freq_params"]
                rf_model = RecencyFrequencyModel(**rf_params)

                # Use training data with dates (if available) or indices
                dates = df_tr.get("date", list(range(len(df_tr))))
                rf_model.fit(df_tr, y_tr, dates)
                proba_rf = rf_model.predict_proba(df_va)

            except Exception as e:
                logger.warning(f"RecencyFrequency training failed: {e}")
                logger.info(
                    "Skipping RecencyFrequency for this fold due to training issues"
                )
                proba_rf = None

        # Blend probabilities
        lgbm_weight = float(cfg["models"].get("lgbm_weight", 0.5))
        svm_weight = float(cfg["models"].get("svm_weight", 0.25))
        rf_weight = float(cfg["models"].get("recency_freq_weight", 0.25))
        n_classes = len(np.unique(y))

        # Pad probabilities to full class space
        if proba_l is not None:
            proba_l = pad_proba(lgbm, proba_l, n_classes)
        if proba_s is not None:
            proba_s = pad_proba(svm, proba_s, n_classes)
        # proba_rf is already in full class space

        # Blend all available models
        total_weight = 0
        proba_va = np.zeros((len(df_va), n_classes))

        if proba_l is not None:
            proba_va += lgbm_weight * proba_l
            total_weight += lgbm_weight

        if proba_s is not None:
            proba_va += svm_weight * proba_s
            total_weight += svm_weight

        if proba_rf is not None:
            proba_va += rf_weight * proba_rf
            total_weight += rf_weight

        # Normalize weights if not all models are available
        if total_weight > 0:
            proba_va = proba_va / total_weight
        else:
            # Fallback to uniform distribution
            proba_va = np.ones((len(df_va), n_classes)) / n_classes

        # Build memories on training fold ONLY
        train_norm = fold_feature_pipe.named_steps["add_text"].transform(df_tr.copy())
        fold_memories = {
            "payee": build_payee_memory(
                train_norm, y_tr, n_min=int(cfg["priors"]["vendor"]["min_count"])
            ),
            "account": build_account_memory(
                train_norm, y_tr, n_min=int(cfg["priors"]["vendor"]["min_count"])
            ),
            "payee_amount": build_payee_amount_patterns(
                train_norm, y_tr, cfg["columns"]["NUM_COLS"][0], n_min=3
            ),
        }

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

        # Apply prefix and priors using fold-specific memories
        val_norm = fold_feature_pipe.named_steps["add_text"].transform(df_va.copy())

        # Get prefix constraints
        prefix_of = np.array(label_prefixes(np.unique(y_tr).astype(str)))
        prefix_to_ids = {
            p: np.where(prefix_of == p)[0].tolist() for p in sorted(set(prefix_of))
        }

        proba_va = apply_prefix_and_priors(
            proba_va,
            pref_pred_va,
            val_norm,
            prefix_to_ids,
            fold_memories["payee"],
            fold_memories["account"],
            fold_memories["payee_amount"],
            amount_col=cfg["columns"]["NUM_COLS"][0],
        )

        return y_va, proba_va, pref_pred_va

    # Run cross-validation sequentially (datasets always < 10K)
    logger.info("Running CV sequentially for optimal performance")
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

    return val_true, val_proba, val_pref


def train_final_models(df, y, cfg, feature_pipe, prefix_le, adjusted_n_folds=3):
    """Train final models on all data."""
    logger.info(f"Final model training: received {len(y)} samples for training")
    # No threadpool limits needed for small datasets (< 10K)
    X_all_ft = feature_pipe.fit_transform(df, y)

    sampler_all = RandomOverSampler(
        random_state=int(cfg["random_state"]),
        sampling_strategy=capped_ros_strategy(y, cfg["sampler"]["ros_cap_percentile"]),
    )
    X_all, y_all = sampler_all.fit_resample(X_all_ft, y)
    logger.info(
        f"Final model training: using {len(y_all)} samples after resampling (original: {len(y)})"
    )

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
        lgbm_full = LGBMClassifier(
            objective="multiclass",
            random_state=int(cfg["random_state"]),
            n_jobs=-1,
            **cfg["models"]["lgbm_params"],  # Use config params directly
        )
        lgbm_full.set_params(num_class=len(np.unique(y)))

        # Create validation split for early stopping in final training
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_all,
            y_all,
            test_size=0.2,
            random_state=int(cfg["random_state"]),
            stratify=y_all,
        )

        lgbm_full.fit(
            X_train_final,
            y_train_final,
            eval_set=[(X_val_final, y_val_final)],
            eval_metric="multi_logloss",
            callbacks=[CancellationCallback()],
        )

    if cfg["models"]["use_svm_blend"]:
        # Determine appropriate CV folds for calibration based on resampled data
        max_folds = int(cfg["cv"]["n_folds"])
        calibration_folds = determine_cv_folds(
            pd.Series(y_all), min(max_folds, adjusted_n_folds)
        )

        svm_base = CancellableEstimator(
            SGDClassifier(
                loss="log_loss",
                alpha=1e-2,
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

    # Train final recency-frequency model
    rf_full = None
    if cfg["models"]["use_recency_frequency"]:
        try:
            rf_params = cfg["models"]["recency_freq_params"]
            rf_full = RecencyFrequencyModel(**rf_params)

            # Use original data (not resampled) for recency-frequency patterns
            dates = df.get("date", list(range(len(df))))
            rf_full.fit(df, y, dates)
            logger.info("Final RecencyFrequency model trained successfully")

        except Exception as e:
            logger.warning(f"Final RecencyFrequency training failed: {e}")
            rf_full = None

    return lgbm_full, svm_full, rf_full, prefix_clf


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


def train_model_bundle(
    cfg: dict[str, Any], train_df: pd.DataFrame
) -> tuple[ModelBundle, dict[str, float]]:
    """Train complete PXBlendSC-RF model bundle."""
    # Setup
    N_FOLDS = int(cfg["cv"]["n_folds"])
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
    val_true, val_proba, val_pref = perform_cross_validation(
        df, y, cfg, prefix_le, adjusted_n_folds
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
    logger.info(f"Training final models on all {len(y)} samples")
    lgbm_full, svm_full, rf_full, prefix_clf = train_final_models(
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

    # Final model trains on all available data (training + validation combined)
    logger.info(f"Final model trained on all {len(y)} samples")

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
        rf_model=rf_full,
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
    }

    return bundle, metrics
