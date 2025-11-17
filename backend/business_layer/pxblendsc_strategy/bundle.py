"""Model bundle for PXBlendSC-RF strategy."""

import json
import logging
import os

import numpy as np
from joblib import dump, load

from .thresholds import apply_prefix_and_priors, apply_thresholds
from .utils import convert_numpy_types, pad_proba, prepare_dataframe_columns

logger = logging.getLogger(__name__)


class ModelBundle:
    """Complete model bundle for PXBlendSC-RF strategy."""

    def __init__(
        self,
        *,
        cfg,
        feature_pipe,
        lgbm_model,
        svm_model,
        rf_model=None,
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
        self.rf = rf_model
        self.lgbm_weight = float(lgbm_weight)
        self.svm_weight = float(cfg["models"].get("svm_weight", 0.25))
        self.rf_weight = float(cfg["models"].get("recency_freq_weight", 0.25))
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
        if self.rf is not None:
            dump(self.rf, os.path.join(folder, "rf.joblib"))

        # Save artifact with full configuration
        artifact = {
            "version": "px-blend-sc:1",
            "cfg": self.cfg,
            "label_encoder_classes": self.label_encoder.classes_.tolist(),
            "feature_pipe": "feature_pipe.joblib",
            "prefix_clf": "prefix_clf.joblib",
            "lgbm": "lgbm.joblib" if self.lgbm is not None else None,
            "svm": "svm.joblib" if self.svm is not None else None,
            "rf": "rf.joblib" if self.rf is not None else None,
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
            "svm_weight": self.svm_weight,
            "rf_weight": self.rf_weight,
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

        rf_model = None
        if artifact.get("rf") is not None:
            rf_model = load(os.path.join(folder, artifact["rf"]))

        return ModelBundle(
            cfg=artifact["cfg"],
            feature_pipe=feature_pipe,
            lgbm_model=lgbm_model,
            svm_model=svm_model,
            rf_model=rf_model,
            lgbm_weight=artifact.get("lgbm_weight", 0.7),
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
        import pandas as pd

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
        proba = self._get_blended_probabilities(Xfeats, df)

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

    def _get_blended_probabilities(self, Xfeats, df):
        """Get blended probabilities from models."""
        n_classes = len(self.classes)

        # Base probabilities
        proba_l = self.lgbm.predict_proba(Xfeats) if self.lgbm is not None else None
        proba_s = self.svm.predict_proba(Xfeats) if self.svm is not None else None
        proba_rf = self.rf.predict_proba(df) if self.rf is not None else None

        # Align to full class space
        if proba_l is not None:
            proba_l = pad_proba(self.lgbm, proba_l, n_classes)
        if proba_s is not None:
            proba_s = pad_proba(self.svm, proba_s, n_classes)
        # proba_rf is already in full class space

        # Blend all available models
        total_weight = 0
        proba = np.zeros((len(df), n_classes))

        if proba_l is not None:
            proba += self.lgbm_weight * proba_l
            total_weight += self.lgbm_weight

        if proba_s is not None:
            proba += self.svm_weight * proba_s
            total_weight += self.svm_weight

        if proba_rf is not None:
            proba += self.rf_weight * proba_rf
            total_weight += self.rf_weight

        # Normalize weights if not all models are available
        if total_weight > 0:
            proba = proba / total_weight
        else:
            # Fallback to uniform distribution
            proba = np.ones((len(df), n_classes)) / n_classes

        return proba
