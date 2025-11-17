"""Feature pipeline construction for PXBlendSC-RF strategy."""

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import (
    AddCombinedText,
    AmountFeaturizer,
    CrossFeaturizer,
    DateFeaturizer,
)
from .text_processing import make_generic_normalizer


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
