"""Model factories for the baseline portfolio."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project.features.registry import build_lead_baseline_registry

__all__ = ["build_baseline_estimators"]


def build_baseline_estimators(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the three baseline estimators for the LEAD point-anomaly task.

    Parameters
    ----------
    config:
        Configuration mapping loaded from configs/experiments/baselines.yaml.

    Returns
    -------
    dict[str, Any]
        Mapping with keys 'dummy_prior', 'logistic_l2', and
        'hist_gradient_boosting'.
    """
    if "seed" not in config:
        raise ValueError("Config must specify top-level 'seed'.")
    seed = int(config["seed"])

    if "features" not in config or not isinstance(config["features"], Mapping):
        raise ValueError("Config must specify 'features' mapping.")
    features_cfg = config["features"]

    if "numeric" not in features_cfg or "categorical" not in features_cfg:
        raise ValueError("Config 'features' mapping must specify 'numeric' and 'categorical'.")

    numeric_cols = list(features_cfg["numeric"])
    categorical_cols = list(features_cfg["categorical"])

    # Validate against approved registry names
    registry = build_lead_baseline_registry()
    expected_categorical = {
        name
        for name in registry.names
        if (meta := registry.get(name)) is not None and meta.family.casefold() == "categorical"
    }
    expected_numeric = {
        name
        for name in registry.names
        if (meta := registry.get(name)) is not None and meta.family.casefold() != "categorical"
    }

    if set(numeric_cols) != expected_numeric:
        raise ValueError(
            f"Config numeric features {set(numeric_cols)} do not match approved numeric features {expected_numeric}"
        )
    if set(categorical_cols) != expected_categorical:
        raise ValueError(
            f"Config categorical features {set(categorical_cols)} do not match approved categorical features {expected_categorical}"
        )

    if "models" not in config or not isinstance(config["models"], Mapping):
        raise ValueError("Config must specify 'models' mapping.")
    models_cfg = config["models"]

    expected_models = {"dummy_prior", "logistic_l2", "hist_gradient_boosting"}
    if set(models_cfg.keys()) != expected_models:
        raise ValueError(
            f"Config models {set(models_cfg.keys())} do not match expected models {expected_models}"
        )

    # 1. Dummy Classifier (Prior prevalence reference)
    dummy_cfg = models_cfg["dummy_prior"]
    dummy_params = dummy_cfg.get("params", {}) if isinstance(dummy_cfg, Mapping) else {}
    strategy = dummy_params.get("strategy", "prior")
    dummy_prior = DummyClassifier(strategy=strategy, random_state=seed)

    # Common categorical transformer: most-frequent imputation + one-hot encoding
    # Unknown categories are ignored (encoded as zeros). Dense output is enforced.
    categorical_transformer = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # 2. Logistic Regression (L2 regularized linear reference)
    # Fold-local numeric median imputation + missing indicators + standard scaling
    lr_cfg = models_cfg["logistic_l2"]
    lr_params = lr_cfg.get("params", {}) if isinstance(lr_cfg, Mapping) else {}
    c_val = float(lr_params.get("C", 1.0))
    l1_ratio = float(lr_params.get("l1_ratio", 0.0))
    class_weight = lr_params.get("class_weight", None)
    max_iter = int(lr_params.get("max_iter", 1000))

    numeric_transformer_lr = Pipeline(
        [
            (
                "union",
                FeatureUnion(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        (
                            "indicator",
                            MissingIndicator(features="all", error_on_new=False),
                        ),
                    ]
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    transformers_lr: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers_lr.append(("numeric", numeric_transformer_lr, numeric_cols))
    if categorical_cols:
        transformers_lr.append(("categorical", categorical_transformer, categorical_cols))

    preprocessor_lr = ColumnTransformer(
        transformers=transformers_lr,
        remainder="drop",
        sparse_threshold=0.0,
    )

    logistic_l2 = Pipeline(
        [
            ("preprocessor", preprocessor_lr),
            (
                "classifier",
                LogisticRegression(
                    C=c_val,
                    l1_ratio=l1_ratio,
                    class_weight=class_weight,
                    max_iter=max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )

    # 3. HistGradientBoostingClassifier (Untuned scikit-learn GBDT reference)
    # Fold-local numeric median imputation + missing indicators (no scaling needed for GBDT)
    hgb_cfg = models_cfg["hist_gradient_boosting"]
    hgb_params = hgb_cfg.get("params", {}) if isinstance(hgb_cfg, Mapping) else {}
    early_stopping = bool(hgb_params.get("early_stopping", False))

    numeric_transformer_hgb = FeatureUnion(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            (
                "indicator",
                MissingIndicator(features="all", error_on_new=False),
            ),
        ]
    )

    transformers_hgb: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers_hgb.append(("numeric", numeric_transformer_hgb, numeric_cols))
    if categorical_cols:
        transformers_hgb.append(("categorical", categorical_transformer, categorical_cols))

    preprocessor_hgb = ColumnTransformer(
        transformers=transformers_hgb,
        remainder="drop",
        sparse_threshold=0.0,
    )

    hist_gradient_boosting = Pipeline(
        [
            ("preprocessor", preprocessor_hgb),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    early_stopping=early_stopping,
                    random_state=seed,
                ),
            ),
        ]
    )

    return {
        "dummy_prior": dummy_prior,
        "logistic_l2": logistic_l2,
        "hist_gradient_boosting": hist_gradient_boosting,
    }
