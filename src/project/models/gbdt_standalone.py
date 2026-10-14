"""Unified interface for standalone GBDT challenger models.

Provides model training and inference across LightGBM, CatBoost, and XGBoost
with native handling of missing values, binary classification logloss objective,
deterministic seeding, and graceful dry-run / mock support when packages are unavailable.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

SUPPORTED_FAMILIES: tuple[str, ...] = ("lightgbm", "catboost", "xgboost")

_FAMILY_PACKAGES: dict[str, str] = {
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "xgboost": "xgboost",
}

_INSTALLATION_GUIDANCE: dict[str, str] = {
    "lightgbm": (
        "LightGBM is not installed in the active environment. "
        "Install it via:\n"
        "  uv pip install lightgbm\n"
        "or:\n"
        "  pip install lightgbm"
    ),
    "catboost": (
        "CatBoost is not installed in the active environment. "
        "Install it via:\n"
        "  uv pip install catboost\n"
        "or:\n"
        "  pip install catboost"
    ),
    "xgboost": (
        "XGBoost is not installed in the active environment. "
        "Install it via:\n"
        "  uv pip install xgboost\n"
        "or:\n"
        "  pip install xgboost"
    ),
}


class MissingGbdtDependencyError(ImportError):
    """Raised when an optional GBDT package is required but missing."""


def normalize_family(family: str) -> str:
    """Normalize and validate GBDT family identifier."""
    if not isinstance(family, str):
        raise TypeError(f"family must be a string, got {type(family).__name__}")
    fam = family.strip().lower()
    alias_map = {
        "lgbm": "lightgbm",
        "cb": "catboost",
        "xgb": "xgboost",
    }
    fam = alias_map.get(fam, fam)
    if fam not in SUPPORTED_FAMILIES:
        raise ValueError(
            f"Unsupported GBDT family '{family}'. Supported families: {list(SUPPORTED_FAMILIES)}"
        )
    return fam


def is_gbdt_available(family: str) -> bool:
    """Check if the underlying package for a GBDT family is installed."""
    fam = normalize_family(family)
    pkg = _FAMILY_PACKAGES[fam]
    try:
        return importlib.util.find_spec(pkg) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def get_installation_guidance(family: str) -> str:
    """Return command-line installation instructions for the requested GBDT family."""
    fam = normalize_family(family)
    return _INSTALLATION_GUIDANCE[fam]


def _to_pandas_data(
    df: pl.DataFrame | pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Extract features and target as Pandas DataFrame and NumPy array."""
    if isinstance(df, pl.DataFrame):
        feat_df = df.select(list(feature_cols)).to_pandas()
        target = df.get_column(target_col).to_numpy() if target_col and target_col in df.columns else None
    elif isinstance(df, pd.DataFrame):
        feat_df = df[list(feature_cols)].copy()
        target = df[target_col].to_numpy() if target_col and target_col in df.columns else None
    else:
        raise TypeError(f"Expected Polars or Pandas DataFrame, got {type(df).__name__}")
    return feat_df, target


def _is_categorical_like(dtype: Any) -> bool:
    """Return True if dtype represents categorical, string, or object data."""
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    if dtype == object or isinstance(dtype, pd.StringDtype):
        return True
    type_str = str(dtype).lower()
    return type_str in ("str", "string") or type_str.startswith("string") or "string" in type_str


def _prepare_matrices_for_lightgbm(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Prepare feature matrices for LightGBM with categorical support."""
    cat_cols: list[str] = []
    X_tr = X_train.copy()
    X_va = X_val.copy()
    for col in X_tr.columns:
        if _is_categorical_like(X_tr[col].dtype) or _is_categorical_like(X_va[col].dtype):
            X_tr[col] = X_tr[col].astype("category")
            X_va[col] = pd.Categorical(X_va[col], categories=X_tr[col].cat.categories)
            cat_cols.append(col)
    return X_tr, X_va, cat_cols


def _prepare_matrices_for_xgboost(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare feature matrices for XGBoost with categorical support."""
    X_tr = X_train.copy()
    X_va = X_val.copy()
    for col in X_tr.columns:
        if _is_categorical_like(X_tr[col].dtype) or _is_categorical_like(X_va[col].dtype):
            X_tr[col] = X_tr[col].astype("category")
            X_va[col] = pd.Categorical(X_va[col], categories=X_tr[col].cat.categories)
    return X_tr, X_va


def _prepare_matrices_for_catboost(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Prepare feature matrices for CatBoost with string categorical support."""
    cat_cols: list[str] = []
    X_tr = X_train.copy()
    X_va = X_val.copy()
    for col in X_tr.columns:
        if _is_categorical_like(X_tr[col].dtype) or _is_categorical_like(X_va[col].dtype):
            s_tr = (
                X_tr[col].astype(object)
                if isinstance(X_tr[col].dtype, pd.CategoricalDtype)
                else X_tr[col]
            )
            s_va = (
                X_va[col].astype(object)
                if isinstance(X_va[col].dtype, pd.CategoricalDtype)
                else X_va[col]
            )
            X_tr[col] = s_tr.fillna("_missing_").astype(str)
            X_va[col] = s_va.fillna("_missing_").astype(str)
            cat_cols.append(col)
    return X_tr, X_va, cat_cols


def _mock_predict_partition(
    y_train: np.ndarray,
    val_len: int,
    seed: int,
) -> np.ndarray:
    """Generate deterministic mock probabilities for dry-run verification."""
    prior = float(np.mean(y_train)) if len(y_train) > 0 else 0.5
    prior = max(1e-4, min(1.0 - 1e-4, prior))
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-0.005, 0.005, size=val_len)
    preds = np.clip(prior + noise, 0.0, 1.0).astype(np.float64)
    return preds


def train_standalone_gbdt_partition(
    family: str,
    train_df: pl.DataFrame | pd.DataFrame,
    val_df: pl.DataFrame | pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    params: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> np.ndarray:
    """Train a standalone GBDT partition model and return validation probabilities.

    Parameters
    ----------
    family:
        GBDT family identifier: 'lightgbm', 'catboost', or 'xgboost'.
    train_df:
        Training fold rows (Polars or Pandas DataFrame).
    val_df:
        Validation fold rows (Polars or Pandas DataFrame).
    feature_cols:
        Ordered feature column names.
    target_col:
        Name of binary classification target column.
    params:
        Hyperparameters for the chosen GBDT model.
    dry_run:
        If True, generates deterministic mock probabilities without training
        or requiring external library installation.

    Returns
    -------
    np.ndarray:
        1D float64 array of validation out-of-fold positive class probabilities.
    """
    fam = normalize_family(family)
    user_params = dict(params or {})

    # Validate feature and target existence
    X_train, y_train = _to_pandas_data(train_df, feature_cols, target_col)
    X_val, _ = _to_pandas_data(val_df, feature_cols, None)

    if y_train is None:
        raise ValueError(f"Target column '{target_col}' not found in training data")

    val_len = len(X_val)
    if val_len == 0:
        return np.empty(0, dtype=np.float64)

    seed = int(user_params.get("seed", user_params.get("random_state", user_params.get("random_seed", 42))))

    if dry_run or user_params.get("mock", False):
        logger.info(
            "Executing partition in dry-run/mock mode for family '%s' (non_comparative=True, n03_eligible=False)",
            fam,
        )
        return _mock_predict_partition(y_train, val_len, seed)

    if not is_gbdt_available(fam):
        raise MissingGbdtDependencyError(get_installation_guidance(fam))

    # Dispatch to installed backend
    if fam == "lightgbm":
        import lightgbm as lgb

        X_tr, X_va, cat_cols = _prepare_matrices_for_lightgbm(X_train, X_val)
        lgb_params: dict[str, Any] = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": user_params.get("learning_rate", 0.03),
            "n_estimators": user_params.get("n_estimators", 1000),
            "num_leaves": user_params.get("num_leaves", 31),
            "max_depth": user_params.get("max_depth", -1),
            "subsample": user_params.get("subsample", 0.8),
            "subsample_freq": user_params.get("subsample_freq", 1),
            "colsample_bytree": user_params.get("colsample_bytree", 0.8),
            "reg_alpha": user_params.get("reg_alpha", 0.1),
            "reg_lambda": user_params.get("reg_lambda", 1.0),
            "min_child_samples": user_params.get("min_child_samples", 20),
            "random_state": seed,
            "verbosity": -1,
            "n_jobs": user_params.get("n_jobs", 1),
        }
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_tr,
            y_train,
            categorical_feature=cat_cols if cat_cols else "auto",
        )
        preds = model.predict_proba(X_va)[:, 1]

    elif fam == "catboost":
        import catboost as cb

        X_tr, X_va, cat_cols = _prepare_matrices_for_catboost(X_train, X_val)
        cb_params: dict[str, Any] = {
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "learning_rate": user_params.get("learning_rate", 0.03),
            "iterations": user_params.get("iterations", user_params.get("n_estimators", 1000)),
            "depth": user_params.get("depth", user_params.get("max_depth", 6)),
            "l2_leaf_reg": user_params.get("l2_leaf_reg", user_params.get("reg_lambda", 3.0)),
            "subsample": user_params.get("subsample", 0.8),
            "bootstrap_type": "Bernoulli" if user_params.get("task_type", "GPU") == "GPU" else "MVS",
            "random_seed": seed,
            "verbose": False,
            "task_type": user_params.get("task_type", "GPU"),
            "thread_count": user_params.get("thread_count", user_params.get("n_jobs", 12)),
        }
        model = cb.CatBoostClassifier(**cb_params)
        model.fit(
            X_tr,
            y_train,
            cat_features=cat_cols if cat_cols else None,
            verbose=False,
        )
        preds = model.predict_proba(X_va)[:, 1]

    elif fam == "xgboost":
        import xgboost as xgb

        X_tr, X_va = _prepare_matrices_for_xgboost(X_train, X_val)
        xgb_params: dict[str, Any] = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": user_params.get("learning_rate", 0.03),
            "n_estimators": user_params.get("n_estimators", 1000),
            "max_depth": user_params.get("max_depth", 6),
            "subsample": user_params.get("subsample", 0.8),
            "colsample_bytree": user_params.get("colsample_bytree", 0.8),
            "reg_alpha": user_params.get("reg_alpha", 0.1),
            "reg_lambda": user_params.get("reg_lambda", 1.0),
            "min_child_weight": user_params.get("min_child_weight", 1.0),
            "random_state": seed,
            "n_jobs": user_params.get("n_jobs", 12),
            "enable_categorical": True,
            "tree_method": user_params.get("tree_method", "hist"),
        }
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr,
            y_train,
            verbose=False,
        )
        preds = model.predict_proba(X_va)[:, 1]

    else:
        raise ValueError(f"Unhandled GBDT family: {fam}")

    result = np.asarray(preds, dtype=np.float64)
    if result.shape != (val_len,):
        raise ValueError(f"Expected prediction shape ({val_len},), got {result.shape}")
    return np.clip(result, 0.0, 1.0)
