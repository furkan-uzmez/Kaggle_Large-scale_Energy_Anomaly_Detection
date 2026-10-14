"""Fold-local GBDT feature selection module.

Implements fold-local feature selection for LightGBM, CatBoost, and XGBoost
with strict isolation between training and validation folds.
Validation data is never accessed during feature importance calculation or ranking.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from project.models.gbdt_standalone import (
    SUPPORTED_FAMILIES,
    MissingGbdtDependencyError,
    _is_categorical_like,
    _prepare_matrices_for_catboost,
    _prepare_matrices_for_lightgbm,
    _prepare_matrices_for_xgboost,
    _to_pandas_data,
    get_installation_guidance,
    is_gbdt_available,
    normalize_family,
    train_standalone_gbdt_partition,
)

logger = logging.getLogger(__name__)

SUPPORTED_REGIMES: tuple[str, ...] = ("passthrough", "selected")

__all__ = (
    "SUPPORTED_FAMILIES",
    "SUPPORTED_REGIMES",
    "normalize_regime",
    "select_top_k_features",
    "train_fs_partition",
)


def normalize_regime(regime: str) -> str:
    """Normalize and validate feature selection regime identifier."""
    if not isinstance(regime, str):
        raise TypeError(f"regime must be a string, got {type(regime).__name__}")
    reg = regime.strip().lower()
    if reg not in SUPPORTED_REGIMES:
        raise ValueError(
            f"Unsupported regime '{regime}'. Supported regimes: {list(SUPPORTED_REGIMES)}"
        )
    return reg


def _compute_fallback_importance(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_cols: Sequence[str],
) -> np.ndarray:
    """Compute deterministic correlation/variance importance when in mock/fallback mode."""
    scores = np.zeros(len(feature_cols), dtype=np.float64)
    y_std = float(np.std(y_train))

    for idx, col in enumerate(feature_cols):
        s = X_train[col]
        if _is_categorical_like(s.dtype):
            # For categorical, use frequency variance
            val_counts = s.value_counts(normalize=True).to_numpy()
            scores[idx] = float(np.var(val_counts)) if len(val_counts) > 0 else 0.0
        else:
            arr = pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            x_std = float(np.std(arr))
            if x_std > 1e-9 and y_std > 1e-9:
                corr = float(np.corrcoef(arr, y_train)[0, 1])
                scores[idx] = abs(corr) if not np.isnan(corr) else 0.0
            else:
                scores[idx] = x_std

    return scores


def select_top_k_features(
    model_family: str,
    train_df: pl.DataFrame | pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "anomaly",
    k: int = 30,
    params: Mapping[str, Any] | None = None,
) -> list[str]:
    """Select top k features strictly on the training partition.

    Validation partitions are never passed to this function, guaranteeing
    zero target or feature leakage across CV boundaries.

    Parameters
    ----------
    model_family:
        GBDT family identifier: 'lightgbm', 'catboost', or 'xgboost'.
    train_df:
        Training fold rows (Polars or Pandas DataFrame).
    feature_cols:
        List of candidate feature column names.
    target_col:
        Name of target column.
    k:
        Number of top features to select. If k >= len(feature_cols), returns all.
    params:
        Optional training hyperparameters for importance scoring.

    Returns
    -------
    list[str]:
        Top k feature names ranked by descending fold-local importance.
    """
    fam = normalize_family(model_family)
    user_params = dict(params or {})
    all_features = list(feature_cols)

    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    if k >= len(all_features):
        return list(all_features)

    # Extract strictly train data
    X_train, y_train = _to_pandas_data(train_df, all_features, target_col)
    if y_train is None:
        raise ValueError(f"Target column '{target_col}' not found in training data")

    seed = int(user_params.get("seed", user_params.get("random_state", user_params.get("random_seed", 42))))
    is_mock = bool(user_params.get("mock", False) or user_params.get("dry_run", False))

    if is_mock or not is_gbdt_available(fam):
        importances = _compute_fallback_importance(X_train, y_train, all_features)
    else:
        # Fit small importance model strictly on train_df
        if fam == "lightgbm":
            import lightgbm as lgb

            X_tr, _, cat_cols = _prepare_matrices_for_lightgbm(X_train, X_train)
            fs_params: dict[str, Any] = {
                "objective": "binary",
                "learning_rate": user_params.get("fs_learning_rate", 0.05),
                "n_estimators": user_params.get("fs_n_estimators", 100),
                "num_leaves": user_params.get("fs_num_leaves", 31),
                "importance_type": "gain",
                "random_state": seed,
                "verbosity": -1,
                "n_jobs": user_params.get("n_jobs", 12),
            }
            model = lgb.LGBMClassifier(**fs_params)
            model.fit(X_tr, y_train, categorical_feature=cat_cols if cat_cols else "auto")
            importances = np.asarray(model.feature_importances_, dtype=np.float64)

        elif fam == "catboost":
            import catboost as cb

            X_tr, _, cat_cols = _prepare_matrices_for_catboost(X_train, X_train)
            fs_params = {
                "loss_function": "Logloss",
                "learning_rate": user_params.get("fs_learning_rate", 0.05),
                "iterations": user_params.get("fs_iterations", 100),
                "depth": user_params.get("fs_depth", 6),
                "random_seed": seed,
                "verbose": False,
                "task_type": user_params.get("task_type", "GPU"),
                "thread_count": user_params.get("thread_count", user_params.get("n_jobs", 12)),
            }
            model = cb.CatBoostClassifier(**fs_params)
            model.fit(X_tr, y_train, cat_features=cat_cols if cat_cols else None, verbose=False)
            importances = np.asarray(
                model.get_feature_importance(type="PredictionValuesChange"), dtype=np.float64
            )

        elif fam == "xgboost":
            import xgboost as xgb

            X_tr, _ = _prepare_matrices_for_xgboost(X_train, X_train)
            fs_params = {
                "objective": "binary:logistic",
                "learning_rate": user_params.get("fs_learning_rate", 0.05),
                "n_estimators": user_params.get("fs_n_estimators", 100),
                "max_depth": user_params.get("fs_max_depth", 6),
                "importance_type": "gain",
                "random_state": seed,
                "n_jobs": user_params.get("n_jobs", 12),
                "enable_categorical": True,
                "tree_method": user_params.get("tree_method", "hist"),
            }
            model = xgb.XGBClassifier(**fs_params)
            model.fit(X_tr, y_train, verbose=False)
            importances = np.asarray(model.feature_importances_, dtype=np.float64)

        else:
            raise MissingGbdtDependencyError(get_installation_guidance(fam))

    # Rank features: descending score, with deterministic feature name tie-breaking
    paired = sorted(
        zip(all_features, importances, strict=True),
        key=lambda pair: (-pair[1], pair[0]),
    )
    selected = [feat for feat, _ in paired[:k]]
    return selected


def train_fs_partition(
    model_family: str,
    regime: str,
    train_df: pl.DataFrame | pd.DataFrame,
    val_df: pl.DataFrame | pd.DataFrame,
    all_features: Sequence[str],
    target_col: str = "anomaly",
    k: int = 30,
    params: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Execute feature selection and partition training on one CV fold.

    Parameters
    ----------
    model_family:
        GBDT family: 'lightgbm', 'catboost', or 'xgboost'.
    regime:
        'passthrough' (use all features) or 'selected' (use fold-local top k).
    train_df:
        Training partition.
    val_df:
        Validation partition.
    all_features:
        Full set of available candidate features.
    target_col:
        Target column name.
    k:
        Number of top features to select if regime is 'selected'.
    params:
        Hyperparameters for model training and selection.
    dry_run:
        If True, executes mock training without library dependencies.

    Returns
    -------
    tuple[np.ndarray, list[str]]:
        Validation out-of-fold probabilities and list of selected feature names.
    """
    fam = normalize_family(model_family)
    reg = normalize_regime(regime)
    user_params = dict(params or {})

    if dry_run or user_params.get("mock", False):
        user_params["mock"] = True

    if reg == "passthrough":
        selected_features = list(all_features)
    else:
        selected_features = select_top_k_features(
            model_family=fam,
            train_df=train_df,
            feature_cols=all_features,
            target_col=target_col,
            k=k,
            params=user_params,
        )

    preds = train_standalone_gbdt_partition(
        family=fam,
        train_df=train_df,
        val_df=val_df,
        feature_cols=selected_features,
        target_col=target_col,
        params=user_params,
        dry_run=dry_run,
    )

    return preds, selected_features
