"""GBDT Hyperparameter Optimization (HPO) module.

Implements leakage-safe 2-arm A/B hyperparameter optimization for LightGBM and XGBoost:
- Control: default champion hyperparameters.
- Challenger: Optuna TPESampler searching hyperparameter space strictly on an inner
  building-grouped validation split carved from outer training data.
- Outer validation partitions are never accessed during HPO trials.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from project.models.gbdt_standalone import (
    _mock_predict_partition,
    _prepare_matrices_for_lightgbm,
    _prepare_matrices_for_xgboost,
    _to_pandas_data,
    is_gbdt_available,
)

logger = logging.getLogger(__name__)

SUPPORTED_FAMILIES: tuple[str, ...] = ("lightgbm", "xgboost")
SUPPORTED_ARMS: tuple[str, ...] = ("control", "challenger")

ARM_ALIASES: dict[str, str] = {
    "default": "control",
    "ctrl": "control",
    "hpo": "challenger",
    "chall": "challenger",
    "tuned": "challenger",
}

FAMILY_ALIASES: dict[str, str] = {
    "lgb": "lightgbm",
    "lgbm": "lightgbm",
    "xgb": "xgboost",
}

VERIFIED_NUMERIC_FEATURES: tuple[str, ...] = (
    "meter_reading",
    "square_feet",
    "year_built",
    "floor_count",
    "hour",
    "weekday",
    "month",
    "year",
    "hour_x",
    "hour_y",
    "month_x",
    "month_y",
    "weekday_x",
    "weekday_y",
    "is_holiday",
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
    "air_temperature_mean_lag7",
    "air_temperature_max_lag7",
    "air_temperature_min_lag7",
    "air_temperature_std_lag7",
    "air_temperature_mean_lag73",
    "air_temperature_max_lag73",
    "air_temperature_min_lag73",
    "air_temperature_std_lag73",
)

VERIFIED_CATEGORICAL_FEATURES: tuple[str, ...] = ("primary_use",)

VERIFIED_FEATURE_POOL: tuple[str, ...] = tuple(
    sorted([*VERIFIED_NUMERIC_FEATURES, *VERIFIED_CATEGORICAL_FEATURES])
)

FORBIDDEN_EXACT_NAMES: frozenset[str] = frozenset({
    "anomaly",
    "building_hour",
    "building_id",
    "building_meter",
    "fold",
    "id",
    "label",
    "repeat",
    "row_index",
    "seed",
    "site_id",
    "split",
    "support_stratum",
    "target",
    "timestamp",
    "y",
})

CONTROL_DEFAULTS: dict[str, dict[str, Any]] = {
    "lightgbm": {
        "learning_rate": 0.03,
        "n_estimators": 1000,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "n_jobs": 12,
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
    },
    "xgboost": {
        "learning_rate": 0.03,
        "n_estimators": 1000,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 1.0,
        "tree_method": "hist",
        "n_jobs": 12,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "enable_categorical": True,
    },
}

__all__ = (
    "ARM_ALIASES",
    "CONTROL_DEFAULTS",
    "FAMILY_ALIASES",
    "FORBIDDEN_EXACT_NAMES",
    "SUPPORTED_ARMS",
    "SUPPORTED_FAMILIES",
    "VERIFIED_CATEGORICAL_FEATURES",
    "VERIFIED_FEATURE_POOL",
    "VERIFIED_NUMERIC_FEATURES",
    "build_inner_split",
    "is_forbidden_predictor",
    "normalize_arm",
    "normalize_family",
    "optimize_hyperparameters",
    "train_hpo_partition",
    "validate_feature_pool",
)


def normalize_family(family: str) -> str:
    """Normalize and validate GBDT family identifier.

    Parameters
    ----------
    family:
        GBDT family identifier (e.g. 'lightgbm', 'xgboost', 'lgbm', 'xgb').

    Returns
    -------
    str:
        Canonical family string.
    """
    if not isinstance(family, str):
        raise TypeError(f"family must be a string, got {type(family).__name__}")
    fam = family.strip().lower()
    fam = FAMILY_ALIASES.get(fam, fam)
    if fam not in SUPPORTED_FAMILIES:
        raise ValueError(
            f"Unsupported GBDT family '{family}'. Supported families: {list(SUPPORTED_FAMILIES)}"
        )
    return fam


def normalize_arm(arm: str) -> str:
    """Normalize and validate A/B experiment arm identifier.

    Parameters
    ----------
    arm:
        Arm identifier (e.g. 'control', 'challenger', 'default', 'hpo').

    Returns
    -------
    str:
        Canonical arm string ('control' or 'challenger').
    """
    if not isinstance(arm, str):
        raise TypeError(f"arm must be a string, got {type(arm).__name__}")
    a = arm.strip().lower()
    a = ARM_ALIASES.get(a, a)
    if a not in SUPPORTED_ARMS:
        raise ValueError(
            f"Unsupported arm '{arm}'. Supported arms: {list(SUPPORTED_ARMS)}"
        )
    return a


def is_forbidden_predictor(name: str) -> bool:
    """Check if a column name is forbidden by anti-leakage or ID/metadata rules.

    Rejects:
    - Columns starting with 'gte_' (unverified global target encoding leak risk).
    - Columns ending with '_id' (site_id, building_id, user_id, etc.).
    - Exact forbidden names (timestamp, anomaly, building_hour, building_meter, etc.).
    """
    lowered = name.casefold()
    return (
        lowered in FORBIDDEN_EXACT_NAMES
        or lowered.endswith("_id")
        or lowered.startswith(("gte_", "source_", "fold_"))
    )


def validate_feature_pool(features: Sequence[str]) -> list[str]:
    """Validate predictor feature list against forbidden names and anti-leakage contracts.

    Parameters
    ----------
    features:
        Sequence of feature column names.

    Returns
    -------
    list[str]:
        Validated unique feature list.
    """
    if not features:
        raise ValueError("Predictor features list cannot be empty")

    seen: set[str] = set()
    validated: list[str] = []
    for col in features:
        if not isinstance(col, str) or not col.strip():
            raise ValueError(f"Invalid predictor feature name: {col!r}")
        if col in seen:
            raise ValueError(f"Duplicate predictor column: {col}")
        seen.add(col)
        if is_forbidden_predictor(col):
            raise ValueError(
                f"Forbidden predictor column '{col}': violates anti-leakage contract "
                f"(starts with 'gte_', ends with '_id', or is a reserved target/metadata field)"
            )
        validated.append(col)
    return validated


def build_inner_split(
    train_df: pl.DataFrame | pd.DataFrame,
    seed: int = 42,
    val_fraction: float = 0.2,
) -> tuple[pl.DataFrame | pd.DataFrame, pl.DataFrame | pd.DataFrame]:
    """Carve a building-grouped inner validation split strictly from training data.

    Guarantees:
    - Set of buildings in inner_val has zero intersection with inner_train.
    - Outer validation partition is never passed or accessed.

    Parameters
    ----------
    train_df:
        Outer training fold DataFrame (Polars or Pandas).
    seed:
        Random seed for group shuffle split.
    val_fraction:
        Fraction of buildings to allocate to inner validation (default: 0.2).

    Returns
    -------
    tuple[pl.DataFrame | pd.DataFrame, pl.DataFrame | pd.DataFrame]:
        (inner_train_df, inner_val_df) of the same type as train_df.
    """
    if not isinstance(train_df, (pl.DataFrame, pd.DataFrame)):
        raise TypeError(
            f"Expected Polars or Pandas DataFrame, got {type(train_df).__name__}"
        )
    if "building_id" not in train_df.columns:
        raise ValueError("train_df must contain 'building_id' column")

    if isinstance(train_df, pl.DataFrame):
        building_ids = train_df["building_id"].to_numpy()
    else:
        building_ids = train_df["building_id"].to_numpy()

    unique_buildings = np.unique(building_ids)
    if len(unique_buildings) < 2:
        raise ValueError("Cannot split with fewer than 2 unique buildings")

    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(
        gss.split(np.zeros(len(train_df)), groups=building_ids)
    )

    if isinstance(train_df, pl.DataFrame):
        inner_train = train_df[train_idx]
        inner_val = train_df[val_idx]
    else:
        inner_train = train_df.iloc[train_idx].copy()
        inner_val = train_df.iloc[val_idx].copy()

    return inner_train, inner_val


def optimize_hyperparameters(
    family: str,
    inner_train_df: pl.DataFrame | pd.DataFrame,
    inner_val_df: pl.DataFrame | pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "anomaly",
    n_trials: int = 15,
    seed: int = 42,
    dry_run: bool = False,
    n_jobs: int = 12,
) -> dict[str, Any]:
    """Optimize hyperparameters on inner train/val split using Optuna TPESampler.

    Objective: Maximize ROC-AUC on the inner validation partition.

    Parameters
    ----------
    family:
        GBDT family identifier: 'lightgbm' or 'xgboost'.
    inner_train_df:
        Inner training partition (Polars or Pandas).
    inner_val_df:
        Inner validation partition (Polars or Pandas).
    feature_cols:
        List of predictor feature names.
    target_col:
        Name of target column.
    n_trials:
        Number of Optuna optimization trials.
    seed:
        Random seed for TPESampler and model fitting.
    dry_run:
        If True, returns default parameters instantly without running Optuna.
    n_jobs:
        Thread count for training.

    Returns
    -------
    dict[str, Any]:
        Dictionary of best hyperparameters.
    """
    fam = normalize_family(family)
    default_cfg = dict(CONTROL_DEFAULTS[fam])
    default_cfg["n_jobs"] = n_jobs
    default_cfg["seed"] = seed

    if dry_run or not is_gbdt_available(fam):
        logger.info(
            "Optuna HPO dry-run mode active for family '%s': returning default params",
            fam,
        )
        fallback = dict(default_cfg)
        fallback["is_fallback"] = True
        return fallback

    if n_trials <= 0:
        raise ValueError(f"n_trials must be a positive integer, got {n_trials}")

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Extract pandas matrices once for inner training and validation
    X_inner_tr, y_inner_tr = _to_pandas_data(
        inner_train_df, feature_cols, target_col
    )
    X_inner_va, y_inner_va = _to_pandas_data(
        inner_val_df, feature_cols, target_col
    )

    if y_inner_tr is None or y_inner_va is None:
        raise ValueError(f"Target column '{target_col}' missing in inner split")

    # Check if inner_val has both binary classes
    unique_val_targets = np.unique(y_inner_va)
    if len(unique_val_targets) < 2:
        logger.warning(
            "Inner validation split contains only one target class; returning default params"
        )
        fallback = dict(default_cfg)
        fallback["is_fallback"] = True
        return fallback
    if fam == "lightgbm":
        import lightgbm as lgb

        X_tr_prep, X_va_prep, cat_cols = _prepare_matrices_for_lightgbm(
            X_inner_tr, X_inner_va
        )

        def objective_lgb(trial: optuna.Trial) -> float:
            trial_params: dict[str, Any] = {
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.015, 0.08, log=True
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 10, 80
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 1e-3, 10.0, log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-3, 10.0, log=True
                ),
                "n_estimators": 1000,
                "n_jobs": n_jobs,
                "random_state": seed,
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
            }
            try:
                clf = lgb.LGBMClassifier(**trial_params)
                clf.fit(
                    X_tr_prep,
                    y_inner_tr,
                    categorical_feature=cat_cols if cat_cols else "auto",
                )
                preds = clf.predict_proba(X_va_prep)[:, 1]
                if len(np.unique(y_inner_va)) < 2:
                    return 0.5
                return float(roc_auc_score(y_inner_va, preds))
            except (ValueError, RuntimeError, TypeError, KeyError) as e:
                logger.warning("LightGBM trial failed: %s", e)
                return 0.0

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        try:
            study.optimize(objective_lgb, n_trials=n_trials)
            if len(study.best_params) == 0 or study.best_value <= 0.0:
                logger.warning(
                    "Optuna trials yielded no valid parameters; falling back to default params"
                )
                fallback = dict(default_cfg)
                fallback["is_fallback"] = True
                return fallback
            best = dict(default_cfg)
            best.update(study.best_params)
            best["is_fallback"] = False
            return best
        except (optuna.exceptions.OptunaError, ValueError, RuntimeError, TypeError, KeyError) as e:
            logger.warning(
                "Optuna study optimization failed: %s; falling back to default params",
                e,
            )
            fallback = dict(default_cfg)
            fallback["is_fallback"] = True
            return fallback

    elif fam == "xgboost":
        import xgboost as xgb

        X_tr_prep, X_va_prep = _prepare_matrices_for_xgboost(
            X_inner_tr, X_inner_va
        )

        def objective_xgb(trial: optuna.Trial) -> float:
            trial_params: dict[str, Any] = {
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.015, 0.08, log=True
                ),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 1.0, 10.0
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 1e-3, 10.0, log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-3, 10.0, log=True
                ),
                "n_estimators": 1000,
                "tree_method": "hist",
                "n_jobs": n_jobs,
                "random_state": seed,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "enable_categorical": True,
            }
            try:
                clf = xgb.XGBClassifier(**trial_params)
                clf.fit(X_tr_prep, y_inner_tr)
                preds = clf.predict_proba(X_va_prep)[:, 1]
                if len(np.unique(y_inner_va)) < 2:
                    return 0.5
                return float(roc_auc_score(y_inner_va, preds))
            except (ValueError, RuntimeError, TypeError, KeyError) as e:
                logger.warning("XGBoost trial failed: %s", e)
                return 0.0

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        try:
            study.optimize(objective_xgb, n_trials=n_trials)
            if len(study.best_params) == 0 or study.best_value <= 0.0:
                logger.warning(
                    "Optuna trials yielded no valid parameters; falling back to default params"
                )
                fallback = dict(default_cfg)
                fallback["is_fallback"] = True
                return fallback
            best = dict(default_cfg)
            best.update(study.best_params)
            best["is_fallback"] = False
            return best
        except (optuna.exceptions.OptunaError, ValueError, RuntimeError, TypeError, KeyError) as e:
            logger.warning(
                "Optuna study optimization failed: %s; falling back to default params",
                e,
            )
            fallback = dict(default_cfg)
            fallback["is_fallback"] = True
            return fallback

    raise ValueError(f"Unhandled GBDT family: {fam}")


def train_hpo_partition(
    model_family: str,
    arm: str,
    train_df: pl.DataFrame | pd.DataFrame,
    val_df: pl.DataFrame | pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "anomaly",
    params: Mapping[str, Any] | None = None,
    n_trials: int = 15,
    seed: int = 42,
    dry_run: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train an HPO partition model and return validation predictions and final hyperparameters.

    Leakage-safe design:
    - Control: trains default model on train_df, evaluates on val_df.
    - Challenger: splits train_df into inner_train/inner_val, tunes hyperparameters
      via Optuna, retrains on FULL train_df with best_params, evaluates on val_df.
    - Outer val_df is strictly isolated and never accessed during HPO trials.

    Parameters
    ----------
    model_family:
        GBDT family: 'lightgbm' or 'xgboost'.
    arm:
        Arm identifier: 'control' or 'challenger'.
    train_df:
        Outer training partition (Polars or Pandas).
    val_df:
        Outer validation partition (Polars or Pandas).
    feature_cols:
        List of predictor feature names.
    target_col:
        Name of target column.
    params:
        Optional hyperparameter overrides.
    n_trials:
        Number of Optuna HPO trials (used only for challenger arm).
    seed:
        Random seed.
    dry_run:
        If True, generates mock predictions and returns default parameters.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]:
        (validation_probabilities, best_or_default_parameters)
    """
    fam = normalize_family(model_family)
    arm_norm = normalize_arm(arm)
    validated_features = validate_feature_pool(feature_cols)

    user_params = dict(params or {})
    effective_seed = int(
        user_params.get("seed", user_params.get("random_state", seed))
    )
    n_jobs = int(user_params.get("n_jobs", 12))

    X_train, y_train = _to_pandas_data(
        train_df, validated_features, target_col
    )
    X_val, _ = _to_pandas_data(val_df, validated_features, None)

    if y_train is None:
        raise ValueError(f"Target column '{target_col}' not found in training data")

    val_len = len(X_val)
    if val_len == 0:
        return np.empty(0, dtype=np.float64), {}

    if dry_run or user_params.get("mock", False):
        logger.info(
            "Executing HPO partition in dry-run/mock mode: family=%s, arm=%s",
            fam,
            arm_norm,
        )
        mock_preds = _mock_predict_partition(y_train, val_len, effective_seed)
        final_params = dict(CONTROL_DEFAULTS[fam])
        final_params.update(user_params)
        final_params["seed"] = effective_seed
        final_params["is_fallback"] = True
        return mock_preds, final_params

    # Control arm: use default parameters
    if arm_norm == "control":
        final_params = dict(CONTROL_DEFAULTS[fam])
        final_params.update(user_params)
        final_params["seed"] = effective_seed
        final_params["n_jobs"] = n_jobs
        final_params.setdefault("is_fallback", False)
    else:
        # Challenger arm: optimize on inner split
        inner_train, inner_val = build_inner_split(
            train_df, seed=effective_seed, val_fraction=0.2
        )
        final_params = optimize_hyperparameters(
            family=fam,
            inner_train_df=inner_train,
            inner_val_df=inner_val,
            feature_cols=validated_features,
            target_col=target_col,
            n_trials=n_trials,
            seed=effective_seed,
            dry_run=False,
            n_jobs=n_jobs,
        )

    # Train model on full outer training data
    if fam == "lightgbm":
        import lightgbm as lgb

        X_tr_prep, X_va_prep, cat_cols = _prepare_matrices_for_lightgbm(
            X_train, X_val
        )
        lgb_params = dict(final_params)
        lgb_params.setdefault("objective", "binary")
        lgb_params.setdefault("metric", "binary_logloss")
        lgb_params.setdefault("verbosity", -1)
        lgb_params.setdefault("subsample_freq", 1)
        lgb_params["random_state"] = effective_seed
        lgb_params["n_jobs"] = n_jobs

        # Clean non-LGBMClassifier arguments if any
        lgb_params.pop("seed", None)
        lgb_params.pop("mock", None)
        lgb_params.pop("is_fallback", None)
        clf = lgb.LGBMClassifier(**lgb_params)
        clf.fit(
            X_tr_prep,
            y_train,
            categorical_feature=cat_cols if cat_cols else "auto",
        )
        preds = clf.predict_proba(X_va_prep)[:, 1]

    elif fam == "xgboost":
        import xgboost as xgb

        X_tr_prep, X_va_prep = _prepare_matrices_for_xgboost(X_train, X_val)
        xgb_params = dict(final_params)
        xgb_params.setdefault("objective", "binary:logistic")
        xgb_params.setdefault("eval_metric", "logloss")
        xgb_params.setdefault("tree_method", "hist")
        xgb_params.setdefault("enable_categorical", True)
        xgb_params["random_state"] = effective_seed
        xgb_params["n_jobs"] = n_jobs

        # Clean non-XGBClassifier arguments if any
        xgb_params.pop("seed", None)
        xgb_params.pop("mock", None)
        xgb_params.pop("is_fallback", None)
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_tr_prep, y_train)
        preds = clf.predict_proba(X_va_prep)[:, 1]
    else:
        raise ValueError(f"Unhandled GBDT family: {fam}")

    return np.clip(preds, 0.0, 1.0), final_params
