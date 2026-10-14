"""End-to-end Kaggle test submission pipeline for champion anomaly detection models.

Generates the 8 champion submission files conforming to the competition submission contract:
- submission_01_ensemble_logit_lr_0.9008.csv (Level-2 Logit L2-Logistic Regression on 6 test probabilities)
- submission_02_lightgbm_champion_0.8946.csv (Single champion LightGBM)
- submission_03_ensemble_caruana_0.8908.csv (Level-2 Caruana Selection on 6 test probabilities)
- submission_04_xgboost_champion_0.8937.csv (Single champion XGBoost)
- submission_05_lightgbm_focal_0.8937.csv (LightGBM with Focal Loss: alpha=0.25, gamma=2.0)
- submission_06_catboost_weighted_0.8746.csv (CatBoost standalone with Logloss on 9 core features)
- submission_07_extratrees_0.8740.csv (ExtraTrees: n_estimators=300, max_depth=None, min_samples_leaf=5)
- submission_08_tabular_mlp_0.8302.csv (PyTorch Tabular MLP: 3 layers, StandardScaler control arm on train)

Submission Contract:
- Exact columns: 'row_id' (integer) and 'anomaly' (continuous float probability in [0.0, 1.0]).
- Exact row count: 1,800,567 matching sample_submission.csv order.
- Zero nulls, zero NaNs, zero infs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from project.models.ensemble_meta_learning import (
    CaruanaEnsembleSelector,
    LogitLogisticRegressionMeta,
    load_base_candidate_predictions,
)
from project.models.gbdt_loss_functions import make_focal_loss_lgb
from project.models.tabular_mlp import FoldLocalPreprocessor, _build_mlp, _require_torch

logger = logging.getLogger(__name__)

SAFE_PREDICTOR_COLUMNS: tuple[str, ...] = (
    "meter_reading",
    "square_feet",
    "year_built",
    "floor_count",
    "primary_use",
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

CATEGORICAL_PREDICTORS: tuple[str, ...] = ("primary_use",)
NUMERIC_PREDICTORS: tuple[str, ...] = tuple(
    c for c in SAFE_PREDICTOR_COLUMNS if c not in CATEGORICAL_PREDICTORS
)
CORE_GBDT_PREDICTOR_COLUMNS: tuple[str, ...] = (
    "meter_reading",
    "square_feet",
    "year_built",
    "floor_count",
    "hour",
    "weekday",
    "month",
    "year",
    "primary_use",
)
CORE_GBDT_CATEGORICAL: tuple[str, ...] = ("primary_use",)
CORE_GBDT_NUMERIC: tuple[str, ...] = tuple(
    c for c in CORE_GBDT_PREDICTOR_COLUMNS if c not in CORE_GBDT_CATEGORICAL
)


BASE_MODEL_ORDER: tuple[str, ...] = (
    "lightgbm_champion",  # matches pred_lightgbm_challenger in OOF
    "xgboost_champion",  # matches pred_xgboost_control in OOF
    "catboost_weighted",  # matches pred_catboost_control in OOF
    "extratrees",  # matches pred_extratrees_natural in OOF
    "lightgbm_focal",  # matches pred_lightgbm_focal in OOF
    "tabular_mlp",  # matches pred_mlp_control in OOF
)

SUBMISSION_SPECS: dict[str, dict[str, Any]] = {
    "submission_01_ensemble_logit_lr": {
        "filename": "submission_01_ensemble_logit_lr_0.9008.csv",
        "type": "ensemble",
        "meta_arm": "challenger_lr",
        "expected_cv_score": 0.9008,
        "description": "Level-2 Logit L2-Logistic Regression on the 6 test probabilities",
    },
    "submission_02_lightgbm_champion": {
        "filename": "submission_02_lightgbm_champion_0.8946.csv",
        "type": "single",
        "model_key": "lightgbm_champion",
        "expected_cv_score": 0.8946,
        "description": "Single champion LightGBM",
    },
    "submission_03_ensemble_caruana": {
        "filename": "submission_03_ensemble_caruana_0.8908.csv",
        "type": "ensemble",
        "meta_arm": "control",
        "expected_cv_score": 0.8908,
        "description": "Level-2 Caruana Selection on the 6 test probabilities",
    },
    "submission_04_xgboost_champion": {
        "filename": "submission_04_xgboost_champion_0.8937.csv",
        "type": "single",
        "model_key": "xgboost_champion",
        "expected_cv_score": 0.8937,
        "description": "Single champion XGBoost",
    },
    "submission_05_lightgbm_focal": {
        "filename": "submission_05_lightgbm_focal_0.8937.csv",
        "type": "single",
        "model_key": "lightgbm_focal",
        "expected_cv_score": 0.8937,
        "description": "LightGBM with Focal Loss: alpha=0.25, gamma=2.0",
    },
    "submission_06_catboost_weighted": {
        "filename": "submission_06_catboost_weighted_0.8746.csv",
        "type": "single",
        "model_key": "catboost_weighted",
        "expected_cv_score": 0.8746,
        "description": "CatBoost standalone with Logloss (no class weights) on 9 core features",
    },
    "submission_07_extratrees": {
        "filename": "submission_07_extratrees_0.8740.csv",
        "type": "single",
        "model_key": "extratrees",
        "expected_cv_score": 0.8740,
        "description": "ExtraTrees: n_estimators=300, max_depth=None, min_samples_leaf=5",
    },
    "submission_08_tabular_mlp": {
        "filename": "submission_08_tabular_mlp_0.8302.csv",
        "type": "single",
        "model_key": "tabular_mlp",
        "expected_cv_score": 0.8302,
        "description": "PyTorch Tabular MLP: 3 layers, StandardScaler control arm on train, PyTorch GPU/CUDA",
    },
}


def validate_submission_contract(
    df: pl.DataFrame | pd.DataFrame,
    expected_row_count: int | None = 1_800_567,
    expected_row_ids: Sequence[int] | None = None,
) -> None:
    """Validate that a DataFrame strictly complies with the Kaggle submission contract.

    Contract:
    - Exact columns: 'row_id' and 'anomaly'.
    - Exact row count: matches expected_row_count (if provided).
    - 'row_id': non-null, integer type, exact sequence match if expected_row_ids provided.
    - 'anomaly': continuous float in [0.0, 1.0].
    - Zero nulls, zero NaNs, zero infs.

    Raises
    ------
    ValueError
        If any submission contract requirement is violated.
    """
    if isinstance(df, pd.DataFrame):
        df_pl = pl.from_pandas(df)
    elif isinstance(df, pl.DataFrame):
        df_pl = df
    else:
        raise TypeError(f"Expected Polars or Pandas DataFrame, got {type(df).__name__}")

    cols = list(df_pl.columns)
    if cols != ["row_id", "anomaly"]:
        raise ValueError(f"Expected exact columns ['row_id', 'anomaly'], got {cols}")

    if expected_row_count is not None and df_pl.height != expected_row_count:
        raise ValueError(
            f"Row count mismatch: expected {expected_row_count}, got {df_pl.height}"
        )

    # Check nulls
    null_counts = df_pl.null_count()
    if null_counts["row_id"][0] > 0:
        raise ValueError(
            f"'row_id' column contains NaN or null ({null_counts['row_id'][0]} values)"
        )
    if null_counts["anomaly"][0] > 0:
        raise ValueError(
            f"'anomaly' column contains NaN or null ({null_counts['anomaly'][0]} values)"
        )
    # Check row_id values and order
    if expected_row_ids is not None:
        actual_ids = df_pl["row_id"].to_list()
        expected_ids_list = list(expected_row_ids)
        if actual_ids != expected_ids_list:
            raise ValueError(
                f"Submission row_id sequence mismatch with expected row_ids "
                f"(first mismatch at index {next((i for i, (a, b) in enumerate(zip(actual_ids, expected_ids_list, strict=False)) if a != b), -1)})"
            )

    # Check anomaly column properties
    anomaly_series = df_pl["anomaly"]
    anomaly_np = anomaly_series.to_numpy()

    if np.any(np.isnan(anomaly_np)):
        raise ValueError("'anomaly' column contains NaN or null values")

    if np.any(np.isinf(anomaly_np)):
        raise ValueError("'anomaly' column contains infinite values")

    min_val = float(np.min(anomaly_np))
    max_val = float(np.max(anomaly_np))

    if min_val < -1e-7 or max_val > 1.0 + 1e-7:
        raise ValueError(
            f"Probabilities out of bounds [0.0, 1.0]: min={min_val}, max={max_val}"
        )


def build_submission(
    row_ids: Sequence[int] | pl.Series | np.ndarray,
    probabilities: Sequence[float] | pl.Series | np.ndarray,
) -> pl.DataFrame:
    """Build a compliant Polars DataFrame from row IDs and continuous probabilities.

    Parameters
    ----------
    row_ids:
        Sequence of integer row IDs.
    probabilities:
        Sequence of continuous anomaly probabilities in [0.0, 1.0].

    Returns
    -------
    pl.DataFrame
        DataFrame with schema {'row_id': Int64, 'anomaly': Float64}.
    """
    row_id_arr = np.asarray(row_ids, dtype=np.int64)
    probs_arr = np.asarray(probabilities, dtype=np.float64)

    if len(row_id_arr) != len(probs_arr):
        raise ValueError(
            f"Length mismatch: row_ids length ({len(row_id_arr)}) != probabilities length ({len(probs_arr)})"
        )

    if not np.all(np.isfinite(probs_arr)):
        raise ValueError("Predicted probabilities contain non-finite values (NaN or Inf)")

    probs_clean = np.clip(probs_arr, 0.0, 1.0)

    sub_df = pl.DataFrame(
        {
            "row_id": pl.Series(row_id_arr, dtype=pl.Int64),
            "anomaly": pl.Series(probs_clean, dtype=pl.Float64),
        }
    )

    validate_submission_contract(sub_df, expected_row_count=len(row_id_arr))
    return sub_df


def save_submission(
    submission_df: pl.DataFrame,
    output_path: Path | str,
    expected_row_count: int | None = None,
) -> Path:
    """Save submission DataFrame to CSV and verify file integrity.

    Parameters
    ----------
    submission_df:
        Submission DataFrame.
    output_path:
        Destination CSV file path.
    expected_row_count:
        Optional expected row count for validation.

    Returns
    -------
    Path
        Resolved path to the saved CSV file.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    validate_submission_contract(submission_df, expected_row_count=expected_row_count)

    submission_df.write_csv(path)

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write submission file or file is empty: {path}")

    logger.info(
        "Saved submission file: %s (%d rows, %.2f MB)",
        path,
        submission_df.height,
        path.stat().st_size / (1024 * 1024),
    )
    return path


def _to_clean_pandas(
    df: pl.DataFrame | pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Extract specified feature columns into a Pandas DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df.select(list(feature_cols)).to_pandas()
    if isinstance(df, pd.DataFrame):
        return df[list(feature_cols)].copy()
    raise TypeError(f"Unsupported DataFrame type: {type(df).__name__}")


def _mock_predict_test(test_len: int, seed: int = 42) -> np.ndarray:
    """Generate deterministic mock probabilities for dry-run verification."""
    rng = np.random.default_rng(seed)
    # realistic anomaly rate around 2% with tail noise
    prior = 0.0213
    noise = rng.uniform(-0.005, 0.005, size=test_len)
    return np.clip(prior + noise, 0.0, 1.0).astype(np.float64)


# ---------------------------------------------------------------------------
# Single Model Predictions
# ---------------------------------------------------------------------------


def predict_lightgbm_champion(
    train_df: pl.DataFrame | pd.DataFrame,
    test_df: pl.DataFrame | pd.DataFrame,
    params: dict[str, Any] | None = None,
    smoke: bool = False,
    dry_run: bool = False,
    random_state: int = 42,
) -> np.ndarray:
    """Train single champion LightGBM on train_df and predict probabilities for test_df."""
    test_len = test_df.height if isinstance(test_df, pl.DataFrame) else len(test_df)
    if dry_run:
        return _mock_predict_test(test_len, seed=random_state + 1)

    import lightgbm as lgb

    p = dict(params or {})
    n_estimators = 20 if smoke else int(p.get("n_estimators", 1000))
    learning_rate = 0.1 if smoke else float(p.get("learning_rate", 0.03))

    X_train = _to_clean_pandas(train_df, CORE_GBDT_PREDICTOR_COLUMNS)
    X_test = _to_clean_pandas(test_df, CORE_GBDT_PREDICTOR_COLUMNS)
    y_train = np.asarray(train_df["anomaly"])

    for cat_col in CORE_GBDT_CATEGORICAL:
        X_train[cat_col] = X_train[cat_col].astype("category")
        X_test[cat_col] = pd.Categorical(
            X_test[cat_col], categories=X_train[cat_col].cat.categories
        )
    clf = lgb.LGBMClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=int(p.get("num_leaves", 31)),
        max_depth=int(p.get("max_depth", -1)),
        subsample=float(p.get("subsample", 0.8)),
        subsample_freq=int(p.get("subsample_freq", 1)),
        colsample_bytree=float(p.get("colsample_bytree", 0.8)),
        reg_alpha=float(p.get("reg_alpha", 0.1)),
        reg_lambda=float(p.get("reg_lambda", 1.0)),
        min_child_samples=int(p.get("min_child_samples", 20)),
        random_state=random_state,
        n_jobs=-1,
        objective="binary",
        verbosity=-1,
    )

    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    return np.clip(probs, 0.0, 1.0)


def predict_xgboost_champion(
    train_df: pl.DataFrame | pd.DataFrame,
    test_df: pl.DataFrame | pd.DataFrame,
    params: dict[str, Any] | None = None,
    smoke: bool = False,
    dry_run: bool = False,
    random_state: int = 42,
) -> np.ndarray:
    """Train single champion XGBoost on train_df and predict probabilities for test_df."""
    test_len = test_df.height if isinstance(test_df, pl.DataFrame) else len(test_df)
    if dry_run:
        return _mock_predict_test(test_len, seed=random_state + 2)

    import xgboost as xgb

    p = dict(params or {})
    n_estimators = 20 if smoke else int(p.get("n_estimators", 1000))
    learning_rate = 0.1 if smoke else float(p.get("learning_rate", 0.03))

    X_train = _to_clean_pandas(train_df, CORE_GBDT_PREDICTOR_COLUMNS)
    X_test = _to_clean_pandas(test_df, CORE_GBDT_PREDICTOR_COLUMNS)
    y_train = np.asarray(train_df["anomaly"])

    for cat_col in CORE_GBDT_CATEGORICAL:
        X_train[cat_col] = X_train[cat_col].astype("category")
        X_test[cat_col] = pd.Categorical(
            X_test[cat_col], categories=X_train[cat_col].cat.categories
        )
    clf = xgb.XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=int(p.get("max_depth", 6)),
        subsample=float(p.get("subsample", 0.8)),
        colsample_bytree=float(p.get("colsample_bytree", 0.8)),
        reg_alpha=float(p.get("reg_alpha", 0.1)),
        reg_lambda=float(p.get("reg_lambda", 1.0)),
        min_child_weight=float(p.get("min_child_weight", 1.0)),
        random_state=random_state,
        tree_method="hist",
        enable_categorical=True,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    return np.clip(probs, 0.0, 1.0)


def predict_lightgbm_focal(
    train_df: pl.DataFrame | pd.DataFrame,
    test_df: pl.DataFrame | pd.DataFrame,
    alpha: float = 0.25,
    gamma: float = 2.0,
    params: dict[str, Any] | None = None,
    smoke: bool = False,
    dry_run: bool = False,
    random_state: int = 42,
) -> np.ndarray:
    """Train LightGBM with custom Focal Loss (alpha=0.25, gamma=2.0) and predict test_df."""
    test_len = test_df.height if isinstance(test_df, pl.DataFrame) else len(test_df)
    if dry_run:
        return _mock_predict_test(test_len, seed=random_state + 3)

    import lightgbm as lgb

    p = dict(params or {})
    n_estimators = 20 if smoke else int(p.get("n_estimators", 1000))
    learning_rate = 0.1 if smoke else float(p.get("learning_rate", 0.03))

    X_train = _to_clean_pandas(train_df, SAFE_PREDICTOR_COLUMNS)
    X_test = _to_clean_pandas(test_df, SAFE_PREDICTOR_COLUMNS)
    y_train = np.asarray(train_df["anomaly"])

    for cat_col in CATEGORICAL_PREDICTORS:
        X_train[cat_col] = X_train[cat_col].astype("category")
        X_test[cat_col] = pd.Categorical(
            X_test[cat_col], categories=X_train[cat_col].cat.categories
        )
    focal_obj = make_focal_loss_lgb(alpha=alpha, gamma=gamma)

    clf = lgb.LGBMClassifier(
        objective=focal_obj,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=int(p.get("num_leaves", 31)),
        max_depth=int(p.get("max_depth", -1)),
        subsample=float(p.get("subsample", 0.8)),
        subsample_freq=int(p.get("subsample_freq", 1)),
        colsample_bytree=float(p.get("colsample_bytree", 0.8)),
        reg_alpha=float(p.get("reg_alpha", 0.1)),
        reg_lambda=float(p.get("reg_lambda", 1.0)),
        min_child_samples=int(p.get("min_child_samples", 20)),
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )

    clf.fit(X_train, y_train)
    # Predict raw margins then apply sigmoid for custom objective
    raw_logits = clf.predict(X_test, raw_score=True)
    probs = 1.0 / (1.0 + np.exp(-np.clip(raw_logits, -30.0, 30.0)))
    return np.clip(probs, 0.0, 1.0)


def predict_catboost_weighted(
    train_df: pl.DataFrame | pd.DataFrame,
    test_df: pl.DataFrame | pd.DataFrame,
    params: dict[str, Any] | None = None,
    smoke: bool = False,
    dry_run: bool = False,
    random_state: int = 42,
) -> np.ndarray:
    """Train CatBoost (Logloss, no auto_class_weights) on 9 core features and predict test_df."""
    test_len = test_df.height if isinstance(test_df, pl.DataFrame) else len(test_df)
    if dry_run:
        return _mock_predict_test(test_len, seed=random_state + 4)

    from catboost import CatBoostClassifier

    p = dict(params or {})
    iterations = 20 if smoke else int(p.get("iterations", 1000))
    learning_rate = 0.1 if smoke else float(p.get("learning_rate", 0.03))

    X_train = _to_clean_pandas(train_df, CORE_GBDT_PREDICTOR_COLUMNS)
    X_test = _to_clean_pandas(test_df, CORE_GBDT_PREDICTOR_COLUMNS)
    y_train = np.asarray(train_df["anomaly"])

    for cat_col in CORE_GBDT_CATEGORICAL:
        X_train[cat_col] = X_train[cat_col].fillna("__missing__").astype(str)
        X_test[cat_col] = X_test[cat_col].fillna("__missing__").astype(str)

    clf = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=int(p.get("depth", 6)),
        l2_leaf_reg=float(p.get("l2_leaf_reg", 3.0)),
        subsample=float(p.get("subsample", 0.8)),
        random_seed=random_state,
        thread_count=-1,
        verbose=False,
        loss_function="Logloss",
        eval_metric="Logloss",
        cat_features=list(CORE_GBDT_CATEGORICAL),
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    return np.clip(probs, 0.0, 1.0)


def predict_extratrees(
    train_df: pl.DataFrame | pd.DataFrame,
    test_df: pl.DataFrame | pd.DataFrame,
    params: dict[str, Any] | None = None,
    smoke: bool = False,
    dry_run: bool = False,
    random_state: int = 42,
) -> np.ndarray:
    """Train ExtraTrees (n_estimators=300, max_depth=None, min_samples_leaf=5) and predict test_df."""
    test_len = test_df.height if isinstance(test_df, pl.DataFrame) else len(test_df)
    if dry_run:
        return _mock_predict_test(test_len, seed=random_state + 5)

    from sklearn.ensemble import ExtraTreesClassifier

    p = dict(params or {})
    n_estimators = 10 if smoke else int(p.get("n_estimators", 300))
    raw_depth = p.get("max_depth", None)
    max_depth = 5 if smoke else (int(raw_depth) if raw_depth is not None else None)

    X_train = _to_clean_pandas(train_df, SAFE_PREDICTOR_COLUMNS)
    X_test = _to_clean_pandas(test_df, SAFE_PREDICTOR_COLUMNS)
    y_train = np.asarray(train_df["anomaly"])

    # Encode categoricals as ordinal codes; impute numerics with training medians
    for cat_col in CATEGORICAL_PREDICTORS:
        categories = pd.Index(X_train[cat_col].dropna().unique())
        X_train[cat_col] = categories.get_indexer(
            X_train[cat_col].astype(object)
        ).astype("float64")
        X_test[cat_col] = categories.get_indexer(X_test[cat_col].astype(object)).astype(
            "float64"
        )

    for num_col in NUMERIC_PREDICTORS:
        med = (
            float(X_train[num_col].median())
            if len(X_train[num_col].dropna()) > 0
            else 0.0
        )
        X_train[num_col] = X_train[num_col].fillna(med).astype("float64")
        X_test[num_col] = X_test[num_col].fillna(med).astype("float64")

    clf = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=int(p.get("min_samples_split", 2)),
        min_samples_leaf=int(p.get("min_samples_leaf", 5)),
        max_features="sqrt",
        bootstrap=False,
        random_state=random_state,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    return np.clip(probs, 0.0, 1.0)


def predict_tabular_mlp(
    train_df: pl.DataFrame | pd.DataFrame,
    test_df: pl.DataFrame | pd.DataFrame,
    params: dict[str, Any] | None = None,
    smoke: bool = False,
    dry_run: bool = False,
    random_state: int = 42,
) -> np.ndarray:
    """Train PyTorch Tabular MLP (3 layers, StandardScaler control arm) and predict test_df."""
    test_len = test_df.height if isinstance(test_df, pl.DataFrame) else len(test_df)
    if dry_run:
        return _mock_predict_test(test_len, seed=random_state + 6)

    torch = _require_torch()
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    p = dict(params or {})
    epochs = 1 if smoke else int(p.get("epochs", 15))
    batch_size = 512 if smoke else int(p.get("batch_size", 1024))
    lr = float(p.get("lr", 1e-3))

    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training Tabular MLP on device: %s", device)

    X_train_df = _to_clean_pandas(train_df, SAFE_PREDICTOR_COLUMNS)
    X_test_df = _to_clean_pandas(test_df, SAFE_PREDICTOR_COLUMNS)
    y_train = np.asarray(train_df["anomaly"], dtype=np.float32)

    # Use control arm (median imputation + StandardScaler, no missing indicators)
    preprocessor = FoldLocalPreprocessor(
        arm="control",
        numeric_cols=list(NUMERIC_PREDICTORS),
        categorical_cols=list(CATEGORICAL_PREDICTORS),
    )
    preprocessor.fit(X_train_df)

    X_train_num, X_train_cat = preprocessor.transform(X_train_df)
    X_test_num, X_test_cat = preprocessor.transform(X_test_df)

    # Build DataLoader
    train_num_t = torch.tensor(X_train_num, dtype=torch.float32)
    train_cat_t = {
        col: torch.tensor(arr, dtype=torch.long) for col, arr in X_train_cat.items()
    }
    train_y_t = torch.tensor(y_train, dtype=torch.float32)

    train_cat_primary = train_cat_t["primary_use"]
    dataset = TensorDataset(train_num_t, train_cat_primary, train_y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = _build_mlp(
        num_numeric_features=X_train_num.shape[1],
        category_vocab_sizes=preprocessor.num_categories,
        embedding_dim=8,
        hidden_dims=(256, 128, 64),
        dropout=0.2,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for ep in range(epochs):
        for b_num, b_cat, b_y in loader:
            b_num = b_num.to(device)
            b_cat_dict = {"primary_use": b_cat.to(device)}
            b_y = b_y.to(device)

            optimizer.zero_grad()
            logits = model(b_num, b_cat_dict)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

    # Predict on test
    model.eval()
    test_num_t = torch.tensor(X_test_num, dtype=torch.float32)
    test_cat_primary = torch.tensor(X_test_cat["primary_use"], dtype=torch.long)
    test_dataset = TensorDataset(test_num_t, test_cat_primary)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

    preds_list: list[np.ndarray] = []
    with torch.no_grad():
        for b_num, b_cat in test_loader:
            b_num = b_num.to(device)
            b_cat_dict = {"primary_use": b_cat.to(device)}
            logits = model(b_num, b_cat_dict)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds_list.append(probs)

    all_preds = (
        np.concatenate(preds_list)
        if preds_list
        else np.zeros(test_len, dtype=np.float64)
    )
    return np.clip(all_preds, 0.0, 1.0).astype(np.float64)

# Canonical aliases matching out-of-fold candidate names
predict_lightgbm_challenger = predict_lightgbm_champion
predict_xgboost_control = predict_xgboost_champion
predict_catboost_control = predict_catboost_weighted
predict_extratrees_natural = predict_extratrees
predict_mlp_control = predict_tabular_mlp


# ---------------------------------------------------------------------------
# Level-2 Meta-Learner Prediction
# ---------------------------------------------------------------------------


def predict_ensemble_meta(
    oof_df: pl.DataFrame,
    oof_pred_cols: list[str],
    test_prob_matrix: np.ndarray,
    arm_id: str,
    random_state: int = 42,
) -> np.ndarray:
    """Fit Level-2 meta-learner on OOF predictions and evaluate on test probability matrix.

    Parameters
    ----------
    oof_df:
        Out-of-fold DataFrame containing target and base model prediction columns.
    oof_pred_cols:
        List of base candidate prediction column names matching columns of test_prob_matrix.
    test_prob_matrix:
        2D NumPy array of shape (n_test, len(oof_pred_cols)).
    arm_id:
        'challenger_lr' for Logistic Regression on Logits, or 'control' for Caruana Selection.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        1D NumPy array of test anomaly probabilities in [0.0, 1.0].
    """
    X_oof = oof_df.select(oof_pred_cols).to_numpy()
    y_oof = oof_df["target"].to_numpy()

    if arm_id == "challenger_lr":
        meta_learner = LogitLogisticRegressionMeta(
            C=1.0,
            clip_eps=1e-7,
            standardize=True,
            max_iter=1000,
            random_state=random_state,
        )
    elif arm_id == "control":
        meta_learner = CaruanaEnsembleSelector(
            n_iterations=50,
            subsample_size=100_000,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported meta arm '{arm_id}'")

    meta_learner.fit(X_oof, y_oof)
    probs = meta_learner.predict_proba(test_prob_matrix)
    return np.clip(probs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# High-Level Pipeline Orchestrator
# ---------------------------------------------------------------------------


def generate_all_submissions(
    train_df: pl.DataFrame | pd.DataFrame | None = None,
    test_df: pl.DataFrame | pd.DataFrame | None = None,
    sample_sub_df: pl.DataFrame | None = None,
    output_dir: Path | str = "outputs/submissions",
    models_mode: str = "all",
    smoke: bool = False,
    dry_run: bool = False,
    sample_size: int | None = None,
    train_path: Path | str = "outputs/prepared/train_joined.parquet",
    test_path: Path | str = "outputs/prepared/test_joined.parquet",
    sample_sub_path: Path | str = "data/energy-anomaly-detection/sample_submission.csv",
) -> dict[str, Path]:
    """Execute end-to-end submission generation pipeline.

    Parameters
    ----------
    train_df:
        Optional training DataFrame. If None, loaded from train_path.
    test_df:
        Optional test DataFrame. If None, loaded from test_path.
    sample_sub_df:
        Optional sample submission DataFrame. If None, loaded from sample_sub_path.
    output_dir:
        Directory where CSV submissions will be saved.
    models_mode:
        One of 'all', 'ensemble', 'single'.
    smoke:
        If True, run fast smoke iterations on models.
    dry_run:
        If True, run instantaneous mock evaluations on samples.
    sample_size:
        Optional integer to truncate test set for verification.
    train_path:
        Path to prepared train Parquet.
    test_path:
        Path to prepared test Parquet.
    sample_sub_path:
        Path to sample submission CSV.

    Returns
    -------
    dict[str, Path]
        Mapping of submission specification key to generated CSV path.
    """
    out_dir = Path(output_dir).resolve()
    if (smoke or dry_run) and out_dir == Path("outputs/submissions").resolve():
        out_dir = Path("outputs/submissions/smoke").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = models_mode.strip().lower()
    if mode not in ("all", "ensemble", "single"):
        raise ValueError(
            f"models_mode must be 'all', 'ensemble', or 'single', got '{models_mode}'"
        )

    # Load test data
    if test_df is None:
        logger.info("Loading test features from %s", test_path)
        t_df = pl.read_parquet(test_path)
    elif isinstance(test_df, pd.DataFrame):
        t_df = pl.from_pandas(test_df)
    else:
        t_df = test_df

    if sample_size is not None and sample_size > 0:
        t_df = t_df.head(sample_size)
    # Determine expected row IDs
    if sample_sub_df is not None:
        expected_row_ids = sample_sub_df["row_id"].to_list()
        if sample_size is not None and sample_size > 0:
            expected_row_ids = expected_row_ids[:sample_size]
    elif test_df is not None:
        expected_row_ids = t_df["row_id"].to_list()
    elif Path(sample_sub_path).is_file():
        sample_sub = pl.read_csv(sample_sub_path)
        if sample_size is not None and sample_size > 0:
            sample_sub = sample_sub.head(sample_size)
        expected_row_ids = sample_sub["row_id"].to_list()
    else:
        expected_row_ids = t_df["row_id"].to_list()
    test_row_ids = t_df["row_id"].to_list()
    n_test = len(test_row_ids)
    logger.info("Generating submissions for %d test observations", n_test)

    # Load train data
    if train_df is None:
        if dry_run:
            logger.info(
                "Dry-run mode: creating synthetic train subset for feature alignment"
            )
            # Minimal train frame for feature inspection
            tr_dict: dict[str, Any] = {"anomaly": [0, 1, 0, 1]}
            for col in SAFE_PREDICTOR_COLUMNS:
                tr_dict[col] = (
                    ["use_1", "use_2", "use_1", "use_2"]
                    if col == "primary_use"
                    else [0.0, 1.0, 0.5, 0.8]
                )
            tr_df = pl.DataFrame(tr_dict)
        else:
            logger.info("Loading train features from %s", train_path)
            tr_df = pl.read_parquet(train_path)
            if smoke:
                tr_df = tr_df.head(2000)
    elif isinstance(train_df, pd.DataFrame):
        tr_df = pl.from_pandas(train_df)
    else:
        tr_df = train_df

    # Dictionary to store single model test probabilities
    test_probs_dict: dict[str, np.ndarray] = {}

    single_runners = {
        "lightgbm_champion": predict_lightgbm_champion,
        "xgboost_champion": predict_xgboost_champion,
        "lightgbm_focal": predict_lightgbm_focal,
        "catboost_weighted": predict_catboost_weighted,
        "extratrees": predict_extratrees,
        "tabular_mlp": predict_tabular_mlp,
    }

    # Run single models if needed with disk caching
    cache_dir = Path(out_dir) / ".cache_test_preds"
    if not (dry_run or smoke):
        cache_dir.mkdir(parents=True, exist_ok=True)

    need_singles = mode in ("all", "single") or mode == "ensemble"
    if need_singles:
        for model_key, runner in single_runners.items():
            cache_path = cache_dir / f"{model_key}.npy"
            if not (dry_run or smoke) and cache_path.is_file():
                try:
                    cached_preds = np.load(cache_path)
                    if len(cached_preds) == t_df.height:
                        logger.info(
                            "Loaded cached test predictions for %s (%d rows)",
                            model_key,
                            len(cached_preds),
                        )
                        test_probs_dict[model_key] = cached_preds
                        continue
                except (OSError, ValueError) as err:
                    logger.debug("Could not load cached predictions for %s: %s", model_key, err)

            logger.info(
                "Computing test predictions for %s (dry_run=%s, smoke=%s)...",
                model_key,
                dry_run,
                smoke,
            )
            preds = runner(tr_df, t_df, smoke=smoke, dry_run=dry_run)
            test_probs_dict[model_key] = preds
            if not (dry_run or smoke):
                np.save(cache_path, preds)
    # Stack test probabilities in canonical BASE_MODEL_ORDER:
    # 0: lightgbm_champion (pred_lightgbm_challenger)
    # 1: xgboost_champion (pred_xgboost_control)
    # 2: catboost_weighted (pred_catboost_control)
    # 3: extratrees (pred_extratrees_natural)
    # 4: lightgbm_focal (pred_lightgbm_focal)
    # 5: tabular_mlp (pred_mlp_control)
    test_prob_matrix = np.column_stack([test_probs_dict[m] for m in BASE_MODEL_ORDER])

    # If ensemble submissions needed, load OOF and fit meta-models
    ensemble_preds_dict: dict[str, np.ndarray] = {}
    if mode in ("all", "ensemble"):
        logger.info("Loading base candidate OOF predictions for meta-model fitting...")
        if dry_run:
            # Synthetic OOF in dry_run mode
            rng = np.random.default_rng(42)
            synth_oof = pl.DataFrame(
                {
                    f"pred_{m}": rng.uniform(0.01, 0.99, size=1000).tolist()
                    for m in BASE_MODEL_ORDER
                }
            ).with_columns(pl.Series("target", rng.integers(0, 2, size=1000)))
            oof_cols = [f"pred_{m}" for m in BASE_MODEL_ORDER]
            oof_df = synth_oof
        else:
            oof_df, oof_cols = load_base_candidate_predictions()

        logger.info("Fitting Level-2 Logit L2-Logistic Regression on OOF...")
        ensemble_preds_dict["challenger_lr"] = predict_ensemble_meta(
            oof_df=oof_df,
            oof_pred_cols=oof_cols,
            test_prob_matrix=test_prob_matrix,
            arm_id="challenger_lr",
        )

        logger.info("Fitting Level-2 Caruana Selection on OOF...")
        ensemble_preds_dict["control"] = predict_ensemble_meta(
            oof_df=oof_df,
            oof_pred_cols=oof_cols,
            test_prob_matrix=test_prob_matrix,
            arm_id="control",
        )

    # Build and save submissions according to specs and mode
    generated_files: dict[str, Path] = {}

    for spec_key, spec in SUBMISSION_SPECS.items():
        spec_type = spec["type"]
        if mode == "single" and spec_type != "single":
            continue
        if mode == "ensemble" and spec_type != "ensemble":
            continue

        filename = spec["filename"]
        target_path = out_dir / filename

        if spec_type == "single":
            m_key = spec["model_key"]
            probs = test_probs_dict[m_key]
        else:  # ensemble
            meta_arm = spec["meta_arm"]
            probs = ensemble_preds_dict[meta_arm]

        sub_df = build_submission(test_row_ids, probs)
        validate_submission_contract(
            sub_df,
            expected_row_count=n_test,
            expected_row_ids=expected_row_ids,
        )

        saved_path = save_submission(sub_df, target_path, expected_row_count=n_test)
        generated_files[spec_key] = saved_path

    logger.info(
        "Successfully generated %d submission files under %s",
        len(generated_files),
        out_dir,
    )
    return generated_files


def audit_submission_file(
    submission_path: Path | str,
    sample_sub_path: Path | str,
    sample_size: int | None = None,
) -> bool:
    """Audit a generated submission file against the official sample_submission.csv.

    Verifies:
    1. File exists and is non-empty.
    2. Column names strictly match ['row_id', 'anomaly'].
    3. Row count strictly matches sample_submission.csv (or sample_size).
    4. row_id sequence strictly matches sample_submission.csv's row_id sequence.
    5. All anomaly values are finite float probabilities in [0.0, 1.0].

    Parameters
    ----------
    submission_path:
        Path to the generated submission CSV to audit.
    sample_sub_path:
        Path to the official sample_submission.csv.
    sample_size:
        Optional row truncation for dry-run verification.

    Returns
    -------
    bool
        True if all audit checks pass; raises ValueError or FileNotFoundError otherwise.
    """
    sub_p = Path(submission_path).resolve()
    if not sub_p.is_file():
        raise FileNotFoundError(f"Submission file not found: {sub_p}")
    if sub_p.stat().st_size == 0:
        raise ValueError(f"Submission file is empty: {sub_p}")

    ref_p = Path(sample_sub_path).resolve()
    if not ref_p.is_file():
        raise FileNotFoundError(f"Reference sample submission file not found: {ref_p}")

    ref_df = pl.read_csv(ref_p)
    if sample_size is not None and sample_size > 0:
        ref_df = ref_df.head(sample_size)

    sub_df = pl.read_csv(sub_p)

    expected_ids = ref_df["row_id"].to_list()
    expected_count = len(expected_ids)

    validate_submission_contract(
        sub_df,
        expected_row_count=expected_count,
        expected_row_ids=expected_ids,
    )
    return True
