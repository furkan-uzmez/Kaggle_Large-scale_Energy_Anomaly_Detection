"""Follow-up experiments recommended in Notebook 04 Section 8.

Implements three targeted post-ensemble experimental pipelines:
1. ``EXP-F01``: Diurnal-Weighted Manifold Mixup
   Concentrates synthetic sample interpolation on 00:00-04:00 UTC and evening
   operational transitions (17:00-21:00 UTC) with adaptive Beta parameters.
2. ``EXP-F02``: Platt-Calibrated Champion
   Fold-local 1D logistic calibration (Platt scaling) fitted on inner out-of-fold
   logits of the champion model (LightGBM Mixup) to guarantee zero leakage.
3. ``EXP-F03``: Skip-Connection Leaf Stacking
   Combines raw standardized domain features with discretized GBDT leaf index
   embeddings into a regularized linear head.

Leakage Contract
----------------
All samplers, encoders, scalers, and calibrators fit strictly on outer training
partitions. Validation partitions are scored without resampling or modification.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project.models.gbdt_standalone import (
    MissingGbdtDependencyError,
    _prepare_matrices_for_lightgbm,
    _to_pandas_data,
    get_installation_guidance,
    is_gbdt_available,
)
from project.models.synthetic_sampling import train_synthetic_sampling_partition

logger = logging.getLogger(__name__)

TRANSITION_HOURS_DEFAULT: tuple[int, ...] = (0, 1, 2, 3, 4, 17, 18, 19, 20, 21)
DEFAULT_TARGET_RATIO: float = 0.08
DEFAULT_ALPHA_BASE: float = 0.4
DEFAULT_ALPHA_TRANS: float = 0.8
DEFAULT_TRANS_WEIGHT: float = 2.0

__all__ = [
    "DEFAULT_ALPHA_BASE",
    "DEFAULT_ALPHA_TRANS",
    "DEFAULT_TARGET_RATIO",
    "DEFAULT_TRANS_WEIGHT",
    "TRANSITION_HOURS_DEFAULT",
    "PlattCalibrator",
    "SkipConnectionLeafClassifier",
    "SkipConnectionLeafStackingEstimator",
    "apply_fold_local_diurnal_mixup",
    "compute_calibration_metrics",
    "compute_diurnal_slice_metrics",
    "run_nested_platt_calibration_partition",
    "train_diurnal_mixup_partition",
    "train_skip_leaf_partition",
]


# ==============================================================================
# Helper utilities
# ==============================================================================


def _extract_hour_series(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    hour_col: str = "hour",
) -> np.ndarray:
    """Extract UTC hour integer array (0-23) from hour or timestamp column."""
    if hour_col in df.columns:
        return np.asarray(df[hour_col].to_numpy(dtype=float) % 24, dtype=int)
    if timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col], utc=True)
        return ts.dt.hour.to_numpy(dtype=int)
    # Default fallback: uniform hour 0
    return np.zeros(len(df), dtype=int)


def _split_columns(
    df: pd.DataFrame, categorical_cols: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Partition DataFrame columns into numeric and categorical lists."""
    cat_set = set(categorical_cols)
    num_cols = [c for c in df.columns if c not in cat_set]
    cat_cols = [c for c in df.columns if c in cat_set]
    return num_cols, cat_cols


def _synthetic_count(n_pos: int, n_neg: int, target_ratio: float) -> int:
    """Number of synthetic positive rows needed to achieve target_ratio."""
    if target_ratio <= 0.0 or target_ratio >= 1.0:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")
    needed = math.ceil((target_ratio * n_neg - (1.0 - target_ratio) * n_pos) / (1.0 - target_ratio))
    return max(0, int(needed))


# ==============================================================================
# EXP-F01: Diurnal-Weighted Manifold Mixup
# ==============================================================================


def apply_fold_local_diurnal_mixup(
    X_train: pd.DataFrame,
    y_train: np.ndarray | Sequence[int],
    categorical_cols: Sequence[str],
    target_ratio: float = DEFAULT_TARGET_RATIO,
    alpha_base: float = DEFAULT_ALPHA_BASE,
    alpha_trans: float = DEFAULT_ALPHA_TRANS,
    trans_hours: Sequence[int] = TRANSITION_HOURS_DEFAULT,
    trans_weight: float = DEFAULT_TRANS_WEIGHT,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fold-local synthetic minority oversampling with diurnal concentration.

    Concentrates synthetic sample generation on 00:00-04:00 UTC and evening
    operational transition hours (trans_hours). Uses adaptive Beta parameter
    alpha(t) elevated to ``alpha_trans`` during transition windows.

    Parameters
    ----------
    X_train:
        Outer training feature DataFrame.
    y_train:
        Outer training binary labels array (0 or 1).
    categorical_cols:
        List of categorical column names.
    target_ratio:
        Desired minority proportion after oversampling (default: 0.08).
    alpha_base:
        Base Dirichlet/Beta shape parameter for non-transition hours.
    alpha_trans:
        Elevated Dirichlet/Beta shape parameter for transition hours.
    trans_hours:
        Hours of the day (0-23 UTC) considered diurnal transitions.
    trans_weight:
        Relative sampling weight for transition-hour positive instances.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Augmented (X_resampled, y_resampled). Real rows precede synthetic rows.
    """
    y = np.asarray(y_train, dtype=np.int64)
    if len(X_train) != len(y):
        raise ValueError(f"Length mismatch: len(X_train)={len(X_train)} vs len(y)={len(y)}")
    if alpha_base <= 0.0 or alpha_trans <= 0.0:
        raise ValueError(f"Alpha parameters must be positive: base={alpha_base}, trans={alpha_trans}")

    X = X_train.reset_index(drop=True)
    pos_idx = np.flatnonzero(y == 1)
    n_pos = len(pos_idx)
    n_syn = _synthetic_count(n_pos, len(y) - n_pos, target_ratio)

    if n_syn == 0 or n_pos < 2:
        return X.copy(), y.copy()

    # Extract hours for positive rows
    hours_all = _extract_hour_series(X)
    pos_hours = hours_all[pos_idx]
    trans_set = set(trans_hours)
    is_trans = np.isin(pos_hours, list(trans_set))

    # Diurnal sampling weights for positive parents
    pos_weights = np.where(is_trans, float(trans_weight), 1.0)
    pos_probs = pos_weights / np.sum(pos_weights)

    rng = np.random.default_rng(seed)

    # Sample parent pairs with replacement according to diurnal weights
    first_local = rng.choice(n_pos, size=n_syn, p=pos_probs)
    second_local = rng.choice(n_pos, size=n_syn, p=pos_probs)

    # Ensure distinct partners when possible
    same_mask = first_local == second_local
    if np.any(same_mask) and n_pos > 1:
        # Re-sample distinct second partners
        second_local[same_mask] = (second_local[same_mask] + rng.integers(1, n_pos, np.sum(same_mask))) % n_pos

    first_global = pos_idx[first_local]
    second_global = pos_idx[second_local]

    # Adaptive alpha per synthetic pair: if either parent is in transition hours, use alpha_trans
    pair_is_trans = is_trans[first_local] | is_trans[second_local]
    alphas = np.where(pair_is_trans, float(alpha_trans), float(alpha_base))

    # Draw lambda ~ Beta(alpha, alpha) vectorised
    lam = rng.beta(alphas, alphas)

    num_cols, cat_cols = _split_columns(X, categorical_cols)

    # Numeric interpolation
    if num_cols:
        pos_num = X[num_cols].to_numpy(dtype=np.float64)
        a, b = pos_num[first_global], pos_num[second_global]
        mixed = lam[:, None] * a + (1.0 - lam[:, None]) * b
        synth_num = np.where(np.isnan(a), b, np.where(np.isnan(b), a, mixed))
        synth_num_df = pd.DataFrame(synth_num, columns=num_cols)
    else:
        synth_num_df = pd.DataFrame(index=range(n_syn))

    # Categorical interpolation
    pick_first = rng.random(n_syn) < lam
    synth_cat_dict: dict[str, np.ndarray] = {}
    for col in cat_cols:
        pos_cat = X[col].to_numpy(dtype=object)
        synth_cat_dict[col] = np.where(
            pick_first, pos_cat[first_global], pos_cat[second_global]
        )
    synth_cat_df = pd.DataFrame(synth_cat_dict)

    synth_df = pd.concat([synth_num_df, synth_cat_df], axis=1)[X.columns]
    X_augmented = pd.concat([X, synth_df], ignore_index=True)
    y_augmented = np.concatenate([y, np.ones(n_syn, dtype=np.int64)])

    return X_augmented, y_augmented


def train_diurnal_mixup_partition(
    train_df: pl.DataFrame | pd.DataFrame,
    val_df: pl.DataFrame | pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    target_col: str = "anomaly",
    params: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Train LightGBM with diurnal-weighted Mixup on train_df and score val_df.

    Zero-leakage guarantee: Resampling occurs strictly inside train_df.
    """
    X_tr_raw, y_tr_raw = _to_pandas_data(train_df, feature_cols, target_col)
    X_va_raw, _ = _to_pandas_data(val_df, feature_cols, None)
    val_len = len(X_va_raw)

    if dry_run:
        rng = np.random.default_rng(seed + 101)
        prior = float(np.mean(y_tr_raw)) if y_tr_raw is not None and len(y_tr_raw) > 0 else 0.021
        noise = rng.uniform(-0.005, 0.005, size=val_len)
        return np.clip(prior + noise, 0.0, 1.0).astype(np.float64)

    if not is_gbdt_available("lightgbm"):
        raise MissingGbdtDependencyError(get_installation_guidance("lightgbm"))
    import lightgbm as lgb

    p = dict(params or {})
    target_ratio = float(p.get("target_ratio", DEFAULT_TARGET_RATIO))
    alpha_base = float(p.get("alpha_base", DEFAULT_ALPHA_BASE))
    alpha_trans = float(p.get("alpha_trans", DEFAULT_ALPHA_TRANS))
    trans_weight = float(p.get("trans_weight", DEFAULT_TRANS_WEIGHT))
    trans_hours = tuple(p.get("trans_hours", TRANSITION_HOURS_DEFAULT))

    X_syn, y_syn = apply_fold_local_diurnal_mixup(
        X_train=X_tr_raw,
        y_train=y_tr_raw,
        categorical_cols=categorical_cols,
        target_ratio=target_ratio,
        alpha_base=alpha_base,
        alpha_trans=alpha_trans,
        trans_hours=trans_hours,
        trans_weight=trans_weight,
        seed=seed,
    )

    X_tr, X_va, cat_cols = _prepare_matrices_for_lightgbm(X_syn, X_va_raw)

    lgb_defaults: dict[str, Any] = {
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
        "random_state": seed,
        "verbosity": -1,
        "n_jobs": 1,
        "objective": "binary",
    }
    sampling_keys = {"target_ratio", "alpha_base", "alpha_trans", "trans_weight", "trans_hours"}
    model_kwargs = {k: v for k, v in p.items() if k not in sampling_keys}
    final_lgb_params = {**lgb_defaults, **model_kwargs, "random_state": seed}
    clf = lgb.LGBMClassifier(**final_lgb_params)
    clf.fit(X_tr, y_syn, categorical_feature=cat_cols if cat_cols else "auto")
    preds = clf.predict_proba(X_va)
    classes = list(clf.classes_)
    if 1 in classes:
        p1 = preds[:, classes.index(1)]
    else:
        p1 = np.zeros(val_len, dtype=np.float64)
    return np.clip(p1, 0.0, 1.0).astype(np.float64)


# ==============================================================================
# EXP-F02: Platt-Calibrated Champion (Nested Zero-Leakage)
# ==============================================================================


class PlattCalibrator:
    """1D logistic calibration (Platt scaling) fitted on probabilities or logits.

    Transforms uncalibrated probabilities p via:
        z = logit(p) = log(p / (1 - p))
        P(y = 1 | z) = 1 / (1 + exp(-(a * z + b)))
    Enforces monotonicity (a >= 1e-4) to guarantee rank order preservation.
    """

    def __init__(
        self,
        c_regularization: float = 1.0,
        eps: float = 1e-7,
        dry_run: bool = False,
    ) -> None:
        self.c_regularization = float(c_regularization)
        self.eps = float(eps)
        self.dry_run = bool(dry_run)
        self.slope_: float = 1.0
        self.intercept_: float = 0.0
        self.is_fitted_: bool = False

    def _to_logit(self, p: np.ndarray) -> np.ndarray:
        p_clipped = np.clip(p, self.eps, 1.0 - self.eps)
        return np.log(p_clipped / (1.0 - p_clipped))

    def fit(
        self,
        probabilities: np.ndarray | Sequence[float],
        y_true: np.ndarray | Sequence[int],
    ) -> PlattCalibrator:
        """Fit univariate logistic regression strictly on inner out-of-fold predictions."""
        if self.dry_run:
            self.slope_ = 1.0
            self.intercept_ = 0.0
            self.is_fitted_ = True
            return self

        p_arr = np.nan_to_num(
            np.asarray(probabilities, dtype=np.float64),
            nan=0.5,
            posinf=1.0 - self.eps,
            neginf=self.eps,
        )
        y_arr = np.asarray(y_true, dtype=np.int64)

        if len(p_arr) != len(y_arr):
            raise ValueError(f"Length mismatch: {len(p_arr)} != {len(y_arr)}")

        if len(p_arr) == 0:
            self.slope_ = 1.0
            self.intercept_ = 0.0
            self.is_fitted_ = True
            return self

        unique_classes = np.unique(y_arr)
        if len(unique_classes) < 2:
            self.slope_ = 1.0
            self.intercept_ = -4.0 if 1 not in unique_classes else 4.0
            self.is_fitted_ = True
            return self

        z = self._to_logit(p_arr)
        z_std = float(np.std(z))
        z_ptp = float(np.ptp(z))

        # Handle degenerate/constant logit cases by falling back to class prevalence
        if z_std < 1e-6 or z_ptp < 1e-6:
            logger.warning(
                "Degenerate or constant logits detected in PlattCalibrator fit (std=%.2e, ptp=%.2e); "
                "falling back to prevalence baseline without NaN/Inf",
                z_std,
                z_ptp,
            )
            pos_rate = float(np.clip(np.mean(y_arr), self.eps, 1.0 - self.eps))
            self.slope_ = 1.0
            self.intercept_ = float(np.log(pos_rate / (1.0 - pos_rate)) - float(np.mean(z)))
            self.is_fitted_ = True
            return self

        z_col = z.reshape(-1, 1)

        lr = LogisticRegression(
            C=self.c_regularization,
            solver="lbfgs",
            max_iter=1000,
            fit_intercept=True,
        )
        lr.fit(z_col, y_arr)

        slope = float(lr.coef_[0, 0])
        intercept = float(lr.intercept_[0])

        # Enforce positive slope (a >= 1e-4) to preserve monotonicity
        # Re-center intercept b = logit(prevalence) - a * z_bar so base-rate is well-behaved
        if slope < 1e-4:
            logger.warning(
                "Fitted Platt slope %.4f < 1e-4; clamping to 1e-4 and re-centering intercept to prevalence",
                slope,
            )
            slope = 1e-4
            pos_rate = float(np.clip(np.mean(y_arr), self.eps, 1.0 - self.eps))
            intercept = float(np.log(pos_rate / (1.0 - pos_rate)) - slope * float(np.mean(z)))
        elif not np.isfinite(intercept):
            logger.warning(
                "Fitted Platt intercept non-finite; falling back to prevalence-aligned intercept"
            )
            pos_rate = float(np.clip(np.mean(y_arr), self.eps, 1.0 - self.eps))
            intercept = float(np.log(pos_rate / (1.0 - pos_rate)) - slope * float(np.mean(z)))

        self.slope_ = slope
        self.intercept_ = intercept
        self.is_fitted_ = True
        return self

    def transform(self, probabilities: np.ndarray | Sequence[float]) -> np.ndarray:
        """Apply fitted Platt scaling to probabilities."""
        if not self.is_fitted_:
            raise RuntimeError("PlattCalibrator is not fitted yet.")
        p_arr = np.asarray(probabilities, dtype=np.float64)
        if p_arr.size == 0:
            return np.empty(0, dtype=np.float64)
        p_arr = np.nan_to_num(p_arr, nan=0.5, posinf=1.0 - self.eps, neginf=self.eps)
        z = self._to_logit(p_arr)
        scaled_z = self.slope_ * z + self.intercept_
        scaled_z = np.nan_to_num(scaled_z, nan=0.0, posinf=30.0, neginf=-30.0)
        # Numerically stable sigmoid: 1 / (1 + exp(-scaled_z))
        calibrated = 1.0 / (1.0 + np.exp(-np.clip(scaled_z, -30.0, 30.0)))
        return np.clip(calibrated, 0.0, 1.0).astype(np.float64)


def run_nested_platt_calibration_partition(
    train_df: pl.DataFrame | pd.DataFrame,
    val_df: pl.DataFrame | pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    target_col: str = "anomaly",
    inner_folds: int = 3,
    params: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Strictly nested zero-leakage Platt calibration pipeline.

    Protocol:
    1. Outer training fold (train_df) is split into ``inner_folds`` disjoint building groups.
    2. Inner cross-validation on train_df generates genuine inner-OOF predictions.
    3. PlattCalibrator is fitted strictly on (inner_oof_preds, train_y).
    4. Champion model (LightGBM Mixup) is fit on all of train_df.
    5. Outer validation fold (val_df) is scored by champion and calibrated by PlattCalibrator.
    """
    _, y_tr_raw = _to_pandas_data(train_df, feature_cols, target_col)

    if not dry_run and not is_gbdt_available("lightgbm"):
        raise MissingGbdtDependencyError(get_installation_guidance("lightgbm"))

    p = dict(params or {})
    c_reg = float(p.get("c_regularization", 1.0))

    # Identify building grouping for inner CV if available
    train_pl = train_df if isinstance(train_df, pl.DataFrame) else pl.from_pandas(train_df)
    n_train = len(train_pl)

    # 1. Inner Cross-Validation strictly within train_df
    inner_oof_preds = np.zeros(n_train, dtype=np.float64)
    evaluated_mask = np.zeros(n_train, dtype=bool)

    if "building_id" in train_pl.columns and len(train_pl["building_id"].unique()) >= 2:
        b_ids = train_pl["building_id"].unique().to_list()
        rng = np.random.default_rng(seed)
        shuffled_b = rng.permutation(b_ids).tolist()
        inner_k = min(inner_folds, len(b_ids))
        fold_buildings = np.array_split(shuffled_b, inner_k)

        for k in range(inner_k):
            val_b = set(fold_buildings[k])
            tr_mask = ~train_pl["building_id"].is_in(list(val_b)).to_numpy()
            va_mask = ~tr_mask

            if np.sum(va_mask) == 0 or np.sum(tr_mask) == 0:
                continue

            inner_tr_df = train_pl.filter(pl.Series(tr_mask))
            inner_va_df = train_pl.filter(pl.Series(va_mask))

            inner_preds = train_synthetic_sampling_partition(
                model_family="lightgbm",
                arm="mixup",
                train_df=inner_tr_df,
                val_df=inner_va_df,
                feature_cols=feature_cols,
                categorical_cols=categorical_cols,
                target_col=target_col,
                params={**p, "seed": seed + k + 1},
                dry_run=dry_run,
            )
            inner_oof_preds[va_mask] = inner_preds
            evaluated_mask[va_mask] = True
    else:
        # Fallback to standard k-fold if building_id not present or < 2 buildings
        indices = np.arange(n_train)
        folds = np.array_split(indices, inner_folds)
        for k in range(inner_folds):
            va_idx = folds[k]
            tr_idx = np.setdiff1d(indices, va_idx)
            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue
            inner_tr_df = train_pl[tr_idx]
            inner_va_df = train_pl[va_idx]
            inner_preds = train_synthetic_sampling_partition(
                model_family="lightgbm",
                arm="mixup",
                train_df=inner_tr_df,
                val_df=inner_va_df,
                feature_cols=feature_cols,
                categorical_cols=categorical_cols,
                target_col=target_col,
                params={**p, "seed": seed + k + 1},
                dry_run=dry_run,
            )
            inner_oof_preds[va_idx] = inner_preds
            evaluated_mask[va_idx] = True

    # Assert genuine coverage over both positive and negative instances
    pos_mask = (y_tr_raw == 1)
    neg_mask = (y_tr_raw == 0)
    total_pos = int(np.sum(pos_mask))
    total_neg = int(np.sum(neg_mask))
    pos_covered = int(np.sum(pos_mask & evaluated_mask))
    neg_covered = int(np.sum(neg_mask & evaluated_mask))

    if total_pos > 0 and pos_covered == 0:
        raise AssertionError(
            f"Inner-OOF calibration predictions lack genuine coverage over positive instances "
            f"({pos_covered}/{total_pos} covered)."
        )
    if total_neg > 0 and neg_covered == 0:
        raise AssertionError(
            f"Inner-OOF calibration predictions lack genuine coverage over negative instances "
            f"({neg_covered}/{total_neg} covered)."
        )

    # 2. Fit PlattCalibrator on evaluated inner-OOF predictions
    calibrator = PlattCalibrator(c_regularization=c_reg)
    calibrator.fit(inner_oof_preds[evaluated_mask], y_tr_raw[evaluated_mask])
    logger.info(
        "Fitted Platt Calibrator: slope=%.4f intercept=%.4f (inner-OOF evaluated %d/%d, pos=%d, neg=%d)",
        calibrator.slope_,
        calibrator.intercept_,
        int(np.sum(evaluated_mask)),
        n_train,
        pos_covered,
        neg_covered,
    )
    # 3. Fit Champion on full outer train_df
    uncalibrated_val_preds = train_synthetic_sampling_partition(
        model_family="lightgbm",
        arm="mixup",
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        target_col=target_col,
        params={**p, "seed": seed},
        dry_run=dry_run,
    )

    # 4. Calibrate outer validation predictions
    calibrated_val_preds = calibrator.transform(uncalibrated_val_preds)
    return np.clip(calibrated_val_preds, 0.0, 1.0).astype(np.float64)


# ==============================================================================
# EXP-F03: Skip-Connection Leaf Stacking
# ==============================================================================


class SkipConnectionLeafStackingEstimator:
    """Combines raw standardized features with GBDT leaf index embeddings.

    Input representation to regularized linear head:
        H = [X_standardized, OneHot(GBDT_Leaves(X))]
    Linear Head:
        P(y = 1 | H) = sigmoid(H @ W + b) trained via SGDClassifier(log_loss).
    Memory Safe: Uses sparse CSR representations to stay well below 8 GB.
    """

    def __init__(
        self,
        gbdt_params: dict[str, Any] | None = None,
        linear_params: dict[str, Any] | None = None,
        categorical_cols: Sequence[str] | None = None,
        dry_run: bool = False,
        seed: int = 42,
    ) -> None:
        self.gbdt_params = dict(gbdt_params or {})
        self.linear_params = dict(linear_params or {})
        self.categorical_cols = list(categorical_cols or ["primary_use"])
        self.dry_run = bool(dry_run)
        self.seed = int(seed)

        self.scaler_: StandardScaler | None = None
        self.imputer_medians_: pd.Series | None = None
        self.leaf_model_: Any = None
        self.leaf_encoder_: OneHotEncoder | None = None
        self.linear_head_: SGDClassifier | None = None
        self.cat_categories_: dict[str, pd.Index] = {}
        self.is_fitted_: bool = False

    def _prepare_frames_fit(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        num_cols, cat_cols = _split_columns(X, self.categorical_cols)
        df = X.copy()
        self.cat_categories_ = {}
        for cat in cat_cols:
            df[cat] = df[cat].astype("category")
            self.cat_categories_[cat] = df[cat].cat.categories
        return df, num_cols, cat_cols

    def _prepare_frames_transform(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        num_cols, cat_cols = _split_columns(X, self.categorical_cols)
        df = X.copy()
        for cat in cat_cols:
            if cat in self.cat_categories_:
                cats = self.cat_categories_[cat]
                cleaned = df[cat].where(df[cat].isin(set(cats)))
                df[cat] = pd.Categorical(cleaned, categories=cats)
            else:
                df[cat] = df[cat].astype("category")
        return df, num_cols, cat_cols

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray | Sequence[int],
    ) -> SkipConnectionLeafStackingEstimator:
        """Fit scaler, leaf extractor, OneHotEncoder, and linear head strictly on training partition."""
        y_arr = np.asarray(y_train, dtype=np.int64)
        if self.dry_run:
            self.is_fitted_ = True
            return self

        if not is_gbdt_available("lightgbm"):
            raise MissingGbdtDependencyError(get_installation_guidance("lightgbm"))
        import lightgbm as lgb

        df_tr, num_cols, cat_cols = self._prepare_frames_fit(X_train)

        # 1. Fit standard scaler and median imputer for numeric skip-connections (memory-safe float32)
        if num_cols:
            num_data = df_tr[num_cols].astype(np.float32)
            self.imputer_medians_ = num_data.median().fillna(0.0).astype(np.float32)
            imputed_tr = num_data.fillna(self.imputer_medians_).astype(np.float32)
            self.scaler_ = StandardScaler(copy=False)
            scaled_num_tr = self.scaler_.fit_transform(imputed_tr).astype(np.float32)
            sparse_skip_tr = sparse.csr_matrix(scaled_num_tr, dtype=np.float32)
        else:
            sparse_skip_tr = sparse.csr_matrix((len(df_tr), 0), dtype=np.float32)
        # 2. Fit LightGBM leaf extractor
        default_gbdt = {
            "n_estimators": 50,
            "max_depth": 5,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.seed,
            "verbosity": -1,
            "n_jobs": 1,
        }
        gbdt_cfg = {**default_gbdt, **self.gbdt_params}
        self.leaf_model_ = lgb.LGBMClassifier(**gbdt_cfg)
        self.leaf_model_.fit(df_tr, y_arr, categorical_feature=cat_cols if cat_cols else "auto")

        # 3. Extract leaf activations and fit OneHotEncoder with float32
        leaves_tr = self.leaf_model_.predict_proba(df_tr, pred_leaf=True)
        self.leaf_encoder_ = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )
        sparse_leaves_tr = self.leaf_encoder_.fit_transform(leaves_tr)

        # 4. Concatenate skip-connection features and leaf embeddings (preserve float32)
        H_tr = sparse.hstack(
            [sparse_skip_tr, sparse_leaves_tr], format="csr", dtype=np.float32
        )
        # 5. Fit regularized linear head
        default_linear = {
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": 1e-4,
            "max_iter": 1000,
            "tol": 1e-3,
            "random_state": self.seed,
        }
        linear_cfg = {**default_linear, **self.linear_params}
        self.linear_head_ = SGDClassifier(**linear_cfg)
        self.linear_head_.fit(H_tr, y_arr)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict positive anomaly probabilities using the stacked skip architecture."""
        if not self.is_fitted_:
            raise RuntimeError("Estimator is not fitted yet.")

        n_samples = len(X)
        if self.dry_run:
            rng = np.random.default_rng(self.seed + 303)
            noise = rng.uniform(-0.005, 0.005, size=n_samples)
            return np.clip(0.021 + noise, 0.0, 1.0).astype(np.float64)

        df_x, num_cols, _cat_cols = self._prepare_frames_transform(X)

        # Transform numeric skip-connections in float32
        if num_cols and self.scaler_ is not None and self.imputer_medians_ is not None:
            num_data = df_x[num_cols].astype(np.float32)
            imputed_x = num_data.fillna(self.imputer_medians_).astype(np.float32)
            scaled_num_x = self.scaler_.transform(imputed_x).astype(np.float32)
            sparse_skip_x = sparse.csr_matrix(scaled_num_x, dtype=np.float32)
        else:
            sparse_skip_x = sparse.csr_matrix((n_samples, 0), dtype=np.float32)

        # Transform leaves
        leaves_x = self.leaf_model_.predict_proba(df_x, pred_leaf=True)
        sparse_leaves_x = self.leaf_encoder_.transform(leaves_x)

        # Concatenate in float32
        H_x = sparse.hstack(
            [sparse_skip_x, sparse_leaves_x], format="csr", dtype=np.float32
        )
        probs = self.linear_head_.predict_proba(H_x)
        classes = list(self.linear_head_.classes_)
        if 1 in classes:
            p1 = probs[:, classes.index(1)]
        else:
            p1 = np.zeros(n_samples, dtype=np.float64)
        return np.clip(p1, 0.0, 1.0).astype(np.float64)



# Canonical alias
SkipConnectionLeafClassifier = SkipConnectionLeafStackingEstimator

def train_skip_leaf_partition(
    train_df: pl.DataFrame | pd.DataFrame,
    val_df: pl.DataFrame | pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    target_col: str = "anomaly",
    gbdt_params: Mapping[str, Any] | None = None,
    linear_params: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Train SkipConnectionLeafStackingEstimator on train_df and predict val_df."""
    X_tr_raw, y_tr_raw = _to_pandas_data(train_df, feature_cols, target_col)
    X_va_raw, _ = _to_pandas_data(val_df, feature_cols, None)

    estimator = SkipConnectionLeafStackingEstimator(
        gbdt_params=dict(gbdt_params or {}),
        linear_params=dict(linear_params or {}),
        categorical_cols=categorical_cols,
        dry_run=dry_run,
        seed=seed,
    )
    estimator.fit(X_tr_raw, y_tr_raw)
    return estimator.predict_proba(X_va_raw)


# ==============================================================================
# Diagnostics and Calibration Metrics
# ==============================================================================


def compute_calibration_metrics(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute Expected Calibration Error (ECE), MCE, Brier Score, and LogLoss."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)

    # Brier Score
    brier = float(np.mean((p - y) ** 2))

    # LogLoss (NLL)
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p)))

    # Equal-width bins for ECE and MCE
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y)

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (p >= low) & (p <= high) if i == n_bins - 1 else (p >= low) & (p < high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = float(np.mean(y[mask]))
            bin_conf = float(np.mean(p[mask]))
            diff = abs(bin_acc - bin_conf)
            ece += (bin_count / n) * diff
            mce = max(mce, diff)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier),
        "log_loss": float(logloss),
    }


def compute_diurnal_slice_metrics(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    hours: np.ndarray | Sequence[int],
    trans_hours: Sequence[int] = TRANSITION_HOURS_DEFAULT,
) -> dict[str, Any]:
    """Compute performance metrics stratified by diurnal transition slices."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    h = np.asarray(hours, dtype=int) % 24

    trans_set = set(trans_hours)
    trans_mask = np.isin(h, list(trans_set))
    offpeak_mask = ~trans_mask

    def _safe_auc(y_sub: np.ndarray, p_sub: np.ndarray) -> float:
        if len(np.unique(y_sub)) < 2:
            return 0.5
        return float(roc_auc_score(y_sub, p_sub))

    def _safe_logloss(y_sub: np.ndarray, p_sub: np.ndarray) -> float:
        if len(y_sub) == 0:
            return 0.0
        p_c = np.clip(p_sub, 1e-7, 1.0 - 1e-7)
        return float(-np.mean(y_sub * np.log(p_c) + (1.0 - y_sub) * np.log(1.0 - p_c)))
    return {
        "transition_auc": _safe_auc(y[trans_mask], p[trans_mask]),
        "transition_logloss": _safe_logloss(y[trans_mask], p[trans_mask]),
        "transition_count": int(np.sum(trans_mask)),
        "offpeak_auc": _safe_auc(y[offpeak_mask], p[offpeak_mask]),
        "offpeak_logloss": _safe_logloss(y[offpeak_mask], p[offpeak_mask]),
        "offpeak_count": int(np.sum(offpeak_mask)),
        "global_auc": _safe_auc(y, p),
        "global_logloss": _safe_logloss(y, p),
    }
