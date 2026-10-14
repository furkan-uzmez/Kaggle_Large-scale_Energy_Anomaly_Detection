"""Loss objective architectures and custom loss functions for GBDT models.

Provides mathematically rigorous, numerically stable custom objective functions
for binary classification:
- Arm 1: logloss (standard unweighted binary logloss)
- Arm 2: weighted (class-weighted binary loss penalizing false negatives 5x-10x)
- Arm 3: focal (focal loss with gamma=2.0, alpha=0.25 down-weighting easy background samples)

Supports LightGBM, CatBoost, and XGBoost across the frozen 4x3 repeated split manifest.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from project.models.gbdt_standalone import (
    SUPPORTED_FAMILIES,
    MissingGbdtDependencyError,
    _prepare_matrices_for_catboost,
    _prepare_matrices_for_lightgbm,
    _prepare_matrices_for_xgboost,
    get_installation_guidance,
    is_gbdt_available,
    normalize_family,
)

logger = logging.getLogger(__name__)

SUPPORTED_LOSS_ARMS: tuple[str, ...] = ("logloss", "weighted", "focal")

DEFAULT_FOCAL_ALPHA: float = 0.25
DEFAULT_FOCAL_GAMMA: float = 2.0
DEFAULT_WEIGHTED_POS_WEIGHT: float = 8.0
HESSIAN_MIN_FLOOR: float = 1e-6

__all__ = [
    "DEFAULT_FOCAL_ALPHA",
    "DEFAULT_FOCAL_GAMMA",
    "DEFAULT_WEIGHTED_POS_WEIGHT",
    "HESSIAN_MIN_FLOOR",
    "SUPPORTED_FAMILIES",
    "SUPPORTED_LOSS_ARMS",
    "CatBoostFocalObjective",
    "binary_focal_loss",
    "binary_focal_loss_grad_hess",
    "focal_loss_lgb",
    "focal_loss_lgb_eval",
    "focal_loss_xgb",
    "focal_loss_xgb_eval",
    "get_loss_arm_params",
    "make_focal_loss_lgb",
    "make_focal_loss_lgb_eval",
    "make_focal_loss_xgb",
    "make_focal_loss_xgb_eval",
    "mock_predict_loss_partition",
    "normalize_family",
    "normalize_loss",
    "train_loss_objective_partition",
]


def normalize_loss(loss: str) -> str:
    """Normalize and validate loss arm identifier.

    Parameters
    ----------
    loss:
        Loss name or alias (e.g. 'logloss', 'weighted', 'focal').

    Returns
    -------
    str:
        Canonical loss arm identifier: 'logloss', 'weighted', or 'focal'.
    """
    if not isinstance(loss, str):
        raise TypeError(f"loss must be a string, got {type(loss).__name__}")
    norm = loss.strip().lower()
    alias_map = {
        "logloss": "logloss",
        "log_loss": "logloss",
        "binary_logloss": "logloss",
        "binary": "logloss",
        "ce": "logloss",
        "cross_entropy": "logloss",
        "weighted": "weighted",
        "weighted_loss": "weighted",
        "class_weighted": "weighted",
        "pos_weighted": "weighted",
        "focal": "focal",
        "focal_loss": "focal",
    }
    canonical = alias_map.get(norm, norm)
    if canonical not in SUPPORTED_LOSS_ARMS:
        raise ValueError(
            f"Unsupported loss arm '{loss}'. Supported loss arms: {list(SUPPORTED_LOSS_ARMS)}"
        )
    return canonical


def binary_focal_loss_grad_hess(
    preds: np.ndarray,
    targets: np.ndarray,
    alpha: float = DEFAULT_FOCAL_ALPHA,
    gamma: float = DEFAULT_FOCAL_GAMMA,
    hess_min: float = HESSIAN_MIN_FLOOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute first and second derivatives of binary focal loss w.r.t. raw margins (logits).

    Mathematical Formulation:
    For logit z, probability p = sigmoid(z) = 1 / (1 + exp(-z)), q = 1 - p:
    Loss:
        y = 1: L_1(z) = -alpha * (1 - p)^gamma * log(p)
        y = 0: L_0(z) = -(1 - alpha) * p^gamma * log(1 - p)

    First derivative (gradient g = dL / dz):
        y = 1: g_1 = alpha * (1 - p)^gamma * [ gamma * p * log(p) + p - 1 ]
        y = 0: g_0 = (1 - alpha) * p^gamma * [ -gamma * (1 - p) * log(1 - p) + p ]

    Second derivative (hessian h = dg / dz = d^2 L / dz^2):
        y = 1: h_1 = alpha * p * (1 - p)^gamma * [ (2*gamma + 1)*(1 - p) + gamma * (1 - (gamma + 1)*p) * log(p) ]
        y = 0: h_0 = (1 - alpha) * (1 - p) * p^gamma * [ (2*gamma + 1)*p + gamma * ((gamma + 1)*p - gamma) * log(1 - p) ]

    Strictly positive clipping is applied to the hessian:
        h = max(h, hess_min)
    guaranteeing finite, non-zero denominator in leaf weight updates: -sum(g) / (sum(h) + lambda).

    Parameters
    ----------
    preds:
        Array of predicted raw logits (margins) z.
    targets:
        Array of binary targets y in {0, 1}.
    alpha:
        Positive class focal weighting coefficient (default 0.25).
    gamma:
        Focusing parameter down-weighting easy background samples (default 2.0).
    hess_min:
        Strictly positive minimum threshold for hessian (default 1e-6).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]:
        (gradient, hessian) as 1D float64 NumPy arrays.
    """
    z = np.asarray(preds, dtype=np.float64).ravel()
    y = np.asarray(targets, dtype=np.float64).ravel()

    if len(z) != len(y):
        raise ValueError(
            f"Shape mismatch: preds length ({len(z)}) does not match targets length ({len(y)})"
        )

    # Numerically stable sigmoid computation
    z_clipped = np.clip(z, -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-z_clipped))
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    q = 1.0 - p

    log_p = np.log(p)
    log_q = np.log(q)

    # y == 1 derivatives
    q_gamma = q**gamma
    g1 = alpha * q_gamma * (gamma * p * log_p + p - 1.0)
    h1 = (
        alpha
        * p
        * q_gamma
        * ((2.0 * gamma + 1.0) * q + gamma * (1.0 - (gamma + 1.0) * p) * log_p)
    )

    # y == 0 derivatives
    p_gamma = p**gamma
    g0 = (1.0 - alpha) * p_gamma * (-gamma * q * log_q + p)
    h0 = (
        (1.0 - alpha)
        * q
        * p_gamma
        * ((2.0 * gamma + 1.0) * p + gamma * ((gamma + 1.0) * p - gamma) * log_q)
    )

    grad = np.where(y > 0.5, g1, g0)
    hess = np.where(y > 0.5, h1, h0)

    # Guard against NaN/Inf and strictly floor hessian
    grad = np.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float64)
    hess = np.nan_to_num(hess, nan=hess_min, posinf=1.0, neginf=hess_min)
    hess = np.clip(hess, hess_min, None).astype(np.float64)

    return grad, hess


def binary_focal_loss(
    preds: np.ndarray,
    targets: np.ndarray,
    alpha: float = DEFAULT_FOCAL_ALPHA,
    gamma: float = DEFAULT_FOCAL_GAMMA,
) -> float:
    """Compute mean binary focal loss value from logits or probabilities.

    Parameters
    ----------
    preds:
        Predicted raw logits z.
    targets:
        Binary targets y in {0, 1}.
    alpha:
        Positive class focal weighting coefficient.
    gamma:
        Focusing parameter.

    Returns
    -------
    float:
        Scalar mean focal loss across all samples.
    """
    z = np.asarray(preds, dtype=np.float64).ravel()
    y = np.asarray(targets, dtype=np.float64).ravel()

    z_clipped = np.clip(z, -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-z_clipped))
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    q = 1.0 - p

    loss_pos = -alpha * (q**gamma) * np.log(p)
    loss_neg = -(1.0 - alpha) * (p**gamma) * np.log(q)
    loss = np.where(y > 0.5, loss_pos, loss_neg)
    return float(np.mean(loss))


def _extract_preds_and_targets(arg1: Any, arg2: Any) -> tuple[np.ndarray, np.ndarray]:
    """Robustly extract (preds, targets) handling both LGB/XGB callback argument orderings."""
    # Check if arg2 is a dataset object providing get_label
    if hasattr(arg2, "get_label"):
        return np.asarray(arg1, dtype=np.float64), np.asarray(
            arg2.get_label(), dtype=np.float64
        )
    # Check if arg1 is a dataset object
    if hasattr(arg1, "get_label"):
        return np.asarray(arg2, dtype=np.float64), np.asarray(
            arg1.get_label(), dtype=np.float64
        )

    a1 = np.asarray(arg1, dtype=np.float64)
    a2 = np.asarray(arg2, dtype=np.float64)

    # Distinguish targets (strictly binary in {0, 1}) from continuous logits
    u1 = np.unique(a1)
    u2 = np.unique(a2)
    is_a1_binary = len(u1) <= 2 and np.all(np.isin(u1, [0.0, 1.0]))
    is_a2_binary = len(u2) <= 2 and np.all(np.isin(u2, [0.0, 1.0]))

    if is_a2_binary and not is_a1_binary:
        return a1, a2
    if is_a1_binary and not is_a2_binary:
        return a2, a1

    # Default convention: arg1 is preds, arg2 is targets
    return a1, a2


def focal_loss_lgb(
    preds: np.ndarray,
    train_data: Any,
    alpha: float = DEFAULT_FOCAL_ALPHA,
    gamma: float = DEFAULT_FOCAL_GAMMA,
) -> tuple[np.ndarray, np.ndarray]:
    """LightGBM custom focal loss objective callback.

    Signature compatible with LightGBM:
        fobj(preds, train_data) -> (grad, hess)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]:
        (grad, hess) of focal loss w.r.t. margins.
    """
    pred_arr, target_arr = _extract_preds_and_targets(preds, train_data)
    return binary_focal_loss_grad_hess(pred_arr, target_arr, alpha=alpha, gamma=gamma)


def focal_loss_lgb_eval(
    preds: np.ndarray,
    eval_data: Any,
    alpha: float = DEFAULT_FOCAL_ALPHA,
    gamma: float = DEFAULT_FOCAL_GAMMA,
) -> tuple[str, float, bool]:
    """LightGBM custom focal loss evaluation metric callback.

    Returns
    -------
    tuple[str, float, bool]:
        ('focal_loss', scalar_loss, is_higher_better=False).
    """
    pred_arr, target_arr = _extract_preds_and_targets(preds, eval_data)
    val = binary_focal_loss(pred_arr, target_arr, alpha=alpha, gamma=gamma)
    return "focal_loss", val, False


def make_focal_loss_lgb(
    alpha: float = DEFAULT_FOCAL_ALPHA,
    gamma: float = DEFAULT_FOCAL_GAMMA,
) -> Callable[[Any, Any], tuple[np.ndarray, np.ndarray]]:
    """Factory creating a LightGBM scikit-learn objective callback with specific hyperparameters.

    Signature compatible with LightGBM LGBMClassifier:
        obj(y_true, y_pred) -> (grad, hess)
    """

    def obj(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(y_true, dtype=np.float64)
        preds = np.asarray(y_pred, dtype=np.float64)
        return binary_focal_loss_grad_hess(
            preds=preds, targets=targets, alpha=alpha, gamma=gamma
        )

    return obj


def make_focal_loss_lgb_eval(
    alpha: float = DEFAULT_FOCAL_ALPHA,
    gamma: float = DEFAULT_FOCAL_GAMMA,
) -> Callable[[np.ndarray, Any], tuple[str, float, bool]]:
    """Factory creating a LightGBM evaluation metric callback with specific hyperparameters."""

    def _eval(preds: np.ndarray, eval_data: Any) -> tuple[str, float, bool]:
        return focal_loss_lgb_eval(preds, eval_data, alpha=alpha, gamma=gamma)

    return _eval


def focal_loss_xgb(
    preds: np.ndarray,
    dtrain: Any,
    alpha: float = DEFAULT_FOCAL_ALPHA,
    gamma: float = DEFAULT_FOCAL_GAMMA,
) -> tuple[np.ndarray, np.ndarray]:
    """XGBoost custom focal loss objective callback.

    Signature compatible with XGBoost:
        obj(preds, dtrain) -> (grad, hess)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]:
        (grad, hess) of focal loss w.r.t. margins.
    """
    pred_arr, target_arr = _extract_preds_and_targets(preds, dtrain)
    return binary_focal_loss_grad_hess(pred_arr, target_arr, alpha=alpha, gamma=gamma)


def focal_loss_xgb_eval(
    preds: np.ndarray,
    dmatrix: Any,
    alpha: float = DEFAULT_FOCAL_ALPHA,
    gamma: float = DEFAULT_FOCAL_GAMMA,
) -> tuple[str, float]:
    """XGBoost custom focal loss evaluation metric callback.

    Returns
    -------
    tuple[str, float]:
        ('focal_loss', scalar_loss).
    """
    pred_arr, target_arr = _extract_preds_and_targets(preds, dmatrix)
    val = binary_focal_loss(pred_arr, target_arr, alpha=alpha, gamma=gamma)
    return "focal_loss", val


def make_focal_loss_xgb(
    alpha: float = DEFAULT_FOCAL_ALPHA,
    gamma: float = DEFAULT_FOCAL_GAMMA,
) -> Callable[[Any, Any], tuple[np.ndarray, np.ndarray]]:
    """Factory creating an XGBoost scikit-learn objective callback with specific hyperparameters.

    Signature compatible with XGBoost XGBClassifier:
        obj(y_true, y_pred) -> (grad, hess)
    """

    def obj(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(y_true, dtype=np.float64)
        preds = np.asarray(y_pred, dtype=np.float64)
        return binary_focal_loss_grad_hess(
            preds=preds, targets=targets, alpha=alpha, gamma=gamma
        )

    return obj


def make_focal_loss_xgb_eval(
    alpha: float = DEFAULT_FOCAL_ALPHA,
    gamma: float = DEFAULT_FOCAL_GAMMA,
) -> Callable[[np.ndarray, Any], tuple[str, float]]:
    """Factory creating an XGBoost evaluation metric callback with specific hyperparameters."""

    def _eval(preds: np.ndarray, dmatrix: Any) -> tuple[str, float]:
        return focal_loss_xgb_eval(preds, dmatrix, alpha=alpha, gamma=gamma)

    return _eval


class CatBoostFocalObjective:
    """CatBoost custom objective callback class for Focal Loss."""

    def __init__(
        self,
        alpha: float = DEFAULT_FOCAL_ALPHA,
        gamma: float = DEFAULT_FOCAL_GAMMA,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma

    def calc_ders_range(
        self,
        approxes: Sequence[float],
        targets: Sequence[float],
        weights: Sequence[float] | None = None,
    ) -> list[tuple[float, float]]:
        """Compute derivatives for CatBoost custom objective.

        CatBoost convention:
        ders1 = -gradient
        ders2 = -hessian (CatBoost maximizes negative loss)
        """
        approx_arr = np.asarray(approxes, dtype=np.float64)
        target_arr = np.asarray(targets, dtype=np.float64)
        grad, hess = binary_focal_loss_grad_hess(
            approx_arr, target_arr, alpha=self.alpha, gamma=self.gamma
        )
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            grad = grad * w
            hess = hess * w
        # CatBoost maximizes negative loss, so return (-g, -h)
        return [(-float(g), -float(h)) for g, h in zip(grad, hess, strict=False)]


def get_loss_arm_params(
    family: str,
    loss: str,
    base_params: Mapping[str, Any] | None = None,
    pos_weight: float = DEFAULT_WEIGHTED_POS_WEIGHT,
    alpha: float = DEFAULT_FOCAL_ALPHA,
    gamma: float = DEFAULT_FOCAL_GAMMA,
) -> dict[str, Any]:
    """Dispatch hyperparameters and objectives for a specific GBDT family and loss arm.

    Parameters
    ----------
    family:
        GBDT family: 'lightgbm', 'catboost', or 'xgboost'.
    loss:
        Loss arm: 'logloss', 'weighted', or 'focal'.
    base_params:
        Base model hyperparameters to clone and adapt.
    pos_weight:
        Scale weight for the positive class under the 'weighted' arm (default 8.0).
    alpha:
        Focal loss alpha coefficient under the 'focal' arm (default 0.25).
    gamma:
        Focal loss gamma focusing parameter under the 'focal' arm (default 2.0).

    Returns
    -------
    dict[str, Any]:
        Configured hyperparameters dict ready for partition training.
    """
    fam = normalize_family(family)
    arm = normalize_loss(loss)
    params = dict(base_params or {})

    if fam == "lightgbm":
        if arm == "logloss":
            params["objective"] = "binary"
            params["metric"] = "binary_logloss"
            params.pop("scale_pos_weight", None)
        elif arm == "weighted":
            params["objective"] = "binary"
            params["metric"] = "binary_logloss"
            params["scale_pos_weight"] = float(pos_weight)
        elif arm == "focal":
            params["objective"] = make_focal_loss_lgb(alpha=alpha, gamma=gamma)
            params["metric"] = "None"
            params["alpha"] = alpha
            params["gamma"] = gamma
            params.pop("scale_pos_weight", None)

    elif fam == "catboost":
        if arm == "logloss":
            params["loss_function"] = "Logloss"
            params["eval_metric"] = "Logloss"
            params.pop("class_weights", None)
            params.pop("scale_pos_weight", None)
        elif arm == "weighted":
            params["loss_function"] = "Logloss"
            params["eval_metric"] = "Logloss"
            params["class_weights"] = [1.0, float(pos_weight)]
        elif arm == "focal":
            # CatBoost native focal loss parameter format
            params["loss_function"] = f"Focal:focal_alpha={alpha};focal_gamma={gamma}"
            params["eval_metric"] = "Logloss"
            params["alpha"] = alpha
            params["gamma"] = gamma
            params.pop("class_weights", None)

    elif fam == "xgboost":
        if arm == "logloss":
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
            params.pop("scale_pos_weight", None)
        elif arm == "weighted":
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
            params["scale_pos_weight"] = float(pos_weight)
        elif arm == "focal":
            params["objective"] = make_focal_loss_xgb(alpha=alpha, gamma=gamma)
            params["eval_metric"] = "logloss"
            params["alpha"] = alpha
            params["gamma"] = gamma
            params.pop("scale_pos_weight", None)

    return params


def mock_predict_loss_partition(
    y_train: np.ndarray,
    val_len: int,
    seed: int,
    family: str = "lightgbm",
    loss: str = "logloss",
) -> np.ndarray:
    """Generate deterministic mock probabilities for dry-run verification.

    Incorporates family and loss arm identity into the pseudo-random generator
    so that distinct configurations generate consistent, deterministic, and
    differentiated probabilities.
    """
    fam = normalize_family(family)
    loss_arm = normalize_loss(loss)
    prior = float(np.mean(y_train)) if len(y_train) > 0 else 0.5
    prior = max(1e-4, min(1.0 - 1e-4, prior))

    # Incorporate loss arm characteristics into prior
    if loss_arm == "weighted":
        prior = min(0.95, prior * 1.5)
    elif loss_arm == "focal":
        prior = max(0.01, prior * 0.9)

    fam_offsets = {"lightgbm": 100, "catboost": 200, "xgboost": 300}
    loss_offsets = {"logloss": 1, "weighted": 2, "focal": 3}
    effective_seed = (int(seed) * 1000 + fam_offsets[fam] + loss_offsets[loss_arm]) % (
        2**31 - 1
    )

    rng = np.random.default_rng(effective_seed)
    noise = rng.uniform(-0.005, 0.005, size=val_len)
    return np.clip(prior + noise, 0.0, 1.0).astype(np.float64)


def _to_pandas_data(
    df: pl.DataFrame | pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Extract features and target as Pandas DataFrame and NumPy array."""
    if isinstance(df, pl.DataFrame):
        feat_df = df.select(list(feature_cols)).to_pandas()
        target = (
            df.get_column(target_col).to_numpy()
            if target_col and target_col in df.columns
            else None
        )
    elif isinstance(df, pd.DataFrame):
        feat_df = df[list(feature_cols)].copy()
        target = (
            df[target_col].to_numpy()
            if target_col and target_col in df.columns
            else None
        )
    else:
        raise TypeError(f"Expected Polars or Pandas DataFrame, got {type(df).__name__}")
    return feat_df, target


def train_loss_objective_partition(
    family: str,
    loss: str,
    train_df: pl.DataFrame | pd.DataFrame,
    val_df: pl.DataFrame | pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    params: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> np.ndarray:
    """Train a single partition model using the specified loss objective arm.

    Parameters
    ----------
    family:
        GBDT family: 'lightgbm', 'catboost', or 'xgboost'.
    loss:
        Loss arm: 'logloss', 'weighted', or 'focal'.
    train_df:
        Training fold rows (Polars or Pandas DataFrame).
    val_df:
        Validation fold rows (Polars or Pandas DataFrame).
    feature_cols:
        Ordered feature column names.
    target_col:
        Target column name.
    params:
        Base hyperparameters.
    dry_run:
        If True, deterministic mock probabilities are generated without
        requiring external library execution.

    Returns
    -------
    np.ndarray:
        1D float64 array of predicted positive probabilities for validation partition.
    """
    fam = normalize_family(family)
    loss_arm = normalize_loss(loss)
    user_params = dict(params or {})

    X_train, y_train = _to_pandas_data(train_df, feature_cols, target_col)
    X_val, _ = _to_pandas_data(val_df, feature_cols, None)

    if y_train is None:
        raise ValueError(f"Target column '{target_col}' not found in training data")

    val_len = len(X_val)
    if val_len == 0:
        return np.empty(0, dtype=np.float64)

    seed = int(
        user_params.get(
            "seed",
            user_params.get("random_state", user_params.get("random_seed", 42)),
        )
    )

    if dry_run or user_params.get("mock", False):
        logger.info(
            "Executing partition in dry-run/mock mode: family='%s', loss='%s' (non_comparative=True, n03_eligible=False)",
            fam,
            loss_arm,
        )
        return mock_predict_loss_partition(
            y_train, val_len, seed, family=fam, loss=loss_arm
        )

    if not is_gbdt_available(fam):
        raise MissingGbdtDependencyError(get_installation_guidance(fam))

    # Resolve loss-specific hyperparameters
    pos_weight = float(
        user_params.get(
            "pos_weight",
            user_params.get("scale_pos_weight", DEFAULT_WEIGHTED_POS_WEIGHT),
        )
    )
    alpha = float(user_params.get("alpha", DEFAULT_FOCAL_ALPHA))
    gamma = float(user_params.get("gamma", DEFAULT_FOCAL_GAMMA))

    arm_params = get_loss_arm_params(
        fam,
        loss_arm,
        base_params=user_params,
        pos_weight=pos_weight,
        alpha=alpha,
        gamma=gamma,
    )

    if fam == "lightgbm":
        import lightgbm as lgb

        X_tr, X_va, cat_cols = _prepare_matrices_for_lightgbm(X_train, X_val)
        lgb_params = {
            "learning_rate": arm_params.get("learning_rate", 0.03),
            "n_estimators": arm_params.get("n_estimators", 1000),
            "num_leaves": arm_params.get("num_leaves", 31),
            "max_depth": arm_params.get("max_depth", -1),
            "subsample": arm_params.get("subsample", 0.8),
            "subsample_freq": arm_params.get("subsample_freq", 1),
            "colsample_bytree": arm_params.get("colsample_bytree", 0.8),
            "reg_alpha": arm_params.get("reg_alpha", 0.1),
            "reg_lambda": arm_params.get("reg_lambda", 1.0),
            "min_child_samples": arm_params.get("min_child_samples", 20),
            "random_state": seed,
            "verbosity": -1,
            "n_jobs": arm_params.get("n_jobs", 12),
            "objective": arm_params.get("objective", "binary"),
        }
        if "scale_pos_weight" in arm_params:
            lgb_params["scale_pos_weight"] = arm_params["scale_pos_weight"]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_tr,
            y_train,
            categorical_feature=cat_cols if cat_cols else "auto",
        )
        if loss_arm == "focal":
            # For custom objective in LightGBM, raw scores are margins; sigmoid converts to probabilities
            raw_scores = model.predict(X_va, raw_score=True)
            preds = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -30.0, 30.0)))
        else:
            preds = model.predict_proba(X_va)[:, 1]

    elif fam == "catboost":
        import catboost as cb

        X_tr, X_va, cat_cols = _prepare_matrices_for_catboost(X_train, X_val)
        cb_params = {
            "learning_rate": arm_params.get("learning_rate", 0.03),
            "iterations": arm_params.get(
                "iterations", arm_params.get("n_estimators", 1000)
            ),
            "depth": arm_params.get("depth", arm_params.get("max_depth", 6)),
            "l2_leaf_reg": arm_params.get(
                "l2_leaf_reg", arm_params.get("reg_lambda", 3.0)
            ),
            "subsample": arm_params.get("subsample", 0.8),
            "bootstrap_type": "Bernoulli" if (arm_params.get("task_type", "GPU") == "GPU" and loss_arm != "focal") else "MVS",
            "task_type": "CPU" if loss_arm == "focal" else arm_params.get("task_type", "GPU"),
            "thread_count": arm_params.get("thread_count", arm_params.get("n_jobs", 12)),
            "loss_function": arm_params.get("loss_function", "Logloss"),
            "eval_metric": arm_params.get("eval_metric", "Logloss"),
        }
        if "class_weights" in arm_params:
            cb_params["class_weights"] = arm_params["class_weights"]

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
        xgb_params = {
            "learning_rate": arm_params.get("learning_rate", 0.03),
            "n_estimators": arm_params.get("n_estimators", 1000),
            "max_depth": arm_params.get("max_depth", 6),
            "subsample": arm_params.get("subsample", 0.8),
            "colsample_bytree": arm_params.get("colsample_bytree", 0.8),
            "reg_alpha": arm_params.get("reg_alpha", 0.1),
            "reg_lambda": arm_params.get("reg_lambda", 1.0),
            "min_child_weight": arm_params.get("min_child_weight", 1.0),
            "random_state": seed,
            "tree_method": arm_params.get("tree_method", "hist"),
            "n_jobs": arm_params.get("n_jobs", 12),
            "objective": arm_params.get("objective", "binary:logistic"),
            "eval_metric": arm_params.get("eval_metric", "logloss"),
            "enable_categorical": True,
        }
        if "scale_pos_weight" in arm_params:
            xgb_params["scale_pos_weight"] = arm_params["scale_pos_weight"]

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_train, verbose=False)
        if loss_arm == "focal":
            raw_scores = model.predict(X_va, output_margin=True)
            preds = 1.0 / (1.0 + np.exp(-np.clip(raw_scores, -30.0, 30.0)))
        else:
            preds = model.predict_proba(X_va)[:, 1]

    else:
        raise ValueError(f"Unhandled GBDT family: {fam}")

    return np.clip(preds, 0.0, 1.0).astype(np.float64)
