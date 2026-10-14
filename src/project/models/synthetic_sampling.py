"""Fold-local synthetic minority sampling for LightGBM and ExtraTrees.

Three experimental arms are supported:

- ``natural``: control, the training fold is used untouched (~2% positives).
- ``smote``: SMOTENC-style oversampling of the positive class to ``target_ratio``.
- ``mixup``: Tabular Mixup oversampling of the positive class to ``target_ratio``.

Leakage contract
----------------
Synthetic rows are generated *only* from the outer-training fold (``train_df``).
The validation fold never reaches a sampler: it is converted to a feature matrix
and scored by the fitted model without being resized, resampled or modified.

Tabular Mixup
-------------
For two distinct positives :math:`x_1, x_2` and
:math:`\\lambda \\sim \\mathrm{Beta}(\\alpha, \\alpha)`:

- numeric features: :math:`x = \\lambda x_1 + (1 - \\lambda) x_2`
- categorical features: take ``x_1`` with probability :math:`\\lambda`, else ``x_2``
"""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from project.models.gbdt_standalone import (
    MissingGbdtDependencyError,
    _prepare_matrices_for_lightgbm,
    _to_pandas_data,
    get_installation_guidance,
    is_gbdt_available,
)

logger = logging.getLogger(__name__)

SUPPORTED_FAMILIES: tuple[str, ...] = ("lightgbm", "extratrees")
SUPPORTED_ARMS: tuple[str, ...] = ("natural", "smote", "mixup")

DEFAULT_TARGET_RATIO: float = 0.08
DEFAULT_K_NEIGHBORS: int = 5
DEFAULT_MIXUP_ALPHA: float = 0.4

_MISSING_SENTINEL = "__missing__"
# Squared distance contributed by a categorical mismatch is ``2 * w**2 == 1``.
_CATEGORICAL_MISMATCH_WEIGHT: float = math.sqrt(0.5)

_FAMILY_ALIASES: dict[str, str] = {
    "lgbm": "lightgbm",
    "lgb": "lightgbm",
    "et": "extratrees",
    "extra_trees": "extratrees",
    "extratree": "extratrees",
}
_ARM_ALIASES: dict[str, str] = {
    "control": "natural",
    "smotenc": "smote",
    "tabular_mixup": "mixup",
}

FORBIDDEN_EXACT_NAMES: frozenset[str] = frozenset(
    {
        "anomaly",
        "building_id",
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
    }
)

__all__ = [
    "DEFAULT_K_NEIGHBORS",
    "DEFAULT_MIXUP_ALPHA",
    "DEFAULT_TARGET_RATIO",
    "SUPPORTED_ARMS",
    "SUPPORTED_FAMILIES",
    "apply_arm_sampling",
    "apply_fold_local_smote",
    "apply_fold_local_tabular_mixup",
    "is_forbidden_predictor",
    "mock_predict_sampling_partition",
    "normalize_arm",
    "normalize_family",
    "train_synthetic_sampling_partition",
    "validate_predictor_cols",
]


# ==============================================================================
# Identifier normalisation and predictor validation
# ==============================================================================


def normalize_family(family: str) -> str:
    """Return the canonical model family (``lightgbm`` or ``extratrees``)."""
    if not isinstance(family, str):
        raise TypeError(f"family must be a string, got {type(family).__name__}")
    norm = family.strip().lower()
    norm = _FAMILY_ALIASES.get(norm, norm)
    if norm not in SUPPORTED_FAMILIES:
        raise ValueError(
            f"Unsupported model family '{family}'. "
            f"Supported families: {list(SUPPORTED_FAMILIES)}"
        )
    return norm


def normalize_arm(arm: str) -> str:
    """Return the canonical sampling arm (``natural``, ``smote`` or ``mixup``)."""
    if not isinstance(arm, str):
        raise TypeError(f"arm must be a string, got {type(arm).__name__}")
    norm = arm.strip().lower()
    norm = _ARM_ALIASES.get(norm, norm)
    if norm not in SUPPORTED_ARMS:
        raise ValueError(
            f"Unsupported sampling arm '{arm}'. Supported arms: {list(SUPPORTED_ARMS)}"
        )
    return norm


def is_forbidden_predictor(name: str) -> bool:
    """Return True when a column name violates the anti-leakage contract."""
    lowered = name.casefold()
    return (
        lowered in FORBIDDEN_EXACT_NAMES
        or lowered.endswith("_id")
        or lowered.startswith(("gte_", "source_", "fold_"))
    )


def validate_predictor_cols(predictor_cols: Sequence[str]) -> list[str]:
    """Validate predictors: non-empty, unique, no ID/target/``gte_*`` columns."""
    if not predictor_cols:
        raise ValueError("Predictor columns list cannot be empty")
    seen: set[str] = set()
    validated: list[str] = []
    for col in predictor_cols:
        if not isinstance(col, str) or not col.strip():
            raise ValueError(f"Invalid predictor column name: {col!r}")
        if col in seen:
            raise ValueError(f"Duplicate predictor column: {col}")
        seen.add(col)
        if is_forbidden_predictor(col):
            raise ValueError(
                f"Forbidden predictor column '{col}': violates anti-leakage contract "
                "(starts with 'gte_', ends with '_id', or is a reserved target/metadata field)"
            )
        validated.append(col)
    return validated


# ==============================================================================
# Sampling helpers
# ==============================================================================


def _validate_sampling_inputs(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    categorical_cols: Sequence[str],
    target_ratio: float,
) -> np.ndarray:
    """Validate sampler inputs and return labels as a 1D int64 array."""
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")
    y = np.asarray(y_train)
    if y.ndim != 1 or len(y) != len(X_train):
        raise ValueError(
            f"y_train must be 1D with len(X_train)={len(X_train)} rows, got shape {y.shape}"
        )
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_train must be binary with labels in {0, 1}")
    missing = [c for c in categorical_cols if c not in X_train.columns]
    if missing:
        raise ValueError(f"categorical_cols not found in X_train: {missing}")
    return y.astype(np.int64)


def _synthetic_count(n_pos: int, n_neg: int, target_ratio: float) -> int:
    """Smallest ``s`` such that ``(P + s) / (P + N + s) >= target_ratio``."""
    needed = (target_ratio * (n_pos + n_neg) - n_pos) / (1.0 - target_ratio)
    return max(0, math.ceil(needed))


def _like_series(values: np.ndarray, like: pd.Series) -> pd.Series:
    """Build a Series of ``values`` carrying the dtype/categories of ``like``."""
    dtype = like.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return pd.Series(
            pd.Categorical(values, categories=dtype.categories, ordered=dtype.ordered)
        )
    return pd.Series(values, dtype=object).astype(dtype)


def _append_synthetic(
    X: pd.DataFrame,
    y: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    synth_num: np.ndarray,
    synth_cat: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Append synthetic positives to the real rows (real rows come first)."""
    base = X.copy()
    if numeric_cols:
        base[numeric_cols] = base[numeric_cols].astype("float64")
    synth = pd.DataFrame(synth_num, columns=numeric_cols)
    for col in categorical_cols:
        synth[col] = _like_series(synth_cat[col], X[col])
    synth = synth[list(X.columns)]
    out = pd.concat([base, synth], ignore_index=True)
    y_out = np.concatenate([y, np.ones(len(synth), dtype=np.int64)])
    return out, y_out


def _positive_numeric(
    X: pd.DataFrame, pos_idx: np.ndarray, numeric_cols: list[str]
) -> np.ndarray:
    """Return positives' numeric block as float64 with NaN for missing values."""
    return (
        X.iloc[pos_idx][numeric_cols]
        .to_numpy(dtype="float64", na_value=np.nan)
        .reshape(len(pos_idx), len(numeric_cols))
    )


def _split_columns(
    X: pd.DataFrame, categorical_cols: Sequence[str]
) -> tuple[list[str], list[str]]:
    cat_cols = list(categorical_cols)
    cat_set = set(cat_cols)
    num_cols = [c for c in X.columns if c not in cat_set]
    # Fail closed: any column in the numeric pool that cannot be cast to float64
    # means an undeclared string/object column slipped through the allowlist.
    bad = [
        c
        for c in num_cols
        if not pd.api.types.is_numeric_dtype(X[c])
    ]
    if bad:
        raise ValueError(
            f"_split_columns: columns {bad} are non-numeric but not listed in "
            f"categorical_cols={cat_cols!r}. Add them to categorical_cols or remove "
            "them from the predictor allowlist."
        )
    return num_cols, cat_cols


# ==============================================================================
# Tabular Mixup
# ==============================================================================


def apply_fold_local_tabular_mixup(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    categorical_cols: Sequence[str],
    target_ratio: float = DEFAULT_TARGET_RATIO,
    alpha: float = DEFAULT_MIXUP_ALPHA,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Oversample positives with vectorized Tabular Mixup until ``target_ratio``.

    Synthetic rows are appended after the (unchanged) real rows. Numeric
    columns are float64 in the result. A missing numeric value in one parent
    is replaced by the other parent's value (NaN only when both are missing).

    Returns the input rows unchanged when fewer than two positives exist or the
    positive rate already meets ``target_ratio``.
    """
    y = _validate_sampling_inputs(X_train, y_train, categorical_cols, target_ratio)
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    X = X_train.reset_index(drop=True)
    pos_idx = np.flatnonzero(y == 1)
    n_pos = len(pos_idx)
    n_syn = _synthetic_count(n_pos, len(y) - n_pos, target_ratio)
    if n_syn == 0:
        return X.copy(), y
    if n_pos < 2:
        logger.warning("Mixup needs >= 2 positives (got %d); returning unchanged", n_pos)
        return X.copy(), y

    num_cols, cat_cols = _split_columns(X, categorical_cols)
    rng = np.random.default_rng(seed)
    first = rng.integers(0, n_pos, n_syn)
    second = rng.integers(0, n_pos - 1, n_syn)
    second += second >= first  # distinct partner
    lam = rng.beta(alpha, alpha, n_syn)

    pos_num = _positive_numeric(X, pos_idx, num_cols)
    a, b = pos_num[first], pos_num[second]
    mixed = lam[:, None] * a + (1.0 - lam[:, None]) * b
    synth_num = np.where(np.isnan(a), b, np.where(np.isnan(b), a, mixed))

    pick_first = rng.random(n_syn) < lam
    synth_cat: dict[str, np.ndarray] = {}
    for col in cat_cols:
        pos_cat = X[col].iloc[pos_idx].to_numpy(dtype=object)
        synth_cat[col] = np.where(pick_first, pos_cat[first], pos_cat[second])

    return _append_synthetic(X, y, num_cols, cat_cols, synth_num, synth_cat)


# ==============================================================================
# SMOTENC
# ==============================================================================


def _load_smotenc() -> Any | None:
    """Return ``imblearn``'s SMOTENC class, or None when unavailable."""
    try:
        module = importlib.import_module("imblearn.over_sampling")
    except ImportError:
        return None
    return module.SMOTENC


def _native_smote_samples(
    pos_num: np.ndarray,
    pos_cat: Mapping[str, np.ndarray],
    n_syn: int,
    k_neighbors: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """SMOTENC-style k-NN interpolation among positives (no imblearn needed).

    Distances use standardized numerics plus a one-hot categorical block with
    unit squared mismatch penalty. Numerics are interpolated between a random
    positive and one of its k nearest positive neighbours; each categorical
    takes the most frequent value among that positive's neighbours (ties random).
    """
    from sklearn.neighbors import NearestNeighbors

    n_pos = pos_num.shape[0]
    k = min(k_neighbors, n_pos - 1)

    with np.errstate(all="ignore"):
        medians = np.nan_to_num(np.nanmedian(pos_num, axis=0), nan=0.0)
        stds = np.nan_to_num(np.nanstd(pos_num, axis=0), nan=1.0)
    stds = np.where(stds > 0.0, stds, 1.0)
    filled = np.where(np.isnan(pos_num), medians, pos_num)
    blocks = [(filled - medians) / stds]

    cat_codes: dict[str, np.ndarray] = {}
    cat_levels: dict[str, np.ndarray] = {}
    for col, values in pos_cat.items():
        codes, uniques = pd.factorize(pd.Series(values, dtype=object), use_na_sentinel=False)
        cat_codes[col] = codes
        cat_levels[col] = np.asarray(uniques, dtype=object)
        onehot = np.zeros((n_pos, len(uniques)), dtype=np.float64)
        onehot[np.arange(n_pos), codes] = _CATEGORICAL_MISMATCH_WEIGHT
        blocks.append(onehot)

    design = np.hstack(blocks) if blocks else np.zeros((n_pos, 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(design)
    neighbours = nn.kneighbors(design, return_distance=False)[:, 1:]

    base = rng.integers(0, n_pos, n_syn)
    partner = neighbours[base, rng.integers(0, k, n_syn)]
    gap = rng.random((n_syn, 1))
    a, b = pos_num[base], pos_num[partner]
    with np.errstate(invalid="ignore"):
        interp = a + gap * (b - a)
    synth_num = np.where(np.isnan(a) | np.isnan(b), a, interp)

    synth_cat: dict[str, np.ndarray] = {}
    rows = np.arange(n_syn)
    for col, codes in cat_codes.items():
        n_levels = len(cat_levels[col])
        neigh_codes = codes[neighbours[base]]
        counts = np.zeros((n_syn, n_levels), dtype=np.float64)
        for j in range(neigh_codes.shape[1]):
            counts[rows, neigh_codes[:, j]] += 1.0
        counts += rng.random(counts.shape) * 0.5  # random tie-break
        synth_cat[col] = cat_levels[col][counts.argmax(axis=1)]
    return synth_num, synth_cat


def _imblearn_smote_samples(
    smotenc_cls: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
    n_syn: int,
    k_neighbors: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate synthetic positives with ``imblearn.over_sampling.SMOTENC``.

    Missing values are imputed (numeric median, categorical sentinel) only in
    the copy handed to SMOTENC; only the appended synthetic rows are used.
    """
    n_rows = len(X)
    n_pos = int(y.sum())
    imputed = pd.DataFrame(index=X.index)
    for col in X.columns:
        if col in cat_cols:
            raw = X[col].astype(object)
            imputed[col] = raw.where(raw.notna(), _MISSING_SENTINEL).astype(str)
        else:
            numeric = X[col].astype("float64")
            median = numeric.median()
            imputed[col] = numeric.fillna(0.0 if pd.isna(median) else median)
    cat_idx = [list(X.columns).index(c) for c in cat_cols]

    sampler = smotenc_cls(
        categorical_features=cat_idx,
        sampling_strategy={1: n_pos + n_syn},
        k_neighbors=min(k_neighbors, n_pos - 1),
        random_state=seed,
    )
    resampled, y_res = sampler.fit_resample(imputed, y)
    if len(resampled) != n_rows + n_syn or len(y_res) != n_rows + n_syn:
        raise RuntimeError(
            f"SMOTENC returned {len(resampled)} rows; expected {n_rows + n_syn}"
        )
    synthetic = pd.DataFrame(resampled).iloc[n_rows:].reset_index(drop=True)
    synthetic.columns = list(X.columns)

    synth_num = synthetic[num_cols].to_numpy(dtype="float64").reshape(n_syn, len(num_cols))
    synth_cat: dict[str, np.ndarray] = {}
    for col in cat_cols:
        lookup: dict[str, Any] = {}
        for value in X[col].dropna().unique():
            lookup[str(value)] = value
        lookup[_MISSING_SENTINEL] = np.nan
        synth_cat[col] = np.array(
            [lookup.get(str(v), np.nan) for v in synthetic[col]], dtype=object
        )
    return synth_num, synth_cat


def apply_fold_local_smote(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    categorical_cols: Sequence[str],
    target_ratio: float = DEFAULT_TARGET_RATIO,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Oversample positives with SMOTENC until ``target_ratio``.

    Uses ``imblearn.over_sampling.SMOTENC`` when installed (and at least one
    categorical column is declared); otherwise a native k-NN interpolation with
    the same semantics. Synthetic rows are appended after the real rows and
    numeric columns are float64 in the result.

    Returns the input rows unchanged when fewer than two positives exist or the
    positive rate already meets ``target_ratio``.
    """
    y = _validate_sampling_inputs(X_train, y_train, categorical_cols, target_ratio)
    if k_neighbors < 1:
        raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
    X = X_train.reset_index(drop=True)
    pos_idx = np.flatnonzero(y == 1)
    n_pos = len(pos_idx)
    n_syn = _synthetic_count(n_pos, len(y) - n_pos, target_ratio)
    if n_syn == 0:
        return X.copy(), y
    if n_pos < 2:
        logger.warning("SMOTE needs >= 2 positives (got %d); returning unchanged", n_pos)
        return X.copy(), y

    num_cols, cat_cols = _split_columns(X, categorical_cols)
    smotenc_cls = _load_smotenc() if cat_cols else None
    if smotenc_cls is not None:
        synth_num, synth_cat = _imblearn_smote_samples(
            smotenc_cls, X, y, num_cols, cat_cols, n_syn, k_neighbors, seed
        )
    else:
        pos_cat = {c: X[c].iloc[pos_idx].to_numpy(dtype=object) for c in cat_cols}
        synth_num, synth_cat = _native_smote_samples(
            _positive_numeric(X, pos_idx, num_cols),
            pos_cat,
            n_syn,
            k_neighbors,
            np.random.default_rng(seed),
        )
    return _append_synthetic(X, y, num_cols, cat_cols, synth_num, synth_cat)


def apply_arm_sampling(
    arm: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    categorical_cols: Sequence[str],
    target_ratio: float = DEFAULT_TARGET_RATIO,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    alpha: float = DEFAULT_MIXUP_ALPHA,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply the arm's fold-local sampler (``natural`` returns the input as-is)."""
    arm = normalize_arm(arm)
    if arm == "smote":
        return apply_fold_local_smote(
            X_train, y_train, categorical_cols, target_ratio, k_neighbors, seed
        )
    if arm == "mixup":
        return apply_fold_local_tabular_mixup(
            X_train, y_train, categorical_cols, target_ratio, alpha, seed
        )
    return X_train.reset_index(drop=True), np.asarray(y_train).astype(np.int64)


# ==============================================================================
# Models
# ==============================================================================


def mock_predict_sampling_partition(
    y_train: np.ndarray,
    val_len: int,
    seed: int,
    family: str = "lightgbm",
    arm: str = "natural",
) -> np.ndarray:
    """Deterministic mock probabilities for dry-run verification.

    The prior is the (resampled) training positive rate, so sampling arms yield
    visibly different outputs from the natural control.
    """
    fam = normalize_family(family)
    arm_id = normalize_arm(arm)
    prior = float(np.mean(y_train)) if len(y_train) > 0 else 0.5
    prior = max(1e-4, min(1.0 - 1e-4, prior))
    fam_offsets = {"lightgbm": 100, "extratrees": 200}
    arm_offsets = {"natural": 1, "smote": 2, "mixup": 3}
    effective_seed = (int(seed) * 1000 + fam_offsets[fam] + arm_offsets[arm_id]) % (
        2**31 - 1
    )
    rng = np.random.default_rng(effective_seed)
    noise = rng.uniform(-0.005, 0.005, size=val_len)
    return np.clip(prior + noise, 0.0, 1.0).astype(np.float64)


def _positive_proba(model: Any, proba: np.ndarray) -> np.ndarray:
    """Extract P(y=1) even when training saw a single class."""
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(proba.shape[0], dtype=np.float64)


def _fit_predict_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    params: Mapping[str, Any],
    seed: int,
) -> np.ndarray:
    if not is_gbdt_available("lightgbm"):
        raise MissingGbdtDependencyError(get_installation_guidance("lightgbm"))
    import lightgbm as lgb

    X_tr, X_va, cat_cols = _prepare_matrices_for_lightgbm(X_train, X_val)
    model = lgb.LGBMClassifier(
        learning_rate=params.get("learning_rate", 0.03),
        n_estimators=params.get("n_estimators", 1000),
        num_leaves=params.get("num_leaves", 31),
        max_depth=params.get("max_depth", -1),
        subsample=params.get("subsample", 0.8),
        subsample_freq=params.get("subsample_freq", 1),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 1.0),
        min_child_samples=params.get("min_child_samples", 20),
        random_state=seed,
        verbosity=-1,
        n_jobs=params.get("n_jobs", 1),
        objective="binary",
    )
    model.fit(X_tr, y_train, categorical_feature=cat_cols if cat_cols else "auto")
    return _positive_proba(model, model.predict_proba(X_va))


def _encode_for_extratrees(
    X_train: pd.DataFrame, X_val: pd.DataFrame, categorical_cols: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ordinal-encode categoricals (unseen/missing -> -1), median-impute numerics.

    All statistics come from the (resampled) training fold only.
    """
    X_tr, X_va = X_train.copy(), X_val.copy()
    cat_set = set(categorical_cols)
    for col in categorical_cols:
        levels = pd.Index(np.asarray(X_tr[col].dropna().unique(), dtype=object))
        for frame in (X_tr, X_va):
            codes = levels.get_indexer(frame[col].astype(object))
            frame[col] = codes.astype("float64")
    num_cols = [c for c in X_tr.columns if c not in cat_set]
    # Fail closed: non-numeric undeclared columns would crash float64 cast with
    # a cryptic error; surface the offending names early.
    bad = [c for c in num_cols if not pd.api.types.is_numeric_dtype(X_tr[c])]
    if bad:
        raise ValueError(
            f"_encode_for_extratrees: columns {bad} are non-numeric but not "
            f"listed in categorical_cols={list(categorical_cols)!r}. Add them to "
            "categorical_cols or remove them from the predictor allowlist."
        )
    if num_cols:
        X_tr[num_cols] = X_tr[num_cols].astype("float64")
        X_va[num_cols] = X_va[num_cols].astype("float64")
        medians = X_tr[num_cols].median().fillna(0.0)
        X_tr[num_cols] = X_tr[num_cols].fillna(medians)
        X_va[num_cols] = X_va[num_cols].fillna(medians)
    return X_tr.astype("float64"), X_va.astype("float64")


def _fit_predict_extratrees(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    categorical_cols: Sequence[str],
    params: Mapping[str, Any],
    seed: int,
) -> np.ndarray:
    from sklearn.ensemble import ExtraTreesClassifier

    X_tr, X_va = _encode_for_extratrees(X_train, X_val, categorical_cols)
    model = ExtraTreesClassifier(
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=params.get("max_depth"),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 5),
        max_features=params.get("max_features", "sqrt"),
        bootstrap=bool(params.get("bootstrap", False)),
        n_jobs=params.get("n_jobs", 1),
        random_state=seed,
    )
    model.fit(X_tr, y_train)
    return _positive_proba(model, model.predict_proba(X_va))


def train_synthetic_sampling_partition(
    model_family: str,
    arm: str,
    train_df: pl.DataFrame | pd.DataFrame,
    val_df: pl.DataFrame | pd.DataFrame,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    target_col: str,
    params: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> np.ndarray:
    """Fold-local sample ``train_df``, fit the model, score the untouched ``val_df``.

    Parameters
    ----------
    model_family:
        ``lightgbm`` or ``extratrees``.
    arm:
        ``natural``, ``smote`` or ``mixup``.
    train_df, val_df:
        Outer-training and validation rows (Polars or Pandas). Only ``train_df``
        is ever resampled; ``val_df`` is scored as-is.
    feature_cols, categorical_cols:
        Predictor allowlist and its categorical subset.
    target_col:
        Binary target column.
    params:
        Model hyperparameters plus optional sampling keys ``target_ratio``,
        ``k_neighbors``, ``alpha``, ``seed`` and ``mock``.
    dry_run:
        Run the real fold-local sampling but score with deterministic mock
        probabilities instead of fitting a model.

    Returns
    -------
    np.ndarray:
        1D float64 positive-class probabilities, one per ``val_df`` row.
    """
    fam = normalize_family(model_family)
    arm_id = normalize_arm(arm)
    user_params = dict(params or {})
    feature_list = validate_predictor_cols(list(feature_cols))
    cat_list = list(categorical_cols)
    outside = [c for c in cat_list if c not in feature_list]
    if outside:
        raise ValueError(f"categorical_cols not in feature_cols: {outside}")

    X_train, y_train = _to_pandas_data(train_df, feature_list, target_col)
    X_val, _ = _to_pandas_data(val_df, feature_list, None)
    if y_train is None:
        raise ValueError(f"Target column '{target_col}' not found in training data")
    if len(X_val) == 0:
        return np.empty(0, dtype=np.float64)

    seed = int(
        user_params.get(
            "seed", user_params.get("random_state", user_params.get("random_seed", 42))
        )
    )
    X_res, y_res = apply_arm_sampling(
        arm_id,
        X_train,
        y_train,
        cat_list,
        target_ratio=float(user_params.get("target_ratio", DEFAULT_TARGET_RATIO)),
        k_neighbors=int(user_params.get("k_neighbors", DEFAULT_K_NEIGHBORS)),
        alpha=float(user_params.get("alpha", DEFAULT_MIXUP_ALPHA)),
        seed=seed,
    )

    if dry_run or user_params.get("mock", False):
        logger.info(
            "Dry-run partition: family=%s arm=%s train_rows=%d->%d "
            "(non_comparative=True, n03_eligible=False)",
            fam,
            arm_id,
            len(X_train),
            len(X_res),
        )
        return mock_predict_sampling_partition(
            y_res, len(X_val), seed, family=fam, arm=arm_id
        )

    if fam == "lightgbm":
        preds = _fit_predict_lightgbm(X_res, y_res, X_val, user_params, seed)
    else:
        preds = _fit_predict_extratrees(X_res, y_res, X_val, cat_list, user_params, seed)
    return np.clip(preds, 0.0, 1.0).astype(np.float64)
