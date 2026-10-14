"""Tabular MLP (PyTorch) for binary anomaly classification.

Provides fold-local preprocessing pipelines and an MLP with embedding for
``primary_use``, BatchNorm, dropout 0.2, AdamW, ReduceLROnPlateau,
BCEWithLogits, and mixed-precision CUDA support.

torch is imported lazily; ``MissingTorchDependencyError`` is raised only when
a real (non-dry-run) training call is made.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------

NUMERIC_FEATURES: tuple[str, ...] = (
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

CATEGORICAL_FEATURES: tuple[str, ...] = ("primary_use",)

ALLOWED_FEATURES: frozenset[str] = frozenset(NUMERIC_FEATURES) | frozenset(CATEGORICAL_FEATURES)

FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {
        "site_id",
        "building_id",
        "building_hour",
        "building_meter",
        "timestamp",
        "anomaly",
    }
) | frozenset(f for f in [] if f.startswith("gte_"))

# Pre-compute a broad forbidden prefix check
_FORBIDDEN_PREFIXES: tuple[str, ...] = ("gte_",)


class MissingTorchDependencyError(ImportError):
    """Raised when torch is required but not installed."""


def is_torch_available() -> bool:
    """Return True if torch is importable."""
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _require_torch() -> Any:
    """Import and return torch, raising MissingTorchDependencyError if absent."""
    if not is_torch_available():
        raise MissingTorchDependencyError(
            "PyTorch is not installed in the active environment. "
            "Install it via:\n"
            "  uv pip install torch\n"
            "or:\n"
            "  pip install torch\n"
            "See https://pytorch.org/get-started/locally/ for platform-specific instructions."
        )
    import torch

    return torch


def validate_feature_allowlist(feature_cols: Sequence[str]) -> None:
    """Raise ValueError if any feature column violates the allowlist.

    Parameters
    ----------
    feature_cols:
        Columns to validate.

    Raises
    ------
    ValueError
        If a forbidden or unrecognized column is passed.
    """
    violations: list[str] = []
    for col in feature_cols:
        is_forbidden_prefix = any(col.startswith(pfx) for pfx in _FORBIDDEN_PREFIXES)
        is_explicit_forbidden = col in FORBIDDEN_FEATURES
        if is_forbidden_prefix or is_explicit_forbidden or col not in ALLOWED_FEATURES:
            violations.append(col)
    if violations:
        raise ValueError(
            f"Feature columns violate the MLP allowlist: {violations}. "
            f"Allowed numeric: {list(NUMERIC_FEATURES)}; "
            f"Allowed categorical: {list(CATEGORICAL_FEATURES)}."
        )


# ---------------------------------------------------------------------------
# Fold-local preprocessing
# ---------------------------------------------------------------------------

_UNKNOWN_CATEGORY = "__unknown__"


class FoldLocalPreprocessor:
    """Fold-local imputer + scaler + category vocabulary.

    Fit is performed exclusively on outer-train rows.  Validation rows are
    transform-only.  Unseen ``primary_use`` values map to an unknown index.

    Parameters
    ----------
    arm:
        ``'control'`` — median imputation + StandardScaler.
        ``'challenger'`` — control + per-column missing indicators +
        QuantileTransformer(output_distribution='normal') on numerics.
    numeric_cols:
        Numeric column names in the same order as they appear in the data.
    categorical_cols:
        Categorical column names (currently only ``primary_use`` is used).
    """

    def __init__(
        self,
        arm: str,
        numeric_cols: Sequence[str],
        categorical_cols: Sequence[str],
    ) -> None:
        if arm not in ("control", "challenger"):
            raise ValueError(f"arm must be 'control' or 'challenger', got '{arm}'")
        self.arm = arm
        self.numeric_cols: list[str] = list(numeric_cols)
        self.categorical_cols: list[str] = list(categorical_cols)
        # Fitted state (populated by .fit())
        self._medians: dict[str, float] = {}
        self._scaler: Any = None  # sklearn scaler
        self._category_vocab: dict[str, dict[str, int]] = {}
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> FoldLocalPreprocessor:
        """Fit imputer, scaler, and category vocabulary on outer-train rows."""
        from sklearn.preprocessing import (
            QuantileTransformer,
            StandardScaler,
        )

        # 1. Compute medians on outer-train (ignoring NaNs)
        for col in self.numeric_cols:
            if col in X.columns:
                series = X[col].dropna()
                self._medians[col] = float(series.median()) if len(series) > 0 else 0.0
            else:
                self._medians[col] = 0.0

        # 2. Impute then scale
        X_num = self._impute_numerics(X)

        if self.arm == "control":
            scaler = StandardScaler()
            scaler.fit(X_num)
            self._scaler = scaler
        else:  # challenger
            scaler = QuantileTransformer(output_distribution="normal", random_state=42)
            scaler.fit(X_num)
            self._scaler = scaler

        # 3. Build category vocabulary from outer-train
        for col in self.categorical_cols:
            if col in X.columns:
                unique_vals = sorted(X[col].dropna().astype(str).unique().tolist())
                # index 0 = unknown, 1..N = known categories
                vocab = {_UNKNOWN_CATEGORY: 0}
                for i, v in enumerate(unique_vals, start=1):
                    vocab[v] = i
                self._category_vocab[col] = vocab
            else:
                self._category_vocab[col] = {_UNKNOWN_CATEGORY: 0}

        self._fitted = True
        return self

    def _impute_numerics(self, X: pd.DataFrame) -> np.ndarray:
        """Return imputed numeric matrix (NaN filled with per-column medians).

        For the ``meter_reading`` column a sign-safe log1p transform is applied
        **after** imputation so that both :meth:`fit` and :meth:`transform` see
        identical values, allowing the scaler to be fitted correctly.
        """
        parts: list[np.ndarray] = []
        for col in self.numeric_cols:
            if col in X.columns:
                col_vals = X[col].to_numpy(dtype=float, na_value=np.nan)
            else:
                col_vals = np.full(len(X), np.nan)
            median = self._medians.get(col, 0.0)
            col_vals = np.where(np.isnan(col_vals), median, col_vals)
            if col == "meter_reading":
                # sign-safe log1p so fit and transform see the same distribution
                col_vals = np.sign(col_vals) * np.log1p(np.abs(col_vals))
            parts.append(col_vals.reshape(-1, 1))
        if not parts:
            return np.empty((len(X), 0), dtype=np.float32)
        return np.hstack(parts).astype(np.float32)

    def _missing_indicators(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary missing-indicator matrix for numeric columns."""
        parts: list[np.ndarray] = []
        for col in self.numeric_cols:
            if col in X.columns:
                indicator = X[col].isna().to_numpy(dtype=np.float32).reshape(-1, 1)
            else:
                indicator = np.ones((len(X), 1), dtype=np.float32)
            parts.append(indicator)
        if not parts:
            return np.empty((len(X), 0), dtype=np.float32)
        return np.hstack(parts)

    def transform(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Transform rows into (scaled_numeric_matrix, {cat_col: int_indices}).

        Parameters
        ----------
        X:
            Input DataFrame (outer-train or validation).

        Returns
        -------
        X_num : np.ndarray, shape (n, num_features) for control or
            (n, 2*num_features) for challenger (numerics + indicators).
        cat_indices : dict mapping each categorical column to an int array.
        """
        if not self._fitted:
            raise RuntimeError("FoldLocalPreprocessor must be fit before transform.")

        X_num_raw = self._impute_numerics(X)
        X_num_scaled = self._scaler.transform(X_num_raw).astype(np.float32)
        # log1p is applied inside _impute_numerics so fit and transform are consistent;
        # no post-hoc rescale needed.

        if self.arm == "challenger":
            indicators = self._missing_indicators(X)
            X_num_scaled = np.hstack([X_num_scaled, indicators])

        cat_indices: dict[str, np.ndarray] = {}
        for col in self.categorical_cols:
            vocab = self._category_vocab.get(col, {_UNKNOWN_CATEGORY: 0})
            if col in X.columns:
                raw_vals = X[col].fillna(_UNKNOWN_CATEGORY).astype(str).tolist()
            else:
                raw_vals = [_UNKNOWN_CATEGORY] * len(X)
            indices = np.array(
                [vocab.get(v, vocab[_UNKNOWN_CATEGORY]) for v in raw_vals],
                dtype=np.int64,
            )
            cat_indices[col] = indices

        return X_num_scaled, cat_indices

    @property
    def num_categories(self) -> dict[str, int]:
        """Return {col: vocab_size} for each categorical column."""
        return {col: len(vocab) for col, vocab in self._category_vocab.items()}


# ---------------------------------------------------------------------------
# Model definition (lazy torch import)
# ---------------------------------------------------------------------------


def _build_mlp(
    num_numeric_features: int,
    category_vocab_sizes: dict[str, int],
    embedding_dim: int = 8,
    hidden_dims: tuple[int, ...] = (256, 128, 64),
    dropout: float = 0.2,
) -> Any:
    """Build and return the MLP model (requires torch).

    Parameters
    ----------
    num_numeric_features:
        Width of the numeric input vector.
    category_vocab_sizes:
        {col: vocab_size} for each categorical column.
    embedding_dim:
        Embedding dimension for each categorical column.
    hidden_dims:
        Widths of the three hidden layers.
    dropout:
        Dropout probability applied after each hidden layer.
    """
    torch = _require_torch()
    nn = torch.nn

    class TabularMLP(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.ModuleDict(
                {
                    col: nn.Embedding(vocab_size, embedding_dim)
                    for col, vocab_size in category_vocab_sizes.items()
                }
            )
            total_cat_dim = len(category_vocab_sizes) * embedding_dim
            input_dim = num_numeric_features + total_cat_dim

            layers: list[nn.Module] = []
            in_dim = input_dim
            for out_dim in hidden_dims:
                layers += [
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ]
                in_dim = out_dim
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x_num: Any, cat_tensors: dict[str, Any]) -> Any:
            parts = [x_num]
            for col, emb in self.embeddings.items():
                parts.append(emb(cat_tensors[col]))
            x = torch.cat(parts, dim=1)
            return self.net(x).squeeze(1)

    return TabularMLP()


# ---------------------------------------------------------------------------
# Training entrypoint
# ---------------------------------------------------------------------------

_INNER_VAL_BUILDING_FRACTION = 0.2  # fraction of buildings held out for inner val
_MIN_INNER_VAL_BUILDINGS = 1


def _split_inner_val(
    train_df: pd.DataFrame,
    building_col: str = "building_id",
    fraction: float = _INNER_VAL_BUILDING_FRACTION,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve an inner validation split from outer-train by grouping on building_id.

    The inner validation fold is never used in the outer validation fold; it
    is carved purely from the outer-train portion.
    """
    buildings = sorted(train_df[building_col].unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(buildings)
    n_inner_val = max(_MIN_INNER_VAL_BUILDINGS, int(np.ceil(len(buildings) * fraction)))
    inner_val_buildings = set(buildings[:n_inner_val])
    inner_train_mask = ~train_df[building_col].isin(inner_val_buildings)
    return train_df[inner_train_mask].copy(), train_df[~inner_train_mask].copy()


def _mock_predict_partition(y_train: np.ndarray, val_len: int, seed: int) -> np.ndarray:
    """Deterministic mock probabilities for dry-run / mock mode."""
    prior = float(np.mean(y_train)) if len(y_train) > 0 else 0.5
    prior = max(1e-4, min(1.0 - 1e-4, prior))
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-0.005, 0.005, size=val_len)
    return np.clip(prior + noise, 0.0, 1.0).astype(np.float64)


def train_mlp_partition(
    arm: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    target_col: str = "anomaly",
    building_col: str = "building_id",
    seed: int = 42,
    params: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> np.ndarray:
    """Train one MLP partition and return out-of-fold probabilities for val_df.

    Parameters
    ----------
    arm:
        ``'control'`` or ``'challenger'``.
    train_df:
        Outer-train partition (pandas DataFrame).
    val_df:
        Outer-validation partition (pandas DataFrame).
    numeric_cols:
        Numeric feature columns (must pass allowlist guard).
    categorical_cols:
        Categorical feature columns (must pass allowlist guard).
    target_col:
        Binary target column name.
    building_col:
        Column used to group buildings for inner-val split and early stopping.
    seed:
        Random seed for reproducibility.
    params:
        Optional hyperparameter overrides. Keys: ``epochs``, ``batch_size``,
        ``lr``, ``patience``, ``embedding_dim``, ``hidden_dims``, ``dropout``.
    dry_run:
        If True, return deterministic mock probabilities without requiring torch.

    Returns
    -------
    np.ndarray
        1-D float64 array of length ``len(val_df)`` with positive-class
        probabilities (clipped to [0, 1]).
    """
    validate_feature_allowlist(list(numeric_cols) + list(categorical_cols))

    p = dict(params or {})
    val_len = len(val_df)

    y_train_np = (
        train_df[target_col].to_numpy(dtype=np.float64)
        if target_col in train_df.columns
        else np.zeros(len(train_df), dtype=np.float64)
    )

    if dry_run or p.get("mock", False):
        logger.info(
            "MLP partition dry-run mode arm='%s' (non_comparative=True, n03_eligible=False, "
            "execution_mode='synthetic_smoke')",
            arm,
        )
        return _mock_predict_partition(y_train_np, val_len, seed)

    # --- real training path requires torch ---
    torch = _require_torch()

    epochs: int = int(p.get("epochs", 30))
    batch_size: int = int(p.get("batch_size", 1024))
    lr: float = float(p.get("lr", 1e-3))
    patience: int = int(p.get("patience", 5))
    embedding_dim: int = int(p.get("embedding_dim", 8))
    hidden_dims: tuple[int, ...] = tuple(p.get("hidden_dims", (256, 128, 64)))
    dropout: float = float(p.get("dropout", 0.2))

    # Reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np_rng = np.random.default_rng(seed)  # noqa: F841

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("MLP arm='%s' using device=%s", arm, device)

    # Preprocessing (fold-local)
    preprocessor = FoldLocalPreprocessor(arm, numeric_cols, categorical_cols)
    preprocessor.fit(train_df)

    # Inner val split (from outer-train only; outer val is predict-only)
    inner_train_df, inner_val_df = _split_inner_val(
        train_df, building_col=building_col, seed=seed
    )

    X_inner_train_num, X_inner_train_cat = preprocessor.transform(inner_train_df)
    y_inner_train = inner_train_df[target_col].to_numpy(dtype=np.float32)

    X_inner_val_num, X_inner_val_cat = preprocessor.transform(inner_val_df)
    y_inner_val = inner_val_df[target_col].to_numpy(dtype=np.float32)

    X_val_num, X_val_cat = preprocessor.transform(val_df)

    # Build model
    model = _build_mlp(
        num_numeric_features=X_inner_train_num.shape[1],
        category_vocab_sizes=preprocessor.num_categories,
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    weight_decay: float = float(p.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    use_amp = torch.cuda.is_available()
    if use_amp:
        scaler_amp = torch.amp.GradScaler("cuda")

    # Tensors for inner-train
    t_X_it_num = torch.from_numpy(X_inner_train_num).to(device)
    t_y_it = torch.from_numpy(y_inner_train).to(device)
    t_X_it_cat = {
        col: torch.from_numpy(arr).to(device) for col, arr in X_inner_train_cat.items()
    }

    # Tensors for inner-val (used only for early stopping)
    t_X_iv_num = torch.from_numpy(X_inner_val_num).to(device)
    t_y_iv = torch.from_numpy(y_inner_val).to(device)
    t_X_iv_cat = {
        col: torch.from_numpy(arr).to(device) for col, arr in X_inner_val_cat.items()
    }

    n_inner_train = len(y_inner_train)
    best_val_loss = float("inf")
    patience_counter = 0
    best_state: dict[str, Any] | None = None

    for epoch in range(epochs):
        model.train()
        # Mini-batch loop (streaming from CPU-resident tensors)
        perm = torch.randperm(n_inner_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_inner_train, batch_size):
            idx = perm[start : start + batch_size]
            # BatchNorm1d requires ≥2 rows in train mode; skip remainder batch.
            if idx.numel() < 2:
                continue
            xb_num = t_X_it_num[idx]
            xb_cat = {col: t_X_it_cat[col][idx] for col in t_X_it_cat}
            yb = t_y_it[idx]

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(xb_num, xb_cat)
                    loss = criterion(logits, yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                logits = model(xb_num, xb_cat)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        # Inner-val loss for early stopping (no outer val rows involved)
        model.eval()
        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast("cuda"):
                    iv_logits = model(t_X_iv_num, t_X_iv_cat)
                    iv_loss = float(criterion(iv_logits, t_y_iv).item())
            else:
                iv_logits = model(t_X_iv_num, t_X_iv_cat)
                iv_loss = float(criterion(iv_logits, t_y_iv).item())

        scheduler.step(iv_loss)
        logger.debug("epoch=%d train_loss=%.4f iv_loss=%.4f", epoch + 1, epoch_loss / max(n_batches, 1), iv_loss)

        if iv_loss < best_val_loss - 1e-6:
            best_val_loss = iv_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict outer validation fold (streaming to avoid OOM)
    model.eval()
    t_X_val_num = torch.from_numpy(X_val_num)
    t_X_val_cat = {col: torch.from_numpy(arr) for col, arr in X_val_cat.items()}

    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, val_len, batch_size):
            xb_num = t_X_val_num[start : start + batch_size].to(device)
            xb_cat = {col: t_X_val_cat[col][start : start + batch_size].to(device) for col in t_X_val_cat}
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(xb_num, xb_cat)
            else:
                logits = model(xb_num, xb_cat)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
            all_probs.append(probs)

    result = np.concatenate(all_probs) if all_probs else np.empty(0, dtype=np.float64)
    return np.clip(result, 0.0, 1.0)
