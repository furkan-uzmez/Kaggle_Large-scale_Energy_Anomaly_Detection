"""Level-3 Meta-Ensemble architecture combining Probability and Representation Meta-Learners.

Architecture:
- Arm A (Meta-2A): Probability-based Meta-Learner (using existing OOF predictions from challenger_lr).
- Arm B (Meta-2B): Representation-based Tree Leaf Embeddings
  (LightGBM leaf extractor -> OneHotEncoder -> sparse SGDClassifier with log-loss).
- Level-3 Meta-Learner: Nested direct ROC-AUC bounded combination of Meta-2A and Meta-2B scores:
  S_L3 = w * S_2A + (1 - w) * S_2B, where w in [0.0, 1.0].

Leakage Invariants:
- Meta-2B tree models and sparse OneHotEncoder fit strictly on outer training folds.
- Validation partitions are transform-only with no retraining or refitting of encoders.
- Output OOF artifact strictly conforms to the canonical 24-column CANDIDATE_OOF_SCHEMA.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from project.models.candidate_artifacts import CANDIDATE_OOF_SCHEMA
from project.submission import build_submission, validate_submission_contract

logger = logging.getLogger(__name__)


def _json_sha256(obj: Any) -> str:
    """Compute deterministic hex SHA-256 digest of a JSON-serializable object."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Arm B (Meta-2B): Tree Leaf Embedding Estimator
# ---------------------------------------------------------------------------


class Meta2BLeafEmbeddingEstimator:
    """Representation-based meta-learner using LightGBM leaf embeddings and sparse linear classifier.

    Extracts high-dimensional non-linear interactions by mapping raw tabular features
    into decision tree leaf activations, encoding them via a sparse one-hot representation,
    and training a linear classifier (SGDClassifier with log-loss).
    """

    def __init__(
        self,
        lgb_params: dict[str, Any] | None = None,
        encoder_params: dict[str, Any] | None = None,
        linear_params: dict[str, Any] | None = None,
        categorical_cols: Sequence[str] | None = None,
        dry_run: bool = False,
    ) -> None:
        self.lgb_params = dict(lgb_params or {})
        self.encoder_params = dict(encoder_params or {})
        self.linear_params = dict(linear_params or {})
        self.categorical_cols = list(categorical_cols or ["primary_use"])
        self.dry_run = bool(dry_run)

        # Fitted estimators
        self.lgb_model_: Any = None
        self.encoder_: OneHotEncoder | None = None
        self.linear_model_: SGDClassifier | None = None
        self.single_class_label_: int | None = None
        self.is_fitted_: bool = False

    def _prepare_features(self, X: np.ndarray | pd.DataFrame | pl.DataFrame) -> pd.DataFrame | np.ndarray:
        """Standardize feature input into a pandas DataFrame or numpy array."""
        if isinstance(X, pl.DataFrame):
            df = X.to_pandas()
            for col in self.categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype("category")
            return df
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            for col in self.categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype("category")
            return df
        return np.asarray(X)

    def extract_leaves(self, X: np.ndarray | pd.DataFrame | pl.DataFrame) -> np.ndarray:
        """Extract tree leaf node indices from the fitted LightGBM model."""
        if self.dry_run:
            n_samples = len(X) if not hasattr(X, "shape") else X.shape[0]
            rng = np.random.default_rng(42)
            return rng.integers(0, 8, size=(n_samples, 5), dtype=np.int32)

        if self.lgb_model_ is None:
            raise RuntimeError("Estimator is not fitted yet.")

        X_prep = self._prepare_features(X)
        leaves = self.lgb_model_.predict_proba(X_prep, pred_leaf=True)
        return np.asarray(leaves, dtype=np.int32)

    def transform_leaves_to_sparse(self, leaves: np.ndarray) -> sparse.csr_matrix:
        """Transform dense leaf indices into a scipy.sparse CSR binary indicator matrix."""
        if self.encoder_ is None:
            raise RuntimeError("OneHotEncoder is not fitted yet.")
        sparse_mat = self.encoder_.transform(leaves)
        if not sparse.isspmatrix_csr(sparse_mat):
            sparse_mat = sparse_mat.tocsr()
        return sparse_mat

    def fit(
        self,
        X: np.ndarray | pd.DataFrame | pl.DataFrame,
        y: np.ndarray | Sequence[int],
    ) -> Meta2BLeafEmbeddingEstimator:
        """Fit LightGBM leaf extractor, OneHotEncoder, and sparse SGDClassifier on training data."""
        y_arr = np.asarray(y, dtype=np.int64)

        if self.dry_run:
            self.is_fitted_ = True
            return self

        unique_classes = np.unique(y_arr)
        if len(unique_classes) < 2:
            self.single_class_label_ = int(unique_classes[0]) if len(unique_classes) == 1 else 0
            self.is_fitted_ = True
            return self

        import lightgbm as lgb

        X_prep = self._prepare_features(X)

        # 1. Fit LightGBM tree leaf extractor
        default_lgb = {
            "n_estimators": 50,
            "max_depth": 5,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        }
        lgb_cfg = {**default_lgb, **self.lgb_params}
        self.lgb_model_ = lgb.LGBMClassifier(**lgb_cfg)
        self.lgb_model_.fit(X_prep, y_arr)

        # 2. Extract training leaves
        leaves_tr = self.extract_leaves(X)

        # 3. Fit OneHotEncoder strictly on outer training leaves
        default_encoder = {"handle_unknown": "ignore", "sparse_output": True}
        enc_cfg = {**default_encoder, **self.encoder_params}
        self.encoder_ = OneHotEncoder(**enc_cfg)
        sparse_tr = self.encoder_.fit_transform(leaves_tr)
        if not sparse.isspmatrix_csr(sparse_tr):
            sparse_tr = sparse_tr.tocsr()

        # 4. Fit sparse linear classifier (log-loss SGDClassifier)
        default_linear = {
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": 1e-4,
            "max_iter": 1000,
            "tol": 1e-3,
            "random_state": 42,
        }
        lin_cfg = {**default_linear, **self.linear_params}
        self.linear_model_ = SGDClassifier(**lin_cfg)
        self.linear_model_.fit(sparse_tr, y_arr)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame | pl.DataFrame) -> np.ndarray:
        """Predict positive anomaly probabilities using fitted leaf embeddings."""
        if not self.is_fitted_:
            raise RuntimeError("Estimator is not fitted yet.")

        n_samples = len(X) if not hasattr(X, "shape") else X.shape[0]

        if self.dry_run:
            rng = np.random.default_rng(42)
            noise = rng.uniform(0.01, 0.05, size=n_samples)
            return np.clip(0.021 + noise, 0.0, 1.0).astype(np.float64)

        if self.single_class_label_ is not None:
            val = 1.0 if self.single_class_label_ == 1 else 0.0
            return np.full(n_samples, val, dtype=np.float64)

        leaves = self.extract_leaves(X)
        sparse_mat = self.transform_leaves_to_sparse(leaves)

        probs = self.linear_model_.predict_proba(sparse_mat)[:, 1]
        return np.clip(probs, 0.0, 1.0).astype(np.float64)


# ---------------------------------------------------------------------------
# Level-3 Meta-Learner (ROC-AUC Bounded Direct Combiner)
# ---------------------------------------------------------------------------


class Level3AucBoundedCombiner:
    """Level-3 Meta-Learner optimizing bounded combination weight w in [0.0, 1.0] for ROC-AUC.

    Blends Arm A (Meta-2A: Probability Meta-Learner) and Arm B (Meta-2B: Representation Leaf Embeddings):
        S_L3 = w * S_2A + (1.0 - w) * S_2B
    """

    def __init__(
        self,
        grid_resolution: int = 101,
        subsample_size: int | None = 100_000,
        random_state: int = 42,
    ) -> None:
        self.grid_resolution = max(2, int(grid_resolution))
        self.subsample_size = subsample_size
        self.random_state = int(random_state)
        self.weight_: float = 0.5
        self.best_auc_: float = 0.5
        self.is_fitted_: bool = False

    def fit(
        self,
        s_2a: np.ndarray | Sequence[float],
        s_2b: np.ndarray | Sequence[float],
        y: np.ndarray | Sequence[int],
    ) -> Level3AucBoundedCombiner:
        """Find optimal weight w in [0.0, 1.0] maximizing ROC-AUC on outer training fold."""
        s_2a_arr = np.asarray(s_2a, dtype=np.float64)
        s_2b_arr = np.asarray(s_2b, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)

        n_samples = len(y_arr)
        if len(s_2a_arr) != n_samples or len(s_2b_arr) != n_samples:
            raise ValueError(
                f"Length mismatch: s_2a ({len(s_2a_arr)}), s_2b ({len(s_2b_arr)}), y ({n_samples})"
            )

        if len(np.unique(y_arr)) < 2:
            self.weight_ = 0.5
            self.best_auc_ = 0.5
            self.is_fitted_ = True
            return self

        # Optional balanced subsampling for fast global grid search on massive data
        if self.subsample_size is not None and n_samples > self.subsample_size:
            rng = np.random.default_rng(self.random_state)
            pos_indices = np.where(y_arr == 1)[0]
            neg_indices = np.where(y_arr == 0)[0]
            half = self.subsample_size // 2
            n_pos = min(len(pos_indices), half)
            n_neg = min(len(neg_indices), self.subsample_size - n_pos)
            chosen_pos = rng.choice(pos_indices, size=n_pos, replace=False)
            chosen_neg = rng.choice(neg_indices, size=n_neg, replace=False)
            sub_idx = np.concatenate([chosen_pos, chosen_neg])
            rng.shuffle(sub_idx)

            eval_2a = s_2a_arr[sub_idx]
            eval_2b = s_2b_arr[sub_idx]
            eval_y = y_arr[sub_idx]
        else:
            eval_2a = s_2a_arr
            eval_2b = s_2b_arr
            eval_y = y_arr

        weights = np.linspace(0.0, 1.0, self.grid_resolution)
        best_w = 0.5
        best_auc = -1.0

        for w in weights:
            pred = w * eval_2a + (1.0 - w) * eval_2b
            try:
                score = float(roc_auc_score(eval_y, pred))
            except (ValueError, TypeError):
                score = 0.5
            if score > best_auc:
                best_auc = score
                best_w = float(w)

        self.weight_ = best_w
        self.best_auc_ = best_auc
        self.is_fitted_ = True
        return self

    def predict_proba(
        self,
        s_2a: np.ndarray | Sequence[float],
        s_2b: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        """Combine Meta-2A and Meta-2B probabilities using learned optimal weight."""
        if not self.is_fitted_:
            raise RuntimeError("Combiner is not fitted yet.")
        s_2a_arr = np.asarray(s_2a, dtype=np.float64)
        s_2b_arr = np.asarray(s_2b, dtype=np.float64)
        combined = self.weight_ * s_2a_arr + (1.0 - self.weight_) * s_2b_arr
        return np.clip(combined, 0.0, 1.0).astype(np.float64)


# ---------------------------------------------------------------------------
# Out-of-Fold Nested Cross-Validation Orchestration
# ---------------------------------------------------------------------------


def run_level3_nested_cv(
    train_dev: pl.DataFrame,
    dev_splits: pl.DataFrame,
    meta_2a_oof_df: pl.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "anomaly",
    categorical_cols: Sequence[str] | None = None,
    repeats: Sequence[int] = (0, 1, 2),
    folds: Sequence[int] = (0, 1, 2, 3),
    meta_2b_params: dict[str, Any] | None = None,
    level3_params: dict[str, Any] | None = None,
    execution_mode: str = "full",
    recipe_id: str = "recipe_level3_meta",
    model_config_id: str = "cfg_level3_auc_bounded",
    experiment_id: str = "level3_ensemble",
    run_id: str = "level3_run",
    config_sha256: str = "0" * 64,
    dry_run: bool = False,
    smoke: bool = False,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Execute leakage-free nested cross-validation for Level-3 Meta-Ensemble.

    Guarantees:
    - Meta-2B tree estimator and sparse encoder fit ONLY on rows where repeat == r and fold != k.
    - Meta-2B transform on validation fold (repeat == r, fold == k) is strictly transform-only.
    - Level-3 combiner fits on outer training fold scores and evaluates strictly on outer validation fold.
    - Output DataFrame conforms strictly to canonical 24-column CANDIDATE_OOF_SCHEMA.
    """
    non_comparative = bool(smoke or dry_run)
    n03_eligible = not non_comparative

    fit_config_payload = {
        "architecture": "level3_meta_ensemble",
        "meta_2a": "challenger_lr_oof",
        "meta_2b_params": meta_2b_params or {},
        "level3_params": level3_params or {},
        "features": list(feature_cols),
    }
    fit_config_sha = _json_sha256(fit_config_payload)

    # Resolve split and data digests
    if "split_sha256" in meta_2a_oof_df.columns:
        split_sha = str(meta_2a_oof_df["split_sha256"][0])
    else:
        split_sha = "0" * 64

    if "data_sha256" in meta_2a_oof_df.columns:
        data_sha = str(meta_2a_oof_df["data_sha256"][0])
    else:
        data_sha = "1" * 64

    m2b_cfg = dict(meta_2b_params or {})
    l3_cfg = dict(level3_params or {})

    oof_dfs: list[pl.DataFrame] = []
    fold_weights: dict[str, float] = {}
    fold_train_aucs: dict[str, float] = {}

    for rep in repeats:
        split_seed = 42 + rep
        meta_2a_rep = meta_2a_oof_df.filter(pl.col("repeat") == rep)
        rep_splits = dev_splits.filter(pl.col("repeat") == rep)

        for fld in folds:
            fold_key = f"r{rep}_f{fld}"
            logger.info("Executing Level-3 partition: repeat=%d, fold=%d...", rep, fld)

            train_b_ids = rep_splits.filter(pl.col("fold") != fld)["building_id"].to_list()
            val_b_ids = rep_splits.filter(pl.col("fold") == fld)["building_id"].to_list()

            if smoke or dry_run:
                train_b_ids = train_b_ids[:4]
                val_b_ids = val_b_ids[:2]

            train_fold_df = train_dev.filter(pl.col("building_id").is_in(train_b_ids)).sort("row_index")
            val_fold_df = train_dev.filter(pl.col("building_id").is_in(val_b_ids)).sort("row_index")

            if val_fold_df.height == 0:
                logger.warning("Empty validation partition for repeat %d, fold %d; skipping.", rep, fld)
                continue

            # Extract features and targets
            X_train = train_fold_df.select(list(feature_cols)).to_pandas()
            y_train = train_fold_df[target_col].to_numpy()

            X_val = val_fold_df.select(list(feature_cols)).to_pandas()

            # Align Meta-2A predictions strictly by row_index
            val_row_indices = val_fold_df["row_index"].to_list()
            train_row_indices = train_fold_df["row_index"].to_list()

            val_2a_df = meta_2a_rep.filter(pl.col("row_index").is_in(val_row_indices)).sort("row_index")
            train_2a_df = meta_2a_rep.filter(pl.col("row_index").is_in(train_row_indices)).sort("row_index")

            # Finding 2: Fail-closed on Meta-2A mismatch in full mode
            if not (smoke or dry_run):
                if val_2a_df.height != len(val_row_indices):
                    raise ValueError(
                        f"Meta-2A OOF validation row count mismatch for repeat {rep}, fold {fld}: "
                        f"expected {len(val_row_indices)} rows, but found {val_2a_df.height} rows."
                    )
                if not np.array_equal(val_2a_df["row_index"].to_numpy(), np.asarray(val_row_indices)):
                    raise ValueError(
                        f"Meta-2A OOF validation row_index alignment mismatch for repeat {rep}, fold {fld}."
                    )
                s_2a_val = val_2a_df["prediction"].to_numpy()

                if train_2a_df.height != len(train_row_indices):
                    raise ValueError(
                        f"Meta-2A OOF training row count mismatch for repeat {rep}, fold {fld}: "
                        f"expected {len(train_row_indices)} rows, but found {train_2a_df.height} rows."
                    )
                if not np.array_equal(train_2a_df["row_index"].to_numpy(), np.asarray(train_row_indices)):
                    raise ValueError(
                        f"Meta-2A OOF training row_index alignment mismatch for repeat {rep}, fold {fld}."
                    )
                s_2a_train = train_2a_df["prediction"].to_numpy()
            else:
                if val_2a_df.height == len(val_row_indices) and np.array_equal(
                    val_2a_df["row_index"].to_numpy(), np.asarray(val_row_indices)
                ):
                    s_2a_val = val_2a_df["prediction"].to_numpy()
                else:
                    rng_fallback = np.random.default_rng(split_seed + fld)
                    s_2a_val = rng_fallback.uniform(0.01, 0.99, size=len(val_row_indices))

                if train_2a_df.height == len(train_row_indices) and np.array_equal(
                    train_2a_df["row_index"].to_numpy(), np.asarray(train_row_indices)
                ):
                    s_2a_train = train_2a_df["prediction"].to_numpy()
                else:
                    rng_fallback = np.random.default_rng(split_seed + fld + 100)
                    s_2a_train = rng_fallback.uniform(0.01, 0.99, size=len(train_row_indices))

            # Finding 1: Compute inner-OOF predictions for Arm B on outer training fold
            # Use building-grouped split so both arms are genuinely out-of-fold when fitting combiner!
            unique_train_b = sorted(train_fold_df["building_id"].unique().to_list())
            n_b = len(unique_train_b)

            inner_lgb_p = dict(m2b_cfg.get("lgb_params") or {})
            if "n_estimators" not in inner_lgb_p:
                inner_lgb_p["n_estimators"] = 30
            else:
                inner_lgb_p["n_estimators"] = min(inner_lgb_p["n_estimators"], 20 if smoke else 30)

            if dry_run:
                estimator_2b_inner = Meta2BLeafEmbeddingEstimator(dry_run=True)
                estimator_2b_inner.fit(X_train, y_train)
                s_2b_inner_oof = estimator_2b_inner.predict_proba(X_train)
            elif n_b >= 2:
                n_inner_splits = 2 if smoke else min(3, n_b)
                kf = KFold(n_splits=n_inner_splits, shuffle=True, random_state=split_seed + fld)
                b_arr = np.array(unique_train_b)
                s_2b_inner_oof = np.zeros(len(train_fold_df), dtype=np.float64)

                for inner_train_idx, inner_val_idx in kf.split(b_arr):
                    inner_train_b = set(b_arr[inner_train_idx])
                    inner_val_b = set(b_arr[inner_val_idx])

                    train_mask = train_fold_df["building_id"].is_in(list(inner_train_b)).to_numpy()
                    val_mask = train_fold_df["building_id"].is_in(list(inner_val_b)).to_numpy()

                    X_in_train = X_train.iloc[train_mask]
                    y_in_train = y_train[train_mask]
                    X_in_val = X_train.iloc[val_mask]

                    inner_estimator = Meta2BLeafEmbeddingEstimator(
                        lgb_params=inner_lgb_p,
                        encoder_params=m2b_cfg.get("encoder_params"),
                        linear_params=m2b_cfg.get("linear_params"),
                        categorical_cols=categorical_cols,
                        dry_run=dry_run,
                    )
                    inner_estimator.fit(X_in_train, y_in_train)
                    s_2b_inner_oof[val_mask] = inner_estimator.predict_proba(X_in_val)
            else:
                inner_estimator = Meta2BLeafEmbeddingEstimator(
                    lgb_params=inner_lgb_p,
                    encoder_params=m2b_cfg.get("encoder_params"),
                    linear_params=m2b_cfg.get("linear_params"),
                    categorical_cols=categorical_cols,
                    dry_run=dry_run,
                )
                inner_estimator.fit(X_train, y_train)
                s_2b_inner_oof = inner_estimator.predict_proba(X_train)

            # Fit Level-3 Combiner on honest outer OOF (Arm A) and inner OOF (Arm B)
            combiner = Level3AucBoundedCombiner(
                grid_resolution=l3_cfg.get("grid_resolution", 101),
                subsample_size=l3_cfg.get("subsample_size", 100_000),
                random_state=split_seed,
            )
            combiner.fit(s_2a_train, s_2b_inner_oof, y_train)
            fold_weights[fold_key] = float(combiner.weight_)
            fold_train_aucs[fold_key] = float(combiner.best_auc_)

            # Fit Meta-2B on full outer training fold to evaluate on outer validation fold
            estimator_2b = Meta2BLeafEmbeddingEstimator(
                lgb_params=m2b_cfg.get("lgb_params"),
                encoder_params=m2b_cfg.get("encoder_params"),
                linear_params=m2b_cfg.get("linear_params"),
                categorical_cols=categorical_cols,
                dry_run=dry_run,
            )
            estimator_2b.fit(X_train, y_train)

            # Predict Meta-2B on validation fold (TRANSFORM ONLY)
            s_2b_val = estimator_2b.predict_proba(X_val)

            # Predict Level-3 on outer validation fold
            s_l3_val = combiner.predict_proba(s_2a_val, s_2b_val)

            # 4. Form canonical OOF slice
            fold_oof = val_fold_df.select([
                pl.col("row_index").cast(pl.Int64),
                pl.col("building_id").cast(pl.String),
                pl.col("timestamp"),
                pl.col(target_col).cast(pl.Int64).alias("target"),
            ]).with_columns([
                pl.lit("level3_ensemble").alias("candidate_id"),
                pl.lit(recipe_id).alias("recipe_id"),
                pl.lit("ensemble_level3").alias("model_family"),
                pl.lit(model_config_id).alias("model_config_id"),
                pl.lit(int(rep), dtype=pl.Int64).alias("repeat"),
                pl.lit(int(fld), dtype=pl.Int64).alias("fold"),
                pl.Series("prediction", s_l3_val, dtype=pl.Float64),
                pl.lit(split_sha).alias("split_sha256"),
                pl.lit(data_sha).alias("data_sha256"),
                pl.lit(config_sha256).alias("config_sha256"),
                pl.lit(run_id).alias("run_id"),
                pl.lit(int(split_seed), dtype=pl.Int64).alias("split_seed"),
                pl.lit(experiment_id).alias("experiment_id"),
                pl.lit("level3_meta").alias("arm_id"),
                pl.lit("positive_probability").alias("score_kind"),
                pl.lit(1, dtype=pl.Int64).alias("positive_label"),
                pl.lit(execution_mode).alias("execution_mode"),
                pl.lit(non_comparative).alias("non_comparative"),
                pl.lit(n03_eligible).alias("n03_eligible"),
                pl.lit(fit_config_sha).alias("model_fit_config_sha256"),
            ])

            oof_dfs.append(fold_oof)

    if not oof_dfs:
        raise RuntimeError("No validation folds were successfully evaluated.")

    combined_oof = (
        pl.concat(oof_dfs, how="vertical")
        .sort(["repeat", "fold", "building_id", "timestamp"])
        .select(CANDIDATE_OOF_SCHEMA.names)
    )

    metadata = {
        "fold_weights": fold_weights,
        "fold_train_aucs": fold_train_aucs,
        "mean_weight": float(np.mean(list(fold_weights.values()))),
        "fit_config_sha256": fit_config_sha,
    }

    return combined_oof, metadata


# ---------------------------------------------------------------------------
# Submission Pipeline Integration
# ---------------------------------------------------------------------------


def build_level3_submission(
    row_ids: Sequence[int] | pl.Series | np.ndarray,
    s_2a: Sequence[float] | pl.Series | np.ndarray,
    s_2b: Sequence[float] | pl.Series | np.ndarray,
    weight: float,
) -> pl.DataFrame:
    """Combine test predictions from Arm A and Arm B and build a validated Kaggle submission DataFrame."""
    s_2a_arr = np.asarray(s_2a, dtype=np.float64)
    s_2b_arr = np.asarray(s_2b, dtype=np.float64)
    row_ids_arr = np.asarray(row_ids, dtype=np.int64)

    if len(s_2a_arr) != len(row_ids_arr) or len(s_2b_arr) != len(row_ids_arr):
        raise ValueError(
            f"Array length mismatch: row_ids={len(row_ids_arr)}, s_2a={len(s_2a_arr)}, s_2b={len(s_2b_arr)}"
        )

    w = float(np.clip(weight, 0.0, 1.0))
    s_l3 = w * s_2a_arr + (1.0 - w) * s_2b_arr
    sub_df = build_submission(row_ids_arr, s_l3)
    validate_submission_contract(sub_df, expected_row_count=len(row_ids_arr))
    return sub_df
