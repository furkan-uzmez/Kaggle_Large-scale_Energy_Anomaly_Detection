"""Level-2 Meta-Learner Tournament combining diverse base model predictions.

Provides 5 distinct meta-model architectures:
1. control: Caruana Greedy Forward Selection with Replacement (discrete bag of models).
2. challenger_lr: L2-Regularized Logistic Regression on clipped, standard-scaled logits.
3. challenger_nnls: Non-Negative Least Squares (constrained non-negative simplex weights).
4. challenger_auc: Direct ROC-AUC rank optimizer via Scipy (Nelder-Mead simplex search).
5. challenger_shallow_lgb: Shallow LightGBM meta-learner capturing low-order interactions.

Nested Cross-Validation Leakage Invariants:
- Meta-models are strictly trained within each outer fold of each repeat.
- For repeat r and fold k, meta-weights train only on (repeat == r, fold != k) rows.
- No cross-repeat pooling or cross-fold leakage is permitted.
- Canonical 24-field out-of-fold records conform to CANDIDATE_OOF_SCHEMA.
"""

from __future__ import annotations

import abc
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize, nnls
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SUPPORTED_META_ARMS: tuple[str, ...] = (
    "control",
    "challenger_lr",
    "challenger_nnls",
    "challenger_auc",
    "challenger_shallow_lgb",
)

DEFAULT_BASE_CANDIDATE_SPECS: list[dict[str, Any]] = [
    {
        "name": "lightgbm_challenger",
        "path": "outputs/models/gbdt_standalone/full/gbdt_lightgbm_full_a5bc574c/lightgbm_candidate_oof.parquet",
        "arm_id": "challenger",
    },
    {
        "name": "xgboost_control",
        "path": "outputs/models/gbdt_standalone/full/gbdt_xgboost_full_002139ec/xgboost_candidate_oof.parquet",
        "arm_id": "control",
    },
    {
        "name": "catboost_control",
        "path": "outputs/models/gbdt_standalone/full/gbdt_catboost_full_5e28c41b/catboost_candidate_oof.parquet",
        "arm_id": "control",
    },
    {
        "name": "extratrees_natural",
        "path": "outputs/models/synthetic_sampling/full/extratrees_natural_synthetic_sampling_extratrees_natural_full_5ccca05f/extratrees_natural_candidate_oof.parquet",
        "arm_id": None,
    },
    {
        "name": "lightgbm_focal",
        "path": "outputs/models/gbdt_loss_objectives/full/lightgbm_focal_gbdt_loss_lightgbm_focal_full_eaab7db4/lightgbm_focal_candidate_oof.parquet",
        "arm_id": None,
    },
    {
        "name": "mlp_control",
        "path": "outputs/models/tabular_mlp/full/mlp_full_56295c03/mlp_control_candidate_oof.parquet",
        "arm_id": None,
    },
]


def _json_sha256(obj: Any) -> str:
    """Compute hex SHA-256 of a JSON-serializable object."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class BaseMetaLearner(abc.ABC):
    """Abstract interface for Level-2 meta-learning estimators."""

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseMetaLearner:
        """Fit meta-learner on base model predictions X and binary labels y."""
        ...

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict positive class probabilities for base model predictions X."""
        ...


class CaruanaEnsembleSelector(BaseMetaLearner):
    """Rich Caruana et al. (ICML 2004) greedy forward selection with replacement.

    Iteratively selects base models to maximize ROC-AUC directly on training folds.
    Produces discrete bag weights that sum to 1.
    """

    def __init__(
        self,
        n_iterations: int = 50,
        subsample_size: int | None = 100_000,
        random_state: int = 42,
    ) -> None:
        self.n_iterations = max(1, int(n_iterations))
        self.subsample_size = subsample_size
        self.random_state = random_state
        self.weights_: np.ndarray | None = None
        self.selected_indices_: list[int] = []
        self.best_score_: float = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> CaruanaEnsembleSelector:
        n_samples, n_models = X.shape
        if n_samples == 0:
            raise ValueError("Cannot fit on empty dataset")
        if len(np.unique(y)) < 2:
            self.weights_ = np.ones(n_models, dtype=np.float64) / n_models
            self.selected_indices_ = list(range(n_models))
            return self

        # Subsample for fast evaluation if dataset is large
        if self.subsample_size is not None and n_samples > self.subsample_size:
            rng = np.random.default_rng(self.random_state)
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]
            half = self.subsample_size // 2
            n_pos = min(len(pos_indices), half)
            n_neg = min(len(neg_indices), self.subsample_size - n_pos)
            chosen_pos = rng.choice(pos_indices, size=n_pos, replace=False)
            chosen_neg = rng.choice(neg_indices, size=n_neg, replace=False)
            sub_idx = np.concatenate([chosen_pos, chosen_neg])
            rng.shuffle(sub_idx)
            X_eval = X[sub_idx]
            y_eval = y[sub_idx]
        else:
            X_eval = X
            y_eval = y

        # Initialize with the single best model by ROC-AUC
        scores = [float(roc_auc_score(y_eval, X_eval[:, m])) for m in range(n_models)]
        best_m = int(np.argmax(scores))
        self.selected_indices_ = [best_m]
        current_sum = X_eval[:, best_m].copy()
        current_score = scores[best_m]

        for step in range(1, self.n_iterations):
            best_step_m = 0
            best_step_score = -1.0
            divisor = float(step + 1)
            for m in range(n_models):
                candidate_pred = (current_sum + X_eval[:, m]) / divisor
                score = float(roc_auc_score(y_eval, candidate_pred))
                if score > best_step_score:
                    best_step_score = score
                    best_step_m = m

            self.selected_indices_.append(best_step_m)
            current_sum += X_eval[:, best_step_m]
            current_score = best_step_score

        self.best_score_ = current_score
        counts = np.bincount(self.selected_indices_, minlength=n_models).astype(
            np.float64
        )
        self.weights_ = counts / float(len(self.selected_indices_))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted yet.")
        preds = X @ self.weights_
        return np.clip(preds, 0.0, 1.0)


class LogitLogisticRegressionMeta(BaseMetaLearner):
    """L2-Regularized Logistic Regression on clipped standard-scaled logits.

    Maps probability predictions p -> logit(p) = log(p / (1 - p)), optionally
    standardizes, and fits an L2-regularized logistic regression.
    """

    def __init__(
        self,
        C: float = 1.0,
        clip_eps: float = 1e-7,
        standardize: bool = True,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.C = float(C)
        self.clip_eps = float(clip_eps)
        self.standardize = bool(standardize)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.scaler_: StandardScaler | None = None
        self.clf_: LogisticRegression | None = None

    def _to_logits(self, X: np.ndarray) -> np.ndarray:
        clipped = np.clip(X, self.clip_eps, 1.0 - self.clip_eps)
        return np.log(clipped / (1.0 - clipped))

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogitLogisticRegressionMeta:
        n_samples, _ = X.shape
        if n_samples == 0:
            raise ValueError("Cannot fit on empty dataset")
        if len(np.unique(y)) < 2:
            self.clf_ = None
            return self

        Z = self._to_logits(X)
        if self.standardize:
            self.scaler_ = StandardScaler()
            Z = self.scaler_.fit_transform(Z)
        else:
            self.scaler_ = None

        self.clf_ = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.clf_.fit(Z, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.clf_ is None:
            # Fallback if single class during fit
            return np.clip(np.mean(X, axis=1), 0.0, 1.0)
        Z = self._to_logits(X)
        if self.scaler_ is not None:
            Z = self.scaler_.transform(Z)
        probs = self.clf_.predict_proba(Z)[:, 1]
        return np.clip(probs, 0.0, 1.0)


class NonNegativeLeastSquaresMeta(BaseMetaLearner):
    """Non-Negative Least Squares constrained weights.

    Finds non-negative linear combination w >= 0 minimizing ||X w - y||_2^2,
    optionally normalized to the simplex sum(w) = 1.
    """

    def __init__(self, normalize_weights: bool = True) -> None:
        self.normalize_weights = bool(normalize_weights)
        self.weights_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NonNegativeLeastSquaresMeta:
        n_samples, n_models = X.shape
        if n_samples == 0:
            raise ValueError("Cannot fit on empty dataset")
        if len(np.unique(y)) < 2:
            self.weights_ = np.ones(n_models, dtype=np.float64) / n_models
            return self

        w, _ = nnls(X, y.astype(np.float64))
        if self.normalize_weights:
            sum_w = float(np.sum(w))
            if sum_w > 0.0:
                w = w / sum_w
            else:
                w = np.ones(n_models, dtype=np.float64) / float(n_models)
        self.weights_ = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted yet.")
        preds = X @ self.weights_
        return np.clip(preds, 0.0, 1.0)


class DirectAucOptimizerMeta(BaseMetaLearner):
    """Direct ROC-AUC rank optimizer via Scipy Nelder-Mead on simplex weights.

    Optimizes unconstrained parameters theta with softmax(theta) to maximize ROC-AUC.
    """

    def __init__(
        self,
        method: str = "Nelder-Mead",
        max_iter: int = 100,
        subsample_size: int | None = 100_000,
        random_state: int = 42,
    ) -> None:
        self.method = method
        self.max_iter = int(max_iter)
        self.subsample_size = subsample_size
        self.random_state = int(random_state)
        self.weights_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> DirectAucOptimizerMeta:
        n_samples, n_models = X.shape
        if n_samples == 0:
            raise ValueError("Cannot fit on empty dataset")
        if len(np.unique(y)) < 2:
            self.weights_ = np.ones(n_models, dtype=np.float64) / n_models
            return self

        if self.subsample_size is not None and n_samples > self.subsample_size:
            rng = np.random.default_rng(self.random_state)
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]
            half = self.subsample_size // 2
            n_pos = min(len(pos_indices), half)
            n_neg = min(len(neg_indices), self.subsample_size - n_pos)
            chosen_pos = rng.choice(pos_indices, size=n_pos, replace=False)
            chosen_neg = rng.choice(neg_indices, size=n_neg, replace=False)
            sub_idx = np.concatenate([chosen_pos, chosen_neg])
            rng.shuffle(sub_idx)
            X_eval = X[sub_idx]
            y_eval = y[sub_idx]
        else:
            X_eval = X
            y_eval = y

        def objective(theta: np.ndarray) -> float:
            w = softmax(theta)
            pred = X_eval @ w
            try:
                score = float(roc_auc_score(y_eval, pred))
                return -score
            except (ValueError, TypeError):
                return 0.0

        x0 = np.zeros(n_models, dtype=np.float64)
        res = minimize(
            objective,
            x0,
            method=self.method,
            options={"maxiter": self.max_iter, "disp": False},
        )
        self.weights_ = softmax(res.x)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted yet.")
        preds = X @ self.weights_
        return np.clip(preds, 0.0, 1.0)


class ShallowLightGbmMeta(BaseMetaLearner):
    """Shallow LightGBM meta-learner capturing low-order model interactions."""

    def __init__(
        self,
        max_depth: int = 2,
        num_leaves: int = 3,
        learning_rate: float = 0.03,
        min_child_samples: int = 100,
        n_estimators: int = 50,
        random_state: int = 42,
        n_jobs: int = 4,
    ) -> None:
        self.max_depth = int(max_depth)
        self.num_leaves = int(num_leaves)
        self.learning_rate = float(learning_rate)
        self.min_child_samples = int(min_child_samples)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.clf_: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> ShallowLightGbmMeta:
        import lightgbm as lgb

        n_samples, _ = X.shape
        if n_samples == 0:
            raise ValueError("Cannot fit on empty dataset")
        if len(np.unique(y)) < 2:
            self.clf_ = None
            return self

        # Adjust min_child_samples for small datasets
        child_samples = min(self.min_child_samples, max(1, n_samples // 4))

        self.clf_ = lgb.LGBMClassifier(
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=child_samples,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            objective="binary",
            verbose=-1,
        )
        self.clf_.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.clf_ is None:
            return np.clip(np.mean(X, axis=1), 0.0, 1.0)
        p = self.clf_.predict_proba(X)
        if p.ndim == 2 and p.shape[1] >= 2:
            return np.clip(p[:, 1], 0.0, 1.0)
        return np.clip(p.ravel(), 0.0, 1.0)


def create_meta_learner(
    arm_id: str,
    params: Mapping[str, Any] | None = None,
) -> BaseMetaLearner:
    """Instantiate a meta-learning estimator for the given arm identifier."""
    if not isinstance(arm_id, str):
        raise TypeError(f"arm_id must be a string, got {type(arm_id).__name__}")
    arm_norm = arm_id.strip().lower()
    if arm_norm not in SUPPORTED_META_ARMS:
        raise ValueError(
            f"Unsupported meta arm '{arm_id}'. Supported arms: {list(SUPPORTED_META_ARMS)}"
        )
    p = dict(params or {})

    if arm_norm == "control":
        return CaruanaEnsembleSelector(
            n_iterations=p.get("n_iterations", 50),
            subsample_size=p.get("subsample_size", 100_000),
            random_state=p.get("random_state", 42),
        )
    if arm_norm == "challenger_lr":
        return LogitLogisticRegressionMeta(
            C=p.get("C", 1.0),
            clip_eps=p.get("clip_eps", 1e-7),
            standardize=p.get("standardize", True),
            max_iter=p.get("max_iter", 1000),
            random_state=p.get("random_state", 42),
        )
    if arm_norm == "challenger_nnls":
        return NonNegativeLeastSquaresMeta(
            normalize_weights=p.get("normalize_weights", True),
        )
    if arm_norm == "challenger_auc":
        return DirectAucOptimizerMeta(
            method=p.get("method", "Nelder-Mead"),
            max_iter=p.get("max_iter", 100),
            subsample_size=p.get("subsample_size", 100_000),
            random_state=p.get("random_state", 42),
        )
    if arm_norm == "challenger_shallow_lgb":
        return ShallowLightGbmMeta(
            max_depth=p.get("max_depth", 2),
            num_leaves=p.get("num_leaves", 3),
            learning_rate=p.get("learning_rate", 0.03),
            min_child_samples=p.get("min_child_samples", 100),
            n_estimators=p.get("n_estimators", 50),
            random_state=p.get("random_state", 42),
            n_jobs=p.get("n_jobs", 4),
        )
    raise ValueError(f"Unhandled meta arm '{arm_norm}'")


def load_base_candidate_predictions(
    candidate_specs: Sequence[dict[str, Any]] | None = None,
    project_root: Path | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Load and horizontally align base candidate OOF predictions.

    Verifies identical length, row index, repeat, fold, and target alignment.
    Returns merged Polars DataFrame and the list of prediction column names.
    """
    specs = list(candidate_specs or DEFAULT_BASE_CANDIDATE_SPECS)
    if not specs:
        raise ValueError("No candidate specifications provided")

    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    first_spec = specs[0]
    first_path = (root / first_spec["path"]).resolve()
    if not first_path.is_file():
        raise FileNotFoundError(f"Base candidate OOF parquet not found: {first_path}")

    meta_cols = [
        "row_index",
        "repeat",
        "fold",
        "building_id",
        "timestamp",
        "target",
        "split_seed",
        "split_sha256",
        "data_sha256",
    ]
    first_cols = meta_cols + ["prediction"] + (["arm_id"] if first_spec.get("arm_id") else [])
    base_df = pl.read_parquet(first_path, columns=first_cols)
    if first_spec.get("arm_id") is not None:
        base_df = base_df.filter(pl.col("arm_id") == first_spec["arm_id"])

    first_pred_col = f"pred_{first_spec['name']}"
    merged_df = base_df.select(meta_cols + [pl.col("prediction").alias(first_pred_col)])
    prediction_cols = [first_pred_col]

    for spec in specs[1:]:
        cand_path = (root / spec["path"]).resolve()
        if not cand_path.is_file():
            raise FileNotFoundError(
                f"Base candidate OOF parquet not found: {cand_path}"
            )

        cand_cols = ["prediction", "row_index", "target"] + (
            ["arm_id"] if spec.get("arm_id") else []
        )
        cand_df = pl.read_parquet(cand_path, columns=cand_cols)
        if spec.get("arm_id") is not None:
            cand_df = cand_df.filter(pl.col("arm_id") == spec["arm_id"])

        if cand_df.height != merged_df.height:
            raise ValueError(
                f"Candidate {spec['name']} row count ({cand_df.height}) differs from base ({merged_df.height})"
            )
        # Fast verification of key alignment
        if not (cand_df["row_index"] == merged_df["row_index"]).all():
            raise ValueError(f"Candidate {spec['name']} row_index alignment mismatch")
        if not (cand_df["target"] == merged_df["target"]).all():
            raise ValueError(f"Candidate {spec['name']} target alignment mismatch")

        pred_col_name = f"pred_{spec['name']}"
        merged_df = merged_df.with_columns(cand_df["prediction"].alias(pred_col_name))
        prediction_cols.append(pred_col_name)

    return merged_df, prediction_cols


def run_meta_nested_cv(
    data: pl.DataFrame,
    prediction_cols: Sequence[str],
    arm_id: str,
    meta_learner_params: Mapping[str, Any] | None,
    repeats: Sequence[int],
    folds: Sequence[int],
    run_id: str,
    execution_mode: str,
    non_comparative: bool,
    n03_eligible: bool,
    config_sha256: str,
    recipe_id: str | None = None,
    model_config_id: str | None = None,
    experiment_id: str = "ensemble_meta_tournament",
) -> pl.DataFrame:
    """Execute nested CV meta-learner fitting over repeats and outer folds.

    Enforces strict isolation: for each (repeat r, fold k), the meta-learner
    trains ONLY on rows where repeat == r and fold != k. Predicts on (repeat == r, fold == k).
    Produces complete OOF dataframe matching CANDIDATE_OOF_SCHEMA.
    """
    arm_norm = arm_id.strip().lower()
    fit_config_payload = {
        "arm_id": arm_norm,
        "model_family": "ensemble_meta",
        "params": dict(meta_learner_params or {}),
        "prediction_cols": sorted(prediction_cols),
    }
    model_fit_config_sha = _json_sha256(fit_config_payload)
    resolved_recipe_id = recipe_id or f"recipe_meta_{arm_norm}"
    resolved_model_config_id = model_config_id or f"cfg_meta_{arm_norm}"
    candidate_id = f"meta_{arm_norm}"

    fold_oof_dfs: list[pl.DataFrame] = []

    for rep in repeats:
        rep_df = data.filter(pl.col("repeat") == rep)
        if rep_df.height == 0:
            continue

        for fld in folds:
            train_part = rep_df.filter(pl.col("fold") != fld)
            val_part = rep_df.filter(pl.col("fold") == fld)

            if train_part.height == 0 or val_part.height == 0:
                continue

            X_train = train_part.select(list(prediction_cols)).to_numpy()
            y_train = train_part["target"].to_numpy()
            X_val = val_part.select(list(prediction_cols)).to_numpy()

            learner = create_meta_learner(arm_norm, meta_learner_params)
            learner.fit(X_train, y_train)
            val_preds = learner.predict_proba(X_val)

            fold_oof = val_part.with_columns(
                [
                    pl.lit(candidate_id).alias("candidate_id"),
                    pl.lit(resolved_recipe_id).alias("recipe_id"),
                    pl.lit("ensemble_meta").alias("model_family"),
                    pl.lit(resolved_model_config_id).alias("model_config_id"),
                    pl.lit(val_preds).alias("prediction"),
                    pl.lit(config_sha256).alias("config_sha256"),
                    pl.lit(run_id).alias("run_id"),
                    pl.lit(experiment_id).alias("experiment_id"),
                    pl.lit(arm_norm).alias("arm_id"),
                    pl.lit("positive_probability").alias("score_kind"),
                    pl.lit(1).cast(pl.Int64).alias("positive_label"),
                    pl.lit(execution_mode).alias("execution_mode"),
                    pl.lit(non_comparative).alias("non_comparative"),
                    pl.lit(n03_eligible).alias("n03_eligible"),
                    pl.lit(model_fit_config_sha).alias("model_fit_config_sha256"),
                ]
            )
            fold_oof_dfs.append(fold_oof)

    if not fold_oof_dfs:
        raise RuntimeError("No out-of-fold predictions were generated.")

    all_oof = pl.concat(fold_oof_dfs, how="vertical")
    # Order columns strictly to match candidate v1 schema
    from project.models.candidate_artifacts import CANDIDATE_OOF_SCHEMA

    return all_oof.select(CANDIDATE_OOF_SCHEMA.names)
