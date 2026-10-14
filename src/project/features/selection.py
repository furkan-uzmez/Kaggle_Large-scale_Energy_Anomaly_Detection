"""Fold-local dimensionality recipes, not campaign execution or model selection.

Callers supply an ordered, typed predictor pool and fit a fresh whole Pipeline
on each training fold. Engineered inputs must themselves be created inside that
fold. SVC uses ranking scores only: probability metrics/promotion remain blocked
until separately approved fold-local calibration. No implicit calibrator or HPO.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from project.features.candidates import build_candidate_registry
from project.features.encodings import build_encoding_registry
from project.features.registry import (
    AvailabilityStatus,
    FeatureRegistry,
    ProvenanceStatus,
    TargetDependence,
    TransformFitBoundary,
    build_lead_baseline_registry,
)

PCA_RECIPE = "dim_pca_vs_passthrough"
SELECTOR_RECIPE = "dim_fold_local_selector_vs_passthrough"
BLOCK_RECIPE = "dim_semantic_block_drop"
SEMANTIC_BLOCKS = {
    "measurement": ("meter_reading",),
    "building_metadata": ("primary_use", "square_feet", "year_built", "floor_count"),
    "calendar": ("hour", "weekday", "month", "year"),
}


def _catalog() -> dict[str, Any]:
    """Authoritative producer definitions, never caller-provided approval flags."""
    registries = [build_lead_baseline_registry()]
    registries.extend(build_candidate_registry(recipe) for recipe in (
        "feat_missingness_indicators", "feat_safe_calendar_numeric_transforms", "feat_categorical_interactions"))
    registries.extend(build_encoding_registry(recipe) for recipe in (
        "feat_missingness_bitmask", "feat_frequency_encoding"))
    return {name: registry.get(name) for registry in registries for name in registry.names}


def _validate_pool(pool: tuple[str, ...], registry: FeatureRegistry) -> dict[str, tuple[str, ...]]:
    if not isinstance(registry, FeatureRegistry):
        raise TypeError("registry must be an authoritative typed FeatureRegistry")
    if (not isinstance(pool, tuple) or not pool or any(not isinstance(n, str) for n in pool)
            or len(set(pool)) != len(pool)):
        raise ValueError("feature_pool must be an explicit nonempty tuple of unique predictor names")
    catalog = _catalog()
    baseline = set(build_lead_baseline_registry().names)

    def roots(name: str, path: tuple[str, ...]) -> tuple[str, ...]:
        meta, approved = registry.get(name), catalog.get(name)
        if meta is None or approved is None:
            raise ValueError(f"unregistered/forbidden predictor or source lineage: {name}")
        if (meta.provenance_status is not ProvenanceStatus.VERIFIED
                or meta.prediction_time_availability is not AvailabilityStatus.AVAILABLE
                or meta.target_dependence is not TargetDependence.INDEPENDENT):
            raise ValueError(f"unsafe provenance/availability/target dependence: {name}")
        if meta.available_at is not None:
            raise ValueError(f"source availability time needs separate runner verification: {name}")
        if (meta.source_columns != approved.source_columns or meta.family != approved.family
                or meta.transform_fit_boundary is not approved.transform_fit_boundary):
            raise ValueError(f"predictor source lineage differs from authoritative producer: {name}")
        if name in baseline:
            return (name,)
        if name in path:
            raise ValueError(f"cyclic predictor source lineage: {name}")
        return tuple(dict.fromkeys(root for source in meta.source_columns for root in roots(source, path + (name,))))

    return {name: roots(name, ()) for name in pool}


def _retained_pool(pool: tuple[str, ...], registry: FeatureRegistry, drop_block: str | None) -> tuple[str, ...]:
    roots = _validate_pool(pool, registry)
    if drop_block is None:
        return pool
    if not isinstance(drop_block, str) or drop_block not in SEMANTIC_BLOCKS:
        raise ValueError("drop_block must name an explicit allowlisted semantic block")
    dropped_sources = set(SEMANTIC_BLOCKS[drop_block])
    kept = tuple(name for name in pool if not dropped_sources.intersection(roots[name]))
    if not kept or kept == pool:
        raise ValueError("semantic block drop is empty or the requested block is absent")
    return kept


def _binary_target(y: Any, frame: pd.DataFrame) -> None:
    if isinstance(y, pd.Series) and not y.index.equals(frame.index):
        raise ValueError("training target index must align exactly with predictor index")
    target = np.asarray(y)
    if (target.ndim != 1 or len(target) != len(frame) or target.dtype.kind not in "iu"
            or set(target.tolist()) != {0, 1}):
        raise ValueError("training target requires both integer binary classes 0 and 1")


def _category_feature_name(feature: str, category: Any) -> str:
    """Lossless pair names avoid underscore collisions across categorical inputs."""
    value = category.item() if isinstance(category, np.generic) else category
    return json.dumps([feature, value], ensure_ascii=False, separators=(",", ":"))


class _CheckedPCA(PCA):
    def fit(self, X: Any, y: Any = None) -> _CheckedPCA:
        array = np.asarray(X)
        count = self.n_components
        if not ((type(count) is int and count > 0) or (type(count) is float and 0 < count < 1)):
            raise ValueError("PCA components must be a positive integer or variance fraction in (0,1)")
        if len(array) < 2:
            raise ValueError("PCA requires at least two training rows")
        if type(count) is int and count > min(array.shape):
            raise ValueError("PCA components exceed training row/numeric-feature bounds")
        if not np.isfinite(array).all() or not np.any(np.var(array, axis=0) > 0):
            raise ValueError("PCA numeric block has no finite positive training variance")
        return super().fit(X, y)

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "components_")
        return np.asarray([f"pca{i}" for i in range(self.n_components_)], dtype=object)


class _NonemptySelector(SelectFromModel):
    def fit(self, X: pd.DataFrame, y: Any = None, **fit_params: Any) -> _NonemptySelector:
        _binary_target(y, X)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                super().fit(X, y, **fit_params)
        except ConvergenceWarning as error:
            raise ValueError("training-fold selector did not converge") from error
        if not self.get_support().any():
            raise ValueError("training-fold selector produced an empty feature mask; no fallback")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "estimator_")
        if not isinstance(X, pd.DataFrame) or tuple(X.columns) != tuple(self.feature_names_in_):
            raise ValueError("selector input schema must match fitted feature names")
        return X.loc[:, self.get_feature_names_out()].copy()


class _SparseFrameSVC(SVC):
    """Present CSR before SVC chooses its sparse gamma/fit path.

    sklearn 1.9 checks SciPy sparsity before validating pandas sparse frames.
    Preserve named preprocessing and inference schema checks while converting
    only at the model boundary: COO/CSR storage scales with nnz, not rows*columns.
    Accelerates large datasets via isolated cuML GPU worker when available, with
    graceful CPU fallback.
    """

    GPU_ROW_THRESHOLD: int = 5000

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _SparseFrameSVC:
        sparse_frame = isinstance(X, pd.DataFrame) and all(
            isinstance(dtype, pd.SparseDtype) for dtype in X.dtypes)
        matrix = X.sparse.to_coo().tocsr() if sparse_frame else X
        row_count = matrix.shape[0] if hasattr(matrix, "shape") else len(matrix)

        if row_count >= self.GPU_ROW_THRESHOLD:
            try:
                from project.features.gpu_svm_bridge import GpuSvmBridge, is_gpu_available
                if is_gpu_available():
                    gamma = self.gamma if isinstance(self.gamma, (int, float, str)) else "scale"
                    bridge = GpuSvmBridge()
                    bridge.fit(matrix, y, C=float(self.C), gamma=gamma, tol=float(self.tol))
                    self._gpu_bridge = bridge
                    self.classes_ = np.array([0, 1])
                    if sparse_frame:
                        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
                    return self
            except Exception:
                import traceback
                from pathlib import Path
                Path("/tmp/gpu_bridge_error.log").write_text(traceback.format_exc())
                self._gpu_bridge = None

        self._gpu_bridge = None
        fitted = super().fit(matrix, y, sample_weight=sample_weight)
        if sparse_frame:
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return fitted

    def decision_function(self, X: Any) -> np.ndarray:
        if getattr(self, "_gpu_bridge", None) is not None:
            sparse_frame = isinstance(X, pd.DataFrame) and all(
                isinstance(dtype, pd.SparseDtype) for dtype in X.dtypes)
            matrix = X.sparse.to_coo().tocsr() if sparse_frame else X
            return self._gpu_bridge.decision_function(matrix)
        return super().decision_function(X)

    def close(self) -> None:
        bridge = getattr(self, "_gpu_bridge", None)
        if bridge is not None:
            bridge.close()
            self._gpu_bridge = None

class FoldLocalPreprocessor(TransformerMixin, BaseEstimator):
    """Strict predictor guard and train-fitted preprocessing for one matched arm.

    Empty transform frames preserve schema; empty training frames fail. Category
    nulls are normalized before most-frequent imputation; unseen values yield
    all-zero OHE. Entirely missing numeric columns retain the imputer's fixed zero
    fill. Individual constants remain; PCA rejects a wholly degenerate block.
    """

    def __init__(self, feature_pool: tuple[str, ...], registry: FeatureRegistry,
                 recipe_id: str, use_pca: bool = False, drop_block: str | None = None,
                 pca_components: float = .95) -> None:
        self.feature_pool = feature_pool
        self.registry = registry
        self.recipe_id = recipe_id
        self.use_pca = use_pca
        self.drop_block = drop_block
        self.pca_components = pca_components

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("selection input must be a pandas DataFrame")
        if X.columns.has_duplicates or tuple(X.columns) != self.feature_pool:
            raise ValueError("selection schema must match the exact ordered feature_pool columns")
        frame = X.copy(deep=True)
        for name in self.feature_pool:
            if self.registry.get(name).family == "categorical":
                dtype = frame[name].dtype
                if isinstance(dtype, pd.CategoricalDtype):
                    if any(not isinstance(v, str) for v in frame[name].cat.categories):
                        raise TypeError(f"{name} must declare string categories")
                elif not (is_object_dtype(dtype) or is_string_dtype(dtype)):
                    raise TypeError(f"{name} must have a string/categorical dtype")
                if any(not isinstance(v, str) for v in frame[name].dropna()):
                    raise TypeError(f"{name} must contain strings or missing values")
                frame[name] = frame[name].astype(object).where(frame[name].notna(), np.nan)
            else:
                dtype = frame[name].dtype
                if not is_numeric_dtype(dtype) or is_bool_dtype(dtype) or is_complex_dtype(dtype):
                    raise TypeError(f"{name} must have a real numeric nonboolean dtype")
                values = frame[name].to_numpy(dtype=float, na_value=np.nan)
                if np.isinf(values).any():
                    raise ValueError(f"{name} contains nonfinite nonmissing values")
                frame[name] = values
        return frame

    def fit(self, X: pd.DataFrame, y: Any = None) -> FoldLocalPreprocessor:
        if (not isinstance(self.recipe_id, str)
                or self.recipe_id not in (PCA_RECIPE, SELECTOR_RECIPE, BLOCK_RECIPE)):
            raise ValueError("unsupported dimensionality/selection recipe")
        if type(self.use_pca) is not bool or (self.use_pca and self.recipe_id != PCA_RECIPE):
            raise ValueError("PCA arm applies only to the PCA recipe")
        if self.drop_block is not None and self.recipe_id != BLOCK_RECIPE:
            raise ValueError("drop_block applies only to the semantic block recipe")
        retained = _retained_pool(self.feature_pool, self.registry, self.drop_block)
        frame = self._frame(X)
        if len(frame) == 0:
            raise ValueError("fit requires at least one training row")
        if y is not None:
            _binary_target(y, frame)
        numeric = tuple(n for n in retained if self.registry.get(n).family != "categorical")
        categorical = tuple(n for n in retained if n not in numeric)
        hgb = self.recipe_id == BLOCK_RECIPE
        branches = []
        if numeric:
            if hgb:
                numeric_pipe = FeatureUnion([
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("indicator", MissingIndicator(features="all", error_on_new=False)),
                ])
            else:
                numeric_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                    ("pca", _CheckedPCA(n_components=self.pca_components, svd_solver="full", whiten=False,
                                         random_state=42) if self.use_pca else "passthrough"),
                ])
            branches.append(("numeric", numeric_pipe, list(numeric)))
        elif self.use_pca:
            raise ValueError("PCA requires at least one numeric predictor")
        if categorical:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=not hgb,
                                          feature_name_combiner=_category_feature_name)),
            ])
            branches.append(("categorical", cat_pipe, list(categorical)))
        transformer = ColumnTransformer(branches, remainder="drop", sparse_threshold=0. if hgb else 1.)
        transformer.fit(frame, y)
        self.column_transformer_ = transformer
        self.feature_names_in_ = np.asarray(self.feature_pool, dtype=object)
        self.n_features_in_ = len(self.feature_pool)
        self.retained_features_ = retained
        self.output_features_ = tuple(transformer.get_feature_names_out())
        self.registry_ = self._output_registry(numeric, categorical)
        self.fitted_policy_ = (self.recipe_id, self.use_pca, self.drop_block, self.pca_components, self.feature_pool)
        return self

    def _output_registry(self, numeric: tuple[str, ...], categorical: tuple[str, ...]) -> FeatureRegistry:
        transformer = self.column_transformer_
        source_sets = []
        if numeric:
            if self.use_pca:
                source_sets.extend([numeric] * transformer.named_transformers_["numeric"].named_steps["pca"].n_components_)
            else:
                source_sets.extend((n,) for n in numeric)
                if self.recipe_id == BLOCK_RECIPE:
                    source_sets.extend((n,) for n in numeric)
        if categorical:
            encoder = transformer.named_transformers_["categorical"].named_steps["one_hot"]
            for name, categories in zip(categorical, encoder.categories_, strict=True):
                source_sets.extend([(name,)] * len(categories))
        definitions = []
        for name, sources in zip(self.output_features_, source_sets, strict=True):
            definitions.append(replace(self.registry.get(sources[0]), name=name,
                family="fold_local_numeric", source_columns=sources,
                semantic_meaning=f"Training-fold fitted {self.recipe_id} output {name}; values from declared sources",
                transform_fit_boundary=TransformFitBoundary.OUTER_TRAIN,
                fallback_behavior="Use training-fitted statistics/vocabulary only; no inference refit"))
        return FeatureRegistry(definitions)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "output_features_")
        policy = (self.recipe_id, self.use_pca, self.drop_block, self.pca_components, self.feature_pool)
        if policy != self.fitted_policy_:
            raise ValueError("preprocessor parameters changed; fit a fresh fold pipeline")
        if input_features is not None and tuple(input_features) != tuple(self.feature_names_in_):
            raise ValueError("input_features must match ordered fit schema")
        return np.asarray(self.output_features_, dtype=object)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.get_feature_names_out()
        _validate_pool(self.feature_pool, self.registry)
        frame = self._frame(X)
        if len(frame) == 0:
            return pd.DataFrame(index=X.index, columns=self.output_features_, dtype=float)
        matrix = self.column_transformer_.transform(frame)
        if sparse.issparse(matrix):
            output = pd.DataFrame.sparse.from_spmatrix(matrix, index=X.index, columns=self.output_features_)
            # SciPy's implicit entries are zero; pandas' float default can be NaN.
            # Set only local fill metadata, retaining sparse indices/stored values.
            for name in output:
                output[name].array.fill_value = 0.0
            return output
        return pd.DataFrame(matrix, index=X.index, columns=self.output_features_)


def make_selection_arms(recipe_id: str, feature_pool: tuple[str, ...], *, registry: FeatureRegistry,
                        drop_block: str | None = None, pca_components: float = .95) -> dict[str, Pipeline]:
    """Create fresh unfitted control/candidate pipelines for one actual matrix ID.

    Pool order is explicit; producers outside baseline/candidates/encodings need
    separate registration before use here. The factory does not fit or read data.
    Nondefault PCA counts are explicit caller settings, not searched or clipped.
    """
    if not isinstance(recipe_id, str) or recipe_id not in (PCA_RECIPE, SELECTOR_RECIPE, BLOCK_RECIPE):
        raise ValueError("unsupported dimensionality/selection recipe")
    _validate_pool(feature_pool, registry)
    if recipe_id == BLOCK_RECIPE:
        if drop_block is None:
            raise ValueError("semantic recipe requires an explicit drop_block")
        _retained_pool(feature_pool, registry, drop_block)
    elif drop_block is not None:
        raise ValueError("drop_block applies only to the semantic block recipe")
    if recipe_id == PCA_RECIPE and not any(registry.get(n).family != "categorical" for n in feature_pool):
        raise ValueError("PCA recipe requires numeric predictors")
    candidate_key = {PCA_RECIPE: "pca", SELECTOR_RECIPE: "selector", BLOCK_RECIPE: f"drop_{drop_block}"}[recipe_id]
    result = {}
    for arm in ("passthrough", candidate_key):
        preprocessing = FoldLocalPreprocessor(feature_pool, registry, recipe_id,
            use_pca=arm == "pca", drop_block=drop_block if arm.startswith("drop_") else None,
            pca_components=pca_components)
        selector = (_NonemptySelector(LogisticRegression(C=1., l1_ratio=1., solver="liblinear",
                    class_weight=None, max_iter=1000, random_state=42), threshold=1e-5, prefit=False)
                    if arm == "selector" else "passthrough")
        if recipe_id == PCA_RECIPE:
            # sklearn 1.9's default disables probabilities without setting its deprecated parameter.
            model = _SparseFrameSVC(C=1., kernel="rbf", gamma="scale", random_state=42)
        elif recipe_id == SELECTOR_RECIPE:
            model = LogisticRegression(C=1., l1_ratio=0., solver="lbfgs", class_weight=None,
                                       max_iter=1000, random_state=42)
        else:
            model = HistGradientBoostingClassifier(early_stopping=False, random_state=42)
        result[arm] = Pipeline([("preprocessor", preprocessing), ("selector", selector), ("model", model)])
    return result


def get_selection_support(pipeline: Pipeline) -> pd.Series:
    """Return a copy of the fitted mask over preprocessed (including OHE) names."""
    preprocessor = pipeline.named_steps["preprocessor"]
    names = preprocessor.get_feature_names_out()
    selector = pipeline.named_steps["selector"]
    mask = np.ones(len(names), dtype=bool) if selector == "passthrough" else selector.get_support()
    return pd.Series(mask.copy(), index=names, name="selected")


def selection_audit(pipeline: Pipeline) -> dict[str, Any]:
    """Resolved parameters and fitted lineage; not metrics, a winner, or approval."""
    check_is_fitted(pipeline.named_steps["model"])
    pre = pipeline.named_steps["preprocessor"]
    support = get_selection_support(pipeline)
    selected = support.index[support].tolist()
    ranking_only = isinstance(pipeline.named_steps["model"], SVC)
    audit = {"recipe_id": pre.recipe_id, "fit_boundary": TransformFitBoundary.OUTER_TRAIN.value,
             "selected_names": selected, "support": support,
             "registry": FeatureRegistry(pre.registry_.get(n) for n in selected),
             "model_params": pipeline.named_steps["model"].get_params(deep=True),
             "preprocessor_params": pre.get_params(deep=False),
             "resolved_preprocessing_params": pre.column_transformer_.get_params(deep=True),
             "selector_params": (None if pipeline.named_steps["selector"] == "passthrough"
                                 else pipeline.named_steps["selector"].get_params(deep=True)),
             "probability_evaluation_status": "blocked_pending_fold_local_calibration" if ranking_only else "not_evaluated",
             "final_promotion_status": "blocked_pending_fold_local_calibration" if ranking_only else "not_authorized"}
    if pre.use_pca:
        pca = pre.column_transformer_.named_transformers_["numeric"].named_steps["pca"]
        audit.update(n_components=pca.n_components_, explained_variance_ratio=pca.explained_variance_ratio_.copy())
    return audit


def selection_stability(supports: Sequence[pd.Series]) -> dict[str, Any]:
    """Frequency/Jaccard of supplied fitted masks only; no data or label fitting.

    Coordinates are preprocessed names. Different fold OHE vocabularies are
    unioned; absent coordinates count as unselected, not an original-column mask.
    """
    if not supports:
        raise ValueError("support masks must be nonempty")
    for support in supports:
        if (not isinstance(support, pd.Series) or support.dtype != bool or not support.index.is_unique
                or not all(isinstance(n, str) for n in support.index) or not support.any()):
            raise ValueError("each support mask must have unique names and nonempty boolean selection")
    names = list(dict.fromkeys(n for support in supports for n in support.index))
    matrix = np.asarray([support.reindex(names, fill_value=False).to_numpy() for support in supports])
    sets = [set(np.flatnonzero(row)) for row in matrix]
    jaccard = np.asarray([[len(a & b) / len(a | b) for b in sets] for a in sets])
    return {"selection_frequency": pd.Series(matrix.mean(axis=0), index=names),
            "jaccard": pd.DataFrame(jaccard)}
