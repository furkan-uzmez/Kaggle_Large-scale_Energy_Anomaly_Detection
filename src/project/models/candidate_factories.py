"""Matched Task 7 model/regime arms; construction is not execution approval.

These pipelines consume already-produced predictors. Supervised target-derived
features remain blocked pending the specific Task 10 assembler authorization.
FitCounter accounts model attempts, not wall time or RSS enforcement.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from project.experiments.candidate_contracts import (
    BlockedExperimentError,
    CandidateSpec,
    FitCounter,
)
from project.experiments.inner_cv import validate_inner_inputs
from project.features.candidates import (
    INTERACTIONS,
    MISSINGNESS,
    SAFE_NUMERIC,
    build_candidate_registry,
)
from project.features.distributions import build_distribution_registry
from project.features.encodings import BITMASK, FREQUENCY, build_encoding_registry
from project.features.input_encoding import (
    ABSOLUTE_CALENDAR_YEAR,
    require_year_built_encoding,
)
from project.features.registry import (
    FeatureRegistry,
    TargetDependence,
    build_lead_baseline_registry,
)
from project.features.value_recipes import RATIO_RECIPE, build_value_recipe_registry

NATIVE = "feat_missingness_passthrough_comparison"
WEIGHTS = "regime_natural_vs_class_weighted"
TREES = "model_extratrees_core_features"
MAX_CATEGORY_COLUMNS = 4096
TREE_DEFAULTS = {
    "n_estimators": 64, "max_depth": 12, "min_samples_leaf": 2,
    "max_features": "sqrt", "bootstrap": False, "class_weight": None,
    "n_jobs": 1, "random_state": 42,
}
TREE_BOUNDS = {"n_estimators": (1, 256), "max_depth": (1, 24), "min_samples_leaf": (1, 128)}


def _labels(y: Any) -> np.ndarray:
    values = y if isinstance(y, pd.Series) else pd.Series(y)
    if (not len(values) or not is_integer_dtype(values.dtype) or is_bool_dtype(values.dtype)
            or values.isna().any() or set(values.tolist()) != {0, 1}):
        raise ValueError("training labels must be nonnull nonboolean binary integers with both classes")
    return values.to_numpy(dtype=np.int64)


def balanced_train_weights(y: Any) -> np.ndarray:
    """Return n/(2*n_class) weights using only the supplied training labels."""
    labels = _labels(y)
    counts = np.bincount(labels, minlength=2)
    return len(labels) / (2.0 * counts[labels])


@lru_cache(maxsize=2)
def _authoritative_metadata(year_built_encoding: str = ABSOLUTE_CALENDAR_YEAR) -> Mapping[str, Any]:
    """Build the nine authoritative registries once; metadata is frozen and read-only."""
    registries = [build_lead_baseline_registry(), build_value_recipe_registry(RATIO_RECIPE)]
    registries.extend(build_distribution_registry(method) for method in ("bin", "quantile"))
    registries.extend(build_encoding_registry(recipe, year_built_encoding=year_built_encoding)
                      for recipe in (BITMASK, FREQUENCY))
    registries.extend(build_candidate_registry(recipe, year_built_encoding=year_built_encoding)
                      for recipe in (MISSINGNESS, SAFE_NUMERIC, INTERACTIONS))
    return MappingProxyType({name: registry.get(name) for registry in registries for name in registry.names})


class _PredictorSchema(TransformerMixin, BaseEstimator):
    """Validate exact ordered domains and normalize nullable storage on a copy."""

    def __init__(self, columns: tuple[str, ...], categorical: tuple[str, ...]):
        self.columns = columns
        self.categorical = categorical

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("predictors must be a pandas DataFrame")
        if tuple(X.columns) != self.columns:
            raise ValueError("predictor schema must match exact ordered admitted columns")
        if not X.index.is_unique or any(X.index.get_level_values(i).isna().any() for i in range(X.index.nlevels)):
            raise ValueError("predictor index must be unique and nonnull")
        result = X.copy(deep=True)
        for name in self.columns:
            column = X[name]
            if name in self.categorical:
                if not (is_object_dtype(column.dtype) or is_string_dtype(column.dtype)
                        or isinstance(column.dtype, pd.CategoricalDtype)):
                    raise TypeError(f"{name} must use string/category storage")
                if isinstance(column.dtype, pd.CategoricalDtype) and any(
                    not isinstance(v, str) for v in column.cat.categories
                ):
                    raise TypeError(f"{name} must declare string categories")
                if any(not isinstance(v, str) for v in column.dropna()):
                    raise TypeError(f"{name} must contain string categories or missing values")
                result[name] = pd.Series([np.nan if pd.isna(v) else v for v in column],
                                         index=X.index, dtype=object)
            else:
                if not is_numeric_dtype(column.dtype) or is_bool_dtype(column.dtype) or is_complex_dtype(column.dtype):
                    raise TypeError(f"{name} must have a real nonboolean numeric dtype")
                values = column.to_numpy(dtype=np.float64, na_value=np.nan)
                if np.isinf(values).any():
                    raise ValueError(f"{name} contains nonfinite raw values")
                result[name] = values
        return result

    def fit(self, X: pd.DataFrame, y: Any = None):
        frame = self._frame(X)
        if not len(frame):
            raise ValueError("fit requires nonempty rows")
        # Bound the entire dense categorical expansion, including all-null columns.
        if sum(max(1, frame[n].nunique(dropna=True)) for n in self.categorical) > MAX_CATEGORY_COLUMNS:
            raise ValueError("categorical cardinality exceeds dense expansion budget")
        self.feature_names_in_ = np.asarray(self.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_in_")
        return self._frame(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        if input_features is not None and tuple(input_features) != self.columns:
            raise ValueError("input_features must match schema")
        return self.feature_names_in_.copy()


class _CandidateArm(Pipeline):
    """Pipeline with an explicit comparison factor and training-only weight regime."""

    def __init__(self, steps, *, weight_mode="natural", comparison_factor="", memory=None,
                 verbose=False, frozen_params=None, native_numeric=()):
        self.weight_mode = weight_mode
        self.comparison_factor = comparison_factor
        self.frozen_params = frozen_params
        self.native_numeric = native_numeric
        super().__init__(steps, memory=memory, verbose=verbose)

    def _check_frozen_configuration(self) -> None:
        """Compare the live steps against the dict resolved at construction time."""
        if not isinstance(self.frozen_params, tuple) or not self.frozen_params:
            raise ValueError("model parameter configuration lacks its construction snapshot")
        current = self.get_params(deep=True)
        current['_core_structure'] = _core_structure(self)
        for name, value in self.frozen_params:
            if name not in current:
                raise ValueError("model parameter configuration changed outside Task 7 bounds")
            live = _canonical_parameter(name, current[name])
            if type(live) is not type(value) or not _same_value(live, value):
                raise ValueError("model parameter configuration changed outside Task 7 bounds")

    def preflight(self, X) -> None:
        """Check the declared regime, then the frozen configuration, then binning."""
        self._check_configuration()
        self.check_native_numeric_support(X)

    def _check_configuration(self) -> None:
        """Validate configuration without applying training-only support checks."""
        if self.weight_mode != _declared_weight_mode(self.comparison_factor, self.frozen_params):
            raise ValueError("weight regime disagrees with the declared comparison factor")
        self._check_frozen_configuration()

    @available_if(lambda self: hasattr(self._final_estimator, 'predict'))
    def predict(self, X, **params):
        self._check_configuration()
        return super().predict(X, **params)

    @available_if(lambda self: hasattr(self._final_estimator, 'predict_proba'))
    def predict_proba(self, X, **params):
        self._check_configuration()
        return super().predict_proba(X, **params)

    @available_if(lambda self: hasattr(self._final_estimator, 'predict_log_proba'))
    def predict_log_proba(self, X, **params):
        self._check_configuration()
        return super().predict_log_proba(X, **params)

    @available_if(lambda self: hasattr(self._final_estimator, 'decision_function'))
    def decision_function(self, X, **params):
        self._check_configuration()
        return super().decision_function(X, **params)

    @available_if(lambda self: hasattr(self._final_estimator, 'score'))
    def score(self, X, y=None, sample_weight=None, **params):
        self._check_configuration()
        return super().score(X, y, sample_weight=sample_weight, **params)

    def check_native_numeric_support(self, X) -> None:
        """Native passthrough cannot bin a numeric column with no finite training value.

        Installed sklearn HGB binning raises an opaque stride error for such a
        column, so this blocks explicitly and never imputes or reroutes.
        """
        if not self.native_numeric:
            return
        frame = self.named_steps["schema"]._frame(X)
        empty = tuple(n for n in self.native_numeric
                      if not np.isfinite(frame[n].to_numpy(dtype=np.float64, na_value=np.nan)).any())
        if empty:
            raise BlockedExperimentError(
                "native_all_null_numeric_column",
                "native passthrough arm cannot bin numeric columns without any finite training "
                f"value: {', '.join(empty)}; imputing or rerouting would destroy the comparison",
            )

    def fit(self, X, y=None, **params):
        labels = _labels(y)
        if not isinstance(y, pd.Series) or not y.index.equals(X.index):
            raise ValueError("target index must align exactly with predictors")
        if "model__sample_weight" in params:
            raise ValueError("external sample_weight cannot override the declared regime")
        if not isinstance(self.named_steps.get("model"),
                          (HistGradientBoostingClassifier, ExtraTreesClassifier)):
            raise TypeError("unsupported model configuration")
        self.preflight(X)
        if self.weight_mode == "balanced":
            params["model__sample_weight"] = balanced_train_weights(labels)
        return super().fit(X, y, **params)


def _same_value(live: Any, frozen: Any) -> bool:
    """Exact equality, treating a frozen NaN sentinel as equal to itself."""
    if type(frozen) is float and np.isnan(frozen):
        return bool(np.isnan(live))
    return bool(live == frozen)


def _declared_weight_mode(factor: Any, frozen: Any) -> str:
    """Resolve the one regime the declared comparison factor permits for this arm."""
    declared = dict(frozen).get("weight_mode") if isinstance(frozen, tuple) else None
    if declared not in ("natural", "balanced"):
        raise ValueError("unknown or unsnapshotted weight regime")
    if declared == "balanced" and factor != WEIGHTS:
        raise ValueError("only the weight comparison factor may declare a balanced regime")
    return declared


def _frozen_snapshot(arm: _CandidateArm) -> tuple[tuple[str, Any], ...]:
    """Freeze every resolved primitive schema/preprocess/model/regime parameter.

    A plain tuple keeps the snapshot immutable, hashable and picklable so that
    sklearn clone/deepcopy of a constructed arm keeps working.
    """
    snapshot = [
        (name, _canonical_parameter(name, value))
        for name, value in sorted(arm.get_params(deep=True).items())
        if name.split("__")[0] in ("schema", "preprocess", "model")
        and (value is None or type(value) in (str, bool, int, float, tuple)
             or isinstance(value, (type, np.dtype)))
    ]
    snapshot.append(("weight_mode", arm.weight_mode))
    snapshot.append(("comparison_factor", arm.comparison_factor))
    snapshot.append(("native_numeric", arm.native_numeric))
    snapshot.append(("_core_structure", _core_structure(arm)))
    return tuple(snapshot)


def _canonical_parameter(name, value):
    """Freeze type-valued parameters without identity or permissive object coercion."""
    if name.endswith('__dtype'):
        if not isinstance(value, (type, np.dtype)):
            raise ValueError('dtype parameter configuration requires a type or numpy dtype')
        try:
            dtype = np.dtype(value)
        except TypeError as exc:
            raise ValueError('invalid dtype parameter configuration') from exc
        # The admitted encoder uses an unstructured float64 dtype. Preserve the
        # full descriptor as well as byte order so structured types cannot alias it.
        return ('dtype', dtype.str, repr(dtype.descr))
    if isinstance(value, type):
        return ('type', value.__module__, value.__qualname__)
    if isinstance(value, np.dtype):
        return ('dtype', value.str, repr(value.descr))
    return value


def _structural_signature(step):
    """Immutable class/branch/selector graph, independent of estimator identity."""
    if isinstance(step, str) or step is None:
        return step
    kind = (type(step).__module__, type(step).__qualname__)
    if isinstance(step, ColumnTransformer):
        return (kind, tuple((name, _structural_signature(transformer), tuple(columns))
                            for name, transformer, columns in step.transformers))
    if isinstance(step, Pipeline):
        return (kind, tuple((name, _structural_signature(child)) for name, child in step.steps))
    return kind


def _core_structure(arm):
    # Producer steps may precede the core (including explicit group consumers).
    # The owned schema/preprocess/model suffix must retain its exact ordered graph.
    return tuple((name, _structural_signature(step)) for name, step in arm.steps[-3:])


def _pipeline(pool, categorical, native, model, factor, weight_mode="natural"):
    numeric = tuple(n for n in pool if n not in categorical)
    category = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transforms = []
    if numeric:
        transforms.append(("numeric", "passthrough" if native else SimpleImputer(
            strategy="median", keep_empty_features=True), numeric))
    if categorical:
        transforms.append(("categorical", category, categorical))
    preprocess = ColumnTransformer(transforms, remainder="drop", sparse_threshold=0.0)
    arm = _CandidateArm([
        ("schema", _PredictorSchema(pool, categorical)), ("preprocess", preprocess), ("model", model),
    ], comparison_factor=factor, weight_mode=weight_mode,
        native_numeric=numeric if native else ())
    arm.frozen_params = _frozen_snapshot(arm)
    _declared_weight_mode(factor, arm.frozen_params)
    return arm


def _model_parameters(spec: CandidateSpec) -> dict:
    params = dict(spec.model_params)
    if spec.experiment_id != TREES:
        expected = {"early_stopping": False, "random_state": 42}
        if any(k not in expected or type(v) is not type(expected[k]) or v != expected[k]
               for k, v in params.items()):
            raise ValueError("HGB configuration must use the resolved untuned baseline parameters")
        return expected
    if any(k not in TREE_DEFAULTS for k in params):
        raise ValueError("unknown ExtraTrees configuration")
    result = {**TREE_DEFAULTS, **params}
    for name, (low, high) in TREE_BOUNDS.items():
        value = result[name]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    for name in set(TREE_DEFAULTS) - set(TREE_BOUNDS):
        if type(result[name]) is not type(TREE_DEFAULTS[name]) or result[name] != TREE_DEFAULTS[name]:
            raise ValueError(f"{name} parameter must retain its fixed ExtraTrees value")
    return result


def build_candidate_arms(spec: CandidateSpec, registry: FeatureRegistry, *,
                         year_built_encoding: str = ABSOLUTE_CALENDAR_YEAR) -> dict[str, Pipeline]:
    """Return Task 7 comparisons under explicit metadata context; raw inputs stay raw."""
    require_year_built_encoding(year_built_encoding)
    if not isinstance(spec, CandidateSpec) or not isinstance(registry, FeatureRegistry):
        raise TypeError("typed CandidateSpec and FeatureRegistry are required")
    families = {NATIVE: "hist_gradient_boosting", WEIGHTS: "hist_gradient_boosting", TREES: "extratrees"}
    if spec.experiment_id not in families or spec.model_family != families[spec.experiment_id]:
        raise ValueError("unknown recipe or mismatched model family")
    if spec.method_params:
        raise ValueError("Task 7 admits no method parameters")
    authoritative = _authoritative_metadata(year_built_encoding)
    for name in spec.feature_pool:
        metadata = registry.get(name)
        if (metadata is None or metadata.target_dependence is not TargetDependence.INDEPENDENT
                or metadata != authoritative.get(name)):
            raise ValueError(f"feature lacks authoritative target-independent metadata: {name}")
        for source in metadata.source_columns:
            if registry.get(source) != authoritative.get(source):
                raise ValueError(f"feature source lacks authoritative metadata: {source}")
    params = _model_parameters(spec)
    categorical = tuple(n for n in spec.feature_pool if registry.get(n).family == "categorical")
    control = _pipeline(spec.feature_pool, categorical, spec.experiment_id == NATIVE,
                        HistGradientBoostingClassifier(early_stopping=False, random_state=42), spec.experiment_id)
    model = (ExtraTreesClassifier(**params) if spec.experiment_id == TREES
             else HistGradientBoostingClassifier(early_stopping=False, random_state=42))
    challenger = _pipeline(spec.feature_pool, categorical, False, model, spec.experiment_id,
                           "balanced" if spec.experiment_id == WEIGHTS else "natural")
    return {"control": control, "challenger": challenger}


def fit_candidate_arm(arm: Pipeline, X: pd.DataFrame, y: pd.Series, *,
                      groups: pd.Series | None = None, fit_counter: FitCounter) -> Pipeline:
    """Debit before the attempted fit; route groups only to explicit consumers."""
    from project.experiments.fit_accounting import FitAccounting

    if not isinstance(arm, Pipeline) or not isinstance(fit_counter, (FitCounter, FitAccounting)):
        raise TypeError("Pipeline and shared fit accounting are required")
    _labels(y)
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series) or not y.index.equals(X.index):
        raise ValueError("target index must align with predictor index")
    params = {}
    if groups is not None:
        validate_inner_inputs(y, groups)
        for name, step in arm.steps[:-1]:
            method = getattr(step, "fit_transform", step.fit)
            if "groups" in inspect.signature(method).parameters:
                params[f"{name}__groups"] = groups
    if isinstance(arm, _CandidateArm):
        arm.preflight(X)
    fit_counter.before_fit()
    return arm.fit(X, y, **params)
