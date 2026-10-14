"""Honestly declared, group-disjoint cross-fitted supervised target encoding.

This is the only supervised feature producer in the candidate set, so its
leakage surface is declared rather than hidden: the appended
``primary_use_target_mean`` is registered ``TARGET_DERIVED`` with an
``OUTER_TRAIN`` fit boundary, and the generic availability guards in
``availability.py`` therefore keep refusing it as an independent predictor. That
refusal is intentional and is not loosened here; only an assembler that
explicitly authorizes this exact producer may route the column into a model.

For any fitting subset the prior is ``mean(y)`` over every row of that subset.
A nonmissing string category is estimated as
``(sum_y(category) + smoothing*prior) / (count(category) + smoothing)``.
Missing and unseen categories fall back to that same subset prior, never to a
learned missing-category target mean, so absence never carries its own signal.

``fit_transform(X, y, *, groups)`` is the only honest training path. It builds
deterministic group-disjoint inner folds with
``experiments.inner_cv.build_inner_folds`` and encodes each held-out row using
only the targets of that fold's own training buildings, so no row -- and no
other row of its building -- contributes to its own encoding. It additionally
stores the full supplied-training map for later ``transform``. ``fit`` stores
only that full map and records ``cross_fitted_ = False``; a ``fit().transform()``
path over training rows is leaky and must never substitute for cross-fitting.
Both paths validate labels and groups; only cross-fitting requires inner-fold support.

``transform`` takes predictors only. It never accepts labels or group metadata
and never refits, so held-out categories and labels cannot alter fitted state.
Group identities stay out of the feature space; an assembled Pipeline passes
``features__groups`` as an explicit fit parameter rather than relying on global
sklearn metadata routing.
"""

from __future__ import annotations

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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from project.experiments.candidate_contracts import BlockedExperimentError
from project.experiments.inner_cv import (
    MAX_SEED,
    MAX_SPLITS,
    MIN_SPLITS,
    build_inner_folds,
    inner_membership_digest,
    validate_inner_inputs,
)
from project.features.registry import (
    AvailabilityStatus,
    FeatureMetadata,
    FeatureRegistry,
    ProvenanceStatus,
    ResourceCost,
    TargetDependence,
    TransformFitBoundary,
    build_lead_baseline_registry,
)

SOURCE_COLUMN = "primary_use"
OUTPUT_NAME = "primary_use_target_mean"
SMOOTHING_BOUNDS = (0.0, 1000.0)
MAX_CATEGORIES_BOUNDS = (1, 4096)
N_SPLITS_BOUNDS = (MIN_SPLITS, MAX_SPLITS)
SEED_BOUNDS = (0, MAX_SEED)
_SEMANTICS = (
    "Supervised primary_use target mean (sum_y + smoothing*prior)/(count + smoothing) "
    "estimated inside group-disjoint inner folds of one outer training partition; "
    "missing and unseen categories take the fitting subset's prior."
)
_FALLBACK = (
    "Missing and unseen categories take the fitting subset's prior mean(y); no "
    "learned missing-category target mean is produced."
)


# --- Parameter validation ------------------------------------------------------


def _require_smoothing(value: object) -> float:
    low, high = SMOOTHING_BOUNDS
    if type(value) not in (int, float) or isinstance(value, bool):
        raise TypeError(
            f"smoothing must be a nonboolean real number in ({low}, {high}]"
        )
    number = float(value)
    if not np.isfinite(number) or number <= low or number > high:
        raise ValueError(f"smoothing must be a finite number in ({low}, {high}]")
    return number


def _require_bounded_integer(value: object, label: str, bounds: tuple[int, int]) -> int:
    low, high = bounds
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer in [{low}, {high}]; booleans are invalid")
    if not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


# --- Registry metadata ---------------------------------------------------------


def build_target_encoding_registry() -> FeatureRegistry:
    """Return verified baseline lineage plus the honestly target-derived feature.

    No caller-supplied flag can upgrade an unsafe baseline source: the raw
    ``primary_use`` producer must stay verified, available, target-independent
    and self-sourced. The appended feature is declared ``TARGET_DERIVED`` so the
    generic availability guards keep rejecting it as an independent predictor.
    """
    baseline = build_lead_baseline_registry()
    source = baseline.get(SOURCE_COLUMN)
    if (source is None
            or source.provenance_status is not ProvenanceStatus.VERIFIED
            or source.prediction_time_availability is not AvailabilityStatus.AVAILABLE
            or source.target_dependence is not TargetDependence.INDEPENDENT
            or source.source_columns != (SOURCE_COLUMN,)):
        raise ValueError(
            f"unsafe baseline source lineage/provenance/availability/target: {SOURCE_COLUMN}"
        )
    definitions = [baseline.get(name) for name in baseline.names]
    definitions.append(FeatureMetadata(
        name=OUTPUT_NAME, family="candidate_target_encoding",
        source_columns=(SOURCE_COLUMN,), semantic_meaning=_SEMANTICS,
        prediction_time_availability=AvailabilityStatus.AVAILABLE,
        target_dependence=TargetDependence.TARGET_DERIVED,
        transform_fit_boundary=TransformFitBoundary.OUTER_TRAIN,
        fallback_behavior=_FALLBACK,
        resource_cost=ResourceCost.LOW, provenance_status=ProvenanceStatus.VERIFIED,
        available_at=source.available_at,
    ))
    return FeatureRegistry(definitions)


# --- Frame, label and group validation ----------------------------------------


def _categories(frame: pd.DataFrame) -> np.ndarray:
    """Return object-dtype categories, rejecting non-string nonmissing values."""
    column = frame[SOURCE_COLUMN]
    if (isinstance(column.dtype, pd.CategoricalDtype)
            and any(not isinstance(value, str) for value in column.cat.categories)):
        raise TypeError(f"{SOURCE_COLUMN} must declare string categories or missing values")
    if not (is_object_dtype(column.dtype) or is_string_dtype(column.dtype)
            or isinstance(column.dtype, pd.CategoricalDtype)):
        raise TypeError(f"{SOURCE_COLUMN} must contain strings or missing values")
    values = np.asarray(
        [None if pd.isna(value) else value for value in column.tolist()], dtype=object
    )
    if any(value is not None and not isinstance(value, str) for value in values):
        raise TypeError(f"{SOURCE_COLUMN} must contain strings or missing values")
    return values


def _validate_frame(frame: object) -> np.ndarray:
    """Accept any ordered subset of approved raw predictors covering primary_use."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("target-encoding input must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError("target-encoding input columns must be unique")
    if not frame.index.is_unique or any(
        frame.index.get_level_values(level).isna().any() for level in range(frame.index.nlevels)
    ):
        raise ValueError("target-encoding index must be unique and nonnull")
    approved = set(build_lead_baseline_registry().names)
    if any(not isinstance(name, str) or name not in approved for name in frame.columns):
        raise ValueError(
            "target-encoding schema columns must be approved baseline predictors only"
        )
    if SOURCE_COLUMN not in frame.columns:
        raise ValueError(f"required target-encoding column missing: {SOURCE_COLUMN}")
    if is_numeric_dtype(frame[SOURCE_COLUMN].dtype) or is_complex_dtype(frame[SOURCE_COLUMN].dtype):
        raise TypeError(f"{SOURCE_COLUMN} must contain strings or missing values")
    for name in frame.columns:
        if name == SOURCE_COLUMN:
            continue
        column = frame[name]
        if (not is_numeric_dtype(column.dtype) or is_bool_dtype(column.dtype)
                or is_complex_dtype(column.dtype)):
            raise TypeError(f"{name} must use a numeric nonboolean noncomplex dtype")
        if not np.isfinite(column.dropna().to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{name} must contain finite numbers or null values")
    return _categories(frame)


def _validate_identities(frame: pd.DataFrame, y: object, groups: object) -> None:
    """Require labels and group metadata aligned to the already validated frame."""
    for label, series in (("target", y), ("group metadata", groups)):
        if not isinstance(series, pd.Series):
            raise TypeError(f"{label} must be a pandas Series aligned to the predictor index")
        if not series.index.equals(frame.index):
            raise ValueError(
                f"{label} index must align exactly with the predictor index"
            )


# --- Smoothed category estimates ----------------------------------------------


def _fit_subset_map(
    categories: np.ndarray, labels: np.ndarray, smoothing: float
) -> tuple[dict[str, float], float]:
    """Return one subset's smoothed category map and its own prior ``mean(y)``.

    The prior is the mean over *every* row of the subset, including rows whose
    category is missing, so a missing category never acquires its own learned
    target mean.
    """
    prior = float(labels.mean())
    known = np.asarray([value is not None for value in categories], dtype=bool)
    frame = pd.DataFrame(
        {"category": categories[known], "label": labels[known].astype(np.float64)}
    )
    aggregated = frame.groupby("category", sort=True)["label"].agg(["sum", "count"])
    mapping = {
        str(category): float(
            (row["sum"] + smoothing * prior) / (row["count"] + smoothing)
        )
        for category, row in aggregated.iterrows()
    }
    return mapping, prior


def _apply_map(
    categories: np.ndarray, mapping: dict[str, float], prior: float
) -> np.ndarray:
    return np.asarray(
        [prior if value is None else mapping.get(value, prior) for value in categories],
        dtype=np.float64,
    )


class GroupCrossFittedTargetEncoder(TransformerMixin, BaseEstimator):
    """Append one honest, group-disjoint cross-fitted supervised category feature.

    Use ``fit_transform(X, y, groups=...)`` for training rows: it returns inner
    out-of-fold encodings and stores the full supplied-training map separately.
    ``fit`` stores only that full map, for arms that score unseen rows. Both
    reject one-class partitions; only cross-fitting requires group support for
    inner folds. Failed refits invalidate previous fitted evidence.
    """

    def __init__(
        self,
        smoothing: float = 10.0,
        n_splits: int = 3,
        seed: int = 42,
        max_categories: int = 4096,
    ) -> None:
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.seed = seed
        self.max_categories = max_categories

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None, *, groups: pd.Series
    ) -> GroupCrossFittedTargetEncoder:
        """Store only the full supplied-training map; this path is not cross-fitted."""
        categories, labels = self._prepare(X, y, groups)
        mapping, prior = _fit_subset_map(categories, labels, self._smoothing)
        self._record(X, mapping, prior)
        self.cross_fitted_ = False
        return self

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None, *, groups: pd.Series
    ) -> pd.DataFrame:
        """Return inner out-of-fold encodings and store the full training map.

        Each held-out row is encoded from its fold's training buildings only, so
        neither the row's own label nor any other label from its building can
        reach its encoding.
        """
        categories, labels = self._prepare(X, y, groups)
        folds = build_inner_folds(y, groups, n_splits=self._n_splits, seed=self._seed)
        digest = inner_membership_digest(folds, groups)
        encoded = np.full(len(categories), np.nan, dtype=np.float64)
        excluded: list[tuple[str, ...]] = []
        for train, valid in folds:
            mapping, prior = _fit_subset_map(
                categories[train], labels[train], self._smoothing
            )
            encoded[valid] = _apply_map(categories[valid], mapping, prior)
            excluded.append(tuple(sorted({str(v) for v in groups.iloc[valid].tolist()})))
        if np.isnan(encoded).any():
            raise ValueError("inner folds left training rows unencoded")
        mapping, prior = _fit_subset_map(categories, labels, self._smoothing)
        self._record(X, mapping, prior)
        self.cross_fitted_ = True
        self.inner_excluded_groups_ = tuple(excluded)
        self.inner_membership_digest_ = digest
        return self._append(X, encoded)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode predictors with the stored full training map; never refit."""
        self._check_unchanged_parameters()
        categories = _validate_frame(X)
        if tuple(X.columns) != tuple(self.feature_names_in_):
            raise ValueError("target-encoding transform schema must match the ordered fit columns")
        return self._append(
            X, _apply_map(categories, self.full_training_map_, self.training_prior_)
        )

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        self._check_unchanged_parameters()
        if input_features is not None and tuple(input_features) != tuple(self.feature_names_in_):
            raise ValueError("input_features must match the ordered fit schema")
        return np.asarray(self.output_features_, dtype=object)

    # --- Internals -------------------------------------------------------------

    def _prepare(
        self, X: pd.DataFrame, y: object, groups: object
    ) -> tuple[np.ndarray, np.ndarray]:
        """Invalidate old evidence and validate base fit inputs without constructing folds."""
        for name in tuple(vars(self)):
            if name.endswith("_"):
                delattr(self, name)
        self._smoothing = _require_smoothing(self.smoothing)
        self._n_splits = _require_bounded_integer(self.n_splits, "n_splits", N_SPLITS_BOUNDS)
        self._seed = _require_bounded_integer(self.seed, "seed", SEED_BOUNDS)
        self._max_categories = _require_bounded_integer(
            self.max_categories, "max_categories", MAX_CATEGORIES_BOUNDS
        )
        build_target_encoding_registry()
        categories = _validate_frame(X)
        _validate_identities(X, y, groups)
        validate_inner_inputs(y, groups)
        if set(y.tolist()) != {0, 1}:
            raise BlockedExperimentError(
                "one_class_inner_partition", "both classes are required for target encoding"
            )
        observed = len({value for value in categories if value is not None})
        if observed > self._max_categories:
            raise ValueError(
                f"{SOURCE_COLUMN} cardinality {observed} exceeds max_categories "
                f"{self._max_categories}"
            )
        return categories, y.to_numpy(dtype=np.int64)

    def _record(self, X: pd.DataFrame, mapping: dict[str, float], prior: float) -> None:
        self.full_training_map_ = mapping
        self.training_prior_ = prior
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.output_features_ = (*tuple(X.columns), OUTPUT_NAME)
        self.smoothing_ = self.smoothing
        self.n_splits_ = self.n_splits
        self.seed_ = self.seed
        self.max_categories_ = self.max_categories
        self.registry_ = build_target_encoding_registry()

    def _append(self, X: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
        output = X.copy(deep=True)
        output[OUTPUT_NAME] = pd.Series(values, index=X.index, dtype=np.float64)
        return output

    def _check_unchanged_parameters(self) -> None:
        check_is_fitted(self, "output_features_")
        if (self.smoothing, self.n_splits, self.seed, self.max_categories) != (
                self.smoothing_, self.n_splits_, self.seed_, self.max_categories_):
            raise ValueError("target-encoding parameter changed; fit again before use")
