"""Bounded candidate encodings; callers must fit frequency on training folds only.

Both recipes require the frozen nine raw predictors before imputation. Bit i
represents nullness of BITMASK_COLUMNS[i], independently of raw input order.
Frequency is nonmissing category count / total training rows (including null
rows in the denominator); missing and unseen categories always receive zero.
Neither recipe consumes labels or authorizes campaign execution.
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

from project.features.input_encoding import (
    ABSOLUTE_CALENDAR_YEAR,
    missing_values,
    require_year_built_encoding,
    year_built_values,
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

BITMASK = "feat_missingness_bitmask"
FREQUENCY = "feat_frequency_encoding"
# Frozen baseline config: numeric predictors followed by categorical predictors.
BITMASK_COLUMNS = (
    "meter_reading", "square_feet", "year_built", "floor_count", "hour",
    "weekday", "month", "year", "primary_use",
)
MAX_CATEGORIES = 4096
OUTPUT_NAMES = {BITMASK: "missingness_bitmask", FREQUENCY: "primary_use_frequency"}


def _require_recipe(recipe_id: str) -> None:
    if not isinstance(recipe_id, str) or recipe_id not in OUTPUT_NAMES:
        raise ValueError(f"unsupported encoding recipe: {recipe_id!r}")


def build_encoding_registry(recipe_id: str, year_built_encoding: str = ABSOLUTE_CALENDAR_YEAR) -> FeatureRegistry:
    """Return verified baseline lineage plus one explicitly bounded feature.

    OUTER_TRAIN denotes training-fold-only fitting, including any inner folds.
    No caller-provided flags can upgrade an unsafe baseline source.
    """
    _require_recipe(recipe_id)
    require_year_built_encoding(year_built_encoding)
    baseline = (build_lead_baseline_registry() if year_built_encoding == ABSOLUTE_CALENDAR_YEAR
                else build_lead_baseline_registry(year_built_encoding))
    definitions = []
    for name in BITMASK_COLUMNS:
        source = baseline.get(name)
        if (source is None or source.provenance_status is not ProvenanceStatus.VERIFIED
                or source.prediction_time_availability is not AvailabilityStatus.AVAILABLE
                or source.target_dependence is not TargetDependence.INDEPENDENT
                or source.source_columns != (name,)):
            raise ValueError(f"unsafe baseline source lineage/provenance/availability/target: {name}")
        definitions.append(source)
    sources = BITMASK_COLUMNS if recipe_id == BITMASK else ("primary_use",)
    dates = [baseline.get(name).available_at for name in sources
             if baseline.get(name).available_at is not None]
    meaning = (
        "Pre-imputation uint16 null bitmask, bits 0..8 in order "
        + ", ".join(BITMASK_COLUMNS) + "; range 0..511, no vocabulary expansion. "
        + f"year_built contract: {year_built_encoding}; encoded 255 or null is missing, never zero."
        if recipe_id == BITMASK else
        "Nonmissing primary_use category count / total training-fold rows including null rows; "
        "fit vocabulary only on training fold, never adapt during transform."
    )
    definitions.append(FeatureMetadata(
        name=OUTPUT_NAMES[recipe_id], family="candidate_encoding", source_columns=sources,
        semantic_meaning=meaning, prediction_time_availability=AvailabilityStatus.AVAILABLE,
        target_dependence=TargetDependence.INDEPENDENT,
        transform_fit_boundary=(TransformFitBoundary.FIXED if recipe_id == BITMASK
                                else TransformFitBoundary.OUTER_TRAIN),
            fallback_behavior=("Native null sets bit to 1; under explicit encoded contract, "
                               "year_built 255 also sets its bit. Known values set bit to 0; "
                               "zero is known; no missing mask values."
                           if recipe_id == BITMASK else "Missing and unseen category: 0.0."),
        resource_cost=ResourceCost.LOW, provenance_status=ProvenanceStatus.VERIFIED,
        available_at=max(dates) if dates else None,
    ))
    return FeatureRegistry(definitions)


def _validate_frame(frame: pd.DataFrame, year_built_encoding: str = ABSOLUTE_CALENDAR_YEAR) -> None:
    require_year_built_encoding(year_built_encoding)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("encoding input must be a pandas DataFrame")
    if (frame.columns.has_duplicates or len(frame.columns) != len(BITMASK_COLUMNS)
            or set(frame.columns) != set(BITMASK_COLUMNS)):
        raise ValueError("schema requires exactly the nine unique approved baseline predictor columns")
    for name in BITMASK_COLUMNS:
        column = frame[name]
        if name == "primary_use":
            if (isinstance(column.dtype, pd.CategoricalDtype)
                    and any(not isinstance(value, str) for value in column.cat.categories)):
                raise TypeError("primary_use must declare string categories or missing values")
            if (not (is_object_dtype(column.dtype) or is_string_dtype(column.dtype)
                     or isinstance(column.dtype, pd.CategoricalDtype))
                    or any(not isinstance(value, str) for value in column.dropna())):
                raise TypeError("primary_use must contain strings or missing values")
        else:
            if name == "year_built" and year_built_encoding != ABSOLUTE_CALENDAR_YEAR:
                year_built_values(column, year_built_encoding)
            if (not is_numeric_dtype(column.dtype) or is_bool_dtype(column.dtype)
                    or is_complex_dtype(column.dtype)):
                raise TypeError(f"{name} must have a real numeric nonboolean dtype")
            if np.isinf(column.to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError(f"{name} contains nonfinite nonmissing values")


class EncodingFeatureTransformer(TransformerMixin, BaseEstimator):
    """Append one matrix recipe, preserving raw values, row index and column order.

    Fit ignores y and rejects zero rows. Transform requires the ordered fit
    schema; numeric widths/nullable storage and category vocabularies may differ
    provided values remain in the same semantic dtype domains. No calendar
    arithmetic is performed. Frequency vocabulary is capped at 4096 observed
    nonmissing categories; neither recipe expands the output beyond ten columns.
    """

    def __init__(self, recipe_id: str, year_built_encoding: str = ABSOLUTE_CALENDAR_YEAR) -> None:
        self.recipe_id = recipe_id
        self.year_built_encoding = year_built_encoding

    def fit(self, X: pd.DataFrame, y: Any = None) -> EncodingFeatureTransformer:
        _require_recipe(self.recipe_id)
        _validate_frame(X, self.year_built_encoding)
        if len(X) == 0:
            raise ValueError("fit requires at least one row")
        registry = build_encoding_registry(self.recipe_id, self.year_built_encoding)
        frequencies = {}
        if self.recipe_id == FREQUENCY:
            if X["primary_use"].nunique(dropna=True) > MAX_CATEGORIES:
                raise ValueError(f"primary_use cardinality exceeds {MAX_CATEGORIES}")
            counts = X["primary_use"].value_counts(dropna=True)
            frequencies = {category: float(count / len(X)) for category, count in counts.items()
                           if count > 0}
        self.frequencies_ = frequencies
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.output_features_ = tuple(X.columns) + (OUTPUT_NAMES[self.recipe_id],)
        self.recipe_id_ = self.recipe_id
        self.year_built_encoding_ = self.year_built_encoding
        self.registry_ = registry
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "output_features_")
        if self.recipe_id != self.recipe_id_ or self.year_built_encoding != self.year_built_encoding_:
            raise ValueError("recipe parameter changed; fit again")
        _validate_frame(X, self.year_built_encoding_)
        if tuple(X.columns) != tuple(self.feature_names_in_):
            raise ValueError("transform schema must match ordered fit columns")
        output = X.copy(deep=True)
        if self.recipe_id_ == BITMASK:
            mask = np.zeros(len(X), dtype=np.uint16)
            for bit, name in enumerate(BITMASK_COLUMNS):
                mask |= missing_values(X[name], self.year_built_encoding_).astype(np.uint16) << bit
            output["missingness_bitmask"] = mask
        else:
            output["primary_use_frequency"] = np.asarray([
                0.0 if pd.isna(value) else self.frequencies_.get(value, 0.0)
                for value in X["primary_use"]
            ], dtype=np.float64)
        return output

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "output_features_")
        if self.recipe_id != self.recipe_id_ or self.year_built_encoding != self.year_built_encoding_:
            raise ValueError("recipe parameter changed; fit again")
        if input_features is not None and tuple(input_features) != tuple(self.feature_names_in_):
            raise ValueError("input_features must match ordered fit schema")
        return np.asarray(self.output_features_, dtype=object)


def make_encoding_transformer(recipe_id: str, year_built_encoding: str = ABSOLUTE_CALENDAR_YEAR) -> EncodingFeatureTransformer:
    """Return a fresh cloneable transformer for one of the two bounded IDs."""
    _require_recipe(recipe_id)
    return EncodingFeatureTransformer(recipe_id, year_built_encoding)
