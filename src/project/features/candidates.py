"""Fixed, row-local candidate recipes; implementation is not compute approval.

Inputs are pandas predictor frames containing only frozen baseline columns.
Outputs retain those columns in order and append deterministic features. Fit
records schema only: no labels, category vocabulary, or population statistics.
The missingness recipe flags every supplied predictor before imputation; there
is deliberately no data-fitted missing-rate threshold.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from project.features.input_encoding import (
    ABSOLUTE_CALENDAR_YEAR,
    construction_years,
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

MISSINGNESS = "feat_missingness_indicators"
SAFE_NUMERIC = "feat_safe_calendar_numeric_transforms"
INTERACTIONS = "feat_categorical_interactions"
HOUR_PERIOD = 24
CALENDAR_DOMAINS = {
    "hour": (0, 23), "weekday": (0, 6), "month": (1, 12),
    "year": (1, 9999), "year_built": (1, 9999),
}
REQUIRED_COLUMNS = {
    MISSINGNESS: (),
    SAFE_NUMERIC: ("meter_reading", "hour", "year", "year_built"),
    INTERACTIONS: ("primary_use", "hour", "weekday", "month"),
}


def _require_recipe(recipe_id: str) -> None:
    if not isinstance(recipe_id, str) or recipe_id not in REQUIRED_COLUMNS:
        raise ValueError(f"unsupported deterministic candidate recipe: {recipe_id!r}")


def _feature_definitions(recipe_id: str, columns: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    if recipe_id == MISSINGNESS:
        return {f"{name}_missing": (name,) for name in columns}
    if recipe_id == SAFE_NUMERIC:
        return {
            "meter_reading_signed_log1p": ("meter_reading",),
            "hour_sin": ("hour",), "hour_cos": ("hour",),
            "building_age": ("year", "year_built"),
        }
    return {f"primary_use_x_{name}": ("primary_use", name) for name in ("hour", "weekday", "month")}


def build_candidate_registry(recipe_id: str, year_built_encoding: str = ABSOLUTE_CALENDAR_YEAR) -> FeatureRegistry:
    """Extend a fresh verified baseline registry with FIXED candidate metadata.

    No flags supplied by callers can approve provenance. Source metadata must
    be verified, available, target-independent, and self-sourced baseline data.
    The original nine-predictor factory is left unchanged.
    """
    _require_recipe(recipe_id)
    require_year_built_encoding(year_built_encoding)
    baseline = (build_lead_baseline_registry() if year_built_encoding == ABSOLUTE_CALENDAR_YEAR
                else build_lead_baseline_registry(year_built_encoding))
    for name in baseline.names:
        source = baseline.get(name)
        if (source.provenance_status is not ProvenanceStatus.VERIFIED
                or source.prediction_time_availability is not AvailabilityStatus.AVAILABLE
                or source.target_dependence is not TargetDependence.INDEPENDENT
                or source.source_columns != (name,)):
            raise ValueError(f"unsafe baseline source provenance/availability/target lineage: {name}")
    definitions = [baseline.get(name) for name in baseline.names]
    for name, sources in _feature_definitions(recipe_id, baseline.names).items():
        availability_dates = [baseline.get(s).available_at for s in sources
                              if baseline.get(s).available_at is not None]
        definitions.append(FeatureMetadata(
            name=name, family="categorical" if recipe_id == INTERACTIONS else "deterministic_candidate",
            source_columns=sources,
            semantic_meaning=(
                f"Observation calendar year minus decoded construction year ({year_built_encoding}); "
                "missing construction years yield missing age." if name == "building_age" else
                f"{recipe_id}: {name} derived row-locally from {', '.join(sources)}"
            ),
            prediction_time_availability=AvailabilityStatus.AVAILABLE,
            target_dependence=TargetDependence.INDEPENDENT,
            transform_fit_boundary=TransformFitBoundary.FIXED,
            fallback_behavior=(f"0/1 pre-imputation missing flag; year_built uses {year_built_encoding}; "
                               "encoded contract includes 255, never zero" if recipe_id == MISSINGNESS
                               else "Propagate missing source values; reject invalid nonmissing domains"),
            resource_cost=ResourceCost.LOW, provenance_status=ProvenanceStatus.VERIFIED,
            available_at=max(availability_dates) if availability_dates else None,
        ))
    return FeatureRegistry(definitions)


def _numeric_array(frame: pd.DataFrame, name: str) -> np.ndarray:
    column = frame[name]
    if (not is_numeric_dtype(column.dtype) or is_bool_dtype(column.dtype)
            or is_complex_dtype(column.dtype)):
        raise TypeError(f"{name} must have a real numeric, nonboolean dtype")
    values = column.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError(f"{name} contains nonfinite nonmissing values")
    if name in CALENDAR_DOMAINS:
        low, high = CALENDAR_DOMAINS[name]
        known = values[~np.isnan(values)]
        if ((known < low) | (known > high) | (known != np.floor(known))).any():
            raise ValueError(f"{name} must be integer-valued in [{low}, {high}] or missing")
    return values


def _validate_frame(frame: pd.DataFrame, recipe_id: str, year_built_encoding: str = ABSOLUTE_CALENDAR_YEAR) -> None:
    require_year_built_encoding(year_built_encoding)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("candidate input must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError("candidate input columns must be unique")
    allowed = build_lead_baseline_registry().names
    if not len(frame.columns) or any(not isinstance(n, str) or n not in allowed for n in frame.columns):
        raise ValueError("candidate schema columns must be nonempty approved baseline predictors only")
    missing = set(REQUIRED_COLUMNS[recipe_id]) - set(frame.columns)
    if missing:
        raise ValueError(f"required candidate columns missing: {sorted(missing)}")
    numeric = {}
    for name in frame.columns:
        if name == "primary_use":
            known = frame[name].dropna()
            if any(not isinstance(value, str) for value in known):
                raise TypeError("primary_use must contain strings or missing values")
        elif name == "year_built":
            numeric[name] = year_built_values(frame[name], year_built_encoding)
        else:
            numeric[name] = _numeric_array(frame, name)
    if ("year" in numeric and "year_built" in numeric
            and (year_built_encoding == ABSOLUTE_CALENDAR_YEAR or recipe_id == SAFE_NUMERIC)
            and (construction_years(frame["year_built"], year_built_encoding) > numeric["year"]).any()):
        raise ValueError("year_built must not exceed the known observation year")


def _add_missingness(frame: pd.DataFrame, year_built_encoding: str = ABSOLUTE_CALENDAR_YEAR) -> pd.DataFrame:
    output = frame.copy(deep=True)
    for name in frame.columns:
        output[f"{name}_missing"] = missing_values(frame[name], year_built_encoding).astype(np.uint8)
    return output


def _add_safe_numeric(frame: pd.DataFrame, year_built_encoding: str = ABSOLUTE_CALENDAR_YEAR) -> pd.DataFrame:
    """Signed log preserves negative zero; null years produce null age.

    Hour uses 2*pi*hour/24. Decode construction years under the explicit
    contract; a known construction year after a known observation is invalid.
    """
    output = frame.copy(deep=True)
    meter = _numeric_array(frame, "meter_reading")
    output["meter_reading_signed_log1p"] = np.copysign(np.log1p(np.abs(meter)), meter)
    angle = 2 * np.pi * _numeric_array(frame, "hour") / HOUR_PERIOD
    output["hour_sin"], output["hour_cos"] = np.sin(angle), np.cos(angle)
    output["building_age"] = _numeric_array(frame, "year") - construction_years(frame["year_built"], year_built_encoding)
    return output


def _add_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode lossless JSON [category, calendar integer] pairs, no vocabulary fit.

    Either missing component yields np.nan, not a synthetic category token.
    Exact category text (including delimiters/empty strings) is preserved.
    """
    output = frame.copy(deep=True)
    for name in ("hour", "weekday", "month"):
        pairs = zip(frame["primary_use"], _numeric_array(frame, name), strict=True)
        output[f"primary_use_x_{name}"] = pd.array([
            np.nan if pd.isna(category) or np.isnan(value) else
            json.dumps([category, int(value)], ensure_ascii=False, separators=(",", ":"))
            for category, value in pairs
        ], dtype=object)
    return output


TRANSFORMS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    MISSINGNESS: _add_missingness,
    SAFE_NUMERIC: _add_safe_numeric,
    INTERACTIONS: _add_interactions,
}


class CandidateFeatureTransformer(TransformerMixin, BaseEstimator):
    """Append one explicit matrix recipe with strict ordered predictor schema.

    Empty typed frames are supported. Fit ignores y and records only input/output
    names; transform preserves row order/index, nulls, and all original columns.
    Place this transformer before any learned imputation or encoding in Pipeline.
    """

    def __init__(self, recipe_id: str, year_built_encoding: str = ABSOLUTE_CALENDAR_YEAR) -> None:
        self.recipe_id = recipe_id
        self.year_built_encoding = year_built_encoding

    def fit(self, X: pd.DataFrame, y: Any = None) -> CandidateFeatureTransformer:
        _require_recipe(self.recipe_id)
        _validate_frame(X, self.recipe_id, self.year_built_encoding)
        registry = build_candidate_registry(self.recipe_id, self.year_built_encoding)
        columns = tuple(X.columns)
        derived = tuple(_feature_definitions(self.recipe_id, columns))
        self.feature_names_in_ = np.asarray(columns, dtype=object)
        self.n_features_in_ = len(columns)
        self.output_features_ = columns + derived
        self.recipe_id_ = self.recipe_id
        self.year_built_encoding_ = self.year_built_encoding
        self.registry_ = registry
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "output_features_")
        if self.recipe_id != self.recipe_id_ or self.year_built_encoding != self.year_built_encoding_:
            raise ValueError("recipe parameter changed; fit again before transform")
        _validate_frame(X, self.recipe_id_, self.year_built_encoding_)
        if tuple(X.columns) != tuple(self.feature_names_in_):
            raise ValueError("candidate transform schema must match the ordered fit columns")
        if self.recipe_id_ == MISSINGNESS:
            return _add_missingness(X, self.year_built_encoding_)
        if self.recipe_id_ == SAFE_NUMERIC:
            return _add_safe_numeric(X, self.year_built_encoding_)
        return _add_interactions(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "output_features_")
        if self.recipe_id != self.recipe_id_ or self.year_built_encoding != self.year_built_encoding_:
            raise ValueError("recipe or input encoding parameter changed; fit again")
        if input_features is not None and tuple(input_features) != tuple(self.feature_names_in_):
            raise ValueError("input_features must match the ordered fit schema")
        return np.asarray(self.output_features_, dtype=object)


def make_candidate_transformer(recipe_id: str, year_built_encoding: str = ABSOLUTE_CALENDAR_YEAR) -> CandidateFeatureTransformer:
    """Return a fresh sklearn-cloneable transformer for one supported matrix ID."""
    _require_recipe(recipe_id)
    return CandidateFeatureTransformer(recipe_id, year_built_encoding)
