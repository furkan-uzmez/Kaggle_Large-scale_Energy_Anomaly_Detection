"""Outer-train-fitted meter distribution features; fitting is not compute approval.

Both methods append exactly one named float64 column and preserve every raw
predictor. Fit consumes training ``meter_reading`` values only: labels, group
metadata, history and held-out rows cannot reach fitted state. Missing meter
values are excluded from fitting and remain missing in the output, so a later
imputer stays the single owner of infill.

Bin: quantile edges at i/n_bins are linearly interpolated on the finite training
values; duplicate interior edges collapse, assignment is
``searchsorted(interior_edges, value, side='right')``, and both tails saturate
in the edge bins. Quantile: the exact empirical midrank CDF
``(count(< x) + 0.5*count(== x))/n`` is learned on training values, stored at up
to ``n_quantiles`` evenly spaced deduplicated knots, interpolated linearly
between knots and saturated at the fitted minimum/maximum CDF. ``n_bins`` and
``n_quantiles`` are maxima; the actual counts are recorded.
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

BIN = "bin"
QUANTILE = "quantile"
OUTPUT_NAMES = {BIN: "meter_reading_bin", QUANTILE: "meter_reading_quantile"}
SOURCE_COLUMN = "meter_reading"
N_BINS_BOUNDS = (2, 64)
N_QUANTILES_BOUNDS = (2, 256)
_SEMANTICS = {
    BIN: (
        "Training-fitted meter_reading quantile bin index; interior edges at i/n_bins "
        "with duplicates collapsed, right-side assignment and saturating tail bins."
    ),
    QUANTILE: (
        "Training-fitted meter_reading empirical midrank CDF (count(<x)+0.5*count(==x))/n, "
        "linearly interpolated between deduplicated knots and saturated at the fitted range."
    ),
}


def _require_method(method: object) -> None:
    if not isinstance(method, str) or method not in OUTPUT_NAMES:
        raise ValueError(f"unsupported distribution method: {method!r}")


def _require_bounded_integer(value: object, label: str, bounds: tuple[int, int]) -> None:
    low, high = bounds
    if type(value) is not int or value < low or value > high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]; booleans are invalid")


def build_distribution_registry(method: str) -> FeatureRegistry:
    """Return verified baseline lineage plus one OUTER_TRAIN distribution feature.

    No caller-supplied flag can upgrade an unsafe baseline source: the raw meter
    producer must stay verified, available, target-independent and self-sourced.
    """
    _require_method(method)
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
        name=OUTPUT_NAMES[method], family="candidate_distribution",
        source_columns=(SOURCE_COLUMN,), semantic_meaning=_SEMANTICS[method],
        prediction_time_availability=AvailabilityStatus.AVAILABLE,
        target_dependence=TargetDependence.INDEPENDENT,
        transform_fit_boundary=TransformFitBoundary.OUTER_TRAIN,
        fallback_behavior="Missing meter values are excluded from fitting and stay missing.",
        resource_cost=ResourceCost.LOW, provenance_status=ProvenanceStatus.VERIFIED,
        available_at=source.available_at,
    ))
    return FeatureRegistry(definitions)


def _numeric_values(frame: pd.DataFrame, name: str) -> np.ndarray:
    column = frame[name]
    if (not is_numeric_dtype(column.dtype) or is_bool_dtype(column.dtype)
            or is_complex_dtype(column.dtype)):
        raise TypeError(f"{name} must have a real numeric, nonboolean dtype")
    values = column.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError(f"{name} contains nonfinite nonmissing values")
    domains = {"hour": (0, 23), "weekday": (0, 6), "month": (1, 12)}
    if name in domains:
        low, high = domains[name]
        known = values[~np.isnan(values)]
        if ((known < low) | (known > high) | (known != np.floor(known))).any():
            raise ValueError(f"{name} must be integer-valued in [{low}, {high}] or missing")
    return values


def _validate_frame(frame: pd.DataFrame) -> np.ndarray:
    """Allow any ordered subset of approved raw predictors that includes the meter."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("distribution input must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError("distribution input columns must be unique")
    if frame.index.has_duplicates or frame.index.to_frame(index=False).isna().any().any():
        raise ValueError("distribution input index must be unique and nonnull")
    approved = set(build_lead_baseline_registry().names)
    if any(not isinstance(name, str) or name not in approved for name in frame.columns):
        raise ValueError("distribution schema columns must be approved baseline predictors only")
    if SOURCE_COLUMN not in frame.columns:
        raise ValueError(f"required distribution column missing: {SOURCE_COLUMN}")
    numeric = {}
    for name in frame.columns:
        column = frame[name]
        if name == "primary_use":
            categorical = isinstance(column.dtype, pd.CategoricalDtype)
            if (not (is_object_dtype(column.dtype) or is_string_dtype(column.dtype) or categorical)
                    or any(not isinstance(value, str) for value in column.dropna())
                    or (categorical and any(not isinstance(value, str) for value in column.cat.categories))):
                raise TypeError("primary_use must contain string categories or missing values")
        else:
            numeric[name] = _numeric_values(frame, name)
    if "year" in numeric and "year_built" in numeric and (numeric["year_built"] > numeric["year"]).any():
        raise ValueError("year_built must not exceed the known observation year")
    return numeric[SOURCE_COLUMN]


def _linear_quantiles(training: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    positions = probabilities * (training.size - 1)
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    weight = positions - lower
    left, right = training[lower], training[upper]
    opposite = (left < 0) & (right > 0)
    result = np.empty_like(weight)
    result[opposite] = (1 - weight[opposite]) * left[opposite] + weight[opposite] * right[opposite]
    same = ~opposite
    result[same] = left[same] + weight[same] * (right[same] - left[same])
    return result


def _interpolate_cdf(values: np.ndarray, knots: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    bounded = np.clip(values, knots[0], knots[-1])
    upper = np.clip(np.searchsorted(knots, bounded, side="right"), 1, len(knots) - 1)
    left, right = knots[upper - 1], knots[upper]
    opposite = (left < 0) & (right > 0)
    weight = np.empty_like(bounded)
    # Scaling opposite-sign intervals avoids both overflowing differences and
    # rounding minimum subnormal endpoints to zero before subtraction.
    scale = np.maximum(-left[opposite], right[opposite])
    scaled_left = left[opposite] / scale
    scaled_right = right[opposite] / scale
    weight[opposite] = (bounded[opposite] / scale - scaled_left) / (scaled_right - scaled_left)
    same = ~opposite
    weight[same] = (bounded[same] - left[same]) / (right[same] - left[same])
    return cdf[upper - 1] + weight * (cdf[upper] - cdf[upper - 1])


def _training_support(values: np.ndarray) -> np.ndarray:
    finite = np.sort(values[~np.isnan(values)])
    if finite.size < 2:
        raise ValueError("fit requires at least two finite training meter values")
    if finite[0] == finite[-1]:
        raise ValueError("training meter values have no distinct variation")
    return finite


def _midrank_cdf(sorted_training: np.ndarray, probe: np.ndarray) -> np.ndarray:
    below = np.searchsorted(sorted_training, probe, side="left")
    at_or_below = np.searchsorted(sorted_training, probe, side="right")
    return (below + 0.5 * (at_or_below - below)) / sorted_training.size


class MeterDistributionTransformer(TransformerMixin, BaseEstimator):
    """Append one bounded, training-fitted meter distribution feature.

    Fit rejects empty or degenerate training support and ignores ``y``; neither
    method accepts group or history data. Transform preserves row order, index,
    nulls and every original column, and never adapts to held-out batches.
    Place this transformer before any learned imputation or encoding.
    """

    def __init__(self, method: str, n_bins: int = 8, n_quantiles: int = 64) -> None:
        self.method = method
        self.n_bins = n_bins
        self.n_quantiles = n_quantiles

    def fit(self, X: pd.DataFrame, y: Any = None) -> MeterDistributionTransformer:
        _require_method(self.method)
        _require_bounded_integer(self.n_bins, "n_bins", N_BINS_BOUNDS)
        _require_bounded_integer(self.n_quantiles, "n_quantiles", N_QUANTILES_BOUNDS)
        training = _training_support(_validate_frame(X))
        registry = build_distribution_registry(self.method)
        if self.method == BIN:
            edges = _linear_quantiles(training, np.arange(self.n_bins + 1) / self.n_bins)
            self.bin_interior_edges_ = np.unique(edges[1:-1])
            self.n_bin_edges_ = int(self.bin_interior_edges_.size)
        else:
            knots = np.unique(_linear_quantiles(
                training, np.linspace(0.0, 1.0, self.n_quantiles)
            ))
            self.quantile_knots_ = knots
            self.quantile_cdf_ = _midrank_cdf(training, knots)
            self.n_quantile_knots_ = int(knots.size)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.output_features_ = (*tuple(X.columns), OUTPUT_NAMES[self.method])
        self.method_ = self.method
        self.n_bins_ = self.n_bins
        self.n_quantiles_ = self.n_quantiles
        self.registry_ = registry
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_unchanged_parameters()
        values = _validate_frame(X)
        if tuple(X.columns) != tuple(self.feature_names_in_):
            raise ValueError("distribution transform schema must match the ordered fit columns")
        known = ~np.isnan(values)
        feature = np.full(values.shape, np.nan, dtype=np.float64)
        if self.method_ == BIN:
            feature[known] = np.searchsorted(
                self.bin_interior_edges_, values[known], side="right"
            ).astype(np.float64)
        else:
            feature[known] = _interpolate_cdf(values[known], self.quantile_knots_, self.quantile_cdf_)
        output = X.copy(deep=True)
        output[OUTPUT_NAMES[self.method_]] = feature
        return output

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        self._check_unchanged_parameters()
        if input_features is not None and tuple(input_features) != tuple(self.feature_names_in_):
            raise ValueError("input_features must match the ordered fit schema")
        return np.asarray(self.output_features_, dtype=object)

    def _check_unchanged_parameters(self) -> None:
        check_is_fitted(self, "output_features_")
        if (self.method, self.n_bins, self.n_quantiles) != (
                self.method_, self.n_bins_, self.n_quantiles_):
            raise ValueError("distribution parameter changed; fit again before use")
