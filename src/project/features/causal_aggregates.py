"""Causal past-only rolling window transformer with zero lookahead bias and cold-start fallback."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from project.features.registry import (
    ABSOLUTE_CALENDAR_YEAR,
    AvailabilityStatus,
    FeatureMetadata,
    FeatureRegistry,
    ProvenanceStatus,
    ResourceCost,
    TargetDependence,
    TransformFitBoundary,
    build_lead_baseline_registry,
)

CAUSAL_ROLLING_FEATURES: tuple[str, ...] = (
    "meter_reading_roll_mean_24h",
    "meter_reading_roll_std_24h",
    "meter_reading_roll_mean_168h",
    "meter_reading_roll_std_168h",
)


def build_causal_aggregates_registry(
    year_built_encoding: str = ABSOLUTE_CALENDAR_YEAR,
) -> FeatureRegistry:
    """Return feature registry augmented with causal rolling aggregate features."""
    baseline = build_lead_baseline_registry(year_built_encoding=year_built_encoding)
    causal_meta = [
        FeatureMetadata(
            name=name,
            family="causal_aggregates",
            source_columns=("meter_reading",),
            semantic_meaning=f"Causal past-only rolling aggregation for {name}.",
            prediction_time_availability=AvailabilityStatus.AVAILABLE,
            target_dependence=TargetDependence.INDEPENDENT,
            transform_fit_boundary=TransformFitBoundary.OUTER_TRAIN,
            fallback_behavior="Fitted category-median cold-start fallback.",
            resource_cost=ResourceCost.LOW,
            provenance_status=ProvenanceStatus.VERIFIED,
        )
        for name in CAUSAL_ROLLING_FEATURES
    ]
    return FeatureRegistry([*(baseline.get(n) for n in baseline.names), *causal_meta])


class CausalRollingWindowTransformer(BaseEstimator, TransformerMixin):
    """Compute strictly past-only rolling aggregations per building.

    Invariants enforced:
    1. Zero lookahead: The value at timestamp t is strictly computed over t-1, t-2, ...
       achieved via shift(1) before any rolling window calculation.
    2. Zero target leakage: Operates only on independent predictor features or past observed series.
    3. Deterministic cold-start fallback: For t=0 or unseen buildings, uses fitted training
       category/global medians rather than future lookahead, zero, or unhandled NaNs.
    """

    def __init__(
        self,
        target_column: str = "meter_reading",
        windows: tuple[int, ...] = (24, 168),
        min_periods: int = 1,
        group_column: str = "building_id",
        timestamp_column: str = "timestamp",
        category_column: str = "primary_use",
    ) -> None:
        self.target_column = target_column
        self.windows = tuple(windows)
        self.min_periods = min_periods
        self.group_column = group_column
        self.timestamp_column = timestamp_column
        self.category_column = category_column

    def fit(self, X: pd.DataFrame, y: Any = None) -> CausalRollingWindowTransformer:
        """Fit deterministic global and category-level fallbacks from training data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if self.target_column not in X.columns:
            raise ValueError(f"target_column '{self.target_column}' missing from X")

        target_series = X[self.target_column].dropna()
        self.global_fallback_ = float(target_series.median()) if not target_series.empty else 0.0

        if self.category_column in X.columns:
            grouped = X.groupby(self.category_column)[self.target_column].median().dropna()
            self.category_fallbacks_ = {cat: float(val) for cat, val in grouped.items()}
        else:
            self.category_fallbacks_ = {}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply causal shift(1) and compute past-only rolling statistics."""
        if not hasattr(self, "global_fallback_"):
            raise ValueError("Transformer must be fitted before calling transform")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        out = X.copy()
        orig_index = out.index

        # If grouping column exists, compute group by group; else compute globally
        has_group = self.group_column in out.columns
        has_time = self.timestamp_column in out.columns
        has_cat = self.category_column in out.columns

        new_features: dict[str, pd.Series] = {}
        for w in self.windows:
            new_features[f"{self.target_column}_roll_mean_{w}h"] = pd.Series(index=orig_index, dtype=float)
            new_features[f"{self.target_column}_roll_std_{w}h"] = pd.Series(index=orig_index, dtype=float)

        groups = out.groupby(self.group_column) if has_group else [(None, out)]

        for _, group in groups:
            sorted_group = group.sort_values(self.timestamp_column) if has_time else group
            sorted_idx = sorted_group.index

            # STRICT CAUSALITY: shift(1) guarantees row t does NOT see row t
            shifted = sorted_group[self.target_column].shift(1)

            # Determine category fallback
            fallback = self.global_fallback_
            if has_cat and not group[self.category_column].empty:
                cat = group[self.category_column].iloc[0]
                fallback = self.category_fallbacks_.get(cat, self.global_fallback_)

            for w in self.windows:
                mean_col = f"{self.target_column}_roll_mean_{w}h"
                std_col = f"{self.target_column}_roll_std_{w}h"

                # Rolling calculation strictly on shifted data
                roll_mean = shifted.rolling(window=w, min_periods=self.min_periods).mean()
                roll_std = shifted.rolling(window=w, min_periods=self.min_periods).std()

                # Fill the initial missing past (e.g. t=0) with deterministic fallback
                roll_mean = roll_mean.fillna(fallback)
                roll_std = roll_std.fillna(0.0)

                new_features[mean_col].loc[sorted_idx] = roll_mean
                new_features[std_col].loc[sorted_idx] = roll_std

        for col_name, series in new_features.items():
            out[col_name] = series.loc[orig_index]

        return out
