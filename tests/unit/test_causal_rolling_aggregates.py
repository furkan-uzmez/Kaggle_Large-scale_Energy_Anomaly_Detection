"""TDD contracts for CausalRollingWindowTransformer (strictly past-only, zero lookahead)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from project.features.causal_aggregates import CausalRollingWindowTransformer


@pytest.fixture
def sample_timeseries_data():
    """Create deterministic hourly time series for two buildings."""
    dates = pd.date_range("2016-01-01", periods=100, freq="h")
    bldg_1 = pd.DataFrame({
        "building_id": "bldg_1",
        "primary_use": "Office",
        "timestamp": dates,
        "meter_reading": np.arange(100, dtype=float),
    })
    bldg_2 = pd.DataFrame({
        "building_id": "bldg_2",
        "primary_use": "Education",
        "timestamp": dates,
        "meter_reading": np.arange(100, dtype=float) * 2.0,
    })
    return pd.concat([bldg_1, bldg_2], ignore_index=True)


def test_causal_transformer_is_available():
    transformer = CausalRollingWindowTransformer(windows=(24, 168))
    assert hasattr(transformer, "fit")
    assert hasattr(transformer, "transform")


def test_causal_rolling_strictly_excludes_current_and_future_timestamps(sample_timeseries_data):
    """Row at time t MUST NOT reflect row at time t or t+k (shift(1) invariant)."""
    transformer = CausalRollingWindowTransformer(target_column="meter_reading", windows=(24,))
    transformer.fit(sample_timeseries_data)
    transformed_1 = transformer.transform(sample_timeseries_data)

    # Modify future rows (index 50 to 99) in bldg_1
    mutated_data = sample_timeseries_data.copy(deep=True)
    bldg_1_mask = mutated_data["building_id"] == "bldg_1"
    mutated_data.loc[bldg_1_mask & (mutated_data.index >= 50), "meter_reading"] += 1000.0

    transformed_2 = transformer.transform(mutated_data)

    # For any row t < 50, the rolling value MUST be completely identical
    # And at row 50, it MUST NOT reflect the mutation at row 50!
    col = "meter_reading_roll_mean_24h"
    b1_t1 = transformed_1.loc[bldg_1_mask, col].to_numpy()
    b1_t2 = transformed_2.loc[bldg_1_mask, col].to_numpy()

    # Invariant: Up to index 50 inclusive, the rolling feature cannot know about mutation at 50!
    np.testing.assert_allclose(b1_t1[:51], b1_t2[:51])


def test_first_row_uses_fallback_not_nan_or_future(sample_timeseries_data):
    """At t=0, there is no past row; it must use fitted fallback, never forward-fill."""
    transformer = CausalRollingWindowTransformer(target_column="meter_reading", windows=(24,))
    transformer.fit(sample_timeseries_data)
    transformed = transformer.transform(sample_timeseries_data)

    col = "meter_reading_roll_mean_24h"
    # First row of each building
    first_rows = transformed.groupby("building_id")[col].first()
    assert not first_rows.isna().any()
    # It must equal the fitted fallback for that building's category
    assert first_rows["bldg_1"] == transformer.category_fallbacks_["Office"]
    assert first_rows["bldg_2"] == transformer.category_fallbacks_["Education"]


def test_unseen_held_out_building_uses_category_cold_start_fallback(sample_timeseries_data):
    """Validation/test building unseen in fit must use category fallback gracefully."""
    transformer = CausalRollingWindowTransformer(target_column="meter_reading", windows=(24,))
    transformer.fit(sample_timeseries_data)

    dates = pd.date_range("2016-01-01", periods=10, freq="h")
    unseen_bldg = pd.DataFrame({
        "building_id": "bldg_unseen",
        "primary_use": "Office",
        "timestamp": dates,
        "meter_reading": np.full(10, 500.0),
    })

    transformed = transformer.transform(unseen_bldg)
    col = "meter_reading_roll_mean_24h"
    assert not transformed[col].isna().any()
    # First row must use fitted fallback
    assert transformed[col].iloc[0] == transformer.category_fallbacks_["Office"]


def test_transformer_generates_mean_and_std_columns(sample_timeseries_data):
    transformer = CausalRollingWindowTransformer(target_column="meter_reading", windows=(24, 168))
    transformer.fit(sample_timeseries_data)
    transformed = transformer.transform(sample_timeseries_data)

    expected_cols = [
        "meter_reading_roll_mean_24h",
        "meter_reading_roll_std_24h",
        "meter_reading_roll_mean_168h",
        "meter_reading_roll_std_168h",
    ]
    for col in expected_cols:
        assert col in transformed.columns
        assert not transformed[col].isna().any()
