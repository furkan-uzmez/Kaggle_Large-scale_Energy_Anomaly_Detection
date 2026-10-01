from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def tiny_lead_tables() -> dict[str, pl.DataFrame]:
    timestamps = [datetime(2022, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(2)]
    train = pl.DataFrame([
        {"building_id": b, "timestamp": t, "meter_reading": float(b + i), "anomaly": i}
        for b in range(10)
        for i, t in enumerate(timestamps)
    ])
    test = pl.DataFrame({
        "row_id": [910, 110],
        "building_id": [100, 101],
        "timestamp": [timestamps[0], timestamps[1]],
        "meter_reading": [2.0, 3.0],
    })
    sample = test.select("row_id").with_columns(pl.Series("anomaly", [0, 0]))
    train_features = train.select("building_id", "timestamp", "meter_reading", "anomaly").with_columns(
        pl.lit(10.0).alias("temperature_mean")
    )
    test_features = test.select("row_id", "building_id", "timestamp", "meter_reading").with_columns(
        pl.Series("temperature_mean", [11.0, 12.0])
    )
    return {
        "train": train,
        "test": test,
        "sample_submission": sample,
        "train_features": train_features,
        "test_features": test_features,
    }


@pytest.fixture
def tiny_lead_tables_with_duplicate_test_features(tiny_lead_tables: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    return dict(tiny_lead_tables)


@pytest.fixture
def tiny_lead_csv_dir(tmp_path: Path, tiny_lead_tables_with_duplicate_test_features: dict[str, pl.DataFrame]) -> Path:
    raw_dir = tmp_path / "energy-anomaly-detection"
    raw_dir.mkdir(parents=True)
    for name, df in tiny_lead_tables_with_duplicate_test_features.items():
        filename = f"{name}.csv"
        df.write_csv(raw_dir / filename)
    return raw_dir
