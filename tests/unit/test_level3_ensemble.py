"""Unit tests for Level-3 Meta-Ensemble system.

Verifies:
1. Meta-2B Leaf Embedding Estimator (LightGBM leaf extractor -> OneHotEncoder -> sparse SGDClassifier log-loss).
2. Zero leakage in Meta-2B (leaf trees and encoder fit ONLY on outer training folds; validation is transform-only).
3. Level-3 Meta-Learner (Nested direct ROC-AUC bounded combination of Meta-2A and Meta-2B scores).
4. Full nested cross-validation pipeline producing canonical 24-column CANDIDATE_OOF_SCHEMA parquets.
5. End-to-end test prediction and submission generation conforming to Kaggle contracts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from scipy import sparse
from sklearn.metrics import roc_auc_score

from project.models.candidate_artifacts import (
    CANDIDATE_OOF_SCHEMA,
    ROW_COLUMNS,
    validate_candidate_oof,
    write_candidate_oof_batches,
)
from project.models.level3_ensemble import (
    Level3AucBoundedCombiner,
    Meta2BLeafEmbeddingEstimator,
    build_level3_submission,
    run_level3_nested_cv,
)
from project.submission import validate_submission_contract


def _make_synthetic_tabular_data(
    n_samples: int = 400,
    n_features: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic numeric features and correlated binary targets."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    logits = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(0.0, 0.2, size=n_samples)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(0.0, 1.0, size=n_samples) < probs).astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Unit Tests for Arm B (Meta-2B): Leaf Embedding Estimator
# ---------------------------------------------------------------------------


def test_meta_2b_fit_predict_probabilities_and_bounds() -> None:
    """Test that Meta2BLeafEmbeddingEstimator fits and predicts valid probabilities in [0, 1]."""
    X, y = _make_synthetic_tabular_data(n_samples=250, n_features=6, seed=42)
    estimator = Meta2BLeafEmbeddingEstimator(
        lgb_params={"n_estimators": 10, "max_depth": 3, "num_leaves": 7, "random_state": 42},
        linear_params={"alpha": 1e-3, "random_state": 42},
    )
    estimator.fit(X, y)
    probs = estimator.predict_proba(X)

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (len(y),)
    assert np.all(np.isfinite(probs))
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    # Check that model learned signal: AUC significantly better than random
    auc = roc_auc_score(y, probs)
    assert auc > 0.65


def test_meta_2b_uses_sparse_csr_matrix() -> None:
    """Test that Meta2B transforms leaf indices to scipy.sparse.csr_matrix."""
    X, y = _make_synthetic_tabular_data(n_samples=100, n_features=4, seed=42)
    estimator = Meta2BLeafEmbeddingEstimator(
        lgb_params={"n_estimators": 5, "max_depth": 2, "num_leaves": 4, "random_state": 42},
        linear_params={"alpha": 1e-2, "random_state": 42},
    )
    estimator.fit(X, y)

    # Check internal encoder produced sparse CSR
    leaves = estimator.extract_leaves(X)
    assert isinstance(leaves, np.ndarray)
    assert leaves.shape == (100, 5)

    sparse_X = estimator.transform_leaves_to_sparse(leaves)
    assert sparse.isspmatrix_csr(sparse_X)
    assert sparse_X.shape[0] == 100


def test_meta_2b_zero_leakage_transform_only() -> None:
    """Verify that validation partition is transform-only and never modifies training encoder."""
    X_tr, y_tr = _make_synthetic_tabular_data(n_samples=150, n_features=5, seed=1)
    X_val, _ = _make_synthetic_tabular_data(n_samples=50, n_features=5, seed=2)

    estimator = Meta2BLeafEmbeddingEstimator(
        lgb_params={"n_estimators": 8, "max_depth": 3, "num_leaves": 6, "random_state": 42},
    )
    estimator.fit(X_tr, y_tr)

    # Record training categories in encoder
    categories_before = [c.copy() for c in estimator.encoder_.categories_]
    n_features_in = estimator.encoder_.n_features_in_

    # Transform validation set (with potentially unseen leaf indices)
    val_probs = estimator.predict_proba(X_val)

    # Encoder state must not have changed at all
    assert estimator.encoder_.n_features_in_ == n_features_in
    for before, after in zip(categories_before, estimator.encoder_.categories_, strict=True):
        np.testing.assert_array_equal(before, after)

    assert len(val_probs) == 50
    assert np.all(np.isfinite(val_probs))


def test_meta_2b_dry_run_mock_mode() -> None:
    """Verify that dry_run mode executes quickly without real fitting."""
    X, y = _make_synthetic_tabular_data(n_samples=30, n_features=4, seed=42)
    estimator = Meta2BLeafEmbeddingEstimator(dry_run=True)
    estimator.fit(X, y)
    probs = estimator.predict_proba(X)

    assert len(probs) == len(y)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


# ---------------------------------------------------------------------------
# Unit Tests for Level-3 Meta-Learner (AUC Bounded Combiner)
# ---------------------------------------------------------------------------


def test_level3_combiner_weight_bounds_and_predictions() -> None:
    """Verify that Level3AucBoundedCombiner restricts weight to [0, 1] and produces valid probabilities."""
    rng = np.random.default_rng(42)
    n = 200
    y = rng.integers(0, 2, size=n)
    s_2a = rng.uniform(0.0, 1.0, size=n)
    s_2b = rng.uniform(0.0, 1.0, size=n)

    combiner = Level3AucBoundedCombiner(grid_resolution=51, random_state=42)
    combiner.fit(s_2a, s_2b, y)

    assert hasattr(combiner, "weight_")
    assert 0.0 <= combiner.weight_ <= 1.0

    preds = combiner.predict_proba(s_2a, s_2b)
    assert preds.shape == (n,)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)
    expected = combiner.weight_ * s_2a + (1.0 - combiner.weight_) * s_2b
    np.testing.assert_allclose(preds, expected, atol=1e-7)


def test_level3_combiner_prefers_superior_arm() -> None:
    """Test that combiner puts dominant weight on the superior arm."""
    rng = np.random.default_rng(42)
    n = 300
    latent = rng.normal(0, 1, size=n)
    y = (latent > 0).astype(np.int64)

    # Arm A is highly predictive (low noise)
    s_2a = 1.0 / (1.0 + np.exp(-(latent + rng.normal(0, 0.1, size=n))))
    # Arm B is pure noise
    s_2b = rng.uniform(0.0, 1.0, size=n)

    combiner_a = Level3AucBoundedCombiner(grid_resolution=101, random_state=42)
    combiner_a.fit(s_2a, s_2b, y)
    # Weight on Arm A should be close to 1.0
    assert combiner_a.weight_ >= 0.85

    # Reverse: Arm B is highly predictive, Arm A is noise
    combiner_b = Level3AucBoundedCombiner(grid_resolution=101, random_state=42)
    combiner_b.fit(s_2b, s_2a, y)
    # Weight on Arm A (which is noise) should be close to 0.0
    assert combiner_b.weight_ <= 0.15


def test_level3_combiner_edge_cases() -> None:
    """Test that combiner handles identical predictions or single class safely."""
    s = np.array([0.2, 0.5, 0.8])
    y_single = np.array([1, 1, 1])

    combiner = Level3AucBoundedCombiner()
    # Single class shouldn't raise exception
    combiner.fit(s, s, y_single)
    assert 0.0 <= combiner.weight_ <= 1.0
    preds = combiner.predict_proba(s, s)
    assert len(preds) == 3


# ---------------------------------------------------------------------------
# Unit Tests for Nested Cross-Validation and Candidate OOF Generation
# ---------------------------------------------------------------------------


def test_run_level3_nested_cv_schema_conformity() -> None:
    """Test that run_level3_nested_cv produces exact 24-column CANDIDATE_OOF_SCHEMA DataFrame."""
    rng = np.random.default_rng(42)
    # Create synthetic development DataFrame matching train_dev structure
    # 4 buildings, 20 timestamps each = 80 rows
    buildings = [101, 102, 103, 104]
    rows: list[dict[str, Any]] = []
    row_idx = 0
    for b in buildings:
        for t in range(20):
            rows.append({
                "row_index": row_idx,
                "building_id": b,
                "timestamp": np.datetime64("2016-01-01T00:00:00", "ns") + np.timedelta64(t, "h"),
                "meter_reading": float(rng.uniform(10.0, 100.0)),
                "square_feet": float(b * 100),
                "primary_use": "Education",
                "anomaly": int(rng.integers(0, 2)),
            })
            row_idx += 1

    train_dev = pl.DataFrame(rows)

    # 1 repeat, 2 folds
    dev_splits = pl.DataFrame({
        "repeat": [0, 0, 0, 0],
        "fold": [0, 0, 1, 1],
        "building_id": [101, 102, 103, 104],
        "split": ["dev", "dev", "dev", "dev"],
    })

    # Synthetic Meta-2A OOF (matching the 80 rows for repeat 0)
    meta_2a_rows: list[dict[str, Any]] = []
    for r in rows:
        fold_id = 0 if r["building_id"] in (101, 102) else 1
        meta_2a_rows.append({
            "candidate_id": "meta_challenger_lr",
            "recipe_id": "recipe_meta_challenger_lr",
            "model_family": "ensemble_meta",
            "model_config_id": "cfg_meta_lr",
            "row_index": r["row_index"],
            "repeat": 0,
            "fold": fold_id,
            "building_id": str(r["building_id"]),
            "timestamp": r["timestamp"],
            "prediction": float(rng.uniform(0.1, 0.9)),
            "target": r["anomaly"],
            "split_sha256": "0" * 64,
            "data_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "run_id": "test_run_2a",
            "split_seed": 42,
            "experiment_id": "test_exp",
            "arm_id": "challenger_lr",
            "score_kind": "positive_probability",
            "positive_label": 1,
            "execution_mode": "synthetic_smoke",
            "non_comparative": True,
            "n03_eligible": False,
            "model_fit_config_sha256": "3" * 64,
        })
    meta_2a_df = pl.DataFrame(meta_2a_rows)

    oof_df, metadata = run_level3_nested_cv(
        train_dev=train_dev,
        dev_splits=dev_splits,
        meta_2a_oof_df=meta_2a_df,
        feature_cols=["meter_reading", "square_feet"],
        target_col="anomaly",
        categorical_cols=["primary_use"],
        repeats=[0],
        folds=[0, 1],
        meta_2b_params={
            "lgb_params": {"n_estimators": 5, "max_depth": 2, "num_leaves": 4, "random_state": 42},
            "linear_params": {"alpha": 1e-3, "random_state": 42},
        },
        level3_params={"grid_resolution": 21, "random_state": 42},
        execution_mode="synthetic_smoke",
        run_id="test_run_l3",
        config_sha256="4" * 64,
        dry_run=False,
        smoke=True,
    )

    # 1. Exact 24 columns in CANDIDATE_OOF_SCHEMA order
    assert oof_df.columns == list(CANDIDATE_OOF_SCHEMA.names)

    # 2. Check no nulls anywhere
    for col in CANDIDATE_OOF_SCHEMA.names:
        assert oof_df[col].null_count() == 0, f"Column {col} has nulls"

    # 3. Check row count matches development partition
    assert oof_df.height == train_dev.height

    # 4. Check probabilities in [0, 1]
    preds = oof_df["prediction"].to_numpy()
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)

    # 5. Check metadata contains per-fold weights
    assert "fold_weights" in metadata
    assert len(metadata["fold_weights"]) == 2


def test_oof_parquet_streaming_and_validation(tmp_path: Path) -> None:
    """Test that OOF DataFrame passes write_candidate_oof_batches and validate_candidate_oof."""
    rng = np.random.default_rng(42)
    n_rows = 50
    records = []
    for i in range(n_rows):
        records.append({
            "candidate_id": "meta_level3",
            "recipe_id": "recipe_level3",
            "model_family": "ensemble_level3",
            "model_config_id": "cfg_level3",
            "row_index": i,
            "repeat": 0,
            "fold": i % 2,
            "building_id": str(100 + (i % 2)),
            "timestamp": i * 3600 * 1_000_000_000,
            "prediction": float(rng.uniform(0.01, 0.99)),
            "target": int(rng.integers(0, 2)),
            "split_sha256": "a" * 64,
            "data_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "run_id": "run_test_artifact",
            "split_seed": 42,
            "experiment_id": "level3_ensemble",
            "arm_id": "level3_meta",
            "score_kind": "positive_probability",
            "positive_label": 1,
            "execution_mode": "synthetic_smoke",
            "non_comparative": True,
            "n03_eligible": False,
            "model_fit_config_sha256": "d" * 64,
        })
    oof_df = pl.DataFrame(records).with_columns([
        pl.col("timestamp").cast(pl.Datetime("ns")),
    ])
    arrow_table = oof_df.select(CANDIDATE_OOF_SCHEMA.names).to_arrow().cast(CANDIDATE_OOF_SCHEMA)

    oof_file = tmp_path / "level3_candidate_oof.parquet"
    write_res = write_candidate_oof_batches(oof_file, [arrow_table], CANDIDATE_OOF_SCHEMA)
    assert write_res["status"] == "written_unvalidated"
    assert write_res["oof_record_count"] == n_rows

    # Validation
    expected_rows = oof_df.select([
        "row_index", "repeat", "fold", "building_id", "timestamp",
        "target", "split_seed", "split_sha256", "data_sha256", "config_sha256", "run_id"
    ])
    manifest = {
        "schema_version": 1,
        "execution_mode": "synthetic_smoke",
        "non_comparative": True,
        "n03_eligible": False,
        "source_notebook": "notebooks/02_baseline_and_candidate_design.ipynb",
        "source_cell_id": "Level-3 Orchestrator",
        "vault_row_indices": [],
        "vault_building_ids": [],
        "run_id": "run_test_artifact",
        "split_sha256": "a" * 64,
        "data_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "protocol": {
            "repeat_ids": [0],
            "fold_ids": [0, 1],
            "split_seeds": {"0": 42},
        },
        "candidates": {
            "meta_level3": {
                "recipe_id": "recipe_level3",
                "model_family": "ensemble_level3",
                "model_config_id": "cfg_level3",
                "experiment_id": "level3_ensemble",
                "arm_id": "level3_meta",
                "score_kind": "positive_probability",
                "positive_label": 1,
                "model_fit_config_sha256": "d" * 64,
            }
        },
    }
    validated = validate_candidate_oof(oof_df, expected_rows, manifest)
    assert validated["oof_record_count"] == n_rows


# ---------------------------------------------------------------------------
# Unit Tests for Submission Pipeline Integration
# ---------------------------------------------------------------------------


def test_build_level3_submission(tmp_path: Path) -> None:
    """Verify build_level3_submission adheres to Kaggle submission contract."""
    row_ids = [0, 1, 2, 3, 4]
    s_2a = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
    s_2b = np.array([0.2, 0.3, 0.5, 0.7, 0.85])
    weight = 0.6

    sub_df = build_level3_submission(
        row_ids=row_ids,
        s_2a=s_2a,
        s_2b=s_2b,
        weight=weight,
    )

    validate_submission_contract(sub_df, expected_row_count=5, expected_row_ids=row_ids)
    assert sub_df.columns == ["row_id", "anomaly"]
    assert sub_df.height == 5
    expected_preds = weight * s_2a + (1.0 - weight) * s_2b
    np.testing.assert_allclose(sub_df["anomaly"].to_numpy(), expected_preds)


def test_fail_closed_meta_2a_mismatch_in_full_mode() -> None:
    """Finding 2: Verify run_level3_nested_cv raises ValueError in full mode on Meta-2A mismatch."""
    rng = np.random.default_rng(42)
    buildings = [101, 102, 103, 104]
    rows: list[dict[str, Any]] = []
    row_idx = 0
    for b in buildings:
        for t in range(10):
            rows.append({
                "row_index": row_idx,
                "building_id": b,
                "timestamp": np.datetime64("2016-01-01T00:00:00", "ns") + np.timedelta64(t, "h"),
                "feat1": float(rng.uniform(10.0, 100.0)),
                "anomaly": int(rng.integers(0, 2)),
            })
            row_idx += 1
    train_dev = pl.DataFrame(rows)
    dev_splits = pl.DataFrame({
        "repeat": [0, 0, 0, 0],
        "fold": [0, 0, 1, 1],
        "building_id": [101, 102, 103, 104],
        "split": ["dev", "dev", "dev", "dev"],
    })

    # Incomplete Meta-2A DataFrame (only 5 rows instead of 40)
    meta_2a_incomplete = pl.DataFrame({
        "repeat": [0] * 5,
        "fold": [0] * 5,
        "row_index": list(range(5)),
        "prediction": [0.5] * 5,
        "building_id": ["101"] * 5,
        "timestamp": [rows[0]["timestamp"]] * 5,
        "target": [0] * 5,
    })

    with pytest.raises(ValueError, match="Meta-2A OOF.*mismatch"):
        run_level3_nested_cv(
            train_dev=train_dev,
            dev_splits=dev_splits,
            meta_2a_oof_df=meta_2a_incomplete,
            feature_cols=["feat1"],
            target_col="anomaly",
            repeats=[0],
            folds=[0, 1],
            execution_mode="full",
            dry_run=False,
            smoke=False,
        )


def test_build_level3_submission_mismatched_lengths() -> None:
    """Finding 3 helper: Verify build_level3_submission raises ValueError on length mismatch."""
    row_ids = [0, 1, 2]
    s_2a = [0.1, 0.2]  # length 2 != 3
    s_2b = [0.3, 0.4, 0.5]
    with pytest.raises(ValueError, match="Array length mismatch"):
        build_level3_submission(row_ids=row_ids, s_2a=s_2a, s_2b=s_2b, weight=0.5)


def test_independent_expected_rows_construction() -> None:
    """Finding 4: Verify independent construction of expected_rows by joining train_dev with dev_splits."""
    rng = np.random.default_rng(42)
    buildings = [101, 102, 103, 104]
    rows: list[dict[str, Any]] = []
    row_idx = 0
    for b in buildings:
        for t in range(5):
            rows.append({
                "row_index": row_idx,
                "building_id": str(b),
                "timestamp": np.datetime64("2016-01-01T00:00:00", "ns") + np.timedelta64(t, "h"),
                "feat1": float(rng.uniform(1.0, 10.0)),
                "anomaly": int(rng.integers(0, 2)),
            })
            row_idx += 1
    train_dev = pl.DataFrame(rows)
    dev_splits = pl.DataFrame({
        "repeat": [0, 0, 0, 0],
        "fold": [0, 0, 1, 1],
        "building_id": ["101", "102", "103", "104"],
        "split": ["dev", "dev", "dev", "dev"],
    })

    repeats = [0]
    folds = [0, 1]
    split_seeds = {"0": 42}
    split_sha = "a" * 64
    data_sha = "b" * 64
    config_sha = "c" * 64
    run_id = "test_run_expected"

    active_splits = dev_splits.filter(
        pl.col("repeat").is_in(repeats) & pl.col("fold").is_in(folds)
    ).select([
        pl.col("building_id").cast(pl.String),
        pl.col("repeat").cast(pl.Int64),
        pl.col("fold").cast(pl.Int64),
    ])

    expected_base = (
        train_dev.select([
            pl.col("row_index").cast(pl.Int64),
            pl.col("building_id").cast(pl.String),
            pl.col("timestamp"),
            pl.col("anomaly").cast(pl.Int64).alias("target"),
        ])
        .join(active_splits, on="building_id", how="inner")
    )

    seed_map = {int(r): int(split_seeds[str(r)]) for r in repeats}
    seed_expr = pl.col("repeat").replace(seed_map).cast(pl.Int64)

    expected_rows = (
        expected_base.with_columns([
            seed_expr.alias("split_seed"),
            pl.lit(split_sha).alias("split_sha256"),
            pl.lit(data_sha).alias("data_sha256"),
            pl.lit(config_sha).alias("config_sha256"),
            pl.lit(run_id).alias("run_id"),
        ])
        .select(list(ROW_COLUMNS))
        .sort(["repeat", "fold", "building_id", "timestamp"])
    )

    assert expected_rows.columns == list(ROW_COLUMNS)
    assert expected_rows.height == len(rows)
    assert expected_rows["row_index"].null_count() == 0
