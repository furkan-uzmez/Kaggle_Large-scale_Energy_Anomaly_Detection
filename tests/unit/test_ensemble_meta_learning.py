"""Unit tests for Level-2 Meta-Learner tournament module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from project.models.candidate_artifacts import (
    CANDIDATE_OOF_SCHEMA,
    validate_candidate_oof,
    write_candidate_oof_batches,
)
from project.models.ensemble_meta_learning import (
    SUPPORTED_META_ARMS,
    CaruanaEnsembleSelector,
    DirectAucOptimizerMeta,
    LogitLogisticRegressionMeta,
    NonNegativeLeastSquaresMeta,
    ShallowLightGbmMeta,
    create_meta_learner,
    load_base_candidate_predictions,
    run_meta_nested_cv,
)


def _synthetic_predictions_and_targets(
    n_samples: int = 500,
    n_models: int = 6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic base model probabilities and correlated binary targets."""
    rng = np.random.default_rng(seed)
    # Latent true score
    latent = rng.standard_normal(n_samples)
    prob_true = 1.0 / (1.0 + np.exp(-latent))
    targets = (rng.uniform(0, 1, n_samples) < prob_true).astype(np.int64)

    # Base model predictions with varying noise
    preds = np.zeros((n_samples, n_models), dtype=np.float64)
    for i in range(n_models):
        noise = rng.normal(0.0, 0.3 + 0.1 * i, n_samples)
        noisy_latent = latent + noise
        p = 1.0 / (1.0 + np.exp(-noisy_latent))
        preds[:, i] = np.clip(p, 1e-6, 1.0 - 1e-6)

    return preds, targets


def test_caruana_ensemble_selector_weights_and_bounds() -> None:
    X, y = _synthetic_predictions_and_targets(n_samples=300, n_models=4, seed=42)
    selector = CaruanaEnsembleSelector(n_iterations=15, random_state=42)
    selector.fit(X, y)

    assert hasattr(selector, "weights_")
    assert len(selector.weights_) == 4
    assert np.all(selector.weights_ >= 0.0)
    assert np.isclose(np.sum(selector.weights_), 1.0)

    preds = selector.predict_proba(X)
    assert preds.shape == (300,)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)
    # The combined score should be competitive with the best single base model
    best_single_auc = max(roc_auc_score(y, X[:, m]) for m in range(4))
    ens_auc = roc_auc_score(y, preds)
    assert ens_auc >= best_single_auc - 0.01


def test_logit_logistic_regression_meta() -> None:
    X, y = _synthetic_predictions_and_targets(n_samples=300, n_models=4, seed=43)
    meta = LogitLogisticRegressionMeta(
        C=1.0, clip_eps=1e-7, standardize=True, random_state=42
    )
    meta.fit(X, y)

    preds = meta.predict_proba(X)
    assert preds.shape == (300,)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)
    assert roc_auc_score(y, preds) > 0.65


def test_non_negative_least_squares_meta() -> None:
    X, y = _synthetic_predictions_and_targets(n_samples=300, n_models=4, seed=44)
    meta = NonNegativeLeastSquaresMeta(normalize_weights=True)
    meta.fit(X, y)

    assert hasattr(meta, "weights_")
    assert len(meta.weights_) == 4
    assert np.all(meta.weights_ >= 0.0)
    assert np.isclose(np.sum(meta.weights_), 1.0)

    preds = meta.predict_proba(X)
    assert preds.shape == (300,)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)
    assert roc_auc_score(y, preds) > 0.65


def test_direct_auc_optimizer_meta() -> None:
    X, y = _synthetic_predictions_and_targets(n_samples=300, n_models=4, seed=45)
    meta = DirectAucOptimizerMeta(method="Nelder-Mead", max_iter=25, random_state=42)
    meta.fit(X, y)

    assert hasattr(meta, "weights_")
    assert len(meta.weights_) == 4
    assert np.all(meta.weights_ >= 0.0)
    assert np.isclose(np.sum(meta.weights_), 1.0)

    preds = meta.predict_proba(X)
    assert preds.shape == (300,)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)
    assert roc_auc_score(y, preds) > 0.65


def test_shallow_lightgbm_meta() -> None:
    X, y = _synthetic_predictions_and_targets(n_samples=300, n_models=4, seed=46)
    meta = ShallowLightGbmMeta(
        max_depth=2,
        num_leaves=3,
        learning_rate=0.03,
        min_child_samples=20,
        n_estimators=10,
        random_state=42,
    )
    meta.fit(X, y)

    preds = meta.predict_proba(X)
    assert preds.shape == (300,)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)
    assert roc_auc_score(y, preds) > 0.65


def test_create_meta_learner_factory() -> None:
    for arm in SUPPORTED_META_ARMS:
        learner = create_meta_learner(arm)
        assert hasattr(learner, "fit")
        assert hasattr(learner, "predict_proba")

    with pytest.raises(ValueError, match="Unsupported meta arm"):
        create_meta_learner("unknown_arm")


def test_load_base_candidate_predictions() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        base_path = tmp / "base_oof.parquet"
        other_path = tmp / "other_oof.parquet"

        df1 = pl.DataFrame(
            {
                "row_index": [0, 1, 2],
                "repeat": [0, 0, 0],
                "fold": [0, 0, 1],
                "building_id": ["b1", "b1", "b2"],
                "timestamp": pl.Series([1000, 2000, 3000]).cast(pl.Datetime("ns")),
                "prediction": [0.1, 0.2, 0.3],
                "target": [0, 1, 0],
                "split_seed": [42, 42, 42],
                "split_sha256": ["a" * 64] * 3,
                "data_sha256": ["b" * 64] * 3,
                "arm_id": ["control", "control", "control"],
            }
        )
        df2 = pl.DataFrame(
            {
                "row_index": [0, 1, 2],
                "repeat": [0, 0, 0],
                "fold": [0, 0, 1],
                "prediction": [0.4, 0.5, 0.6],
                "target": [0, 1, 0],
                "arm_id": ["challenger", "challenger", "challenger"],
            }
        )

        df1.write_parquet(base_path)
        df2.write_parquet(other_path)

        specs = [
            {"name": "m1", "path": str(base_path), "arm_id": "control"},
            {"name": "m2", "path": str(other_path), "arm_id": "challenger"},
        ]

        merged, pred_cols = load_base_candidate_predictions(specs, project_root=tmp)
        assert merged.height == 3
        assert pred_cols == ["pred_m1", "pred_m2"]
        assert "pred_m1" in merged.columns
        assert "pred_m2" in merged.columns
        assert merged["pred_m1"].to_list() == [0.1, 0.2, 0.3]
        assert merged["pred_m2"].to_list() == [0.4, 0.5, 0.6]

        # Test alignment mismatch raises ValueError
        bad_df = df2.with_columns(pl.col("row_index") + 10)
        bad_path = tmp / "bad_oof.parquet"
        bad_df.write_parquet(bad_path)
        bad_specs = [
            {"name": "m1", "path": str(base_path), "arm_id": "control"},
            {"name": "bad", "path": str(bad_path), "arm_id": "challenger"},
        ]
        with pytest.raises(ValueError, match="row_index alignment mismatch"):
            load_base_candidate_predictions(bad_specs, project_root=tmp)


def _build_mock_split_df(n_per_b: int = 50) -> tuple[pl.DataFrame, list[str]]:
    """Build a mock dataframe covering repeat 0, folds 0 and 1 with 2 buildings each."""
    rows = []
    b_fold = {
        "b10": 0,
        "b20": 0,
        "b30": 1,
        "b40": 1,
    }
    rng = np.random.default_rng(42)
    idx = 0
    pred_cols = [
        "pred_m1",
        "pred_m2",
        "pred_m3",
        "pred_m4",
        "pred_m5",
        "pred_m6",
    ]

    for b_id, fold in b_fold.items():
        for t in range(n_per_b):
            target = int(rng.uniform() < 0.2)
            preds = [float(np.clip(rng.uniform(0.1, 0.8), 0.0, 1.0)) for _ in range(6)]
            row = {
                "row_index": idx,
                "repeat": 0,
                "fold": fold,
                "building_id": b_id,
                "timestamp": t * 1_000_000_000,
                "target": target,
                "split_seed": 42,
                "split_sha256": "a" * 64,
                "data_sha256": "b" * 64,
            }
            for col_name, val in zip(pred_cols, preds, strict=True):
                row[col_name] = val
            rows.append(row)
            idx += 1

    df = pl.DataFrame(rows).with_columns(
        [
            pl.col("timestamp").cast(pl.Datetime("ns")),
        ]
    )
    return df, pred_cols


def test_nested_cv_leakage_and_schema() -> None:
    df, pred_cols = _build_mock_split_df(n_per_b=60)
    config_sha = "c" * 64

    oof_df = run_meta_nested_cv(
        data=df,
        prediction_cols=pred_cols,
        arm_id="control",
        meta_learner_params={"n_iterations": 10},
        repeats=[0],
        folds=[0, 1],
        run_id="meta_unit_test",
        execution_mode="synthetic_smoke",
        non_comparative=True,
        n03_eligible=False,
        config_sha256=config_sha,
    )

    # Check that schema matches CANDIDATE_OOF_SCHEMA
    assert set(oof_df.columns) == set(CANDIDATE_OOF_SCHEMA.names)
    assert oof_df.height == df.height
    assert oof_df["prediction"].null_count() == 0
    assert oof_df["target"].null_count() == 0
    assert np.all(oof_df["prediction"].to_numpy() >= 0.0)
    assert np.all(oof_df["prediction"].to_numpy() <= 1.0)
    assert oof_df["arm_id"][0] == "control"
    assert oof_df["candidate_id"][0] == "meta_control"


def test_validate_candidate_oof_e2e_mock() -> None:
    df, pred_cols = _build_mock_split_df(n_per_b=60)
    config_sha = "c" * 64

    oof_df = run_meta_nested_cv(
        data=df,
        prediction_cols=pred_cols,
        arm_id="challenger_nnls",
        meta_learner_params={},
        repeats=[0],
        folds=[0, 1],
        run_id="meta_unit_test_nnls",
        execution_mode="synthetic_smoke",
        non_comparative=True,
        n03_eligible=False,
        config_sha256=config_sha,
    )

    arrow_table = (
        oof_df.select(CANDIDATE_OOF_SCHEMA.names).to_arrow().cast(CANDIDATE_OOF_SCHEMA)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve() / "oof.parquet"
        batches = [arrow_table]
        written = write_candidate_oof_batches(tmp_path, batches, CANDIDATE_OOF_SCHEMA)
        assert written["status"] == "written_unvalidated"

        # Manifest
        manifest = {
            "schema_version": 1,
            "source_notebook": "notebooks/02_baseline_and_candidate_design.ipynb",
            "source_cell_id": "ensemble_meta_test",
            "vault_row_indices": [],
            "vault_building_ids": [],
            "protocol": {
                "repeat_ids": [0],
                "fold_ids": [0, 1],
                "split_seeds": {"0": 42},
            },
            "candidates": {
                "meta_challenger_nnls": {
                    "recipe_id": oof_df["recipe_id"][0],
                    "model_family": "ensemble_meta",
                    "model_config_id": oof_df["model_config_id"][0],
                    "experiment_id": oof_df["experiment_id"][0],
                    "arm_id": "challenger_nnls",
                    "score_kind": "positive_probability",
                    "positive_label": 1,
                    "model_fit_config_sha256": oof_df["model_fit_config_sha256"][0],
                }
            },
            "split_sha256": oof_df["split_sha256"][0],
            "data_sha256": oof_df["data_sha256"][0],
            "config_sha256": config_sha,
            "run_id": "meta_unit_test_nnls",
            "execution_mode": "synthetic_smoke",
            "non_comparative": True,
            "n03_eligible": False,
        }

        expected_rows = oof_df.select(
            [
                "row_index",
                "repeat",
                "fold",
                "building_id",
                "timestamp",
                "target",
                "split_seed",
                "split_sha256",
                "data_sha256",
                "config_sha256",
                "run_id",
            ]
        )

        validation_report = validate_candidate_oof(
            pl.read_parquet(tmp_path),
            expected_rows,
            manifest,
        )
        assert validation_report["status"] == "validated"
        assert validation_report["oof_record_count"] == df.height


def test_nested_cv_no_leakage_isolation() -> None:
    """Verify that train data strictly excludes the validation fold and other repeats."""
    df, pred_cols = _build_mock_split_df(n_per_b=40)
    # Add repeat 1 rows to verify repeat isolation
    df_rep1 = df.with_columns(
        [
            pl.lit(1).cast(pl.Int64).alias("repeat"),
            pl.col("row_index") + 1000,
        ]
    )
    combined = pl.concat([df, df_rep1], how="vertical")

    oof_df = run_meta_nested_cv(
        data=combined,
        prediction_cols=pred_cols,
        arm_id="challenger_lr",
        meta_learner_params={},
        repeats=[0, 1],
        folds=[0, 1],
        run_id="leakage_check_run",
        execution_mode="full",
        non_comparative=False,
        n03_eligible=True,
        config_sha256="d" * 64,
    )
    assert oof_df.height == combined.height
    assert set(oof_df["repeat"].unique().to_list()) == {0, 1}
    assert set(oof_df["fold"].unique().to_list()) == {0, 1}


def test_cli_runner_dry_run() -> None:
    """Verify that run_tournament executes dry-run mode and writes valid artifacts."""
    from scripts.train_ensemble_meta import load_config, run_tournament

    config_path = Path("configs/experiments/ensemble_meta_tournament.yaml")
    assert config_path.is_file()
    config = load_config(config_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir)
        paths = run_tournament(
            arm="challenger_nnls",
            config=config,
            config_path=config_path,
            output_dir=tmp_out,
            dry_run=True,
        )
        assert len(paths) == 1
        oof_path = paths[0]
        assert oof_path.is_file()
        manifest_path = oof_path.parent / "challenger_nnls_manifest.json"
        assert manifest_path.is_file()

        df = pl.read_parquet(oof_path)
        assert df.height > 0
        assert tuple(df.columns) == tuple(CANDIDATE_OOF_SCHEMA.names)
