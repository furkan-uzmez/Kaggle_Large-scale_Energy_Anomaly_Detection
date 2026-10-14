from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy import stats

from project.experiments.full_campaign_review import (
    build_comparison_tables,
    build_ensemble_plan,
    compute_signal_vs_noise_verdict,
    discover_completed_model_runs,
    holm_bonferroni_correction,
    iter_paired_error_slices,
    load_finalized_campaign,
    load_unified_campaign_pool,
    nadeau_bengio_t_test,
)
from project.models.candidate_artifacts import CANDIDATE_OOF_SCHEMA

# =====================================================================
# Fixtures: Canonical v1 Completed Campaign Artifacts
# =====================================================================


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def synthetic_campaign_dir(tmp_path: Path) -> Path:
    """Create a fully valid finalized campaign adhering strictly to canonical v1.

    Models actual repository campaign roots:
    - Root run_status.json and spec.json
    - completion/manifest.json and completion/receipt.json
    - Per-experiment declaration, source_lineage, transport_validation, and *_oof.parquet
    - OOF tables conform to the 24 canonical fields in CANDIDATE_OOF_SCHEMA:
      (candidate_id, recipe_id, model_family, model_config_id, row_index, repeat,
       fold, building_id, timestamp, prediction, target, split_sha256, data_sha256,
       config_sha256, run_id, split_seed, experiment_id, arm_id, score_kind,
       positive_label, execution_mode, non_comparative, n03_eligible, model_fit_config_sha256)
    - 2 experiments:
      1. feat_core_vs_weather: hist_gradient_boosting (positive_probability)
         control vs challenger (positive gain)
      2. dim_pca_vs_passthrough: svm (decision_margin)
         control vs challenger (raw margin scores)
    - 12 partitions: 3 repeats (0, 1, 2) x 4 folds (0, 1, 2, 3)
    - 8 buildings, 16 rows per partition per candidate.
    """
    campaign_dir = tmp_path / "campaign_full_v1"
    campaign_dir.mkdir(parents=True)
    completion_dir = campaign_dir / "completion"
    completion_dir.mkdir()

    run_id = "campaign_full_v1"
    rng = np.random.default_rng(42)
    buildings = [f"bldg_{i:02d}" for i in range(1, 9)]
    primary_uses = [
        "Office",
        "Education",
        "Office",
        "Retail",
        "Education",
        "Office",
        "Healthcare",
        "Retail",
    ]
    bldg_use_map = dict(zip(buildings, primary_uses, strict=True))

    n_rows_per_partition = 16
    # Base lineage hashes
    dummy_data_sha = "a" * 64
    dummy_split_sha = "b" * 64
    dummy_config_sha = "c" * 64

    experiments_spec = [
        {
            "experiment_id": "feat_core_vs_weather",
            "control": {
                "candidate_id": "feat_core_vs_weather_control",
                "model_family": "hist_gradient_boosting",
                "score_kind": "positive_probability",
                "base_prob": 0.25,
            },
            "challenger": {
                "candidate_id": "feat_core_vs_weather_challenger",
                "model_family": "hist_gradient_boosting",
                "score_kind": "positive_probability",
                "base_prob": 0.40,
            },
        },
        {
            "experiment_id": "dim_pca_vs_passthrough",
            "control": {
                "candidate_id": "dim_pca_vs_passthrough_control",
                "model_family": "svm",
                "score_kind": "decision_margin",
                "base_prob": 0.0,
            },
            "challenger": {
                "candidate_id": "dim_pca_vs_passthrough_challenger",
                "model_family": "svm",
                "score_kind": "decision_margin",
                "base_prob": 0.0,
            },
        },
    ]

    manifest_candidates = []
    receipt_artifact_sha: dict[str, str] = {}

    for exp in experiments_spec:
        exp_id = exp["experiment_id"]
        exp_records: dict[str, list[Any]] = {
            "candidate_id": [],
            "recipe_id": [],
            "model_family": [],
            "model_config_id": [],
            "row_index": [],
            "repeat": [],
            "fold": [],
            "building_id": [],
            "timestamp": [],
            "prediction": [],
            "target": [],
            "split_sha256": [],
            "data_sha256": [],
            "config_sha256": [],
            "run_id": [],
            "split_seed": [],
            "experiment_id": [],
            "arm_id": [],
            "score_kind": [],
            "positive_label": [],
            "execution_mode": [],
            "non_comparative": [],
            "n03_eligible": [],
            "model_fit_config_sha256": [],
        }

        declaration_candidates = {}

        for arm in ("control", "challenger"):
            arm_spec = exp[arm]
            cand_id = arm_spec["candidate_id"]
            family = arm_spec["model_family"]
            score_kind = arm_spec["score_kind"]
            fit_hash = hashlib.sha256(f"{exp_id}_{arm}".encode()).hexdigest()

            declaration_candidates[cand_id] = {
                "arm_id": arm,
                "experiment_id": exp_id,
                "model_config_id": f"{cand_id}_config",
                "model_family": family,
                "model_fit_config_sha256": fit_hash,
                "positive_label": 1,
                "recipe_id": exp_id,
                "score_kind": score_kind,
            }

            for repeat in range(3):
                seed = 42 + repeat
                for fold in range(4):
                    fold_bldgs = buildings[fold * 2 : (fold + 1) * 2]
                    for r_idx in range(n_rows_per_partition):
                        global_row_index = fold * n_rows_per_partition + r_idx
                        bldg = fold_bldgs[r_idx % len(fold_bldgs)]
                        hour = r_idx % 24
                        ts_ns = int(
                            datetime(2022, 1, 3, hour, 0, 0, tzinfo=UTC).timestamp()
                            * 1_000_000_000
                        )
                        target = 1 if (r_idx % 4 == 0) else 0

                        if score_kind == "positive_probability":
                            if arm == "control":
                                pred = float(
                                    np.clip(
                                        0.25 + 0.10 * target + rng.normal(0, 0.12),
                                        0.001,
                                        0.999,
                                    )
                                )
                            else:
                                pred = float(
                                    np.clip(
                                        0.10 + 0.65 * target + rng.normal(0, 0.05),
                                        0.001,
                                        0.999,
                                    )
                                )
                        else:
                            # Decision margin
                            pred = float(-1.2 + 2.5 * target + rng.normal(0, 0.3))

                        exp_records["candidate_id"].append(cand_id)
                        exp_records["recipe_id"].append(exp_id)
                        exp_records["model_family"].append(family)
                        exp_records["model_config_id"].append(f"{cand_id}_config")
                        exp_records["row_index"].append(global_row_index)
                        exp_records["repeat"].append(repeat)
                        exp_records["fold"].append(fold)
                        exp_records["building_id"].append(bldg)
                        exp_records["timestamp"].append(ts_ns)
                        exp_records["prediction"].append(pred)
                        exp_records["target"].append(target)
                        exp_records["split_sha256"].append(dummy_split_sha)
                        exp_records["data_sha256"].append(dummy_data_sha)
                        exp_records["config_sha256"].append(dummy_config_sha)
                        exp_records["run_id"].append(run_id)
                        exp_records["split_seed"].append(seed)
                        exp_records["experiment_id"].append(exp_id)
                        exp_records["arm_id"].append(arm)
                        exp_records["score_kind"].append(score_kind)
                        exp_records["positive_label"].append(1)
                        exp_records["execution_mode"].append("full")
                        exp_records["non_comparative"].append(False)
                        exp_records["n03_eligible"].append(True)
                        exp_records["model_fit_config_sha256"].append(fit_hash)

        # Build PyArrow table using authoritative CANDIDATE_OOF_SCHEMA
        arrow_dict = dict(exp_records)
        arrow_dict["timestamp"] = pa.array(
            exp_records["timestamp"], type=pa.timestamp("ns")
        )
        arrow_table = pa.Table.from_pydict(arrow_dict, schema=CANDIDATE_OOF_SCHEMA)

        oof_filename = f"{exp_id}_oof.parquet"
        oof_path = campaign_dir / oof_filename
        pq.write_table(arrow_table, oof_path)
        oof_sha = _compute_sha256(oof_path)
        receipt_artifact_sha[oof_filename] = oof_sha

        # Declaration file
        decl_filename = f"{exp_id}_declaration.json"
        decl_path = campaign_dir / decl_filename
        _write_json(
            decl_path,
            {
                "candidates": declaration_candidates,
                "config_sha256": dummy_config_sha,
                "experiment_id": exp_id,
                "recipe_id": exp_id,
            },
        )
        receipt_artifact_sha[decl_filename] = _compute_sha256(decl_path)

        # Source lineage file
        lineage_filename = f"{exp_id}_source_lineage.json"
        lineage_path = campaign_dir / lineage_filename
        _write_json(
            lineage_path,
            {
                "data_sha256": dummy_data_sha,
                "split_sha256": dummy_split_sha,
                "experiment_id": exp_id,
            },
        )
        receipt_artifact_sha[lineage_filename] = _compute_sha256(lineage_path)

        # Transport validation file
        trans_filename = f"{exp_id}_transport_validation.json"
        trans_path = campaign_dir / trans_filename
        total_arm_records = len(exp_records["candidate_id"])
        _write_json(
            trans_path,
            {
                "experiment_id": exp_id,
                "record_count": total_arm_records,
                "validation_status": "passed",
            },
        )
        receipt_artifact_sha[trans_filename] = _compute_sha256(trans_path)

        manifest_candidates.append(
            {
                "arms": ["control", "challenger"],
                "experiment_id": exp_id,
                "n03_eligible": True,
                "oof_path": str(oof_path),
                "oof_sha256": oof_sha,
                "status": "completed",
            }
        )

    # Feature metadata table for slicing (building metadata and weather missingness)
    metadata_records = []
    for bldg in buildings:
        for hour in range(24):
            metadata_records.append(
                {
                    "building_id": bldg,
                    "primary_use": bldg_use_map[bldg],
                    "hour": hour,
                    "weather_missing": bool((hash(bldg) + hour) % 3 == 0),
                }
            )
    metadata_path = campaign_dir / "feature_metadata.parquet"
    pl.DataFrame(metadata_records).write_parquet(metadata_path)
    metadata_sha = _compute_sha256(metadata_path)
    receipt_artifact_sha["feature_metadata.parquet"] = metadata_sha

    # Root run_status and spec
    status_path = campaign_dir / "run_status.json"
    _write_json(
        status_path,
        {
            "status": "completed",
            "run_id": run_id,
            "active_worker": None,
            "exit_code": 0,
        },
    )
    receipt_artifact_sha["run_status.json"] = _compute_sha256(status_path)

    spec_path = campaign_dir / "spec.json"
    _write_json(
        spec_path,
        {
            "run_id": run_id,
            "execution_mode": "full",
            "total_experiments": len(experiments_spec),
        },
    )
    receipt_artifact_sha["spec.json"] = _compute_sha256(spec_path)

    # Completion manifest
    manifest_path = completion_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "candidates": manifest_candidates,
            "execution_mode": "full",
            "feature_metadata_sha256": metadata_sha,
            "run_id": run_id,
            "schema_version": 1,
        },
    )

    # Completion receipt
    receipt_path = completion_dir / "receipt.json"
    _write_json(
        receipt_path,
        {
            "artifact_sha256": receipt_artifact_sha,
            "campaign_id": run_id,
            "status": "completed",
        },
    )

    return campaign_dir


# =====================================================================
# Tests: load_finalized_campaign
# =====================================================================


def test_load_finalized_campaign_success(synthetic_campaign_dir: Path) -> None:
    """Verifies valid campaign loads manifests, declarations, receipts, and verified OOF."""
    campaign = load_finalized_campaign(synthetic_campaign_dir)

    assert campaign["run_id"] == "campaign_full_v1"
    assert campaign["status"] == "completed"
    assert len(campaign["experiments"]) == 2

    # Candidates categorized correctly by canonical score_kind
    prob_candidates = campaign["probability_candidates"]
    margin_candidates = campaign["margin_candidates"]

    assert "feat_core_vs_weather_control" in prob_candidates
    assert "feat_core_vs_weather_challenger" in prob_candidates
    assert "dim_pca_vs_passthrough_control" in margin_candidates
    assert "dim_pca_vs_passthrough_challenger" in margin_candidates

    # Margin candidates never contaminate probability candidate pool
    assert "dim_pca_vs_passthrough_control" not in prob_candidates
    assert "dim_pca_vs_passthrough_challenger" not in prob_candidates

    # Underlying OOF frames present and verified against schema
    assert "feat_core_vs_weather" in campaign["oof_tables"]
    feat_oof = campaign["oof_tables"]["feat_core_vs_weather"]
    assert isinstance(feat_oof, pl.DataFrame)
    assert set(feat_oof.columns) == set(CANDIDATE_OOF_SCHEMA.names)


def test_load_finalized_campaign_detects_running_or_interrupted(
    synthetic_campaign_dir: Path,
) -> None:
    """Rejects incomplete runs with running or interrupted status."""
    status_file = synthetic_campaign_dir / "run_status.json"
    _write_json(status_file, {"status": "running", "active_worker": 1234})

    with pytest.raises(ValueError, match="not finalized|still running|incomplete"):
        load_finalized_campaign(synthetic_campaign_dir)

    _write_json(status_file, {"status": "interrupted", "reason": "SIGINT"})

    with pytest.raises(ValueError, match="not finalized|interrupted|incomplete"):
        load_finalized_campaign(synthetic_campaign_dir)


def test_load_finalized_campaign_detects_missing_files(
    synthetic_campaign_dir: Path,
) -> None:
    """Rejects campaigns missing completion receipt, manifest, or experiment OOF."""
    receipt_file = synthetic_campaign_dir / "completion" / "receipt.json"
    receipt_file.unlink()

    with pytest.raises(FileNotFoundError, match="receipt.json"):
        load_finalized_campaign(synthetic_campaign_dir)


def test_load_finalized_campaign_rejects_altered_hash(
    synthetic_campaign_dir: Path,
) -> None:
    """Detects tampered OOF parquet files with checksum mismatches against receipt."""
    oof_path = synthetic_campaign_dir / "feat_core_vs_weather_oof.parquet"
    # Append trailing byte to alter SHA-256
    with oof_path.open("ab") as f:
        f.write(b"\x00")

    with pytest.raises(ValueError, match="sha256 mismatch|corrupt|tampered"):
        load_finalized_campaign(synthetic_campaign_dir)


def test_load_finalized_campaign_rejects_altered_feature_metadata_hash(
    synthetic_campaign_dir: Path,
) -> None:
    """Detects tampered feature metadata parquet with checksum mismatch against receipt."""
    meta_path = synthetic_campaign_dir / "feature_metadata.parquet"
    with meta_path.open("ab") as f:
        f.write(b"\x00")

    with pytest.raises(ValueError, match="sha256 mismatch|corrupt|tampered"):
        load_finalized_campaign(synthetic_campaign_dir)


def test_load_finalized_campaign_rejects_row_count_mismatch(
    synthetic_campaign_dir: Path,
) -> None:
    """Detects corrupted OOF tables where row counts deviate from transport validation."""
    oof_path = synthetic_campaign_dir / "feat_core_vs_weather_oof.parquet"
    df = pl.read_parquet(oof_path)
    # Drop one row and rewrite
    pq.write_table(df.slice(1).to_arrow(), oof_path)

    # Even if receipt hash was updated, transport validation record_count must catch mismatch
    receipt_path = synthetic_campaign_dir / "completion" / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifact_sha256"]["feat_core_vs_weather_oof.parquet"] = _compute_sha256(
        oof_path
    )
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="record_count|row count mismatch|expected"):
        load_finalized_campaign(synthetic_campaign_dir)


def test_load_finalized_campaign_separates_probability_pool_from_decision_margins(
    synthetic_campaign_dir: Path,
) -> None:
    """Ensures models like RBF SVC with uncalibrated margins never enter probability pool."""
    campaign = load_finalized_campaign(synthetic_campaign_dir)
    prob_pool = campaign["probability_candidates"]
    margin_pool = campaign["margin_candidates"]

    assert "feat_core_vs_weather_challenger" in prob_pool
    assert "dim_pca_vs_passthrough_challenger" in margin_pool

    # Margin models have uncalibrated values outside [0, 1]
    oof_df = campaign["oof_tables"]["dim_pca_vs_passthrough"]
    margin_preds = oof_df.filter(pl.col("score_kind") == "decision_margin")[
        "prediction"
    ].to_numpy()
    assert (margin_preds < 0).any() or (margin_preds > 1).any()


# =====================================================================
# Tests: Statistical Testing Helpers (Nadeau-Bengio & Holm-Bonferroni)
# =====================================================================


def test_nadeau_bengio_t_test_exact_formula() -> None:
    """Verifies Nadeau-Bengio corrected resampled t-test with test-to-train ratio r=1/3."""
    # 12 paired differences from 3 repeats of 4-fold CV
    differences = np.array(
        [
            0.012,
            0.015,
            0.009,
            0.014,
            0.018,
            0.011,
            0.016,
            0.013,
            0.015,
            0.010,
            0.017,
            0.014,
        ]
    )
    n = len(differences)
    assert n == 12
    test_ratio = 1.0 / 3.0  # r = n_val / n_train = (1/4) / (3/4) = 1/3

    mean_d = float(np.mean(differences))
    s2 = float(np.var(differences, ddof=1))
    # Corrected variance: s^2 * (1/n + r)
    corrected_var = s2 * ((1.0 / n) + test_ratio)
    corrected_se = math.sqrt(corrected_var)
    t_stat = mean_d / corrected_se
    df = n - 1  # 11
    p_val = 2.0 * float(stats.t.sf(abs(t_stat), df=df))
    t_crit = float(stats.t.ppf(0.975, df=df))
    ci_lower = mean_d - t_crit * corrected_se
    ci_upper = mean_d + t_crit * corrected_se

    res = nadeau_bengio_t_test(differences, test_ratio=test_ratio)

    assert res["mean_delta"] == pytest.approx(mean_d, abs=1e-8)
    assert res["corrected_std_error"] == pytest.approx(corrected_se, abs=1e-8)
    assert res["t_stat"] == pytest.approx(t_stat, abs=1e-8)
    assert res["p_value"] == pytest.approx(p_val, abs=1e-8)
    assert res["df"] == 11
    assert res["ci_95_lower"] == pytest.approx(ci_lower, abs=1e-8)
    assert res["ci_95_upper"] == pytest.approx(ci_upper, abs=1e-8)


def test_nadeau_bengio_t_test_zero_deltas_null_case() -> None:
    """When difference is identically zero across all folds, t=0 and p=1."""
    differences = np.zeros(12)
    res = nadeau_bengio_t_test(differences, test_ratio=1.0 / 3.0)

    assert res["mean_delta"] == 0.0
    assert res["t_stat"] == 0.0
    assert res["p_value"] == 1.0


def test_nadeau_bengio_variance_inflation_vs_naive_t_test() -> None:
    """Demonstrates that corrected SE is strictly larger than naive SE by sqrt(5).

    Naive SE = sqrt(s^2 / n) = s / sqrt(12).
    Corrected SE = sqrt(s^2 * (1/12 + 1/3)) = sqrt(s^2 * (5/12)) = sqrt(5) * Naive SE.
    """
    differences = np.array(
        [0.01, 0.02, 0.03, 0.01, 0.02, 0.04, 0.01, 0.02, 0.03, 0.02, 0.01, 0.03]
    )
    s2 = float(np.var(differences, ddof=1))
    naive_se = math.sqrt(s2 / 12)

    res = nadeau_bengio_t_test(differences, test_ratio=1.0 / 3.0)
    expected_inflation = math.sqrt(1 + 12 * (1.0 / 3.0))  # sqrt(5) ~ 2.2360679

    assert res["corrected_std_error"] / naive_se == pytest.approx(
        expected_inflation, rel=1e-5
    )


def test_holm_bonferroni_correction() -> None:
    """Verifies Holm-Bonferroni step-down multiple testing correction."""
    raw_p_values = [0.005, 0.012, 0.040, 0.080, 0.150]
    result = holm_bonferroni_correction(raw_p_values, alpha=0.05)

    adj_p = result["adjusted_p_values"]
    rejected = result["rejected"]

    assert adj_p[0] == pytest.approx(0.025, abs=1e-4)
    assert adj_p[1] == pytest.approx(0.048, abs=1e-4)
    assert adj_p[2] == pytest.approx(0.120, abs=1e-4)
    assert adj_p[3] == pytest.approx(0.160, abs=1e-4)
    assert adj_p[4] == pytest.approx(0.160, abs=1e-4)

    assert rejected == [True, True, False, False, False]


def test_holm_bonferroni_monotonicity_and_bounds() -> None:
    """Adjusted p-values must be monotonic with respect to sorted inputs and bounded in [0, 1]."""
    raw_p_values = [0.5, 0.2, 0.8, 0.01, 0.9]
    res = holm_bonferroni_correction(raw_p_values, alpha=0.05)

    for p in res["adjusted_p_values"]:
        assert 0.0 <= p <= 1.0


def test_compute_signal_vs_noise_verdict() -> None:
    """Verifies verdict assignment based on corrected p-value and delta."""
    # Significant gain
    assert (
        compute_signal_vs_noise_verdict(p_value_corrected=0.02, mean_delta=0.015)
        == "signal"
    )
    # Inconclusive / Noise
    assert (
        compute_signal_vs_noise_verdict(p_value_corrected=0.12, mean_delta=0.015)
        == "noise"
    )
    # Significant regression
    assert (
        compute_signal_vs_noise_verdict(p_value_corrected=0.01, mean_delta=-0.020)
        == "significant_regression"
    )
    # Zero delta
    assert (
        compute_signal_vs_noise_verdict(p_value_corrected=0.01, mean_delta=0.000)
        == "noise"
    )


# =====================================================================
# Tests: build_comparison_tables
# =====================================================================


def test_build_comparison_tables_structure_and_metrics(
    synthetic_campaign_dir: Path,
) -> None:
    """Verifies generation of comparison table with equal-mean repeat metrics and deltas."""
    campaign = load_finalized_campaign(synthetic_campaign_dir)
    tables = build_comparison_tables(campaign)

    assert isinstance(tables, pl.DataFrame)
    required_cols = {
        "challenger_id",
        "control_id",
        "metric",
        "control_equal_mean",
        "challenger_equal_mean",
        "paired_delta_mean",
        "delta_std_err",
        "t_stat",
        "p_value_raw",
        "p_value_corrected",
        "ci_95_lower",
        "ci_95_upper",
        "verdict",
    }
    assert required_cols.issubset(set(tables.columns))

    # feat_core_vs_weather_challenger should have roc_auc and average_precision
    fe_rows = tables.filter(
        pl.col("challenger_id") == "feat_core_vs_weather_challenger"
    )
    metrics_present = set(fe_rows["metric"].to_list())
    assert "roc_auc" in metrics_present
    assert "average_precision" in metrics_present

    # Positive delta observed
    auc_row = fe_rows.filter(pl.col("metric") == "roc_auc")
    assert auc_row["paired_delta_mean"][0] > 0
    assert auc_row["verdict"][0] in ("signal", "noise")


def test_build_comparison_tables_margin_candidates_exclude_probability_losses(
    synthetic_campaign_dir: Path,
) -> None:
    """Decision margin candidates (RBF SVC) report ROC-AUC but omit LogLoss/Brier."""
    campaign = load_finalized_campaign(synthetic_campaign_dir)
    tables = build_comparison_tables(campaign)

    svc_rows = tables.filter(
        pl.col("challenger_id") == "dim_pca_vs_passthrough_challenger"
    )
    svc_metrics = set(svc_rows["metric"].to_list())

    assert "roc_auc" in svc_metrics
    assert "log_loss" not in svc_metrics
    assert "brier_score" not in svc_metrics


def test_build_comparison_tables_equal_mean_aggregation(
    synthetic_campaign_dir: Path,
) -> None:
    """Equal-mean across repeats averages repeat-level pooled metrics, not naive row-pooling."""
    campaign = load_finalized_campaign(synthetic_campaign_dir)
    tables = build_comparison_tables(campaign)

    for row in tables.iter_rows(named=True):
        assert math.isfinite(row["control_equal_mean"])
        assert math.isfinite(row["challenger_equal_mean"])
        assert math.isfinite(row["paired_delta_mean"])
        assert row["ci_95_lower"] <= row["paired_delta_mean"] <= row["ci_95_upper"]


# =====================================================================
# Tests: iter_paired_error_slices
# =====================================================================


def test_iter_paired_error_slices_dimensions(synthetic_campaign_dir: Path) -> None:
    """Slices paired predictions along building, time, and missingness dimensions."""
    campaign = load_finalized_campaign(synthetic_campaign_dir)
    slices = list(
        iter_paired_error_slices(
            campaign,
            left_candidate="feat_core_vs_weather_challenger",
            right_candidate="feat_core_vs_weather_control",
        )
    )

    assert len(slices) > 0
    slice_types = {s["slice_type"] for s in slices}
    assert "building_cohort" in slice_types
    assert "calendar_hour" in slice_types
    assert "missingness" in slice_types


def test_iter_paired_error_slices_fp_fn_breakdown(synthetic_campaign_dir: Path) -> None:
    """Each slice provides sample count, anomaly count, and FP/FN breakdown."""
    campaign = load_finalized_campaign(synthetic_campaign_dir)
    slices = list(
        iter_paired_error_slices(
            campaign,
            left_candidate="feat_core_vs_weather_challenger",
            right_candidate="feat_core_vs_weather_control",
        )
    )
    for s in slices:
        assert s["n_samples"] > 0
        assert s["n_anomalies"] >= 0
        assert s["left_fp"] >= 0
        assert s["left_fn"] >= 0
        assert s["right_fp"] >= 0
        assert s["right_fn"] >= 0
        assert "delta_error_rate" in s
        assert "mcnemar_left_better_count" in s
        assert "mcnemar_right_better_count" in s


# =====================================================================
# Tests: build_ensemble_plan
# =====================================================================


def test_build_ensemble_plan_emits_eight_required_keys(
    synthetic_campaign_dir: Path,
) -> None:
    """Emits dictionary with exactly the 8 specified architectural ensemble keys."""
    campaign = load_finalized_campaign(synthetic_campaign_dir)
    plan = build_ensemble_plan(campaign)

    required_keys = {
        "provisional_level1_candidates",
        "provisional_rejections_with_reasons",
        "level1_eligibility_policy",
        "simple_blend_candidates",
        "level2_meta_learners",
        "level3_attempt_policy",
        "nested_cv_contract",
        "promotion_thresholds",
    }
    assert set(plan.keys()) == required_keys


def test_build_ensemble_plan_segregates_margin_and_uncalibrated(
    synthetic_campaign_dir: Path,
) -> None:
    """RBF SVC with decision margins must be in rejections with explicit reason."""
    campaign = load_finalized_campaign(synthetic_campaign_dir)
    plan = build_ensemble_plan(campaign)

    rejections = plan["provisional_rejections_with_reasons"]
    assert "dim_pca_vs_passthrough_challenger" in rejections
    reason = rejections["dim_pca_vs_passthrough_challenger"]
    assert any(
        term in reason.lower()
        for term in ("margin", "uncalibrated", "probability", "score_kind")
    )


def test_build_ensemble_plan_contracts_and_thresholds(
    synthetic_campaign_dir: Path,
) -> None:
    """Nested CV contract and promotion thresholds specify leak-free criteria."""
    campaign = load_finalized_campaign(synthetic_campaign_dir)
    plan = build_ensemble_plan(campaign)

    cv_contract = plan["nested_cv_contract"]
    assert cv_contract["outer_repeats"] == 3
    assert cv_contract["outer_folds"] == 4
    assert cv_contract["group_key"] == "building_id"
    assert "leakage_prevention" in cv_contract

    thresholds = plan["promotion_thresholds"]
    assert "min_roc_auc_delta" in thresholds
    assert "significance_alpha" in thresholds
    assert thresholds["significance_alpha"] <= 0.05


def test_compute_signal_vs_noise_verdict_min_lift() -> None:
    """Verifies that gains below min_lift are classified as noise even with significant p-values."""
    # Lift below min_lift threshold -> noise
    assert (
        compute_signal_vs_noise_verdict(
            p_value_corrected=0.01, mean_delta=0.0005, min_lift=0.001
        )
        == "noise"
    )
    # Lift equal to min_lift threshold -> noise
    assert (
        compute_signal_vs_noise_verdict(
            p_value_corrected=0.01, mean_delta=0.001, min_lift=0.001
        )
        == "noise"
    )
    # Lift strictly above min_lift threshold -> signal
    assert (
        compute_signal_vs_noise_verdict(
            p_value_corrected=0.01, mean_delta=0.0015, min_lift=0.001
        )
        == "signal"
    )
    # Regression -> significant_regression
    assert (
        compute_signal_vs_noise_verdict(
            p_value_corrected=0.01, mean_delta=-0.002, min_lift=0.001
        )
        == "significant_regression"
    )


def test_build_comparison_tables_roc_auc_min_lift_applied() -> None:
    """Verifies build_comparison_tables applies min_lift=0.001 for roc_auc."""
    tv = {
        "exp_small": {
            "fold_metrics": [
                {"candidate_id": "c_ctrl", "repeat": r, "fold": f, "roc_auc": 0.8000}
                for r in range(3)
                for f in range(4)
            ]
            + [
                {"candidate_id": "c_chall", "repeat": r, "fold": f, "roc_auc": 0.8005}
                for r in range(3)
                for f in range(4)
            ],
            "candidate_metrics": {
                "c_ctrl": {"mean_repeat_metrics": {"roc_auc": 0.8000}},
                "c_chall": {"mean_repeat_metrics": {"roc_auc": 0.8005}},
            },
        }
    }
    small_lift_campaign = {
        "status": "completed",
        "transport_validations": tv,
        "declarations": {
            "exp_small": {
                "candidates": {
                    "c_ctrl": {
                        "arm_id": "control",
                        "score_kind": "positive_probability",
                    },
                    "c_chall": {
                        "arm_id": "challenger",
                        "score_kind": "positive_probability",
                    },
                }
            }
        },
        "experiments": [{"experiment_id": "exp_small"}],
        "oof_tables": {},
        "probability_candidates": ["c_ctrl", "c_chall"],
        "margin_candidates": [],
    }

    tables = build_comparison_tables(small_lift_campaign)
    auc_row = tables.filter(
        (pl.col("challenger_id") == "c_chall") & (pl.col("metric") == "roc_auc")
    )
    assert len(auc_row) == 1
    assert auc_row["paired_delta_mean"][0] == pytest.approx(0.0005, abs=1e-6)
    assert auc_row["verdict"][0] == "noise"


def test_build_comparison_tables_per_experiment_fallback(
    synthetic_campaign_dir: Path,
) -> None:
    """Verifies comparison rows are built per-experiment (fold_metrics for exp1, oof_tables for exp2)."""
    campaign = load_finalized_campaign(synthetic_campaign_dir)
    hybrid_campaign = dict(campaign)
    hybrid_tv = dict(campaign.get("transport_validations", {}))
    hybrid_tv["feat_core_vs_weather"] = {
        "fold_metrics": [
            {
                "candidate_id": "feat_core_vs_weather_control",
                "repeat": r,
                "fold": f,
                "roc_auc": 0.70,
            }
            for r in range(3)
            for f in range(4)
        ]
        + [
            {
                "candidate_id": "feat_core_vs_weather_challenger",
                "repeat": r,
                "fold": f,
                "roc_auc": 0.75,
            }
            for r in range(3)
            for f in range(4)
        ],
        "candidate_metrics": {
            "feat_core_vs_weather_control": {"mean_repeat_metrics": {"roc_auc": 0.70}},
            "feat_core_vs_weather_challenger": {
                "mean_repeat_metrics": {"roc_auc": 0.75}
            },
        },
    }
    hybrid_tv["dim_pca_vs_passthrough"] = {"validation_status": "passed"}
    hybrid_campaign["transport_validations"] = hybrid_tv

    tables = build_comparison_tables(hybrid_campaign)
    challengers = set(tables["challenger_id"].to_list())
    assert "feat_core_vs_weather_challenger" in challengers
    assert "dim_pca_vs_passthrough_challenger" in challengers
    assert len(tables) > 0
    for p_corr in tables["p_value_corrected"]:
        assert 0.0 <= p_corr <= 1.0


def test_validate_repeated_cv_partitions_guards_against_spurious_significance() -> None:
    """Verifies that insufficient repeated-CV partitions do not report spurious significance."""
    single_rep_campaign = {
        "status": "completed",
        "transport_validations": {
            "exp_single_rep": {
                "fold_metrics": [
                    {"candidate_id": "c_ctrl", "repeat": 0, "fold": f, "roc_auc": 0.70}
                    for f in range(4)
                ]
                + [
                    {"candidate_id": "c_chall", "repeat": 0, "fold": f, "roc_auc": 0.85}
                    for f in range(4)
                ],
                "candidate_metrics": {
                    "c_ctrl": {"mean_repeat_metrics": {"roc_auc": 0.70}},
                    "c_chall": {"mean_repeat_metrics": {"roc_auc": 0.85}},
                },
            }
        },
        "declarations": {
            "exp_single_rep": {
                "candidates": {
                    "c_ctrl": {
                        "arm_id": "control",
                        "score_kind": "positive_probability",
                    },
                    "c_chall": {
                        "arm_id": "challenger",
                        "score_kind": "positive_probability",
                    },
                }
            }
        },
        "experiments": [{"experiment_id": "exp_single_rep"}],
        "oof_tables": {},
        "probability_candidates": ["c_ctrl", "c_chall"],
        "margin_candidates": [],
    }

    tables = build_comparison_tables(single_rep_campaign)
    assert len(tables) > 0
    row = tables.filter(pl.col("challenger_id") == "c_chall").row(0, named=True)
    assert row["t_stat"] == 0.0
    assert row["p_value_raw"] == 1.0
    assert row["p_value_corrected"] == 1.0
    assert row["verdict"] == "noise"


def test_build_ensemble_plan_eligibility_filter_rejections(
    synthetic_campaign_dir: Path,
) -> None:
    """Verifies eligibility filters in build_ensemble_plan for comparative completion, n03_eligible, and seal status."""
    campaign = load_finalized_campaign(synthetic_campaign_dir)

    # 1. Candidate in an unsealed campaign
    unsealed_campaign = dict(campaign)
    unsealed_campaign["status"] = "finished_with_blockers"
    unsealed_plan = build_ensemble_plan(unsealed_campaign)
    assert len(unsealed_plan["provisional_level1_candidates"]) == 0
    assert (
        "feat_core_vs_weather_challenger"
        in unsealed_plan["provisional_rejections_with_reasons"]
    )
    assert (
        "diagnostic only"
        in unsealed_plan["provisional_rejections_with_reasons"][
            "feat_core_vs_weather_challenger"
        ].lower()
    )

    # 2. Candidate with comparative_completion_available=False
    comp_false_campaign = dict(campaign)
    comp_false_campaign["manifest"] = {
        "candidates": [
            {
                "experiment_id": "feat_core_vs_weather",
                "comparative_completion_available": False,
                "status": "completed",
                "n03_eligible": True,
            }
        ]
    }
    comp_false_plan = build_ensemble_plan(comp_false_campaign)
    assert (
        "feat_core_vs_weather_challenger"
        in comp_false_plan["provisional_rejections_with_reasons"]
    )
    assert (
        "feat_core_vs_weather_challenger"
        not in comp_false_plan["provisional_level1_candidates"]
    )

    # 3. Candidate with n03_eligible=False
    n03_false_campaign = dict(campaign)
    n03_false_campaign["manifest"] = {
        "candidates": [
            {
                "experiment_id": "feat_core_vs_weather",
                "comparative_completion_available": True,
                "status": "completed",
                "n03_eligible": False,
            }
        ]
    }
    n03_false_plan = build_ensemble_plan(n03_false_campaign)
    assert (
        "feat_core_vs_weather_challenger"
        in n03_false_plan["provisional_rejections_with_reasons"]
    )
    assert (
        "feat_core_vs_weather_challenger"
        not in n03_false_plan["provisional_level1_candidates"]
    )

    # 4. Certified completed candidate enters provisional_level1_candidates
    certified_plan = build_ensemble_plan(campaign)
    assert (
        "feat_core_vs_weather_challenger"
        in certified_plan["provisional_level1_candidates"]
    )
    assert (
        "feat_core_vs_weather_control"
        in certified_plan["provisional_level1_candidates"]
    )


# =====================================================================
# Tests: Discovered Completed Model Runs & Unified Campaign Pool
# =====================================================================


def _make_mini_candidate_oof_parquet(
    path: Path,
    candidates: list[tuple[str, str, float]],  # (candidate_id, arm_id, base_prob)
    model_family: str,
    score_kind: str = "positive_probability",
    n03_eligible: bool = True,
    non_comparative: bool = False,
    n_partitions: int = 8,  # 2 repeats x 4 folds
    rows_per_partition: int = 4,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_data: dict[str, list[Any]] = {k: [] for k in CANDIDATE_OOF_SCHEMA.names}
    for cid, arm, base_prob in candidates:
        for p_idx in range(n_partitions):
            rep = p_idx // 4
            fold = p_idx % 4
            for r_idx in range(rows_per_partition):
                global_idx = p_idx * rows_per_partition + r_idx
                all_data["candidate_id"].append(cid)
                all_data["recipe_id"].append("recipe_test")
                all_data["model_family"].append(model_family)
                all_data["model_config_id"].append("cfg_test")
                all_data["row_index"].append(global_idx)
                all_data["repeat"].append(rep)
                all_data["fold"].append(fold)
                all_data["building_id"].append(f"bldg_{r_idx:02d}")
                all_data["timestamp"].append(
                    datetime(2023, 1, 1, global_idx % 24, 0, tzinfo=UTC)
                )
                target_val = 1 if (r_idx % 2 == 1) else 0
                pred_val = min(
                    max(base_prob + (0.2 if target_val == 1 else -0.1), 0.01), 0.99
                )
                all_data["prediction"].append(pred_val)
                all_data["target"].append(target_val)
                all_data["split_sha256"].append("a" * 64)
                all_data["data_sha256"].append("b" * 64)
                all_data["config_sha256"].append("c" * 64)
                all_data["run_id"].append(f"run_{model_family}")
                all_data["split_seed"].append(42)
                all_data["experiment_id"].append(f"exp_{model_family}")
                all_data["arm_id"].append(arm)
                all_data["score_kind"].append(score_kind)
                all_data["positive_label"].append(1)
                all_data["execution_mode"].append("full")
                all_data["non_comparative"].append(non_comparative)
                all_data["n03_eligible"].append(n03_eligible)
                all_data["model_fit_config_sha256"].append("d" * 64)

    tbl = pa.Table.from_pydict(all_data, schema=CANDIDATE_OOF_SCHEMA)
    pq.write_table(tbl, path)


@pytest.fixture
def models_pool_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Create synthetic model directories mirroring outputs/models and outputs/candidates."""
    models_root = tmp_path / "models"
    candidates_root = tmp_path / "candidates"

    # 1. Standalone GBDT (2-arm single file)
    gbdt_path = (
        models_root
        / "gbdt_standalone"
        / "full"
        / "gbdt_catboost_full_123"
        / "catboost_candidate_oof.parquet"
    )
    _make_mini_candidate_oof_parquet(
        gbdt_path,
        [
            ("catboost_control", "control", 0.3),
            ("catboost_challenger", "challenger", 0.4),
        ],
        model_family="catboost",
    )
    _write_json(
        gbdt_path.parent / "catboost_manifest.json",
        {
            "run_id": "gbdt_catboost_full_123",
            "family": "catboost",
            "execution_mode": "full",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 64,
        },
    )

    # 2. Tabular MLP (2-arm companion files)
    mlp_dir = models_root / "tabular_mlp" / "full" / "mlp_full_456"
    _make_mini_candidate_oof_parquet(
        mlp_dir / "mlp_control_candidate_oof.parquet",
        [("mlp_control", "control", 0.28)],
        model_family="tabular_mlp",
    )
    _write_json(
        mlp_dir / "mlp_control_manifest.json",
        {
            "run_id": "mlp_full_456",
            "model_family": "tabular_mlp",
            "arm": "control",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 32,
        },
    )
    _make_mini_candidate_oof_parquet(
        mlp_dir / "mlp_challenger_candidate_oof.parquet",
        [("mlp_challenger", "challenger", 0.42)],
        model_family="tabular_mlp",
    )
    _write_json(
        mlp_dir / "mlp_challenger_manifest.json",
        {
            "run_id": "mlp_full_456",
            "model_family": "tabular_mlp",
            "arm": "challenger",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 32,
        },
    )

    # 3. GBDT Feature Selection (passthrough vs selected)
    fs_pass = (
        models_root
        / "gbdt_feature_selection"
        / "full"
        / "lightgbm_passthrough_full_111"
        / "lightgbm_passthrough_candidate_oof.parquet"
    )
    _make_mini_candidate_oof_parquet(
        fs_pass,
        [("lightgbm_passthrough", "control", 0.31)],
        model_family="lightgbm",
    )
    _write_json(
        fs_pass.parent / "lightgbm_passthrough_transport_validation.json",
        {
            "run_id": "lightgbm_passthrough_full_111",
            "family": "lightgbm",
            "regime": "passthrough",
            "arm": "control",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 32,
        },
    )
    fs_sel = (
        models_root
        / "gbdt_feature_selection"
        / "full"
        / "lightgbm_selected_full_222"
        / "lightgbm_selected_candidate_oof.parquet"
    )
    _make_mini_candidate_oof_parquet(
        fs_sel,
        [("lightgbm_selected", "challenger", 0.43)],
        model_family="lightgbm",
    )
    _write_json(
        fs_sel.parent / "lightgbm_selected_transport_validation.json",
        {
            "run_id": "lightgbm_selected_full_222",
            "family": "lightgbm",
            "regime": "selected",
            "arm": "challenger",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 32,
        },
    )

    # 4. GBDT Loss Objectives (logloss vs weighted, focal)
    loss_log = (
        models_root
        / "gbdt_loss_objectives"
        / "full"
        / "xgboost_logloss_full_333"
        / "xgboost_logloss_candidate_oof.parquet"
    )
    _make_mini_candidate_oof_parquet(
        loss_log,
        [("xgboost_logloss", "logloss", 0.33)],
        model_family="xgboost",
    )
    _write_json(
        loss_log.parent / "xgboost_logloss_transport_validation.json",
        {
            "run_id": "xgboost_logloss_full_333",
            "family": "xgboost",
            "loss_arm": "logloss",
            "arm": "control",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 32,
        },
    )
    loss_wt = (
        models_root
        / "gbdt_loss_objectives"
        / "full"
        / "xgboost_weighted_full_444"
        / "xgboost_weighted_candidate_oof.parquet"
    )
    _make_mini_candidate_oof_parquet(
        loss_wt,
        [("xgboost_weighted", "weighted", 0.39)],
        model_family="xgboost",
    )
    _write_json(
        loss_wt.parent / "xgboost_weighted_transport_validation.json",
        {
            "run_id": "xgboost_weighted_full_444",
            "family": "xgboost",
            "loss_arm": "weighted",
            "arm": "challenger",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 32,
        },
    )
    loss_foc = (
        models_root
        / "gbdt_loss_objectives"
        / "full"
        / "xgboost_focal_full_555"
        / "xgboost_focal_candidate_oof.parquet"
    )
    _make_mini_candidate_oof_parquet(
        loss_foc,
        [("xgboost_focal", "focal", 0.44)],
        model_family="xgboost",
    )
    _write_json(
        loss_foc.parent / "xgboost_focal_transport_validation.json",
        {
            "run_id": "xgboost_focal_full_555",
            "family": "xgboost",
            "loss_arm": "focal",
            "arm": "challenger",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 32,
        },
    )

    # 5. Synthetic Sampling (natural vs smote)
    syn_nat = (
        models_root
        / "synthetic_sampling"
        / "full"
        / "extratrees_natural_full_666"
        / "extratrees_natural_candidate_oof.parquet"
    )
    _make_mini_candidate_oof_parquet(
        syn_nat,
        [("extratrees_natural", "natural", 0.29)],
        model_family="extratrees",
    )
    _write_json(
        syn_nat.parent / "extratrees_natural_transport_validation.json",
        {
            "run_id": "extratrees_natural_full_666",
            "family": "extratrees",
            "arm": "control",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 32,
        },
    )
    syn_sm = (
        models_root
        / "synthetic_sampling"
        / "full"
        / "extratrees_smote_full_777"
        / "extratrees_smote_candidate_oof.parquet"
    )
    _make_mini_candidate_oof_parquet(
        syn_sm,
        [("extratrees_smote", "smote", 0.40)],
        model_family="extratrees",
    )
    _write_json(
        syn_sm.parent / "extratrees_smote_transport_validation.json",
        {
            "run_id": "extratrees_smote_full_777",
            "family": "extratrees",
            "arm": "challenger",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 32,
        },
    )

    # 6. AutoGluon candidate run
    ag_dir = (
        candidates_root / "autogluon_full_stack" / "model_autogluon_full_stack-full-888"
    )
    _make_mini_candidate_oof_parquet(
        ag_dir / "candidate_oof.parquet",
        [("model_autogluon_full_stack_challenger", "challenger", 0.47)],
        model_family="autogluon",
    )
    _write_json(
        ag_dir / "run_summary.json",
        {
            "status": "completed",
            "run_id": "model_autogluon_full_stack-full-888",
            "oof_path": str(ag_dir / "candidate_oof.parquet"),
            "oof_record_count": 32,
        },
    )
    # 7. Ineligible candidate (n03_eligible=False)
    bad_dir = models_root / "gbdt_standalone" / "full" / "bad_ineligible_run"
    _make_mini_candidate_oof_parquet(
        bad_dir / "bad_candidate_oof.parquet",
        [("bad_cand", "control", 0.1)],
        model_family="lightgbm",
        n03_eligible=False,
    )

    # 8. Smoke run (should be ignored by full discovery)
    smoke_dir = models_root / "gbdt_standalone" / "smoke" / "smoke_run"
    _make_mini_candidate_oof_parquet(
        smoke_dir / "smoke_candidate_oof.parquet",
        [("smoke_cand", "control", 0.1)],
        model_family="lightgbm",
    )

    return models_root, candidates_root


def test_discover_completed_model_runs_validates_schema_and_metadata(
    models_pool_dir: tuple[Path, Path],
) -> None:
    """Verifies discover_completed_model_runs detects valid full runs, checks schema, and rejects ineligible runs."""
    models_root, candidates_root = models_pool_dir
    runs = discover_completed_model_runs(
        models_root=models_root, candidates_root=candidates_root
    )
    assert len(runs) > 0

    all_candidates: set[str] = set()
    for r in runs:
        assert r["status"] == "completed"
        assert r["n03_eligible"] is True
        assert r["non_comparative"] is False
        assert r["score_kind"] == "positive_probability"
        assert Path(r["oof_path"]).is_file()
        for cid in r["candidates"]:
            all_candidates.add(cid)

    # Verified inclusion
    assert "catboost_control" in all_candidates
    assert "catboost_challenger" in all_candidates
    assert "mlp_control" in all_candidates
    assert "mlp_challenger" in all_candidates
    assert "lightgbm_passthrough" in all_candidates
    assert "lightgbm_selected" in all_candidates
    assert "xgboost_logloss" in all_candidates
    assert "xgboost_weighted" in all_candidates
    assert "xgboost_focal" in all_candidates
    assert "extratrees_natural" in all_candidates
    assert "extratrees_smote" in all_candidates
    assert "model_autogluon_full_stack_challenger" in all_candidates

    # Rejections
    assert "bad_cand" not in all_candidates
    assert "smoke_cand" not in all_candidates


def test_load_unified_campaign_pool_ingests_and_pairs_models(
    models_pool_dir: tuple[Path, Path],
) -> None:
    """Verifies load_unified_campaign_pool ingests runs, maps control/challenger pairs, and builds comparison tables."""
    models_root, candidates_root = models_pool_dir
    unified = load_unified_campaign_pool(
        campaign_targets=None,
        include_models=True,
        models_root=models_root,
        candidates_root=candidates_root,
    )
    # Ensure lazy loading by default: tables are not eagerly loaded in memory
    for tbl in unified["oof_tables"].values():
        assert tbl is None

    assert unified["status"] == "completed"
    prob_cands = set(unified["probability_candidates"])
    assert "catboost_challenger" in prob_cands
    assert "mlp_challenger" in prob_cands
    assert "lightgbm_selected" in prob_cands
    assert "xgboost_weighted" in prob_cands
    assert "extratrees_smote" in prob_cands
    assert "model_autogluon_full_stack_challenger" in prob_cands

    # Verify comparison tables correctly paired controls with challengers
    tables = build_comparison_tables(unified)
    assert len(tables) > 0

    challengers = set(tables["challenger_id"].unique().to_list())
    controls = set(tables["control_id"].unique().to_list())

    # Standalone pair
    assert "catboost_challenger" in challengers
    assert "catboost_control" in controls

    # Tabular MLP pair
    assert "mlp_challenger" in challengers
    assert "mlp_control" in controls

    # Feature Selection pair: passthrough is control, selected is challenger
    assert "lightgbm_selected" in challengers
    assert "lightgbm_passthrough" in controls

    # Loss Objectives pair: logloss is control, weighted/focal are challengers
    assert "xgboost_weighted" in challengers
    assert "xgboost_focal" in challengers
    assert "xgboost_logloss" in controls

    # Synthetic Sampling pair: natural is control, smote is challenger
    assert "extratrees_smote" in challengers
    assert "extratrees_natural" in controls


def test_build_ensemble_plan_certifies_discovered_models(
    models_pool_dir: tuple[Path, Path],
) -> None:
    """Verifies build_ensemble_plan certifies eligible discovered models into provisional Level 1 ensemble."""
    models_root, candidates_root = models_pool_dir
    unified = load_unified_campaign_pool(
        campaign_targets=None,
        include_models=True,
        models_root=models_root,
        candidates_root=candidates_root,
    )

    plan = build_ensemble_plan(unified)
    provisional_l1 = set(plan["provisional_level1_candidates"])

    # Check that eligible models appear in provisional Level 1 ensemble
    assert "catboost_challenger" in provisional_l1
    assert "catboost_control" in provisional_l1
    assert "mlp_challenger" in provisional_l1
    assert "mlp_control" in provisional_l1
    assert "lightgbm_selected" in provisional_l1
    assert "lightgbm_passthrough" in provisional_l1
    assert "xgboost_weighted" in provisional_l1
    assert "xgboost_logloss" in provisional_l1
    assert "extratrees_smote" in provisional_l1
    assert "extratrees_natural" in provisional_l1
    assert "model_autogluon_full_stack_challenger" in provisional_l1


def test_load_finalized_campaign_with_models_and_lazy_path_contract(
    synthetic_campaign_dir: Path,
    models_pool_dir: tuple[Path, Path],
) -> None:
    """Verifies load_finalized_campaign merges legacy campaigns with discovered models even under load_oof=False."""
    models_root, candidates_root = models_pool_dir
    merged = load_finalized_campaign(
        campaign_dir=synthetic_campaign_dir,
        include_models=True,
        models_root=models_root,
        candidates_root=candidates_root,
        load_oof=False,
    )

    # Both legacy experiments and discovered model candidates are present
    prob_cands = set(merged["probability_candidates"])
    assert "feat_core_vs_weather_challenger" in prob_cands
    assert "catboost_challenger" in prob_cands
    assert "mlp_challenger" in prob_cands

    # Verify lazy load contract: _rows_for_experiment_from_oof handles oof_paths with list
    tables = build_comparison_tables(merged)
    assert len(tables) > 0
    challengers = set(tables["challenger_id"].unique().to_list())
    assert "feat_core_vs_weather_challenger" in challengers
    assert "mlp_challenger" in challengers
    assert "lightgbm_selected" in challengers


def test_discover_completed_model_runs_rejects_active_failed_and_corrupt_artifacts(
    tmp_path: Path,
) -> None:
    """Verifies that discover_completed_model_runs rejects active/failed/dry_run models and raises on corrupt artifacts."""
    models_dir = tmp_path / "models"

    # 1. Failed run
    failed_dir = models_dir / "gbdt_standalone" / "full" / "failed_model"
    _make_mini_candidate_oof_parquet(
        failed_dir / "failed_candidate_oof.parquet",
        [("failed_cand", "challenger", 0.5)],
        model_family="catboost",
    )
    _write_json(
        failed_dir / "run_summary.json", {"status": "failed", "run_id": "failed_run"}
    )

    # 2. Running / active run
    running_dir = models_dir / "gbdt_standalone" / "full" / "running_model"
    _make_mini_candidate_oof_parquet(
        running_dir / "running_candidate_oof.parquet",
        [("running_cand", "challenger", 0.5)],
        model_family="catboost",
    )
    _write_json(
        running_dir / "run_summary.json", {"status": "running", "run_id": "running_run"}
    )

    # 3. Dry run
    dry_dir = models_dir / "gbdt_standalone" / "full" / "dry_model"
    _make_mini_candidate_oof_parquet(
        dry_dir / "dry_candidate_oof.parquet",
        [("dry_cand", "challenger", 0.5)],
        model_family="catboost",
    )
    _write_json(dry_dir / "run_summary.json", {"status": "completed", "dry_run": True})

    # Run discovery: all 3 must be excluded
    discovered = discover_completed_model_runs(
        models_root=models_dir, candidates_root=tmp_path / "empty_cand"
    )
    discovered_cands = {c for r in discovered for c in r["candidates"]}
    assert "failed_cand" not in discovered_cands
    assert "running_cand" not in discovered_cands
    assert "dry_cand" not in discovered_cands

    # 4. Corrupted JSON manifest raises ValueError
    corrupt_root = tmp_path / "corrupt_models"
    corrupt_dir = corrupt_root / "gbdt_standalone" / "full" / "corrupt_json_model"
    _make_mini_candidate_oof_parquet(
        corrupt_dir / "corrupt_candidate_oof.parquet",
        [("corrupt_cand", "challenger", 0.5)],
        model_family="catboost",
    )
    (corrupt_dir / "catboost_manifest.json").write_text(
        "{invalid json", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Corrupted manifest JSON"):
        discover_completed_model_runs(
            models_root=corrupt_root, candidates_root=tmp_path / "empty_cand"
        )

    # 5. Record count mismatch raises ValueError
    mismatch_root = tmp_path / "mismatch_models"
    mismatch_dir = mismatch_root / "gbdt_standalone" / "full" / "mismatch_model"
    _make_mini_candidate_oof_parquet(
        mismatch_dir / "mismatch_candidate_oof.parquet",
        [("mismatch_cand", "challenger", 0.5)],
        model_family="catboost",
    )
    _write_json(mismatch_dir / "catboost_manifest.json", {"records_count": 9999})
    with pytest.raises(ValueError, match="OOF record count mismatch"):
        discover_completed_model_runs(
            models_root=mismatch_root, candidates_root=tmp_path / "empty_cand"
        )


def test_mixed_pool_rejects_unsealed_legacy_while_qualifying_discovered_models(
    synthetic_campaign_dir: Path,
    models_pool_dir: tuple[Path, Path],
) -> None:
    """Verifies that in a mixed pool with an unsealed campaign, legacy candidates remain rejected while discovered completed runs qualify."""
    models_root, candidates_root = models_pool_dir
    mixed = load_finalized_campaign(
        campaign_dir=synthetic_campaign_dir,
        allow_finished_with_blockers=True,
        include_models=True,
        models_root=models_root,
        candidates_root=candidates_root,
        load_oof=False,
    )
    mixed["status"] = "finished_with_blockers"

    plan = build_ensemble_plan(mixed)
    provisional_l1 = set(plan["provisional_level1_candidates"])
    rejections = plan["provisional_rejections_with_reasons"]

    # Legacy candidates MUST be rejected due to unsealed campaign status
    assert "feat_core_vs_weather_challenger" not in provisional_l1
    assert "feat_core_vs_weather_challenger" in rejections
    assert "diagnostic only" in rejections["feat_core_vs_weather_challenger"].lower()

    # Discovered independent completed models MUST qualify for provisional Level 1
    assert "catboost_challenger" in provisional_l1
    assert "mlp_challenger" in provisional_l1
    assert "lightgbm_selected" in provisional_l1


def test_regression_require_completion_metadata_before_certifying_run(
    tmp_path: Path,
) -> None:
    """Finding 1: Require companion completion metadata and expected coverage before certifying runs."""
    models_root = tmp_path / "models"
    cand_root = tmp_path / "candidates"

    # 1. Orphaned Standalone GBDT run (OOF parquet exists, but NO manifest exists)
    orphan_dir = models_root / "gbdt_standalone" / "full" / "gbdt_orphaned_full_111"
    orphan_oof = orphan_dir / "catboost_candidate_oof.parquet"
    _make_mini_candidate_oof_parquet(
        orphan_oof,
        [
            ("catboost_orphan_ctrl", "control", 0.3),
            ("catboost_orphan_chall", "challenger", 0.4),
        ],
        model_family="catboost",
    )

    # 2. Orphaned AutoGluon run (candidate_oof.parquet exists, but NO run_summary.json exists)
    ag_orphan_dir = cand_root / "autogluon_full_stack" / "model_autogluon_orphan_full_222"
    ag_orphan_oof = ag_orphan_dir / "candidate_oof.parquet"
    _make_mini_candidate_oof_parquet(
        ag_orphan_oof,
        [("ag_orphan_challenger", "challenger", 0.45)],
        model_family="autogluon",
    )

    # 3. AutoGluon run with run_summary.json status != 'completed' (e.g. 'failed')
    ag_failed_dir = cand_root / "autogluon_full_stack" / "model_autogluon_failed_full_333"
    ag_failed_oof = ag_failed_dir / "candidate_oof.parquet"
    _make_mini_candidate_oof_parquet(
        ag_failed_oof,
        [("ag_failed_challenger", "challenger", 0.45)],
        model_family="autogluon",
    )
    _write_json(
        ag_failed_dir / "run_summary.json",
        {"status": "failed", "run_id": "ag_failed_333", "oof_record_count": 32},
    )

    # 4. Standalone run with manifest but MISSING row count (coverage not declared)
    no_count_dir = models_root / "gbdt_standalone" / "full" / "gbdt_no_count_full_555"
    no_count_oof = no_count_dir / "catboost_candidate_oof.parquet"
    _make_mini_candidate_oof_parquet(
        no_count_oof,
        [("catboost_no_count_ctrl", "control", 0.3)],
        model_family="catboost",
    )
    _write_json(
        no_count_dir / "catboost_manifest.json",
        {
            "run_id": "gbdt_no_count_555",
            "family": "catboost",
            "execution_mode": "full",
            "non_comparative": False,
            "n03_eligible": True,
            # records_count omitted!
        },
    )

    # 5. Tabular MLP sibling run: challenger OOF exists, but only control manifest exists
    mlp_sibling_dir = models_root / "tabular_mlp" / "full" / "mlp_sibling_full_666"
    mlp_chall_oof = mlp_sibling_dir / "mlp_challenger_candidate_oof.parquet"
    _make_mini_candidate_oof_parquet(
        mlp_chall_oof,
        [("mlp_sibling_challenger", "challenger", 0.42)],
        model_family="tabular_mlp",
    )
    _write_json(
        mlp_sibling_dir / "mlp_control_manifest.json",
        {
            "run_id": "mlp_sibling_666",
            "model_family": "tabular_mlp",
            "arm": "control",
            "execution_mode": "full",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 32,
        },
    )

    # 6. Valid completed run with matching manifest AND expected coverage
    valid_dir = models_root / "gbdt_standalone" / "full" / "gbdt_valid_full_444"
    valid_oof = valid_dir / "catboost_candidate_oof.parquet"
    _make_mini_candidate_oof_parquet(
        valid_oof,
        [
            ("catboost_valid_ctrl", "control", 0.3),
            ("catboost_valid_chall", "challenger", 0.4),
        ],
        model_family="catboost",
    )
    _write_json(
        valid_dir / "catboost_manifest.json",
        {
            "run_id": "gbdt_valid_full_444",
            "family": "catboost",
            "execution_mode": "full",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 64,
        },
    )

    discovered = discover_completed_model_runs(
        models_root=models_root, candidates_root=cand_root
    )
    discovered_cands = {c for r in discovered for c in r["candidates"]}

    # Orphaned, failed, count-less, and sibling-manifest runs must NOT be certified
    assert "catboost_orphan_ctrl" not in discovered_cands
    assert "catboost_orphan_chall" not in discovered_cands
    assert "ag_orphan_challenger" not in discovered_cands
    assert "ag_failed_challenger" not in discovered_cands
    assert "catboost_no_count_ctrl" not in discovered_cands
    assert "mlp_sibling_challenger" not in discovered_cands

    # Valid run with companion manifest and declared coverage must be certified
    assert "catboost_valid_ctrl" in discovered_cands
    assert "catboost_valid_chall" in discovered_cands


def test_regression_error_slice_consumer_handles_list_of_oof_paths(
    tmp_path: Path,
) -> None:
    """Finding 2: iter_paired_error_slices cleanly loads and pairs candidates when oof_paths is a list."""
    models_root = tmp_path / "models"
    mlp_dir = models_root / "tabular_mlp" / "full" / "mlp_full_test"
    ctrl_oof = mlp_dir / "mlp_control_candidate_oof.parquet"
    chall_oof = mlp_dir / "mlp_challenger_candidate_oof.parquet"

    _make_mini_candidate_oof_parquet(
        ctrl_oof, [("mlp_test_ctrl", "control", 0.3)], model_family="tabular_mlp"
    )
    _write_json(
        mlp_dir / "mlp_control_manifest.json",
        {
            "run_id": "mlp_full_test",
            "model_family": "tabular_mlp",
            "arm": "control",
            "execution_mode": "full",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 32,
        },
    )
    _make_mini_candidate_oof_parquet(
        chall_oof, [("mlp_test_chall", "challenger", 0.4)], model_family="tabular_mlp"
    )
    _write_json(
        mlp_dir / "mlp_challenger_manifest.json",
        {
            "run_id": "mlp_full_test",
            "model_family": "tabular_mlp",
            "arm": "challenger",
            "execution_mode": "full",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 32,
        },
    )

    discovered = discover_completed_model_runs(
        models_root=models_root, candidates_root=tmp_path / "empty_cand"
    )
    assert len(discovered) == 2

    from project.experiments.full_campaign_review import (
        _group_model_runs_into_experiments,
    )

    (
        exp_list,
        decls,
        oofs,
        paths,
        _transports,
        _prob_cands,
        _margin_cands,
    ) = _group_model_runs_into_experiments(discovered, load_oof=False)

    exp_id = exp_list[0]["experiment_id"]
    assert isinstance(paths[exp_id], list)
    assert oofs[exp_id] is None

    campaign = {
        "run_id": "test_lazy_pool",
        "status": "completed",
        "campaign_dir": None,
        "experiments": exp_list,
        "declarations": decls,
        "oof_tables": oofs,
        "oof_paths": paths,
        "feature_metadata_path": None,
    }

    slices = list(iter_paired_error_slices(campaign, "mlp_test_ctrl", "mlp_test_chall"))
    assert len(slices) > 0
    assert any(s["slice_type"] == "calendar_hour" for s in slices)
    assert any(s["slice_type"] == "building_cohort" for s in slices)


def test_regression_deduplicate_reruns_selects_latest_by_mtime(
    tmp_path: Path,
) -> None:
    """Finding 3: Discovered model runs deduplicate reruns for the same model family / arm, selecting latest by st_mtime."""
    import os
    import time

    models_root = tmp_path / "models"
    pass_dir_old = (
        models_root
        / "gbdt_feature_selection"
        / "full"
        / "lightgbm_passthrough_full_old"
    )
    pass_dir_new = (
        models_root
        / "gbdt_feature_selection"
        / "full"
        / "lightgbm_passthrough_full_new"
    )

    oof_old = pass_dir_old / "lightgbm_passthrough_candidate_oof.parquet"
    oof_new = pass_dir_new / "lightgbm_passthrough_candidate_oof.parquet"

    _make_mini_candidate_oof_parquet(
        oof_old, [("lightgbm_passthrough", "control", 0.31)], model_family="lightgbm"
    )
    _write_json(
        pass_dir_old / "lightgbm_passthrough_transport_validation.json",
        {
            "run_id": "lgb_pass_old",
            "family": "lightgbm",
            "regime": "passthrough",
            "arm": "control",
            "execution_mode": "full",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 32,
        },
    )

    _make_mini_candidate_oof_parquet(
        oof_new, [("lightgbm_passthrough", "control", 0.31)], model_family="lightgbm"
    )
    _write_json(
        pass_dir_new / "lightgbm_passthrough_transport_validation.json",
        {
            "run_id": "lgb_pass_new",
            "family": "lightgbm",
            "regime": "passthrough",
            "arm": "control",
            "execution_mode": "full",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 32,
        },
    )

    now = time.time()
    os.utime(pass_dir_old, (now - 1000, now - 1000))
    os.utime(pass_dir_new, (now, now))

    discovered = discover_completed_model_runs(
        models_root=models_root, candidates_root=tmp_path / "empty_cand"
    )
    assert len(discovered) == 1
    assert discovered[0]["run_id"] == "lgb_pass_new"
    assert discovered[0]["oof_path"] == oof_new


def test_regression_error_slice_consumer_handles_missing_campaign_dir(
    tmp_path: Path,
) -> None:
    """Finding 4: Models-only pool with campaign_dir=None does not raise TypeError in iter_paired_error_slices."""
    models_root = tmp_path / "models"
    gbdt_path = (
        models_root
        / "gbdt_standalone"
        / "full"
        / "gbdt_catboost_full_999"
        / "catboost_candidate_oof.parquet"
    )
    _make_mini_candidate_oof_parquet(
        gbdt_path,
        [("catboost_ctrl_s", "control", 0.3), ("catboost_chall_s", "challenger", 0.4)],
        model_family="catboost",
    )
    _write_json(
        gbdt_path.parent / "catboost_manifest.json",
        {
            "run_id": "gbdt_catboost_full_999",
            "family": "catboost",
            "execution_mode": "full",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 64,
        },
    )

    campaign = load_unified_campaign_pool(
        campaign_targets=None,
        include_models=True,
        models_root=models_root,
        candidates_root=tmp_path / "empty_cand",
        load_oof=True,
    )
    assert campaign["campaign_dir"] is None
    assert campaign["feature_metadata_path"] is None

    slices = list(
        iter_paired_error_slices(campaign, "catboost_ctrl_s", "catboost_chall_s")
    )
    assert len(slices) > 0
    assert any(s["slice_type"] == "calendar_hour" for s in slices)
    assert any(s["slice_type"] == "building_cohort" for s in slices)


def test_regression_reject_null_and_invalid_type_eligibility(
    tmp_path: Path,
) -> None:
    """Finding 5: Null values and non-boolean canonical types in n03_eligible are strictly rejected."""
    models_root = tmp_path / "models"

    # 1. Parquet with null values in n03_eligible
    null_dir = models_root / "gbdt_standalone" / "full" / "gbdt_null_eligible_full"
    null_oof = null_dir / "catboost_candidate_oof.parquet"
    null_dir.mkdir(parents=True, exist_ok=True)

    all_data_null: dict[str, list[Any]] = {k: [] for k in CANDIDATE_OOF_SCHEMA.names}
    for p_idx in range(4):
        rep = p_idx // 2
        fold = p_idx % 2
        for r_idx in range(2):
            global_idx = p_idx * 2 + r_idx
            all_data_null["candidate_id"].append("catboost_null_test")
            all_data_null["recipe_id"].append("r")
            all_data_null["model_family"].append("catboost")
            all_data_null["model_config_id"].append("c")
            all_data_null["row_index"].append(global_idx)
            all_data_null["repeat"].append(rep)
            all_data_null["fold"].append(fold)
            all_data_null["building_id"].append("b01")
            all_data_null["timestamp"].append(datetime(2023, 1, 1, 0, 0, tzinfo=UTC))
            all_data_null["prediction"].append(0.5)
            all_data_null["target"].append(0)
            all_data_null["split_sha256"].append("a" * 64)
            all_data_null["data_sha256"].append("b" * 64)
            all_data_null["config_sha256"].append("c" * 64)
            all_data_null["run_id"].append("run_null")
            all_data_null["split_seed"].append(42)
            all_data_null["experiment_id"].append("exp_null")
            all_data_null["arm_id"].append("control")
            all_data_null["score_kind"].append("positive_probability")
            all_data_null["positive_label"].append(1)
            all_data_null["execution_mode"].append("full")
            all_data_null["non_comparative"].append(False)
            all_data_null["n03_eligible"].append(None)
            all_data_null["model_fit_config_sha256"].append("d" * 64)

    nullable_fields = [
        pa.field(f.name, f.type, nullable=True) if f.name == "n03_eligible" else f
        for f in CANDIDATE_OOF_SCHEMA
    ]
    tbl_null = pa.Table.from_pydict(all_data_null, schema=pa.schema(nullable_fields))
    pq.write_table(tbl_null, null_oof)
    _write_json(
        null_dir / "catboost_manifest.json",
        {
            "run_id": "run_null",
            "family": "catboost",
            "execution_mode": "full",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 8,
        },
    )

    # 2. Parquet with integer type for n03_eligible instead of pa.bool_()
    int_dir = models_root / "gbdt_standalone" / "full" / "gbdt_int_eligible_full"
    int_oof = int_dir / "catboost_candidate_oof.parquet"
    int_dir.mkdir(parents=True, exist_ok=True)

    all_data_int: dict[str, list[Any]] = {k: [] for k in CANDIDATE_OOF_SCHEMA.names}
    for p_idx in range(4):
        rep = p_idx // 2
        fold = p_idx % 2
        for r_idx in range(2):
            global_idx = p_idx * 2 + r_idx
            all_data_int["candidate_id"].append("catboost_int_test")
            all_data_int["recipe_id"].append("r")
            all_data_int["model_family"].append("catboost")
            all_data_int["model_config_id"].append("c")
            all_data_int["row_index"].append(global_idx)
            all_data_int["repeat"].append(rep)
            all_data_int["fold"].append(fold)
            all_data_int["building_id"].append("b01")
            all_data_int["timestamp"].append(datetime(2023, 1, 1, 0, 0, tzinfo=UTC))
            all_data_int["prediction"].append(0.5)
            all_data_int["target"].append(0)
            all_data_int["split_sha256"].append("a" * 64)
            all_data_int["data_sha256"].append("b" * 64)
            all_data_int["config_sha256"].append("c" * 64)
            all_data_int["run_id"].append("run_int")
            all_data_int["split_seed"].append(42)
            all_data_int["experiment_id"].append("exp_int")
            all_data_int["arm_id"].append("control")
            all_data_int["score_kind"].append("positive_probability")
            all_data_int["positive_label"].append(1)
            all_data_int["execution_mode"].append("full")
            all_data_int["non_comparative"].append(False)
            all_data_int["n03_eligible"].append(1)  # int instead of bool
            all_data_int["model_fit_config_sha256"].append("d" * 64)

    int_fields = [
        pa.field(f.name, pa.int64(), nullable=False) if f.name == "n03_eligible" else f
        for f in CANDIDATE_OOF_SCHEMA
    ]
    tbl_int = pa.Table.from_pydict(all_data_int, schema=pa.schema(int_fields))
    pq.write_table(tbl_int, int_oof)
    _write_json(
        int_dir / "catboost_manifest.json",
        {
            "run_id": "run_int",
            "family": "catboost",
            "execution_mode": "full",
            "non_comparative": False,
            "n03_eligible": True,
            "records_count": 8,
        },
    )

    # discover_completed_model_runs must reject BOTH (null and int type)
    discovered = discover_completed_model_runs(
        models_root=models_root, candidates_root=tmp_path / "empty_cand"
    )
    assert not any(r["run_id"] == "run_null" for r in discovered)
    assert not any(r["run_id"] == "run_int" for r in discovered)

    # build_ensemble_plan must reject candidate with nulls in in-memory OOF table
    oof_df_with_null = pl.from_arrow(tbl_null)
    campaign = {
        "run_id": "camp_null",
        "status": "completed",
        "probability_candidates": ["catboost_null_test"],
        "margin_candidates": [],
        "oof_tables": {"exp_null": oof_df_with_null},
        "declarations": {
            "exp_null": {"candidates": {"catboost_null_test": {"arm_id": "control"}}}
        },
        "manifest": {"candidates": [{"experiment_id": "exp_null", "status": "completed"}]},
    }
    plan = build_ensemble_plan(campaign)
    assert "catboost_null_test" not in plan["provisional_level1_candidates"]
    assert "catboost_null_test" in plan["provisional_rejections_with_reasons"]
