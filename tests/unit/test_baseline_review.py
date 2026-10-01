from __future__ import annotations

import hashlib
import importlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def reader():
    try:
        return importlib.import_module("project.experiments.baseline_review").load_baseline_review
    except ModuleNotFoundError:
        pytest.fail("Validated baseline review reader is missing")


def write_json(path, payload):
    path.write_text(json.dumps(payload))


@pytest.fixture
def artifact(tmp_path):
    config_dir = tmp_path / "configs/experiments"
    config_dir.mkdir(parents=True)
    config_bytes = (ROOT / "configs/experiments/baselines.yaml").read_bytes()
    (config_dir / "baselines.yaml").write_bytes(config_bytes)
    config = yaml.safe_load(config_bytes)
    prepared = tmp_path / "outputs/prepared"
    splits = tmp_path / "outputs/splits"
    prepared.mkdir(parents=True)
    splits.mkdir()
    manifest_rows = []
    for r, seed in enumerate([42, 43, 44]):
        for b in range(4):
            manifest_rows.append({"building_id": b, "repeat": r, "seed": seed, "split": "dev", "fold": b})
        manifest_rows.append({"building_id": 4, "repeat": r, "seed": seed, "split": "vault", "fold": None})
    manifest = pl.DataFrame(manifest_rows)
    manifest.write_parquet(splits / "building_folds_repeated.parquet")
    rows = pl.DataFrame({"building_id": [b for b in range(5) for _ in range(2)], "timestamp": [datetime.fromisoformat(f"2022-01-03T{h:02}:00:00") for _ in range(5) for h in (8, 9)], "anomaly": [0, 1] * 5})
    rows.write_parquet(prepared / "train_joined.parquet")
    hashes = {
        "data_sha256": hashlib.sha256((prepared / "train_joined.parquet").read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256((splits / "building_folds_repeated.parquet").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    run = tmp_path / "outputs/runs/20260101-test"
    run.mkdir(parents=True)
    (run / "baselines.yaml").write_bytes(config_bytes)
    records = []
    for candidate, metadata in config["models"].items():
        for r, seed in enumerate([42, 43, 44]):
            for index, row in enumerate(rows.filter(pl.col("building_id") < 4).iter_rows(named=True)):
                records.append({
                    "candidate_id": candidate,
                    **{k: metadata[k] for k in ("recipe_id", "model_family", "model_config_id")},
                    "row_index": index,
                    "repeat": r,
                    "split_seed": seed,
                    "fold": row["building_id"],
                    "building_id": row["building_id"],
                    "timestamp": row["timestamp"],
                    "target": row["anomaly"],
                    "prediction": 0.5 if candidate == "dummy_prior" else (0.2 + 0.6 * row["anomaly"]),
                    **hashes,
                    "run_id": run.name,
                    "execution_mode": "full",
                    "non_comparative": False,
                })
    pl.DataFrame(records).write_parquet(run / "oof_predictions.parquet")
    write_json(run / "run_status.json", {"status": "completed", "run_id": run.name})
    candidate_metrics = {}
    for c in config["models"]:
        candidate_metrics[c] = {
            "pooled": {
                "roc_auc": 0.5 if c == "dummy_prior" else 1.0,
                "average_precision": 0.5 if c == "dummy_prior" else 1.0,
                "log_loss": 0.6931471805599453 if c == "dummy_prior" else float(-math.log(0.8)),
                "brier_score": 0.25 if c == "dummy_prior" else float((0.2)**2),
            },
            "repeat_std": {
                "roc_auc": 0.0,
                "average_precision": 0.0,
                "log_loss": 0.0,
                "brier_score": 0.0,
            },
            "wall_time_seconds": 1.25,
            "repeats": [
                {
                    "repeat": r,
                    "split_seed": seed,
                    "roc_auc": 0.5 if c == "dummy_prior" else 1.0,
                    "average_precision": 0.5 if c == "dummy_prior" else 1.0,
                    "log_loss": 0.6931471805599453 if c == "dummy_prior" else float(-math.log(0.8)),
                    "brier_score": 0.25 if c == "dummy_prior" else float((0.2)**2),
                }
                for r, seed in enumerate([42, 43, 44])
            ],
            "folds": [
                {
                    "repeat": r,
                    "split_seed": seed,
                    "fold": f,
                    "roc_auc": 0.5 if c == "dummy_prior" else 1.0,
                    "average_precision": 0.5 if c == "dummy_prior" else 1.0,
                    "log_loss": 0.6931471805599453 if c == "dummy_prior" else float(-math.log(0.8)),
                    "brier_score": 0.25 if c == "dummy_prior" else float((0.2)**2),
                }
                for r, seed in enumerate([42, 43, 44])
                for f in range(4)
            ],
        }
    write_json(run / "metrics.json", {
        "run_id": run.name, "mode": "full", "non_comparative": False,
        "fold_ids": [0, 1, 2, 3],
        "repeat_ids": [0, 1, 2],
        "repeat_count": 3,
        "split_seeds": [42, 43, 44],
        "development_oof_row_count": 8,
        "oof_record_count": 72,
        "metric_aggregation": "equal_mean_of_repeat_pooled_metrics",
        "candidate_metrics": candidate_metrics,
    })
    write_json(run / "resource_profile.json", {
        "run_id": run.name, "mode": "full", "non_comparative": False,
        "model_fit_count": 36,
        "fold_ids": [0, 1, 2, 3],
        "repeat_ids": [0, 1, 2],
        "repeat_count": 3,
        "split_seeds": [42, 43, 44],
    })
    (run / "model_card.md").write_text("Baseline model card")
    for candidate in config["models"]:
        (run / f"model_card_{candidate}.md").write_text(f"Model card for {candidate}")
    budget = yaml.safe_load((ROOT / "configs/experiments/baseline_execution_budget.yaml").read_bytes())
    # Keep legacy bounded artifacts independent of the current execution mode.
    budget["max_rss_bytes"] = 4 * 1024**3
    budget.pop("rss_limit_enforced", None)
    write_json(run / "baseline_execution_budget.json", budget)
    write_json(run / "guard_observations.json", {
        "run_id": run.name, "status": "completed", "reason": None,
        "elapsed_seconds": 5.0, "peak_process_tree_rss_bytes": 100 * 1024**2,
        "cpu_threads": 4, "max_rss_bytes": 4 * 1024**3, "max_wall_seconds": 2700,
        "worker_exitcode": 0,
    })
    return tmp_path, run


def test_no_artifact_is_not_run(tmp_path):
    result = reader()(tmp_path)
    assert result["status"] == "not_run"
    assert result["reason"]


def test_actual_oof_metrics_and_population_are_recomputed(artifact):
    root, run = artifact
    result = reader()(root, run)
    assert set(result) == {
        "status", "run_id", "run_path", "metrics", "fold_metrics", "resources",
        "repeat_metrics", "repeat_count", "split_seeds",
        "reference_candidate_id", "positive_prevalence", "development_rows",
        "development_buildings", "fold_count",
    }
    assert result["status"] == "completed"
    assert result["run_path"] == run.relative_to(root).as_posix()
    assert result["development_rows"] == 8
    assert result["development_buildings"] == 4
    assert result["positive_prevalence"] == 0.5
    assert result["fold_count"] == 4
    metrics = result["metrics"]
    assert metrics.columns == [
        "candidate_id", "model_family", "roc_auc", "average_precision",
        "log_loss", "brier_score", "fold_roc_auc_mean", "fold_roc_auc_std",
        "fit_wall_seconds", "repeat_roc_auc_mean", "repeat_roc_auc_std",
    ]
    assert metrics.filter(pl.col("candidate_id") == "dummy_prior")["roc_auc"][0] == 0.5
    assert result["fold_metrics"].height == 36
    assert metrics.filter(pl.col("candidate_id") == "logistic_l2")["log_loss"][0] == pytest.approx(-math.log(0.8))


def telemetry_artifact(run):
    budget_path = run / "baseline_execution_budget.json"
    budget = json.loads(budget_path.read_text())
    budget.update(max_rss_bytes=None, rss_limit_enforced=False)
    write_json(budget_path, budget)
    guard_path = run / "guard_observations.json"
    guard = json.loads(guard_path.read_text())
    guard.update(max_rss_bytes=None, rss_limit_enforced=False, peak_process_tree_rss_bytes=5 * 1024**3)
    write_json(guard_path, guard)
    return guard


def test_reader_accepts_approved_telemetry_only_large_rss(artifact):
    root, run = artifact
    telemetry_artifact(run)
    result = reader()(root, run)
    assert result["status"] == "completed"
    assert result["resources"]["execution_guard"]["peak_process_tree_rss_bytes"] == 5 * 1024**3
    assert result["metrics"].filter(pl.col("candidate_id") == "dummy_prior")["roc_auc"][0] == 0.5


@pytest.mark.parametrize("field,value", [
    ("rss_limit_enforced", True), ("rss_limit_enforced", 0),
    ("rss_limit_enforced", "false"), ("max_rss_bytes", 4 * 1024**3),
    ("peak_process_tree_rss_bytes", None), ("peak_process_tree_rss_bytes", 0),
    ("peak_process_tree_rss_bytes", -1), ("peak_process_tree_rss_bytes", True),
    ("peak_process_tree_rss_bytes", float("inf")), ("peak_process_tree_rss_bytes", float("nan")),
    ("elapsed_seconds", 2700), ("cpu_threads", 5),
])
def test_reader_rejects_forged_or_invalid_telemetry_observation(artifact, field, value):
    root, run = artifact
    guard = telemetry_artifact(run)
    guard[field] = value
    write_json(run / "guard_observations.json", guard)
    with pytest.raises(ValueError):
        reader()(root, run)
    assert reader()(root)["status"] == "not_run"


def test_reader_requires_explicit_telemetry_guard_mode(artifact):
    root, run = artifact
    guard = telemetry_artifact(run)
    del guard["rss_limit_enforced"]
    write_json(run / "guard_observations.json", guard)
    with pytest.raises(ValueError):
        reader()(root, run)


@pytest.mark.parametrize("corruption", ["smoke", "in_progress", "stale", "missing_row", "wrong_label", "missing_candidate", "config", "split", "budget", "guard_failed", "guard_cap", "guard_missing", "current_labels"])
def test_invalid_explicit_run_raises_and_discovery_skips(artifact, corruption):
    root, run = artifact
    if corruption in {"smoke", "in_progress"}:
        filename = "metrics.json" if corruption == "smoke" else "run_status.json"
        payload = json.loads((run / filename).read_text())
        payload.update({"mode": "smoke", "non_comparative": True} if corruption == "smoke" else {"status": "in_progress"})
        write_json(run / filename, payload)
    elif corruption in {"budget", "guard_failed", "guard_cap", "guard_missing"}:
        filename = "baseline_execution_budget.json" if corruption == "budget" else "guard_observations.json"
        if corruption == "guard_missing":
            (run / filename).unlink()
        else:
            payload = json.loads((run / filename).read_text())
            payload.update({"approval_status": "pending"} if corruption == "budget" else {"status": "failed"} if corruption == "guard_failed" else {"peak_process_tree_rss_bytes": 5 * 1024**3})
            write_json(run / filename, payload)
    elif corruption == "current_labels":
        path = root / "outputs/prepared/train_joined.parquet"
        pl.read_parquet(path).with_columns((1 - pl.col("anomaly")).alias("anomaly")).write_parquet(path)
        # A forged updated data hash must not hide label drift.
        path = run / "oof_predictions.parquet"
        pl.read_parquet(path).with_columns(pl.lit(hashlib.sha256((root / "outputs/prepared/train_joined.parquet").read_bytes()).hexdigest()).alias("data_sha256")).write_parquet(path)
    elif corruption == "config":
        with (root / "configs/experiments/baselines.yaml").open("a") as stream:
            stream.write("\n# changed\n")
    elif corruption == "split":
        path = root / "outputs/splits/building_folds_repeated.parquet"
        pl.read_parquet(path).with_columns(pl.when(pl.col("split") == "dev").then((pl.col("fold") + 1) % 4).otherwise(pl.col("fold")).alias("fold")).write_parquet(path)
    else:
        path = run / "oof_predictions.parquet"
        frame = pl.read_parquet(path)
        if corruption == "stale":
            frame = frame.with_columns(pl.lit("stale").alias("data_sha256"))
        elif corruption == "wrong_label":
            frame = frame.with_columns((1 - pl.col("target")).alias("target"))
        elif corruption == "missing_candidate":
            frame = frame.filter(pl.col("candidate_id") != "dummy_prior")
        else:
            frame = frame.slice(1)
        frame.write_parquet(path)
    with pytest.raises(ValueError):
        reader()(root, run)
    assert reader()(root)["status"] == "not_run"


def test_latest_invalid_run_does_not_hide_valid_completed_run(artifact):
    root, run = artifact
    latest = run.with_name("20270101-smoke")
    shutil.copytree(run, latest)
    write_json(latest / "run_status.json", {"status": "in_progress", "run_id": latest.name})
    assert reader()(root)["run_id"] == run.name


@pytest.mark.parametrize("change", ["feature", "target", "positive_label", "primary_metric"])
def test_matching_hashes_cannot_admit_a_nonbaseline_contract(artifact, change):
    root, run = artifact
    path = root / "configs/experiments/baselines.yaml"
    config = yaml.safe_load(path.read_bytes())
    if change == "feature":
        config["features"]["numeric"].append("unapproved_engineered_feature")
    elif change == "target":
        config["target"] = "other_target"
        data_path = root / "outputs/prepared/train_joined.parquet"
        pl.read_parquet(data_path).with_columns(pl.col("anomaly").alias("other_target")).write_parquet(data_path)
    else:
        config[change] = 0 if change == "positive_label" else "average_precision"
    content = yaml.safe_dump(config, sort_keys=False).encode()
    path.write_bytes(content)
    (run / "baselines.yaml").write_bytes(content)
    oof_path = run / "oof_predictions.parquet"
    pl.read_parquet(oof_path).with_columns(
        pl.lit(hashlib.sha256(content).hexdigest()).alias("config_sha256"),
        pl.lit(hashlib.sha256((root / "outputs/prepared/train_joined.parquet").read_bytes()).hexdigest()).alias("data_sha256"),
    ).write_parquet(oof_path)
    with pytest.raises(ValueError):
        reader()(root, run)
