"""Bounded SMOKE execution tested only with synthetic projected Parquet data."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_REFERENCE = "configs/experiments/campaign_smoke_v1.yaml"
EXPERIMENT = "feat_missingness_indicators"


def api():
    name = "project.experiments.campaign_smoke"
    assert importlib.util.find_spec(name), "real bounded SMOKE runner API is missing"
    return importlib.import_module(name)


def config():
    path = ROOT / CONFIG_REFERENCE
    assert path.exists(), "versioned smoke config is missing"
    return yaml.safe_load(path.read_text())


@pytest.fixture
def synthetic_project(tmp_path):
    """Persist a tiny 200-building/4x3 contract, never use repository real data."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path
    for directory in ("configs/experiments", "outputs/splits", "outputs/prepared", "outputs/runs"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for reference in ("configs/experiments/feature_campaign.yaml", "configs/experiments/baselines.yaml",
                      "configs/resource_budget.yaml"):
        path = root / reference
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / reference).read_bytes())
    settings = config()
    settings.update(launch_review_status="approved", train_rows=60, validation_rows=40,
                    batch_size=16)
    (root / CONFIG_REFERENCE).write_text(yaml.safe_dump(settings))
    rows = []
    metadata = {"repeats": 3, "n_dev_folds": 4, "split_seeds": [42, 43, 44],
                "vault_building_ids": list(range(160, 200)), "per_repeat": [],
                "source_manifest": {"path": "outputs/splits/building_folds.parquet", "sha256": "historical"}}
    for repeat in range(3):
        order = np.random.RandomState(42 + repeat).permutation(160).tolist()
        fold_by_id = {building: index // 40 for index, building in enumerate(order)}
        part = [{"building_id": b, "repeat": repeat, "seed": 42 + repeat,
                 "split": "dev" if b < 160 else "vault", "fold": fold_by_id.get(b),
                 "row_count": 4, "anomaly_count": 2, "normal_count": 2,
                 "anomaly_prevalence": 0.5, "support_stratum": 0} for b in range(200)]
        rows.extend(part)
        reference = f"outputs/splits/building_folds_r{repeat}_seed{42 + repeat}.parquet"
        pq.write_table(pa.Table.from_pylist(part), root / reference)
        summary = lambda fold: {"fold": fold, "building_count": 40, "row_count": 160,
                                "positive_rows": 80, "negative_rows": 80}
        metadata["per_repeat"].append({"repeat": repeat, "seed": 42 + repeat, "path": reference,
            "sha256": hashlib.sha256((root / reference).read_bytes()).hexdigest(),
            "class_support": {"vault": summary(None), "development_folds": [summary(f) for f in range(4)]},
            "per_building_support": [{k: v for k, v in r.items() if k not in {"repeat", "seed"}} for r in part]})
    master_reference = "outputs/splits/building_folds_repeated.parquet"
    pq.write_table(pa.Table.from_pylist(rows), root / master_reference)
    digest = hashlib.sha256((root / master_reference).read_bytes()).hexdigest()
    metadata.update(family_sha256=digest, artifacts={"building_folds_path": master_reference,
                                                   "building_folds_sha256": digest})
    (root / "outputs/splits/split_metadata_repeated.json").write_text(json.dumps(metadata))
    baseline = yaml.safe_load((ROOT / "configs/experiments/baselines.yaml").read_text())
    sample = []
    for building in range(200):
        for row in range(4):
            timestamp = datetime(2024, 1, 1, 0, row, tzinfo=UTC).replace(tzinfo=None)
            record = {name: float(row + building) for name in baseline["features"]["numeric"]}
            # train.py:67-70 requires Datetime and Monday=0 calendar semantics.
            # The previous fixture incorrectly made calendar fields row+building.
            record.update(building_id=building, timestamp=timestamp,
                          hour=timestamp.hour, weekday=timestamp.weekday(),
                          month=timestamp.month, year=timestamp.year,
                          anomaly=row % 2, primary_use="office", unused_secret="not projected",
                          meter_reading=None if row == 0 else float(row + building))
            sample.append(record)
    pq.write_table(pa.Table.from_pylist(sample), root / "outputs/prepared/train_joined.parquet",
                   row_group_size=32)
    return root, settings


def test_checked_in_config_is_smoke_only_and_parent_review_approved():
    settings = config()
    report = api().validate_smoke_config(ROOT, settings, EXPERIMENT)
    assert report["config_valid"] is True
    assert settings["launch_review_status"] == "approved"
    assert report["executable"] is True
    assert report["status"] == "admitted_smoke_only"
    assert settings["scope"] == "campaign_smoke"
    assert settings["source_design_sha256"] == hashlib.sha256(
        (ROOT / "configs/experiments/feature_campaign.yaml").read_bytes()).hexdigest()


def test_synthetic_pending_parent_review_remains_blocked():
    settings = config() | {"launch_review_status": "pending"}
    report = api().validate_smoke_config(ROOT, settings, EXPERIMENT)
    assert report["config_valid"] is True
    assert report["executable"] is False
    assert report["status"] == "blocked_parent_review"


@pytest.mark.parametrize("key,value", [
    ("maximum_cardinality", 10001), ("cpu_threads", 3), ("max_ram_gb", 2),
    ("max_wall_time_seconds", 121), ("max_model_fits", 4), ("max_hpo_trials", 1),
    ("seed", 43), ("repeat_id", 1), ("fold_id", True), ("device", "cuda"),
    ("execution_authorized", False), ("user_approval_status", "pending"),
    ("source_design_sha256", "0" * 64), ("scope", "baseline_only"),
    ("train_rows", 9999), ("batch_size", 10001), ("max_model_fits", 1),
])
def test_invalid_smoke_config_denied_before_worker(synthetic_project, monkeypatch, key, value):
    root, settings = synthetic_project
    settings[key] = value
    (root / CONFIG_REFERENCE).write_text(yaml.safe_dump(settings))
    calls = []
    monkeypatch.setattr(api(), "run_smoke_guarded", lambda *args: calls.append(args))
    report = api().launch_smoke(root, CONFIG_REFERENCE, EXPERIMENT)
    assert report["returncode"] == 2
    assert not calls
    assert list((root / "outputs/runs").iterdir()) == []


def test_selection_and_design_checksum_are_explicit(synthetic_project):
    root, settings = synthetic_project
    assert not api().validate_smoke_config(root, settings, "model_extratrees_core_features")["config_valid"]
    bad = copy.deepcopy(settings)
    bad["source_design"] = "configs/experiments/baselines.yaml"
    assert not api().validate_smoke_config(root, bad, EXPERIMENT)["config_valid"]


def test_projected_bounded_sample_uses_original_grouped_membership(synthetic_project, monkeypatch):
    import polars as pl
    import pyarrow.dataset as ds

    root, settings = synthetic_project
    manifest = pl.read_parquet(root / "outputs/splits/building_folds_repeated.parquet")
    calls = []
    original = ds.dataset

    class DatasetProxy:
        def __init__(self, dataset):
            self.schema = dataset.schema
            self._dataset = dataset

        def scanner(self, **kwargs):
            calls.append(kwargs)
            assert "unused_secret" not in kwargs["columns"]
            assert kwargs["filter"] is not None
            assert kwargs["use_threads"] is False
            assert kwargs["batch_readahead"] == kwargs["fragment_readahead"] == 0
            return self._dataset.scanner(**kwargs)

    monkeypatch.setattr(ds, "dataset", lambda *args, **kwargs: DatasetProxy(original(*args, **kwargs)))
    sample = api().load_grouped_sample(root / settings["data_reference"], manifest, settings)
    again = api().load_grouped_sample(root / settings["data_reference"], manifest, settings)
    train, valid = sample["train"], sample["validation"]
    assert len(train) == 60 and len(valid) == 40
    assert len(train) + len(valid) <= settings["maximum_cardinality"]
    assert set(train.building_id).isdisjoint(set(valid.building_id))
    assert not (set(train.building_id) | set(valid.building_id)) & set(range(160, 200))
    dev = manifest.filter((pl.col("repeat") == 0) & (pl.col("split") == "dev"))
    val_ids = set(dev.filter(pl.col("fold") == settings["fold_id"])["building_id"])
    assert set(valid.building_id) <= val_ids
    assert set(train.building_id).isdisjoint(val_ids)
    assert "unused_secret" not in train
    assert train.equals(again["train"]) and valid.equals(again["validation"])
    assert calls


def test_no_viable_sample_classes_has_no_fallback(synthetic_project):
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    root, settings = synthetic_project
    path = root / settings["data_reference"]
    frame = pq.read_table(path)
    frame = frame.set_column(frame.schema.get_field_index("anomaly"), "anomaly",
                             pa.array([0] * frame.num_rows, type=pa.int64()))
    pq.write_table(frame, path)
    manifest = pl.read_parquet(root / "outputs/splits/building_folds_repeated.parquet")
    with pytest.raises(ValueError, match="both classes"):
        api().load_grouped_sample(path, manifest, settings)


def test_real_two_arm_worker_with_synthetic_parquet(synthetic_project, monkeypatch):
    import pyarrow.parquet as pq

    root, settings = synthetic_project
    monkeypatch.setattr(api(), "host_preflight", lambda: {"timestamp_utc": "synthetic-test",
        "available_ram_bytes": 4 * 1024**3, "logical_cpus": 2})
    before = {str(p): p.read_bytes() for p in (root / "outputs/splits").iterdir()}
    report = api().launch_smoke(root, CONFIG_REFERENCE, EXPERIMENT)
    assert report["returncode"] == 0, report
    run = root / report["run_reference"]
    status = json.loads((run / "run_status.json").read_text())
    assert status["status"] == "smoke_completed"
    assert status["mode"] == "smoke" and status["non_comparative"] is True
    assert status["n03_eligible"] is False
    assert status["model_fit_count"] == 2
    predictions = pq.read_table(run / "sample_predictions.parquet").to_pandas()
    assert len(predictions) == 2 * settings["validation_rows"]
    assert set(predictions.arm) == {"raw", "missingness"}
    assert predictions.probability.between(0, 1).all()
    selected = pq.read_table(run / "sample_selection.parquet").to_pandas()
    assert len(selected) == 100
    assert not selected.duplicated(["building_id", "timestamp"]).any()
    metrics = json.loads((run / "metrics.json").read_text())
    assert set(metrics["arms"]) == {"raw", "missingness"}
    assert metrics["parameters_identical_both_arms"] is True
    assert metrics["smoke_only_adjustments"]
    evidence = api().validate_smoke_artifacts(run, settings)
    assert evidence["model_fit_count"] == 2
    for reference, digest in status["artifact_sha256"].items():
        assert hashlib.sha256((run / reference).read_bytes()).hexdigest() == digest
    guard = json.loads((run / "guard_observations.json").read_text())
    assert guard["rss_limit_enforced"] is True
    assert guard["peak_process_tree_rss_bytes"] <= 1024**3
    assert not (root / "outputs/runs/experiment_inventory.jsonl").exists()
    assert before == {str(p): p.read_bytes() for p in (root / "outputs/splits").iterdir()}


def test_preflight_remeasures_and_denies_ram_shortage(synthetic_project, monkeypatch):
    root, _ = synthetic_project
    monkeypatch.setattr(api(), "host_preflight", lambda: {"timestamp_utc": "synthetic-test",
        "available_ram_bytes": 1, "logical_cpus": 2})
    calls = []
    monkeypatch.setattr(api(), "run_smoke_guarded", lambda *args: calls.append(args))
    report = api().launch_smoke(root, CONFIG_REFERENCE, EXPERIMENT)
    assert report["returncode"] == 2
    assert "RAM" in str(report["errors"])
    assert not calls


def sleeping_worker(args):
    time.sleep(10)
    return 0


def descendant_worker(args):
    child = subprocess.Popen([sys.executable, "-B", "-c",
                              "import time; a=bytearray(400*1024**2); time.sleep(10)"])
    Path(args["run_dir"]).joinpath("descendant_pid").write_text(str(child.pid))
    time.sleep(10)
    return 0


def fake_completed_worker(args):
    Path(args["run_dir"]).joinpath("run_status.json").write_text('{"status":"completed"}')
    return 0


@pytest.mark.parametrize("worker,changes,reason", [
    (sleeping_worker, {"max_wall_time_seconds": 0.3}, "wall"),
    (descendant_worker, {"max_ram_gb": 0.5, "max_wall_time_seconds": 8}, "rss"),
    (fake_completed_worker, {}, "artifact"),
])
def test_guard_includes_startup_descendants_and_rejects_fake_completion(tmp_path, worker, changes, reason):
    settings = config() | changes
    result = api().run_smoke_guarded(worker, {"run_dir": str(tmp_path)}, tmp_path, settings)
    assert result != 0
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["status"] == "failed"
    assert reason in status["reason"]
    guard = json.loads((tmp_path / "guard_observations.json").read_text())
    assert guard["status"] == "failed"
    if worker is descendant_worker:
        from project.experiments.baseline_budget import process_tree_rss_bytes

        pid = int((tmp_path / "descendant_pid").read_text())
        assert process_tree_rss_bytes(pid)[0] == 0


def test_worker_receives_cpu_limits_without_gpu(synthetic_project, monkeypatch):
    root, _ = synthetic_project
    monkeypatch.setattr(api(), "host_preflight", lambda: {"timestamp_utc": "synthetic-test",
        "available_ram_bytes": 4 * 1024**3, "logical_cpus": 2})
    report = api().launch_smoke(root, CONFIG_REFERENCE, EXPERIMENT)
    assert report["returncode"] == 0
    profile = json.loads((root / report["run_reference"] / "resource_profile.json").read_text())
    assert profile["cpu_threads"] == 2 and profile["device"] == "cpu"
    assert profile["environment"]["OMP_NUM_THREADS"] == "2"
    assert profile["environment"]["CUDA_VISIBLE_DEVICES"] == ""


def test_transformations_fit_only_training_and_flags_are_preimputation(synthetic_project, monkeypatch):
    import polars as pl
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder
    from threadpoolctl import threadpool_limits

    root, settings = synthetic_project
    sample = api().load_grouped_sample(root / settings["data_reference"],
        pl.read_parquet(root / "outputs/splits/building_folds_repeated.parquet"), settings)
    sample["validation"] = sample["validation"].copy()
    sample["validation"]["primary_use"] = "validation_only"
    fit_rows = []
    encoded_fit_rows, unseen_encoded = [], []
    original = SimpleImputer.fit

    def capture_fit(self, frame, y=None):
        fit_rows.append(len(frame))
        if "meter_reading_missing" in frame:
            assert frame.meter_reading_missing.equals(frame.meter_reading.isna().astype(float))
        return original(self, frame, y)

    original_encoder_fit, original_transform = OrdinalEncoder.fit, OrdinalEncoder.transform

    def capture_encoder_fit(self, frame, y=None):
        encoded_fit_rows.append(len(frame))
        assert set(frame.primary_use) == {"office"}
        return original_encoder_fit(self, frame, y)

    def capture_transform(self, frame):
        output = original_transform(self, frame)
        if set(frame.primary_use) == {"validation_only"}:
            unseen_encoded.append((output == -1).all())
        return output

    monkeypatch.setattr(SimpleImputer, "fit", capture_fit)
    monkeypatch.setattr(OrdinalEncoder, "fit", capture_encoder_fit)
    monkeypatch.setattr(OrdinalEncoder, "transform", capture_transform)
    with threadpool_limits(limits=2):
        _, metrics = api()._fit_arms(sample, settings)
    assert fit_rows == [settings["train_rows"], settings["train_rows"]]
    assert encoded_fit_rows == [settings["train_rows"], settings["train_rows"]]
    assert unseen_encoded == [True, True]
    assert "meter_reading_missing" not in metrics["arms"]["raw"]["predictors"]
    assert "meter_reading_missing" in metrics["arms"]["missingness"]["predictors"]
    registry = api()._missingness_registry()
    assert registry.get("meter_reading_missing").source_columns == ("meter_reading",)
    assert registry.get("meter_reading_missing").transform_fit_boundary.value == "fixed"


def test_public_loader_rejects_expanded_batch_even_without_launch(synthetic_project):
    import polars as pl

    root, settings = synthetic_project
    settings["batch_size"] = 10001
    manifest = pl.read_parquet(root / "outputs/splits/building_folds_repeated.parquet")
    with pytest.raises(ValueError, match="batch"):
        api().load_grouped_sample(root / settings["data_reference"], manifest, settings)


def test_artifact_completion_checks_noncomparative_and_n03_flags(synthetic_project, monkeypatch):
    root, settings = synthetic_project
    monkeypatch.setattr(api(), "host_preflight", lambda: {"timestamp_utc": "synthetic-test",
        "available_ram_bytes": 4 * 1024**3, "logical_cpus": 2})
    report = api().launch_smoke(root, CONFIG_REFERENCE, EXPERIMENT)
    assert report["returncode"] == 0
    run = root / report["run_reference"]
    path = run / "worker_result.json"
    manifest = json.loads(path.read_text())
    manifest["n03_eligible"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="artifact"):
        api().validate_smoke_artifacts(run, settings)


def test_nonfinite_model_probabilities_never_become_completed_evidence(synthetic_project, monkeypatch):
    import numpy as np
    import polars as pl
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    root, settings = synthetic_project
    sample = api().load_grouped_sample(root / settings["data_reference"],
        pl.read_parquet(root / "outputs/splits/building_folds_repeated.parquet"), settings)
    monkeypatch.setattr(HistGradientBoostingClassifier, "predict_proba",
                        lambda self, frame: np.full((len(frame), 2), np.nan))
    with threadpool_limits(limits=2), pytest.raises(ValueError, match="finite"):
        api()._fit_arms(sample, settings)
    assert list((root / "outputs/runs").iterdir()) == []


def failing_descendant_worker(args):
    child = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(10)"])
    Path(args["run_dir"]).joinpath("descendant_pid").write_text(str(child.pid))
    return 17


def test_nonzero_worker_reaps_its_live_child(tmp_path):
    from project.experiments.baseline_budget import process_tree_rss_bytes

    assert api().run_smoke_guarded(failing_descendant_worker, {"run_dir": str(tmp_path)},
                                  tmp_path, config()) != 0
    pid = int((tmp_path / "descendant_pid").read_text())
    try:
        assert process_tree_rss_bytes(pid)[0] == 0
        assert json.loads((tmp_path / "run_status.json").read_text())["status"] == "failed"
    finally:
        # A red regression must not leave the intentionally spawned child alive.
        import os
        import signal

        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def stalled_verification_entry(state):
    module = api()
    original = module._artifact_manifest

    def stalled_verify(run):
        time.sleep(10)
        return original(run)

    module._artifact_manifest = stalled_verify
    module._finalization_entry(state)


def stalled_completion_entry(state):
    module = api()
    original = module._write_json

    def stalled_write(path, value):
        if path.name == module.CANDIDATE_COMPLETION and value.get("status") == "pending_supervisor_acceptance":
            time.sleep(10)
        return original(path, value)

    module._write_json = stalled_write
    module._finalization_entry(state)


@pytest.mark.parametrize("entry", [stalled_verification_entry, stalled_completion_entry])
def test_finalization_stall_is_interrupted_by_total_wall(synthetic_project, monkeypatch, entry):
    root, settings = synthetic_project
    settings["max_wall_time_seconds"] = 6
    (root / CONFIG_REFERENCE).write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(api(), "host_preflight", lambda: {"timestamp_utc": "synthetic-test",
        "available_ram_bytes": 4 * 1024**3, "logical_cpus": 2})
    monkeypatch.setattr(api(), "_finalization_entry", entry)
    started = time.monotonic()
    report = api().launch_smoke(root, CONFIG_REFERENCE, EXPERIMENT)
    assert time.monotonic() - started < 7
    assert report["returncode"] != 0
    status = json.loads((root / report["run_reference"] / "run_status.json").read_text())
    assert status["status"] == "failed" and "wall" in status["reason"]


@pytest.mark.parametrize("column,value", [("timestamp", "not-a-timestamp"), ("timestamp", 17),
                                        ("hour", 23), ("weekday", 1), ("month", 12), ("year", 2000)])
def test_bad_timestamp_calendar_fail_before_fit_without_retry(synthetic_project, monkeypatch, column, value):
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sklearn.ensemble import HistGradientBoostingClassifier

    root, settings = synthetic_project
    path = root / settings["data_reference"]
    data = pq.read_table(path)
    data = data.set_column(data.schema.get_field_index(column), column,
                           pa.array([value] * data.num_rows))
    pq.write_table(data, path)
    fits = []
    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", lambda *args: fits.append(args))
    with pytest.raises(ValueError, match="timestamp|calendar"):
        sample = api().load_grouped_sample(path,
            pl.read_parquet(root / "outputs/splits/building_folds_repeated.parquet"), settings)
        api()._fit_arms(sample, settings)
    assert fits == []
    assert list((root / "outputs/runs").iterdir()) == []


@pytest.mark.parametrize("field,value", [("target", "other"), ("positive_label", True),
    ("primary_metric", "accuracy"), ("guardrails", []), ("seed", 43), ("folds", 3),
    ("split_seeds", [42]), ("split_manifest", "outputs/splits/building_folds.parquet"),
    ("model_params", {"early_stopping": True})])
def test_corrupt_frozen_baseline_denied_before_worker(synthetic_project, monkeypatch, field, value):
    root, _ = synthetic_project
    path = root / "configs/experiments/baselines.yaml"
    baseline = yaml.safe_load(path.read_text())
    if field == "model_params":
        baseline["models"]["hist_gradient_boosting"]["params"] = value
    else:
        baseline[field] = value
    path.write_text(yaml.safe_dump(baseline))
    calls = []
    monkeypatch.setattr(api(), "host_preflight", lambda: {"timestamp_utc": "synthetic-test",
        "available_ram_bytes": 4 * 1024**3, "logical_cpus": 2})
    monkeypatch.setattr(api(), "run_smoke_guarded", lambda *args: calls.append(args))
    report = api().launch_smoke(root, CONFIG_REFERENCE, EXPERIMENT)
    assert report["returncode"] == 2
    assert not calls


@pytest.fixture
def synthetic_completed_run(synthetic_project, monkeypatch):
    root, settings = synthetic_project
    monkeypatch.setattr(api(), "host_preflight", lambda: {"timestamp_utc": "synthetic-test",
        "available_ram_bytes": 4 * 1024**3, "logical_cpus": 2})
    report = api().launch_smoke(root, CONFIG_REFERENCE, EXPERIMENT)
    assert report["returncode"] == 0, report
    return root / report["run_reference"], settings


@pytest.mark.parametrize("tamper", ["one_arm_target", "both_arm_targets", "selection_target",
                                  "roc_auc", "average_precision", "log_loss", "brier_score"])
def test_targets_and_metrics_tampering_rejected_even_with_refreshed_hashes(synthetic_completed_run, tamper):
    import pyarrow as pa
    import pyarrow.parquet as pq

    run, settings = synthetic_completed_run
    manifest = json.loads((run / "worker_result.json").read_text())
    if tamper in {"one_arm_target", "both_arm_targets"}:
        path = run / "sample_predictions.parquet"
        frame = pq.read_table(path).to_pandas()
        mask = frame.arm == "raw" if tamper == "one_arm_target" else frame.arm.notna()
        frame.loc[mask, "anomaly"] = 1 - frame.loc[mask, "anomaly"]
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    elif tamper == "selection_target":
        path = run / "sample_selection.parquet"
        frame = pq.read_table(path).to_pandas()
        assert "anomaly" in frame, "selected rows must retain canonical target evidence"
        mask = frame.sample_partition == "validation"
        frame.loc[mask, "anomaly"] = 1 - frame.loc[mask, "anomaly"]
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
        lineage_path = run / "lineage.json"
        lineage = json.loads(lineage_path.read_text())
        lineage["selected_sample_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        lineage_path.write_text(json.dumps(lineage))
        manifest["artifact_sha256"][lineage_path.name] = hashlib.sha256(lineage_path.read_bytes()).hexdigest()
    else:
        path = run / "metrics.json"
        metrics = json.loads(path.read_text())
        metrics["arms"]["raw"][tamper] += 0.123
        path.write_text(json.dumps(metrics))
    manifest["artifact_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "worker_result.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="target|metric"):
        api().validate_smoke_artifacts(run, settings)


def test_validated_baseline_hash_bound_to_worker_launch(synthetic_project, monkeypatch):
    root, _ = synthetic_project
    monkeypatch.setattr(api(), "host_preflight", lambda: {"timestamp_utc": "synthetic-test",
        "available_ram_bytes": 4 * 1024**3, "logical_cpus": 2})
    original_guard = api().run_smoke_guarded

    def change_after_admission(target, args, run, settings):
        path = root / "configs/experiments/baselines.yaml"
        baseline = yaml.safe_load(path.read_text())
        baseline["note"] = "Semantically valid but not the admitted bytes"
        path.write_text(yaml.safe_dump(baseline))
        return original_guard(target, args, run, settings)

    monkeypatch.setattr(api(), "run_smoke_guarded", change_after_admission)
    report = api().launch_smoke(root, CONFIG_REFERENCE, EXPERIMENT)
    assert report["returncode"] != 0
    run = root / report["run_reference"]
    assert json.loads((run / "run_status.json").read_text())["status"] == "failed"
    assert not (run / "metrics.json").exists()
    assert not (run / "sample_predictions.parquet").exists()


def test_stalled_admission_hash_obeys_launch_deadline(synthetic_project, monkeypatch):
    root, settings = synthetic_project
    settings["max_wall_time_seconds"] = 0.3
    (root / CONFIG_REFERENCE).write_text(yaml.safe_dump(settings))
    original = api()._read_yaml_hash

    def stalled_source(root, reference, label):
        if reference == "configs/experiments/feature_campaign.yaml":
            time.sleep(10)
        return original(root, reference, label)

    monkeypatch.setattr(api(), "_read_yaml_hash", stalled_source)
    started = time.monotonic()
    report = api().launch_smoke(root, CONFIG_REFERENCE, EXPERIMENT)
    assert time.monotonic() - started < 1
    assert report["returncode"] != 0
    assert "wall" in str(report)
    assert list((root / "outputs/runs").iterdir()) == []


def sleeping_descendant_worker(args):
    child = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(10)"])
    Path(args["run_dir"]).joinpath("descendant_pid").write_text(str(child.pid))
    time.sleep(10)
    return 0


def test_monitor_error_still_terminates_live_tree_and_reports_failure(tmp_path, monkeypatch):
    original = api().process_tree_rss_bytes
    failed = False

    def fail_once(pid):
        nonlocal failed
        if not failed and (tmp_path / "descendant_pid").exists():
            failed = True
            raise OSError("injected monitor failure")
        return original(pid)

    monkeypatch.setattr(api(), "process_tree_rss_bytes", fail_once)
    code = api().run_smoke_guarded(sleeping_descendant_worker, {"run_dir": str(tmp_path)},
                                   tmp_path, config())
    assert code != 0
    pid = int((tmp_path / "descendant_pid").read_text())
    assert original(pid)[0] == 0
    assert json.loads((tmp_path / "run_status.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("when", ["before_snapshot", "before_fit"])
def test_worker_never_rebaselines_admitted_inputs(synthetic_project, monkeypatch, when):
    root, settings = synthetic_project
    run = root / "outputs/runs/binding-regression"
    run.mkdir()
    module = api()
    admitted = module.validate_smoke_config(root, settings, EXPERIMENT)["validated_input_sha256"]
    admitted[CONFIG_REFERENCE] = hashlib.sha256((root / CONFIG_REFERENCE).read_bytes()).hexdigest()
    args = {"root": str(root), "run_dir": str(run), "config_reference": CONFIG_REFERENCE,
            "experiment_id": EXPERIMENT, "settings": settings,
            "validated_input_sha256": admitted, "preflight": {"timestamp_utc": "synthetic-test"}}
    original_split, original_sample = module.validate_split_contract, module.load_grouped_sample
    fits = []

    def mutate_baseline():
        path = root / "configs/experiments/baselines.yaml"
        baseline = yaml.safe_load(path.read_text())
        baseline["note"] = "Changed after initial worker admission binding"
        path.write_text(yaml.safe_dump(baseline))

    def split_then_mutate(*args):
        result = original_split(*args)
        mutate_baseline()
        return result

    def sample_then_mutate(*args):
        result = original_sample(*args)
        mutate_baseline()
        return result

    monkeypatch.setattr(module, "validate_split_contract" if when == "before_snapshot" else "load_grouped_sample",
                        split_then_mutate if when == "before_snapshot" else sample_then_mutate)

    def unexpected_fit(*args):
        fits.append(args)
        pytest.fail("fit reached after admitted baseline bytes changed")

    monkeypatch.setattr(module, "_fit_arms", unexpected_fit)
    with pytest.raises(ValueError, match="admitted|snapshot|hash|binding"):
        module._smoke_worker(args)
    assert fits == []
    assert not (run / "metrics.json").exists()


def stall_after_guard_entry(state):
    module = api()
    original = module._write_json

    def write_then_stall(path, value):
        original(path, value)
        if path.name == "guard_observations.json":
            time.sleep(10)

    module._write_json = write_then_stall
    module._finalization_entry(state)


@pytest.mark.parametrize("reporter", ["stall", "fail"])
def test_failed_launch_retained_bundle_has_no_authoritative_success(synthetic_project, monkeypatch, reporter):
    root, settings = synthetic_project
    settings["max_wall_time_seconds"] = 6
    (root / CONFIG_REFERENCE).write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(api(), "host_preflight", lambda: {"timestamp_utc": "synthetic-test",
        "available_ram_bytes": 4 * 1024**3, "logical_cpus": 2})
    monkeypatch.setattr(api(), "_finalization_entry", stall_after_guard_entry)
    program = "import time; time.sleep(10)" if reporter == "stall" else "raise SystemExit(17)"
    monkeypatch.setattr(api(), "FAILURE_REPORT_PROGRAM", program)
    report = api().launch_smoke(root, CONFIG_REFERENCE, EXPERIMENT)
    assert report["returncode"] != 0
    run = root / report["run_reference"]
    guard = json.loads((run / "guard_observations.json").read_text())
    assert guard["status"] != "completed"
    assert guard["artifact_validation_passed"] is not True
    assert not (run / "supervisor_acceptance.json").exists()
    assert hasattr(api(), "read_smoke_completion"), "completion reader must require supervisor acceptance"
    with pytest.raises(ValueError, match="supervisor|acceptance|provisional"):
        api().read_smoke_completion(run)
