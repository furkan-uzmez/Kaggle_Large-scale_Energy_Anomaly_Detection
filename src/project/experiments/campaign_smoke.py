"""Two-fit CPU SMOKE with bounded sampling and a live process-tree guard.

All data/schema/hash/model/artifact work runs in the spawned guarded worker.
The source design remains unauthorized for full execution. A separate versioned
config and explicit parent launch review authorize only this non-comparative path.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from project.experiments.baseline_budget import (
    BASELINE_MODELS,
    BASELINE_PARAMS,
    THREAD_ENVIRONMENT,
    process_tree_rss_bytes,
)
from project.experiments.baseline_budget import (
    _worker as baseline_worker_entry,
)
from project.experiments.campaign_budget import (
    validate_admission,
    validate_split_contract,
)
from project.experiments.campaign_ledger import (
    file_sha256,
    referenced_file,
    validate_predictors,
)

EXPERIMENT = "feat_missingness_indicators"
SOURCE_DESIGN = "configs/experiments/feature_campaign.yaml"
SMOKE_CONFIG = "configs/experiments/campaign_smoke_v1.yaml"
HGB_PARAMS = {"max_iter": 20, "max_leaf_nodes": 7, "min_samples_leaf": 10,
              "learning_rate": 0.1, "early_stopping": False, "random_state": 42,
              "categorical_features": None}
NUMERIC = ("meter_reading", "square_feet", "year_built", "floor_count", "hour",
           "weekday", "month", "year")
CATEGORICAL = ("primary_use",)
CORE = NUMERIC + CATEGORICAL
PROJECTED = ("building_id", "timestamp", "anomaly") + CORE
ARTIFACTS = {"sample_selection.parquet", "sample_predictions.parquet", "metrics.json",
             "resource_profile.json", "lineage.json", "smoke_config.yaml", "preflight.json"}
CANDIDATE_COMPLETION = "smoke_completion_candidate.json"
SUPERVISOR_ACCEPTANCE = "supervisor_acceptance.json"


def _read_yaml_hash(root: Path, reference: str, label: str):
    raw = referenced_file(root, reference, label).read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def _validate_frozen_baseline(baseline: Any) -> None:
    if not isinstance(baseline, Mapping):
        raise TypeError("frozen baseline must be a mapping")
    expected = {"seed": 42, "folds": 4, "repeats": 3, "split_seeds": [42, 43, 44],
                "split_manifest": "outputs/splits/building_folds_repeated.parquet",
                "target": "anomaly", "positive_label": 1, "primary_metric": "roc_auc",
                "guardrails": ["average_precision", "log_loss", "brier_score"],
                "features": {"numeric": list(NUMERIC), "categorical": list(CATEGORICAL)}}
    # JSON comparison distinguishes booleans from integers in nested contracts.
    for key, value in expected.items():
        if json.dumps(baseline.get(key), sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError(f"frozen baseline semantic contract mismatch: {key}")
    models = baseline.get("models")
    if not isinstance(models, Mapping) or set(models) != set(BASELINE_MODELS):
        raise ValueError("frozen baseline models mismatch")
    for name in BASELINE_MODELS:
        model = models[name]
        if (not isinstance(model, Mapping) or
                json.dumps(model.get("params"), sort_keys=True) != json.dumps(BASELINE_PARAMS[name], sort_keys=True)):
            raise ValueError(f"frozen baseline parameters mismatch: {name}")


def _positive(value: Any, ceiling: float, integer: bool = False) -> bool:
    return (type(value) in ((int,) if integer else (int, float)) and 0 < value <= ceiling
            and (type(value) is int or math.isfinite(value)))


def _config_errors(settings: Any) -> list[str]:
    if not isinstance(settings, Mapping):
        return ["smoke config must be a mapping"]
    errors = []
    exact = {"schema_version": 1, "version": 1, "scope": "campaign_smoke", "mode": "smoke",
             "execution_authorized": True, "host_preflight_status": "approved",
             "user_approval_status": "approved", "max_fold_count": 1, "max_seed_count": 1,
             "seed": 42, "repeat_id": 0, "device": "cpu", "rss_measurement": "process_tree",
             "max_hpo_trials": 0, "non_comparative": True, "n03_eligible": False,
             "experiment_ids": [EXPERIMENT], "source_design": SOURCE_DESIGN,
             "data_reference": "outputs/prepared/train_joined.parquet"}
    for key, expected in exact.items():
        value = settings.get(key)
        if type(value) is not type(expected) or value != expected:
            errors.append(f"smoke requires {key}={expected!r}")
    for key, ceiling, integer in (("maximum_cardinality", 10000, True), ("max_model_fits", 3, True),
                                 ("cpu_threads", 2, True), ("max_wall_time_seconds", 120, False),
                                 ("max_ram_gb", 1, False), ("batch_size", 256, True),
                                 ("train_rows", 10000, True), ("validation_rows", 10000, True)):
        if not _positive(settings.get(key), ceiling, integer):
            errors.append(f"invalid or expanded smoke cap: {key}")
    if type(settings.get("fold_id")) is not int or settings["fold_id"] not in range(4):
        errors.append("fold_id must select one original grouped fold")
    if not errors and settings["train_rows"] + settings["validation_rows"] > settings["maximum_cardinality"]:
        errors.append("train+validation rows exceed maximum_cardinality")
    if type(settings.get("max_model_fits")) is int and settings["max_model_fits"] < 2:
        errors.append("two arms require two admitted fits")
    params = settings.get("model_params")
    if (not isinstance(params, Mapping) or set(params) != set(HGB_PARAMS)
            or any(type(params[k]) is not type(v) or params[k] != v for k, v in HGB_PARAMS.items())):
        errors.append("both arms require the identical fixed smoke-only HGB params")
    if settings.get("launch_review_status") not in ("pending", "approved"):
        errors.append("invalid parent launch review status")
    return errors


def validate_smoke_config(root: Path, settings: Mapping[str, Any], experiment_id: str) -> dict[str, Any]:
    """Validate separate smoke permission without opening the full campaign gate."""
    errors = _config_errors(settings)
    validated_hashes = {}
    if experiment_id != EXPERIMENT:
        errors.append("explicit supported real experiment ID must be selected")
    if not errors:
        try:
            source, source_hash = _read_yaml_hash(root, settings["source_design"], "source design")
            if settings.get("source_design_sha256") != source_hash:
                raise ValueError("source design checksum mismatch")
            baseline, baseline_hash = _read_yaml_hash(root, source["baseline_config"], "baseline")
            _validate_frozen_baseline(baseline)
            budget, budget_hash = _read_yaml_hash(root, source["resource_budget"], "design budget")
            design = validate_admission(source, budget, baseline)
            if not design["design_valid"]:
                raise ValueError(f"source design invalid: {design['design_errors']}")
            row = next(row for row in source["experiment_matrix"] if row["id"] == EXPERIMENT)
            if (row["feature_set"], row["model_family"], row["status"]) != ("core_predictors", "hist_gradient_boosting", "planned"):
                raise ValueError("selected source experiment recipe mismatch")
            if baseline.get("features") != {"numeric": list(NUMERIC), "categorical": list(CATEGORICAL)}:
                raise ValueError("smoke core features must match authoritative baseline registry")
            validated_hashes = {settings["source_design"]: source_hash,
                                source["baseline_config"]: baseline_hash,
                                source["resource_budget"]: budget_hash}
        except (OSError, ValueError, TypeError, KeyError, StopIteration, yaml.YAMLError) as error:
            errors.append(str(error))
    reviewed = isinstance(settings, Mapping) and settings.get("launch_review_status") == "approved"
    return {"config_valid": not errors, "executable": not errors and reviewed,
            "status": "rejected_by_gate" if errors else
            ("admitted_smoke_only" if reviewed else "blocked_parent_review"), "errors": errors,
            "validated_input_sha256": validated_hashes}


def host_preflight() -> dict[str, Any]:
    """Remeasure MemAvailable immediately before launch; no framework/GPU import."""
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        memory[key] = int(value.split()[0]) * 1024
    return {"timestamp_utc": datetime.now(UTC).isoformat(),
            "available_ram_bytes": memory["MemAvailable"],
            "logical_cpus": len(os.sched_getaffinity(0))}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def _assert_admitted_hashes(input_hashes: Mapping[str, str], admitted: Mapping[str, str], stage: str) -> None:
    for reference, digest in admitted.items():
        current = input_hashes.get(reference)
        if current is None:
            raise ValueError(f"admitted input missing from smoke snapshot during {stage}: {reference}")
        if current != digest:
            raise ValueError(f"admitted input hash changed during {stage}: {reference}")


def _current_hashes(root: Path, references: list[str]) -> dict[str, str]:
    return {reference: file_sha256(referenced_file(root, reference, "input")) for reference in references}


def _scan_sample(dataset: Any, building_ids: list[int], quota: int, batch_size: int):
    import pyarrow as pa
    import pyarrow.dataset as ds

    scanner = dataset.scanner(columns=list(PROJECTED), filter=ds.field("building_id").isin(building_ids),
                              batch_size=min(batch_size, quota), batch_readahead=0,
                              fragment_readahead=0, use_threads=False)
    batches, remaining = [], quota
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        selected = batch.slice(0, min(remaining, batch.num_rows))
        batches.append(selected)
        remaining -= selected.num_rows
        if remaining == 0:
            break
    if remaining:
        raise ValueError("eligible development rows do not fill the predeclared sample quota")
    return pa.Table.from_batches(batches).to_pandas()


def _validate_calendar(frame: Any) -> None:
    import pandas as pd

    if not pd.api.types.is_datetime64_any_dtype(frame.timestamp.dtype) or frame.timestamp.isna().any():
        raise ValueError("timestamp must be a valid Datetime, matching the frozen baseline")
    stamp = pd.to_datetime(frame.timestamp, errors="raise")
    expected = {"hour": stamp.dt.hour, "weekday": stamp.dt.weekday,
                "month": stamp.dt.month, "year": stamp.dt.year}
    for name, values in expected.items():
        if frame[name].isna().any() or not (frame[name].to_numpy() == values.to_numpy()).all():
            raise ValueError(f"calendar {name} disagrees with timestamp (weekday Monday=0)")


def load_grouped_sample(data_path: Path, manifest: Any, settings: Mapping[str, Any]) -> dict[str, Any]:
    """First eligible physical-order rows, without label-based retry or global sort.

    Arrow filtering happens before any sample frame/materialization: no vault rows
    or labels reach the experiment. Parquet may decode shared compressed pages;
    only projected, filtered batches are retained, with readahead disabled.
    """
    import numpy as np
    import polars as pl
    import pyarrow as pa
    import pyarrow.dataset as ds

    if (not _positive(settings.get("maximum_cardinality"), 10000, True)
            or not _positive(settings.get("train_rows"), 10000, True)
            or not _positive(settings.get("validation_rows"), 10000, True)
            or settings["train_rows"] + settings["validation_rows"] > settings["maximum_cardinality"]):
        raise ValueError("sample rows exceed admitted total")
    if not _positive(settings.get("batch_size"), 256, True):
        raise ValueError("sample batch_size exceeds bounded projection limit")
    dev = manifest.filter((pl.col("repeat") == settings["repeat_id"]) & (pl.col("split") == "dev"))
    train_ids = sorted(dev.filter(pl.col("fold") != settings["fold_id"])["building_id"].to_list())
    val_ids = sorted(dev.filter(pl.col("fold") == settings["fold_id"])["building_id"].to_list())
    vault_ids = set(manifest.filter(pl.col("split") == "vault")["building_id"])
    if not train_ids or not val_ids or set(train_ids) & set(val_ids) or vault_ids & (set(train_ids) | set(val_ids)):
        raise ValueError("invalid grouped development/vault membership")
    dataset = ds.dataset(data_path, format="parquet")
    if not set(PROJECTED) <= set(dataset.schema.names):
        raise ValueError("projected data schema is missing core columns")
    if not pa.types.is_timestamp(dataset.schema.field("timestamp").type):
        raise ValueError("timestamp must use the frozen baseline Datetime datatype")
    for name in NUMERIC + ("anomaly", "building_id"):
        dtype = dataset.schema.field(name).type
        if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)):
            raise ValueError(f"invalid numeric schema: {name}")
    if not (pa.types.is_string(dataset.schema.field("primary_use").type)
            or pa.types.is_large_string(dataset.schema.field("primary_use").type)):
        raise ValueError("primary_use must have string schema")
    train = _scan_sample(dataset, train_ids, settings["train_rows"], settings["batch_size"])
    valid = _scan_sample(dataset, val_ids, settings["validation_rows"], settings["batch_size"])
    for frame in (train, valid):
        _validate_calendar(frame)
        if frame[["building_id", "timestamp", "anomaly"]].isna().any().any():
            raise ValueError("sample keys and targets must be nonnull")
        if set(frame.anomaly.unique()) != {0, 1}:
            raise ValueError("predeclared sample must contain both classes; no fallback")
        if frame.duplicated(["building_id", "timestamp"]).any():
            raise ValueError("sample contains duplicate natural keys")
        if np.isinf(frame[list(NUMERIC)].to_numpy(dtype=float)).any():
            raise ValueError("sample contains infinite predictors")
    return {"train": train, "validation": valid,
            "selection_policy": "first eligible physical-order rows; no class/score-based retries",
            "projected_columns": list(PROJECTED)}


def _missingness_registry():
    from project.features.registry import (
        FeatureMetadata,
        FeatureRegistry,
        ResourceCost,
        TransformFitBoundary,
        build_lead_baseline_registry,
    )

    registry = build_lead_baseline_registry()
    definitions = [registry.get(name) for name in registry.names]
    for name in CORE:
        source = registry.get(name)
        definitions.append(FeatureMetadata(name=f"{name}_missing", family="missingness",
            source_columns=(name,), semantic_meaning=f"Missingness of observed {name} before imputation",
            prediction_time_availability=source.prediction_time_availability,
            target_dependence=source.target_dependence, transform_fit_boundary=TransformFitBoundary.FIXED,
            fallback_behavior="Deterministic 0/1 null flag; no fitted statistics",
            resource_cost=ResourceCost.LOW, provenance_status=source.provenance_status))
    return FeatureRegistry(definitions)


def _fit_arms(sample: Mapping[str, Any], settings: Mapping[str, Any]):
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    rows, metrics = [], {}
    train, valid = sample["train"], sample["validation"]
    _validate_calendar(train)
    _validate_calendar(valid)
    registry = _missingness_registry()
    fit_count = 0
    for arm in ("raw", "missingness"):
        if fit_count >= settings["max_model_fits"]:
            raise ValueError("smoke fit count cap reached")
        train_x, valid_x = train[list(CORE)].copy(), valid[list(CORE)].copy()
        predictors = list(CORE)
        if arm == "missingness":
            for name in CORE:
                flag = f"{name}_missing"
                train_x[flag] = train_x[name].isna().astype(float)
                valid_x[flag] = valid_x[name].isna().astype(float)
                predictors.append(flag)
        validate_predictors(predictors, registry, datetime.now(UTC).isoformat())
        for frame in (train_x, valid_x):
            frame["primary_use"] = frame["primary_use"].where(frame["primary_use"].notna(), np.nan)
        numeric = list(NUMERIC) + [name for name in predictors if name.endswith("_missing")]
        preprocessing = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                           encoded_missing_value=-1), list(CATEGORICAL)),
        ], remainder="drop", sparse_threshold=0)
        model = Pipeline([("preprocessing", preprocessing),
                          ("classifier", HistGradientBoostingClassifier(**settings["model_params"]))])
        model.fit(train_x, train.anomaly)
        fit_count += 1
        probability = model.predict_proba(valid_x)[:, 1]
        if not np.isfinite(probability).all() or not ((0 <= probability) & (probability <= 1)).all():
            raise ValueError("smoke predicted probabilities must be finite in [0,1]")
        metrics[arm] = {"roc_auc": roc_auc_score(valid.anomaly, probability),
                        "average_precision": average_precision_score(valid.anomaly, probability),
                        "log_loss": log_loss(valid.anomaly, probability, labels=[0, 1]),
                        "brier_score": brier_score_loss(valid.anomaly, probability),
                        "predictors": predictors, "model_fit_count": 1}
        prediction = valid[["building_id", "timestamp", "anomaly"]].copy()
        prediction["arm"], prediction["probability"] = arm, probability
        rows.append(prediction)
    return pd.concat(rows, ignore_index=True), {
        "arms": metrics, "model_fit_count": fit_count, "parameters_identical_both_arms": True,
        "model_params": dict(settings["model_params"]), "non_comparative": True, "n03_eligible": False,
        "smoke_only_adjustments": "20 boosting iterations, 7 leaves; fold-local median/ordinal preprocessing; not frozen full baseline recipe",
    }


def _artifact_manifest(run_dir: Path) -> dict[str, Any]:
    """Small streaming-hash check usable by the lightweight supervisor."""
    result_path = run_dir / "worker_result.json"
    if not result_path.is_file():
        raise ValueError("smoke artifact manifest missing")
    result = json.loads(result_path.read_text())
    if not isinstance(result, Mapping):
        raise TypeError("smoke artifact manifest must be a mapping")
    hashes = result.get("artifact_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != ARTIFACTS:
        raise ValueError("smoke artifact manifest incomplete")
    for reference, digest in hashes.items():
        path = run_dir / reference
        if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
            raise ValueError("smoke artifact hash mismatch")
    if (result.get("run_id") != run_dir.name or result.get("experiment_id") != EXPERIMENT
            or result.get("model_fit_count") != 2 or result.get("non_comparative") is not True
            or result.get("mode") != "smoke" or result.get("n03_eligible") is not False):
        raise ValueError("smoke artifact run/fit linkage mismatch")
    return dict(result)


def validate_smoke_artifacts(run_dir: Path, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Validate actual small sample evidence INSIDE the guarded worker."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    result = _artifact_manifest(run_dir)
    selected = pq.read_table(run_dir / "sample_selection.parquet").to_pandas()
    predictions = pq.read_table(run_dir / "sample_predictions.parquet").to_pandas()
    if "anomaly" not in selected or "anomaly" not in predictions:
        raise ValueError("smoke target alignment evidence missing")
    for frame in (selected, predictions):
        if (not pd.api.types.is_integer_dtype(frame.anomaly.dtype)
                or frame.anomaly.isna().any() or set(frame.anomaly.unique()) != {0, 1}):
            raise ValueError("smoke target evidence must be integer binary labels")
    if len(selected) != settings["train_rows"] + settings["validation_rows"] or len(selected) > 10000:
        raise ValueError("smoke artifact sample row cap mismatch")
    if selected.duplicated(["building_id", "timestamp"]).any():
        raise ValueError("smoke artifact selected keys are not unique")
    train = selected[selected.sample_partition == "train"]
    valid = selected[selected.sample_partition == "validation"]
    if (len(train) != settings["train_rows"] or len(valid) != settings["validation_rows"]
            or set(train.building_id) & set(valid.building_id)):
        raise ValueError("smoke artifact grouped sample mismatch")
    if len(predictions) != 2 * len(valid) or set(predictions.arm) != {"raw", "missingness"}:
        raise ValueError("smoke artifact predictions count mismatch")
    expected_keys = set(zip(valid.building_id, valid.timestamp, strict=True))
    expected_targets = valid.set_index(["building_id", "timestamp"]).anomaly
    metrics = json.loads((run_dir / "metrics.json").read_text())
    arms = metrics.get("arms")
    if not isinstance(arms, Mapping) or set(arms) != {"raw", "missingness"}:
        raise ValueError("smoke metric arms missing")
    for arm in ("raw", "missingness"):
        part = predictions[predictions.arm == arm]
        if len(part) != len(valid) or set(zip(part.building_id, part.timestamp, strict=True)) != expected_keys:
            raise ValueError("smoke artifact prediction keys mismatch")
        aligned = part.set_index(["building_id", "timestamp"]).reindex(expected_targets.index)
        if not (aligned.anomaly.to_numpy() == expected_targets.to_numpy()).all():
            raise ValueError("smoke arm targets disagree with selected validation targets")
        if not np.isfinite(aligned.probability).all() or not aligned.probability.between(0, 1).all():
            raise ValueError("smoke artifact probabilities invalid")
        recomputed = {"roc_auc": roc_auc_score(expected_targets, aligned.probability),
                      "average_precision": average_precision_score(expected_targets, aligned.probability),
                      "log_loss": log_loss(expected_targets, aligned.probability, labels=[0, 1]),
                      "brier_score": brier_score_loss(expected_targets, aligned.probability)}
        recorded = arms[arm]
        if not isinstance(recorded, Mapping):
            raise TypeError("smoke metric arm must be a mapping")
        for name, value in recomputed.items():
            actual = recorded.get(name)
            if type(actual) not in (int, float) or not np.isfinite(actual) or not np.isclose(actual, value, rtol=1e-12, atol=1e-12):
                raise ValueError(f"smoke metric {arm}/{name} disagrees with predictions")
    if not np.isfinite(predictions.probability).all() or not predictions.probability.between(0, 1).all():
        raise ValueError("smoke artifact probabilities invalid")
    if metrics.get("model_fit_count") != 2 or metrics.get("model_params") != HGB_PARAMS:
        raise ValueError("smoke artifact metric/fit contract mismatch")
    return result


def _smoke_worker(args: Mapping[str, Any]) -> int:
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = time.monotonic()
    root, run_dir = Path(args["root"]), Path(args["run_dir"])
    settings, config_hash = _read_yaml_hash(root, args["config_reference"], "smoke config")
    report = validate_smoke_config(root, settings, args["experiment_id"])
    if not report["executable"]:
        raise ValueError(f"smoke worker admission blocked: {report}")
    if settings != args["settings"]:
        raise ValueError("smoke config changed between admission and worker")
    current_binding = {**report["validated_input_sha256"], args["config_reference"]: config_hash}
    if current_binding != args.get("validated_input_sha256"):
        raise ValueError("validated source/baseline/config hashes changed before worker launch")
    source = yaml.safe_load(referenced_file(root, settings["source_design"], "source design").read_text())
    pa.set_cpu_count(settings["cpu_threads"])
    pa.set_io_thread_count(1)
    split_report = validate_split_contract(root, source["prediction_contract"])
    if not split_report["valid"]:
        raise ValueError(f"smoke active split contract invalid: {split_report}")
    data_path = referenced_file(root, settings["data_reference"], "data")
    source_hash = file_sha256(data_path)
    split_path = referenced_file(root, source["prediction_contract"]["split_manifest"], "split")
    split_metadata = json.loads(split_path.with_name("split_metadata_repeated.json").read_text())
    input_references = [settings["source_design"], args["config_reference"], source["baseline_config"], source["resource_budget"],
                        source["prediction_contract"]["split_manifest"],
                        str(split_path.with_name("split_metadata_repeated.json").relative_to(root))]
    input_references.extend(entry["path"] for entry in split_metadata["per_repeat"])
    input_hashes = _current_hashes(root, input_references)
    admitted_hashes = args["validated_input_sha256"]
    _assert_admitted_hashes(input_hashes, admitted_hashes, "pre-fit binding")
    manifest = pl.read_parquet(split_path)
    sample = load_grouped_sample(data_path, manifest, settings)
    input_hashes = _current_hashes(root, input_references)
    _assert_admitted_hashes(input_hashes, admitted_hashes, "pre-fit sample binding")
    predictions, metrics = _fit_arms(sample, settings)
    selected = pd.concat([sample[partition][["building_id", "timestamp", "anomaly"]].assign(sample_partition=partition)
                          for partition in ("train", "validation")], ignore_index=True)
    pq.write_table(pa.Table.from_pandas(selected, preserve_index=False), run_dir / "sample_selection.parquet")
    pq.write_table(pa.Table.from_pandas(predictions, preserve_index=False), run_dir / "sample_predictions.parquet")
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "preflight.json", args["preflight"])
    (run_dir / "smoke_config.yaml").write_text(yaml.safe_dump(dict(settings)))
    _write_json(run_dir / "lineage.json", {
        "source_data_reference": settings["data_reference"], "source_parquet_sha256": source_hash,
        "source_design_sha256": file_sha256(root / settings["source_design"]),
        "config_sha256": file_sha256(root / args["config_reference"]),
        "baseline_config_sha256": file_sha256(root / source["baseline_config"]),
        "split_sha256": split_report["split_sha256"],
        "split_metadata_sha256": file_sha256(split_path.with_name("split_metadata_repeated.json")),
        "repeat_id": settings["repeat_id"], "fold_id": settings["fold_id"], "seed": 42,
        "selection_policy": sample["selection_policy"], "projected_columns": sample["projected_columns"],
        "selected_sample_sha256": file_sha256(run_dir / "sample_selection.parquet"),
        "input_artifact_sha256": input_hashes,
        "row_identity_contract": "Natural keys (building_id,timestamp), partition and target plus source Parquet SHA256; SMOKE only, not full OOF row_index",
        "validated_launch_sha256": admitted_hashes,
    })
    _write_json(run_dir / "resource_profile.json", {
        "mode": "smoke", "non_comparative": True, "n03_eligible": False, "model_fit_count": 2,
        "train_rows": len(sample["train"]), "validation_rows": len(sample["validation"]),
        "cpu_threads": settings["cpu_threads"], "device": "cpu", "worker_elapsed_seconds": time.monotonic() - started,
        "environment": {name: os.environ.get(name) for name in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "CUDA_VISIBLE_DEVICES")},
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
    })
    if file_sha256(data_path) != source_hash or file_sha256(split_path) != split_report["split_sha256"]:
        raise ValueError("data/split changed during smoke execution")
    input_hashes = _current_hashes(root, input_references)
    _assert_admitted_hashes(input_hashes, admitted_hashes, "completion binding")
    result = {"run_id": run_dir.name, "experiment_id": EXPERIMENT, "model_fit_count": 2,
              "mode": "smoke", "non_comparative": True, "n03_eligible": False,
              "artifact_sha256": {name: file_sha256(run_dir / name) for name in sorted(ARTIFACTS)}}
    _write_json(run_dir / "worker_result.json", result)
    validate_smoke_artifacts(run_dir, settings)
    _write_json(run_dir / "run_status.json", {**result, "status": "artifacts_ready",
        "worker_result_sha256": file_sha256(run_dir / "worker_result.json")})
    return 0


def _guard_entry(args: Mapping[str, Any]) -> int:
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, affinity[:args["cpu_threads"]])
    _write_json(Path(args["worker_args"]["run_dir"]) / "worker_identity.json", {"pid": os.getpid()})
    return args["target"](args["worker_args"])


def _runtime_tree(pid: int, tracked: set[int]) -> tuple[int, set[int]]:
    """Continue accounting for observed descendants even after their parent exits."""
    _, current = process_tree_rss_bytes(pid)
    selected = current | tracked
    for child in tracked - current:
        _, descendants = process_tree_rss_bytes(child)
        selected.update(descendants)
    rss = 0
    for child in selected:
        try:
            raw = Path(f"/proc/{child}/stat").read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            rss += max(0, int(fields[21])) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            continue
    return rss, selected


def _terminate_phase(process: Any, tracked: set[int]) -> None:
    """Every terminal path kills group AND known tree; no normal-exit bypass."""
    if process.pid is None:
        return
    try:
        _, selected = _runtime_tree(process.pid, tracked)
    except (OSError, ValueError):
        selected = tracked | {process.pid}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        for pid in selected:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
    process.join(timeout=0.05)


def _watch_phase(process: Any, limits: Mapping[str, Any], peak: Any) -> str | None:
    tracked = set()
    try:
        process.start()
        while True:
            rss, current = _runtime_tree(process.pid, tracked)
            tracked.update(current)
            peak.value = max(peak.value, rss)
            if time.monotonic() >= limits["deadline"]:
                return "total smoke wall cap exceeded"
            if rss >= limits["max_rss_bytes"]:
                return "process-tree rss cap exceeded"
            if not process.is_alive():
                process.join(timeout=0)
                if process.exitcode != 0:
                    return f"{limits['phase']} worker exited with code {process.exitcode}"
                remaining_rss, _ = _runtime_tree(process.pid, tracked)
                if remaining_rss:
                    return f"{limits['phase']} left live descendants"
                return None
            process.join(timeout=min(0.02, max(0, limits["deadline"] - time.monotonic())))
    except (OSError, ValueError, RuntimeError) as error:
        return f"{limits['phase']} monitoring failed: {error}"
    finally:
        _terminate_phase(process, tracked)


def _observation(state: Mapping[str, Any], success: bool, reason: str | None) -> dict[str, Any]:
    return {"run_id": state["run_dir"].name, "status": "completed" if success else "failed",
            "resource_guard_status": "completed" if success else "failed", "reason": reason,
            "mode": "smoke", "non_comparative": True, "n03_eligible": False,
            "artifact_validation_passed": success, "rss_limit_enforced": True,
            "max_rss_bytes": state["max_rss_bytes"], "cpu_threads": state["settings"]["cpu_threads"],
            "max_wall_seconds": state["settings"]["max_wall_time_seconds"],
            "elapsed_seconds": time.monotonic() - state["started"],
            "peak_process_tree_rss_bytes": state["peak"].value,
            "worker_exitcode": state["worker_exitcode"], "poll_seconds": 0.02,
            "runtime_scope": "compute and supervised finalization; cleanup/reporting reserve inside total deadline"}


def _pending_observation(state: Mapping[str, Any]) -> dict[str, Any]:
    observation = _observation(state, False, "awaiting supervisor acceptance")
    observation["status"] = "pending_supervisor_acceptance"
    observation["resource_guard_status"] = "pending_supervisor_acceptance"
    observation["artifact_validation_passed"] = False
    return observation


def _finalization_entry(state: Mapping[str, Any]) -> None:
    """Hash/read/recompute/write in a killable process, never in the supervisor."""
    os.setsid()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:state["settings"]["cpu_threads"]])
    run_dir = state["run_dir"]
    try:
        result = validate_smoke_artifacts(run_dir, state["settings"])
        stage = json.loads((run_dir / "run_status.json").read_text())
        if (stage.get("status") != "artifacts_ready" or
                stage.get("worker_result_sha256") != file_sha256(run_dir / "worker_result.json")):
            raise ValueError("artifact completion validation attestation missing")
        _write_json(run_dir / "guard_observations.json", _pending_observation(state))
        _write_json(run_dir / CANDIDATE_COMPLETION, {**result, "status": "pending_supervisor_acceptance"})
    except (OSError, ValueError, TypeError, KeyError) as error:
        _write_json(run_dir / "run_status.json", {"run_id": run_dir.name, "status": "failed",
            "reason": str(error), "mode": "smoke", "non_comparative": True, "n03_eligible": False})
        raise SystemExit(1) from error


def _commit_supervisor_success(state: Mapping[str, Any]) -> None:
    run_dir = state["run_dir"]
    candidate = json.loads((run_dir / CANDIDATE_COMPLETION).read_text())
    if candidate.get("status") != "pending_supervisor_acceptance":
        raise ValueError("smoke completion candidate missing supervisor handoff")
    acceptance = {"run_id": run_dir.name, "status": "accepted", "candidate_sha256": file_sha256(run_dir / CANDIDATE_COMPLETION),
                  "accepted_at_utc": datetime.now(UTC).isoformat()}
    _write_json(run_dir / SUPERVISOR_ACCEPTANCE, acceptance)
    _write_json(run_dir / "guard_observations.json", _observation(state, True, None))
    guard_hash = file_sha256(run_dir / "guard_observations.json")
    acceptance_hash = file_sha256(run_dir / SUPERVISOR_ACCEPTANCE)
    _write_json(run_dir / "run_status.json", {**candidate, "status": "smoke_completed",
        "artifact_sha256": {**candidate["artifact_sha256"], "guard_observations.json": guard_hash,
                            SUPERVISOR_ACCEPTANCE: acceptance_hash}})


def read_smoke_completion(run_dir: Path) -> dict[str, Any]:
    status_path = run_dir / "run_status.json"
    acceptance_path = run_dir / SUPERVISOR_ACCEPTANCE
    if not status_path.exists() or not acceptance_path.exists():
        raise ValueError("smoke completion requires supervisor acceptance")
    status = json.loads(status_path.read_text())
    acceptance = json.loads(acceptance_path.read_text())
    if status.get("status") != "smoke_completed" or acceptance.get("status") != "accepted":
        raise ValueError("smoke completion is provisional without supervisor acceptance")
    artifact_hashes = status.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping):
        raise TypeError("smoke completion artifact hashes missing")
    if artifact_hashes.get(SUPERVISOR_ACCEPTANCE) != file_sha256(acceptance_path):
        raise ValueError("supervisor acceptance hash mismatch")
    if acceptance.get("candidate_sha256") != file_sha256(run_dir / CANDIDATE_COMPLETION):
        raise ValueError("smoke completion candidate hash mismatch")
    return status


FAILURE_REPORT_PROGRAM = """
import json, os, sys
from pathlib import Path
request = json.loads(sys.argv[1])
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:request['cpu_threads']])
for name, payload in request['payloads'].items():
    path = Path(request['run_dir']) / name
    temporary = path.with_suffix('.failure.tmp')
    temporary.write_text(json.dumps(payload, allow_nan=False) + '\\n')
    temporary.replace(path)
"""


class _FailureReporter:
    """Spawn stdlib-only reporter without importing the ML harness or forking threads."""

    def __init__(self, state: Mapping[str, Any], reason: str) -> None:
        self.request = {"run_dir": str(state["run_dir"]), "cpu_threads": state["settings"]["cpu_threads"],
            "payloads": {"guard_observations.json": _observation(state, False, reason),
                "run_status.json": {"run_id": state["run_dir"].name, "status": "failed", "reason": reason,
                                    "mode": "smoke", "non_comparative": True, "n03_eligible": False}}}
        self.process = None

    @property
    def pid(self):
        return self.process.pid if self.process is not None else None

    @property
    def exitcode(self):
        return self.process.poll() if self.process is not None else None

    def start(self):
        self.process = subprocess.Popen([sys.executable, "-I", "-S", "-B", "-c", FAILURE_REPORT_PROGRAM,
                                         json.dumps(self.request, allow_nan=False)], start_new_session=True)

    def is_alive(self):
        return self.exitcode is None

    def join(self, timeout=None):
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass


def run_smoke_guarded(target: Callable, args: Mapping[str, Any], run_dir: Path,
                      settings: Mapping[str, Any]) -> int:
    """Smoke-specific watchdog reuses baseline worker/RSS primitives, not approval.

    Compute and finalizer are sequential, sharing one absolute deadline and RSS
    ceiling. Failure cleanup/reporting gets a reserve WITHIN that deadline. The
    parent performs no artifact hashing/reading/writing after worker completion.
    """
    for key, ceiling, integer in (("cpu_threads", 2, True), ("max_wall_time_seconds", 120, False),
                                 ("max_ram_gb", 1, False)):
        if not _positive(settings.get(key), ceiling, integer):
            raise ValueError(f"invalid smoke runtime cap: {key}")
    started = args.get("launch_started_monotonic", time.monotonic())
    deadline = min(started + settings["max_wall_time_seconds"],
                   time.monotonic() + settings["max_wall_time_seconds"])
    remaining = max(0, deadline - time.monotonic())
    reserve = min(1.0, remaining * 0.2)
    work_deadline = deadline - reserve
    spawn = multiprocessing.get_context("spawn")
    peak = spawn.Value("Q", 0, lock=False)
    state = {"run_dir": run_dir, "settings": dict(settings), "started": started,
             "max_rss_bytes": int(settings["max_ram_gb"] * 1024**3), "peak": peak,
             "worker_exitcode": None}
    previous = {name: os.environ.get(name) for name in (*THREAD_ENVIRONMENT, "CUDA_VISIBLE_DEVICES")}
    try:
        for name in THREAD_ENVIRONMENT:
            os.environ[name] = str(settings["cpu_threads"])
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        worker = spawn.Process(target=baseline_worker_entry,
            args=(_guard_entry, {"target": target, "worker_args": dict(args),
                                "cpu_threads": settings["cpu_threads"]}, settings["cpu_threads"]))
        reason = _watch_phase(worker, {"deadline": work_deadline,
            "max_rss_bytes": state["max_rss_bytes"], "phase": "compute"}, peak)
        state["worker_exitcode"] = worker.exitcode
        if reason is None:
            finalizer = spawn.Process(target=_finalization_entry, args=(state,))
            reason = _watch_phase(finalizer, {"deadline": work_deadline,
                "max_rss_bytes": state["max_rss_bytes"], "phase": "artifact finalization"}, peak)
        if reason is None:
            try:
                _commit_supervisor_success(state)
            except (OSError, ValueError, TypeError, KeyError) as error:
                reason = f"supervisor acceptance failed: {error}"
        if reason is not None:
            reporter = _FailureReporter(state, reason)
            _watch_phase(reporter, {"deadline": deadline, "max_rss_bytes": state["max_rss_bytes"],
                                    "phase": "failure report"}, peak)
            return 1
        return 0
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _parent_timeout(signum, frame):
    raise TimeoutError("total smoke wall cap exceeded during admission/supervision")


def _arm_parent_deadline(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("total smoke wall cap exceeded before launch")
    signal.setitimer(signal.ITIMER_REAL, remaining)


@contextmanager
def _parent_deadline(deadline: float):
    """Interrupt parent metadata I/O; data/finalization use process watchdogs."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("bounded smoke launch requires the main supervisor thread")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    try:
        signal.signal(signal.SIGALRM, _parent_timeout)
        _arm_parent_deadline(deadline)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        previous_remaining = max(0, previous_timer[0] - (time.monotonic() - started))
        signal.setitimer(signal.ITIMER_REAL, previous_remaining, previous_timer[1])


def _launch_request(request: dict[str, Any]) -> dict[str, Any]:
    root = Path(request["root"]).resolve()
    config_reference, experiment_id = request["config_reference"], request["experiment_id"]
    started = request["started"]
    if config_reference != SMOKE_CONFIG:
        return {"returncode": 2, "status": "blocked_unapproved_compute",
                "errors": ["--smoke requires explicit versioned campaign_smoke_v1.yaml"]}
    settings, config_hash = _read_yaml_hash(root, config_reference, "smoke config")
    if isinstance(settings, Mapping) and _positive(settings.get("max_wall_time_seconds"), 120):
        _arm_parent_deadline(started + settings["max_wall_time_seconds"])
    report = validate_smoke_config(root, settings, experiment_id)
    if not report["executable"]:
        return {**report, "returncode": 2}
    preflight = host_preflight()
    # Leave one additional process-tree budget as host headroom; never expand cap.
    if preflight["available_ram_bytes"] < 2 * settings["max_ram_gb"] * 1024**3:
        return {"returncode": 2, "status": "blocked_preflight", "errors": ["insufficient available RAM/headroom"],
                "preflight": preflight}
    if preflight["logical_cpus"] < settings["cpu_threads"]:
        return {"returncode": 2, "status": "blocked_preflight", "errors": ["insufficient available CPU affinity"]}
    runs = root / "outputs/runs"
    if runs.is_symlink() or (root / "outputs").is_symlink():
        raise ValueError("smoke output path must not follow symlinks")
    runs.mkdir(parents=True, exist_ok=True)
    run_dir = runs / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-smoke-" + uuid.uuid4().hex[:8])
    run_dir.mkdir(exist_ok=False)
    request["run_reference"] = str(run_dir.relative_to(root))
    args = {"root": str(root), "run_dir": str(run_dir), "config_reference": config_reference,
            "experiment_id": experiment_id, "settings": dict(settings), "preflight": preflight,
            "launch_started_monotonic": started,
            "validated_input_sha256": {**report["validated_input_sha256"], config_reference: config_hash}}
    code = run_smoke_guarded(_smoke_worker, args, run_dir, settings)
    return {"returncode": code, "status": "smoke_completed" if code == 0 else "failed",
            "run_reference": str(run_dir.relative_to(root)), "non_comparative": True, "n03_eligible": False}


def launch_smoke(root: Path, config_reference: str, experiment_id: str) -> dict[str, Any]:
    """Launch reviewed SMOKE under one deadline, including admission metadata I/O."""
    request = {"root": root, "config_reference": config_reference, "experiment_id": experiment_id,
               "started": time.monotonic()}
    try:
        with _parent_deadline(request["started"] + 120):
            return _launch_request(request)
    except TimeoutError as error:
        return {"returncode": 1, "status": "failed", "errors": [str(error)],
                "run_reference": request.get("run_reference"), "non_comparative": True, "n03_eligible": False}
