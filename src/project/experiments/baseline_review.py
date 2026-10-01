"""Read completed frozen baseline artifacts against the current development data."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from project.experiments.baseline_budget import (
    BASELINE_MODELS,
    BASELINE_PARAMS,
    validate_baseline_budget,
)
from project.models.artifacts import load_repeated_manifest, validate_oof_parquet


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _current_population(root: Path, config: dict) -> tuple[pl.DataFrame, dict[str, str], pl.DataFrame]:
    data_path = root / "outputs/prepared/train_joined.parquet"
    split_path = (root / config["split_manifest"]).resolve()
    if not split_path.is_relative_to(root / "outputs"):
        raise ValueError("Split manifest must be within outputs/")
    manifest = load_repeated_manifest(split_path)
    dev = manifest.filter(pl.col("split") == "dev")
    first = dev.filter(pl.col("repeat") == 0)
    # Filter before collecting labels: vault targets are never loaded.
    rows = (
        pl.scan_parquet(data_path)
        .filter(pl.col("building_id").is_in(first["building_id"].to_list()))
        .select("building_id", "timestamp", pl.col(config["target"]).alias("target"))
        .collect(engine="streaming")
        .sort(["building_id", "timestamp"])
        .with_row_index("row_index")
        .with_columns(pl.col("row_index").cast(pl.Int64))
    )
    if rows["building_id"].n_unique() != first.height:
        raise ValueError("Current data do not cover all development buildings")
    return rows, {
        "data_sha256": _sha256(data_path), "split_sha256": _sha256(split_path),
        "config_sha256": _sha256(root / "configs/experiments/baselines.yaml"),
    }, dev


def _read_run(root: Path, run: Path) -> dict[str, Any]:
    if not run.is_relative_to(root / "outputs/runs") or not run.is_dir():
        raise ValueError("Baseline run must be a directory within outputs/runs/")
    status = _json(run / "run_status.json")
    if status.get("status") != "completed" or status.get("run_id") != run.name:
        raise ValueError("Baseline run is not completed")
    summary = _json(run / "metrics.json")
    resources = _json(run / "resource_profile.json")
    for payload in (summary, resources):
        if payload.get("run_id") != run.name or payload.get("mode") != "full" or payload.get("non_comparative") is not False or payload.get("fold_ids") != [0, 1, 2, 3]:
            raise ValueError("Only full comparative four-fold baseline artifacts are eligible")
    if resources.get("model_fit_count") != 36 or summary.get("used_synthetic_fallback", False):
        raise ValueError("Expected exactly thirty-six real development fits")
    for payload in (summary, resources):
        if payload.get("repeat_ids") != [0, 1, 2] or payload.get("repeat_count") != 3 or payload.get("split_seeds") != [42, 43, 44]:
            raise ValueError("Expected three approved split repetitions")
    if not (run / "model_card.md").is_file():
        raise ValueError("Incomplete baseline model card")
    config_path = root / "configs/experiments/baselines.yaml"
    if (run / "baselines.yaml").read_bytes() != config_path.read_bytes():
        raise ValueError("Stale baseline config snapshot")
    config = yaml.safe_load(config_path.read_bytes())
    if not isinstance(config, dict) or tuple(config.get("models", {})) != BASELINE_MODELS or (config.get("seed"), config.get("folds"), config.get("repeats")) != (42, 4, 3) or config.get("split_seeds") != [42, 43, 44]:
        raise ValueError("Baseline contract is not frozen")
    for candidate in BASELINE_MODELS:
        if config["models"][candidate].get("params") != BASELINE_PARAMS[candidate]:
            raise ValueError("Baseline parameters are not frozen")
    budget = _json(run / "baseline_execution_budget.json")
    validate_baseline_budget(budget, config)
    guard = _json(run / "guard_observations.json")
    if guard.get("status") != "completed" or guard.get("run_id") != run.name or guard.get("reason") is not None or guard.get("worker_exitcode") != 0:
        raise ValueError("Baseline execution guard did not complete successfully")
    rss_limit_enforced = budget.get("rss_limit_enforced", True)
    if guard.get("rss_limit_enforced", True) is not rss_limit_enforced:
        raise ValueError("Guard RSS mode differs from approved baseline budget")
    for name in ("cpu_threads", "max_rss_bytes", "max_wall_seconds"):
        if name not in guard or guard[name] != budget[name]:
            raise ValueError("Guard caps differ from approved baseline budget")
    for name, cap in (("elapsed_seconds", budget["max_wall_seconds"]), ("peak_process_tree_rss_bytes", budget["max_rss_bytes"])):
        value = guard.get(name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0 or (cap is not None and value >= cap):
            raise ValueError(f"Baseline guard observation exceeds or lacks {name}")
    resources["execution_guard"] = guard
    resources["execution_budget"] = budget
    population, hashes, assignments = _current_population(root, config)

    def expected_rows():
        for repeat, seed in enumerate(config["split_seeds"]):
            yield repeat, population.join(assignments.filter(pl.col("repeat") == repeat).select(
                "building_id", "fold"), on="building_id", how="left").with_columns(
                *(pl.lit(value).alias(name) for name, value in hashes.items()),
                pl.lit(run.name).alias("run_id"), pl.lit(seed).alias("split_seed"))

    validated = validate_oof_parquet(run / "oof_predictions.parquet", expected_rows(),
        expected_candidate_ids=BASELINE_MODELS, candidate_metadata=config["models"])
    if summary.get("metric_aggregation") != "equal_mean_of_repeat_pooled_metrics":
        raise ValueError("Invalid repeat metric aggregation")
    for name in ("development_oof_row_count", "oof_record_count"):
        if summary.get(name) != validated[name]:
            raise ValueError("Recorded OOF coverage differs from actual records")
    metric_rows = []
    for candidate in BASELINE_MODELS:
        if not (run / f"model_card_{candidate}.md").is_file():
            raise ValueError("Incomplete per-candidate model cards")
        values = validated["candidate_metrics"][candidate]
        recorded = summary["candidate_metrics"][candidate]
        for name, value in values["pooled"].items():
            if not np.isclose(recorded["pooled"][name], value, rtol=0, atol=1e-12):
                raise ValueError("Recorded pooled metric differs from recomputed repeat mean")
        for field, keys in (("repeats", ("repeat",)), ("folds", ("repeat", "fold"))):
            rows = recorded[field]
            lookup = {tuple(row[key] for key in keys): row for row in rows}
            if len(rows) != len(values[field]) or len(lookup) != len(rows):
                raise ValueError("Recorded repeat/fold metric coverage differs")
            for row in values[field]:
                saved = lookup.get(tuple(row[key] for key in keys), {})
                for name in ("split_seed", "roc_auc", "average_precision", "log_loss", "brier_score"):
                    if name not in saved or not np.isclose(saved[name], row[name], rtol=0, atol=1e-12):
                        raise ValueError("Recorded repeat/fold metric differs from OOF")
        fold_auc = [row["roc_auc"] for row in values["folds"]]
        fit_seconds = summary["candidate_metrics"][candidate]["wall_time_seconds"]
        if isinstance(fit_seconds, bool) or not isinstance(fit_seconds, (float, int)) or not math.isfinite(fit_seconds) or fit_seconds < 0:
            raise ValueError("Invalid candidate fit wall time")
        metric_rows.append({
            "candidate_id": candidate, "model_family": config["models"][candidate]["model_family"],
            **values["pooled"], "fold_roc_auc_mean": float(np.mean(fold_auc)),
            "fold_roc_auc_std": float(np.std(fold_auc, ddof=1)), "fit_wall_seconds": float(fit_seconds),
            "repeat_roc_auc_mean": values["pooled"]["roc_auc"],
            "repeat_roc_auc_std": values["repeat_std"]["roc_auc"],
        })
    metrics = pl.DataFrame(metric_rows)
    return {
        "status": "completed", "run_id": run.name, "run_path": run.relative_to(root).as_posix(),
        "metrics": metrics, "fold_metrics": pl.DataFrame(validated["fold_metrics"]), "resources": resources,
        "repeat_metrics": pl.DataFrame(validated["repeat_metrics"]),
        "repeat_count": 3, "split_seeds": [42, 43, 44],
        "reference_candidate_id": max(metric_rows, key=lambda row: row["roc_auc"])["candidate_id"],
        "positive_prevalence": float(population["target"].mean()),
        "development_rows": population.height, "development_buildings": population["building_id"].n_unique(),
        "fold_count": 4,
    }


def load_baseline_review(project_root: Path, run_path: Path | None = None) -> dict:
    """Return the latest valid full baseline review, or a reasoned ``not_run``.

    Explicit invalid paths raise ValueError. Metrics are recomputed from validated
    OOF scores against current dev keys, labels, split/data/config hashes. The
    highest pooled ROC-AUC is a nominal reference, never a promotion decision.
    """
    root = project_root.resolve()
    if run_path is not None:
        run = (run_path if run_path.is_absolute() else root / run_path).resolve()
        try:
            return _read_run(root, run)
        except (OSError, KeyError, TypeError, ValueError, pl.exceptions.PolarsError) as error:
            raise ValueError(f"Invalid baseline run {run}: {error}") from error
    runs_root = root / "outputs/runs"
    runs = sorted((path for path in runs_root.iterdir() if path.is_dir()), reverse=True) if runs_root.is_dir() else []
    for run in runs:
        try:
            return _read_run(root, run.resolve())
        except (OSError, KeyError, TypeError, ValueError, pl.exceptions.PolarsError):
            continue
    return {"status": "not_run", "reason": "No completed eligible full baseline artifact matches the current development contract."}
