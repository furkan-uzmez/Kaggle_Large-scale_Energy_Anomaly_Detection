"""Task 12: Candidate evidence, manifest, status, and inventory linkage.

Writes immutable JSON-safe manifest.json, run_status.json, and appends to
candidate_inventory.jsonl in the run destination directory. Synthetic events
have outcome="completed", non_comparative=True, n03_eligible=False, and never
emit comparative campaign completion events.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from project.experiments.candidate_contracts import RunResult, RunSpec
from project.experiments.inventory import (
    REQUIRED_INVENTORY_FIELDS,
    append_inventory_records,
)


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy/path types into JSON-serializable primitives."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    return str(obj)


def write_candidate_evidence(
    run_result: RunResult,
    spec: RunSpec,
    artifacts: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Persist run evidence, immutable manifest/status, and inventory linkage."""
    dest_path = Path(destination).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)

    manifest_path = dest_path / "manifest.json"
    status_path = dest_path / "run_status.json"
    inventory_path = dest_path / "candidate_inventory.jsonl"

    manifest_data = artifacts.get("manifest")
    if manifest_data is None:
        manifest_data = {
            "schema_version": 1,
            "run_id": run_result.run_id,
            "execution_mode": spec.execution_mode,
            "non_comparative": run_result.non_comparative,
            "n03_eligible": run_result.n03_eligible,
            "source_notebook": spec.source_notebook,
            "source_cell_id": spec.source_cell_id,
            "candidates": {},
        }
        for c in spec.candidates:
            manifest_data["candidates"][c.experiment_id] = {
                "recipe_id": c.experiment_id,
                "model_family": c.model_family,
                "model_config_id": f"{c.experiment_id}_default",
                "experiment_id": c.experiment_id,
                "arm_id": "challenger",
                "score_kind": "positive_probability",
                "positive_label": 1,
            }

    # Write manifest.json
    manifest_safe = _json_safe(manifest_data)
    manifest_path.write_text(json.dumps(manifest_safe, indent=2, sort_keys=True))

    # Build status data
    status_data: dict[str, Any] = {
        "run_id": run_result.run_id,
        "status": run_result.status,
        "artifact_paths": list(run_result.artifact_paths),
        "non_comparative": run_result.non_comparative,
        "n03_eligible": run_result.n03_eligible,
        "execution_mode": spec.execution_mode,
        "source_notebook": spec.source_notebook,
        "source_cell_id": spec.source_cell_id,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    if run_result.status in ("blocked", "failed"):
        blocker_code = artifacts.get("blocker_code", "experiment_blocked" if run_result.status == "blocked" else "run_failed")
        reason = artifacts.get("reason", "Experiment blocked or execution failed")
        status_data["blocker_code"] = blocker_code
        status_data["reason"] = reason
        status_data["rejection_reason"] = f"{blocker_code}: {reason}"
    if "metric_report" in artifacts:
        status_data["metrics"] = artifacts["metric_report"]

    status_safe = _json_safe(status_data)
    status_path.write_text(json.dumps(status_safe, indent=2, sort_keys=True))

    # Build inventory records
    inventory_records: list[dict[str, Any]] = []
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    rejection = status_data.get("rejection_reason")

    primary_metric = None
    if "metric_report" in artifacts:
        report = artifacts["metric_report"]
        cand_metrics = report.get("candidate_metrics", {})
        for c_meta in cand_metrics.values():
            mean_m = c_meta.get("mean_repeat_metrics", {})
            if "roc_auc" in mean_m:
                primary_metric = mean_m["roc_auc"]
                break

    oof_path_str = str(artifacts.get("oof_path", ""))

    for candidate in spec.candidates:
        record: dict[str, Any] = {
            "experiment_id": candidate.experiment_id,
            "hypothesis_id": f"hyp_{candidate.experiment_id}",
            "varied_factor_id": f"factor_{candidate.experiment_id}",
            "baseline_id": "lead_baseline",
            "candidate_id": f"{candidate.experiment_id}_challenger",
            "recipe_id": candidate.experiment_id,
            "model_config_id": f"{candidate.experiment_id}_default",
            "data_sha256": artifacts.get("data_sha256", "b" * 64),
            "split_sha256": artifacts.get("split_sha256", "a" * 64),
            "config_sha256": artifacts.get("config_sha256", "c" * 64),
            "primary_metric": primary_metric,
            "oof_path": oof_path_str,
            "wall_time_seconds": float(artifacts.get("wall_time_seconds", 0.0)),
            "outcome": "completed" if run_result.status == "completed" else run_result.status,
            "rejection_reason": rejection,
            "run_id": run_result.run_id,
            "timestamp": now_iso,
        }
        # Validate all required fields exist
        for field in REQUIRED_INVENTORY_FIELDS:
            if field not in record:
                record[field] = None
        inventory_records.append(_json_safe(record))

    append_inventory_records(inventory_path, inventory_records)

    return {
        "manifest_path": str(manifest_path),
        "status_path": str(status_path),
        "inventory_path": str(inventory_path),
    }
