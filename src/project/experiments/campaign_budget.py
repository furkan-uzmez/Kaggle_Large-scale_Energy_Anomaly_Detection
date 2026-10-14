"""Read-only campaign design validation and bounded compute admission contracts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from project.experiments.campaign_ledger import file_sha256, referenced_file

# This stage can admit only the explicitly approved future CPU smoke contract.
# max_ram_gb uses binary GiB, matching the process-tree 1 GiB approval.
SMOKE_CAPS = {
    "maximum_cardinality": 10000, "max_ram_gb": 1,
    "max_wall_time_seconds": 120, "max_model_fits": 3,
    "max_seed_count": 1, "max_fold_count": 1, "cpu_threads": 2,
}
INTEGER_CAPS = set(SMOKE_CAPS) - {"max_ram_gb", "max_wall_time_seconds"}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _prediction_errors(contract: Any) -> list[str]:
    if not isinstance(contract, Mapping):
        return ["prediction_contract must be a mapping"]
    errors = []
    if contract.get("target") != "anomaly":
        errors.append("frozen target must be anomaly")
    if type(contract.get("positive_label")) is not int or contract["positive_label"] != 1:
        errors.append("frozen positive_label must be integer 1")
    if contract.get("primary_metric") != "roc_auc":
        errors.append("frozen primary_metric must be roc_auc")
    guardrails = contract.get("guardrails")
    if (not isinstance(guardrails, list) or any(not isinstance(g, str) for g in guardrails)
            or len(guardrails) != 3 or set(guardrails) != {"average_precision", "log_loss", "brier_score"}):
        errors.append("frozen guardrails mismatch")
    if (type(contract.get("folds")) is not int or contract.get("folds") != 4
            or type(contract.get("repeats")) is not int or contract.get("repeats") != 3
            or contract.get("split_seeds") != [42, 43, 44]
            or contract.get("split_manifest") != "outputs/splits/building_folds_repeated.parquet"):
        errors.append("frozen active 4x3 split contract mismatch")
    return errors


def _positive_cap(value: Any, integer: bool = False) -> bool:
    types = (int,) if integer else (int, float)
    return (type(value) in types and value > 0
            and (type(value) is int or math.isfinite(value)))


def _design_errors(campaign: Mapping[str, Any], baseline: Mapping[str, Any]) -> list[str]:
    errors = []
    if type(campaign.get("schema_version")) is not int or campaign["schema_version"] != 1:
        errors.append("invalid schema_version")
    for key in ("campaign_id", "baseline_config", "resource_budget"):
        if not isinstance(campaign.get(key), str) or not campaign[key].strip():
            errors.append(f"missing {key}")
    if type(campaign.get("execution_authorized")) is not bool:
        errors.append("execution_authorized must be boolean")
    errors.extend(_prediction_errors(campaign.get("prediction_contract")))
    nested = {}
    for key in ("model_family_audit", "feature_sets"):
        value = campaign.get(key)
        if not isinstance(value, Mapping) or any(not isinstance(v, Mapping) for v in value.values()):
            errors.append(f"{key} must map names to mappings")
            nested[key] = {}
        else:
            nested[key] = value
    models = baseline.get("models")
    if not isinstance(models, Mapping) or any(not isinstance(v, Mapping) for v in models.values()):
        errors.append("baseline models must map names to mappings")
        models = {}
    matrix = campaign.get("experiment_matrix")
    if not isinstance(matrix, list) or not matrix:
        return errors + ["experiment_matrix must be nonempty"]
    seen = set()
    for row in matrix:
        if not isinstance(row, Mapping):
            errors.append("experiment row must be mapping")
            continue
        identity = row.get("id")
        if not isinstance(identity, str) or not identity.strip() or identity in seen:
            errors.append("invalid or duplicate experiment ID")
            continue
        seen.add(identity)
        if not isinstance(row.get("status"), str) or row["status"] not in {"planned", "conditional", "blocked", "N/A"}:
            errors.append(f"invalid design status: {identity}")
        if not isinstance(row.get("execution_status"), str) or row["execution_status"] not in {"blocked_unapproved_compute", "blocked", "prohibited"}:
            errors.append(f"invalid closed execution status: {identity}")
        family = row.get("model_family")
        known = set(nested["model_family_audit"]) | set(models)
        if family is not None and (not isinstance(family, str) or family not in known):
            errors.append(f"unknown model family: {identity}")
        if row.get("status") == "N/A" and not any(
            isinstance(row.get(key), str) and row[key].strip()
            for key in ("rationale", "rejection_reason", "reason")
        ):
            errors.append(f"N/A requires reason: {identity}")
        if row.get("result") is not None or row.get("oof_artifact") is not None:
            errors.append(f"design cannot claim completion artifacts: {identity}")
        if row.get("feature_set") is not None and (not isinstance(row["feature_set"], str) or row["feature_set"] not in nested["feature_sets"]):
            errors.append(f"unregistered feature set: {identity}")
    return errors


def validate_admission(campaign: Mapping[str, Any], budget: Mapping[str, Any],
                       baseline: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Report schema validity independently from compute authorization.

    Admission is not a runner and not predictor approval. The later runner must
    also validate proposal registry and active splits before any data/model work.
    """
    if (not isinstance(campaign, Mapping) or not isinstance(budget, Mapping)
            or not isinstance(baseline, Mapping)):
        return {"design_valid": False, "executable": False, "status": "rejected_by_gate",
                "errors": ["campaign, budget, and baseline must be mappings"],
                "design_errors": ["campaign, budget, and baseline must be mappings"],
                "runner_implemented": False}
    design_errors = _design_errors(campaign, baseline)
    errors = list(design_errors)
    for key in ("candidate_target", "candidate_cap"):
        if key in budget and not _positive_cap(budget[key], True):
            design_errors.append(f"invalid {key}")
            errors.append(f"invalid {key}")
    for key in SMOKE_CAPS:
        value = budget.get(key)
        if value is not None and not _positive_cap(value, key in INTEGER_CAPS):
            design_errors.append(f"invalid typed cap: {key}")
            errors.append(f"invalid typed cap: {key}")
    hpo = budget.get("max_hpo_trials")
    if hpo is not None and (type(hpo) is not int or hpo < 0):
        design_errors.append("invalid max_hpo_trials")
        errors.append("invalid max_hpo_trials")
    authorized = (
        campaign.get("execution_authorized") is True
        and budget.get("execution_authorized") is True
        and budget.get("host_preflight_status") == "approved"
        and budget.get("user_approval_status") == "approved"
    )
    if not authorized:
        errors.append("blocked_unapproved_compute")
    if campaign.get("mode") != "smoke" or budget.get("mode") != "smoke" or budget.get("scope") != "campaign_smoke":
        errors.append("only campaign_smoke scope is approved; baseline_only/full cannot authorize")
    for key, ceiling in SMOKE_CAPS.items():
        value = budget.get(key)
        if not _positive_cap(value, key in INTEGER_CAPS) or value > ceiling:
            errors.append(f"smoke cap missing or exceeds approved limit: {key}")
    if type(hpo) is not int or hpo != 0:
        errors.append("smoke requires max_hpo_trials=0")
    if budget.get("device") != "cpu" or budget.get("rss_measurement") != "process_tree":
        errors.append("smoke requires CPU and process_tree RSS")
    executable = not errors
    return {"design_valid": not design_errors, "executable": executable,
            "status": "admitted_smoke_contract_only" if executable else
            ("blocked_unapproved_compute" if not authorized else "rejected_by_gate"),
            "errors": errors, "design_errors": design_errors, "runner_implemented": False}


def validate_split_contract(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Check only the small active 4x3 split artifacts; never recreate splits."""
    from polars.exceptions import PolarsError

    try:
        errors = _prediction_errors(contract)
        if errors:
            raise ValueError("; ".join(errors))
        result = _check_splits(Path(root).resolve(), contract)
    except (ValueError, KeyError, TypeError, OSError, PolarsError) as error:
        return {"valid": False, "errors": [str(error)]}
    return {"valid": True, "errors": [], **result}


def _check_splits(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    import polars as pl

    if (contract.get("folds"), contract.get("repeats"), contract.get("split_seeds")) != (4, 3, [42, 43, 44]):
        raise ValueError("expected authoritative 4x3 split contract")
    path = referenced_file(root, contract.get("split_manifest"), "split")
    metadata_path = referenced_file(root, str(path.with_name("split_metadata_repeated.json").relative_to(root)), "metadata")
    metadata = _mapping(json.loads(metadata_path.read_text()), "split metadata")
    artifacts = _mapping(metadata.get("artifacts"), "split metadata artifacts")
    digest = file_sha256(path)
    if (metadata.get("family_sha256") != digest or
            artifacts.get("building_folds_sha256") != digest or
            artifacts.get("building_folds_path") != contract["split_manifest"] or
            metadata.get("repeats") != 3 or metadata.get("n_dev_folds") != 4 or
            metadata.get("split_seeds") != [42, 43, 44]):
        raise ValueError("active master metadata/hash mismatch")
    frame = pl.read_parquet(path)
    required = {"building_id", "repeat", "seed", "split", "fold", "row_count", "anomaly_count", "normal_count"}
    if not required <= set(frame.columns):
        raise ValueError("split membership/support columns missing")
    if frame.height != 600 or frame.select("repeat", "building_id").is_duplicated().any():
        raise ValueError("invalid master membership count/duplicates")
    if sorted(frame["repeat"].unique()) != [0, 1, 2]:
        raise ValueError("invalid repeat IDs")
    entries = metadata["per_repeat"]
    if (not isinstance(entries, list) or len(entries) != 3
            or any(not isinstance(e, Mapping) for e in entries)
            or [entry["repeat"] for entry in entries] != [0, 1, 2]):
        raise ValueError("invalid per-repeat metadata")
    vault_values = metadata["vault_building_ids"]
    if not isinstance(vault_values, list) or any(type(v) is not int for v in vault_values):
        raise ValueError("invalid vault metadata IDs")
    vault_ids = set(vault_values)
    if len(vault_ids) != 40:
        raise ValueError("invalid vault metadata")
    dev_ids = None
    partitions = set()
    for repeat, entry in enumerate(entries):
        part_path = referenced_file(root, entry["path"], "per-repeat")
        if file_sha256(part_path) != entry["sha256"] or entry["seed"] != 42 + repeat:
            raise ValueError("per-repeat hash/seed mismatch")
        part = frame.filter(pl.col("repeat") == repeat)
        if not pl.read_parquet(part_path).sort("building_id").equals(part.sort("building_id")):
            raise ValueError("per-repeat membership differs from master")
        support_columns = ["building_id", "split", "fold", "row_count", "anomaly_count",
                           "normal_count", "anomaly_prevalence", "support_stratum"]
        recorded = entry["per_building_support"]
        if (not isinstance(recorded, list) or len(recorded) != 200
                or any(not isinstance(r, Mapping) for r in recorded)):
            raise ValueError("per-building support incomplete")
        observed = part.select(support_columns).sort("building_id").to_dicts()
        if sorted(recorded, key=lambda row: row["building_id"]) != observed:
            raise ValueError("per-building membership/support metadata mismatch")
        if part["seed"].unique().to_list() != [42 + repeat] or set(part["split"]) != {"dev", "vault"}:
            raise ValueError("invalid seed or split scope")
        dev = part.filter(pl.col("split") == "dev")
        vault = part.filter(pl.col("split") == "vault")
        current_dev = set(dev["building_id"])
        if dev.height != 160 or vault.height != 40 or set(vault["building_id"]) != vault_ids:
            raise ValueError("vault/development membership mismatch")
        if dev_ids is not None and current_dev != dev_ids:
            raise ValueError("development membership changed across repeats")
        dev_ids = current_dev
        if vault["fold"].null_count() != 40 or dev["fold"].null_count() or set(dev["fold"]) != {0, 1, 2, 3}:
            raise ValueError("invalid fold assignments")
        support = _mapping(entry["class_support"], "class support")
        folds = support["development_folds"]
        if (not isinstance(folds, list) or len(folds) != 4
                or any(not isinstance(s, Mapping) for s in folds)
                or {s["fold"] for s in folds} != {0, 1, 2, 3}):
            raise ValueError("per-repeat fold support incomplete")
        _check_support(vault, support["vault"])
        for summary in folds:
            _check_support(dev.filter(pl.col("fold") == summary["fold"]), summary)
        partition = tuple(sorted(tuple(sorted(dev.filter(pl.col("fold") == f)["building_id"])) for f in range(4)))
        if partition in partitions:
            raise ValueError("repeats are copied partitions")
        partitions.add(partition)
    return {"repeat_count": 3, "vault_building_count": 40, "split_sha256": digest}


def _check_support(frame: Any, summary: Mapping[str, Any]) -> None:
    summary = _mapping(summary, "partition support")
    rows = frame["row_count"].sum()
    positives = frame["anomaly_count"].sum()
    expected = {"building_count": frame.height, "row_count": rows,
                "positive_rows": positives, "negative_rows": rows - positives}
    if (any(summary.get(key) != value for key, value in expected.items())
            or frame["normal_count"].sum() != rows - positives or not 0 < positives < rows):
        raise ValueError("class support mismatch or single-class partition")
