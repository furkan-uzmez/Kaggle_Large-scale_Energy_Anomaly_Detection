"""Artifact-only full campaign review and analytical comparison tools.

This module inspects, validates, and evaluates finalized full campaign artifacts
from disk. It strictly enforces artifact-only analysis: no model training, no parameter
fitting, no threshold searching, and no access to unauthorized datasets.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from project.models.candidate_artifacts import CANDIDATE_OOF_SCHEMA
from project.models.metrics import METRIC_CONTRACT

PROMOTION_THRESHOLDS: dict[str, Any] = {
    "min_roc_auc_delta": 0.001,
    "significance_alpha": 0.05,
    "require_non_negative_pr_auc_delta": True,
}


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_unsealed_campaign_dir(
    resolved_dir: Path, run_status: dict[str, Any], load_oof: bool = True
) -> dict[str, Any]:
    """Load and validate completed experiments from an unsealed or finished-with-blockers campaign."""
    experiments_raw = run_status.get("experiments", [])
    oof_tables: dict[str, pl.DataFrame | None] = {}
    oof_paths: dict[str, Path] = {}
    transport_validations: dict[str, dict[str, Any]] = {}
    probability_candidates: set[str] = set()
    margin_candidates: set[str] = set()
    declarations: dict[str, dict[str, Any]] = {}
    valid_experiments: list[dict[str, Any]] = []
    expected_oof_columns = set(CANDIDATE_OOF_SCHEMA.names)

    for exp_entry in experiments_raw:
        exp_id = exp_entry.get("experiment_id")
        if not exp_id:
            continue
        exp_status = exp_entry.get("status")
        if exp_status not in ("awaiting_validation", "completed"):
            continue
        oof_file = resolved_dir / f"{exp_id}_oof.parquet"
        if not oof_file.is_file():
            continue
        transport_file = resolved_dir / f"{exp_id}_transport_validation.json"
        if not transport_file.is_file():
            continue

        # Verify hash against run_status if available
        expected_hash = exp_entry.get("oof_sha256")
        if expected_hash and _sha256(oof_file) != expected_hash:
            raise ValueError(
                f"sha256 mismatch: artifact '{oof_file.name}' is corrupt or tampered"
            )

        transport_data = json.loads(transport_file.read_text(encoding="utf-8"))
        val_status = transport_data.get("validation_status") or transport_data.get(
            "status"
        )
        if val_status not in ("passed", "validated", "completed"):
            continue

        if load_oof:
            table = pl.read_parquet(oof_file)
            cols = set(table.columns)
            actual_count = len(table)
            oof_tables[exp_id] = table
        else:
            schema = pq.read_schema(oof_file)
            meta = pq.read_metadata(oof_file)
            cols = set(schema.names)
            actual_count = meta.num_rows
            oof_tables[exp_id] = None

        missing_cols = expected_oof_columns - cols
        if missing_cols:
            raise ValueError(
                f"OOF schema violation for {exp_id}: missing columns {missing_cols}"
            )

        expected_count = (
            transport_data.get("record_count")
            or transport_data.get("oof_record_count")
            or transport_data.get("expected_row_repeat_count")
        )
        if expected_count is not None and actual_count != expected_count:
            raise ValueError(
                f"OOF row count mismatch for {exp_id}: expected {expected_count}, got {actual_count}"
            )

        oof_paths[exp_id] = oof_file
        transport_validations[exp_id] = transport_data

        decl_file = resolved_dir / f"{exp_id}_declaration.json"
        if decl_file.is_file():
            decl_data = json.loads(decl_file.read_text(encoding="utf-8"))
            declarations[exp_id] = decl_data
            cand_dict = decl_data.get("candidates", {})
            for cand_id, cand_meta in cand_dict.items():
                score_kind = cand_meta.get("score_kind", "positive_probability")
                if score_kind == "positive_probability":
                    probability_candidates.add(cand_id)
                elif score_kind == "decision_margin":
                    margin_candidates.add(cand_id)

        valid_experiments.append(exp_entry)

    overlap = probability_candidates.intersection(margin_candidates)
    if overlap:
        raise ValueError(
            f"Candidate cannot be both probability and margin score: {overlap}"
        )

    return {
        "run_id": run_status.get("run_id", resolved_dir.name),
        "status": "finished_with_blockers",
        "campaign_dir": resolved_dir,
        "manifest": {"candidates": valid_experiments},
        "receipt": {"artifact_sha256": {}},
        "run_status": run_status,
        "experiments": valid_experiments,
        "oof_tables": oof_tables,
        "oof_paths": oof_paths,
        "transport_validations": transport_validations,
        "probability_candidates": sorted(probability_candidates),
        "margin_candidates": sorted(margin_candidates),
        "declarations": declarations,
        "feature_metadata_path": None,
    }


def _load_single_campaign_dir(
    resolved_dir: Path,
    allow_finished_with_blockers: bool = False,
    load_oof: bool = True,
) -> dict[str, Any]:
    """Load and strictly validate a single finalized campaign directory."""
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Campaign directory does not exist: {resolved_dir}")
    if not resolved_dir.is_dir():
        raise NotADirectoryError(f"Campaign path is not a directory: {resolved_dir}")

    # Validate run status
    status_file = resolved_dir / "run_status.json"
    if not status_file.is_file():
        raise FileNotFoundError(f"run_status.json not found in campaign: {status_file}")
    run_status = json.loads(status_file.read_text(encoding="utf-8"))
    status = run_status.get("status")

    # Locate completion directory or staged completion directory
    completion_dir = resolved_dir / "completion"
    if not completion_dir.is_dir():
        stages = sorted(resolved_dir.glob(".completion-stage-*"))
        if stages and (stages[-1] / "receipt.json").is_file():
            completion_dir = stages[-1]

    if completion_dir.is_dir() and (completion_dir / "run_status.json").is_file():
        run_status = json.loads(
            (completion_dir / "run_status.json").read_text(encoding="utf-8")
        )
        status = run_status.get("status")

    receipt_file = completion_dir / "receipt.json" if completion_dir.is_dir() else None
    manifest_file = (
        completion_dir / "manifest.json" if completion_dir.is_dir() else None
    )

    if status in ("running", "interrupted"):
        if not allow_finished_with_blockers:
            raise ValueError(
                f"Campaign is not finalized; status is '{status}' (still running, interrupted, or incomplete)"
            )
        return _load_unsealed_campaign_dir(resolved_dir, run_status, load_oof=load_oof)

    if not receipt_file or not receipt_file.is_file():
        if not allow_finished_with_blockers:
            raise FileNotFoundError(
                f"Missing receipt.json in completion directory: {resolved_dir / 'completion' / 'receipt.json'}"
            )
        # Handle finished-with-blockers campaign where receipt is not sealed
        return _load_unsealed_campaign_dir(resolved_dir, run_status, load_oof=load_oof)
    if not manifest_file or not manifest_file.is_file():
        raise FileNotFoundError(
            f"Missing manifest.json in completion directory: {completion_dir / 'manifest.json'}"
        )

    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    # Verify cryptographic hashes against receipt
    artifact_hashes = receipt.get("artifact_sha256", {})
    for rel_path, expected_hash in artifact_hashes.items():
        artifact_path = resolved_dir / rel_path
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Receipt references missing artifact: {rel_path}")
        actual_hash = _sha256(artifact_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"sha256 mismatch: artifact '{rel_path}' is corrupt or tampered "
                f"(expected {expected_hash}, got {actual_hash})"
            )
    # Check manifest-referenced feature_metadata hash if present
    meta_path = resolved_dir / "feature_metadata.parquet"
    manifest_meta_hash = manifest.get("feature_metadata_sha256")
    if meta_path.is_file() and manifest_meta_hash:
        actual_meta_hash = _sha256(meta_path)
        if actual_meta_hash != manifest_meta_hash:
            raise ValueError(
                f"sha256 mismatch: feature_metadata.parquet is corrupt or tampered "
                f"(expected {manifest_meta_hash}, got {actual_meta_hash})"
            )

    # Process and validate experiments
    experiments = manifest.get("candidates", [])
    oof_tables: dict[str, pl.DataFrame | None] = {}
    oof_paths: dict[str, Path] = {}
    transport_validations: dict[str, dict[str, Any]] = {}
    probability_candidates: set[str] = set()
    margin_candidates: set[str] = set()
    declarations: dict[str, dict[str, Any]] = {}

    expected_oof_columns = set(CANDIDATE_OOF_SCHEMA.names)
    for exp_entry in experiments:
        exp_id = exp_entry.get("experiment_id")
        if not exp_id:
            continue

        oof_file = resolved_dir / f"{exp_id}_oof.parquet"
        if not oof_file.is_file():
            raise FileNotFoundError(f"Experiment OOF file missing: {oof_file}")

        # Mandatory transport validation check
        transport_file = resolved_dir / f"{exp_id}_transport_validation.json"
        if not transport_file.is_file():
            raise FileNotFoundError(
                f"Mandatory transport validation missing for {exp_id}: {transport_file}"
            )

        transport_data = json.loads(transport_file.read_text(encoding="utf-8"))
        val_status = transport_data.get("validation_status") or transport_data.get(
            "status"
        )
        if val_status not in ("passed", "validated", "completed"):
            raise ValueError(
                f"Transport validation status for {exp_id} is '{val_status}', expected 'passed' or 'validated'"
            )

        if load_oof:
            df = pl.read_parquet(oof_file)
            actual_count = len(df)
            actual_columns = set(df.columns)
            oof_tables[exp_id] = df
        else:
            schema = pq.read_schema(oof_file)
            meta = pq.read_metadata(oof_file)
            actual_count = meta.num_rows
            actual_columns = set(schema.names)
            oof_tables[exp_id] = None

        expected_count = (
            transport_data.get("record_count")
            or transport_data.get("oof_record_count")
            or transport_data.get("expected_row_repeat_count")
        )
        if expected_count is not None and actual_count != expected_count:
            raise ValueError(
                f"Row count mismatch in {exp_id} OOF table: "
                f"expected record_count {expected_count}, got {actual_count}"
            )
        oof_paths[exp_id] = oof_file
        transport_validations[exp_id] = transport_data

        # Cross-check manifest oof_sha256 if present
        manifest_oof_sha = exp_entry.get("oof_sha256")
        if manifest_oof_sha:
            computed_oof_sha = _sha256(oof_file)
            if computed_oof_sha != manifest_oof_sha:
                raise ValueError(
                    f"sha256 mismatch: manifest oof_sha256 for {exp_id} "
                    f"(expected {manifest_oof_sha}, got {computed_oof_sha})"
                )

        # Mandatory source lineage check
        lineage_file = resolved_dir / f"{exp_id}_source_lineage.json"
        if not lineage_file.is_file():
            raise FileNotFoundError(
                f"Mandatory source lineage missing for {exp_id}: {lineage_file}"
            )

        # Validate schema columns
        if not expected_oof_columns.issubset(actual_columns):
            missing_cols = expected_oof_columns - actual_columns
            raise ValueError(
                f"OOF table for {exp_id} missing canonical columns: {missing_cols}"
            )
        # Mandatory declaration check
        decl_file = resolved_dir / f"{exp_id}_declaration.json"
        if not decl_file.is_file():
            raise FileNotFoundError(
                f"Mandatory declaration file missing for {exp_id}: {decl_file}"
            )

        decl_data = json.loads(decl_file.read_text(encoding="utf-8"))
        declarations[exp_id] = decl_data
        cand_declarations = decl_data.get("candidates", {})
        for cand_id, cand_meta in cand_declarations.items():
            score_kind = cand_meta.get("score_kind")
            if score_kind == "positive_probability":
                probability_candidates.add(cand_id)
            elif score_kind == "decision_margin":
                margin_candidates.add(cand_id)

    # Ensure mutual exclusivity
    overlap = probability_candidates.intersection(margin_candidates)
    if overlap:
        raise ValueError(
            f"Candidate cannot have multiple contradictory score kinds: {overlap}"
        )

    run_id = manifest.get("run_id") or run_status.get("run_id")

    return {
        "run_id": run_id,
        "status": status,
        "campaign_dir": resolved_dir,
        "manifest": manifest,
        "receipt": receipt,
        "run_status": run_status,
        "experiments": experiments,
        "declarations": declarations,
        "oof_tables": oof_tables,
        "oof_paths": oof_paths,
        "transport_validations": transport_validations,
        "probability_candidates": sorted(probability_candidates),
        "margin_candidates": sorted(margin_candidates),
        "feature_metadata_path": meta_path if meta_path.is_file() else None,
    }


def discover_completed_model_runs(
    models_root: Path | None = None,
    candidates_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Discover completed full-run candidate models with verified OOF artifacts.

    Scans model directories for completed runs adhering to CANDIDATE_OOF_SCHEMA,
    requiring n03_eligible == True, non_comparative == False, and score_kind == 'positive_probability'.

    Args:
        models_root: Path to outputs/models root, or None to use default.
        candidates_root: Path to outputs/candidates root, or None to use default.

    Returns:
        List of completed model run descriptors with verified OOF paths and companion metadata.
    """
    m_root = Path("outputs/models") if models_root is None else Path(models_root)
    c_root = (
        Path("outputs/candidates") if candidates_root is None else Path(candidates_root)
    )

    expected_cols = set(CANDIDATE_OOF_SCHEMA.names)
    candidate_runs: list[dict[str, Any]] = []

    search_roots = []
    if m_root.is_dir():
        search_roots.append((m_root, "models"))
    if c_root.is_dir():
        search_roots.append((c_root, "candidates"))

    for root_dir, _ in search_roots:
        for oof_path in sorted(root_dir.rglob("*oof*.parquet")):
            parts = set(oof_path.parts)
            if any(s in parts for s in ("smoke", "synthetic_smoke", ".tmp")):
                continue
            if "full" not in oof_path.as_posix():
                continue

            try:
                schema = pq.read_schema(oof_path)
            except Exception as e:
                raise ValueError(
                    f"Corrupted or unreadable OOF parquet schema in {oof_path}: {e}"
                ) from e

            if not expected_cols.issubset(set(schema.names)):
                continue

            # Validate canonical schema field types and non-nullability
            schema_mismatch = False
            for field in CANDIDATE_OOF_SCHEMA:
                actual_field = schema.field(field.name)
                if field.name == "building_id":
                    if actual_field.type not in (pa.string(), pa.int64()):
                        schema_mismatch = True
                        break
                elif field.name == "timestamp":
                    if not pa.types.is_timestamp(actual_field.type):
                        schema_mismatch = True
                        break
                else:
                    if actual_field.type != field.type:
                        schema_mismatch = True
                        break
                if actual_field.nullable:
                    schema_mismatch = True
                    break
            if schema_mismatch:
                continue
            try:
                meta = pl.read_parquet(
                    oof_path,
                    columns=[
                        "candidate_id",
                        "arm_id",
                        "score_kind",
                        "n03_eligible",
                        "non_comparative",
                        "execution_mode",
                    ],
                )
            except Exception as e:
                raise ValueError(
                    f"Corrupted or unreadable OOF parquet table in {oof_path}: {e}"
                ) from e

            col_n03 = meta["n03_eligible"]
            if col_n03.null_count() > 0 or not col_n03.all(ignore_nulls=False):
                continue
            col_nc = meta["non_comparative"]
            if col_nc.null_count() > 0 or not (col_nc == False).all(ignore_nulls=False):
                continue
            col_sk = meta["score_kind"]
            if col_sk.null_count() > 0 or not (col_sk == "positive_probability").all(
                ignore_nulls=False
            ):
                continue
            if "execution_mode" in meta.columns:
                col_em = meta["execution_mode"]
                if col_em.null_count() > 0 or not (col_em == "full").all(
                    ignore_nulls=False
                ):
                    continue

            candidates = meta["candidate_id"].unique().to_list()
            arms = meta["arm_id"].unique().to_list()
            run_dir = oof_path.parent

            rel_parts = oof_path.relative_to(root_dir).parts
            category = rel_parts[0] if len(rel_parts) > 1 else "model"

            stem = (
                oof_path.name.replace("_candidate_oof.parquet", "")
                .replace("_oof.parquet", "")
                .replace(".parquet", "")
            )
            specific_manifest = run_dir / f"{stem}_manifest.json"
            if category in ("tabular_mlp", "mlp"):
                manifest_file = (
                    specific_manifest if specific_manifest.is_file() else None
                )
            else:
                manifest_file = (
                    specific_manifest
                    if specific_manifest.is_file()
                    else next(run_dir.glob("*manifest.json"), None)
                )
            specific_transport = run_dir / f"{stem}_transport_validation.json"
            if category in (
                "gbdt_feature_selection",
                "gbdt_loss_objectives",
                "synthetic_sampling",
            ):
                transport_file = (
                    specific_transport if specific_transport.is_file() else None
                )
            else:
                transport_file = (
                    specific_transport
                    if specific_transport.is_file()
                    else next(run_dir.glob("*transport_validation.json"), None)
                )
            summary_file = (
                run_dir / "run_summary.json"
                if (run_dir / "run_summary.json").is_file()
                else None
            )

            # Require producer's matching completion artifact per category convention
            if category in ("gbdt_standalone", "tabular_mlp", "mlp"):
                if not manifest_file or not manifest_file.is_file():
                    continue
            elif category in (
                "gbdt_feature_selection",
                "gbdt_loss_objectives",
                "synthetic_sampling",
            ):
                if not transport_file or not transport_file.is_file():
                    continue
            elif "autogluon" in category or "autogluon" in oof_path.as_posix():
                if not summary_file or not summary_file.is_file():
                    continue
            else:
                if not (
                    (manifest_file and manifest_file.is_file())
                    or (transport_file and transport_file.is_file())
                    or (summary_file and summary_file.is_file())
                ):
                    continue

            manifest_data: dict[str, Any] = {}
            if manifest_file and manifest_file.is_file():
                try:
                    manifest_data = json.loads(
                        manifest_file.read_text(encoding="utf-8")
                    )
                except Exception as e:
                    raise ValueError(
                        f"Corrupted manifest JSON in {manifest_file}: {e}"
                    ) from e

                m_status = manifest_data.get("status")
                if m_status is not None and m_status not in (
                    "completed",
                    "passed",
                    "validated",
                ):
                    continue
                if manifest_data.get("execution_mode") not in (None, "full"):
                    continue
                if (
                    manifest_data.get("dry_run") is True
                    or manifest_data.get("smoke") is True
                ):
                    continue
                if manifest_data.get("n03_eligible") is False:
                    continue
                if manifest_data.get("non_comparative") is True:
                    continue

            transport_data: dict[str, Any] = {}
            if transport_file and transport_file.is_file():
                try:
                    transport_data = json.loads(
                        transport_file.read_text(encoding="utf-8")
                    )
                except Exception as e:
                    raise ValueError(
                        f"Corrupted transport JSON in {transport_file}: {e}"
                    ) from e

                t_status = transport_data.get(
                    "validation_status"
                ) or transport_data.get("status")
                if t_status is not None and t_status not in (
                    "passed",
                    "validated",
                    "completed",
                ):
                    continue
                if transport_data.get("execution_mode") not in (None, "full"):
                    continue
                if (
                    transport_data.get("dry_run") is True
                    or transport_data.get("smoke") is True
                ):
                    continue
                if transport_data.get("n03_eligible") is False:
                    continue
                if transport_data.get("non_comparative") is True:
                    continue

            summary_data: dict[str, Any] = {}
            if summary_file and summary_file.is_file():
                try:
                    summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
                except Exception as e:
                    raise ValueError(
                        f"Corrupted run_summary JSON in {summary_file}: {e}"
                    ) from e

                s_status = summary_data.get("status")
                if s_status is not None and s_status != "completed":
                    continue
                if s_status is None and (
                    "autogluon" in category or "autogluon" in oof_path.as_posix()
                ):
                    continue
                if (
                    summary_data.get("dry_run") is True
                    or summary_data.get("smoke") is True
                ):
                    continue
                if summary_data.get("execution_mode") not in (None, "full"):
                    continue

            actual_count = len(meta)
            if actual_count == 0:
                continue

            expected_count = (
                manifest_data.get("records_count")
                or manifest_data.get("record_count")
                or transport_data.get("records_count")
                or transport_data.get("record_count")
                or summary_data.get("oof_record_count")
                or summary_data.get("records_count")
                or summary_data.get("record_count")
            )
            if expected_count is None:
                continue
            if actual_count != expected_count:
                raise ValueError(
                    f"OOF record count mismatch for {oof_path}: expected {expected_count}, got {actual_count}"
                )
            run_id = (
                manifest_data.get("run_id")
                or transport_data.get("run_id")
                or summary_data.get("run_id")
                or run_dir.name
            )

            model_family = (
                manifest_data.get("family")
                or manifest_data.get("model_family")
                or transport_data.get("family")
                or ""
            )
            if not model_family:
                for cand in candidates:
                    for fam in (
                        "catboost",
                        "lightgbm",
                        "xgboost",
                        "extratrees",
                        "autogluon",
                        "tabular_mlp",
                        "mlp",
                    ):
                        if fam in cand.lower() or fam in oof_path.stem.lower():
                            model_family = (
                                "tabular_mlp" if fam in ("tabular_mlp", "mlp") else fam
                            )
                            break
                    if model_family:
                        break
            if not model_family:
                model_family = "unknown"

            if len(arms) > 1 and "control" in arms and "challenger" in arms:
                role = "both"
            elif any(
                a in ("control", "passthrough", "logloss", "natural") for a in arms
            ):
                role = "control"
            else:
                role = "challenger"

            if category == "gbdt_feature_selection":
                experiment_id = f"gbdt_fs_{model_family}"
            elif category == "gbdt_loss_objectives":
                experiment_id = f"gbdt_loss_{model_family}"
            elif category == "synthetic_sampling":
                experiment_id = f"synthetic_sampling_{model_family}"
            elif category in ("tabular_mlp", "mlp"):
                experiment_id = f"tabular_mlp_{run_dir.name}"
            elif category == "gbdt_standalone":
                experiment_id = f"gbdt_{model_family}_standalone"
            elif "autogluon" in category or "autogluon" in oof_path.as_posix():
                experiment_id = f"autogluon_full_stack_{run_id}"
            else:
                experiment_id = f"{category}_{run_dir.name}"

            candidate_runs.append(
                {
                    "run_id": run_id,
                    "experiment_id": experiment_id,
                    "model_family": model_family,
                    "category": category,
                    "candidates": candidates,
                    "arms": arms,
                    "role": role,
                    "score_kind": "positive_probability",
                    "n03_eligible": True,
                    "non_comparative": False,
                    "status": "completed",
                    "comparative_completion_available": True,
                    "records_count": len(meta),
                    "oof_path": oof_path,
                    "manifest_path": manifest_file,
                    "transport_validation_path": transport_file,
                    "run_summary_path": summary_file,
                    "manifest_data": manifest_data,
                    "transport_data": transport_data,
                    "summary_data": summary_data,
                    "st_mtime": run_dir.stat().st_mtime,
                }
            )

    # Deduplicate reruns: select latest completed run by directory modification time (st_mtime)
    candidate_runs.sort(
        key=lambda r: (r["st_mtime"], r["oof_path"].as_posix()),
        reverse=True,
    )

    discovered_runs: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    seen_family_arms: set[tuple[str, str, str]] = set()

    for r in candidate_runs:
        cand_already_seen = any(c in seen_candidates for c in r["candidates"])
        arm_already_seen = any(
            (r["category"], r["model_family"], a) in seen_family_arms
            for a in r["arms"]
        )
        if cand_already_seen or arm_already_seen:
            continue

        seen_candidates.update(r["candidates"])
        for a in r["arms"]:
            seen_family_arms.add((r["category"], r["model_family"], a))
        discovered_runs.append(r)

    discovered_runs.sort(key=lambda r: r["oof_path"].as_posix())
    return discovered_runs


def _group_model_runs_into_experiments(
    runs: list[dict[str, Any]],
    load_oof: bool = False,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, pl.DataFrame | None],
    dict[str, Path | list[Path]],
    dict[str, dict[str, Any]],
    list[str],
    list[str],
]:
    """Group discovered model runs into structured experiment suites."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        grouped[r["experiment_id"]].append(r)

    experiments: list[dict[str, Any]] = []
    declarations: dict[str, dict[str, Any]] = {}
    oof_tables: dict[str, pl.DataFrame | None] = {}
    oof_paths: dict[str, Path | list[Path]] = {}
    transport_validations: dict[str, dict[str, Any]] = {}
    probability_candidates: set[str] = set()
    margin_candidates: set[str] = set()

    for exp_id, exp_runs in sorted(grouped.items()):
        primary_run = exp_runs[0]
        exp_entry: dict[str, Any] = {
            "experiment_id": exp_id,
            "status": "completed",
            "comparative_completion_available": True,
            "n03_eligible": True,
            "model_family": primary_run["model_family"],
            "records_count": sum(r["records_count"] for r in exp_runs),
            "provenance": "discovered_model",
        }
        for r in exp_runs:
            m_data = r.get("manifest_data") or {}
            for k in (
                "split_sha256",
                "data_sha256",
                "config_sha256",
                "model_fit_config_sha256",
            ):
                if k in m_data and k not in exp_entry:
                    exp_entry[k] = m_data[k]

        experiments.append(exp_entry)

        decl_candidates: dict[str, dict[str, Any]] = {}
        for r in exp_runs:
            if r["role"] == "both":
                for cid in r["candidates"]:
                    arm = (
                        "control"
                        if (cid.endswith("_control") or "control" in cid)
                        else "challenger"
                    )
                    decl_candidates[cid] = {
                        "arm_id": arm,
                        "role": arm,
                        "score_kind": r["score_kind"],
                    }
                    if r["score_kind"] == "positive_probability":
                        probability_candidates.add(cid)
                    else:
                        margin_candidates.add(cid)
            else:
                for cid in r["candidates"]:
                    arm = "control" if r["role"] == "control" else "challenger"
                    decl_candidates[cid] = {
                        "arm_id": arm,
                        "role": arm,
                        "score_kind": r["score_kind"],
                        "raw_arm_id": r["arms"][0] if r["arms"] else arm,
                    }
                    if r["score_kind"] == "positive_probability":
                        probability_candidates.add(cid)
                    else:
                        margin_candidates.add(cid)

        declarations[exp_id] = {
            "experiment_id": exp_id,
            "provenance": "discovered_model",
            "candidates": decl_candidates,
        }

        paths = [r["oof_path"] for r in exp_runs]
        oof_paths[exp_id] = paths[0] if len(paths) == 1 else paths

        if load_oof:
            if len(paths) == 1:
                oof_tables[exp_id] = pl.read_parquet(paths[0])
            else:
                oof_tables[exp_id] = pl.concat([pl.read_parquet(p) for p in paths])
        else:
            oof_tables[exp_id] = None

        for r in exp_runs:
            t_data = r.get("transport_data")
            if t_data:
                transport_validations[exp_id] = t_data
                break
            m_data = r.get("manifest_data")
            if m_data and exp_id not in transport_validations:
                transport_validations[exp_id] = dict(m_data)
                transport_validations[exp_id].setdefault("validation_status", "passed")

    return (
        experiments,
        declarations,
        oof_tables,
        oof_paths,
        transport_validations,
        sorted(probability_candidates),
        sorted(margin_candidates),
    )


def _merge_discovered_models(
    campaign: dict[str, Any],
    models_root: Path | None = None,
    candidates_root: Path | None = None,
    load_oof: bool = True,
) -> None:
    """Merge discovered model runs directly into an existing campaign dictionary."""
    model_runs = discover_completed_model_runs(
        models_root=models_root, candidates_root=candidates_root
    )
    if not model_runs:
        return
    (
        exp_list,
        decls,
        oofs,
        paths,
        transports,
        prob_cands,
        margin_cands,
    ) = _group_model_runs_into_experiments(model_runs, load_oof=load_oof)

    campaign_exps = campaign.setdefault("experiments", [])
    for exp in exp_list:
        if exp not in campaign_exps:
            campaign_exps.append(exp)

    if "manifest" in campaign and isinstance(campaign["manifest"], dict):
        manifest_cands = campaign["manifest"].setdefault("candidates", [])
        for exp in exp_list:
            if exp not in manifest_cands:
                manifest_cands.append(exp)

    campaign.setdefault("declarations", {}).update(decls)
    campaign.setdefault("oof_tables", {}).update(oofs)
    campaign.setdefault("oof_paths", {}).update(paths)
    campaign.setdefault("transport_validations", {}).update(transports)

    all_prob = set(campaign.get("probability_candidates", [])) | set(prob_cands)
    all_margin = set(campaign.get("margin_candidates", [])) | set(margin_cands)
    overlap = all_prob.intersection(all_margin)
    if overlap:
        raise ValueError(
            f"Candidate cannot have multiple contradictory score kinds across roots: {overlap}"
        )
    campaign["probability_candidates"] = sorted(all_prob)
    campaign["margin_candidates"] = sorted(all_margin)


def load_finalized_campaign(
    campaign_dir: Path | str | list[Path | str] | None = None,
    allow_finished_with_blockers: bool = False,
    load_oof: bool = True,
    include_models: bool = False,
    models_root: Path | None = None,
    candidates_root: Path | None = None,
) -> dict[str, Any]:
    """Load and validate completed campaign artifacts against manifests and receipts.

    Validates:
    - Root run status is strictly completed (or finished_with_blockers if allowed).
    - Manifest and receipt files exist.
    - Every artifact checksum in receipt matches actual file content on disk.
    - Mandatory transport validation, declaration, and source lineage files exist.
    - Transport validation status is strictly 'passed' or 'validated' and record count matches OOF table.
    - Candidates are cleanly partitioned into probability candidates and margin candidates.

    Args:
        campaign_dir: Path to campaign directory, list of paths, or None to discover default.
        allow_finished_with_blockers: If True, allow loading finished experiments from campaigns
            with unresolved blockers or unsealed receipts.
        load_oof: If True, eagerly read OOF parquet files into polars DataFrames.
            If False, validate schema and metadata without eager array loading.

    Returns:
        Dictionary containing campaign metadata, verified OOF tables, and candidate pools.

    Raises:
        FileNotFoundError: If required manifest, receipt, or artifact files are missing.
        ValueError: If campaign is incomplete/running/interrupted, or if checksum/row count fails.
    """
    if campaign_dir is None:
        default_dir = Path("outputs/campaigns")
        if default_dir.is_dir():
            candidates = [p for p in default_dir.iterdir() if p.is_dir()]
            completed = [
                p for p in candidates if (p / "completion" / "receipt.json").is_file()
            ]
            target_dirs = completed or candidates
            if not target_dirs:
                raise FileNotFoundError(
                    "No campaign directories found in outputs/campaigns"
                )
            resolved_dirs = [max(target_dirs, key=lambda p: p.stat().st_mtime)]
        else:
            raise FileNotFoundError(
                "Default campaign directory outputs/campaigns does not exist"
            )
    elif isinstance(campaign_dir, (list, tuple)):
        if not campaign_dir:
            raise ValueError("campaign_dir list is empty")
        resolved_dirs = [Path(p) for p in campaign_dir]
    else:
        resolved_dirs = [Path(campaign_dir)]

    if len(resolved_dirs) == 1:
        campaign = _load_single_campaign_dir(
            resolved_dirs[0],
            allow_finished_with_blockers=allow_finished_with_blockers,
            load_oof=load_oof,
        )
        if include_models:
            _merge_discovered_models(
                campaign,
                models_root=models_root,
                candidates_root=candidates_root,
                load_oof=load_oof,
            )
        return campaign
    # Multiple campaign directories: validate all and merge
    loaded_campaigns = [
        _load_single_campaign_dir(
            d,
            allow_finished_with_blockers=allow_finished_with_blockers,
            load_oof=load_oof,
        )
        for d in resolved_dirs
    ]
    primary = loaded_campaigns[0]

    merged_experiments = list(primary["experiments"])
    merged_oof_tables = dict(primary.get("oof_tables", {}))
    merged_oof_paths = dict(primary.get("oof_paths", {}))
    merged_transport = dict(primary.get("transport_validations", {}))
    merged_declarations = dict(primary.get("declarations", {}))
    merged_prob_cands = set(primary.get("probability_candidates", []))
    merged_margin_cands = set(primary.get("margin_candidates", []))

    for other in loaded_campaigns[1:]:
        for exp in other.get("experiments", []):
            if exp not in merged_experiments:
                merged_experiments.append(exp)
        merged_oof_tables.update(other.get("oof_tables", {}))
        merged_oof_paths.update(other.get("oof_paths", {}))
        merged_transport.update(other.get("transport_validations", {}))
        merged_declarations.update(other.get("declarations", {}))
        merged_prob_cands.update(other.get("probability_candidates", []))
        merged_margin_cands.update(other.get("margin_candidates", []))

    overlap = merged_prob_cands.intersection(merged_margin_cands)
    if overlap:
        raise ValueError(
            f"Candidate cannot have multiple contradictory score kinds across roots: {overlap}"
        )

    campaign = {
        "run_id": primary["run_id"],
        "status": primary["status"],
        "campaign_dir": primary["campaign_dir"],
        "manifest": primary["manifest"],
        "receipt": primary["receipt"],
        "run_status": primary["run_status"],
        "experiments": merged_experiments,
        "declarations": merged_declarations,
        "oof_tables": merged_oof_tables,
        "oof_paths": merged_oof_paths,
        "transport_validations": merged_transport,
        "probability_candidates": sorted(merged_prob_cands),
        "margin_candidates": sorted(merged_margin_cands),
        "feature_metadata_path": primary.get("feature_metadata_path"),
    }

    if include_models:
        _merge_discovered_models(
            campaign,
            models_root=models_root,
            candidates_root=candidates_root,
            load_oof=load_oof,
        )

    return campaign


def load_unified_campaign_pool(
    campaign_targets: Path | str | list[Path | str] | None = None,
    include_models: bool = True,
    models_root: Path | None = None,
    candidates_root: Path | None = None,
    allow_finished_with_blockers: bool = False,
    load_oof: bool = False,
) -> dict[str, Any]:
    """Load unified campaign pool combining finalized campaigns and discovered model runs.

    Args:
        campaign_targets: Path, list of paths, or None to discover default campaigns.
        include_models: Whether to discover and ingest completed full-run candidate models.
        models_root: Optional custom root for outputs/models.
        candidates_root: Optional custom root for outputs/candidates.
        allow_finished_with_blockers: Allow loading campaigns with blockers.
        load_oof: Whether to eagerly load OOF arrays into polars DataFrames.

    Returns:
        Unified campaign dictionary containing all verified experiments and candidate models.
    """
    has_explicit_campaign = campaign_targets is not None
    has_default_campaign = False
    if not has_explicit_campaign and models_root is None and candidates_root is None:
        default_dir = Path("outputs/campaigns")
        if default_dir.is_dir() and any(p.is_dir() for p in default_dir.iterdir()):
            has_default_campaign = True
    if has_explicit_campaign or has_default_campaign:
        return load_finalized_campaign(
            campaign_dir=campaign_targets,
            allow_finished_with_blockers=allow_finished_with_blockers,
            load_oof=load_oof,
            include_models=include_models,
            models_root=models_root,
            candidates_root=candidates_root,
        )

    if not include_models:
        raise FileNotFoundError(
            "No campaign directories found and include_models is False"
        )

    model_runs = discover_completed_model_runs(
        models_root=models_root, candidates_root=candidates_root
    )
    if not model_runs:
        raise FileNotFoundError("No completed model runs or campaign directories found")

    (
        exp_list,
        decls,
        oofs,
        paths,
        transports,
        prob_cands,
        margin_cands,
    ) = _group_model_runs_into_experiments(model_runs, load_oof=load_oof)

    return {
        "run_id": "unified_models_pool",
        "status": "completed",
        "campaign_dir": None,
        "manifest": {"candidates": exp_list},
        "receipt": {"artifact_sha256": {}},
        "run_status": {"status": "completed", "experiments": exp_list},
        "experiments": exp_list,
        "declarations": decls,
        "oof_tables": oofs,
        "oof_paths": paths,
        "transport_validations": transports,
        "probability_candidates": sorted(prob_cands),
        "margin_candidates": sorted(margin_cands),
        "feature_metadata_path": None,
    }


def nadeau_bengio_t_test(
    differences: np.ndarray | list[float],
    test_ratio: float = 1.0 / 3.0,
) -> dict[str, Any]:
    """Compute Nadeau-Bengio corrected resampled t-test for cross-validation deltas.

    Adjusts standard error for cross-validation sample non-independence using:
        corrected_var = s^2 * (1/n + test_ratio)
    where test_ratio is n_val / n_train (e.g. 1 / (K - 1)).

    Args:
        differences: Paired metric differences across all repeat/fold partitions.
        test_ratio: Ratio of test set size to train set size (default 1/3 for 4-fold CV).

    Returns:
        Dictionary containing mean_delta, corrected_std_error, t_stat, p_value, df,
        and 95% confidence intervals.
    """
    diff_arr = np.asarray(differences, dtype=float)
    n = len(diff_arr)
    if n == 0 or not np.all(np.isfinite(diff_arr)):
        raise ValueError("differences must be nonempty and contain only finite values")
    if test_ratio < 0 or not math.isfinite(test_ratio):
        raise ValueError("test_ratio must be a finite non-negative float")

    df = max(1, n - 1)
    mean_d = float(np.mean(diff_arr))

    if n < 2:
        return {
            "mean_delta": mean_d,
            "corrected_std_error": 0.0,
            "t_stat": 0.0,
            "p_value": 1.0,
            "df": df,
            "ci_95_lower": mean_d,
            "ci_95_upper": mean_d,
        }

    if np.all(diff_arr == diff_arr[0]):
        if math.isclose(mean_d, 0.0, abs_tol=1e-12):
            t_stat = 0.0
            p_val = 1.0
        else:
            t_stat = math.copysign(float("inf"), mean_d)
            p_val = 0.0
        return {
            "mean_delta": mean_d,
            "corrected_std_error": 0.0,
            "t_stat": t_stat,
            "p_value": p_val,
            "df": df,
            "ci_95_lower": mean_d,
            "ci_95_upper": mean_d,
        }

    s2 = float(np.var(diff_arr, ddof=1))
    corrected_var = s2 * ((1.0 / n) + test_ratio)
    corrected_se = math.sqrt(max(0.0, corrected_var))

    if math.isclose(corrected_se, 0.0, abs_tol=1e-12):
        if math.isclose(mean_d, 0.0, abs_tol=1e-12):
            t_stat = 0.0
            p_val = 1.0
        else:
            t_stat = math.copysign(float("inf"), mean_d)
            p_val = 0.0
        return {
            "mean_delta": mean_d,
            "corrected_std_error": 0.0,
            "t_stat": t_stat,
            "p_value": p_val,
            "df": df,
            "ci_95_lower": mean_d,
            "ci_95_upper": mean_d,
        }

    t_stat = mean_d / corrected_se
    p_val = 2.0 * float(stats.t.sf(abs(t_stat), df=df))
    t_crit = float(stats.t.ppf(0.975, df=df))
    ci_lower = mean_d - t_crit * corrected_se
    ci_upper = mean_d + t_crit * corrected_se

    return {
        "mean_delta": mean_d,
        "corrected_std_error": corrected_se,
        "t_stat": t_stat,
        "p_value": p_val,
        "df": df,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
    }


def holm_bonferroni_correction(
    raw_p_values: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Compute Holm-Bonferroni step-down multiple testing correction.

    Guarantees monotonicity and bounds in [0, 1], preserving input sequence order.

    Args:
        raw_p_values: List of unadjusted p-values from statistical tests.
        alpha: Family-wise error rate significance cutoff.

    Returns:
        Dictionary containing adjusted_p_values, rejected boolean list, and alpha.
    """
    m = len(raw_p_values)
    if m == 0:
        return {"adjusted_p_values": [], "rejected": [], "alpha": alpha}

    # Sort p-values tracking original positions
    indexed = sorted(enumerate(raw_p_values), key=lambda x: x[1])

    sorted_adj = [0.0] * m
    running_max = 0.0

    for k, (_, p_val) in enumerate(indexed):
        multiplier = m - k
        adj = p_val * multiplier
        adj = max(adj, running_max)
        adj = min(1.0, max(0.0, adj))
        running_max = adj
        sorted_adj[k] = adj

    adjusted_p_values = [0.0] * m
    rejected = [False] * m

    for (orig_idx, _), p_adj in zip(indexed, sorted_adj, strict=True):
        adjusted_p_values[orig_idx] = p_adj
        rejected[orig_idx] = bool(p_adj <= alpha)

    return {
        "adjusted_p_values": adjusted_p_values,
        "rejected": rejected,
        "alpha": alpha,
    }


def compute_signal_vs_noise_verdict(
    p_value_corrected: float,
    mean_delta: float,
    min_lift: float = 0.0,
) -> str:
    """Classify model comparison into signal, noise, or significant_regression.

    Args:
        p_value_corrected: Multiple-testing-adjusted p-value.
        mean_delta: Average metric gain of challenger over control.
        min_lift: Practical significance threshold.

    Returns:
        One of 'signal', 'noise', or 'significant_regression'.
    """
    if p_value_corrected <= 0.05:
        if mean_delta > min_lift:
            return "signal"
        if mean_delta < -min_lift or mean_delta < 0.0:
            return "significant_regression"
    return "noise"


def _validate_repeated_cv_partitions(
    ctrl_partitions: list[tuple[Any, Any]],
    chall_partitions: list[tuple[Any, Any]],
) -> tuple[bool, str]:
    """Validate repeated cross-validation partitions between control and challenger arms.

    Checks:
    1. Partitions within each arm are distinct (no duplicate (repeat, fold) pairs).
    2. Partitions between control and challenger are paired strictly 1-to-1.
    3. At least 2 distinct repeats are present (repeated CV contract).

    Returns:
        Tuple of (is_valid, failure_reason).
    """
    if not ctrl_partitions or not chall_partitions:
        return False, "Empty partition list"

    # 1. Distinct partitions
    if len(ctrl_partitions) != len(set(ctrl_partitions)):
        return False, "Control arm contains duplicate (repeat, fold) partitions"
    if len(chall_partitions) != len(set(chall_partitions)):
        return False, "Challenger arm contains duplicate (repeat, fold) partitions"

    # 2. Strict 1-to-1 pairing
    if len(ctrl_partitions) != len(chall_partitions) or set(ctrl_partitions) != set(
        chall_partitions
    ):
        return (
            False,
            "Partitions are not paired 1-to-1 between control and challenger arms",
        )

    # 3. At least 2 distinct repeats
    repeats = {r for r, _ in ctrl_partitions if r is not None}
    if len(repeats) < 2:
        return False, f"Insufficient repeats for repeated-CV test: {len(repeats)} < 2"

    return True, ""


def _rows_for_experiment_from_fold_metrics(
    exp_id: str,
    campaign: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract comparison rows for a single experiment from transport validation fold_metrics."""
    transport_validations = campaign.get("transport_validations", {})
    tv = transport_validations.get(exp_id, {})
    fold_metrics = tv.get("fold_metrics", [])
    if not fold_metrics:
        return []

    declarations = campaign.get("declarations", {})
    margin_candidates = set(campaign.get("margin_candidates", []))
    decl = declarations.get(exp_id, {}).get("candidates", {})

    by_cand: dict[str, list[dict[str, Any]]] = {}
    for m in fold_metrics:
        cid = m.get("candidate_id")
        if cid:
            by_cand.setdefault(cid, []).append(m)

    control_id = None
    for cid in by_cand:
        arm = decl.get(cid, {}).get("arm_id")
        if arm == "control" or cid.endswith("_control"):
            control_id = cid
            break
    if not control_id and by_cand:
        control_id = min(by_cand.keys())
    if not control_id:
        return []

    ctrl_records = sorted(
        by_cand[control_id],
        key=lambda x: (x.get("repeat", 0), x.get("fold", 0)),
    )
    challenger_ids = [c for c in by_cand if c != control_id]
    cand_metrics_summary = tv.get("candidate_metrics", {})
    ctrl_summary = cand_metrics_summary.get(control_id, {}).get(
        "mean_repeat_metrics", {}
    )

    rows: list[dict[str, Any]] = []

    for chall_id in challenger_ids:
        chall_records = sorted(
            by_cand[chall_id],
            key=lambda x: (x.get("repeat", 0), x.get("fold", 0)),
        )
        ctrl_partitions = [(r.get("repeat"), r.get("fold")) for r in ctrl_records]
        chall_partitions = [(r.get("repeat"), r.get("fold")) for r in chall_records]

        is_valid_cv, _ = _validate_repeated_cv_partitions(
            ctrl_partitions, chall_partitions
        )

        is_margin = chall_id in margin_candidates or control_id in margin_candidates
        metrics_to_evaluate = (
            ["roc_auc", "average_precision"]
            if is_margin
            else ["roc_auc", "average_precision", "log_loss", "brier_score"]
        )

        n_folds = len({r.get("fold", 0) for r in ctrl_records}) or 4
        test_ratio = 1.0 / (n_folds - 1) if n_folds > 1 else 1.0 / 3.0
        chall_summary = cand_metrics_summary.get(chall_id, {}).get(
            "mean_repeat_metrics", {}
        )

        for metric in metrics_to_evaluate:
            ctrl_vals = [
                r.get(metric) for r in ctrl_records if r.get(metric) is not None
            ]
            chall_vals = [
                r.get(metric) for r in chall_records if r.get(metric) is not None
            ]
            if len(ctrl_vals) != len(ctrl_records) or len(chall_vals) != len(
                chall_records
            ):
                continue

            is_loss = metric in ("log_loss", "brier_score")
            ctrl_eq_mean = float(ctrl_summary.get(metric, float(np.mean(ctrl_vals))))
            chall_eq_mean = float(chall_summary.get(metric, float(np.mean(chall_vals))))

            if is_valid_cv:
                paired_diffs = [
                    (ct - ch) if is_loss else (ch - ct)
                    for ch, ct in zip(chall_vals, ctrl_vals, strict=True)
                ]
                nb_res = nadeau_bengio_t_test(paired_diffs, test_ratio=test_ratio)
            else:
                mean_d = float(
                    ctrl_eq_mean - chall_eq_mean
                    if is_loss
                    else chall_eq_mean - ctrl_eq_mean
                )
                nb_res = {
                    "mean_delta": mean_d,
                    "corrected_std_error": 0.0,
                    "t_stat": 0.0,
                    "p_value": 1.0,
                    "ci_95_lower": mean_d,
                    "ci_95_upper": mean_d,
                }

            rows.append(
                {
                    "challenger_id": chall_id,
                    "control_id": control_id,
                    "metric": metric,
                    "control_equal_mean": ctrl_eq_mean,
                    "challenger_equal_mean": chall_eq_mean,
                    "paired_delta_mean": nb_res["mean_delta"],
                    "delta_std_err": nb_res["corrected_std_error"],
                    "t_stat": nb_res["t_stat"],
                    "p_value_raw": nb_res["p_value"],
                    "ci_95_lower": nb_res["ci_95_lower"],
                    "ci_95_upper": nb_res["ci_95_upper"],
                }
            )

    return rows


def _rows_for_experiment_from_oof(
    exp_id: str,
    campaign: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute comparison rows for a single experiment from OOF tables."""
    oof_tables = campaign.get("oof_tables", {})
    oof_paths = campaign.get("oof_paths", {})
    margin_candidates = set(campaign.get("margin_candidates", []))

    df = oof_tables.get(exp_id)
    if df is None:
        p = oof_paths.get(exp_id)
        if isinstance(p, (list, tuple)):
            existing = [Path(x) for x in p if Path(x).is_file()]
            if existing:
                df = pl.concat([pl.read_parquet(x) for x in existing])
            else:
                return []
        elif p and Path(p).is_file():
            df = pl.read_parquet(p)
        else:
            return []

    candidates = df["candidate_id"].unique().to_list()
    control_candidates = (
        df.filter(pl.col("arm_id") == "control")["candidate_id"].unique().to_list()
    )
    declarations = campaign.get("declarations", {})
    decl = declarations.get(exp_id, {}).get("candidates", {})
    if not control_candidates:
        for cid in candidates:
            cand_meta = decl.get(cid, {})
            if (
                cand_meta.get("arm_id") == "control"
                or cand_meta.get("role") == "control"
                or cid.endswith(("_control", "_passthrough", "_logloss", "_natural"))
            ):
                control_candidates.append(cid)
                break
    if not control_candidates and candidates:
        control_candidates = [min(candidates)]
    if not control_candidates:
        return []

    control_id = control_candidates[0]
    challenger_candidates = [
        c for c in candidates if c != control_id and c not in control_candidates
    ]
    if not challenger_candidates:
        challenger_candidates = [
            c
            for c in df.filter(pl.col("arm_id") == "challenger")["candidate_id"]
            .unique()
            .to_list()
            if c != control_id
        ]
    ctrl_full = df.filter(pl.col("candidate_id") == control_id)
    rows: list[dict[str, Any]] = []

    for chall_id in challenger_candidates:
        chall_full = df.filter(pl.col("candidate_id") == chall_id)

        join_keys = ["repeat", "fold", "row_index"]
        paired_df = ctrl_full.select(
            join_keys
            + [
                pl.col("target").alias("ctrl_target"),
                pl.col("prediction").alias("ctrl_prediction"),
                pl.col("building_id").alias("ctrl_bldg"),
                pl.col("timestamp").alias("ctrl_ts"),
            ]
        ).join(
            chall_full.select(
                join_keys
                + [
                    pl.col("target").alias("chall_target"),
                    pl.col("prediction").alias("chall_prediction"),
                    pl.col("building_id").alias("chall_bldg"),
                    pl.col("timestamp").alias("chall_ts"),
                ]
            ),
            on=join_keys,
            how="inner",
        )

        if len(paired_df) != len(ctrl_full) or len(paired_df) != len(chall_full):
            raise ValueError(
                f"One-to-one row pairing failure between {control_id} and {chall_id}: "
                f"ctrl={len(ctrl_full)}, chall={len(chall_full)}, paired={len(paired_df)}"
            )
        if not (paired_df["ctrl_target"] == paired_df["chall_target"]).all():
            raise ValueError(f"Mismatched targets between {control_id} and {chall_id}")

        ctrl_partitions = sorted(ctrl_full.select(["repeat", "fold"]).unique().rows())
        chall_partitions = sorted(chall_full.select(["repeat", "fold"]).unique().rows())
        is_valid_cv, _ = _validate_repeated_cv_partitions(
            ctrl_partitions, chall_partitions
        )

        is_margin = chall_id in margin_candidates or control_id in margin_candidates
        metrics_to_evaluate = (
            ["roc_auc", "average_precision"]
            if is_margin
            else ["roc_auc", "average_precision", "log_loss", "brier_score"]
        )

        repeats = sorted(paired_df["repeat"].unique().to_list())
        folds = sorted(paired_df["fold"].unique().to_list())
        n_folds = len(folds)
        test_ratio = 1.0 / (n_folds - 1) if n_folds > 1 else 1.0 / 3.0

        for metric in metrics_to_evaluate:
            ctrl_rep_metrics: list[float] = []
            chall_rep_metrics: list[float] = []

            for r in repeats:
                rep_slice = paired_df.filter(pl.col("repeat") == r)
                y_rep = rep_slice["ctrl_target"].to_numpy()
                p_ctrl_rep = rep_slice["ctrl_prediction"].to_numpy()
                p_chall_rep = rep_slice["chall_prediction"].to_numpy()

                if metric == "roc_auc":
                    ctrl_rep_metrics.append(float(roc_auc_score(y_rep, p_ctrl_rep)))
                    chall_rep_metrics.append(float(roc_auc_score(y_rep, p_chall_rep)))
                elif metric == "average_precision":
                    ctrl_rep_metrics.append(
                        float(average_precision_score(y_rep, p_ctrl_rep))
                    )
                    chall_rep_metrics.append(
                        float(average_precision_score(y_rep, p_chall_rep))
                    )
                elif metric == "log_loss":
                    ctrl_rep_metrics.append(
                        float(log_loss(y_rep, p_ctrl_rep, labels=[0, 1]))
                    )
                    chall_rep_metrics.append(
                        float(log_loss(y_rep, p_chall_rep, labels=[0, 1]))
                    )
                elif metric == "brier_score":
                    ctrl_rep_metrics.append(float(brier_score_loss(y_rep, p_ctrl_rep)))
                    chall_rep_metrics.append(
                        float(brier_score_loss(y_rep, p_chall_rep))
                    )

            control_equal_mean = (
                float(np.mean(ctrl_rep_metrics)) if ctrl_rep_metrics else 0.0
            )
            challenger_equal_mean = (
                float(np.mean(chall_rep_metrics)) if chall_rep_metrics else 0.0
            )

            is_loss = METRIC_CONTRACT["directions"].get(metric) == "minimize"

            if is_valid_cv:
                paired_diffs: list[float] = []
                for r in repeats:
                    for f in folds:
                        fold_slice = paired_df.filter(
                            (pl.col("repeat") == r) & (pl.col("fold") == f)
                        )
                        if len(fold_slice) == 0:
                            continue
                        y_f = fold_slice["ctrl_target"].to_numpy()
                        p_ctrl_f = fold_slice["ctrl_prediction"].to_numpy()
                        p_chall_f = fold_slice["chall_prediction"].to_numpy()

                        if metric == "roc_auc":
                            m_ctrl = float(roc_auc_score(y_f, p_ctrl_f))
                            m_chall = float(roc_auc_score(y_f, p_chall_f))
                        elif metric == "average_precision":
                            m_ctrl = float(average_precision_score(y_f, p_ctrl_f))
                            m_chall = float(average_precision_score(y_f, p_chall_f))
                        elif metric == "log_loss":
                            m_ctrl = float(log_loss(y_f, p_ctrl_f, labels=[0, 1]))
                            m_chall = float(log_loss(y_f, p_chall_f, labels=[0, 1]))
                        elif metric == "brier_score":
                            m_ctrl = float(brier_score_loss(y_f, p_ctrl_f))
                            m_chall = float(brier_score_loss(y_f, p_chall_f))
                        else:
                            m_ctrl = 0.0
                            m_chall = 0.0

                        diff = (m_ctrl - m_chall) if is_loss else (m_chall - m_ctrl)
                        paired_diffs.append(diff)

                nb_res = nadeau_bengio_t_test(paired_diffs, test_ratio=test_ratio)
            else:
                mean_d = float(
                    control_equal_mean - challenger_equal_mean
                    if is_loss
                    else challenger_equal_mean - control_equal_mean
                )
                nb_res = {
                    "mean_delta": mean_d,
                    "corrected_std_error": 0.0,
                    "t_stat": 0.0,
                    "p_value": 1.0,
                    "ci_95_lower": mean_d,
                    "ci_95_upper": mean_d,
                }

            rows.append(
                {
                    "challenger_id": chall_id,
                    "control_id": control_id,
                    "metric": metric,
                    "control_equal_mean": control_equal_mean,
                    "challenger_equal_mean": challenger_equal_mean,
                    "paired_delta_mean": nb_res["mean_delta"],
                    "delta_std_err": nb_res["corrected_std_error"],
                    "t_stat": nb_res["t_stat"],
                    "p_value_raw": nb_res["p_value"],
                    "ci_95_lower": nb_res["ci_95_lower"],
                    "ci_95_upper": nb_res["ci_95_upper"],
                }
            )

    return rows


def _rows_from_transport_validations(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract comparison rows from all transport validations having fold_metrics."""
    rows: list[dict[str, Any]] = []
    for exp_id in campaign.get("transport_validations", {}):
        rows.extend(_rows_for_experiment_from_fold_metrics(exp_id, campaign))
    return rows


def _rows_from_oof_tables(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute comparison rows from all OOF tables."""
    rows: list[dict[str, Any]] = []
    exp_ids = set(campaign.get("oof_tables", {}).keys()) | set(
        campaign.get("oof_paths", {}).keys()
    )
    for exp_id in sorted(exp_ids):
        rows.extend(_rows_for_experiment_from_oof(exp_id, campaign))
    return rows


def build_comparison_tables(campaign: dict[str, Any]) -> pl.DataFrame:
    """Generate statistical comparison tables across all experiments in a campaign.

    Computes equal-mean repeat-level pooled metrics for control and challenger arms,
    paired fold differences on strictly joined rows, Nadeau-Bengio corrected resampled t-test
    statistics, and Holm-Bonferroni adjusted p-values and verdicts.

    For each experiment, uses transport validation fold_metrics if available, falling
    back to out-of-fold predictions. Repeated-CV partitions are strictly validated
    (distinct, paired 1-to-1, at least 2 repeats) before computing t-test statistics to
    prevent spurious significance. Multiple-testing correction is applied across the
    entire merged family.

    Decision margin models report ROC-AUC and Average Precision, but omit probability loss
    metrics (LogLoss and Brier score).

    Args:
        campaign: Finalized campaign dictionary returned by load_finalized_campaign.

    Returns:
        Polars DataFrame containing structured statistical comparison rows.
    """
    exp_ids: list[str] = []
    for exp_entry in campaign.get("experiments", []):
        eid = (
            exp_entry.get("experiment_id")
            if isinstance(exp_entry, dict)
            else str(exp_entry)
        )
        if eid and eid not in exp_ids:
            exp_ids.append(eid)
    for source_dict in (
        campaign.get("declarations", {}),
        campaign.get("transport_validations", {}),
        campaign.get("oof_tables", {}),
        campaign.get("oof_paths", {}),
    ):
        for eid in source_dict:
            if eid and eid not in exp_ids:
                exp_ids.append(eid)

    all_rows: list[dict[str, Any]] = []
    transport_validations = campaign.get("transport_validations", {})

    for exp_id in exp_ids:
        tv = transport_validations.get(exp_id, {})
        fold_metrics = tv.get("fold_metrics", [])
        if fold_metrics:
            exp_rows = _rows_for_experiment_from_fold_metrics(exp_id, campaign)
        else:
            exp_rows = _rows_for_experiment_from_oof(exp_id, campaign)
        all_rows.extend(exp_rows)

    if not all_rows:
        return pl.DataFrame(
            schema={
                "challenger_id": pl.Utf8,
                "control_id": pl.Utf8,
                "metric": pl.Utf8,
                "control_equal_mean": pl.Float64,
                "challenger_equal_mean": pl.Float64,
                "paired_delta_mean": pl.Float64,
                "delta_std_err": pl.Float64,
                "t_stat": pl.Float64,
                "p_value_raw": pl.Float64,
                "p_value_corrected": pl.Float64,
                "ci_95_lower": pl.Float64,
                "ci_95_upper": pl.Float64,
                "verdict": pl.Utf8,
            }
        )

    # Compute multiple-testing corrections across comparisons
    raw_p_values = [r["p_value_raw"] for r in all_rows]
    hb_res = holm_bonferroni_correction(raw_p_values, alpha=0.05)
    adjusted_p_values = hb_res["adjusted_p_values"]

    min_roc_auc_delta = PROMOTION_THRESHOLDS["min_roc_auc_delta"]
    for row, p_corr in zip(all_rows, adjusted_p_values, strict=True):
        row["p_value_corrected"] = p_corr
        min_lift = min_roc_auc_delta if row["metric"] == "roc_auc" else 0.0
        row["verdict"] = compute_signal_vs_noise_verdict(
            p_corr, row["paired_delta_mean"], min_lift=min_lift
        )

    return pl.DataFrame(all_rows)


def iter_paired_error_slices(
    campaign: dict[str, Any],
    left_candidate: str,
    right_candidate: str,
    metadata_path: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Iterate through paired prediction error slices across key operational dimensions.

    Slices errors along:
    1. building_cohort: Primary use classification or building cohort.
    2. calendar_hour: Hourly diurnal cycle.
    3. missingness: Weather and telemetry missingness flags.

    Provides false positive/false negative breakdowns, delta error rates, and McNemar counts.

    Args:
        campaign: Finalized campaign dictionary.
        left_candidate: Candidate ID for left model.
        right_candidate: Candidate ID for right model.
        metadata_path: Optional path to feature_metadata.parquet.

    Yields:
        Dictionary for each slice with sample count, anomaly count, FP/FN breakdown,
        delta error rate, and McNemar better counts.
    """
    oof_tables = campaign.get("oof_tables", {})
    matching_df: pl.DataFrame | None = None
    for df in oof_tables.values():
        if df is None:
            continue
        cands = set(df["candidate_id"].unique().to_list())
        if left_candidate in cands and right_candidate in cands:
            matching_df = df
            break

    if matching_df is None:
        # Load on-demand from oof_paths
        oof_paths = campaign.get("oof_paths", {})
        declarations = campaign.get("declarations", {})
        for exp_id, decl in declarations.items():
            cands = set(decl.get("candidates", {}).keys())
            if left_candidate in cands and right_candidate in cands:
                p = oof_paths.get(exp_id)
                paths = list(p) if isinstance(p, (list, tuple)) else ([p] if p else [])
                existing = [Path(x) for x in paths if Path(x).is_file()]
                if existing:
                    matching_df = (
                        pl.read_parquet(existing[0])
                        if len(existing) == 1
                        else pl.concat([pl.read_parquet(x) for x in existing])
                    )
                    break

    if matching_df is None:
        return
    left_df = matching_df.filter(pl.col("candidate_id") == left_candidate)
    right_df = matching_df.filter(pl.col("candidate_id") == right_candidate)

    left_sub = left_df.select(
        [
            pl.col("repeat"),
            pl.col("fold"),
            pl.col("row_index"),
            pl.col("building_id"),
            pl.col("timestamp"),
            pl.col("target"),
            pl.col("prediction").alias("left_prediction"),
            pl.col("score_kind").alias("left_score_kind"),
        ]
    )
    right_sub = right_df.select(
        [
            pl.col("repeat"),
            pl.col("fold"),
            pl.col("row_index"),
            pl.col("prediction").alias("right_prediction"),
            pl.col("score_kind").alias("right_score_kind"),
        ]
    )

    paired = left_sub.join(right_sub, on=["repeat", "fold", "row_index"])
    paired = paired.with_columns(
        pl.col("timestamp").dt.hour().cast(pl.Int64).alias("hour")
    )

    # Resolve metadata table
    if metadata_path is None:
        metadata_path = campaign.get("feature_metadata_path")
        campaign_dir = campaign.get("campaign_dir")
        if metadata_path is None and campaign_dir is not None:
            cand_meta = Path(campaign_dir) / "feature_metadata.parquet"
            if cand_meta.is_file():
                metadata_path = cand_meta

    if metadata_path is not None and Path(metadata_path).is_file():
        meta_df = pl.read_parquet(metadata_path)
        if "building_id" in meta_df.columns and "hour" in meta_df.columns:
            paired = paired.join(meta_df, on=["building_id", "hour"], how="left")
        elif "building_id" in meta_df.columns:
            paired = paired.join(meta_df, on=["building_id"], how="left")

    left_score_kind = (
        paired["left_score_kind"][0] if len(paired) > 0 else "positive_probability"
    )
    right_score_kind = (
        paired["right_score_kind"][0] if len(paired) > 0 else "positive_probability"
    )

    left_thresh = 0.5 if left_score_kind == "positive_probability" else 0.0
    right_thresh = 0.5 if right_score_kind == "positive_probability" else 0.0

    left_bin = (pl.col("left_prediction") >= left_thresh).cast(pl.Int64)
    right_bin = (pl.col("right_prediction") >= right_thresh).cast(pl.Int64)
    y = pl.col("target").cast(pl.Int64)

    left_error = (left_bin != y).cast(pl.Int64)
    right_error = (right_bin != y).cast(pl.Int64)
    left_fp = ((left_bin == 1) & (y == 0)).cast(pl.Int64)
    left_fn = ((left_bin == 0) & (y == 1)).cast(pl.Int64)
    right_fp = ((right_bin == 1) & (y == 0)).cast(pl.Int64)
    right_fn = ((right_bin == 0) & (y == 1)).cast(pl.Int64)

    mcnemar_left_better = ((left_error == 0) & (right_error == 1)).cast(pl.Int64)
    mcnemar_right_better = ((left_error == 1) & (right_error == 0)).cast(pl.Int64)

    paired = paired.with_columns(
        [
            left_bin.alias("left_bin"),
            right_bin.alias("right_bin"),
            left_error.alias("left_error"),
            right_error.alias("right_error"),
            left_fp.alias("left_fp"),
            left_fn.alias("left_fn"),
            right_fp.alias("right_fp"),
            right_fn.alias("right_fn"),
            mcnemar_left_better.alias("mcnemar_left_better"),
            mcnemar_right_better.alias("mcnemar_right_better"),
        ]
    )

    def _calc_slice(
        sub_df: pl.DataFrame, slice_type: str, slice_key: Any
    ) -> dict[str, Any] | None:
        n_samples = len(sub_df)
        if n_samples == 0:
            return None
        n_anomalies = int(sub_df["target"].sum())
        l_fp = int(sub_df["left_fp"].sum())
        l_fn = int(sub_df["left_fn"].sum())
        r_fp = int(sub_df["right_fp"].sum())
        r_fn = int(sub_df["right_fn"].sum())
        left_err_rate = (l_fp + l_fn) / n_samples
        right_err_rate = (r_fp + r_fn) / n_samples
        delta_err = left_err_rate - right_err_rate
        mcnemar_l = int(sub_df["mcnemar_left_better"].sum())
        mcnemar_r = int(sub_df["mcnemar_right_better"].sum())
        return {
            "slice_type": slice_type,
            "slice_key": slice_key,
            "n_samples": n_samples,
            "n_anomalies": n_anomalies,
            "left_fp": l_fp,
            "left_fn": l_fn,
            "right_fp": r_fp,
            "right_fn": r_fn,
            "delta_error_rate": delta_err,
            "mcnemar_left_better_count": mcnemar_l,
            "mcnemar_right_better_count": mcnemar_r,
        }

    # 1. Building cohort slices
    if "primary_use" in paired.columns:
        cohorts = sorted(paired["primary_use"].unique().to_list())
        for cohort in cohorts:
            if cohort is not None:
                s = _calc_slice(
                    paired.filter(pl.col("primary_use") == cohort),
                    "building_cohort",
                    str(cohort),
                )
                if s is not None:
                    yield s
    else:
        bldgs = sorted(paired["building_id"].unique().to_list())
        for bldg in bldgs:
            s = _calc_slice(
                paired.filter(pl.col("building_id") == bldg),
                "building_cohort",
                str(bldg),
            )
            if s is not None:
                yield s

    # 2. Calendar hour slices
    hours = sorted(paired["hour"].unique().to_list())
    for hour in hours:
        s = _calc_slice(
            paired.filter(pl.col("hour") == hour), "calendar_hour", int(hour)
        )
        if s is not None:
            yield s

    # 3. Missingness slices
    if "weather_missing" in paired.columns:
        missing_vals = sorted(paired["weather_missing"].unique().to_list())
        for missing_val in missing_vals:
            if missing_val is not None:
                s = _calc_slice(
                    paired.filter(pl.col("weather_missing") == missing_val),
                    "missingness",
                    f"weather_missing={missing_val}",
                )
                if s is not None:
                    yield s
    else:
        missing_cols = [c for c in paired.columns if c.endswith("_missing")]
        for mc in missing_cols:
            for val in [True, False]:
                s = _calc_slice(
                    paired.filter(pl.col(mc) == val), "missingness", f"{mc}={val}"
                )
                if s is not None:
                    yield s


def build_ensemble_plan(
    campaign: dict[str, Any],
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Emit strict 8-key architectural ensemble plan for downstream integration.

    Segregates calibrated probability models into provisional Level 1 ensemble candidates
    while explicitly rejecting decision margin or uncalibrated candidates with reasons.

    Args:
        campaign: Finalized campaign dictionary.
        output_path: Optional output path to write ensemble_plan.json.

    Returns:
        Dictionary containing exactly the 8 required ensemble plan keys.
    """
    prob_candidates = set(campaign.get("probability_candidates", []))
    margin_candidates = set(campaign.get("margin_candidates", []))

    provisional_l1: list[str] = []
    rejections: dict[str, str] = {}

    # 1. Reject margin candidates with explicit explanation
    for cand_id in sorted(margin_candidates):
        rejections[cand_id] = (
            "Excluded from probability ensemble pool: score_kind is decision_margin "
            "(uncalibrated decision margins require probability calibration)."
        )

    # 2. Build mapping from candidate_id to experiment_id
    cand_to_exp: dict[str, str] = {}
    declarations = campaign.get("declarations", {})
    for exp_id, decl in declarations.items():
        for cid in decl.get("candidates", {}):
            cand_to_exp[cid] = exp_id
    for exp_id, df in campaign.get("oof_tables", {}).items():
        if df is not None and "candidate_id" in df.columns:
            for cid in df["candidate_id"].unique().to_list():
                if cid not in cand_to_exp:
                    cand_to_exp[cid] = exp_id

    # 3. Gather experiment-level metadata lookups
    manifest = campaign.get("manifest", {})
    manifest_cands = manifest.get("candidates", [])
    manifest_by_exp: dict[str, dict[str, Any]] = {}
    for entry in manifest_cands:
        eid = entry.get("experiment_id")
        if eid:
            manifest_by_exp[eid] = entry

    experiments_list = campaign.get("experiments", [])
    exp_by_id: dict[str, dict[str, Any]] = {}
    for entry in experiments_list:
        if isinstance(entry, dict):
            eid = entry.get("experiment_id")
            if eid:
                exp_by_id[eid] = entry

    run_status = campaign.get("run_status", {})
    campaign_status = campaign.get("status")

    # 4. Filter probability candidates for certified completion and eligibility
    for cand_id in sorted(prob_candidates):
        exp_id = cand_to_exp.get(cand_id)
        if not exp_id:
            known_ids = (
                list(manifest_by_exp.keys())
                + list(exp_by_id.keys())
                + list(declarations.keys())
            )
            for known_exp in known_ids:
                if cand_id.startswith(known_exp):
                    exp_id = known_exp
                    break

        exp_manifest = manifest_by_exp.get(exp_id) if exp_id else None
        exp_info = exp_by_id.get(exp_id) if exp_id else None
        exp_decl = declarations.get(exp_id, {})
        is_independent_model = (
            (
                exp_manifest is not None
                and exp_manifest.get("provenance") == "discovered_model"
            )
            or (
                exp_info is not None
                and exp_info.get("provenance") == "discovered_model"
            )
            or exp_decl.get("provenance") == "discovered_model"
        )

        # Criterion 1: Unsealed status
        if is_independent_model:
            is_unsealed = (
                exp_manifest is not None and exp_manifest.get("status") != "completed"
            ) or (
                exp_info is not None
                and exp_info.get("status") not in (None, "completed")
            )
            comparative_unavailable = (
                exp_manifest is not None
                and exp_manifest.get("comparative_completion_available") is False
            ) or (
                exp_info is not None
                and exp_info.get("comparative_completion_available") is False
            )
            n03_ineligible = (
                exp_manifest is not None and exp_manifest.get("n03_eligible") is False
            ) or (exp_info is not None and exp_info.get("n03_eligible") is False)
        else:
            is_unsealed = (
                campaign_status != "completed"
                or (
                    exp_manifest is not None
                    and exp_manifest.get("status") != "completed"
                )
                or (
                    exp_info is not None
                    and exp_info.get("status") not in (None, "completed")
                )
            )
            comparative_unavailable = (
                (
                    exp_manifest is not None
                    and exp_manifest.get("comparative_completion_available") is False
                )
                or (
                    exp_info is not None
                    and exp_info.get("comparative_completion_available") is False
                )
                or manifest.get("comparative_completion_available") is False
                or run_status.get("comparative_completion_available") is False
                or campaign.get("comparative_completion_available") is False
            )
            n03_ineligible = (
                (exp_manifest is not None and exp_manifest.get("n03_eligible") is False)
                or (exp_info is not None and exp_info.get("n03_eligible") is False)
                or manifest.get("n03_eligible") is False
                or run_status.get("n03_eligible") is False
            )
            oof_df = campaign.get("oof_tables", {}).get(exp_id)
            if oof_df is not None and "n03_eligible" in oof_df.columns:
                cand_slice = oof_df.filter(pl.col("candidate_id") == cand_id)
                if not cand_slice.is_empty():
                    col_n03 = cand_slice["n03_eligible"]
                    if col_n03.null_count() > 0 or not col_n03.all(
                        ignore_nulls=False
                    ):
                        n03_ineligible = True
                    if "non_comparative" in cand_slice.columns:
                        col_nc = cand_slice["non_comparative"]
                        if col_nc.null_count() > 0 or not (col_nc == False).all(
                            ignore_nulls=False
                        ):
                            n03_ineligible = True
        if is_unsealed or comparative_unavailable or n03_ineligible:
            rejections[cand_id] = (
                "Unfinalized run / comparative completion not certified; retained as diagnostic only"
            )
        else:
            provisional_l1.append(cand_id)

    plan = {
        "provisional_level1_candidates": provisional_l1,
        "provisional_rejections_with_reasons": rejections,
        "level1_eligibility_policy": {
            "required_score_kind": "positive_probability",
            "require_completed_status": True,
            "require_verified_oof_lineage": True,
            "require_zero_leakage": True,
        },
        "simple_blend_candidates": list(provisional_l1),
        "level2_meta_learners": [
            {"name": "logistic_regression", "regularization": "l2", "solver": "lbfgs"},
            {"name": "ridge_classifier", "alpha": 1.0},
            {"name": "lightgbm_shallow", "max_depth": 3, "n_estimators": 50},
        ],
        "level3_attempt_policy": {
            "allow_level3_stacking": False,
            "reason": (
                "Level 3 stacking is restricted to prevent complexity explosion unless Level 2 "
                "demonstrates statistically significant gain over Level 1 blend."
            ),
        },
        "nested_cv_contract": {
            "outer_repeats": 3,
            "outer_folds": 4,
            "group_key": "building_id",
            "leakage_prevention": (
                "Strict GroupKFold partitioning by building_id ensures out-of-fold predictions "
                "never leak training representations across folds or repeats."
            ),
        },
        "promotion_thresholds": {
            "min_roc_auc_delta": PROMOTION_THRESHOLDS["min_roc_auc_delta"],
            "significance_alpha": PROMOTION_THRESHOLDS["significance_alpha"],
            "require_non_negative_pr_auc_delta": PROMOTION_THRESHOLDS[
                "require_non_negative_pr_auc_delta"
            ],
        },
    }

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")

    return plan
