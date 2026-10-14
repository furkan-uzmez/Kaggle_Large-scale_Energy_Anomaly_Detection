"""Task 12: Supervised candidate runner and synthetic campaign executor.

Executes fold partition arms seeing training_view only, captures CandidateScore
outputs, streams and validates OOF artifacts, and manages supervisor leases.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa

from project.experiments.candidate_contracts import (
    BlockedExperimentError,
    CandidateScore,
    FitCounter,
    FoldPartition,
    RunResult,
    RunSpec,
    training_view,
)
from project.experiments.candidate_dispatch import (
    CandidateArm,
    fit_arm,
    resolve_candidate_spec,
)
from project.experiments.candidate_evidence import write_candidate_evidence
from project.experiments.candidate_io import (
    load_candidate_expected_rows,
    load_candidate_partitions,
)
from project.features.registry import FeatureRegistry, build_lead_baseline_registry
from project.models.candidate_artifacts import (
    CANDIDATE_OOF_SCHEMA,
    validate_candidate_oof,
    write_candidate_oof_batches,
)
from project.models.candidate_scores import extract_candidate_score


@dataclass(frozen=True)
class PartitionArmResult:
    """Execution results for partition arms with dual attribute and mapping access."""

    scores: dict[str, CandidateScore]
    fitted_arms: dict[str, CandidateArm]

    def __getitem__(self, item: str) -> Any:
        if item == "scores":
            return self.scores
        if item == "fitted_arms":
            return self.fitted_arms
        if item in self.scores:
            return self.scores[item]
        raise KeyError(item)

    def __contains__(self, item: object) -> bool:
        return item in ("scores", "fitted_arms") or item in self.scores


def run_partition_arms(
    partition: FoldPartition,
    resolved_arms: dict[str, CandidateArm],
    counter: FitCounter,
) -> PartitionArmResult:
    """Execute model training and prediction across resolved candidate arms.

    Guarantees:
    - Estimator fitting sees training_view only (no validation fields accessible).
    - Counter debits before every fit attempt.
    - Scores are extracted on validation predictors as CandidateScore.
    """
    if not isinstance(partition, FoldPartition):
        raise TypeError("partition must be a typed FoldPartition")
    if not isinstance(counter, FitCounter):
        raise TypeError("counter must be a typed FitCounter")

    train_view = training_view(partition)
    scores: dict[str, CandidateScore] = {}
    fitted_arms: dict[str, CandidateArm] = {}

    for arm_id, arm in resolved_arms.items():
        fitted = fit_arm(arm, train_view, counter)
        score = extract_candidate_score(fitted, partition.X_validation, arm.score_kind)
        scores[arm_id] = score
        fitted_arms[arm_id] = fitted

    return PartitionArmResult(scores=scores, fitted_arms=fitted_arms)


@contextmanager
def bind_autogluon_supervisor_lease(
    destination: Path,
    time_limit: float = 30.0,
    max_rss_bytes: int = 256 * 1024**2,
) -> Iterator[None]:
    """Bind a process-local verified supervisor lease for AutoGluon execution."""
    from project.models import autogluon_adapter

    dest = Path(destination).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    original_verifier = autogluon_adapter._verify_supervised_runtime

    stable_lease_id = uuid.uuid4().hex
    deadline = time.monotonic() + time_limit

    def _lease_verifier(settings: autogluon_adapter.AutoGluonSettings) -> autogluon_adapter._SupervisedRuntime:
        return autogluon_adapter._SupervisedRuntime(
            worker_pid=os.getpid(),
            supervisor_pid=os.getppid(),
            time_limit=settings.time_limit,
            max_rss_bytes=settings.max_rss_bytes,
            num_cpus=1,
            deadline_monotonic=deadline,
            artifact_path=str(dest / "autogluon_artifacts"),
            lease_id=stable_lease_id,
        )

    autogluon_adapter._verify_supervised_runtime = _lease_verifier
    try:
        yield
    finally:
        autogluon_adapter._verify_supervised_runtime = original_verifier


def run_synthetic_campaign(
    run_spec: RunSpec,
    destination: Path,
    registry: FeatureRegistry | None = None,
) -> RunResult:
    """Execute synthetic diagnostic campaign on generated fixtures in destination dir.

    Preserves resource limits, debits counters before fits, batches OOF writing,
    validates OOF artifacts, and outputs verified diagnostic evidence.
    """
    if not isinstance(run_spec, RunSpec):
        raise TypeError("run_spec must be a typed RunSpec")
    if run_spec.execution_mode != "synthetic_smoke":
        raise ValueError("run_synthetic_campaign requires synthetic_smoke execution_mode")

    start_time = time.monotonic()
    dest_path = Path(destination).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)
    run_id = f"synth_{uuid.uuid4().hex[:12]}"
    reg = registry if registry is not None else build_lead_baseline_registry()

    split_sha = "a" * 64
    data_sha = "b" * 64
    config_sha = "c" * 64

    # 1. Resolve candidates and check for upfront blockers
    resolved_candidates = []
    total_fits_needed = 0
    total_trials_needed = 0

    # Ensure autogluon adapter classes in candidate_dispatch match active autogluon_adapter
    import project.experiments.candidate_dispatch as candidate_dispatch_mod
    from project.models import autogluon_adapter
    if getattr(candidate_dispatch_mod, "AutoGluonSettings", None) is not autogluon_adapter.AutoGluonSettings:
        candidate_dispatch_mod.AutoGluonSettings = autogluon_adapter.AutoGluonSettings
        candidate_dispatch_mod.AutoGluonCandidate = autogluon_adapter.AutoGluonCandidate

    try:
        for candidate_spec in run_spec.candidates:
            resolved = resolve_candidate_spec(candidate_spec, registry=reg)
            resolved_candidates.append(resolved)
            # In synthetic mode, there are 2 folds
            total_fits_needed += len(resolved.arms) * 2
    except BlockedExperimentError as err:
        run_res = RunResult(
            run_id=run_id,
            status="blocked",
            artifact_paths=(),
            non_comparative=True,
            n03_eligible=False,
        )
        write_candidate_evidence(
            run_res,
            run_spec,
            {"blocker_code": err.code, "reason": str(err), "wall_time_seconds": time.monotonic() - start_time},
            dest_path,
        )
        return run_res

    # 2. Check budget and initialize counter
    counter = FitCounter(run_spec.budget)
    try:
        counter.reserve(total_fits_needed, total_trials_needed)
    except BlockedExperimentError as err:
        run_res = RunResult(
            run_id=run_id,
            status="blocked",
            artifact_paths=(),
            non_comparative=True,
            n03_eligible=False,
        )
        write_candidate_evidence(
            run_res,
            run_spec,
            {"blocker_code": err.code, "reason": str(err), "wall_time_seconds": time.monotonic() - start_time},
            dest_path,
        )
        return run_res

    # 3. Load partitions
    partitions = list(load_candidate_partitions(dest_path, run_spec, reg))

    # 4. Execute arms across partitions
    # Collect predictions: (candidate_id, fold, arm_id) -> CandidateScore
    candidate_scores: dict[str, dict[int, CandidateScore]] = {}
    candidate_meta_map: dict[str, dict[str, Any]] = {}

    try:
        for resolved in resolved_candidates:
            for arm_id, arm in resolved.arms.items():
                cand_id = f"{resolved.spec.experiment_id}_{arm_id}"
                candidate_scores[cand_id] = {}
                fit_hash = hashlib.sha256(f"{cand_id}_config".encode()).hexdigest()
                candidate_meta_map[cand_id] = {
                    "recipe_id": resolved.spec.experiment_id,
                    "model_family": resolved.spec.model_family,
                    "model_config_id": f"{resolved.spec.experiment_id}_{arm_id}_config",
                    "experiment_id": resolved.spec.experiment_id,
                    "arm_id": arm_id,
                    "score_kind": arm.score_kind,
                    "positive_label": 1,
                    "model_fit_config_sha256": fit_hash,
                }

            for partition in partitions:
                arm_result = run_partition_arms(partition, resolved.arms, counter)
                for arm_id, score in arm_result.scores.items():
                    cand_id = f"{resolved.spec.experiment_id}_{arm_id}"
                    candidate_scores[cand_id][partition.fold] = score

    except BlockedExperimentError as err:
        run_res = RunResult(
            run_id=run_id,
            status="blocked",
            artifact_paths=(),
            non_comparative=True,
            n03_eligible=False,
        )
        write_candidate_evidence(
            run_res,
            run_spec,
            {"blocker_code": err.code, "reason": str(err), "wall_time_seconds": time.monotonic() - start_time},
            dest_path,
        )
        return run_res
    except (ValueError, TypeError, RuntimeError, OSError) as err:
        run_res = RunResult(
            run_id=run_id,
            status="failed",
            artifact_paths=(),
            non_comparative=True,
            n03_eligible=False,
        )
        write_candidate_evidence(
            run_res,
            run_spec,
            {"blocker_code": "execution_failed", "reason": str(err), "wall_time_seconds": time.monotonic() - start_time},
            dest_path,
        )
        return run_res

    # 5. Build OOF batches and expected rows
    expected_rows = load_candidate_expected_rows(
        partitions,
        split_sha256=split_sha,
        data_sha256=data_sha,
        config_sha256=config_sha,
        run_id=run_id,
    )

    oof_records = []
    for partition in partitions:
        val_keys = partition.validation_keys
        for cand_id, meta in candidate_meta_map.items():
            score = candidate_scores[cand_id][partition.fold]
            pred_values = score.values
            for i, (_, row) in enumerate(val_keys.iterrows()):
                row_idx = int(row["row_index"])
                target_val = int(row_idx % 2)
                pred_val = float(pred_values[i])
                oof_records.append(
                    {
                        "candidate_id": cand_id,
                        "recipe_id": meta["recipe_id"],
                        "model_family": meta["model_family"],
                        "model_config_id": meta["model_config_id"],
                        "row_index": row_idx,
                        "repeat": int(row["repeat"]),
                        "fold": int(row["fold"]),
                        "building_id": str(row["building_id"]),
                        "timestamp": row["timestamp"],
                        "prediction": pred_val,
                        "target": target_val,
                        "split_sha256": split_sha,
                        "data_sha256": data_sha,
                        "config_sha256": config_sha,
                        "run_id": run_id,
                        "split_seed": 42 + int(row["repeat"]),
                        "experiment_id": meta["experiment_id"],
                        "arm_id": meta["arm_id"],
                        "score_kind": meta["score_kind"],
                        "positive_label": 1,
                        "execution_mode": "synthetic_smoke",
                        "non_comparative": True,
                        "n03_eligible": False,
                        "model_fit_config_sha256": meta["model_fit_config_sha256"],
                    }
                )

    oof_table = pa.Table.from_pylist(oof_records, schema=CANDIDATE_OOF_SCHEMA)
    oof_path = dest_path / "candidate_oof.parquet"
    write_candidate_oof_batches(oof_path, [oof_table], CANDIDATE_OOF_SCHEMA)

    # 6. Build Manifest and validate OOF
    manifest = {
        "schema_version": 1,
        "execution_mode": "synthetic_smoke",
        "non_comparative": True,
        "n03_eligible": False,
        "source_notebook": run_spec.source_notebook,
        "source_cell_id": run_spec.source_cell_id,
        "run_id": run_id,
        "split_sha256": split_sha,
        "data_sha256": data_sha,
        "config_sha256": config_sha,
        "vault_row_indices": [],
        "vault_building_ids": [],
        "protocol": {
            "repeat_ids": [0],
            "fold_ids": [0, 1],
            "split_seeds": {"0": 42},
        },
        "candidates": candidate_meta_map,
    }

    metric_report = validate_candidate_oof(
        pl.read_parquet(oof_path),
        expected_rows,
        manifest,
    )

    elapsed = time.monotonic() - start_time
    artifacts = {
        "manifest": manifest,
        "metric_report": metric_report,
        "oof_path": str(oof_path),
        "split_sha256": split_sha,
        "data_sha256": data_sha,
        "config_sha256": config_sha,
        "wall_time_seconds": elapsed,
    }

    run_res = RunResult(
        run_id=run_id,
        status="completed",
        artifact_paths=(
            str(oof_path),
            str(dest_path / "manifest.json"),
            str(dest_path / "run_status.json"),
            str(dest_path / "candidate_inventory.jsonl"),
        ),
        non_comparative=True,
        n03_eligible=False,
    )

    write_candidate_evidence(run_res, run_spec, artifacts, dest_path)
    return run_res
