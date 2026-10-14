"""Unit tests for candidate evidence and inventory linkage (Task 12)."""

import json
from pathlib import Path

from candidate_fixtures import CORE_COLUMNS
from project.experiments.candidate_contracts import (
    CandidateSpec,
    FitBudget,
    RunResult,
    RunSpec,
)
from project.experiments.inventory import REQUIRED_INVENTORY_FIELDS


def make_run_spec(recipe: str = "feat_meter_deviation_ratio") -> RunSpec:
    budget = FitBudget(4, 0, 30.0, 256 * 1024**2, 1)
    return RunSpec(
        1,
        (CandidateSpec(recipe, CORE_COLUMNS, "hist_gradient_boosting", (), ()),),
        budget,
        "synthetic_smoke",
        "notebooks/02_baseline_and_candidate_design.ipynb",
        "cell-evidence",
    )


def test_write_candidate_evidence_completed(tmp_path: Path):
    from project.experiments.candidate_evidence import write_candidate_evidence

    spec = make_run_spec()
    run_result = RunResult(
        run_id="run-001",
        status="completed",
        artifact_paths=(str(tmp_path / "candidate_oof.parquet"),),
        non_comparative=True,
        n03_eligible=False,
    )
    artifacts = {
        "manifest": {
            "schema_version": 1,
            "run_id": "run-001",
            "execution_mode": "synthetic_smoke",
            "candidates": {"feat_meter_deviation_ratio_challenger": {"score_kind": "positive_probability"}},
        },
        "metric_report": {
            "candidate_metrics": {
                "feat_meter_deviation_ratio_challenger": {
                    "mean_repeat_metrics": {"roc_auc": 0.85, "brier_score": 0.12},
                }
            }
        },
    }

    evidence = write_candidate_evidence(run_result, spec, artifacts, tmp_path)
    assert evidence["manifest_path"].endswith("manifest.json")
    assert evidence["status_path"].endswith("run_status.json")

    # Check manifest.json
    manifest_file = tmp_path / "manifest.json"
    assert manifest_file.exists()
    manifest_data = json.loads(manifest_file.read_text())
    assert manifest_data["run_id"] == "run-001"

    # Check run_status.json
    status_file = tmp_path / "run_status.json"
    assert status_file.exists()
    status_data = json.loads(status_file.read_text())
    assert status_data["status"] == "completed"
    assert status_data["non_comparative"] is True
    assert status_data["n03_eligible"] is False

    # Check candidate_inventory.jsonl
    inv_file = tmp_path / "candidate_inventory.jsonl"
    assert inv_file.exists()
    lines = inv_file.read_text().strip().split("\n")
    assert len(lines) >= 1
    record = json.loads(lines[0])
    for f in REQUIRED_INVENTORY_FIELDS:
        assert f in record
    assert record["outcome"] == "completed"
    assert record["rejection_reason"] is None


def test_write_candidate_evidence_blocked(tmp_path: Path):
    from project.experiments.candidate_evidence import write_candidate_evidence

    spec = make_run_spec(recipe="regime_same_family_hpo_vs_default")
    run_result = RunResult(
        run_id="run-blocked",
        status="blocked",
        artifact_paths=(),
        non_comparative=True,
        n03_eligible=False,
    )
    artifacts = {
        "blocker_code": "hpo_not_ready",
        "reason": "Task 9 accounting/resume integration requires parent approval",
    }

    _ = write_candidate_evidence(run_result, spec, artifacts, tmp_path)
    status_file = tmp_path / "run_status.json"
    status_data = json.loads(status_file.read_text())
    assert status_data["status"] == "blocked"
    assert status_data["blocker_code"] == "hpo_not_ready"
    assert "Task 9" in status_data["reason"]

    inv_file = tmp_path / "candidate_inventory.jsonl"
    record = json.loads(inv_file.read_text().strip().split("\n")[0])
    assert record["outcome"] == "blocked"
    assert record["rejection_reason"] == "hpo_not_ready: Task 9 accounting/resume integration requires parent approval"


def test_synthetic_events_never_emit_n03_eligible(tmp_path: Path):
    from project.experiments.candidate_evidence import write_candidate_evidence

    spec = make_run_spec()
    run_result = RunResult(
        run_id="run-synth",
        status="completed",
        artifact_paths=(str(tmp_path / "oof.parquet"),),
        non_comparative=True,
        n03_eligible=False,
    )
    write_candidate_evidence(run_result, spec, {}, tmp_path)

    status_data = json.loads((tmp_path / "run_status.json").read_text())
    assert status_data["n03_eligible"] is False
    assert status_data["non_comparative"] is True
