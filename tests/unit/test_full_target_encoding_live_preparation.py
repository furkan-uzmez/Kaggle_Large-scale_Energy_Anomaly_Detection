"""Nonfit TE preparation is not production capability or D/R authority."""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from project.experiments import full_campaign_worker as worker
from project.experiments.candidate_contracts import CandidateSpec
from project.experiments.full_campaign_capabilities import (
    ORDINARY_EXPERIMENTS,
    build_reviewed_ordinary_arms,
    resolve_full_campaign_capability,
    verify_full_campaign_capability,
)
from project.experiments.full_run_contracts import FullCandidateSnapshot

ROOT = Path(__file__).resolve().parents[2]
TARGET = "feat_conditional_cross_fitted_target_encoding"
METHODS = (("smoothing", 10.0), ("n_splits", 3), ("seed", 42), ("max_categories", 4096))
POOL = ("meter_reading", "square_feet", "year_built", "floor_count", "hour",
        "weekday", "month", "year", "primary_use")
ENCODING = "lead_year_offset_1900_uint8_v1"


def snapshot():
    return FullCandidateSnapshot(
        CandidateSpec(TARGET, POOL, "hist_gradient_boosting",
                      (("early_stopping", False), ("random_state", 42)), METHODS),
        ("control", "challenger"), "notebooks/02_baseline_and_candidate_design.ipynb",
        "cand-matrix-render")


def test_exact_unfitted_te_graph_without_capability_promotion():
    arms = build_reviewed_ordinary_arms(snapshot(), year_built_encoding=ENCODING)
    assert tuple(arms) == ("control", "challenger")
    assert "features" not in arms["control"].named_steps
    encoder = arms["challenger"].named_steps["features"]
    assert (encoder.n_splits, encoder.seed, encoder.smoothing, encoder.max_categories) == (3, 42, 10.0, 4096)
    assert not hasattr(encoder, "cross_fitted_")
    assert arms["challenger"].fit_cost_metadata["encoder_fitted_objects"] == 4
    assert all(not hasattr(arm.named_steps["model"], "classes_") for arm in arms.values())
    assert len(snapshot().planned_units) == 24


@pytest.mark.parametrize("methods", [(), METHODS[:-1], tuple(reversed(METHODS)),
    (("smoothing", 10),) + METHODS[1:], METHODS[:1] + (("n_splits", 3.0),) + METHODS[2:],
    METHODS[:2] + (("seed", 43),) + METHODS[3:]])
def test_te_method_drift_rejected(methods):
    original = snapshot()
    with pytest.raises(ValueError, match="reviewed|frozen"):
        build_reviewed_ordinary_arms(replace(original, spec=replace(original.spec, method_params=methods)),
                                     year_built_encoding=ENCODING)


def test_existing_fourteen_capability_and_unfitted_graphs_unchanged():
    capability = resolve_full_campaign_capability(ROOT)
    assert capability.available and len(ORDINARY_EXPERIMENTS) == 14
    assert capability.supported_experiment_ids == ORDINARY_EXPERIMENTS
    assert TARGET not in capability.supported_experiment_ids
    verify_full_campaign_capability(ROOT, capability)
    config = yaml.safe_load((ROOT / "configs/experiments/candidate_full_v2.yaml").read_text())
    for row in config["candidates"]:
        candidate = FullCandidateSnapshot(
            CandidateSpec(row["experiment_id"], tuple(row["feature_pool"]), row["model_family"],
                          tuple(row["model_params"].items()), tuple(row["method_params"].items())),
            tuple(row["arms"]), row["source_notebook"], row["source_cell_id"],
            selector_arms=tuple(row["selector_arms"]))
        assert tuple(build_reviewed_ordinary_arms(candidate, year_built_encoding=ENCODING)) == candidate.arms


@pytest.mark.parametrize("state", [{}, {"cross_fitted_": False}, {"cross_fitted_": 1},
    {"cross_fitted_": True}, {"cross_fitted_": True, "inner_membership_digest_": "b" * 64}])
def test_postfit_state_rejected_without_fitting(state):
    guard = getattr(worker, "_verify_fitted_target_encoding", None)
    assert callable(guard), "worker requires a fitted-state guard"
    with pytest.raises(ValueError, match="cross.fitted|membership"):
        guard(SimpleNamespace(arm_id="challenger", named_steps={"features": SimpleNamespace(**state)}), "a" * 64)


def test_control_postfit_guard_without_encoder():
    guard = getattr(worker, "_verify_fitted_target_encoding", None)
    assert callable(guard), "worker requires a fitted-state guard"
    guard(SimpleNamespace(arm_id="control", named_steps={}), "a" * 64)




def test_selector_ack_budget_preserved():
    assert worker.SELECTOR_SUCCESS_ACK_TIMEOUT_SECONDS == 300.0
    assert worker._ack_budget_seconds("selector_success", 3.0) == 300.0
    assert worker._ack_budget_seconds("debit", 3.0) == 3.0


def test_te_only_preflight_config():
    path = ROOT / "configs/experiments/target_encoding_full_preflight_v1.yaml"
    assert path.is_file(), "new TE-only config required; old configs are immutable"
    config = yaml.safe_load(path.read_text())
    assert len(config["candidates"]) == 1
    row = config["candidates"][0]
    assert row["experiment_id"] == TARGET and tuple(row["method_params"].items()) == METHODS
    assert row["arms"] == ["control", "challenger"] and row["selector_arms"] == []


@pytest.mark.parametrize("mutation", ["labels", "groups", "encoding"])
def test_real_partition_audit_guard_rejects_mutation_before_any_debit(tmp_path, monkeypatch, mutation):
    from full_campaign_fixtures import write_admission_fixture
    from project.experiments import full_development_readiness as readiness
    from project.experiments.full_candidate_io import (
        build_development_index,
        load_full_training_partition,
    )
    from project.experiments.full_development_audits import (
        AuditPlan,
        audit_full_development,
        write_audit_evidence,
    )

    fixture = write_admission_fixture(tmp_path / "synthetic-source")
    candidate = replace(fixture.spec.candidates[0], spec=snapshot().spec)
    spec = replace(fixture.spec, candidates=(candidate,))
    report = audit_full_development(fixture.root, spec, AuditPlan((2, 3, 4, 5), 42))
    # Fixture-only authority; never alter production pin/root or claim a grant.
    path = fixture.root / readiness.EVIDENCE_PATH
    digest = write_audit_evidence(path, report)
    monkeypatch.setattr(readiness, "_REFERENCE_ROOT", fixture.root)
    monkeypatch.setattr(readiness, "EVIDENCE_SHA256", digest)
    index = build_development_index(fixture.root, spec, fixture.root / "synthetic-index.sqlite3")
    partition = load_full_training_partition(index, spec, 0, 0)
    assert len(readiness.verify_target_encoding_partition(fixture.root, spec, partition)) == 64
    if mutation == "labels":
        labels = partition.y_train.copy()
        labels.iloc[0] = 1 - labels.iloc[0]
        partition = replace(partition, y_train=labels)
    elif mutation == "groups":
        groups = partition.groups_train.copy()
        groups.iloc[0] = "synthetic-unbound-group"
        partition = replace(partition, groups_train=groups)
    else:
        binding = next(s for s in spec.sources if s.role == "encoding")
        (fixture.root / binding.path).write_text("changed synthetic encoding")
    with pytest.raises(ValueError):
        readiness.verify_target_encoding_partition(fixture.root, spec, partition)
    assert not (fixture.root / "journal").exists()
