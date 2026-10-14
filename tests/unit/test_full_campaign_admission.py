"""Real temporary source bytes protect read-only admission and provenance."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from project.experiments import full_campaign_admission as admission_module
from project.experiments.candidate_contracts import CandidateSpec, StudySpec
from project.experiments.full_campaign_admission import (
    FullCampaignIntent,
    inspect_full_campaign,
)
from project.experiments.full_run_contracts import (
    FullCandidateSnapshot,
    FullResourcePolicy,
    FullRunSpec,
    SourceBinding,
)

ROOT = Path(__file__).resolve().parents[2]
MATRIX = "configs/experiments/feature_campaign.yaml"
NOTEBOOK = "notebooks/02_baseline_and_candidate_design.ipynb"
BASELINE = "configs/experiments/baselines.yaml"
PRODUCER = "src/project/features/registry.py"


def binding(root, role, path):
    return SourceBinding(role, path, hashlib.sha256((root / path).read_bytes()).hexdigest())


@pytest.fixture
def request_source(tmp_path):
    for path in (MATRIX, BASELINE, PRODUCER):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / path).read_bytes())
    # Copy actual source cells, excluding outputs: provenance is source, not a
    # notebook's rendered table or execution count.
    notebook = json.loads((ROOT / NOTEBOOK).read_bytes())
    cells = [{"id": c["id"], "cell_type": c["cell_type"], "source": c["source"]}
             for c in notebook["cells"]]
    destination = tmp_path / NOTEBOOK
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps({"cells": cells}))
    baseline = yaml.safe_load((tmp_path / BASELINE).read_bytes())
    pool = tuple(baseline["features"]["numeric"] + baseline["features"]["categorical"])
    candidate = FullCandidateSnapshot(
        CandidateSpec("feat_missingness_indicators", pool, "hist_gradient_boosting", ()),
        ("control", "challenger"), NOTEBOOK, "cand-matrix-render",
    )
    sources = tuple(binding(tmp_path, role, path) for role, path in (
        ("matrix", MATRIX), ("baseline", BASELINE), ("producer", PRODUCER), ("notebook", NOTEBOOK)))
    # Explicit declarations only; admission must not open these absent files.
    sources += (SourceBinding("data", "outputs/prepared/train_joined.parquet", "a" * 64),
                SourceBinding("split", "outputs/splits/building_folds_repeated.parquet", "b" * 64))
    spec = FullRunSpec("full-request", (candidate,), sources, (("python", "fixture"),), FullResourcePolicy(True))
    return tmp_path, FullCampaignIntent(spec, True)


def row(report, recipe):
    return next(r for r in report.rows if r.record.experiment_id == recipe)


def test_complete_inventory_is_source_order_not_completion(request_source):
    root, intent = request_source
    report = inspect_full_campaign(root, intent)
    matrix = yaml.safe_load((root / MATRIX).read_bytes())["experiment_matrix"]
    assert tuple(r.record.experiment_id for r in report.rows) == tuple(r["id"] for r in matrix)
    ordinary = row(report, "feat_missingness_indicators")
    assert ordinary.record.status == "pending_implementation"
    assert ordinary.planned_model_fits == 24
    assert ordinary.planned_selector_fits == 0
    assert not report.executed and not report.execution_authorized and not report.n03_eligible
    assert all(not r.record.n03_eligible and r.record.status != "completed" for r in report.rows)
    assert {"random_forest", "ebm_gam", "knn", "tabular_neural_net"} <= dict(report.audit_families).keys()
    assert dict(report.context_anchors)["business_incumbent"] == "pending_input"


@pytest.mark.parametrize("change,code", [
    ("missing_cell", "source_cell_missing"),
    ("unrelated_cell", "source_cell_linkage_mismatch"),
    ("forged_digest", "source_hash_mismatch"),
    ("nonmatrix", "experiment_not_in_source_matrix"),
    ("symlink", "source_path_invalid"),
    ("producer", "unsupported_producer"),
    ("baseline_missing", "baseline_source_missing"),
    ("no_request", "explicit_full_request_missing"),
    ("not_sequential", "sequential_execution_required"),
])
def test_source_failures_are_explicit_before_fitting(request_source, change, code):
    root, intent = request_source
    spec = intent.spec
    candidate = spec.candidates[0]
    if change == "missing_cell":
        spec = replace(spec, candidates=(replace(candidate, source_cell_id="missing"),))
    elif change == "unrelated_cell":
        spec = replace(spec, candidates=(replace(candidate, source_cell_id="cand-title"),))
    elif change == "forged_digest":
        spec = replace(spec, sources=tuple(replace(s, sha256="c" * 64) if s.path == MATRIX else s for s in spec.sources))
    elif change == "nonmatrix":
        spec = replace(spec, candidates=(replace(candidate, spec=replace(candidate.spec, experiment_id="invented")),))
    elif change == "symlink":
        original = root / PRODUCER
        original.rename(root / "producer_copy.py")
        original.symlink_to(root / "producer_copy.py")
    elif change == "producer":
        (root / "unknown.py").write_text("# not an approved producer")
        spec = replace(spec, sources=tuple(binding(root, "producer", "unknown.py") if s.role == "producer" else s for s in spec.sources))
    elif change == "baseline_missing":
        spec = replace(spec, sources=tuple(s for s in spec.sources if s.role != "baseline"))
    elif change == "no_request":
        intent = replace(intent, explicit_full_request=False)
    else:
        intent = replace(intent, sequential=False)
    report = inspect_full_campaign(root, replace(intent, spec=spec))
    codes = set(report.blocker_codes) | {c for r in report.rows for c in r.record.blocker_codes}
    assert code in codes
    assert not any(r.record.status == "ready" for r in report.rows)


def test_unresolved_rows_have_real_blockers_and_unknown_hpo_cost(request_source):
    report = inspect_full_campaign(*request_source)
    expected = {
        "feat_core_vs_provided": "no_verified_additional_features",
        "feat_preserve_raw_vs_invalid_sentinel": "sentinel_evidence_unverified",
        "dim_semantic_block_drop": "missing_drop_block",
        "regime_same_family_hpo_vs_default": "hpo_cost_unknown",
        "model_autogluon_core_features": "full_lease_unsupported",
        "blocked_unverified_gte_family": "blocked_unverified_gte",
    }
    for recipe, code in expected.items():
        assert code in row(report, recipe).record.blocker_codes
    hpo = row(report, "regime_same_family_hpo_vs_default")
    assert hpo.planned_model_fits is None and hpo.planned_trials is None
    assert row(report, "prohibited_opaque_building_id_digit_extraction").record.status == "prohibited"


def test_inspection_never_reads_outputs_or_constructs_fits(request_source, monkeypatch):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    def forbidden_fit(*args, **kwargs):
        pytest.fail("admission must never fit")

    for estimator in (ExtraTreesClassifier, HistGradientBoostingClassifier, LogisticRegression, SVC):
        monkeypatch.setattr(estimator, "fit", forbidden_fit)
    root, intent = request_source
    original = Path.open
    def guarded_open(path, *args, **kwargs):
        assert "outputs" not in path.parts and "prepared" not in path.parts
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    before = tuple(root.rglob("*"))
    report = inspect_full_campaign(root, intent)
    assert len(report.rows) == 21
    assert tuple(root.rglob("*")) == before


def test_forged_baseline_protocol_cannot_authenticate_by_self_hash(request_source):
    root, intent = request_source
    path = root / BASELINE
    baseline = yaml.safe_load(path.read_bytes())
    baseline["folds"] = 2
    path.write_text(yaml.safe_dump(baseline))
    spec = replace(intent.spec, sources=tuple(binding(root, s.role, s.path) if s.path == BASELINE else s for s in intent.spec.sources))
    report = inspect_full_campaign(root, replace(intent, spec=spec))
    assert "prediction_contract_mismatch" in report.blocker_codes


def test_matrix_mutation_after_authentication_never_reports_unauthenticated_content(
    request_source, monkeypatch,
):
    root, intent = request_source
    matrix_path = root / MATRIX
    authenticated_digest = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    original_read_source = admission_module._read_source
    state = {"matrix_reads": 0}

    def mutating_read_source(path_root, relative, role):
        if relative == MATRIX:
            state["matrix_reads"] += 1
            if state["matrix_reads"] >= 2:
                # Simulate a hostile writer swapping the matrix between the
                # authenticated read and a later unchecked reread.
                mutated = yaml.safe_load(matrix_path.read_bytes())
                mutated["context_baselines"]["business_incumbent"]["status"] = "changed_after_auth"
                matrix_path.write_text(yaml.safe_dump(mutated))
        return original_read_source(path_root, relative, role)

    monkeypatch.setattr(admission_module, "_read_source", mutating_read_source)
    report = inspect_full_campaign(root, intent)

    matrix_binding = next(s for s in report.authenticated_sources if s.role == "matrix")
    assert matrix_binding.sha256 == authenticated_digest
    context = dict(report.context_source_metadata)
    incumbent = json.loads(context["business_incumbent"])
    # The report must reflect the authenticated bytes, never the mutated ones.
    assert incumbent["status"] != "changed_after_auth"
    assert state["matrix_reads"] == 1


@pytest.mark.parametrize("source_path", [MATRIX, BASELINE])
@pytest.mark.parametrize("field,value", [
    ("primary_metric", "accuracy"),
    ("guardrails", []),
    ("guardrails", ["brier_score", "log_loss", "average_precision"]),
    ("guardrails", ["average_precision", "log_loss", "brier_score", "brier_score"]),
    ("guardrails", ["average_precision", "log_loss", "brier_score", "accuracy"]),
    ("split_manifest", "outputs/splits/other.parquet"),
])
def test_rehashed_source_cannot_change_frozen_evaluation_protocol(
    request_source, monkeypatch, source_path, field, value,
):
    root, intent = request_source
    path = root / source_path
    config = yaml.safe_load(path.read_bytes())
    protocol = config["prediction_contract"] if source_path == MATRIX else config
    protocol[field] = value
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    spec = replace(intent.spec, sources=tuple(
        binding(root, s.role, s.path) if s.path == source_path else s
        for s in intent.spec.sources
    ))
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert "outputs" not in path.parts and "prepared" not in path.parts
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    report = inspect_full_campaign(root, replace(intent, spec=spec))
    assert "prediction_contract_mismatch" in report.blocker_codes
    assert "source_hash_mismatch" not in report.blocker_codes
    assert report.deferred_sources == tuple(s for s in spec.sources if s.role in ("data", "split"))
    assert all("prediction_contract_mismatch" in r.record.blocker_codes for r in report.rows)


def test_declared_split_binding_must_match_both_frozen_sources_without_reads(
    request_source, monkeypatch,
):
    root, intent = request_source
    spec = replace(intent.spec, sources=tuple(
        replace(s, path="outputs/splits/other.parquet") if s.role == "split" else s
        for s in intent.spec.sources
    ))
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert "outputs" not in path.parts and "prepared" not in path.parts
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    report = inspect_full_campaign(root, replace(intent, spec=spec))
    assert "split_source_path_mismatch" in report.blocker_codes
    assert "prediction_contract_mismatch" in report.blocker_codes
    assert report.deferred_sources == tuple(s for s in spec.sources if s.role in ("data", "split"))


@pytest.mark.parametrize("change", ["context", "audit", "protocol", "matrix"])
def test_matrix_mutation_after_authentication_uses_authenticated_snapshot(
    request_source, monkeypatch, change,
):
    root, intent = request_source
    original_bytes = (root / MATRIX).read_bytes()
    original_campaign = yaml.safe_load(original_bytes)
    changed_campaign = yaml.safe_load(original_bytes)
    if change == "context":
        changed_campaign["context_baselines"]["business_incumbent"]["status"] = "changed_after_authentication"
    elif change == "audit":
        changed_campaign["model_family_audit"]["random_forest"]["decision_reason"] = "changed_after_authentication"
    elif change == "protocol":
        changed_campaign["prediction_contract"]["guardrails"] = []
    else:
        changed_campaign["experiment_matrix"][1]["model_family"] = "random_forest"
    changed_bytes = yaml.safe_dump(changed_campaign, sort_keys=False).encode()
    original_read_bytes = Path.read_bytes
    original_open = Path.open
    mutated = False

    def mutate_after_matrix_verification(path):
        nonlocal mutated
        payload = original_read_bytes(path)
        # Notebook is authenticated after the matrix. Mutate the real temporary
        # file at this IO boundary, without replacing admission/authentication.
        if path == root / NOTEBOOK and not mutated:
            (root / MATRIX).write_bytes(changed_bytes)
            mutated = True
        return payload

    def guarded_open(path, *args, **kwargs):
        assert "outputs" not in path.parts and "prepared" not in path.parts
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", mutate_after_matrix_verification)
    monkeypatch.setattr(Path, "open", guarded_open)
    report = inspect_full_campaign(root, intent)
    assert mutated
    matrix_binding = next(s for s in report.authenticated_sources if s.role == "matrix")
    assert matrix_binding.sha256 == hashlib.sha256(original_bytes).hexdigest()
    assert {key: json.loads(value) for key, value in report.context_source_metadata} == original_campaign["context_baselines"]
    assert {key: json.loads(value) for key, value in report.audit_source_metadata} == original_campaign["model_family_audit"]
    assert tuple(r.model_family for r in report.rows) == tuple(r["model_family"] for r in original_campaign["experiment_matrix"])
    assert "prediction_contract_mismatch" not in report.blocker_codes
    assert "source_matrix_mismatch" not in report.blocker_codes


@pytest.mark.parametrize("rejection", ["forged_digest", "missing_matrix_path"])
def test_rejected_matrix_binding_fallback_preserves_inventory_without_authentication(
    request_source, monkeypatch, rejection,
):
    root, intent = request_source
    changes = {"sha256": "c" * 64} if rejection == "forged_digest" else {
        "path": "configs/experiments/missing_matrix.yaml",
    }
    sources = tuple(
        replace(s, **changes) if s.role == "matrix" else s
        for s in intent.spec.sources
    )
    spec = replace(intent.spec, sources=sources)
    matrix = yaml.safe_load((root / MATRIX).read_bytes())["experiment_matrix"]
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert "outputs" not in path.parts and "prepared" not in path.parts
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    report = inspect_full_campaign(root, replace(intent, spec=spec))
    assert tuple(r.record.experiment_id for r in report.rows) == tuple(r["id"] for r in matrix)
    assert "matrix_source_missing" in report.blocker_codes
    assert not any(s.role == "matrix" for s in report.authenticated_sources)
    assert all(r.record.status in ("blocked", "prohibited") for r in report.rows)


@pytest.mark.parametrize("recipe,methods,selector,arms,models,selectors,encoders", [
    ("feat_histogram_quantile", (), (), ("control", "bin", "quantile"), 36, 0, 0),
    ("dim_fold_local_selector_vs_passthrough", (), ("challenger",), ("control", "challenger"), 24, 12, 0),
    ("feat_conditional_cross_fitted_target_encoding", (("n_splits", 4),), (), ("control", "challenger"), 24, 0, 60),
])
def test_exact_accounting_remains_proposed(request_source, recipe, methods, selector, arms, models, selectors, encoders):
    root, intent = request_source
    candidate = intent.spec.candidates[0]
    family = "logistic_l2" if "selector" in recipe else "hist_gradient_boosting"
    candidate = replace(candidate, spec=replace(candidate.spec, experiment_id=recipe, model_family=family, method_params=methods), arms=arms, selector_arms=selector)
    report = inspect_full_campaign(root, replace(intent, spec=replace(intent.spec, candidates=(candidate,))))
    result = row(report, recipe)
    assert (result.planned_model_fits, result.planned_selector_fits, result.encoder_objects) == (models, selectors, encoders)
    assert not result.record.n03_eligible


@pytest.mark.parametrize("methods,code", [
    ((("data_path", "outputs/prepared/train_joined.parquet"),), "invalid_candidate_parameters"),
    ((("n_splits", True),), "invalid_candidate_parameters"),
    ((("n_splits", 0),), "invalid_candidate_parameters"),
])
def test_invalid_methods_never_receive_planned_costs(request_source, methods, code):
    root, intent = request_source
    candidate = intent.spec.candidates[0]
    recipe = "feat_conditional_cross_fitted_target_encoding"
    candidate = replace(candidate, spec=replace(candidate.spec, experiment_id=recipe, method_params=methods))
    report = inspect_full_campaign(root, replace(intent, spec=replace(intent.spec, candidates=(candidate,))))
    result = row(report, recipe)
    assert code in result.record.blocker_codes
    assert result.planned_model_fits is None


def test_missing_sklearn_dependency_is_not_caller_override(request_source, monkeypatch):
    import importlib.util
    original = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "sklearn" else original(name))
    report = inspect_full_campaign(*request_source)
    assert "required_dependency_unavailable" in report.blocker_codes


def test_matrix_family_tampering_is_blocked_even_when_rehashed(request_source):
    root, intent = request_source
    path = root / MATRIX
    campaign = yaml.safe_load(path.read_bytes())
    campaign["experiment_matrix"][1]["model_family"] = "random_forest"
    path.write_text(yaml.safe_dump(campaign, sort_keys=False))
    spec = replace(intent.spec, sources=tuple(binding(root, s.role, s.path) if s.path == MATRIX else s for s in intent.spec.sources))
    report = inspect_full_campaign(root, replace(intent, spec=spec))
    assert "source_matrix_mismatch" in report.blocker_codes


def test_baseline_core_tampering_is_not_verified_producer_evidence(request_source):
    root, intent = request_source
    path = root / BASELINE
    baseline = yaml.safe_load(path.read_bytes())
    baseline["features"]["numeric"].append("made_up_predictor")
    path.write_text(yaml.safe_dump(baseline))
    spec = replace(intent.spec, sources=tuple(binding(root, s.role, s.path) if s.path == BASELINE else s for s in intent.spec.sources))
    report = inspect_full_campaign(root, replace(intent, spec=spec))
    assert "baseline_feature_contract_mismatch" in report.blocker_codes


def test_root_parent_alias_is_rejected(request_source):
    root, intent = request_source
    (root / "alias").mkdir()
    with pytest.raises(ValueError, match="source_path_invalid"):
        inspect_full_campaign(root / "alias" / "..", intent)


def test_explicit_test_hpo_cost_does_not_approve_random_search(request_source):
    root, intent = request_source
    candidate = intent.spec.candidates[0]
    recipe = "regime_same_family_hpo_vs_default"
    study = StudySpec(2, 3, 42, (("learning_rate", (.03, .1)),))
    candidate = replace(candidate, spec=replace(candidate.spec, experiment_id=recipe), study=study, study_method="random_search")
    report = inspect_full_campaign(root, replace(intent, spec=replace(intent.spec, candidates=(candidate,))))
    result = row(report, recipe)
    assert result.planned_model_fits == 12 * (2 * 3 + 2)
    assert result.planned_trials == 24
    assert "hpo_method_space_unapproved" in result.record.blocker_codes
    assert "hpo_dispatch_missing" in result.record.blocker_codes
    assert result.record.status == "blocked"


def test_misbound_dataset_reference_is_blocked_without_opening(request_source):
    root, intent = request_source
    spec = replace(intent.spec, sources=tuple(replace(s, path="outputs/prepared/test_joined.parquet") if s.role == "data" else s for s in intent.spec.sources))
    report = inspect_full_campaign(root, replace(intent, spec=spec))
    assert "data_source_path_mismatch" in report.blocker_codes


def test_source_metadata_retains_optout_and_binding(request_source):
    root, intent = request_source
    report = inspect_full_campaign(root, intent)
    assert report.requested_spec_sha256 == intent.spec.binding_sha256
    assert report.resource_policy == FullResourcePolicy(True)
    assert report.partition_count == 12
    assert report.vault_excluded is True
    assert report.full_request_recorded is True
    campaign = yaml.safe_load((root / MATRIX).read_bytes())
    assert {key: json.loads(value) for key, value in report.audit_source_metadata} == campaign["model_family_audit"]
    assert {key: json.loads(value) for key, value in report.context_source_metadata} == campaign["context_baselines"]
    assert {s.role for s in report.deferred_sources} == {"data", "split"}
    assert all(s.role not in ("data", "split") for s in report.authenticated_sources)


def test_frozen_explicit_model_parameters_are_not_rejected_as_unknown(request_source):
    root, intent = request_source
    candidate = intent.spec.candidates[0]
    candidate = replace(candidate, spec=replace(candidate.spec, model_params=(("early_stopping", False), ("random_state", 42))))
    report = inspect_full_campaign(root, replace(intent, spec=replace(intent.spec, candidates=(candidate,))))
    result = row(report, candidate.spec.experiment_id)
    assert result.record.status == "pending_implementation"
    assert "model_profile_unverified" not in result.record.blocker_codes


def test_changed_model_identity_cannot_advertise_exact_fit_plan(request_source):
    root, intent = request_source
    candidate = intent.spec.candidates[0]
    candidate = replace(candidate, spec=replace(candidate.spec, model_params=(("early_stopping", True),)))
    report = inspect_full_campaign(root, replace(intent, spec=replace(intent.spec, candidates=(candidate,))))
    result = row(report, candidate.spec.experiment_id)
    assert "model_profile_unverified" in result.record.blocker_codes
    assert result.planned_model_fits is None


def test_duplicate_source_cell_ids_fail_authentication(request_source):
    root, intent = request_source
    path = root / NOTEBOOK
    notebook = json.loads(path.read_bytes())
    cell = next(c for c in notebook["cells"] if c["id"] == "cand-matrix-render")
    notebook["cells"].append(cell)
    path.write_text(json.dumps(notebook))
    spec = replace(intent.spec, sources=tuple(binding(root, s.role, s.path) if s.path == NOTEBOOK else s for s in intent.spec.sources))
    report = inspect_full_campaign(root, replace(intent, spec=spec))
    assert "source_cell_missing" in row(report, "feat_missingness_indicators").record.blocker_codes
