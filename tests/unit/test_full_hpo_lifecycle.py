"""No estimator fits: authenticated transport and real parent journal tests."""

import hashlib
from dataclasses import replace

import pytest
from test_full_nested_tuning import _make_hpo_journal_fixture

from project.experiments.full_hpo_lifecycle import (
    HpoLifecyclePlan,
    ParentHpoLifecycle,
    WorkerHpoLifecycle,
    sign_message,
)
from project.experiments.full_nested_tuning import full_hpo_study_spec
from project.experiments.full_run_contracts import ArtifactEvidence


@pytest.fixture
def lane(tmp_path):
    journal, spec = _make_hpo_journal_fixture(tmp_path, full_hpo_study_spec())
    plan = HpoLifecyclePlan(spec, 0, 0, "work-session", "nonce-session")
    def verify(identity, record, evidence):
        assert record.selected_parameters == identity.parameters
        return 0.7 if identity.phase == "inner" else None
    parent = ParentHpoLifecycle(journal, plan, "test-key", verify_completion=verify)
    worker = WorkerHpoLifecycle(plan, "test-key", parent.handle)
    yield journal, plan, parent, worker, verify
    journal.close()


def evidence(journal, record):
    path = journal.root / record.artifact_directory / "proof.json"
    path.write_text('{"fixture":true}')
    return (ArtifactEvidence(str(path.relative_to(journal.root)), hashlib.sha256(path.read_bytes()).hexdigest()),)


def test_debit_before_constructor_and_authenticated_completion(lane):
    journal, plan, _parent, worker, _ = lane
    calls = []
    def construct():
        assert journal.used_model_fits == 1
        calls.append("constructor")
        return object()
    def perform(model, record):
        calls.append("fit")
        return evidence(journal, record)
    assert worker.run_fit(plan.default(), construct, perform) is not None
    assert calls == ["constructor", "fit"]
    assert journal.attempts()[0].status == "completed"
    assert worker.used_model_fits == 1


def test_replay_and_identity_mutation_do_not_debit(lane):
    journal, plan, parent, _, _ = lane
    payload = {"version": 1, "kind": "hpo_event", "event": "acquire", "seq": 1, "identity": plan.default().payload(),
               "evidence": [], "reason": None, "attempt_id": None}
    envelope = sign_message(payload, "test-key")
    parent.handle(envelope)
    with pytest.raises(ValueError, match="sequence|transition"):
        parent.handle(envelope)
    assert journal.used_model_fits == 1
    changed = dict(payload, seq=2, identity=replace(plan.default(), nonce="other").payload())
    with pytest.raises(ValueError, match="identity|transition"):
        parent.handle(sign_message(changed, "test-key"))
    assert journal.used_model_fits == 1


def test_forged_ack_prevents_constructor(lane):
    journal, plan, parent, _, _ = lane
    def forged(envelope):
        reply = parent.handle(envelope)
        reply["signature"] = "0" * 64
        return reply
    worker = WorkerHpoLifecycle(plan, "test-key", forged)
    calls = []
    with pytest.raises(ValueError, match="signature"):
        worker.run_fit(plan.default(), lambda: calls.append("constructor"), lambda *_: ())
    assert not calls
    assert journal.used_model_fits == 1  # uncertain authorization remains charged


def test_explicit_parent_retry_charges_again(lane):
    journal, plan, _parent, worker, verify = lane
    def broken():
        raise RuntimeError("constructor failure")
    with pytest.raises(RuntimeError, match="constructor failure"):
        worker.run_fit(plan.default(), broken, lambda *_: ())
    assert journal.attempts()[0].status == "failed"
    fresh = replace(plan, work_id="retry-session", nonce="retry-nonce")
    denied = ParentHpoLifecycle(journal, fresh, "test-key", verify_completion=verify)
    with pytest.raises(ValueError, match="retry"):
        WorkerHpoLifecycle(fresh, "test-key", denied.handle).run_fit(fresh.default(), object, lambda *_: ())
    allowed = ParentHpoLifecycle(journal, fresh, "test-key", verify_completion=verify,
                                retry_reasons={fresh.default().unit: "explicit parent retry"})
    retry = WorkerHpoLifecycle(fresh, "test-key", allowed.handle)
    retry.run_fit(fresh.default(), object, lambda _, record: evidence(journal, record))
    assert journal.used_model_fits == 2


def test_reuse_requires_verified_parent_proof_and_no_constructor(lane):
    journal, plan, _parent, worker, verify = lane
    worker.run_fit(plan.default(), object, lambda _, record: evidence(journal, record))
    fresh = replace(plan, work_id="resume-session", nonce="resume-nonce")
    service = ParentHpoLifecycle(journal, fresh, "test-key", verify_completion=verify)
    resumed = WorkerHpoLifecycle(fresh, "test-key", service.handle)
    assert resumed.run_fit(fresh.default(), lambda: pytest.fail("reuse constructed"), lambda *_: ()) is None
    assert journal.used_model_fits == 1
    assert resumed.used_model_fits == 0
    assert hasattr(resumed, "completed_proof"), "reuse must expose authentic completed evidence"
    assert resumed.completed_proof == journal.attempts()[0]


def test_complete_ordered_plan_and_tie_break_without_training(lane):
    journal, plan, _parent, worker, _ = lane
    units = [plan.default()] + [plan.inner(t, f) for t in range(8) for f in range(3)]
    units.append(plan.selected(0))
    for identity in units:
        worker.run_fit(identity, object, lambda _, record: evidence(journal, record))
    assert journal.used_model_fits == 26
    assert len(journal.verify_completed()) == 26
    assert worker.reserved_model_fits == 26


def test_trial_config_and_selected_winner_are_parent_checked(lane):
    journal, plan, parent, worker, _ = lane
    wrong = replace(plan.default(), config_sha256="0" * 64)
    with pytest.raises(ValueError, match="identity"):
        worker.run_fit(wrong, object, lambda *_: ())
    assert journal.used_model_fits == 0
    fresh_worker = WorkerHpoLifecycle(plan, "test-key", parent.handle)
    with pytest.raises(ValueError, match="identity|transition"):
        fresh_worker.run_fit(plan.selected(0), object, lambda *_: ())
    assert journal.used_model_fits == 0


def test_completion_attempt_identity_is_not_optional(lane):
    journal, plan, parent, _, _ = lane
    acquire = {"version": 1, "kind": "hpo_event", "event": "acquire", "seq": 1, "identity": plan.default().payload(),
               "evidence": [], "reason": None, "attempt_id": None}
    reply = parent.handle(sign_message(acquire, "test-key"))
    completion = dict(acquire, event="completed", seq=2, attempt_id="forged-attempt")
    with pytest.raises(ValueError, match="attempt"):
        parent.handle(sign_message(completion, "test-key"))
    assert journal.attempts()[0].status == "running"
    assert reply["payload"]["record"]["attempt_id"] != "forged-attempt"


@pytest.mark.parametrize("field,value", [("seq", True), ("seq", 2), ("event", "shard_closed")])
def test_strict_event_schema_and_sequence(lane, field, value):
    journal, plan, parent, _, _ = lane
    request = {"version": 1, "kind": "hpo_event", "event": "acquire", "seq": 1, "identity": plan.default().payload(),
               "evidence": [], "reason": None, "attempt_id": None}
    request[field] = value
    with pytest.raises(ValueError):
        parent.handle(sign_message(request, "test-key"))
    assert journal.used_model_fits == 0


def test_tampered_completed_proof_cannot_be_reused(lane):
    journal, plan, _parent, worker, verify = lane
    worker.run_fit(plan.default(), object, lambda _, record: evidence(journal, record))
    record = journal.attempts()[0]
    (journal.root / record.evidence[0].path).write_text("tampered")
    fresh = replace(plan, work_id="new-work", nonce="new-nonce")
    service = ParentHpoLifecycle(journal, fresh, "test-key", verify_completion=verify)
    with pytest.raises(ValueError, match="hash"):
        WorkerHpoLifecycle(fresh, "test-key", service.handle).run_fit(
            fresh.default(), lambda: pytest.fail("tampered reuse constructed"), lambda *_: ())
    assert journal.used_model_fits == 1


def test_independent_completion_verifier_failure_keeps_debit_unfinished(lane):
    journal, plan, _, _, _ = lane
    def reject(*args):
        raise ValueError("scientific proof rejected")
    service = ParentHpoLifecycle(journal, plan, "test-key", verify_completion=reject)
    worker = WorkerHpoLifecycle(plan, "test-key", service.handle)
    with pytest.raises(ValueError, match="scientific proof"):
        worker.run_fit(plan.default(), object, lambda _, record: evidence(journal, record))
    assert journal.used_model_fits == 1
    assert journal.attempts()[0].status == "awaiting_validation"
    with pytest.raises(ValueError, match="fresh"):
        worker.run_fit(plan.default(), lambda: pytest.fail("poisoned constructor"), lambda *_: ())


def test_selected_wrong_winner_rejected_after_inner_proofs(lane):
    journal, plan, _parent, worker, _ = lane
    for identity in [plan.default()] + [plan.inner(t, f) for t in range(8) for f in range(3)]:
        worker.run_fit(identity, object, lambda _, record: evidence(journal, record))
    with pytest.raises(ValueError, match="identity"):
        worker.run_fit(plan.selected(1), lambda: pytest.fail("wrong winner constructed"), lambda *_: ())
    assert journal.used_model_fits == 25


def test_nonapproved_study_and_boolean_partition_rejected(lane):
    _, plan, _, _, _ = lane
    with pytest.raises(ValueError, match="partition"):
        replace(plan, fold=True)
    snapshot = replace(plan.spec.candidates[0], study=full_hpo_study_spec(n_trials=2))
    with pytest.raises(ValueError, match="StudySpec"):
        replace(plan, spec=replace(plan.spec, candidates=(snapshot,)))


def test_keyboard_interrupt_is_owned_interruption_not_failed_fit(lane):
    journal, plan, _parent, worker, _ = lane
    def interrupt():
        raise KeyboardInterrupt("fixture interruption")
    with pytest.raises(KeyboardInterrupt):
        worker.run_fit(plan.default(), interrupt, lambda *_: ())
    assert journal.attempts()[0].status == "interrupted"


def test_supervisor_reconciles_only_its_owned_attempt(lane):
    journal, plan, parent, _, verify = lane
    assert callable(getattr(parent, "terminalize_owned", None)), "owned-only supervisor terminalization missing"
    unrelated = journal.begin(plan.inner(7, 2).unit)
    parent.terminalize_owned("interrupted", reason="fixture supervisor exit confirmed")
    assert journal.attempts()[0] == unrelated
    journal.finish(unrelated.attempt_id, "failed", reason="fixture unrelated cleanup")
    plan = replace(plan, work_id="fresh-owner", nonce="fresh-owner-nonce")
    parent = ParentHpoLifecycle(journal, plan, "test-key", verify_completion=verify)
    payload = {"version": 1, "kind": "hpo_event", "event": "acquire", "seq": 1,
               "identity": plan.default().payload(), "evidence": [], "reason": None, "attempt_id": None}
    parent.handle(sign_message(payload, "test-key"))
    parent.terminalize_owned("interrupted", reason="fixture supervisor exit confirmed")
    assert journal.attempts()[-1].status == "interrupted"
    assert journal.attempts()[0].status == "failed"


def test_declared_fixed_model_parameters_cannot_be_silently_ignored(lane):
    _, plan, _, _, _ = lane
    snapshot = plan.spec.candidates[0]
    wrong = replace(snapshot, spec=replace(snapshot.spec, model_params=(("random_state", 41),)))
    with pytest.raises(ValueError, match="configuration|parameters"):
        replace(plan, spec=replace(plan.spec, candidates=(wrong,)))


def test_approved_study_space_preserves_exact_numeric_types(lane):
    _, plan, _, _, _ = lane
    snapshot = plan.spec.candidates[0]
    study = snapshot.study
    parameters = tuple((name, (0, 1) if name == "l2_regularization" else values)
                       for name, values in study.parameter_lists)
    changed = replace(snapshot, study=replace(study, parameter_lists=parameters))
    assert changed.study == study  # Python numeric equality is not exact identity.
    with pytest.raises(ValueError, match="StudySpec"):
        replace(plan, spec=replace(plan.spec, candidates=(changed,)))
