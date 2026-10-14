"""Contract tests with actual SQLite and confined evidence files."""

import hashlib
import json
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from project.experiments import full_run_contracts as contracts
from project.experiments.candidate_contracts import CandidateSpec, FitBudget


def api(name):
    value = getattr(contracts, name, None)
    assert value is not None, f"missing full-run behavior: {name}"
    return value


def make_spec(root):
    binding = api("SourceBinding")
    sources = []
    for role in ("matrix", "data", "split", "producer", "notebook"):
        path = root / f"{role}.json"
        content = json.dumps({"cells": [{"id": "real-cell"}]}) if role == "notebook" else role
        path.write_text(content)
        sources.append(binding(role, path.name, hashlib.sha256(path.read_bytes()).hexdigest()))
    candidate = CandidateSpec("feat_missingness_indicators", ("meter_reading",), "hist_gradient_boosting", ())
    snapshot = api("FullCandidateSnapshot")(candidate, ("control", "challenger"), "notebook.json", "real-cell")
    return api("FullRunSpec")("run-one", (snapshot,), tuple(sources), (("python", "3.13"),), api("FullResourcePolicy")(informed_opt_out=True))


def test_policy_requires_exact_informed_null_false_optout():
    policy = api("FullResourcePolicy")
    assert policy(informed_opt_out=True).max_rss_bytes is None
    for kwargs in ({"informed_opt_out": False}, {"cpu_threads": True}, {"max_wall_seconds": float("inf")}, {"rss_limit_enforced": 0}):
        with pytest.raises(ValueError):
            policy(**{"informed_opt_out": True, **kwargs})
    with pytest.raises(ValueError):
        FitBudget(12, 0, 10, None)


def test_resource_optout_must_be_supplied_explicitly():
    with pytest.raises(TypeError):
        api("FullResourcePolicy")()


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "/absolute", "..", "a\\b"])
def test_run_id_cannot_be_a_path(tmp_path, run_id):
    with pytest.raises(ValueError, match="run_id"):
        replace(make_spec(tmp_path), run_id=run_id)


def test_spec_is_immutable_authenticated_and_frozen_protocol(tmp_path):
    spec = make_spec(tmp_path)
    spec.verify_bindings(tmp_path)
    assert len(spec.planned_units) == 24
    with pytest.raises(FrozenInstanceError):
        spec.run_id = "changed"
    for kwargs in ({"development_rows": True}, {"folds": 3}, {"split_seeds": (42, 43, 45)}, {"environment": (("python", float("nan")),)}):
        with pytest.raises((ValueError, TypeError)):
            replace(spec, **kwargs)
    with pytest.raises(ValueError, match="cell"):
        replace(spec, candidates=(replace(spec.candidates[0], source_cell_id="forged"),)).verify_bindings(tmp_path)
    (tmp_path / "data.json").write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        spec.verify_bindings(tmp_path)


def test_attempts_persist_and_retry_is_explicit(tmp_path):
    spec = make_spec(tmp_path)
    journal_type = api("FullFitJournal")
    with journal_type(tmp_path / "run", spec, source_root=tmp_path) as journal:
        unit = spec.planned_units[0]
        first = journal.begin(unit)
        journal.finish(first.attempt_id, "failed", reason="construction failed")
        with pytest.raises(ValueError, match="retry"):
            journal.begin(unit)
        second = journal.retry(unit, reason="explicit retry")
        journal.finish(second.attempt_id, "interrupted", reason="worker disappeared")
        assert journal.used_model_fits == 2
    with journal_type(tmp_path / "run", spec, source_root=tmp_path) as journal:
        assert [a.status for a in journal.attempts()] == ["failed", "interrupted"]
        assert journal.used_model_fits == 2
    with pytest.raises(ValueError, match="binding"):
        journal_type(tmp_path / "run", replace(spec, environment=(("python", "other"),)), source_root=tmp_path)


def test_journal_rejects_modified_events(tmp_path):
    import sqlite3

    spec = make_spec(tmp_path)
    journal_type = api("FullFitJournal")
    root = tmp_path / "run"
    with journal_type(root, spec, source_root=tmp_path) as journal:
        attempt = journal.begin(spec.planned_units[0])
        journal.finish(attempt.attempt_id, "failed", reason="failure")
    with sqlite3.connect(root / "full_fit.sqlite3") as connection:
        connection.execute("UPDATE events SET payload='{}' WHERE sequence=1")
    with pytest.raises(ValueError, match="hash"):
        journal_type(root, spec, source_root=tmp_path)


def test_finite_costs_and_selector_accounting(tmp_path):
    from project.experiments.candidate_contracts import StudySpec

    spec = make_spec(tmp_path)
    snapshot = spec.candidates[0]
    assert len(replace(snapshot, arms=("control", "bin", "quantile")).planned_units) == 36
    selector = replace(snapshot, selector_arms=("challenger",))
    assert len(selector.planned_units) == 36
    hpo = replace(snapshot, spec=replace(snapshot.spec, experiment_id="regime_same_family_hpo_vs_default"), study=StudySpec(2, 3, 42, (("max_iter", (100, 200)),)), study_method="random_search")
    assert len(hpo.planned_units) == 12 * (2 * 3 + 2)
    assert replace(spec, candidates=(hpo,)).planned_trials == 24
    with pytest.raises(ValueError):
        replace(hpo, study=None)
    with api("FullFitJournal")(tmp_path / "run", replace(spec, candidates=(selector,)), source_root=tmp_path) as journal:
        unit = next(u for u in selector.planned_units if u.endswith("/selector"))
        attempt = journal.begin(unit)
        journal.finish(attempt.attempt_id, "failed", reason="selector failed")
        assert journal.used_selector_fits == 1
        assert journal.used_model_fits == 0


def test_seal_refuses_wrong_hash_external_file_and_resealing(tmp_path):
    spec = make_spec(tmp_path)
    evidence_type = api("ArtifactEvidence")
    with api("FullFitJournal")(tmp_path / "run", spec, source_root=tmp_path) as journal:
        attempt = journal.begin(spec.planned_units[0])
        path = tmp_path / "run" / attempt.artifact_directory / "proof"
        path.write_bytes(b"proof")
        relative = str(path.relative_to(tmp_path / "run"))
        with pytest.raises(ValueError, match="hash"):
            journal.seal(attempt.attempt_id, (evidence_type(relative, "0" * 64),))
        outside = tmp_path / "run" / "other"
        outside.write_bytes(b"proof")
        digest = hashlib.sha256(b"proof").hexdigest()
        with pytest.raises(ValueError, match="exclusive"):
            journal.seal(attempt.attempt_id, (evidence_type("other", digest),))
        proof = (evidence_type(relative, digest),)
        journal.seal(attempt.attempt_id, proof)
        with pytest.raises(ValueError, match="once"):
            journal.seal(attempt.attempt_id, proof)
        journal.finish(attempt.attempt_id, "completed")
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        api("FullFitJournal")(tmp_path / "run", spec, source_root=tmp_path)


def test_completion_verification_rechecks_source_binding(tmp_path):
    spec = make_spec(tmp_path)
    with api("FullFitJournal")(tmp_path / "run", spec, source_root=tmp_path) as journal:
        (tmp_path / "split.json").write_text("changed split")
        with pytest.raises(ValueError, match="hash"):
            journal.verify_completed()


def test_structured_status_never_grants_eligibility_without_completion_proof():
    status_type = api("FullStatusRecord")
    with pytest.raises(ValueError):
        status_type("experiment", "running", n03_eligible=True)
    with pytest.raises(ValueError):
        status_type("experiment", "completed", n03_eligible=True)
    blocked = status_type("experiment", "blocked", blocker_codes=("missing_source",))
    assert blocked.n03_eligible is False


def test_selected_parameter_snapshots_remain_deeply_immutable(tmp_path):
    spec = make_spec(tmp_path)
    with api("FullFitJournal")(tmp_path / "run", spec, source_root=tmp_path) as journal:
        parameters = (("nested", (1, (2, 3))),)
        attempt = journal.begin(spec.planned_units[0], selected_parameters=parameters)
        assert attempt.selected_parameters == parameters
        journal.finish(attempt.attempt_id, "failed", reason="failure")
    with api("FullFitJournal")(tmp_path / "run", spec, source_root=tmp_path) as journal:
        assert journal.attempts()[0].selected_parameters == parameters


def test_corrupted_completed_evidence_freezes_further_work(tmp_path):
    spec = make_spec(tmp_path)
    with api("FullFitJournal")(tmp_path / "run", spec, source_root=tmp_path) as journal:
        attempt = journal.begin(spec.planned_units[0])
        path = tmp_path / "run" / attempt.artifact_directory / "proof"
        path.write_bytes(b"proof")
        journal.seal(attempt.attempt_id, (api("ArtifactEvidence")(str(path.relative_to(tmp_path / "run")), hashlib.sha256(b"proof").hexdigest()),))
        journal.finish(attempt.attempt_id, "completed")
        path.write_bytes(b"changed")
        with pytest.raises(ValueError, match="hash"):
            journal.begin(spec.planned_units[1])


def test_completion_requires_verified_seal_and_cannot_overwrite(tmp_path):
    spec = make_spec(tmp_path)
    with api("FullFitJournal")(tmp_path / "run", spec, source_root=tmp_path) as journal:
        attempt = journal.begin(spec.planned_units[0], selected_parameters=(("max_iter", 100),))
        with pytest.raises(ValueError, match="seal"):
            journal.finish(attempt.attempt_id, "completed")
        evidence = tmp_path / "run" / attempt.artifact_directory / "evidence.json"
        evidence.write_text("verified fit evidence")
        digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
        journal.seal(attempt.attempt_id, (api("ArtifactEvidence")(str(evidence.relative_to(tmp_path / "run")), digest),))
        journal.finish(attempt.attempt_id, "completed")
        assert journal.verify_completed() == (spec.planned_units[0],)
        for action in (lambda: journal.begin(attempt.unit), lambda: journal.retry(attempt.unit, reason="again"), lambda: journal.finish(attempt.attempt_id, "failed")):
            with pytest.raises(ValueError):
                action()
        evidence.write_text("tampered")
        with pytest.raises(ValueError, match="hash"):
            journal.verify_completed()


def test_exclusive_ownership_serial_attempts_and_changed_plan(tmp_path):
    spec = make_spec(tmp_path)
    journal_type = api("FullFitJournal")
    with journal_type(tmp_path / "run", spec, source_root=tmp_path) as journal:
        with pytest.raises(ValueError, match="busy"):
            journal_type(tmp_path / "run", spec, source_root=tmp_path)
        journal.begin(spec.planned_units[0])
        with pytest.raises(ValueError, match="active"):
            journal.begin(spec.planned_units[1])
        with pytest.raises(ValueError):
            journal.begin("unknown/unit")
    changed = replace(spec, candidates=(replace(spec.candidates[0], arms=("control",)),))
    with pytest.raises(ValueError, match="binding"):
        journal_type(tmp_path / "run", changed, source_root=tmp_path)


def test_source_confinement_and_duplicate_units(tmp_path):
    spec = make_spec(tmp_path)
    with pytest.raises(ValueError):
        api("SourceBinding")("data", "../outside", "a" * 64)
    with pytest.raises(ValueError):
        replace(spec.candidates[0], arms=("control", "control"))
    original = tmp_path / "data.json"
    original.unlink()
    original.symlink_to(tmp_path / "matrix.json")
    with pytest.raises(ValueError, match="symlink"):
        spec.verify_bindings(tmp_path)


@pytest.mark.parametrize("binding_name", ("candidate_spec", "environment_spec", "root", "source_root"))
def test_open_journal_bindings_are_read_only(tmp_path, binding_name):
    spec = make_spec(tmp_path)
    root = tmp_path / "run"
    other = tmp_path / "other"
    other.mkdir()
    make_spec(other)
    with api("FullFitJournal")(root, spec, source_root=tmp_path) as journal:
        if binding_name == "candidate_spec":
            name = "spec"
            replacement = replace(spec, candidates=(replace(spec.candidates[0], arms=("control", "challenger", "added")),))
        elif binding_name == "environment_spec":
            name = "spec"
            replacement = replace(spec, environment=(("python", "changed"),))
        else:
            name = binding_name
            replacement = other
        original = getattr(journal, name)
        with pytest.raises(AttributeError):
            setattr(journal, name, replacement)
        assert getattr(journal, name) == original
        with pytest.raises(ValueError, match="unknown planned unit"):
            journal.begin("feat_missingness_indicators/added/r0/f0/model")
        attempt = journal.begin(spec.planned_units[0])
        assert (root / attempt.artifact_directory).is_dir()
        assert not (other / "attempts").exists()
        journal.finish(attempt.attempt_id, "failed", reason="fixture failure")


@pytest.mark.parametrize("operation", ("begin", "retry", "seal", "finish", "verify_completed"))
def test_open_journal_checks_persisted_snapshot_before_authorizing(tmp_path, operation):
    import sqlite3

    spec = make_spec(tmp_path)
    root = tmp_path / "run"
    with api("FullFitJournal")(root, spec, source_root=tmp_path) as journal:
        attempt = None
        evidence = ()
        if operation in ("retry", "seal", "finish"):
            attempt = journal.begin(spec.planned_units[0])
            if operation == "retry":
                journal.finish(attempt.attempt_id, "failed", reason="fixture failure")
            else:
                path = root / attempt.artifact_directory / "proof"
                path.write_bytes(b"proof")
                evidence = (api("ArtifactEvidence")(str(path.relative_to(root)), hashlib.sha256(b"proof").hexdigest()),)
                if operation == "finish":
                    journal.seal(attempt.attempt_id, evidence)
        with sqlite3.connect(root / "full_fit.sqlite3") as connection:
            snapshot, = connection.execute("SELECT snapshot FROM binding").fetchone()
            changed = json.loads(snapshot)
            changed["spec"]["environment"] = [["python", "changed"]]
            replacement = json.dumps(changed, sort_keys=True, separators=(",", ":"))
            # Keep the chain internally consistent: comparison with the
            # original operational binding must independently reject changes.
            head = hashlib.sha256(replacement.encode()).hexdigest()
            for sequence, payload in connection.execute("SELECT sequence, payload FROM events ORDER BY sequence").fetchall():
                head = hashlib.sha256((head + payload).encode()).hexdigest()
                connection.execute("UPDATE events SET digest=? WHERE sequence=?", (head, sequence))
            connection.execute("UPDATE binding SET snapshot=?, head=?", (replacement, head))
            event_count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        with pytest.raises(ValueError, match="binding"):
            if operation == "begin":
                journal.begin(spec.planned_units[0])
            elif operation == "retry":
                journal.retry(attempt.unit, reason="explicit retry")
            elif operation == "seal":
                journal.seal(attempt.attempt_id, evidence)
            elif operation == "finish":
                journal.finish(attempt.attempt_id, "completed")
            else:
                journal.verify_completed()
        with sqlite3.connect(root / "full_fit.sqlite3") as connection:
            assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == event_count
        if operation in ("begin", "verify_completed"):
            assert not (root / "attempts").exists()


def observe_durability(monkeypatch, fail_directory=None, connection=None):
    """Observe real fsync calls and real SQLite statements at their boundaries."""
    observed = []
    original_fsync = contracts.os.fsync
    original_connect = contracts.sqlite3.connect

    def fsync(fd):
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if path == fail_directory:
            raise OSError("directory fsync failed")
        original_fsync(fd)
        observed.append(("fsync", path))

    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connection.set_trace_callback(lambda statement: observed.append(("sql", statement)))
        return connection

    monkeypatch.setattr(contracts.os, "fsync", fsync)
    monkeypatch.setattr(contracts.sqlite3, "connect", connect)
    if connection is not None:
        connection.set_trace_callback(lambda statement: observed.append(("sql", statement)))
    return observed


def assert_synced_before(observed, directories, sql_fragment):
    persistence = next(index for index, event in enumerate(observed) if event[0] == "sql" and sql_fragment in event[1])
    synced = [event[1] for event in observed[:persistence] if event[0] == "fsync"]
    assert all(directory in synced for directory in directories), f"missing durable directories before {sql_fragment}: {synced}"
    assert [synced.index(directory) for directory in directories] == sorted(synced.index(directory) for directory in directories)


def test_new_run_ancestry_is_synced_before_initial_binding(monkeypatch, tmp_path):
    spec = make_spec(tmp_path)
    root = tmp_path / "new" / "campaign" / "run"
    observed = observe_durability(monkeypatch)
    with api("FullFitJournal")(root, spec, source_root=tmp_path):
        assert_synced_before(observed, (root, root.parent, root.parent.parent, tmp_path), "INSERT INTO binding")


def test_attempt_directory_chain_is_synced_before_begin_charge(monkeypatch, tmp_path):
    spec = make_spec(tmp_path)
    root = tmp_path / "run"
    with api("FullFitJournal")(root, spec, source_root=tmp_path) as journal:
        observed = observe_durability(monkeypatch, connection=journal._conn)
        attempt = journal.begin(spec.planned_units[0])
        directory = root / attempt.artifact_directory
        assert_synced_before(observed, (directory, directory.parent, root), '"kind":"begin"')


def test_nested_evidence_directory_chain_precedes_seal_persistence(monkeypatch, tmp_path):
    spec = make_spec(tmp_path)
    root = tmp_path / "run"
    with api("FullFitJournal")(root, spec, source_root=tmp_path) as journal:
        attempt = journal.begin(spec.planned_units[0])
        directory = root / attempt.artifact_directory
        path = directory / "artifacts" / "nested" / "proof"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"proof")
        observed = observe_durability(monkeypatch, connection=journal._conn)
        journal.seal(attempt.attempt_id, (api("ArtifactEvidence")(str(path.relative_to(root)), hashlib.sha256(b"proof").hexdigest()),))
        assert_synced_before(observed, (path, path.parent, path.parent.parent, directory, directory.parent, root), '"kind":"seal"')


@pytest.mark.parametrize("failure_location", ("nested", "attempts"))
def test_directory_fsync_failure_does_not_commit_seal(monkeypatch, tmp_path, failure_location):
    spec = make_spec(tmp_path)
    root = tmp_path / "run"
    with api("FullFitJournal")(root, spec, source_root=tmp_path) as journal:
        attempt = journal.begin(spec.planned_units[0])
        directory = root / attempt.artifact_directory
        path = directory / "nested" / "proof"
        path.parent.mkdir()
        path.write_bytes(b"proof")
        failed_directory = path.parent if failure_location == "nested" else directory.parent
        observed = observe_durability(monkeypatch, failed_directory, journal._conn)
        with pytest.raises(OSError, match="directory fsync failed"):
            journal.seal(attempt.attempt_id, (api("ArtifactEvidence")(str(path.relative_to(root)), hashlib.sha256(b"proof").hexdigest()),))
        assert not any(event[0] == "sql" and '"kind":"seal"' in event[1] for event in observed)
        record, = journal.attempts()
        assert record.status == "running" and record.evidence == ()
