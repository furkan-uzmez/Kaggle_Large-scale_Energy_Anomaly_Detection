"""Bound outer HPO transport: one tiny real subprocess, never full data."""

import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import MethodType

import numpy as np
import pyarrow.parquet as pq
import pytest

from full_campaign_fixtures import write_admission_fixture
from project.experiments import full_hpo_campaign as campaign
from project.experiments import full_development_readiness as readiness
from project.experiments.full_development_audits import AuditPlan, audit_full_development, write_audit_evidence
from project.experiments import full_candidate_io as io
from project.experiments import full_hpo_worker as backend
from project.experiments.full_campaign_preflight import current_environment
from project.experiments.full_campaign_evidence import sign_parent_record
from project.experiments.full_hpo_lifecycle import EXPERIMENT
from project.experiments.full_nested_tuning import full_hpo_study_spec
from project.experiments.full_run_contracts import FullFitJournal
from project.experiments.full_snapshot_codec import canonical_json
from project.experiments.nested_tuning import build_hgb_tuning_arm


def _authority(root, monkeypatch, *, nested_journal=False):
    fixture = write_admission_fixture(root / "source")
    snapshot = fixture.spec.candidates[0]
    snapshot = replace(snapshot, spec=replace(snapshot.spec, experiment_id=EXPERIMENT),
                       study=full_hpo_study_spec(), study_method="random_search")
    spec = replace(fixture.spec, candidates=(snapshot,), environment=current_environment())
    report = audit_full_development(fixture.root, spec, AuditPlan((3,), 42))
    digest = write_audit_evidence(fixture.root / readiness.EVIDENCE_PATH, report)
    # Confined fixture-only pin: both parent and real child verify this actual audit.
    # Production readiness source and its immutable audit are never modified.
    copied = fixture.root / "src/project/experiments/full_development_readiness.py"
    lines = copied.read_text().splitlines()
    lines = [f'EVIDENCE_SHA256 = "{digest}"' if line.startswith("EVIDENCE_SHA256 = ") else line for line in lines]
    copied.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(readiness, "_REFERENCE_ROOT", fixture.root)
    monkeypatch.setattr(readiness, "EVIDENCE_SHA256", digest)
    monkeypatch.syspath_prepend(str(fixture.root / "src"))
    run_root = root / "run"
    journal = FullFitJournal(run_root / "journal" if nested_journal else run_root, spec, source_root=fixture.root)
    index = io.build_development_index(fixture.root, spec,
        run_root / "dev_index.sqlite3" if nested_journal else journal.root / "development.sqlite3")
    return fixture.root, spec, journal, index


def test_worker_produces_outer_shards_before_losing_fresh_models(tmp_path, monkeypatch):
    module = campaign
    source, spec, journal, index = _authority(tmp_path, monkeypatch)
    try:
        context = module.HpoPartitionContext(source, journal.root, spec, index, 0, 0)
        directory, request, service = module.prepare_hpo_partition(context, journal, "fixture-key")
        calls = []
        def recording_factory(params):
            model = build_hgb_tuning_arm(params)
            def record_fit(self, X, y):
                calls.append(tuple(X.index))
                self.named_steps["model"].classes_ = np.array([0, 1])
                return self
            model.fit = MethodType(record_fit, model)
            model.predict_proba = MethodType(lambda self, X: np.full((len(X), 2), 0.5), model)
            return model
        result = backend.execute_hpo_request(directory, "fixture-key", service.lifecycle.handle,
                                             arm_factory=recording_factory)
        monkeypatch.setenv("FULL_HPO_WORKER_KEY", "fixture-key")
        monkeypatch.setattr(backend.os, "getppid", lambda: request["parent_pid"])
        monkeypatch.setattr(backend, "execute_hpo_request", lambda *_: result)
        backend.run(request["work_id"], directory)
        payload = json.loads((directory / "hpo-result.json").read_text())["payload"]
        assert len(calls) == journal.used_model_fits == 26
        assert set(payload["shards"]) == {"control", "challenger"}
        expected = sum(len(batch["target"]) for batch in io.iter_full_validation_batches(index, spec, 0, 0))
        for item in payload["shards"].values():
            assert pq.ParquetFile(directory / item["path"]).metadata.num_rows == expected
    finally:
        journal.close()


@pytest.fixture(scope="module")
def completed_partition(tmp_path_factory):
    module = campaign
    monkeypatch = pytest.MonkeyPatch()
    source, spec, journal, index = _authority(tmp_path_factory.mktemp("hpo-26-fit"), monkeypatch)
    context = module.HpoPartitionContext(source, journal.root, spec, index, 0, 0)
    try:
        result = module.run_hpo_partition(context, journal, "ephemeral-fixture-key")
        assert journal.used_model_fits == 26
        assert len(journal.verify_completed()) == 26
        journal.close()
        yield module, context, result
    finally:
        journal.close()
        monkeypatch.undo()


def test_real_dedicated_worker_26_fit_outer_partition(completed_partition):
    module, context, result = completed_partition
    assert set(result["shard_paths"]) == {"control", "challenger"}
    assert result["outcome"]["exit_code"] == 0
    assert result["request"]["kind"] == "full_hpo"
    assert result["result"]["kind"] == "hpo_result"
    verified = module.verify_hpo_partition(context, result["proof"])
    assert verified["shard_paths"] == result["shard_paths"]
    assert verified["candidates"][f"{EXPERIMENT}_challenger"]["fold_fit_config_sha256"]["0:0"]
    assert "ephemeral-fixture-key" not in canonical_json(result["proof"])


@pytest.mark.parametrize("mutation", ["selected_trial_id", "inner_score", "outer_target", "missing_outer"])
def test_independent_verification_rejects_tampered_partition(completed_partition, mutation):
    module, context, result = completed_partition
    proof_path = Path(result["proof"]["path"])
    original_proof = proof_path.read_bytes()
    envelope = json.loads(original_proof)
    if mutation in ("selected_trial_id", "inner_score"):
        path = Path(result["result"]["study_db_path"])
        original = path.read_bytes()
        try:
            with sqlite3.connect(path) as connection:
                if mutation == "selected_trial_id":
                    connection.execute("UPDATE study SET selected_trial_id = (selected_trial_id + 1) % 8")
                else:
                    connection.execute("UPDATE fold_score SET score = 0.123 WHERE trial_id=0 AND fold_index=0")
            # Bypass mere metadata hashing: independently sealed scores must still win.
            envelope["payload"]["study_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            proof_path.write_text(canonical_json(sign_parent_record(context.source_root, envelope["payload"])))
            proof = {"path": str(proof_path), "sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest()}
            with pytest.raises((ValueError, RuntimeError), match="study|score|winner|partition"):
                module.verify_hpo_partition(context, proof)
        finally:
            path.write_bytes(original)
            proof_path.write_bytes(original_proof)
    else:
        import pyarrow as pa
        path = Path(result["shard_paths"]["challenger"])
        original = path.read_bytes()
        try:
            if mutation == "missing_outer":
                path.unlink()
            else:
                table = pq.ParquetFile(path).read()
                position = table.schema.get_field_index("target")
                table = table.set_column(position, table.schema.field(position),
                                         pa.array([1 - value for value in table["target"].to_pylist()], type=pa.int64()))
                pq.write_table(table, path)
            with pytest.raises((ValueError, RuntimeError, OSError)):
                module.verify_hpo_partition(context, result["proof"])
        finally:
            path.write_bytes(original)


def test_bound_index_tamper_fails_before_debit(tmp_path, monkeypatch):
    module = campaign
    source, spec, journal, index = _authority(tmp_path, monkeypatch)
    try:
        with sqlite3.connect(index) as connection:
            connection.execute("UPDATE dev_row SET target=1-target WHERE row_index=0")
        context = module.HpoPartitionContext(source, journal.root, spec, index, 0, 0)
        with pytest.raises(ValueError, match="integrity|index"):
            module.run_hpo_partition(context, journal, "fixture-key")
        assert journal.used_model_fits == 0
    finally:
        journal.close()


def test_parent_terminalizes_only_failed_owned_attempt(tmp_path, monkeypatch):
    source, spec, journal, index = _authority(tmp_path, monkeypatch)
    context = campaign.HpoPartitionContext(source, journal.root, spec, index, 0, 0)
    def failing_runtime(*args, **kwargs):
        lifecycle = kwargs["poll_callback"].__self__.lifecycle
        lifecycle.handle(backend.sign_message({
            "version": 1, "kind": "hpo_event", "event": "acquire", "seq": 1,
            "identity": lifecycle.plan.default().payload(), "evidence": [], "reason": None,
            "attempt_id": None,
        }, "fixture-key"))
        raise RuntimeError("fixture crash before model construction")
    monkeypatch.setattr(campaign, "run_work_unit", failing_runtime)
    try:
        with pytest.raises(RuntimeError, match="fixture crash"):
            campaign.run_hpo_partition(context, journal, "fixture-key")
        assert journal.used_model_fits == 1
        assert len(journal.attempts()) == 1
        assert journal.attempts()[0].status == "failed"
        with pytest.raises(ValueError, match="resume|proof"):
            campaign.run_hpo_partition(context, journal, "fixture-key")
        assert journal.used_model_fits == 1
    finally:
        journal.close()


def test_verified_resume_does_not_refit(completed_partition):
    module, context, result = completed_partition
    with FullFitJournal(context.run_root, context.spec, source_root=context.source_root) as journal:
        with pytest.raises(ValueError, match="resume|proof"):
            module.run_hpo_partition(context, journal, "new-ephemeral-key")
        verified = module.verify_hpo_partition(context, result["proof"], journal=journal)
        assert verified["result"]["selected_trial_id"] == result["result"]["selected_trial_id"]
        assert journal.used_model_fits == 26


def test_nested_campaign_journal_executes_and_verifies_outer_hpo(tmp_path, monkeypatch):
    source, spec, journal, index = _authority(tmp_path, monkeypatch, nested_journal=True)
    context = campaign.HpoPartitionContext(source, journal.root.parent, spec, index, 0, 0)
    try:
        result = campaign.run_hpo_partition(context, journal, "nested-fixture-key")
        assert journal.used_model_fits == 26
        assert len(journal.verify_completed()) == 26
        journal.close()
        verified = campaign.verify_hpo_partition(context, result["proof"])
        assert verified["shard_paths"] == result["shard_paths"]
        assert set(verified["shard_paths"]) == {"control", "challenger"}
    finally:
        journal.close()


def test_nested_campaign_rejects_noncanonical_sibling_index_before_debit(tmp_path, monkeypatch):
    import shutil

    source, spec, journal, index = _authority(tmp_path, monkeypatch, nested_journal=True)
    sibling = journal.root.parent / "unrelated.sqlite3"
    shutil.copyfile(index, sibling)
    context = campaign.HpoPartitionContext(source, journal.root.parent, spec, sibling, 0, 0)
    try:
        with pytest.raises(ValueError, match="managed root"):
            campaign.prepare_hpo_partition(context, journal, "nested-fixture-key")
        assert journal.used_model_fits == 0
    finally:
        journal.close()
