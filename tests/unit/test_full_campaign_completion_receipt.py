"""Independent ordinary evidence verification; no repository dataset access."""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from full_campaign_fixtures import write_admission_fixture
from project.experiments import full_campaign_evidence as evidence
from project.experiments.full_campaign_admission import FullCampaignIntent
from project.experiments.full_campaign_capabilities import (
    resolve_full_campaign_capability,
)
from project.experiments.full_campaign_preflight import preflight_full_campaign
from project.experiments.full_candidate_runner import run_full_campaign


@pytest.fixture(scope="module")
def genuine_campaign(tmp_path_factory):
    directory = tmp_path_factory.mktemp("completion-receipt")
    fixture = write_admission_fixture(directory / "source")
    admitted = preflight_full_campaign(fixture.root, FullCampaignIntent(
        fixture.spec, True, resource_opt_out_record="user: fixture-only completion acceptance"))
    run = directory / "run"
    run_full_campaign(fixture.spec, fixture.root, run, data_preflight=admitted,
                      capability=resolve_full_campaign_capability(fixture.root))
    return fixture, admitted, run


@pytest.fixture(scope="module")
def publication_transport(genuine_campaign):
    fixture, admitted, previous = genuine_campaign
    run = previous.parent / "publication-run"
    run_full_campaign(fixture.spec, fixture.root, run, data_preflight=admitted,
                      capability=resolve_full_campaign_capability(fixture.root), publish_completion=False)
    return fixture, admitted, run


def test_independent_receipt_verifies_all_ordinary_evidence_without_writes(genuine_campaign):
    fixture, admitted, run = genuine_campaign
    verifier = getattr(evidence, "verify_ordinary_campaign_receipt", None)
    assert callable(verifier), "independent reopening verifier is required"
    before = sorted(str(path) for path in run.rglob("*"))
    receipt = verifier(fixture.root, run, admitted)
    assert receipt["status"] == "independently_verified_ordinary_receipt"
    assert receipt["verified_model_fits"] == 24
    assert receipt["n03_eligible"] is False
    assert len(receipt["inventory"]) == 21
    assert sorted(str(path) for path in run.rglob("*")) == before


def test_genuine_parent_completion_is_published_atomically(genuine_campaign):
    fixture, admitted, run = genuine_campaign
    issuer = getattr(evidence, "issue_ordinary_campaign_completion", None)
    assert callable(issuer), "parent-only completion gate must be explicit"
    destination = run / "completion"
    result = evidence.verify_ordinary_campaign_completion(fixture.root, run, admitted)
    assert result["status"] == "finished_with_blockers"
    assert result["experiments"][0]["status"] == "completed"
    assert len(result["inventory"]) == 21
    assert len([row for row in result["inventory"] if row["status"] == "completed"]) == 1
    assert result["comparative_completion_available"] is True
    assert (destination / "manifest.json").is_file()
    assert json.loads((run / "run_status.json").read_text())["status"] == "awaiting_validation"


@pytest.mark.parametrize("mutation", ["oof_label", "oof_probability", "shard", "supervisor", "encoding", "source", "journal_unit", "inventory"])
def test_independent_verifier_rejects_actual_evidence_corruption(genuine_campaign, mutation):
    import pyarrow as pa
    import pyarrow.parquet as pq

    fixture, admitted, run = genuine_campaign
    attempt = next((run / "journal/attempts").iterdir())
    if mutation.startswith("oof_"):
        path = run / "feat_missingness_indicators_oof.parquet"
    elif mutation == "shard":
        path = attempt / "shard.parquet"
    elif mutation == "supervisor":
        path = attempt / "supervisor.json"
    elif mutation == "encoding":
        path = fixture.root / "docs/evidence/year_built_encoding_v1.json"
    elif mutation == "source":
        path = fixture.root / "src/project/features/registry.py"
    elif mutation == "journal_unit":
        path = run / "journal/full_fit.sqlite3"
    else:
        path = run / "run_status.json"
    original = path.read_bytes()
    try:
        if mutation.startswith("oof_"):
            table = pq.read_table(path)
            column = "target" if mutation == "oof_label" else "prediction"
            values = table[column].to_numpy().copy()
            values[0] = 1 - values[0] if column == "target" else 0.987
            field = table.schema.field(column)
            table = table.set_column(table.schema.get_field_index(column), field, pa.array(values, type=field.type))
            pq.write_table(table, path)
        elif mutation == "journal_unit":
            with sqlite3.connect(path) as connection:
                last = connection.execute("SELECT MAX(sequence) FROM events").fetchone()[0]
                connection.execute("DELETE FROM events WHERE sequence=?", (last,))
                digest = connection.execute("SELECT digest FROM events WHERE sequence=?", (last - 1,)).fetchone()[0]
                connection.execute("UPDATE binding SET head=?, count=?", (digest, last - 1))
        elif mutation == "inventory":
            document = json.loads(original)
            document["inventory"][0]["status"] = "completed"
            path.write_text(json.dumps(document))
        else:
            path.write_bytes(b"corrupted owned fixture evidence")
        with pytest.raises((ValueError, OSError)):
            evidence.verify_ordinary_campaign_receipt(fixture.root, run, admitted)
        with pytest.raises((ValueError, OSError)):
            evidence.verify_ordinary_campaign_completion(fixture.root, run, admitted)
    finally:
        path.write_bytes(original)


def test_missing_supervisor_fails_without_recreating_evidence(genuine_campaign):
    fixture, admitted, run = genuine_campaign
    path = next((run / "journal/attempts").iterdir()) / "supervisor.json"
    original = path.read_bytes()
    try:
        path.unlink()
        with pytest.raises((ValueError, OSError)):
            evidence.verify_ordinary_campaign_receipt(fixture.root, run, admitted)
        with pytest.raises((ValueError, OSError)):
            evidence.verify_ordinary_campaign_completion(fixture.root, run, admitted)
        assert not path.exists()
    finally:
        path.write_bytes(original)


def test_forged_or_replayed_receipt_cannot_grant_completion(genuine_campaign):
    fixture, admitted, run = genuine_campaign
    receipt = evidence.verify_ordinary_campaign_receipt(fixture.root, run, admitted)
    for changed in ({**receipt, "seal": "truthy"}, {**receipt, "run_root": "/another/run"},
                    {**receipt, "n03_eligible": True}):
        with pytest.raises(ValueError, match="forged, replayed or mismatched"):
            evidence.verify_ordinary_campaign_receipt(fixture.root, run, admitted, expected_receipt=changed)


def test_receipt_can_be_reverified_in_fresh_process_with_fresh_admission(genuine_campaign, tmp_path):
    fixture, admitted, run = genuine_campaign
    receipt = evidence.verify_ordinary_campaign_receipt(fixture.root, run, admitted)
    saved = tmp_path / "receipt.json"
    saved.write_text(json.dumps(receipt))
    program = """
import json, sys
from pathlib import Path
sys.path.insert(0, 'src')
from project.experiments.full_snapshot_codec import decode_snapshot
from project.experiments.full_campaign_admission import FullCampaignIntent
from project.experiments.full_campaign_preflight import preflight_full_campaign
from project.experiments.full_campaign_evidence import verify_ordinary_campaign_receipt, verify_ordinary_campaign_completion
from project.experiments.full_candidate_runner import resume_full_campaign
from project.experiments.full_campaign_capabilities import resolve_full_campaign_capability
source, run, receipt = map(Path, sys.argv[1:])
spec = decode_snapshot(json.loads((run / 'spec.json').read_text()))
admitted = preflight_full_campaign(source, FullCampaignIntent(spec, True,
    resource_opt_out_record='user: fixture-only completion acceptance'))
result = verify_ordinary_campaign_receipt(source, run, admitted,
    expected_receipt=json.loads(receipt.read_text()))
assert result['verified_model_fits'] == 24 and result['n03_eligible'] is False
before = {str(p): p.read_bytes() for p in run.rglob('*') if p.is_file()}
completed = verify_ordinary_campaign_completion(source, run, admitted)
resumed = resume_full_campaign(spec, source, run, data_preflight=admitted,
    capability=resolve_full_campaign_capability(source))
assert resumed == completed
assert completed['status'] == 'finished_with_blockers'
assert completed['experiments'][0]['status'] == 'completed'
assert {str(p): p.read_bytes() for p in run.rglob('*') if p.is_file()} == before
print(result['status'])
"""
    result = subprocess.run([sys.executable, "-B", "-c", program, str(fixture.root), str(run), str(saved)],
                            cwd=Path(__file__).resolve().parents[2], env=dict(os.environ),
                            text=True, capture_output=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "independently_verified_ordinary_receipt"


def test_completion_gate_preserves_occupied_destination(genuine_campaign, tmp_path):
    fixture, admitted, run = genuine_campaign
    destination = run / "completion"
    before = {path.name: path.read_bytes() for path in destination.iterdir()}
    with pytest.raises(FileExistsError):
        evidence.issue_ordinary_campaign_completion(fixture.root, run, admitted, destination)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before


@pytest.mark.parametrize("filename", ["manifest.json", "run_status.json", "receipt.json", "admission.json"])
def test_signed_completion_bytes_cannot_be_changed(genuine_campaign, filename):
    fixture, admitted, run = genuine_campaign
    path = run / "completion" / filename
    original = path.read_bytes()
    try:
        document = json.loads(original)
        if filename == "admission.json":
            document["signature"] = "a" * 64
        else:
            document["caller_supplied_seal"] = True
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError):
            evidence.verify_ordinary_campaign_completion(fixture.root, run, admitted)
    finally:
        path.write_bytes(original)


def test_completion_admission_cannot_be_replayed_for_another_run(publication_transport, genuine_campaign):
    fixture, admitted, other = publication_transport
    _, _, original = genuine_campaign
    from dataclasses import replace

    from project.experiments.full_run_contracts import OrdinaryCompletionAdmission

    permit = OrdinaryCompletionAdmission(**json.loads((original / "completion/admission.json").read_text()))
    with pytest.raises(ValueError, match="forged, replayed or mismatched"):
        admitted.verify_bindings(fixture.root, phase="completion", completion_admission=replace(
            permit, run_root=str(other), destination=str(other / "completion")))
    assert not (other / "completion").exists()


@pytest.mark.parametrize("kind", ["oof", "shard", "journal", "supervisor", "receipt", "source", "encoding"])
def test_tampering_prevents_new_completion_publication(publication_transport, kind):
    fixture, admitted, run = publication_transport
    attempt = next((run / "journal/attempts").iterdir())
    paths = {"oof": run / "feat_missingness_indicators_oof.parquet",
             "shard": attempt / "shard.parquet", "journal": run / "journal/full_fit.sqlite3",
             "supervisor": attempt / "supervisor.json", "receipt": run / "run_status.json",
             "source": fixture.root / "src/project/features/registry.py",
             "encoding": fixture.root / "docs/evidence/year_built_encoding_v1.json"}
    path = paths[kind]
    original = path.read_bytes()
    try:
        path.write_bytes(b"corrupt fixture before parent issuance")
        with pytest.raises((ValueError, OSError, sqlite3.DatabaseError)):
            evidence.issue_ordinary_campaign_completion(fixture.root, run, admitted, run / "completion")
        assert not (run / "completion").exists()
    finally:
        path.write_bytes(original)


def test_interrupted_publication_has_no_visible_completed_state(publication_transport, monkeypatch):
    fixture, admitted, run = publication_transport

    def interrupt_before_commit(stage, destination):
        assert (stage / "manifest.json").is_file()
        raise KeyboardInterrupt("fixture publication interrupted")

    monkeypatch.setattr(evidence, "_publish_completion_directory", interrupt_before_commit)
    with pytest.raises(KeyboardInterrupt):
        evidence.issue_ordinary_campaign_completion(fixture.root, run, admitted, run / "completion")
    assert not (run / "completion").exists()
    assert list(run.glob(".completion-stage-*"))
    with pytest.raises(OSError):
        evidence.verify_ordinary_campaign_completion(fixture.root, run, admitted)


def test_publication_never_replaces_concurrent_destination(publication_transport, monkeypatch):
    fixture, admitted, run = publication_transport
    original = evidence._publish_completion_directory
    destination = run / "completion"

    def race_before_atomic_commit(stage, target):
        target.mkdir()
        original(stage, target)

    monkeypatch.setattr(evidence, "_publish_completion_directory", race_before_atomic_commit)
    try:
        with pytest.raises(FileExistsError):
            evidence.issue_ordinary_campaign_completion(fixture.root, run, admitted, destination)
        assert destination.is_dir()
        assert list(destination.iterdir()) == []
    finally:
        destination.rmdir()  # Only the empty destination created by this fixture race.


def test_corruption_during_staging_is_denied_before_commit(publication_transport, monkeypatch):
    fixture, admitted, run = publication_transport
    path = run / "feat_missingness_indicators_oof.parquet"
    original_bytes = path.read_bytes()
    original_sync = evidence._fsync_directory

    def corrupt_after_staged_files(directory):
        original_sync(directory)
        if directory.name.startswith(".completion-stage-"):
            path.write_bytes(b"corrupt before commit")

    monkeypatch.setattr(evidence, "_fsync_directory", corrupt_after_staged_files)
    try:
        with pytest.raises((ValueError, OSError)):
            evidence.issue_ordinary_campaign_completion(fixture.root, run, admitted, run / "completion")
        assert not (run / "completion").exists()
    finally:
        path.write_bytes(original_bytes)


def test_two_parent_publishers_have_exactly_one_immutable_winner(publication_transport):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    fixture, admitted, run = publication_transport
    barrier = Barrier(2)

    def publish():
        barrier.wait()
        try:
            return evidence.issue_ordinary_campaign_completion(fixture.root, run, admitted, run / "completion")
        except OSError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: publish(), range(2)))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert evidence.verify_ordinary_campaign_completion(fixture.root, run, admitted) == winners[0]
    assert len(winners[0]["experiments"]) == 1
    assert winners[0]["experiments"][0]["status"] == "completed"


def test_explicit_completed_resume_verifies_without_fits_or_writes(genuine_campaign, monkeypatch):
    from project.experiments import full_candidate_runner as runner

    fixture, admitted, run = genuine_campaign
    before = {str(path): path.read_bytes() for path in run.rglob("*") if path.is_file()}

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("verified completed resume must not fit")

    monkeypatch.setattr(runner, "run_work_unit", forbidden_fit)
    resume = getattr(runner, "resume_full_campaign", None)
    assert callable(resume), "explicit verified resume interface required"
    result = resume(fixture.spec, fixture.root, run, data_preflight=admitted,
                    capability=resolve_full_campaign_capability(fixture.root))
    assert result["experiments"][0]["status"] == "completed"
    assert result["status"] == "finished_with_blockers"
    assert {str(path): path.read_bytes() for path in run.rglob("*") if path.is_file()} == before


def test_resume_cannot_skip_corrupt_completed_oof(genuine_campaign):
    from project.experiments import full_candidate_runner as runner

    fixture, admitted, run = genuine_campaign
    path = run / "feat_missingness_indicators_oof.parquet"
    original = path.read_bytes()
    try:
        path.write_bytes(b"corrupt completed OOF")
        with pytest.raises((ValueError, OSError)):
            runner.resume_full_campaign(fixture.spec, fixture.root, run, data_preflight=admitted,
                                        capability=resolve_full_campaign_capability(fixture.root))
    finally:
        path.write_bytes(original)


def test_incomplete_resume_never_creates_unaccounted_retry(genuine_campaign, tmp_path):
    from project.experiments.full_candidate_runner import resume_full_campaign

    fixture, admitted, _run = genuine_campaign
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    marker = incomplete / "interrupted-owner-evidence"
    marker.write_text("retain")
    with pytest.raises(ValueError, match="proven process exit and explicit journal retry required"):
        resume_full_campaign(fixture.spec, fixture.root, incomplete, data_preflight=admitted,
                             capability=resolve_full_campaign_capability(fixture.root),
                             retry_reason="user explicitly requests retry")
    assert list(incomplete.iterdir()) == [marker]
    assert marker.read_text() == "retain"
