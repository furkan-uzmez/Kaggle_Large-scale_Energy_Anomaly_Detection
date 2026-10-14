"""Real fixture interruption/recovery; never repository datasets."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from full_campaign_fixtures import write_admission_fixture
from project.experiments import full_candidate_runner as runner
from project.experiments.full_campaign_admission import FullCampaignIntent
from project.experiments.full_campaign_capabilities import (
    resolve_full_campaign_capability,
)
from project.experiments.full_campaign_preflight import preflight_full_campaign
from project.experiments.full_run_contracts import FullFitJournal, _file_hash


@pytest.fixture(scope="module")
def interrupted_campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("interrupted-resume")
    fixture = write_admission_fixture(root / "source")
    admitted = preflight_full_campaign(fixture.root, FullCampaignIntent(
        fixture.spec, True, resource_opt_out_record="user: fixture resume only"))
    run = root / "run"
    original = runner.run_work_unit
    calls = []

    def interrupt_second(*args, **kwargs):
        calls.append(kwargs["unit"])
        if len(calls) == 2:
            callback = kwargs["poll_callback"]

            def interrupt_after_debit():
                callback()
                if (Path(kwargs["work_dir"]) / "ack-0001.json").exists():
                    raise KeyboardInterrupt("fixture midworker interruption")

            kwargs["poll_callback"] = interrupt_after_debit
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "run_work_unit", interrupt_second)
        with pytest.raises(KeyboardInterrupt):
            runner.run_full_campaign(fixture.spec, fixture.root, run, data_preflight=admitted,
                                     capability=resolve_full_campaign_capability(fixture.root))
    with FullFitJournal.open_readonly(run / "journal", fixture.spec, source_root=fixture.root) as journal:
        attempts = journal.attempts()
    assert len(attempts) == 2
    assert attempts[0].status == "completed"
    assert attempts[1].status == "interrupted"
    return fixture, admitted, run, attempts


def test_resume_requires_reason_before_any_writes(interrupted_campaign):
    fixture, admitted, run, _ = interrupted_campaign
    before = {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="explicit retry reason"):
        runner.resume_full_campaign(fixture.spec, fixture.root, run, data_preflight=admitted,
                                    capability=resolve_full_campaign_capability(fixture.root))
    assert {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()} == before


def test_corrupt_completed_shard_blocks_resume_before_retry(interrupted_campaign):
    fixture, admitted, run, attempts = interrupted_campaign
    shard = run / "journal" / attempts[0].artifact_directory / "shard.parquet"
    original = shard.read_bytes()
    try:
        shard.write_bytes(b"corrupted authenticated completed shard")
        with pytest.raises(ValueError):
            runner.resume_full_campaign(fixture.spec, fixture.root, run, data_preflight=admitted,
                                        capability=resolve_full_campaign_capability(fixture.root),
                                        retry_reason="user: retry fixture")
    finally:
        shard.write_bytes(original)
    with FullFitJournal.open_readonly(run / "journal", fixture.spec, source_root=fixture.root) as journal:
        assert len(journal.attempts()) == 2


def test_changed_snapshot_refuses_resume_before_mutation(interrupted_campaign):
    from dataclasses import replace

    fixture, _, run, _ = interrupted_campaign
    spec = replace(fixture.spec, run_id="different-resume-run")
    admitted = preflight_full_campaign(fixture.root, FullCampaignIntent(
        spec, True, resource_opt_out_record="user: fixture resume only"))
    before = {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="snapshot"):
        runner.resume_full_campaign(spec, fixture.root, run, data_preflight=admitted,
                                    capability=resolve_full_campaign_capability(fixture.root), retry_reason="user: explicit")
    assert {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()} == before


def test_alive_owned_worker_refuses_recovery_without_signalling(interrupted_campaign):
    from project.experiments.full_campaign_evidence import (
        sign_parent_record,
        verify_parent_record,
    )
    from project.experiments.full_campaign_runtime import _process_identity

    fixture, admitted, run, _ = interrupted_campaign
    attempt = max((run / "attempts").iterdir())
    path = attempt / "worker-launch.json"
    original = path.read_bytes()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        launch = verify_parent_record(fixture.root, path, run)
        launch["identity_record"]["pid"] = process.pid
        launch["identity_record"]["identity"] = list(_process_identity(process.pid))
        path.write_text(json.dumps(sign_parent_record(fixture.root, launch)))
        before = {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()}
        with pytest.raises(ValueError, match="still alive"):
            runner.resume_full_campaign(fixture.spec, fixture.root, run, data_preflight=admitted,
                                        capability=resolve_full_campaign_capability(fixture.root), retry_reason="user: explicit")
        assert process.poll() is None
        assert {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()} == before
    finally:
        path.write_bytes(original)
        process.kill()
        process.wait(timeout=5)


def test_fresh_process_resume_reuses_completed_unit_and_debits_exact_retry(interrupted_campaign):
    fixture, _, run, attempts = interrupted_campaign
    shard = run / "journal" / attempts[0].artifact_directory / "shard.parquet"
    before = _file_hash(shard)
    program = """
import json,sys
from pathlib import Path
sys.path.insert(0, 'src')
from project.experiments.full_snapshot_codec import decode_snapshot
from project.experiments.full_campaign_admission import FullCampaignIntent
from project.experiments.full_campaign_preflight import preflight_full_campaign
from project.experiments.full_campaign_capabilities import resolve_full_campaign_capability
from project.experiments.full_candidate_runner import resume_full_campaign
source,run=map(Path,sys.argv[1:])
spec=decode_snapshot(json.loads((run/'spec.json').read_text()))
admitted=preflight_full_campaign(source,FullCampaignIntent(spec,True,resource_opt_out_record='user: fixture resume only'))
result=resume_full_campaign(spec,source,run,data_preflight=admitted,capability=resolve_full_campaign_capability(source),retry_reason='user: explicit interrupted fixture retry')
assert result['experiments'][0]['status']=='completed'
print(result['status'])
"""
    result = subprocess.run([sys.executable, "-B", "-c", program, str(fixture.root), str(run)],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
                            timeout=300, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "finished_with_blockers"
    assert _file_hash(shard) == before
    with FullFitJournal.open_readonly(run / "journal", fixture.spec, source_root=fixture.root) as journal:
        recovered = journal.attempts()
    assert len(recovered) == 25  # 24 planned fits plus exactly one explicit retry.
    assert sum(a.unit == attempts[0].unit for a in recovered) == 1
    retries = [a for a in recovered if a.unit == attempts[1].unit]
    assert len(retries) == 2
    assert retries[-1].artifact_directory != retries[0].artifact_directory
    assert len(list((run / "attempts").iterdir())) == 25
    receipt = json.loads((run / "completion/receipt.json").read_text())
    assert receipt["verified_model_fits"] == 25


def test_completion_publication_repair_does_not_refit(interrupted_campaign, monkeypatch):
    fixture, admitted, run, _ = interrupted_campaign
    # Simulate loss of the commit point, retaining all staged bytes. This moves
    # only the completion directory created by this fixture, not user work.
    (run / "completion").rename(run / ".completion-stage-fixture")
    with FullFitJournal.open_readonly(run / "journal", fixture.spec, source_root=fixture.root) as journal:
        before = journal.attempts()
    monkeypatch.setattr(runner, "run_work_unit", lambda **kwargs: pytest.fail("publication repair must not fit"))
    result = runner.resume_full_campaign(fixture.spec, fixture.root, run, data_preflight=admitted,
                                        capability=resolve_full_campaign_capability(fixture.root))
    assert result["experiments"][0]["status"] == "completed"
    with FullFitJournal.open_readonly(run / "journal", fixture.spec, source_root=fixture.root) as journal:
        assert journal.attempts() == before
    assert (run / ".completion-stage-fixture").is_dir()


def test_abrupt_parent_death_restarts_from_signed_running_attempt(tmp_path):
    import time

    from project.experiments.full_campaign_evidence import verify_interrupted_exit
    from project.experiments.full_snapshot_codec import encode_snapshot

    fixture = write_admission_fixture(tmp_path / "source")
    descriptor = tmp_path / "driver-spec.json"
    descriptor.write_text(json.dumps(encode_snapshot(fixture.spec)))
    run = tmp_path / "run"
    program = """
import json,os,signal,sys
from pathlib import Path
sys.path[:0]=['src','tests']
from project.experiments import full_candidate_runner as runner
from project.experiments.full_snapshot_codec import decode_snapshot
from project.experiments.full_campaign_admission import FullCampaignIntent
from project.experiments.full_campaign_preflight import preflight_full_campaign
from project.experiments.full_campaign_capabilities import resolve_full_campaign_capability
source,run,descriptor=map(Path,sys.argv[1:])
spec=decode_snapshot(json.loads(descriptor.read_text()))
admitted=preflight_full_campaign(source,FullCampaignIntent(spec,True,resource_opt_out_record='user: abrupt fixture only'))
original=runner.run_work_unit
calls=[]
def kill_second(**kwargs):
    calls.append(kwargs['unit'])
    if len(calls)==2:
        original_callback=kwargs['poll_callback']
        def kill_after_debit():
            original_callback()
            if (Path(kwargs['work_dir'])/'ack-0001.json').exists():
                os.kill(os.getpid(),signal.SIGKILL)
        kwargs['poll_callback']=kill_after_debit
    return original(**kwargs)
runner.run_work_unit=kill_second
runner.run_full_campaign(spec,source,run,data_preflight=admitted,capability=resolve_full_campaign_capability(source))
"""
    result = subprocess.run([sys.executable, "-B", "-c", program, str(fixture.root), str(run), str(descriptor)],
                            cwd=Path(__file__).resolve().parents[2], env=dict(os.environ), capture_output=True,
                            text=True, timeout=60, check=False)
    assert result.returncode == -9, result.stderr
    with FullFitJournal.open_readonly(run / "journal", fixture.spec, source_root=fixture.root) as journal:
        before = journal.attempts()
    assert [attempt.status for attempt in before] == ["completed", "running"]
    deadline = time.monotonic() + 5
    while True:
        try:
            verify_interrupted_exit(fixture.root, run, fixture.spec, before[-1])
            break
        except ValueError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
    admitted = preflight_full_campaign(fixture.root, FullCampaignIntent(
        fixture.spec, True, resource_opt_out_record="user: abrupt fixture only"))
    result = runner.resume_full_campaign(fixture.spec, fixture.root, run, data_preflight=admitted,
                                        capability=resolve_full_campaign_capability(fixture.root),
                                        retry_reason="user: restart after proven parent death")
    assert result["experiments"][0]["status"] == "completed"
    with FullFitJournal.open_readonly(run / "journal", fixture.spec, source_root=fixture.root) as journal:
        after = journal.attempts()
    assert len(after) == 25
    assert after[1].status == "interrupted"
    assert len([attempt for attempt in after if attempt.unit == before[0].unit]) == 1
