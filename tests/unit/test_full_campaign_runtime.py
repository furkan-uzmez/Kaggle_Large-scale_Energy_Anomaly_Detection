"""Sequential runtime isolation, cancellation and terminal-evidence tests."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from full_campaign_workers import (
    fail_worker,
    sleep_forever,
    spawn_child_and_exit_zero,
    spawn_child_and_fail,
    spawn_child_and_sleep,
    spawn_sigterm_ignoring_child_and_exit_zero,
    write_nothing,
    write_oof_original,
    write_result,
    write_stderr_and_result,
)
from project.experiments import full_campaign_runtime as runtime


def _is_pid_dead(pid: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    try:
        os.kill(pid, 0)
        return False
    except ProcessLookupError:
        return True


def test_outer_supervisor_death_kills_owned_worker_group(tmp_path):
    program = """
import sys
from project.experiments.full_campaign_runtime import run_work_unit
run_work_unit('parent-death', 'full_campaign_workers:spawn_child_and_sleep', sys.argv[1])
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(p) for p in sys.path if p))
    parent = subprocess.Popen([sys.executable, "-c", program, str(tmp_path)], env=env)
    identities = {}
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            _, children = runtime._observe(parent.pid)
            identities = {pid: runtime._process_identity(pid) for pid in children}
            if len(identities) >= 2 and (tmp_path / "child.pid").exists():
                break
            time.sleep(0.05)
        assert len(identities) >= 2, "real worker and child must start"
        assert int((tmp_path / "child.pid").read_text()) in identities
        parent.kill()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and any(runtime._process_identity(pid) == identity for pid, identity in identities.items()):
            time.sleep(0.05)
        assert all(runtime._process_identity(pid) != identity for pid, identity in identities.items())
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        for pid, identity in identities.items():
            if identity is not None and runtime._process_identity(pid) == identity:
                os.kill(pid, 9)


def test_timeout_free_unit_requires_parent_validation_despite_real_artifacts(tmp_path):
    outcome = runtime.run_work_unit(
        "exp/arm/r0/f0/model",
        write_result,
        tmp_path,
        expected_artifacts=("result.json",),
    )
    assert outcome.status == "awaiting_validation"
    assert outcome.artifacts and outcome.artifacts[0].path == "result.json"
    assert outcome.observation["rss_limit_enforced"] is False
    assert outcome.observation["wall_limit_enforced"] is False
    assert outcome.observation["informed_opt_out"] is True
    assert (
        runtime.verify_resume(tmp_path, "exp/arm/r0/f0/model", ("result.json",))[0]
        is False
    )


def test_exit_zero_without_expected_artifact_is_not_completed(tmp_path):
    outcome = runtime.run_work_unit(
        "missing/artifact", write_nothing, tmp_path, expected_artifacts=("needed.json",)
    )
    assert outcome.status == "awaiting_validation"
    assert "needed.json" in (outcome.reason or "")
    assert outcome.artifacts == ()
    completed, reason = runtime.verify_resume(
        tmp_path, "missing/artifact", ("needed.json",)
    )
    assert completed is False and reason is not None


def test_nonzero_exit_is_failed(tmp_path):
    outcome = runtime.run_work_unit(
        "nonzero", fail_worker, tmp_path, expected_artifacts=()
    )
    assert outcome.status == "failed"
    assert outcome.exit_code not in (None, 0)


def test_cancellation_terminates_and_marks_cancelled(tmp_path):
    calls = {"n": 0}

    def cancel_check():
        calls["n"] += 1
        return calls["n"] >= 3

    outcome = runtime.run_work_unit(
        "cancel",
        sleep_forever,
        tmp_path,
        cancel_check=cancel_check,
        expected_artifacts=(),
    )
    assert outcome.status == "cancelled"
    completed, _ = runtime.verify_resume(tmp_path, "cancel", ())
    assert completed is False


def test_forged_completion_file_is_rejected(tmp_path):
    unit = "forged"
    completion = tmp_path / f"completion_{runtime._slug(unit)}.json"
    completion.write_text('{"unit": "forged", "status": "completed", "artifacts": []}')
    ok, reason = runtime.verify_resume(tmp_path, unit, ("evidence.json",))
    assert ok is False and reason is not None


def test_changed_artifact_hash_invalidates_resume(tmp_path):
    outcome = runtime.run_work_unit(
        "hash-check", write_oof_original, tmp_path, expected_artifacts=("oof.parquet",)
    )
    # Exercise the low-level transport receipt verifier, not a scientific grant.
    # Runtime no longer fabricates this receipt from exit zero.
    runtime.write_completion(tmp_path, "hash-check", "completed", outcome.artifacts, outcome.observation)
    (tmp_path / "oof.parquet").write_bytes(b"tampered")
    ok, reason = runtime.verify_resume(tmp_path, "hash-check", ("oof.parquet",))
    assert ok is False and "hash" in (reason or "")


def test_second_campaign_lock_is_rejected(tmp_path):
    first = runtime.CampaignLock(tmp_path)
    first.__enter__()
    try:
        with pytest.raises(runtime.CampaignLockError):
            runtime.CampaignLock(tmp_path).__enter__()
    finally:
        first.__exit__(None, None, None)
    with runtime.CampaignLock(tmp_path):
        pass


def test_leader_exit_zero_cleans_surviving_child(tmp_path):
    outcome = runtime.run_work_unit(
        "leader-zero-child",
        spawn_child_and_exit_zero,
        tmp_path,
        expected_artifacts=("result.json",),
    )
    assert outcome.status == "awaiting_validation"
    child_pid = int((tmp_path / "child.pid").read_text())
    assert _is_pid_dead(child_pid) is True


def test_leader_nonzero_exit_cleans_surviving_child(tmp_path):
    outcome = runtime.run_work_unit(
        "leader-fail-child",
        spawn_child_and_fail,
        tmp_path,
        expected_artifacts=(),
    )
    assert outcome.status == "failed"
    child_pid = int((tmp_path / "child.pid").read_text())
    assert _is_pid_dead(child_pid) is True


def test_cancellation_cleans_surviving_child(tmp_path):
    calls = {"n": 0}

    def cancel_check():
        calls["n"] += 1
        return calls["n"] >= 3

    outcome = runtime.run_work_unit(
        "cancel-child",
        spawn_child_and_sleep,
        tmp_path,
        cancel_check=cancel_check,
        expected_artifacts=(),
    )
    assert outcome.status == "cancelled"
    child_pid = int((tmp_path / "child.pid").read_text())
    assert _is_pid_dead(child_pid) is True


def test_terminate_does_not_stop_waiting_when_leader_exits_before_sigterm_ignoring_child(
    tmp_path,
):
    outcome = runtime.run_work_unit(
        "sigterm-ignoring-child",
        spawn_sigterm_ignoring_child_and_exit_zero,
        tmp_path,
        expected_artifacts=("result.json",),
    )
    assert outcome.status == "awaiting_validation"
    child_pid = int((tmp_path / "child.pid").read_text())
    assert _is_pid_dead(child_pid) is True


def test_cleanup_failure_blocks_completed_outcome(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_terminate", lambda *args, **kwargs: False)
    outcome = runtime.run_work_unit(
        "cleanup-fail",
        write_result,
        tmp_path,
        expected_artifacts=("result.json",),
    )
    assert outcome.status == "failed"
    assert "cleanup" in (outcome.reason or "").lower()
    ok, _ = runtime.verify_resume(tmp_path, "cleanup-fail", ("result.json",))
    assert ok is False


def test_lock_crash_release(tmp_path):
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from full_campaign_workers import hold_lock_and_signal; import sys; hold_lock_and_signal('unit', sys.argv[1])",
            str(tmp_path),
        ],
        env=dict(
            os.environ,
            PYTHONPATH=f".:src:tests:{os.environ.get('PYTHONPATH', '')}",
        ),
    )
    signal_file = tmp_path / "locked.signal"
    deadline = time.monotonic() + 5.0
    while not signal_file.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert signal_file.is_file()

    with pytest.raises(runtime.CampaignLockError):
        runtime.CampaignLock(tmp_path).__enter__()

    proc.kill()
    proc.wait()

    with runtime.CampaignLock(tmp_path):
        pass


def test_shared_project_campaign_lock_helper(tmp_path):
    lock_file = tmp_path / "campaign_project.lock"
    with (
        runtime.shared_project_campaign_lock(lock_file),
        pytest.raises(runtime.CampaignLockError),
        runtime.shared_project_campaign_lock(lock_file),
    ):
        pass
    with runtime.shared_project_campaign_lock(lock_file):
        pass


def test_stderr_persisted(tmp_path):
    outcome = runtime.run_work_unit(
        "unit-stderr",
        write_stderr_and_result,
        tmp_path,
        expected_artifacts=("result.json",),
    )
    assert outcome.status == "awaiting_validation"
    stderr_path = Path(
        outcome.observation.get("stderr_path", tmp_path / "stderr.log")
    )
    assert stderr_path.is_file()
    assert "worker stderr diagnostic log" in stderr_path.read_text()


def test_empty_expected_artifact_set_is_rejected(tmp_path):
    outcome = runtime.run_work_unit(
        "empty-expected",
        write_result,
        tmp_path,
        expected_artifacts=(),
    )
    assert outcome.status != "completed"
    assert "empty" in (outcome.reason or "").lower()
    ok, reason = runtime.verify_resume(
        tmp_path, "empty-expected", expected_artifacts=()
    )
    assert ok is False
    assert "empty" in (reason or "").lower()


def test_worker_forged_completion_rejection(tmp_path):
    completion = tmp_path / f"completion_{runtime._slug('forged-unit')}.json"
    completion.write_text(
        '{"unit": "forged-unit", "status": "completed", "artifacts": [{"path": "result.json", "sha256": "fake"}]}'
    )
    (tmp_path / "result.json").write_text('{"ok": true}')
    ok, reason = runtime.verify_resume(
        tmp_path,
        "forged-unit",
        ("result.json",),
        supervisor_token="parent-secret-token",
    )
    assert ok is False
    assert reason is not None


def test_stale_or_mutated_resume_binding_rejected(tmp_path):
    outcome = runtime.run_work_unit(
        "bound-unit",
        write_result,
        tmp_path,
        expected_artifacts=("result.json",),
        attempt_id="attempt-001",
        spec_binding="spec-v1-hash",
        environment_binding={"py": "3.13"},
        supervisor_token="sup-secret",
    )
    assert outcome.status == "awaiting_validation"
    # Explicit fixture receipt keeps attempt/spec/environment/hash rejection
    # coverage. A successful low-level check is not admission or resume authority.
    runtime.write_completion(tmp_path, "bound-unit", "completed", outcome.artifacts,
                             outcome.observation, attempt_id="attempt-001",
                             spec_binding="spec-v1-hash", environment_binding={"py": "3.13"},
                             supervisor_token="sup-secret")

    ok, _ = runtime.verify_resume(
        tmp_path,
        "bound-unit",
        ("result.json",),
        attempt_id="attempt-001",
        spec_binding="spec-v1-hash",
        environment_binding={"py": "3.13"},
        supervisor_token="sup-secret",
    )
    assert ok is True

    ok, reason = runtime.verify_resume(
        tmp_path,
        "bound-unit",
        ("result.json",),
        attempt_id="attempt-002",
        spec_binding="spec-v1-hash",
        supervisor_token="sup-secret",
    )
    assert ok is False and "attempt" in (reason or "").lower()

    ok, reason = runtime.verify_resume(
        tmp_path,
        "bound-unit",
        ("result.json",),
        attempt_id="attempt-001",
        spec_binding="spec-v2-hash",
        supervisor_token="sup-secret",
    )
    assert ok is False and "spec" in (reason or "").lower()

    ok, reason = runtime.verify_resume(
        tmp_path,
        "bound-unit",
        ("result.json", "extra.json"),
        attempt_id="attempt-001",
        spec_binding="spec-v1-hash",
        supervisor_token="sup-secret",
    )
    assert ok is False and (
        "inventory" in (reason or "").lower() or "missing" in (reason or "").lower()
    )


def test_journal_attempt_terminalized_on_failure_and_cancellation(tmp_path):
    class MockJournal:
        def __init__(self):
            self.calls = []

        def seal(self, attempt_id, artifacts):
            self.calls.append(("seal", attempt_id, artifacts))

        def finish(self, attempt_id, status, *, reason=None):
            self.calls.append(("finish", attempt_id, status, reason))

    journal1 = MockJournal()
    outcome_fail = runtime.run_work_unit(
        "unit-fail",
        fail_worker,
        tmp_path / "fail",
        expected_artifacts=("res.json",),
        attempt_id="attempt-fail-1",
        journal=journal1,
    )
    assert outcome_fail.status == "failed"
    assert any(
        c[0] == "finish" and c[1] == "attempt-fail-1" and c[2] == "failed"
        for c in journal1.calls
    )

    journal2 = MockJournal()
    outcome_cancel = runtime.run_work_unit(
        "unit-cancel",
        sleep_forever,
        tmp_path / "cancel",
        expected_artifacts=("res.json",),
        attempt_id="attempt-cancel-1",
        journal=journal2,
        cancel_check=lambda: True,
    )
    assert outcome_cancel.status == "cancelled"
    assert any(
        c[0] == "finish" and c[1] == "attempt-cancel-1" and c[2] == "cancelled"
        for c in journal2.calls
    )


def test_streaming_hash(tmp_path):
    large_file = tmp_path / "stream_test.bin"
    chunk = b"A" * 1024 * 64
    with large_file.open("wb") as f:
        for _ in range(16):
            f.write(chunk)
    expected_sha = hashlib.sha256(large_file.read_bytes()).hexdigest()
    assert runtime.stream_file_hash(large_file) == expected_sha
def test_cleanup_never_signals_reused_process_identities(monkeypatch):
    from types import SimpleNamespace

    from project.experiments import full_campaign_runtime as runtime

    process = SimpleNamespace(pid=987001, poll=lambda: 0,
                              _full_process_identities={987001: (1, 987001, 987001),
                                                        987002: (2, 987001, 987001)})
    monkeypatch.setattr(runtime, "_find_live_targets", lambda *args: {987001, 987002})
    monkeypatch.setattr(runtime, "_process_identity", lambda pid: (999, 987001, 987001), raising=False)
    signals = []
    monkeypatch.setattr(runtime.os, "kill", lambda *args: signals.append(args))
    monkeypatch.setattr(runtime.os, "killpg", lambda *args: signals.append(args))
    assert runtime._terminate(process, {987002}, grace_seconds=0) is False
    assert signals == [], "start-time mismatch must never authorize signalling a reused PID/group"
