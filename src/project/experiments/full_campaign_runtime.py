"""Sequential full-campaign runtime: process isolation and terminal evidence.

Each work unit runs as a fresh subprocess in its own session/process group. The
parent observes optional RSS and elapsed telemetry but, under the user's explicit
resource opt-out, never invents or enforces a RAM/CPU/wall cap and never forges an
AutoGluon lease. Exit code zero produces transport-only awaiting_validation
evidence. The parent independently validates scientific contents before
completing a fit journal entry; runtime never grants scientific completion.

The campaign holds a stable source-wide advisory lock across orchestration so
different output roots cannot launch concurrent work. Its inode is never unlinked.
Interrupted/cancelled attempts remain ineligible;
resume verifies each completed unit's file hashes and never overwrites evidence.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from project.experiments.baseline_budget import process_tree_rss_bytes
from project.experiments.candidate_contracts import _validate_managed_path
from project.experiments.full_run_contracts import (
    ArtifactEvidence,
    FullFitJournal,
)

POLL_SECONDS = 0.05
TERMINAL_STATES = ("completed", "failed", "cancelled", "interrupted")

# The worker is a real subprocess (never fork from a threaded parent). It imports
# a target function by dotted path and calls it with the unit id and directory.
_WORKER_BOOTSTRAP = (
    "import importlib, json, sys, os, signal, time\n"
    "parent = int(sys.argv[2])\n"
    "leader = os.getpid()\n"
    "guardian = os.fork()\n"
    "if guardian == 0:\n"
    "    while True:\n"
    "        if os.getppid() != leader:\n"
    "            os._exit(0)\n"
    "        try:\n"
    "            fields = open('/proc/%d/stat' % leader).read().rsplit(')', 1)[1].split()\n"
    "            alive = fields[0] != 'Z' and int(fields[1]) == parent\n"
    "        except (OSError, ValueError, IndexError):\n"
    "            alive = False\n"
    "        if not alive:\n"
    "            os.killpg(leader, signal.SIGKILL)\n"
    "            os._exit(1)\n"
    "        time.sleep(0.05)\n"
    "while not os.path.exists(os.path.join(os.path.dirname(sys.argv[1]), 'startup-authorized.json')):\n"
    "    time.sleep(0.01)\n"
    "request = json.load(open(sys.argv[1]))\n"
    "module_name, function_name = request['target'].split(':')\n"
    "module = importlib.import_module(module_name)\n"
    "getattr(module, function_name)(request['unit'], request['directory'])\n"
)


class CampaignLockError(RuntimeError):
    """Raised when a second campaign invocation is already active."""


@dataclass
class WorkUnitOutcome:
    unit: str
    status: str
    exit_code: int | None
    artifacts: tuple[ArtifactEvidence, ...] = ()
    observation: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None


class CampaignLock:
    """Exclusive POSIX advisory flock for campaign runs; resilient across crashes."""

    def __init__(self, path_or_root: str | Path, *, source_root: str | Path | None = None) -> None:
        target = Path(path_or_root)
        if source_root is not None:
            source = _validate_managed_path(source_root, "campaign source", directory=True)
            self.path = source / "campaign.lock"
        elif target.suffix == ".lock" or (target.exists() and not target.is_dir()):
            self.path = target
        else:
            self.path = target / "campaign.lock"
        self._fd: int | None = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            os.close(self._fd)
            self._fd = None
            raise CampaignLockError(
                "campaign already holds an exclusive launch lock"
            ) from error
        try:
            os.ftruncate(self._fd, 0)
            os.lseek(self._fd, 0, os.SEEK_SET)
            os.write(self._fd, f"{os.getpid()} {time.time()}\n".encode())
            os.fsync(self._fd)
        except OSError:
            pass
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        # Keep the inode permanently. Unlinking lets a process holding an open
        # descriptor and a later invocation lock different inodes concurrently.


def shared_project_campaign_lock(path_or_root: str | Path) -> CampaignLock:
    """Explicit helper to acquire a shared project-wide or campaign-wide advisory flock."""
    return CampaignLock(path_or_root)


def stream_file_hash(path: str | Path, chunk_size: int = 65536) -> str:
    """Compute sha256 checksum by streaming in chunks instead of in-memory read_bytes."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()



def run_work_unit(
    unit: str,
    target: str | Callable[[str, Path], None],
    work_dir: str | Path,
    *,
    journal: FullFitJournal | None = None,
    attempt_id: str | None = None,
    expected_artifacts: tuple[str, ...] = (),
    cancel_check: Callable[[], bool] | None = None,
    python_executable: str | None = None,
    spec_binding: str | None = None,
    environment_binding: Mapping[str, Any] | None = None,
    supervisor_token: str | None = None,
    poll_callback: Callable[[], None] | None = None,
    attempt_binding: Mapping[str, Any] | None = None,
    on_started: Callable[[dict[str, Any]], None] | None = None,
) -> WorkUnitOutcome:
    """Run one unit in a fresh subprocess/process group; seal only with evidence.

    ``target`` is a ``"module:function"`` dotted path (production) or a top-level
    callable (tests). It must write files under ``work_dir``; no return value is
    trusted. The parent decides the outcome from the process exit, cancellation
    signal and expected artifact set.
    """
    directory = Path(work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    request = directory / "request.json"
    _write_request(request, unit, directory, target)
    command = [
        python_executable or sys.executable,
        "-c",
        _WORKER_BOOTSTRAP,
        str(request),
        str(os.getpid()),
    ]
    started = time.monotonic()
    peak_rss = 0
    status = "failed"
    exit_code: int | None = None
    reason: str | None = None
    tracked_descendants: set[int] = set()

    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    stdout_f = stdout_path.open("ab")
    stderr_f = stderr_path.open("ab")
    env = dict(os.environ)
    if environment_binding is not None:
        for k, v in environment_binding.items():
            env[str(k)] = str(v)
    python_paths = [p for p in sys.path if p]
    if env.get("PYTHONPATH"):
        for p in env["PYTHONPATH"].split(os.pathsep):
            if p and p not in python_paths:
                python_paths.append(p)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    try:
        process = subprocess.Popen(
            command,
            env=env,
            start_new_session=True,
            stdout=stdout_f,
            stderr=stderr_f,
        )
    finally:
        stdout_f.close()
        stderr_f.close()

    if process.pid > 0:
        tracked_descendants.add(process.pid)
    process._full_process_identities = {}
    initial_identity = _process_identity(process.pid)
    if initial_identity is not None:
        process._full_process_identities[process.pid] = initial_identity
    try:
        if initial_identity is None:
            raise ValueError("worker process identity unavailable before supervision")
        identity_record = {"unit": unit, "pid": process.pid, "identity": list(initial_identity),
                           "parent_pid": os.getpid(), "binding": dict(attempt_binding or {})}
        _atomic_json(directory / "worker-identity.json", identity_record)
        if on_started is not None:
            on_started(identity_record)
        _atomic_json(directory / "startup-authorized.json", {"pid": process.pid, "identity": list(initial_identity)})
    except BaseException:
        _terminate(process, tracked_descendants)
        raise
    try:
        while True:
            rss, descendants = _observe(process.pid)
            peak_rss = max(peak_rss, rss)
            tracked_descendants.update(descendants)
            _remember_process_identities(process, descendants, observed_tree=True)
            if cancel_check is not None and cancel_check():
                status = "cancelled"
                reason = "explicit cancellation"
                break
            if poll_callback is not None:
                poll_callback()
            if process.poll() is not None:
                exit_code = process.returncode
                break
            time.sleep(POLL_SECONDS)
        if poll_callback is not None:
            poll_callback()
    except BaseException as error:
        _terminate(process, tracked_descendants)
        _seal_failure(
            directory,
            unit,
            "interrupted",
            time.monotonic() - started,
            peak_rss,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
        if journal is not None and attempt_id is not None:
            try:
                journal.finish(
                    attempt_id,
                    "interrupted",
                    reason=str(error) or "interrupted by exception",
                )
            except Exception:  # noqa: BLE001, S110
                pass
        raise
    duration = round(time.monotonic() - started, 6)

    cleanup_ok = _terminate(process, tracked_descendants)
    if not cleanup_ok:
        status = "failed"
        cleanup_msg = "descendant cleanup failed: surviving processes remain"
        reason = f"{reason}; {cleanup_msg}" if reason else cleanup_msg
    elif status != "cancelled":
        if exit_code != 0:
            status = "failed"
            reason = f"worker exited with code {exit_code}"
        else:
            _artifacts, missing_reason = _verify_artifacts(
                directory, expected_artifacts
            )
            if missing_reason is not None:
                status = "awaiting_validation"
                reason = missing_reason
            else:
                status = "awaiting_validation"
    return _finalize(
        directory,
        unit,
        status,
        exit_code,
        duration,
        peak_rss,
        reason,
        journal,
        attempt_id,
        expected_artifacts,
        spec_binding=spec_binding,
        environment_binding=environment_binding,
        supervisor_token=supervisor_token,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        process_identities=process._full_process_identities,
        cleanup_confirmed=cleanup_ok,
    )


def _write_request(
    path: Path, unit: str, directory: Path, target: str | Callable
) -> None:
    if isinstance(target, str):
        spec = target
    elif callable(target):
        resolved = getattr(target, "__qualname__", "")
        module = getattr(target, "__module__", None)
        if "." in resolved or module is None:
            raise TypeError(
                "callable targets must be importable top-level functions or 'module:function' strings"
            )
        spec = f"{module}:{resolved}"
    else:
        raise TypeError(
            "target must be a 'module:function' string or a top-level callable"
        )
    _atomic_json(path, {"unit": unit, "directory": str(directory), "target": spec})


def _observe(pid: int) -> tuple[int, set[int]]:
    try:
        return process_tree_rss_bytes(pid)
    except (OSError, ValueError):
        return 0, set()


def _is_alive(pid: int) -> bool:
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat_path = Path(f"/proc/{pid}/stat")
        if stat_path.is_file():
            stat_text = stat_path.read_text()
            state = stat_text[stat_text.rfind(")") + 2 :].split()[0]
            if state == "Z":
                return False
    except (OSError, IndexError, ValueError):
        pass
    return True


def _process_identity(pid: int) -> tuple[int, int, int] | None:
    """Linux start tick, process group and session: a PID alone is not identity."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return int(fields[19]), int(fields[2]), int(fields[3])
    except (OSError, ValueError, IndexError):
        return None


def confirm_owned_attempt_exit(record: Mapping[str, Any]) -> None:
    """Read-only recovery check; never signal a live or reused PID.

    Fixed ordinary workers must not detach descendants into other sessions.
    The bootstrap guardian enforces parent death for the owned process group.
    Recovery rejects any still-live member of that session, including a reused
    group number, rather than guessing that it is safe to retry.
    """
    pid = record.get("pid")
    identity = record.get("identity")
    if (type(pid) is not int or pid <= 0 or type(identity) is not list or len(identity) != 3
            or any(type(value) is not int or value <= 0 for value in identity)
            or identity[1:] != [pid, pid]):
        raise ValueError("exact owned session/start-time identity required")
    if _process_identity(pid) == tuple(identity):
        raise ValueError("owned worker is still alive; recovery refused")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        current = _process_identity(int(entry.name))
        if current is not None and (current[1] == pid or current[2] == pid):
            raise ValueError("owned worker session/descendant cleanup not confirmed")


def _remember_process_identities(process: subprocess.Popen, descendants: set[int], *, observed_tree: bool = False) -> None:
    identities = process._full_process_identities
    leader = identities.get(process.pid)
    leader_now = _process_identity(process.pid)
    for pid in descendants:
        if pid in identities:
            continue
        identity = _process_identity(pid)
        if identity is None or leader is None:
            continue
        if ((leader_now == leader and (observed_tree or identity[2] == process.pid))
                or (leader_now is None and identity[2] == process.pid and identity[0] >= leader[0])):
            identities[pid] = identity


def _find_live_targets(pgid: int, known_pids: set[int]) -> set[int]:
    """Find live descendants and processes belonging to the session/process group."""
    live: set[int] = {pid for pid in known_pids if _is_alive(pid)}
    proc_root = Path("/proc")
    if proc_root.is_dir():
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            child_pid = int(entry.name)
            if child_pid == os.getpid() or child_pid in live:
                continue
            try:
                stat = (entry / "stat").read_text()
                fields = stat[stat.rfind(")") + 2 :].split()
                state, ppid, pgrp = fields[0], int(fields[1]), int(fields[2])
                if state == "Z":
                    continue
                if pgrp == pgid or ppid == pgid or ppid in known_pids:
                    live.add(child_pid)
            except (OSError, ValueError, IndexError):
                continue
    return live


def _terminate(
    process: subprocess.Popen, descendants: set[int], *, grace_seconds: float = 0.5
) -> bool:
    """Terminate the process group and all descendants, ensuring none survive."""
    pgid = process.pid
    tracked = set(descendants)
    if pgid > 0:
        tracked.add(pgid)

    live_targets = _find_live_targets(pgid, tracked)
    if not live_targets and process.poll() is not None:
        return True

    for sig in (signal.SIGTERM, signal.SIGKILL):
        _remember_process_identities(process, live_targets)
        identities = process._full_process_identities
        leader_identity = identities.get(pgid)
        if leader_identity is not None and _process_identity(pgid) == leader_identity:
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        for child in list(live_targets):
            identity = identities.get(child)
            if identity is None or _process_identity(child) != identity:
                continue
            try:
                os.kill(child, sig)
            except (ProcessLookupError, PermissionError):
                pass

        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if process.poll() is None:
                try:
                    process.wait(timeout=0.02)
                except subprocess.TimeoutExpired:
                    pass
            live_targets = _find_live_targets(pgid, tracked | live_targets)
            if not live_targets and process.poll() is not None:
                return True
            time.sleep(0.02)

    if process.poll() is None:
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            pass

    surviving = _find_live_targets(pgid, tracked | live_targets)
    if process.poll() is None:
        surviving.add(process.pid)
    return len(surviving) == 0


def _verify_artifacts(
    directory: Path, names: tuple[str, ...]
) -> tuple[tuple[ArtifactEvidence, ...], str | None]:
    if not names:
        return (), "empty expected artifact set"
    evidence: list[ArtifactEvidence] = []
    for name in names:
        path = directory / name
        if not path.is_file() or path.is_symlink():
            return (), f"missing expected artifact: {name}"
        digest = stream_file_hash(path)
        evidence.append(ArtifactEvidence(name, digest))
    return tuple(evidence), None


def _finalize(
    directory: Path,
    unit: str,
    status: str,
    exit_code: int | None,
    duration: float,
    peak_rss: int,
    reason: str | None,
    journal: FullFitJournal | None,
    attempt_id: str | None,
    expected_names: tuple[str, ...],
    *,
    spec_binding: str | None = None,
    environment_binding: Mapping[str, Any] | None = None,
    supervisor_token: str | None = None,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
    process_identities: Mapping[int, tuple[int, int, int]] | None = None,
    cleanup_confirmed: bool = False,
) -> WorkUnitOutcome:
    observation = {
        "unit": unit,
        "status": status,
        "exit_code": exit_code,
        "elapsed_seconds": duration,
        "peak_process_tree_rss_bytes": peak_rss,
        "rss_limit_enforced": False,
        "max_rss_bytes": None,
        "wall_limit_enforced": False,
        "max_wall_seconds": None,
        "cpu_limit_enforced": False,
        "cpu_threads": None,
        "informed_opt_out": True,
        "reason": reason,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "process_identities": {str(pid): list(identity) for pid, identity in (process_identities or {}).items()},
        "cleanup_confirmed": cleanup_confirmed,
    }
    write_status(directory, unit, status, observation)
    artifacts: tuple[ArtifactEvidence, ...] = ()
    if status == "awaiting_validation":
        artifacts, _ = _verify_artifacts(directory, expected_names)
        # This is transport evidence only. Parent must independently validate
        # scientific contents before sealing its journal attempt.
    elif status == "completed":
        artifacts, _ = _verify_artifacts(directory, expected_names)
        write_completion(
            directory,
            unit,
            status,
            artifacts,
            observation,
            attempt_id=attempt_id,
            spec_binding=spec_binding,
            environment_binding=environment_binding,
            supervisor_token=supervisor_token,
        )
        if journal is not None:
            if attempt_id is None:
                raise ValueError(
                    "a completed unit with a journal requires an attempt id"
                )
            journal.seal(attempt_id, artifacts)
            journal.finish(attempt_id, "completed")
    else:
        if journal is not None and attempt_id is not None:
            try:
                journal.finish(attempt_id, status, reason=reason)
            except Exception:  # noqa: BLE001, S110
                pass
    return WorkUnitOutcome(unit, status, exit_code, artifacts, observation, reason)


def _seal_failure(
    directory: Path,
    unit: str,
    status: str,
    duration: float,
    peak_rss: int,
    *,
    stdout_path: str | None = None,
    stderr_path: str | None = None,
) -> None:
    write_status(
        directory,
        unit,
        status,
        {
            "unit": unit,
            "status": status,
            "elapsed_seconds": duration,
            "peak_process_tree_rss_bytes": peak_rss,
            "informed_opt_out": True,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
        },
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    _fsync_dir(path.parent)


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_status(
    directory: str | Path, unit: str, status: str, observation: Mapping[str, Any]
) -> Path:
    if status not in (*TERMINAL_STATES, "awaiting_validation", "running"):
        raise ValueError("invalid work-unit status")
    path = Path(directory) / f"status_{_slug(unit)}.json"
    _atomic_json(path, {"unit": unit, "status": status, **dict(observation)})
    return path


def write_completion(
    directory: str | Path,
    unit: str,
    status: str,
    artifacts: tuple[ArtifactEvidence, ...],
    observation: Mapping[str, Any],
    *,
    attempt_id: str | None = None,
    spec_binding: str | None = None,
    environment_binding: Mapping[str, Any] | None = None,
    supervisor_token: str | None = None,
) -> Path:
    if status != "completed":
        raise ValueError("completion evidence requires completed status")
    path = Path(directory) / f"completion_{_slug(unit)}.json"
    _atomic_json(
        path,
        {
            "unit": unit,
            "status": status,
            "attempt_id": attempt_id,
            "spec_binding": spec_binding,
            "environment_binding": dict(environment_binding) if environment_binding is not None else None,
            "supervisor_token": supervisor_token,
            "artifacts": [
                {"path": item.path, "sha256": item.sha256} for item in artifacts
            ],
            "observation": dict(observation),
        },
    )
    return path


def verify_resume(
    directory: str | Path,
    unit: str,
    expected_artifacts: tuple[str, ...] = (),
    *,
    attempt_id: str | None = None,
    spec_binding: str | None = None,
    environment_binding: Mapping[str, Any] | None = None,
    supervisor_token: str | None = None,
) -> tuple[bool, str | None]:
    """Return (completed, reason). A forged/old status file alone is insufficient."""
    if not expected_artifacts:
        return False, "empty expected artifact set"
    path = Path(directory) / f"completion_{_slug(unit)}.json"
    if not path.is_file() or path.is_symlink():
        return False, "no completion evidence"
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False, "completion evidence is unreadable"
    if payload.get("status") != "completed":
        return False, "completion evidence is not completed"
    if payload.get("unit") != unit:
        return False, "completion evidence unit mismatch"
    if supervisor_token is not None and payload.get("supervisor_token") != supervisor_token:
        return False, "supervisor token mismatch"
    if attempt_id is not None and payload.get("attempt_id") != attempt_id:
        return False, "attempt id mismatch"
    if spec_binding is not None and payload.get("spec_binding") != spec_binding:
        return False, "spec binding mismatch"
    if environment_binding is not None:
        rec_env = payload.get("environment_binding")
        if rec_env != dict(environment_binding):
            return False, "environment binding mismatch"

    recorded = {item["path"]: item["sha256"] for item in payload.get("artifacts", [])}
    if set(recorded.keys()) != set(expected_artifacts):
        return False, "artifact inventory mismatch"
    for name in expected_artifacts:
        artifact = Path(directory) / name
        if not artifact.is_file():
            return False, f"completed artifact missing: {name}"
        actual = stream_file_hash(artifact)
        if recorded.get(name) != actual:
            return False, f"completed artifact hash mismatch: {name}"
    return True, None


def _slug(unit: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in unit
    )
