"""Importable worker targets for full-campaign runtime subprocess tests.

These top-level functions live outside the test file so the runtime can launch
them as real subprocesses via ``full_campaign_runtime``. Fixture data is only
used by tests; this module never touches prepared or output datasets.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def write_result(unit: str, directory: str) -> None:
    Path(directory, "result.json").write_text('{"ok": true}')


def write_nothing(unit: str, directory: str) -> None:
    return None


def fail_worker(unit: str, directory: str) -> None:
    raise RuntimeError("intentional worker failure")


def sleep_forever(unit: str, directory: str) -> None:
    time.sleep(30)


def write_oof_original(unit: str, directory: str) -> None:
    Path(directory, "oof.parquet").write_bytes(b"original")


def spawn_child_and_exit_zero(unit: str, directory: str) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path(directory, "child.pid").write_text(str(child.pid))
    Path(directory, "result.json").write_text('{"ok": true}')


def spawn_child_and_fail(unit: str, directory: str) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path(directory, "child.pid").write_text(str(child.pid))
    raise RuntimeError("intentional leader failure with live child")


def spawn_child_and_sleep(unit: str, directory: str) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    Path(directory, "child.pid").write_text(str(child.pid))
    time.sleep(60)


def spawn_sigterm_ignoring_child_and_exit_zero(unit: str, directory: str) -> None:
    code = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(60)\n"
    )
    child = subprocess.Popen([sys.executable, "-c", code])
    Path(directory, "child.pid").write_text(str(child.pid))
    Path(directory, "result.json").write_text('{"ok": true}')


def write_stderr_and_result(unit: str, directory: str) -> None:
    sys.stderr.write("worker stderr diagnostic log\n")
    sys.stderr.flush()
    Path(directory, "result.json").write_text('{"ok": true}')


def forge_completion_attempt(unit: str, directory: str) -> None:
    Path(directory, f"completion_{unit}.json").write_text(
        '{"unit": "'
        + unit
        + '", "status": "completed", "artifacts": [{"path": "result.json", "sha256": "fake"}]}'
    )
    Path(directory, "result.json").write_text('{"ok": true}')


def hold_lock_and_signal(unit: str, directory: str) -> None:
    from project.experiments.full_campaign_runtime import CampaignLock

    lock = CampaignLock(directory)
    lock.__enter__()
    Path(directory, "locked.signal").write_text("ok")
    time.sleep(60)

