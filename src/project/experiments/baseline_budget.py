"""Baseline-only approval contract and Linux live process-tree resource guard."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import signal
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

BASELINE_MODELS = ("dummy_prior", "logistic_l2", "hist_gradient_boosting")
BASELINE_PARAMS = {
    "dummy_prior": {"strategy": "prior"},
    "logistic_l2": {"C": 1.0, "l1_ratio": 0.0, "class_weight": None, "max_iter": 1000},
    "hist_gradient_boosting": {"early_stopping": False},
}
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
)
POLL_SECONDS = 0.05


def _rss_limit_enforced(budget: Mapping[str, Any]) -> bool:
    """Require an explicit null cap and boolean opt-out for RSS telemetry only."""
    mode = budget.get("rss_limit_enforced", True)
    if type(mode) is not bool or "max_rss_bytes" not in budget:
        raise ValueError("RSS mode requires a boolean rss_limit_enforced and max_rss_bytes key")
    if (budget["max_rss_bytes"] is None) != (mode is False):
        raise ValueError("RSS telemetry-only mode requires max_rss_bytes=null and rss_limit_enforced=false")
    return mode


def validate_baseline_budget(budget: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Fail closed unless the approved budget describes exactly the frozen 36 fits."""
    from project.features.registry import build_lead_baseline_registry

    if not isinstance(budget, Mapping) or not isinstance(config, Mapping):
        raise TypeError("Baseline budget and config must be mappings")
    contract = {
        "scope": "baseline_only", "approval_status": "approved",
        "resource_status": "approved", "models": list(BASELINE_MODELS),
        "fold_ids": [0, 1, 2, 3], "seed": 42, "planned_fits": 36,
        "repeat_ids": [0, 1, 2], "split_seeds": [42, 43, 44],
    }
    for name, value in contract.items():
        if budget.get(name) != value or isinstance(budget.get(name), bool):
            raise ValueError(f"Baseline execution budget requires {name}={value!r}")
    rss_limit_enforced = _rss_limit_enforced(budget)
    for name, maximum in (
        ("cpu_threads", 4), ("max_rss_bytes", 4 * 1024**3), ("max_wall_seconds", 2700),
    ):
        value = budget.get(name)
        if name == "max_rss_bytes" and not rss_limit_enforced:
            continue
        if type(value) is not int or not 0 < value <= maximum:
            raise ValueError(f"Baseline budget {name} must be a positive integer <= {maximum}")
    models = config.get("models")
    if not isinstance(models, Mapping) or tuple(models) != BASELINE_MODELS:
        raise ValueError("Baseline config must specify the three frozen models")
    if (config.get("seed"), config.get("folds"), config.get("repeats")) != (42, 4, 3) or config.get("split_seeds") != [42, 43, 44]:
        raise ValueError("Baseline config requires estimator seed 42, four folds and three split repeats")
    for name, value in {"target": "anomaly", "positive_label": 1, "primary_metric": "roc_auc", "guardrails": ["average_precision", "log_loss", "brier_score"]}.items():
        if config.get(name) != value:
            raise ValueError(f"Frozen baseline config requires {name}={value!r}")
    registry = build_lead_baseline_registry()
    features = config.get("features")
    if not isinstance(features, Mapping) or set(features) != {"numeric", "categorical"}:
        raise ValueError("Baseline features must contain numeric and categorical lists")
    for kind in ("numeric", "categorical"):
        names = features[kind]
        expected = {name for name in registry.names if (registry.get(name).family == "categorical") == (kind == "categorical")}
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names) or len(names) != len(expected) or set(names) != expected:
            raise ValueError(f"Baseline {kind} features differ from frozen registry")
    for candidate in BASELINE_MODELS:
        if not isinstance(models[candidate], Mapping) or models[candidate].get("params") != BASELINE_PARAMS[candidate]:
            raise ValueError(f"Baseline {candidate} parameters differ from frozen contract")


def load_baseline_budget(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Read and validate a separate baseline approval file."""
    budget = yaml.safe_load(path.read_bytes())
    validate_baseline_budget(budget, config)
    return dict(budget)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _worker(target: Callable, args: Any, threads: int) -> None:
    os.setsid()
    # sklearn already depends on threadpoolctl; it also limits libraries loaded on import.
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=threads):
        raise SystemExit(target(args))


def process_tree_rss_bytes(pid: int) -> tuple[int, set[int]]:
    """Sum live RSS for the child, descendants and its process group using /proc."""
    processes: dict[int, tuple[int, int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            processes[int(entry.name)] = (int(fields[1]), int(fields[2]), int(fields[21]))
        except (OSError, ValueError, IndexError):
            continue  # A process can exit between enumeration and reading its stat.
    selected = {pid}
    selected.update(child for child, (_, group, _) in processes.items() if group == pid)
    while True:
        descendants = {child for child, (parent, _, _) in processes.items() if parent in selected}
        if descendants.issubset(selected):
            break
        selected.update(descendants)
    page_size = os.sysconf("SC_PAGE_SIZE")
    return sum(max(0, processes[child][2]) * page_size for child in selected if child in processes), selected


def _stop_worker(process: multiprocessing.Process, descendants: set[int]) -> None:
    """Terminate the whole worker group, then kill it after a bounded grace period."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        for pid in descendants:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        process.join(timeout=0.5)


def run_guarded(target: Callable, args: Any, run_dir: Path, budget: Mapping[str, Any]) -> int:
    """Spawn a guarded worker; record RSS even when its cap is explicitly disabled.

    Approval is validated by the CLI before calling this generic runtime boundary.
    Polling observes live process-tree RSS (not tracemalloc or historical ru_maxrss).
    """
    if sys.platform != "linux" or not Path("/proc/self/stat").is_file():
        raise RuntimeError("Baseline live RSS guard requires Linux /proc")
    rss_limit_enforced = _rss_limit_enforced(budget)
    for name in ("max_wall_seconds", "max_rss_bytes"):
        if name == "max_rss_bytes" and not rss_limit_enforced:
            continue
        value = budget[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid runtime cap: {name}")
    threads = budget["cpu_threads"]
    previous = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    process = multiprocessing.get_context("spawn").Process(target=_worker, args=(target, args, threads))
    started = time.monotonic()
    peak_rss = 0
    reason = None
    descendants: set[int] = set()
    try:
        for name in THREAD_ENVIRONMENT:
            os.environ[name] = str(threads)
        process.start()
        while True:
            elapsed = time.monotonic() - started
            rss, descendants = process_tree_rss_bytes(process.pid)
            peak_rss = max(peak_rss, rss)
            if elapsed >= budget["max_wall_seconds"]:
                reason = "wall time cap exceeded"
            elif rss_limit_enforced and rss >= budget["max_rss_bytes"]:
                reason = "process-tree rss cap exceeded"
            if reason:
                _stop_worker(process, descendants)
                break
            if not process.is_alive():
                process.join()
                if process.exitcode != 0:
                    reason = f"worker exited with code {process.exitcode}"
                break
            process.join(timeout=POLL_SECONDS)
    except BaseException:
        if process.pid is not None:
            _stop_worker(process, descendants | {process.pid})
        reason = "guard interrupted or monitoring failed"
        raise
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        observation = {
            "run_id": run_dir.name, "status": "failed" if reason else "completed",
            "reason": reason, "elapsed_seconds": round(time.monotonic() - started, 6),
            "peak_process_tree_rss_bytes": peak_rss, "cpu_threads": threads,
            "max_wall_seconds": budget["max_wall_seconds"], "max_rss_bytes": budget["max_rss_bytes"],
            "rss_limit_enforced": rss_limit_enforced,
            "poll_seconds": POLL_SECONDS, "worker_exitcode": process.exitcode,
        }
        _write_json(run_dir / "guard_observations.json", observation)
        if reason:
            _write_json(run_dir / "run_status.json", {"run_id": run_dir.name, "status": "failed", "reason": reason})
    if reason:
        print(f"Baseline guard failed: {reason}; {run_dir}", file=sys.stderr)
        return 1
    return 0
