from __future__ import annotations

import copy
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def budget_module():
    try:
        return importlib.import_module("project.experiments.baseline_budget")
    except ModuleNotFoundError:
        pytest.fail("Baseline execution approval and live resource guard are missing")


def approved_budget():
    return {
        "scope": "baseline_only", "approval_status": "approved",
        "resource_status": "approved", "models": ["dummy_prior", "logistic_l2", "hist_gradient_boosting"],
        "fold_ids": [0, 1, 2, 3], "seed": 42, "planned_fits": 36,
        "repeat_ids": [0, 1, 2], "split_seeds": [42, 43, 44],
        "cpu_threads": 4, "max_rss_bytes": 4 * 1024**3, "max_wall_seconds": 2700,
    }


@pytest.mark.parametrize("field,value", [
    ("approval_status", "pending"), ("resource_status", "pending"),
    ("scope", "feature_campaign"), ("models", ["dummy_prior"]),
    ("fold_ids", [0]), ("seed", 7), ("planned_fits", 13),
    ("cpu_threads", None), ("max_rss_bytes", None), ("max_wall_seconds", None),
    ("cpu_threads", 5), ("max_rss_bytes", 5 * 1024**3), ("max_wall_seconds", 2701),
])
def test_rejects_unapproved_or_expanded_scope(field, value):
    module = budget_module()
    budget = approved_budget()
    budget[field] = value
    config = yaml.safe_load((ROOT / "configs/experiments/baselines.yaml").read_text())
    with pytest.raises(ValueError):
        module.validate_baseline_budget(budget, config)


def test_approved_contract_and_frozen_config():
    module = budget_module()
    config = yaml.safe_load((ROOT / "configs/experiments/baselines.yaml").read_text())
    module.validate_baseline_budget(approved_budget(), config)
    changed = copy.deepcopy(config)
    changed["models"]["logistic_l2"]["params"]["C"] = 2
    with pytest.raises(ValueError):
        module.validate_baseline_budget(approved_budget(), changed)


def test_explicit_approved_telemetry_only_budget():
    config = yaml.safe_load((ROOT / "configs/experiments/baselines.yaml").read_text())
    budget_module().validate_baseline_budget(
        approved_budget() | {"max_rss_bytes": None, "rss_limit_enforced": False}, config
    )


@pytest.mark.parametrize("limits", [
    {"max_rss_bytes": None},
    {"max_rss_bytes": None, "rss_limit_enforced": True},
    {"rss_limit_enforced": False},
    {"max_rss_bytes": None, "rss_limit_enforced": 0},
    {"rss_limit_enforced": "false"},
    {"max_rss_bytes": None, "rss_limit_enforced": False, "resource_status": "pending"},
])
def test_invalid_rss_modes_fail_closed(limits):
    config = yaml.safe_load((ROOT / "configs/experiments/baselines.yaml").read_text())
    with pytest.raises(ValueError):
        budget_module().validate_baseline_budget(approved_budget() | limits, config)


def test_telemetry_mode_requires_explicit_null_cap_key():
    budget = approved_budget() | {"rss_limit_enforced": False}
    del budget["max_rss_bytes"]
    config = yaml.safe_load((ROOT / "configs/experiments/baselines.yaml").read_text())
    with pytest.raises(ValueError):
        budget_module().validate_baseline_budget(budget, config)


def test_telemetry_only_large_observed_rss_does_not_stop_worker(tmp_path, monkeypatch):
    module = budget_module()
    actual_rss = module.process_tree_rss_bytes

    def large_rss(pid):
        rss, descendants = actual_rss(pid)
        return rss + 5 * 1024**3, descendants

    monkeypatch.setattr(module, "process_tree_rss_bytes", large_rss)
    budget = approved_budget() | {"max_rss_bytes": None, "rss_limit_enforced": False}
    assert module.run_guarded(successful_worker, str(tmp_path), tmp_path, budget) == 0
    observation = json.loads((tmp_path / "guard_observations.json").read_text())
    assert observation["rss_limit_enforced"] is False
    assert observation["max_rss_bytes"] is None
    assert observation["peak_process_tree_rss_bytes"] > 4 * 1024**3
    assert observation["worker_exitcode"] == 0
    environment = json.loads((tmp_path / "worker_environment.json").read_text())
    assert environment["OMP_NUM_THREADS"] == "4"


def test_telemetry_only_wall_cap_still_kills_worker(tmp_path):
    budget = approved_budget() | {
        "max_rss_bytes": None, "rss_limit_enforced": False, "max_wall_seconds": 1,
    }
    assert budget_module().run_guarded(ignoring_term_worker, str(tmp_path), tmp_path, budget) == 1
    observation = json.loads((tmp_path / "guard_observations.json").read_text())
    assert observation["reason"] == "wall time cap exceeded"
    assert observation["worker_exitcode"] == -9
    assert observation["peak_process_tree_rss_bytes"] > 0


@pytest.mark.parametrize("limits", [
    {"rss_limit_enforced": False},
    {"max_rss_bytes": None, "rss_limit_enforced": True},
    {"max_rss_bytes": None},
])
def test_runtime_rejects_contradictory_rss_modes(tmp_path, limits):
    with pytest.raises(ValueError):
        budget_module().run_guarded(successful_worker, str(tmp_path), tmp_path, approved_budget() | limits)
    assert not (tmp_path / "worker_environment.json").exists()


def sleeping_worker(args):
    time.sleep(10)
    return 0


def memory_worker(args):
    allocation = bytearray(80 * 1024**2)
    time.sleep(10)
    return len(allocation) * 0


def successful_worker(args):
    Path(args).joinpath("worker_environment.json").write_text(
        json.dumps(__import__("os").environ.copy())
    )
    return 0


def descendant_worker(args):
    child = subprocess.Popen([sys.executable, "-c", "import time; allocation = bytearray(160 * 1024**2); time.sleep(10)"])
    Path(args).joinpath("descendant_pid").write_text(str(child.pid))
    time.sleep(10)
    return 0


def ignoring_term_worker(args):
    import signal

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(args).joinpath("ready").write_text(str(os.getpid()))
    time.sleep(10)
    return 0


def test_cap_counts_descendant_rss_and_stops_whole_tree(tmp_path):
    module = budget_module()
    budget = approved_budget() | {"max_rss_bytes": 180 * 1024**2, "max_wall_seconds": 5}
    assert module.run_guarded(descendant_worker, str(tmp_path), tmp_path, budget) != 0
    observation = json.loads((tmp_path / "guard_observations.json").read_text())
    assert observation["reason"] == "process-tree rss cap exceeded"
    pid = int((tmp_path / "descendant_pid").read_text())
    assert module.process_tree_rss_bytes(pid)[0] == 0


def test_wall_cap_kills_worker_ignoring_terminate(tmp_path):
    module = budget_module()
    budget = approved_budget() | {"max_wall_seconds": 1, "max_rss_bytes": 1024**3}
    assert module.run_guarded(ignoring_term_worker, str(tmp_path), tmp_path, budget) != 0
    assert (tmp_path / "ready").exists()
    observation = json.loads((tmp_path / "guard_observations.json").read_text())
    assert observation["worker_exitcode"] == -9


@pytest.mark.parametrize("worker,limits,reason", [
    (sleeping_worker, {"max_wall_seconds": 0.3, "max_rss_bytes": 1024**3}, "wall"),
    (memory_worker, {"max_wall_seconds": 10, "max_rss_bytes": 60 * 1024**2}, "rss"),
])
def test_real_spawned_worker_is_stopped_at_cap(tmp_path, worker, limits, reason):
    module = budget_module()
    budget = approved_budget() | limits
    result = module.run_guarded(worker, str(tmp_path), tmp_path, budget)
    assert result != 0
    observation = json.loads((tmp_path / "guard_observations.json").read_text())
    assert observation["status"] == "failed"
    assert reason in observation["reason"]
    assert json.loads((tmp_path / "run_status.json").read_text())["status"] == "failed"
    assert observation["elapsed_seconds"] < 5


def test_worker_receives_thread_environment_before_spawn(tmp_path):
    module = budget_module()
    assert module.run_guarded(successful_worker, str(tmp_path), tmp_path, approved_budget()) == 0
    environment = json.loads((tmp_path / "worker_environment.json").read_text())
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
        assert environment[name] == "4"
    observation = json.loads((tmp_path / "guard_observations.json").read_text())
    assert observation["status"] == "completed"
    assert observation["peak_process_tree_rss_bytes"] > 0


def test_full_cli_denies_missing_budget_before_fitting(tmp_path, monkeypatch):
    from scripts import train

    called = []
    monkeypatch.setattr(train, "_run", lambda args: called.append(args) or 0)
    assert train.main(["--full", "--project-root", str(tmp_path)]) != 0
    assert not called
