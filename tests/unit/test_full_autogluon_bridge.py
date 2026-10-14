"""Transport tests only: no estimator fitting or production data."""
import hashlib
import importlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def bridge():
    try:
        return importlib.import_module("project.models.full_autogluon_bridge")
    except ModuleNotFoundError:
        pytest.fail("full AutoGluon bridge contract is not implemented")


def test_fixed_profile_is_uncapped_and_sequential():
    module = bridge()
    params = module.full_xt_profile()
    assert params["hyperparameters"] == {"XT": {"n_estimators": 64, "max_depth": 12, "min_samples_leaf": 2, "random_state": 42}}
    assert params["time_limit"] is None and params["memory_limit"] is None
    assert params["fit_strategy"] == "sequential"
    assert params["num_cpus"] == 1
    module.validate_profile(params)
    params["num_bag_folds"] = 2
    with pytest.raises(ValueError):
        module.validate_profile(params)


def make_request(module, root):
    root.mkdir(mode=0o700)
    pq.write_table(pa.table({"meter_reading": [1., 2.], "anomaly": [0, 1]}), root / "train.parquet")
    pq.write_table(pa.table({"row_index": [10, 11], "meter_reading": [3., 4.]}), root / "predict.parquet")
    binding = {"run_id": "run", "attempt_id": "attempt", "config_hash": "a" * 64,
               "data_hash": "b" * 64, "split_hash": "c" * 64}
    interpreter = str(Path(__file__).resolve().parents[2] / ".venv-autogluon/bin/python")
    return module.prepare_request(root, binding, module.inspect_environment(interpreter))


def test_authenticated_request_and_real_parquet(tmp_path):
    module = bridge()
    root = tmp_path / "attempt"
    request = make_request(module, root)
    signature = module.authenticate(request, b"secret")
    module.verify_request(root, request, (b"secret", signature))
    assert request["train"]["schema"][1] == {"name": "anomaly", "type": "int64", "nullable": True}
    pq.write_table(pa.table({"meter_reading": [9.], "anomaly": [1]}), root / "train.parquet")
    with pytest.raises(ValueError, match="hash"):
        module.verify_request(root, request, (b"secret", signature))


@pytest.mark.parametrize("path", ["../outside.parquet", "/tmp/outside.parquet", "payload.pkl"])
def test_confined_paths(tmp_path, path):
    module = bridge()
    with pytest.raises(ValueError):
        module.confined_path(tmp_path, path)


def test_symlink_and_unknown_profile_rejected(tmp_path):
    module = bridge()
    root = tmp_path / "attempt"
    request = make_request(module, root)
    (root / "alias.parquet").symlink_to(root / "train.parquet")
    with pytest.raises(ValueError, match="symlink"):
        module.confined_path(root, "alias.parquet")
    changed = deepcopy(request)
    changed["profile"]["presets"] = "best_quality"
    with pytest.raises(ValueError):
        module.verify_request(root, changed, (b"secret", module.authenticate(changed, b"secret")))


def test_response_binds_request_and_probability_semantics(tmp_path):
    module = bridge()
    root = tmp_path / "attempt"
    request = make_request(module, root)
    pq.write_table(pa.table({"row_index": [10, 11], "probability": [0., 1.]}), root / "scores.parquet")
    response = module.prepare_response(root, request)
    module.verify_response(root, request, response)
    response["request_hash"] = "0" * 64
    with pytest.raises(ValueError, match="request"):
        module.verify_response(root, request, response)


def test_probe_rejects_wrong_python_without_fitting():
    module = bridge()
    facts = module.inspect_environment(sys.executable)
    assert facts["executable"] == sys.executable
    with pytest.raises(ValueError):
        module.validate_environment(facts)


def test_json_exclusive_and_no_pickle(tmp_path):
    module = bridge()
    root = tmp_path / "attempt"
    request = make_request(module, root)
    module.write_request(root, request)
    assert json.loads((root / "request.json").read_text()) == request
    with pytest.raises(FileExistsError):
        module.write_request(root, request)


def test_missing_arrow_dependency_is_rejected():
    module = bridge()
    with pytest.raises(ValueError, match="dependencies"):
        module.validate_environment({"python": "3.12.12", "pandas": "2.3.3",
                                     "autogluon.tabular": "1.4.0"})


def test_command_requires_exact_probed_environment_and_worker_hash(tmp_path):
    module = bridge()
    worker = tmp_path / "worker.py"
    worker.write_text("print('harmless fixture')\n")
    facts = module.inspect_environment(sys.executable)
    with pytest.raises(ValueError):
        module.supervisor_command(sys.executable, worker, {"environment": facts, "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest()})


def test_invalid_probability_and_reordered_rows_rejected(tmp_path):
    module = bridge()
    root = tmp_path / "attempt"
    request = make_request(module, root)
    for rows, probabilities in [([11, 10], [.1, .2]), ([10, 11], [.1, float("nan")]), ([10, 11], [.1, 1.1])]:
        pq.write_table(pa.table({"row_index": rows, "probability": probabilities}), root / "scores.parquet")
        with pytest.raises(ValueError):
            module.prepare_response(root, request)


def test_verified_secondary_interpreter_runs_only_harmless_stub(tmp_path):
    module = bridge()
    interpreter = str(Path(__file__).resolve().parents[2] / ".venv-autogluon/bin/python")
    facts = module.inspect_environment(interpreter)
    module.validate_environment(facts)
    worker = tmp_path / "worker.py"
    worker.write_text("print('transport-only fixture')\n")
    binding = {"environment": facts, "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest()}
    argv = module.supervisor_command(interpreter, worker, binding)
    result = subprocess.run(argv, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "transport-only fixture"
    worker.write_text("print('changed')\n")
    with pytest.raises(ValueError, match="hash"):
        module.supervisor_command(interpreter, worker, binding)
