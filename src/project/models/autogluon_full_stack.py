"""Full-stack AutoGluon profile, isolated request builder, and fold worker.

The profile fits GBM (LightGBM), CAT (CatBoost), XGB (XGBoost), NN_TORCH
(NeuralNetTorch), RF (RandomForest) and XT (ExtraTrees) plus a weighted
ensemble, for a binary ``roc_auc`` objective.

The module top level imports only the standard library and pyarrow so it can
be executed by the isolated ``.venv-autogluon`` interpreter as
``python -I autogluon_full_stack.py --attempt-directory DIR``. pandas and
AutoGluon are imported lazily inside :func:`run_worker`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

FULL_STACK_MODELS: tuple[str, ...] = ("GBM", "CAT", "XGB", "NN_TORCH", "RF", "XT")

# AutoGluon leaderboard name prefix produced by each hyperparameter key.
MODEL_NAME_PREFIXES: dict[str, str] = {
    "GBM": "LightGBM",
    "CAT": "CatBoost",
    "XGB": "XGBoost",
    "NN_TORCH": "NeuralNetTorch",
    "RF": "RandomForest",
    "XT": "ExtraTrees",
}
WEIGHTED_ENSEMBLE_PREFIX = "WeightedEnsemble"

DEFAULT_EVAL_METRIC = "roc_auc"
DEFAULT_PROBLEM_TYPE = "binary"
DEFAULT_LABEL = "anomaly"
DEFAULT_SEED = 42
DEFAULT_MEMORY_LIMIT_GB = 14
PROTOCOL_VERSION = 1
HASH_CHUNK_BYTES = 1 << 16

REQUEST_FILE = "request.json"
RESPONSE_FILE = "response.json"
SCORES_FILE = "scores.parquet"
MODELS_DIRECTORY = "models"
TRAIN_FILE = "train.parquet"
PREDICT_FILE = "predict.parquet"
ROW_INDEX_COLUMN = "row_index"

SCORE_SEMANTICS: dict[str, Any] = {
    "score_type": "probability",
    "positive_label": 1,
    "class_order": [0, 1],
    "row_order": "input",
}


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    """Serialise ``value`` deterministically for hashing."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def request_sha256(request: dict[str, Any]) -> str:
    """Return the canonical content hash of a request payload."""
    return hashlib.sha256(canonical_json(request)).hexdigest()


def _descriptor(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing required transport table: {path}")
    parquet = pq.ParquetFile(path)
    return {
        "path": name,
        "sha256": file_sha256(path),
        "rows": parquet.metadata.num_rows,
        "schema": [
            {"name": f.name, "type": str(f.type), "nullable": f.nullable}
            for f in parquet.schema_arrow
        ],
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_profile_inputs(
    time_limit: float | None,
    seed: int,
    memory_limit_gb: float | str,
    num_gpus: int | str,
) -> None:
    if time_limit is not None and not (_is_number(time_limit) and time_limit > 0):
        raise ValueError("time_limit must be None or a positive finite number")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if memory_limit_gb != "auto" and not (_is_number(memory_limit_gb) and memory_limit_gb > 0):
        raise ValueError("memory_limit_gb is a mandatory guard: positive number or 'auto'")
    if num_gpus != "auto" and not (type(num_gpus) is int and num_gpus >= 0):
        raise ValueError("num_gpus must be 'auto' or a nonnegative integer")


def full_standard_autogluon_profile(
    *,
    time_limit: float | None = None,
    seed: int = DEFAULT_SEED,
    presets: str | None = None,
    memory_limit_gb: float | str = DEFAULT_MEMORY_LIMIT_GB,
    num_gpus: int | str = "auto",
    hyperparameter_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Configure the standard full-stack AutoGluon TabularPredictor.

    Enforces the binary ``roc_auc`` objective, deterministic seeds, a mandatory
    memory guard and a weighted ensemble over all six model families.
    ``raise_on_model_failure`` is on so a missing backend can never silently
    shrink the stack.
    """
    _validate_profile_inputs(time_limit, seed, memory_limit_gb, num_gpus)
    hyperparameters: dict[str, dict[str, Any]] = {
        "GBM": {"seed_value": seed},
        "CAT": {"random_seed": seed},
        "XGB": {"random_state": seed},
        "NN_TORCH": {"seed_value": seed},
        "RF": {"random_state": seed},
        "XT": {"random_state": seed},
    }
    for key, override in (hyperparameter_overrides or {}).items():
        if key not in hyperparameters or not isinstance(override, dict):
            raise ValueError(f"Unsupported hyperparameter override for {key!r}")
        hyperparameters[key].update(override)

    return {
        "constructor": {
            "label": DEFAULT_LABEL,
            "problem_type": DEFAULT_PROBLEM_TYPE,
            "eval_metric": DEFAULT_EVAL_METRIC,
            "positive_class": 1,
            "learner_kwargs": {"random_state": seed},
        },
        "fit": {
            "hyperparameters": hyperparameters,
            "fit_weighted_ensemble": True,
            "time_limit": time_limit,
            "presets": presets,
            "memory_limit": memory_limit_gb,
            "num_bag_folds": 0,
            "num_stack_levels": 0,
            "auto_stack": False,
            "dynamic_stacking": False,
            "calibrate": False,
            "calibrate_decision_threshold": False,
            "hyperparameter_tune_kwargs": None,
            "num_cpus": "auto",
            "num_gpus": num_gpus,
            "fit_strategy": "sequential",
            "raise_on_model_failure": True,
            "raise_on_no_models_fitted": True,
        },
    }


def prepare_full_stack_request(
    root: Path,
    binding: dict[str, Any],
    environment: dict[str, Any],
    *,
    time_limit: float | None = None,
    seed: int = DEFAULT_SEED,
    presets: str | None = None,
    memory_limit_gb: float | str = DEFAULT_MEMORY_LIMIT_GB,
    num_gpus: int | str = "auto",
    hyperparameter_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON request for one isolated fold worker.

    ``root`` must contain ``train.parquet`` and ``predict.parquet``.
    """
    root = Path(root).resolve()
    request: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "binding": dict(binding),
        "environment": dict(environment),
        "profile": full_standard_autogluon_profile(
            time_limit=time_limit,
            seed=seed,
            presets=presets,
            memory_limit_gb=memory_limit_gb,
            num_gpus=num_gpus,
            hyperparameter_overrides=hyperparameter_overrides,
        ),
        "score_semantics": dict(SCORE_SEMANTICS),
        "train": _descriptor(root, TRAIN_FILE),
        "predict": _descriptor(root, PREDICT_FILE),
    }
    return json.loads(canonical_json(request))


def _read_bound_table(root: Path, descriptor: dict[str, Any]) -> pa.Table:
    path = root / descriptor["path"]
    if path.parent != root or file_sha256(path) != descriptor["sha256"]:
        raise ValueError(f"Transport table hash mismatch: {descriptor['path']}")
    table = pq.read_table(path)
    if table.num_rows != descriptor["rows"]:
        raise ValueError(f"Transport table row count mismatch: {descriptor['path']}")
    return table


def assert_full_stack_fitted(leaderboard_models: Sequence[str]) -> None:
    names = list(leaderboard_models)
    missing = [
        key
        for key, prefix in MODEL_NAME_PREFIXES.items()
        if not any(name.startswith(prefix) for name in names)
    ]
    if missing:
        raise RuntimeError(f"Full stack is missing fitted model families: {missing}")
    if not any(name.startswith(WEIGHTED_ENSEMBLE_PREFIX) for name in names):
        raise RuntimeError("Weighted ensemble was not fitted")


def run_worker(attempt_directory: Path) -> dict[str, Any]:
    """Fit AutoGluon on the bound train table and score the bound predict table.

    Writes ``scores.parquet`` (row_index, probability in input order) and
    ``response.json`` bound to the request hash. Raises on any contract breach.
    """
    import pandas as pd
    from autogluon.tabular import TabularPredictor

    root = Path(attempt_directory).resolve()
    request = json.loads((root / REQUEST_FILE).read_text(encoding="utf-8"))
    if request["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("Unsupported request protocol version")
    bound_request_hash = request_sha256(request)
    train = _read_bound_table(root, request["train"]).to_pandas()
    predict_table = _read_bound_table(root, request["predict"])
    row_index = predict_table.column(ROW_INDEX_COLUMN).to_numpy()
    predict = predict_table.drop_columns([ROW_INDEX_COLUMN]).to_pandas()

    profile = request["profile"]
    fit_kwargs = dict(profile["fit"])

    predictor = TabularPredictor(path=str(root / MODELS_DIRECTORY), **profile["constructor"])
    predictor.fit(train_data=train, **fit_kwargs)

    leaderboard = predictor.leaderboard(silent=True)
    models = [str(name) for name in leaderboard["model"]]
    assert_full_stack_fitted(models)

    probabilities = predictor.predict_proba(predict, as_pandas=True)[1].to_numpy(dtype="float64")
    if len(probabilities) != len(row_index) or not pd.Series(probabilities).between(0, 1).all():
        raise ValueError("Predicted probabilities must be finite in [0, 1], one per row")

    scores = pa.table(
        {
            ROW_INDEX_COLUMN: pa.array(row_index, type=pa.int64()),
            "probability": pa.array(probabilities, type=pa.float64()),
        }
    )
    pq.write_table(scores, root / SCORES_FILE)
    response = {
        "protocol_version": PROTOCOL_VERSION,
        "request_hash": bound_request_hash,
        "score_semantics": dict(SCORE_SEMANTICS),
        "scores_sha256": file_sha256(root / SCORES_FILE),
        "models": models,
        "best_model": str(predictor.model_best),
    }
    (root / RESPONSE_FILE).write_text(json.dumps(response, indent=2), encoding="utf-8")
    return response


def main(argv: Sequence[str] | None = None) -> int:
    """Worker entrypoint executed by the isolated AutoGluon interpreter."""
    parser = argparse.ArgumentParser(description="Full-stack AutoGluon fold worker.")
    parser.add_argument("--attempt-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    run_worker(args.attempt_directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
