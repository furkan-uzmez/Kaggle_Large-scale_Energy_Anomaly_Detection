"""Unit tests for the full-stack AutoGluon profile, request, CLI and fold dispatch."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from scripts import train_autogluon_full_stack as cli

from project.models import autogluon_full_stack as stack
from project.models.candidate_artifacts import CANDIDATE_OOF_SCHEMA

START = datetime(2016, 1, 1, tzinfo=UTC).replace(tzinfo=None)
BUILDINGS = 16
HOURS = 24
FEATURES = ("meter_reading", "square_feet")
ENVIRONMENT = {"python": "3.12.12", "pandas": "2.3.3", "autogluon.tabular": "1.4.0"}
BINDING = {
    "run_id": "run", "attempt_id": "a1",
    "config_hash": "a" * 64, "data_hash": "b" * 64, "split_hash": "c" * 64,
}


def test_profile_includes_full_stack_and_weighted_ensemble() -> None:
    profile = stack.full_standard_autogluon_profile(time_limit=120.0, seed=7)
    constructor, fit = profile["constructor"], profile["fit"]
    assert constructor["problem_type"] == "binary"
    assert constructor["eval_metric"] == "roc_auc"
    assert constructor["learner_kwargs"]["random_state"] == 7
    assert set(fit["hyperparameters"]) == {"GBM", "CAT", "XGB", "NN_TORCH", "RF", "XT"}
    assert fit["fit_weighted_ensemble"] is True
    assert fit["time_limit"] == 120.0
    assert fit["raise_on_model_failure"] is True
    assert fit["hyperparameters"]["NN_TORCH"] == {"seed_value": 7}


def test_profile_memory_guard_is_mandatory_and_overrides_validated() -> None:
    with pytest.raises(ValueError, match="memory"):
        stack.full_standard_autogluon_profile(memory_limit_gb=None)
    with pytest.raises(ValueError, match="Unsupported"):
        stack.full_standard_autogluon_profile(hyperparameter_overrides={"KNN": {}})
    profile = stack.full_standard_autogluon_profile(
        memory_limit_gb=8, presets="medium_quality",
        hyperparameter_overrides={"GBM": {"num_leaves": 31}},
    )
    assert profile["fit"]["memory_limit"] == 8
    assert profile["fit"]["presets"] == "medium_quality"
    assert profile["fit"]["hyperparameters"]["GBM"] == {"seed_value": 42, "num_leaves": 31}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_limit": -1.0}, {"time_limit": float("nan")}, {"time_limit": 0},
        {"seed": -1}, {"seed": True}, {"seed": 1.5}, {"seed": 2**32},
        {"memory_limit_gb": 0}, {"memory_limit_gb": "lots"}, {"memory_limit_gb": float("inf")},
        {"num_gpus": -1}, {"num_gpus": "all"},
    ],
)
def test_profile_rejects_invalid_scalars(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        stack.full_standard_autogluon_profile(**kwargs)


def _write_transport(root: Path) -> None:
    train = pa.table({"meter_reading": [1.0, 2.0], "anomaly": [0, 1]})
    predict = pa.table({"row_index": [0, 1], "meter_reading": [1.5, 2.5]})
    pq.write_table(train, root / stack.TRAIN_FILE)
    pq.write_table(predict, root / stack.PREDICT_FILE)


def test_prepare_request_binds_tables_and_is_json_stable(tmp_path: Path) -> None:
    _write_transport(tmp_path)
    request = stack.prepare_full_stack_request(tmp_path, BINDING, ENVIRONMENT, time_limit=60.0)
    assert request["train"]["sha256"] == stack.file_sha256(tmp_path / stack.TRAIN_FILE)
    assert request["predict"]["sha256"] == stack.file_sha256(tmp_path / stack.PREDICT_FILE)
    assert request["profile"]["fit"]["fit_weighted_ensemble"] is True
    assert json.loads(json.dumps(request)) == request
    assert stack.request_sha256(request) == stack.request_sha256(json.loads(json.dumps(request)))
def test_prepare_request_missing_table(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        stack.prepare_full_stack_request(tmp_path, BINDING, ENVIRONMENT)


def test_assert_full_stack_fitted_rejects_missing_families() -> None:
    full = ["LightGBM", "CatBoost", "XGBoost", "NeuralNetTorch", "RandomForest",
            "ExtraTrees", "WeightedEnsemble_L2"]
    stack.assert_full_stack_fitted(full)
    with pytest.raises(RuntimeError, match="CAT"):
        stack.assert_full_stack_fitted([m for m in full if m != "CatBoost"])
    with pytest.raises(RuntimeError, match="Weighted"):
        stack.assert_full_stack_fitted(full[:-1])


def test_cli_parser_defaults_and_overrides() -> None:
    parser = cli.build_parser()
    defaults = parser.parse_args([])
    assert defaults.config == Path("configs/experiments/autogluon_full_stack.yaml")
    assert (defaults.smoke, defaults.dry_run, defaults.time_limit) == (False, False, None)
    args = parser.parse_args(
        ["--config", "c.yaml", "--smoke", "--dry-run", "--time-limit", "30", "--output-dir", "o"]
    )
    assert (args.config, args.smoke, args.dry_run) == (Path("c.yaml"), True, True)
    assert (args.time_limit, args.output_dir) == (30.0, Path("o"))


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Synthetic 16-building data, 4x3 manifest, config, and a fake isolated worker."""
    rows = [
        {"building_id": b, "timestamp": START + timedelta(hours=h), "anomaly": h % 2,
         "meter_reading": float(b + h), "square_feet": 1000 + b}
        for b in range(BUILDINGS) for h in range(HOURS)
    ]
    data = tmp_path / "data.parquet"
    pl.DataFrame(rows).write_parquet(data)
    manifest = pl.DataFrame([
        {"building_id": b, "repeat": r, "seed": 42 + r, "split": "dev", "fold": (b + r) % 4}
        for r in range(3) for b in range(BUILDINGS)
    ])
    splits = tmp_path / "splits.parquet"
    manifest.write_parquet(splits)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "experiment_id": "model_autogluon_full_stack", "seed": 42,
        "time_limit_per_fold": 10, "smoke_time_limit": 5, "presets": "medium_quality",
        "memory_limit_gb": 8,
        "splits": {"path": str(splits)},
        "data": {"features_path": str(data), "feature_pool": list(FEATURES)},
    }))
    interpreter = tmp_path / "python"
    interpreter.write_text("")
    monkeypatch.setattr(cli, "AUTOGLUON_INTERPRETER", interpreter)
    monkeypatch.setattr(cli.bridge, "inspect_environment", lambda _: dict(ENVIRONMENT))
    monkeypatch.setattr(cli.bridge, "validate_environment", lambda _: None)
    return {"config": config, "tmp": tmp_path}


def _fake_worker(interpreter: Path, directory: Path) -> None:
    """Stand-in for the isolated worker honouring the real response contract."""
    request = json.loads((directory / stack.REQUEST_FILE).read_text())
    predict = pq.read_table(directory / stack.PREDICT_FILE)
    scores = pa.table({
        "row_index": predict["row_index"],
        "probability": pa.array([0.25] * predict.num_rows, type=pa.float64()),
    })
    pq.write_table(scores, directory / stack.SCORES_FILE)
    models = ["LightGBM", "CatBoost", "XGBoost", "NeuralNetTorch", "RandomForest",
              "ExtraTrees", "WeightedEnsemble_L2"]
    (directory / stack.RESPONSE_FILE).write_text(json.dumps({
        "protocol_version": 1, "request_hash": stack.request_sha256(request),
        "score_semantics": stack.SCORE_SEMANTICS,
        "scores_sha256": stack.file_sha256(directory / stack.SCORES_FILE),
        "models": models, "best_model": "WeightedEnsemble_L2",
    }))


def _args(workspace: dict[str, Path], *, smoke: bool, dry_run: bool):
    return cli.build_parser().parse_args([
        "--config", str(workspace["config"]), "--output-dir", str(workspace["tmp"] / "out"),
        *(["--smoke"] if smoke else []), *(["--dry-run"] if dry_run else []),
    ])


def test_dry_run_plans_twelve_folds_without_training(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "launch_worker", lambda *_: pytest.fail("dry run must not train"))
    result = cli.run_training_pipeline(_args(workspace, smoke=False, dry_run=True))
    assert result["status"] == "dry_run_completed"
    assert [(u["repeat"], u["fold"]) for u in result["folds_planned"]] == [
        (r, f) for r in range(3) for f in range(4)
    ]
    assert result["profile_summary"]["fit_weighted_ensemble"] is True
    assert Path(result["plan_path"]).is_file()


def test_dry_run_smoke_plans_repeat_zero_two_folds(workspace: dict[str, Path]) -> None:
    result = cli.run_training_pipeline(_args(workspace, smoke=True, dry_run=True))
    assert [(u["repeat"], u["fold"]) for u in result["folds_planned"]] == [(0, 0), (0, 1)]
    assert result["time_limit"] == 5.0


def test_distinct_smoke_output_path_routing() -> None:
    parser = cli.build_parser()
    smoke_settings = cli.resolve_settings(parser.parse_args(["--smoke"]))
    full_settings = cli.resolve_settings(parser.parse_args([]))
    custom_settings = cli.resolve_settings(parser.parse_args(["--smoke", "--output-dir", "custom_dir"]))

    assert smoke_settings.output_dir.parent == (cli.DEFAULT_OUTPUT_DIR / "smoke").resolve()
    assert full_settings.output_dir.parent == cli.DEFAULT_OUTPUT_DIR.resolve()
    assert custom_settings.output_dir.parent == Path("custom_dir").resolve()
    assert smoke_settings.run_id != full_settings.run_id

def test_full_dispatch_writes_canonical_oof_over_all_folds(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def worker(interpreter: Path, directory: Path) -> None:
        calls.append(directory.name)
        train = pl.read_parquet(directory / stack.TRAIN_FILE)
        assert set(train["anomaly"]) == {0, 1}
        _fake_worker(interpreter, directory)

    monkeypatch.setattr(cli, "launch_worker", worker)
    result = cli.run_training_pipeline(_args(workspace, smoke=False, dry_run=False))
    assert calls == [f"r{r}_f{f}" for r in range(3) for f in range(4)]
    oof_path = Path(result["oof_path"])
    assert pq.ParquetFile(oof_path).schema_arrow.equals(CANDIDATE_OOF_SCHEMA, check_metadata=True)
    oof = pl.read_parquet(oof_path)
    assert oof.height == BUILDINGS * HOURS * 3
    assert oof.select("row_index", "repeat").is_duplicated().sum() == 0
    assert set(oof["execution_mode"]) == {"full"} and not oof["non_comparative"].any()
    assert set(oof["split_seed"]) == {42, 43, 44}


def test_smoke_dispatch_is_non_comparative_and_restricted(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "launch_worker", _fake_worker)
    result = cli.run_training_pipeline(_args(workspace, smoke=True, dry_run=False))
    oof = pl.read_parquet(result["oof_path"])
    assert set(oof["execution_mode"]) == {"synthetic_smoke"}
    assert oof["non_comparative"].all() and not oof["n03_eligible"].any()
    assert set(zip(oof["repeat"], oof["fold"], strict=True)) == {(0, 0), (0, 1)}


def test_missing_interpreter_and_worker_failure_raise(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "AUTOGLUON_INTERPRETER", workspace["tmp"] / "absent-python")
    with pytest.raises(FileNotFoundError, match="interpreter"):
        cli.run_training_pipeline(_args(workspace, smoke=True, dry_run=False))


def test_worker_failure_aborts_without_continuing(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def failing(interpreter: Path, directory: Path) -> None:
        seen.append(directory.name)
        raise RuntimeError("worker boom")

    monkeypatch.setattr(cli, "launch_worker", failing)
    with pytest.raises(RuntimeError, match="boom"):
        cli.run_training_pipeline(_args(workspace, smoke=False, dry_run=False))
    assert seen == ["r0_f0"]


def test_tampered_scores_are_rejected(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def tampering(interpreter: Path, directory: Path) -> None:
        _fake_worker(interpreter, directory)
        scores = pq.read_table(directory / stack.SCORES_FILE)
        pq.write_table(scores.set_column(1, "probability", pa.array([0.9] * scores.num_rows)),
                       directory / stack.SCORES_FILE)

    monkeypatch.setattr(cli, "launch_worker", tampering)
    with pytest.raises(ValueError, match="authentication"):
        cli.run_training_pipeline(_args(workspace, smoke=True, dry_run=False))


def test_materialized_fold_is_building_isolated(tmp_path: Path) -> None:
    rows = pl.DataFrame([
        {"building_id": b, "timestamp": START + timedelta(hours=h),
         "anomaly": h % 2, "meter_reading": 1.0, "square_feet": 1}
        for b in range(8) for h in range(6)
    ]).sort("building_id", "timestamp").with_row_index("row_index").with_columns(
        pl.col("row_index").cast(pl.Int64)
    )
    assignments = pl.DataFrame({"building_id": list(range(8)), "fold": [b % 4 for b in range(8)]})
    keys = cli.materialize_fold(rows, assignments, fold=1, directory=tmp_path,
                                features=FEATURES)
    train = pl.read_parquet(tmp_path / stack.TRAIN_FILE)
    assert set(keys["building_id"]) == {1, 5}
    assert train.height == rows.height - keys.height
    assert pl.read_parquet(tmp_path / stack.PREDICT_FILE)["row_index"].to_list() == \
        keys["row_index"].to_list()
    assert "anomaly" not in pl.read_parquet(tmp_path / stack.PREDICT_FILE).columns
