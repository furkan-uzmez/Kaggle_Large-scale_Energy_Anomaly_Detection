"""Train the full-stack AutoGluon challenger over the frozen 4x3 outer folds.

For every (repeat, fold) the parent materialises building-isolated transport
tables, runs one isolated worker in ``.venv-autogluon`` sequentially, verifies
the authenticated response, and streams the out-of-fold probabilities into the
canonical ``CANDIDATE_OOF_SCHEMA`` parquet. A missing interpreter, a failed
worker, or any contract violation raises; nothing falls back to ``.venv``.

``--dry-run`` only plans; ``--smoke`` uses repeat 0, folds 0-1 on a small
building subset and tags rows ``synthetic_smoke`` / non-comparative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import yaml

from project.models import autogluon_full_stack as stack
from project.models import full_autogluon_bridge as bridge
from project.models.candidate_artifacts import (
    CANDIDATE_OOF_SCHEMA,
    write_candidate_oof_batches,
)

logger = logging.getLogger("autogluon_full_stack")

DEFAULT_CONFIG_PATH = Path("configs/experiments/autogluon_full_stack.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/candidates/autogluon_full_stack")
DEFAULT_SPLITS_PATH = Path("outputs/splits/building_folds_repeated.parquet")
DEFAULT_DATA_PATH = Path("outputs/prepared/train_joined.parquet")
AUTOGLUON_INTERPRETER = Path(".venv-autogluon/bin/python")
CORE_COLUMNS = (
    "meter_reading", "square_feet", "year_built", "floor_count",
    "hour", "weekday", "month", "year", "primary_use",
)
BUILDING_COLUMN = "building_id"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "anomaly"
FULL_REPEATS = (0, 1, 2)
FULL_FOLDS = (0, 1, 2, 3)
SMOKE_FOLDS = (0, 1)
SMOKE_BUILDINGS_PER_FOLD = 6
SMOKE_TIME_LIMIT_SECONDS = 20.0
OOF_BATCH_ROWS = 64_000
CANDIDATE_ARM = "challenger"
MODEL_FAMILY = "autogluon"
MODEL_CONFIG_ID = "autogluon_full_stack_v1"


@dataclass(frozen=True)
class Settings:
    """Resolved run settings; the single source for profile and lineage."""

    run_id: str
    experiment_id: str
    config_path: Path
    splits_path: Path
    data_path: Path
    output_dir: Path
    features: tuple[str, ...]
    time_limit: float
    presets: str | None
    seed: int
    memory_limit_gb: float | str
    num_gpus: int | str
    hyperparameters: dict[str, Any] | None
    smoke: bool

    def profile(self) -> dict[str, Any]:
        """Return the worker profile for these settings."""
        return stack.full_standard_autogluon_profile(
            time_limit=self.time_limit,
            seed=self.seed,
            presets=self.presets,
            memory_limit_gb=self.memory_limit_gb,
            num_gpus=self.num_gpus,
            hyperparameter_overrides=self.hyperparameters,
        )


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Train full-stack AutoGluon over the frozen 4x3 outer folds."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="YAML experiment configuration.")
    parser.add_argument("--smoke", action="store_true",
                        help="Repeat 0, folds 0-1, small building subset, non-comparative.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan folds and write dry_run_plan.json without training.")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="Per-fold AutoGluon time limit in seconds.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for plans, per-fold work files, and OOF output.")
    return parser


def load_config(path: Path) -> dict[str, Any]:
    """Load the YAML experiment configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    return content


def resolve_settings(args: argparse.Namespace) -> Settings:
    """Merge YAML configuration with CLI overrides."""
    config = load_config(args.config)
    if args.time_limit is not None:
        time_limit = float(args.time_limit)
    elif args.smoke:
        time_limit = float(config.get("smoke_time_limit", SMOKE_TIME_LIMIT_SECONDS))
    else:
        time_limit = float(config.get("time_limit_per_fold", 300.0))
    if time_limit <= 0:
        raise ValueError("time limit must be positive")
    experiment_id = str(config.get("experiment_id", "model_autogluon_full_stack"))
    config_sha = stack.file_sha256(Path(args.config))
    mode = "smoke" if args.smoke else "full"
    run_id = f"{experiment_id}-{mode}-{config_sha[:8]}-{uuid.uuid4().hex[:8]}"

    if args.output_dir is not None:
        base_dir = Path(args.output_dir).resolve()
    else:
        base_dir = (DEFAULT_OUTPUT_DIR / "smoke" if args.smoke else DEFAULT_OUTPUT_DIR).resolve()
    output_dir = (base_dir / run_id).resolve()

    return Settings(
        run_id=run_id,
        experiment_id=experiment_id,
        config_path=Path(args.config),
        splits_path=Path(config.get("splits", {}).get("path", DEFAULT_SPLITS_PATH)),
        data_path=Path(config.get("data", {}).get("features_path", DEFAULT_DATA_PATH)),
        output_dir=output_dir,
        features=tuple(config.get("data", {}).get("feature_pool", CORE_COLUMNS)),
        time_limit=time_limit,
        presets=config.get("presets"),
        seed=int(config.get("seed", stack.DEFAULT_SEED)),
        memory_limit_gb=config.get("memory_limit_gb", stack.DEFAULT_MEMORY_LIMIT_GB),
        num_gpus=config.get("num_gpus", "auto"),
        hyperparameters=config.get("hyperparameters"),
        smoke=bool(args.smoke),
    )


def load_manifest(path: Path) -> pl.DataFrame:
    """Read the frozen repeated-fold manifest (never regenerated here)."""
    if not path.is_file():
        raise FileNotFoundError(f"Frozen split manifest not found: {path}")
    manifest = pl.read_parquet(path)
    missing = {BUILDING_COLUMN, "repeat", "seed", "split", "fold"} - set(manifest.columns)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    return manifest.filter(pl.col("split") == "dev")


def plan_folds(dev_manifest: pl.DataFrame, *, smoke: bool) -> list[dict[str, int]]:
    """Enumerate outer (repeat, fold, split seed) work units in execution order."""
    plan = (
        dev_manifest.select("repeat", "fold", "seed").unique().sort("repeat", "fold").to_dicts()
    )
    if smoke:
        plan = [u for u in plan if u["repeat"] == 0 and u["fold"] in SMOKE_FOLDS]
    expected = (
        {(0, f) for f in SMOKE_FOLDS} if smoke
        else {(r, f) for r in FULL_REPEATS for f in FULL_FOLDS}
    )
    if {(u["repeat"], u["fold"]) for u in plan} != expected or len(plan) != len(expected):
        raise ValueError("Split manifest does not provide the expected 4x3 outer folds")
    return plan


def _config_sha256(settings: Settings) -> str:
    return stack.file_sha256(settings.config_path)


def _run_id(settings: Settings) -> str:
    return settings.run_id

def _dry_run(settings: Settings, plan: list[dict[str, int]]) -> dict[str, Any]:
    profile = settings.profile()
    result = {
        "status": "dry_run_completed",
        "dry_run": True,
        "smoke": settings.smoke,
        "experiment_id": settings.experiment_id,
        "run_id": settings.run_id,
        "run_dir": str(settings.output_dir),
        "plan_path": str(settings.output_dir / "dry_run_plan.json"),
        "time_limit": settings.time_limit,
        "presets": settings.presets,
        "seed": settings.seed,
        "config_sha256": _config_sha256(settings),
        "split_sha256": stack.file_sha256(settings.splits_path),
        "data_available": settings.data_path.is_file(),
        "interpreter_available": AUTOGLUON_INTERPRETER.is_file(),
        "folds_planned": plan,
        "profile_summary": {
            "constructor": profile["constructor"],
            "fit_models": list(profile["fit"]["hyperparameters"]),
            "fit_weighted_ensemble": profile["fit"]["fit_weighted_ensemble"],
        },
    }
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "dry_run_plan.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def load_development_rows(
    settings: Settings, dev_manifest: pl.DataFrame
) -> pl.DataFrame:
    """Load dev rows with the canonical (building_id, timestamp) row_index."""
    buildings = dev_manifest.filter(pl.col("repeat") == 0).sort(BUILDING_COLUMN)
    if settings.smoke:
        buildings = buildings.group_by("fold", maintain_order=True).head(SMOKE_BUILDINGS_PER_FOLD)
    columns = (BUILDING_COLUMN, TIMESTAMP_COLUMN, TARGET_COLUMN, *settings.features)
    rows = (
        pl.scan_parquet(settings.data_path)
        .filter(pl.col(BUILDING_COLUMN).is_in(buildings[BUILDING_COLUMN].to_list()))
        .select(columns)
        .collect(engine="streaming")
    )
    if set(rows[BUILDING_COLUMN].unique().to_list()) != set(buildings[BUILDING_COLUMN].to_list()):
        raise ValueError("Prepared rows do not cover every selected development building")
    if rows.select(BUILDING_COLUMN, TIMESTAMP_COLUMN).is_duplicated().any():
        raise ValueError("Prepared development natural keys are not unique")
    return (
        rows.sort(BUILDING_COLUMN, TIMESTAMP_COLUMN)
        .with_row_index(stack.ROW_INDEX_COLUMN)
        .with_columns(pl.col(stack.ROW_INDEX_COLUMN).cast(pl.Int64))
    )


def materialize_fold(
    rows: pl.DataFrame,
    assignments: pl.DataFrame,
    *,
    fold: int,
    directory: Path,
    features: Sequence[str],
) -> pl.DataFrame:
    """Write train/predict transport tables; return the validation key frame."""
    joined = rows.join(assignments.select(BUILDING_COLUMN, "fold"), on=BUILDING_COLUMN, how="inner")
    if joined.height != rows.height:
        raise ValueError("Every development row must have exactly one fold assignment")
    validation = joined.filter(pl.col("fold") == fold)
    training = joined.filter(pl.col("fold") != fold)
    if validation.height == 0 or training.height == 0:
        raise ValueError("Outer fold must have nonempty train and validation rows")
    if set(validation[BUILDING_COLUMN]) & set(training[BUILDING_COLUMN]):
        raise ValueError("Building isolation violated between train and validation")
    if set(training[TARGET_COLUMN].unique().to_list()) != {0, 1}:
        raise ValueError("Training rows must contain both target classes")
    directory.mkdir(parents=True, exist_ok=True)
    columns = [*features, TARGET_COLUMN]
    training.select(columns).write_parquet(directory / stack.TRAIN_FILE)
    validation.select(stack.ROW_INDEX_COLUMN, *features).write_parquet(
        directory / stack.PREDICT_FILE
    )
    return validation.select(
        stack.ROW_INDEX_COLUMN, BUILDING_COLUMN, TIMESTAMP_COLUMN, TARGET_COLUMN
    )


def launch_worker(interpreter: Path, directory: Path) -> None:
    """Run one isolated worker; any failure raises."""
    worker = Path(stack.__file__).resolve()
    command = [str(interpreter), "-I", str(worker), "--attempt-directory", str(directory)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (directory / "worker.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (directory / "worker.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"AutoGluon worker failed in {directory} (exit {completed.returncode}): "
            f"{completed.stderr[-2000:]}"
        )


def collect_scores(directory: Path, validation_keys: pl.DataFrame) -> pl.Series:
    """Verify the worker response and return probabilities aligned to ``validation_keys``."""
    request = json.loads((directory / stack.REQUEST_FILE).read_text(encoding="utf-8"))
    response = json.loads((directory / stack.RESPONSE_FILE).read_text(encoding="utf-8"))
    scores_path = directory / stack.SCORES_FILE
    if (
        response["request_hash"] != stack.request_sha256(request)
        or response["score_semantics"] != stack.SCORE_SEMANTICS
        or response["scores_sha256"] != stack.file_sha256(scores_path)
    ):
        raise ValueError(f"Worker response failed authentication in {directory}")
    stack.assert_full_stack_fitted(response["models"])
    scores = pl.read_parquet(scores_path)
    if scores[stack.ROW_INDEX_COLUMN].to_list() != validation_keys[stack.ROW_INDEX_COLUMN].to_list():
        raise ValueError(f"Worker scores do not match validation row order in {directory}")
    probability = scores["probability"]
    if probability.null_count() or not probability.is_finite().all() or not (
        (probability >= 0) & (probability <= 1)
    ).all():
        raise ValueError(f"Worker probabilities invalid in {directory}")
    return probability


def _oof_table(
    keys: pl.DataFrame, probability: pl.Series, constants: dict[str, Any]
) -> pa.Table:
    frame = keys.with_columns(
        probability.alias("prediction"),
        pl.col(TARGET_COLUMN).alias("target"),
        pl.col(TIMESTAMP_COLUMN).cast(pl.Datetime("ns")),
        *[pl.lit(value).alias(name) for name, value in constants.items()],
    )
    return frame.select(CANDIDATE_OOF_SCHEMA.names).to_arrow().cast(CANDIDATE_OOF_SCHEMA)


def _iter_oof_batches(
    settings: Settings,
    rows: pl.DataFrame,
    dev_manifest: pl.DataFrame,
    plan: list[dict[str, int]],
    interpreter: Path,
    environment: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> Iterator[pa.Table]:
    config_sha = _config_sha256(settings)
    split_sha = stack.file_sha256(settings.splits_path)
    data_sha = stack.file_sha256(settings.data_path)
    run_id = _run_id(settings)
    fit_sha = hashlib.sha256(stack.canonical_json(settings.profile())).hexdigest()
    smoke = settings.smoke
    candidate_id = f"{settings.experiment_id}_{CANDIDATE_ARM}"
    for unit in plan:
        repeat, fold, split_seed = unit["repeat"], unit["fold"], unit["seed"]
        assignments = dev_manifest.filter(pl.col("repeat") == repeat)
        directory = settings.output_dir / f"r{repeat}_f{fold}"
        keys = materialize_fold(
            rows, assignments, fold=fold, directory=directory,
            features=settings.features,
        )
        binding = {
            "run_id": run_id, "attempt_id": f"r{repeat}f{fold}",
            "config_hash": config_sha, "data_hash": data_sha, "split_hash": split_sha,
        }
        request = stack.prepare_full_stack_request(
            directory, binding, environment, time_limit=settings.time_limit,
            seed=settings.seed, presets=settings.presets,
            memory_limit_gb=settings.memory_limit_gb, num_gpus=settings.num_gpus,
            hyperparameter_overrides=settings.hyperparameters,
        )
        (directory / stack.REQUEST_FILE).write_text(json.dumps(request, indent=2), encoding="utf-8")
        logger.info("Fitting repeat %s fold %s (%s validation rows)", repeat, fold, keys.height)
        launch_worker(interpreter, directory)
        probability = collect_scores(directory, keys)
        response = json.loads((directory / stack.RESPONSE_FILE).read_text(encoding="utf-8"))
        summaries.append({
            "repeat": repeat, "fold": fold, "rows": keys.height,
            "models": response["models"], "best_model": response["best_model"],
        })
        table = _oof_table(keys, probability, {
            "candidate_id": candidate_id, "recipe_id": settings.experiment_id,
            "model_family": MODEL_FAMILY, "model_config_id": MODEL_CONFIG_ID,
            "repeat": repeat, "fold": fold, "split_sha256": split_sha,
            "data_sha256": data_sha, "config_sha256": config_sha, "run_id": run_id,
            "split_seed": split_seed, "experiment_id": settings.experiment_id,
            "arm_id": CANDIDATE_ARM, "score_kind": "positive_probability",
            "positive_label": 1,
            "execution_mode": "synthetic_smoke" if smoke else "full",
            "non_comparative": smoke, "n03_eligible": not smoke,
            "model_fit_config_sha256": fit_sha,
        })
        for start in range(0, table.num_rows, OOF_BATCH_ROWS):
            yield table.slice(start, OOF_BATCH_ROWS)


def _train(settings: Settings, plan: list[dict[str, int]], dev_manifest: pl.DataFrame) -> dict[str, Any]:
    if not AUTOGLUON_INTERPRETER.is_file():
        raise FileNotFoundError(f"Isolated AutoGluon interpreter missing: {AUTOGLUON_INTERPRETER}")
    interpreter = AUTOGLUON_INTERPRETER.absolute()
    environment = bridge.inspect_environment(str(interpreter))
    bridge.validate_environment(environment)
    rows = load_development_rows(settings, dev_manifest)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = settings.output_dir / "candidate_oof.parquet"
    summaries: list[dict[str, Any]] = []
    batches = _iter_oof_batches(
        settings, rows, dev_manifest, plan, interpreter, environment, summaries
    )
    written = write_candidate_oof_batches(oof_path, batches, CANDIDATE_OOF_SCHEMA)
    expected_rows = sum(s["rows"] for s in summaries)
    if written["oof_record_count"] != expected_rows or len(summaries) != len(plan):
        raise RuntimeError("OOF coverage differs from the planned folds")
    if not settings.smoke:
        per_repeat = {r: sum(s["rows"] for s in summaries if s["repeat"] == r) for r in FULL_REPEATS}
        if set(per_repeat.values()) != {rows.height}:
            raise RuntimeError("Each repeat must cover every development row exactly once")
    result = {
        "status": "completed",
        "dry_run": False,
        "smoke": settings.smoke,
        "run_id": settings.run_id,
        "run_dir": str(settings.output_dir),
        "oof_path": str(oof_path),
        "oof_record_count": written["oof_record_count"],
        "folds": summaries,
    }
    (settings.output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run_training_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Plan (``--dry-run``) or sequentially execute every outer fold."""
    settings = resolve_settings(args)
    dev_manifest = load_manifest(settings.splits_path)
    plan = plan_folds(dev_manifest, smoke=settings.smoke)
    if args.dry_run:
        return _dry_run(settings, plan)
    return _train(settings, plan, dev_manifest)


def main(argv: Sequence[str] | None = None) -> int:
    """Script entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        run_training_pipeline(args)
    except Exception:
        logger.exception("AutoGluon full-stack run failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
