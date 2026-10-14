#!/usr/bin/env python3
"""Run the tabular MLP 2-arm A/B experiment (control vs. challenger).

Iterates over the frozen 4x3 building-grouped repeated split manifest,
fits out-of-fold MLP partitions, and records canonical OOF predictions
conforming to CANDIDATE_OOF_SCHEMA.

torch is imported lazily; ``--dry-run`` mode works without it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from project.models.candidate_artifacts import (
    CANDIDATE_OOF_SCHEMA,
    write_candidate_oof_batches,
)
from project.models.tabular_mlp import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    MissingTorchDependencyError,
    train_mlp_partition,
    validate_feature_allowlist,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/experiments/tabular_mlp_campaign.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/models/tabular_mlp")
EXPERIMENT_ID = "exp_tabular_mlp"
MAX_BATCH_ROWS = 64_000


def _file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file; of the path string when the file does not exist."""
    if not path.is_file():
        return hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(obj: Any) -> str:
    """Hex SHA-256 of a canonical JSON serialisation."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the tabular MLP runner."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate tabular MLP (control vs. challenger) on repeated splits."
    )
    parser.add_argument(
        "--arm",
        choices=["control", "challenger", "both"],
        default="both",
        help="Arm to train: control, challenger, or both (default: both).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke mode: 1 repeat, 2 folds, subset of buildings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute in dry-run mode: mock partition training without torch.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for predictions and manifests.",
    )
    return parser.parse_args(args)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the YAML configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a dictionary")
    return config


def run_tabular_mlp(
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path | None = None,
    smoke: bool = False,
    dry_run: bool = False,
    arm: str = "both",
) -> list[Path]:
    """Execute MLP cross-validation over the repeated split manifest.

    Parameters
    ----------
    config:
        Parsed YAML configuration dict.
    config_path:
        Path to the YAML file (for SHA-256 lineage).
    output_dir:
        Destination directory for OOF parquet + manifest files.
    smoke:
        If True, limit to 1 repeat, 2 folds, and a small building subset.
    dry_run:
        If True, generate deterministic mock predictions without requiring torch.
    arm:
        Which arm(s) to run: ``'control'``, ``'challenger'``, or ``'both'``.

    Returns
    -------
    list[Path]
        Paths to the written OOF parquet files (one per arm).
    """
    arm_norm = arm.lower().strip()
    if arm_norm not in ("control", "challenger", "both"):
        raise ValueError(f"Unsupported arm '{arm}'. Must be one of: control, challenger, both")
    arms_to_run = ["control", "challenger"] if arm_norm == "both" else [arm_norm]

    features_cfg = config.get("features", {})
    numeric_cols: list[str] = list(features_cfg.get("numeric", list(NUMERIC_FEATURES)))
    categorical_cols: list[str] = list(features_cfg.get("categorical", list(CATEGORICAL_FEATURES)))
    validate_feature_allowlist(numeric_cols + categorical_cols)

    target_col: str = config.get("target", "anomaly")
    split_path = Path(config.get("split_manifest", "outputs/splits/building_folds_repeated.parquet"))
    data_path = Path(config.get("data_path", "outputs/prepared/train_joined.parquet"))

    if not split_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    split_sha = _file_sha256(split_path)
    data_sha = _file_sha256(data_path)
    config_sha = _file_sha256(config_path)

    model_cfg = config.get("model", {})
    arch_cfg = model_cfg.get("architecture", {})
    train_cfg = model_cfg.get("training", {})
    model_params: dict[str, Any] = {
        **arch_cfg,
        **train_cfg,
        "seed": config.get("seed", 42),
    }
    fit_config_sha = _json_sha256({"model": "tabular_mlp", "params": model_params})

    arm_defs: dict[str, Any] = config.get("arm_definitions", {})
    experiment_id: str = config.get("experiment_id", EXPERIMENT_ID)

    is_synthetic = bool(smoke or dry_run)
    execution_mode = "synthetic_smoke" if is_synthetic else "full"
    non_comparative = is_synthetic
    n03_eligible = not is_synthetic
    run_id = f"mlp_{execution_mode}_{uuid.uuid4().hex[:8]}"

    split_df = pl.read_parquet(split_path)
    train_joined = pl.read_parquet(data_path)

    dev_splits = split_df.filter(pl.col("split") == "dev")
    dev_building_ids = dev_splits["building_id"].unique().to_list()

    train_dev = train_joined.filter(pl.col("building_id").is_in(dev_building_ids))
    train_dev = (
        train_dev.drop("row_index", strict=False)
        .sort(["building_id", "timestamp"])
        .with_row_index("row_index")
        .with_columns(pl.col("row_index").cast(pl.Int64))
    )

    if is_synthetic:
        repeats = [0]
        folds = [0, 1]
    else:
        repeats = sorted(dev_splits["repeat"].unique().to_list())
        folds = sorted(dev_splits["fold"].unique().to_list())

    logger.info(
        "Starting tabular MLP evaluation: mode=%s, dry_run=%s, arms=%s, repeats=%s, folds=%s",
        execution_mode,
        dry_run,
        arms_to_run,
        repeats,
        folds,
    )

    # Collect OOF rows per arm
    oof_dfs_by_arm: dict[str, list[pl.DataFrame]] = {a: [] for a in arms_to_run}

    for rep in repeats:
        split_seed = 42 + rep
        for fld in folds:
            fold_manifest = dev_splits.filter(pl.col("repeat") == rep)
            train_b_ids = fold_manifest.filter(pl.col("fold") != fld)["building_id"].to_list()
            val_b_ids = fold_manifest.filter(pl.col("fold") == fld)["building_id"].to_list()

            if is_synthetic:
                train_b_ids = train_b_ids[:4]
                val_b_ids = val_b_ids[:2]

            train_fold_pl = train_dev.filter(pl.col("building_id").is_in(train_b_ids))
            val_fold_pl = train_dev.filter(pl.col("building_id").is_in(val_b_ids))

            if val_fold_pl.height == 0:
                continue

            train_fold_pd = train_fold_pl.to_pandas()
            val_fold_pd = val_fold_pl.to_pandas()

            for arm_name in arms_to_run:
                partition_seed = split_seed
                arm_params = dict(model_params)
                arm_params["seed"] = partition_seed

                preds = train_mlp_partition(
                    arm=arm_name,
                    train_df=train_fold_pd,
                    val_df=val_fold_pd,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    target_col=target_col,
                    building_col="building_id",
                    seed=partition_seed,
                    params=arm_params,
                    dry_run=dry_run,
                )

                arm_def = arm_defs.get(arm_name, {})
                candidate_id = arm_def.get("candidate_id", f"mlp_{arm_name}")
                recipe_id = arm_def.get("recipe_id", f"recipe_tabular_mlp_{arm_name}")
                model_config_id = arm_def.get("model_config_id", f"cfg_tabular_mlp_{arm_name}_default")

                fold_oof_df = val_fold_pl.select([
                    pl.col("row_index").cast(pl.Int64),
                    pl.col("building_id").cast(pl.Utf8),
                    pl.col("timestamp").cast(pl.Datetime("ns")),
                    pl.col(target_col).cast(pl.Int64).alias("target"),
                ]).with_columns([
                    pl.lit(candidate_id).alias("candidate_id"),
                    pl.lit(recipe_id).alias("recipe_id"),
                    pl.lit("tabular_mlp").alias("model_family"),
                    pl.lit(model_config_id).alias("model_config_id"),
                    pl.lit(int(rep)).alias("repeat"),
                    pl.lit(int(fld)).alias("fold"),
                    pl.Series("prediction", preds, dtype=pl.Float64),
                    pl.lit(split_sha).alias("split_sha256"),
                    pl.lit(data_sha).alias("data_sha256"),
                    pl.lit(config_sha).alias("config_sha256"),
                    pl.lit(run_id).alias("run_id"),
                    pl.lit(int(split_seed)).alias("split_seed"),
                    pl.lit(experiment_id).alias("experiment_id"),
                    pl.lit(arm_name).alias("arm_id"),
                    pl.lit("positive_probability").alias("score_kind"),
                    pl.lit(1).alias("positive_label"),
                    pl.lit(execution_mode).alias("execution_mode"),
                    pl.lit(non_comparative).alias("non_comparative"),
                    pl.lit(n03_eligible).alias("n03_eligible"),
                    pl.lit(fit_config_sha).alias("model_fit_config_sha256"),
                ])
                oof_dfs_by_arm[arm_name].append(fold_oof_df)

    # Write one parquet per arm
    if output_dir is not None:
        resolved_output_dir = Path(output_dir).resolve()
    else:
        base = DEFAULT_OUTPUT_DIR / "smoke" if smoke else DEFAULT_OUTPUT_DIR
        resolved_output_dir = base.resolve()

    dest_dir = (resolved_output_dir / execution_mode / run_id).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    result_paths: list[Path] = []

    for arm_name in arms_to_run:
        oof_dfs = oof_dfs_by_arm[arm_name]
        if not oof_dfs:
            raise RuntimeError(f"No OOF predictions generated for arm '{arm_name}'.")

        oof_df = pl.concat(oof_dfs, how="vertical")
        arrow_table = oof_df.select(CANDIDATE_OOF_SCHEMA.names).to_arrow().cast(CANDIDATE_OOF_SCHEMA)

        oof_path = (dest_dir / f"mlp_{arm_name}_candidate_oof.parquet").resolve()
        batches = [
            arrow_table.slice(offset, min(MAX_BATCH_ROWS, arrow_table.num_rows - offset))
            for offset in range(0, arrow_table.num_rows, MAX_BATCH_ROWS)
        ]
        write_candidate_oof_batches(oof_path, batches, CANDIDATE_OOF_SCHEMA)
        result_paths.append(oof_path)

        manifest_data: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "model_family": "tabular_mlp",
            "arm": arm_name,
            "execution_mode": execution_mode,
            "non_comparative": non_comparative,
            "n03_eligible": n03_eligible,
            "records_count": arrow_table.num_rows,
            "split_sha256": split_sha,
            "data_sha256": data_sha,
            "config_sha256": config_sha,
            "model_fit_config_sha256": fit_config_sha,
            "oof_parquet": str(oof_path),
        }
        manifest_path = dest_dir / f"mlp_{arm_name}_manifest.json"
        manifest_path.write_text(json.dumps(manifest_data, indent=2))
        logger.info("Wrote manifest: %s", manifest_path)
        logger.info("Wrote OOF: %s (%d rows)", oof_path, arrow_table.num_rows)

    return result_paths


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    try:
        config = load_config(args.config)
        paths = run_tabular_mlp(
            config=config,
            config_path=args.config,
            output_dir=args.output_dir,
            smoke=args.smoke,
            dry_run=args.dry_run,
            arm=args.arm,
        )
        logger.info("Done. OOF files: %s", [str(p) for p in paths])
    except MissingTorchDependencyError as exc:
        logger.error("Missing torch dependency: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
