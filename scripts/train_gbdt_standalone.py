#!/usr/bin/env python3
"""Run standalone GBDT challenger experiments (LightGBM, CatBoost, XGBoost).

Iterates over the frozen 4x3 repeated split manifest, fits out-of-fold partitions,
and records canonical OOF predictions conforming to CANDIDATE_OOF_SCHEMA.
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
from project.models.gbdt_standalone import (
    SUPPORTED_FAMILIES,
    normalize_family,
    train_standalone_gbdt_partition,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/experiments/gbdt_standalone_challengers.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/models/gbdt_standalone")


def _file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file on disk."""
    if not path.is_file():
        # Fallback pseudo-hash for non-existent mock paths
        return hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(obj: Any) -> str:
    """Compute hex SHA-256 of a JSON-serializable object."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the standalone GBDT runner."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate standalone GBDT challenger models on repeated splits."
    )
    parser.add_argument(
        "--family",
        required=True,
        choices=list(SUPPORTED_FAMILIES),
        help="GBDT model family: lightgbm, catboost, or xgboost.",
    )
    parser.add_argument(
        "--arm",
        choices=["control", "challenger", "both"],
        default="both",
        help="Arm to train: control, challenger, or both.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke mode: 1 repeat, 2 folds, subset of buildings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute in dry-run mode: mock partition training without library dependencies.",
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


def run_standalone_gbdt(
    family: str,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path | None = None,
    smoke: bool = False,
    dry_run: bool = False,
    arm: str = "both",
) -> Path:
    """Execute GBDT cross-validation over the repeated split manifest."""
    fam = normalize_family(family)
    models_cfg = config.get("models", {})
    if fam not in models_cfg:
        raise ValueError(f"Family '{fam}' not configured in {config_path}")

    arm_norm = arm.lower().strip()
    if arm_norm not in ("control", "challenger", "both"):
        raise ValueError(f"Unsupported arm '{arm}'. Must be one of: 'control', 'challenger', 'both'")
    arms_to_run = ["control", "challenger"] if arm_norm == "both" else [arm_norm]

    model_spec = models_cfg[fam]
    model_params = dict(model_spec.get("params", {}))
    recipe_id = model_spec.get("recipe_id", f"recipe_gbdt_{fam}")
    model_config_id = model_spec.get("model_config_id", f"cfg_gbdt_{fam}_default")

    features_cfg = config.get("features", {})
    feature_cols = list(features_cfg.get("numeric", [])) + list(features_cfg.get("categorical", []))
    target_col = config.get("target", "anomaly")

    split_path = Path(config.get("split_manifest", "outputs/splits/building_folds_repeated.parquet"))
    data_path = Path(config.get("data_path", "outputs/prepared/train_joined.parquet"))

    if not split_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    split_sha = _file_sha256(split_path)
    data_sha = _file_sha256(data_path)
    config_sha = _file_sha256(config_path)
    fit_config_sha = _json_sha256({"family": fam, "params": model_params})

    split_df = pl.read_parquet(split_path)
    train_joined = pl.read_parquet(data_path)

    dev_splits = split_df.filter(pl.col("split") == "dev")
    dev_building_ids = dev_splits["building_id"].unique().to_list()

    # Assign canonical row_index strictly to development population
    train_dev = train_joined.filter(pl.col("building_id").is_in(dev_building_ids))
    train_dev = (
        train_dev.drop("row_index", strict=False)
        .sort(["building_id", "timestamp"])
        .with_row_index("row_index")
        .with_columns(pl.col("row_index").cast(pl.Int64))
    )

    is_mock = bool(model_params.get("mock", False))
    is_synthetic = bool(smoke or dry_run or is_mock)
    execution_mode = "synthetic_smoke" if is_synthetic else "full"
    non_comparative = is_synthetic
    n03_eligible = not is_synthetic
    run_id = f"gbdt_{fam}_{execution_mode}_{uuid.uuid4().hex[:8]}"

    # Filter repeats and folds
    if is_synthetic:
        repeats = [0]
        folds = [0, 1]
    else:
        repeats = sorted(dev_splits["repeat"].unique().to_list())
        folds = sorted(dev_splits["fold"].unique().to_list())
    oof_dfs: list[pl.DataFrame] = []
    logger.info(
        "Starting standalone GBDT evaluation: family=%s, mode=%s, dry_run=%s, repeats=%s, folds=%s",
        fam,
        execution_mode,
        dry_run,
        repeats,
        folds,
    )

    for rep in repeats:
        split_seed = 42 + rep
        for fld in folds:
            fold_manifest = dev_splits.filter(pl.col("repeat") == rep)
            train_b_ids = fold_manifest.filter(pl.col("fold") != fld)["building_id"].to_list()
            val_b_ids = fold_manifest.filter(pl.col("fold") == fld)["building_id"].to_list()

            if is_synthetic:
                # In synthetic mode, limit to maximum 4 buildings for speed
                train_b_ids = train_b_ids[:4]
                val_b_ids = val_b_ids[:2]
            train_fold_df = train_dev.filter(pl.col("building_id").is_in(train_b_ids))
            val_fold_df = train_dev.filter(pl.col("building_id").is_in(val_b_ids))
            if val_fold_df.height == 0:
                continue

            # Update seed for current partition
            partition_params = dict(model_params)
            partition_params["seed"] = split_seed
            for arm_idx, arm_name in enumerate(arms_to_run, start=1):
                logger.info(
                    "[%s] Repeat %d/%d, Fold %d/%d, Arm '%s' (%d/%d) starting...",
                    fam,
                    rep + 1,
                    len(repeats),
                    fld + 1,
                    len(folds),
                    arm_name,
                    arm_idx,
                    len(arms_to_run),
                )
                preds = train_standalone_gbdt_partition(
                    family=fam,
                    train_df=train_fold_df,
                    val_df=val_fold_df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    params=partition_params,
                    dry_run=dry_run,
                )

                fold_oof_df = val_fold_df.select([
                    pl.col("row_index").cast(pl.Int64),
                    pl.col("building_id").cast(pl.Utf8),
                    pl.col("timestamp").cast(pl.Datetime("ns")),
                    pl.col(target_col).cast(pl.Int64).alias("target"),
                ]).with_columns([
                    pl.lit(f"{fam}_{arm_name}").alias("candidate_id"),
                    pl.lit(f"{recipe_id}_{arm_name}").alias("recipe_id"),
                    pl.lit(fam).alias("model_family"),
                    pl.lit(f"{model_config_id}_{arm_name}").alias("model_config_id"),
                    pl.lit(int(rep)).alias("repeat"),
                    pl.lit(int(fld)).alias("fold"),
                    pl.Series("prediction", preds, dtype=pl.Float64),
                    pl.lit(split_sha).alias("split_sha256"),
                    pl.lit(data_sha).alias("data_sha256"),
                    pl.lit(config_sha).alias("config_sha256"),
                    pl.lit(run_id).alias("run_id"),
                    pl.lit(int(split_seed)).alias("split_seed"),
                    pl.lit(f"exp_gbdt_{fam}").alias("experiment_id"),
                    pl.lit(arm_name).alias("arm_id"),
                    pl.lit("positive_probability").alias("score_kind"),
                    pl.lit(1).alias("positive_label"),
                    pl.lit(execution_mode).alias("execution_mode"),
                    pl.lit(non_comparative).alias("non_comparative"),
                    pl.lit(n03_eligible).alias("n03_eligible"),
                    pl.lit(fit_config_sha).alias("model_fit_config_sha256"),
                ])
                oof_dfs.append(fold_oof_df)
    if not oof_dfs:
        raise RuntimeError("No out-of-fold predictions were generated.")

    oof_df = pl.concat(oof_dfs, how="vertical")
    arrow_table = oof_df.select(CANDIDATE_OOF_SCHEMA.names).to_arrow().cast(CANDIDATE_OOF_SCHEMA)

    if output_dir is not None:
        resolved_output_dir = Path(output_dir).resolve()
    else:
        base = DEFAULT_OUTPUT_DIR / "smoke" if smoke else DEFAULT_OUTPUT_DIR
        resolved_output_dir = base.resolve()

    dest_dir = (resolved_output_dir / execution_mode / run_id).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    oof_path = (dest_dir / f"{fam}_candidate_oof.parquet").resolve()
    # Stream in chunks of at most 64,000 rows to satisfy write_candidate_oof_batches
    max_batch_size = 64_000
    batches = [
        arrow_table.slice(offset, min(max_batch_size, arrow_table.num_rows - offset))
        for offset in range(0, arrow_table.num_rows, max_batch_size)
    ]

    write_candidate_oof_batches(oof_path, batches, CANDIDATE_OOF_SCHEMA)
    manifest_data = {
        "schema_version": 1,
        "run_id": run_id,
        "family": fam,
        "execution_mode": execution_mode,
        "arm": arm_norm,
        "arms": arms_to_run,
        "non_comparative": non_comparative,
        "n03_eligible": n03_eligible,
        "records_count": arrow_table.num_rows,
        "split_sha256": split_sha,
        "data_sha256": data_sha,
        "config_sha256": config_sha,
        "model_fit_config_sha256": fit_config_sha,
        "oof_parquet": str(oof_path),
    }
    manifest_path = dest_dir / f"{fam}_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2))
    logger.info("Successfully wrote manifest to %s", manifest_path)

    return oof_path


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    run_standalone_gbdt(
        family=args.family,
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        smoke=args.smoke,
        dry_run=args.dry_run,
        arm=args.arm,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
