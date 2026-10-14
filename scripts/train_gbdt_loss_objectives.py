#!/usr/bin/env python3
"""Run GBDT Loss Objectives experiments (LogLoss vs. Weighted Loss vs. Focal Loss).

Evaluates LightGBM, CatBoost, and XGBoost across the frozen 4x3 repeated Stratified
Group K-Fold split manifest for 3 loss objective arms:
- Arm 1: logloss (Standard unweighted binary logloss)
- Arm 2: weighted (Class-weighted binary loss, penalizing missed anomalies 8x)
- Arm 3: focal (Focal loss with gamma=2.0, alpha=0.25 down-weighting easy background samples)

Generates canonical OOF parquets conforming to CANDIDATE_OOF_SCHEMA and transport
validation JSON files recording run lineage, model configs, and SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from project.models.candidate_artifacts import (
    CANDIDATE_OOF_SCHEMA,
    write_candidate_oof_batches,
)
from project.models.gbdt_loss_functions import (
    SUPPORTED_FAMILIES,
    SUPPORTED_LOSS_ARMS,
    normalize_family,
    normalize_loss,
    train_loss_objective_partition,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/experiments/gbdt_loss_objectives_campaign.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/models/gbdt_loss_objectives")


def _file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file on disk."""
    if not path.is_file():
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


FORBIDDEN_EXACT_NAMES: frozenset[str] = frozenset(
    {
        "anomaly",
        "building_hour",
        "building_id",
        "building_meter",
        "fold",
        "id",
        "label",
        "repeat",
        "row_index",
        "seed",
        "site_id",
        "split",
        "support_stratum",
        "target",
        "timestamp",
        "y",
    }
)


def is_forbidden_predictor(name: str) -> bool:
    """Check if a column name violates anti-leakage or ID/metadata constraints."""
    lowered = name.casefold()
    return (
        lowered in FORBIDDEN_EXACT_NAMES
        or lowered.endswith("_id")
        or lowered.startswith(("gte_", "source_", "fold_"))
    )


def validate_predictor_cols(predictor_cols: Sequence[str]) -> list[str]:
    """Validate predictor columns against forbidden names and anti-leakage contracts.

    Guarantees:
    - Predictor columns list is non-empty.
    - No duplicate columns.
    - No column starts with 'gte_' (unverified global target encoding leak risk).
    - No column ends with '_id' (site_id, building_id, etc.).
    - No reserved target/metadata/fold identifiers.
    """
    if not predictor_cols:
        raise ValueError("Predictor columns list cannot be empty")

    seen: set[str] = set()
    validated: list[str] = []
    for col in predictor_cols:
        if not isinstance(col, str) or not col.strip():
            raise ValueError(f"Invalid predictor column name: {col!r}")
        if col in seen:
            raise ValueError(f"Duplicate predictor column: {col}")
        seen.add(col)
        if is_forbidden_predictor(col):
            raise ValueError(
                f"Forbidden predictor column '{col}': violates anti-leakage contract "
                f"(starts with 'gte_', ends with '_id', or is a reserved target/metadata field)"
            )
        validated.append(col)
    return validated


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GBDT Loss Objectives cross-validation campaign.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--family",
        type=str,
        default="all",
        choices=["all", "lightgbm", "catboost", "xgboost"],
        help="GBDT model family to evaluate, or 'all'.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="all",
        choices=["all", "logloss", "weighted", "focal"],
        help="Loss objective arm to evaluate, or 'all'.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke mode: 1 repeat, 2 folds, subset of buildings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute dry-run mode: deterministic mock training without full fits.",
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
        help="Custom output directory for predictions and transport artifacts.",
    )
    return parser.parse_args(args)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and validate campaign configuration file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a dictionary")
    return config


def run_gbdt_loss_objectives(
    family: str,
    loss: str,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path | None = None,
    smoke: bool = False,
    dry_run: bool = False,
    split_df: pl.DataFrame | None = None,
    train_df: pl.DataFrame | None = None,
) -> list[Path]:
    """Execute GBDT loss objectives cross-validation.

    Parameters
    ----------
    family:
        Model family filter: 'lightgbm', 'catboost', 'xgboost', or 'all'.
    loss:
        Loss arm filter: 'logloss', 'weighted', 'focal', or 'all'.
    config:
        Parsed campaign YAML configuration mapping.
    config_path:
        Path to campaign YAML configuration file on disk.
    output_dir:
        Optional custom root directory for persisting OOF parquets and validation JSONs.
    smoke:
        If True, restricts execution to repeat 0, folds 0-1, and 4 train / 2 val buildings.
    dry_run:
        If True, executes deterministic mock training without fitting real GBDTs.
    split_df:
        Optional preloaded split manifest DataFrame (for testing).
    train_df:
        Optional preloaded joined training DataFrame (for testing).

    Returns
    -------
    list[Path]:
        List of generated OOF parquet artifact paths.
    """
    models_cfg = config.get("models", {})
    families_to_run = (
        list(SUPPORTED_FAMILIES) if family == "all" else [normalize_family(family)]
    )
    losses_to_run = (
        list(SUPPORTED_LOSS_ARMS) if loss == "all" else [normalize_loss(loss)]
    )

    target_col = config.get("target", "anomaly")
    split_path = Path(
        config.get("split_manifest", "outputs/splits/building_folds_repeated.parquet")
    )
    data_path = Path(config.get("data_path", "outputs/prepared/train_joined.parquet"))

    if split_df is None:
        if not split_path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {split_path}")
        split_df = pl.read_parquet(split_path)

    if train_df is None:
        if not data_path.is_file():
            raise FileNotFoundError(f"Training data not found: {data_path}")
        train_joined = pl.read_parquet(data_path)
    else:
        train_joined = train_df

    split_sha = (
        _file_sha256(split_path)
        if split_path.is_file()
        else hashlib.sha256(b"mock_split").hexdigest()
    )
    data_sha = (
        _file_sha256(data_path)
        if data_path.is_file()
        else hashlib.sha256(b"mock_data").hexdigest()
    )
    config_sha = (
        _file_sha256(config_path)
        if config_path.is_file()
        else hashlib.sha256(b"mock_config").hexdigest()
    )

    # Resolve predictor columns strictly from declared allowlist
    features_cfg = config.get("features", {})
    if features_cfg.get("predictors"):
        declared_predictors = list(features_cfg["predictors"])
    elif "numeric" in features_cfg or "categorical" in features_cfg:
        declared_predictors = list(features_cfg.get("numeric", [])) + list(
            features_cfg.get("categorical", [])
        )
    else:
        raise ValueError(
            "Configuration must explicitly declare 'features.predictors' allowlist"
        )

    predictor_cols = validate_predictor_cols(declared_predictors)

    # Ensure all declared predictors exist in train_joined
    missing_cols = [c for c in predictor_cols if c not in train_joined.columns]
    if missing_cols:
        raise ValueError(
            f"Declared predictors not found in training dataset: {missing_cols}"
        )

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

    is_synthetic = bool(smoke or dry_run)
    execution_mode = "synthetic_smoke" if is_synthetic else "full"
    non_comparative = is_synthetic
    n03_eligible = not is_synthetic

    if is_synthetic:
        repeats = [0]
        folds = [0, 1]
    else:
        repeats = sorted(dev_splits["repeat"].unique().to_list())
        folds = sorted(dev_splits["fold"].unique().to_list())

    result_paths: list[Path] = []
    loss_definitions = config.get("loss_definitions", {})

    for fam in families_to_run:
        if fam not in models_cfg:
            raise ValueError(f"Family '{fam}' not configured in {config_path}")

        model_spec = models_cfg[fam]
        base_params = dict(model_spec.get("base_params", model_spec.get("params", {})))
        recipe_id = model_spec.get("recipe_id", f"recipe_gbdt_loss_{fam}")
        model_config_id = model_spec.get(
            "model_config_id", f"cfg_gbdt_loss_{fam}_default"
        )
        model_arms_cfg = model_spec.get("arms", {})

        for loss_arm in losses_to_run:
            run_id = (
                f"gbdt_loss_{fam}_{loss_arm}_{execution_mode}_{uuid.uuid4().hex[:8]}"
            )
            arm_specific_params = dict(model_arms_cfg.get(loss_arm, {}))
            fit_config_sha = _json_sha256(
                {
                    "family": fam,
                    "loss_arm": loss_arm,
                    "base_params": base_params,
                    "arm_params": arm_specific_params,
                }
            )

            oof_dfs: list[pl.DataFrame] = []

            logger.info(
                "Executing GBDT Loss Objectives: family=%s, loss=%s, mode=%s, repeats=%s, folds=%s",
                fam,
                loss_arm,
                execution_mode,
                repeats,
                folds,
            )

            for rep in repeats:
                split_seed = 42 + rep
                for fld in folds:
                    fold_manifest = dev_splits.filter(pl.col("repeat") == rep)
                    train_b_ids = fold_manifest.filter(pl.col("fold") != fld)[
                        "building_id"
                    ].to_list()
                    val_b_ids = fold_manifest.filter(pl.col("fold") == fld)[
                        "building_id"
                    ].to_list()

                    if is_synthetic:
                        train_b_ids = train_b_ids[:4]
                        val_b_ids = val_b_ids[:2]

                    train_fold_df = train_dev.filter(
                        pl.col("building_id").is_in(train_b_ids)
                    )
                    val_fold_df = train_dev.filter(
                        pl.col("building_id").is_in(val_b_ids)
                    )

                    if val_fold_df.height == 0:
                        continue

                    partition_params = dict(base_params)
                    partition_params.update(arm_specific_params)
                    partition_params["seed"] = split_seed

                    preds = train_loss_objective_partition(
                        family=fam,
                        loss=loss_arm,
                        train_df=train_fold_df,
                        val_df=val_fold_df,
                        feature_cols=predictor_cols,
                        target_col=target_col,
                        params=partition_params,
                        dry_run=dry_run,
                    )

                    fold_oof_df = val_fold_df.select(
                        [
                            pl.col("row_index").cast(pl.Int64),
                            pl.col("building_id").cast(pl.Utf8),
                            pl.col("timestamp").cast(pl.Datetime("ns")),
                            pl.col(target_col).cast(pl.Int64).alias("target"),
                        ]
                    ).with_columns(
                        [
                            pl.lit(f"{fam}_{loss_arm}").alias("candidate_id"),
                            pl.lit(f"{recipe_id}_{loss_arm}").alias("recipe_id"),
                            pl.lit(fam).alias("model_family"),
                            pl.lit(f"{model_config_id}_{loss_arm}").alias(
                                "model_config_id"
                            ),
                            pl.lit(int(rep)).alias("repeat"),
                            pl.lit(int(fld)).alias("fold"),
                            pl.Series("prediction", preds, dtype=pl.Float64),
                            pl.lit(split_sha).alias("split_sha256"),
                            pl.lit(data_sha).alias("data_sha256"),
                            pl.lit(config_sha).alias("config_sha256"),
                            pl.lit(run_id).alias("run_id"),
                            pl.lit(int(split_seed)).alias("split_seed"),
                            pl.lit("regime_gbdt_loss_objectives").alias(
                                "experiment_id"
                            ),
                            pl.lit(loss_arm).alias("arm_id"),
                            pl.lit("positive_probability").alias("score_kind"),
                            pl.lit(1).alias("positive_label"),
                            pl.lit(execution_mode).alias("execution_mode"),
                            pl.lit(non_comparative).alias("non_comparative"),
                            pl.lit(n03_eligible).alias("n03_eligible"),
                            pl.lit(fit_config_sha).alias("model_fit_config_sha256"),
                        ]
                    )
                    oof_dfs.append(fold_oof_df)

            if not oof_dfs:
                raise RuntimeError(
                    f"No out-of-fold predictions generated for {fam}_{loss_arm}."
                )

            oof_df = pl.concat(oof_dfs, how="vertical")
            arrow_table = (
                oof_df.select(CANDIDATE_OOF_SCHEMA.names)
                .to_arrow()
                .cast(CANDIDATE_OOF_SCHEMA)
            )

            # Ensure schema conformity and null-free invariant
            if arrow_table.schema.names != CANDIDATE_OOF_SCHEMA.names:
                raise ValueError(
                    "Generated OOF schema does not match CANDIDATE_OOF_SCHEMA"
                )
            if any(oof_df[n].null_count() > 0 for n in CANDIDATE_OOF_SCHEMA.names):
                raise ValueError("Generated OOF records contain null values")

            if output_dir is not None:
                resolved_output_dir = Path(output_dir).resolve()
            else:
                base = DEFAULT_OUTPUT_DIR / "smoke" if smoke else DEFAULT_OUTPUT_DIR
                resolved_output_dir = base.resolve()

            dest_dir = (
                resolved_output_dir / execution_mode / f"{fam}_{loss_arm}_{run_id}"
            ).resolve()
            dest_dir.mkdir(parents=True, exist_ok=True)
            oof_path = (dest_dir / f"{fam}_{loss_arm}_candidate_oof.parquet").resolve()

            # Stream out in max 64,000 batches
            max_batch_size = 64_000
            batches = [
                arrow_table.slice(
                    offset, min(max_batch_size, arrow_table.num_rows - offset)
                )
                for offset in range(0, arrow_table.num_rows, max_batch_size)
            ]
            write_candidate_oof_batches(oof_path, batches, CANDIDATE_OOF_SCHEMA)

            manifest_data = {
                "schema_version": 1,
                "run_id": run_id,
                "family": fam,
                "loss_arm": loss_arm,
                "arm": loss_arm,
                "execution_mode": execution_mode,
                "non_comparative": non_comparative,
                "n03_eligible": n03_eligible,
                "records_count": arrow_table.num_rows,
                "split_sha256": split_sha,
                "data_sha256": data_sha,
                "config_sha256": config_sha,
                "model_fit_config_sha256": fit_config_sha,
                "oof_parquet": str(oof_path),
                "loss_definition": loss_definitions.get(loss_arm, {}),
                "arm_params": arm_specific_params,
            }

            manifest_path = dest_dir / f"{fam}_{loss_arm}_transport_validation.json"
            manifest_path.write_text(json.dumps(manifest_data, indent=2))
            logger.info(
                "Successfully wrote transport validation JSON to %s", manifest_path
            )

            result_paths.append(oof_path)

    return result_paths


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    results = run_gbdt_loss_objectives(
        family=args.family,
        loss=args.loss,
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        smoke=args.smoke,
        dry_run=args.dry_run,
    )
    logger.info(
        "GBDT loss objectives completed successfully (%d OOF files generated).",
        len(results),
    )
    for p in results:
        logger.info("Artifact: %s", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
