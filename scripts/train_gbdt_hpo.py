#!/usr/bin/env python3
"""Run GBDT Hyperparameter Optimization (HPO) experiments.

Evaluates LightGBM and XGBoost in a 2-arm A/B experiment:
- Control: Default champion hyperparameters.
- Challenger: Optuna TPESampler searching hyperparameter space strictly on an inner
  building-grouped validation split carved from outer training data.

Outer validation partitions are never accessed during HPO trials, guaranteeing zero leakage.
Emits canonical 24-field CANDIDATE_OOF_SCHEMA parquets, transport validation JSON,
and manifest JSON recording best_params per fold.
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
    HPO_EXPERIMENT,
    MAX_BATCH_ROWS,
    ROW_COLUMNS,
    validate_candidate_oof,
    write_candidate_oof_batches,
)
from project.models.gbdt_hpo import (
    SUPPORTED_ARMS,
    SUPPORTED_FAMILIES,
    normalize_arm,
    normalize_family,
    train_hpo_partition,
    validate_feature_pool,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/experiments/gbdt_hpo_campaign.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/models/gbdt_hpo")


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


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GBDT Hyperparameter Optimization (Control vs. Challenger) cross-validation."
    )
    parser.add_argument(
        "--family",
        default="all",
        choices=[*SUPPORTED_FAMILIES, "all"],
        help="GBDT family to train: lightgbm, xgboost, or all.",
    )
    parser.add_argument(
        "--arm",
        default="both",
        choices=[*SUPPORTED_ARMS, "both", "all"],
        help="Experiment arm: control, challenger, both, or all.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna HPO trials (overrides config defaults).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke mode: 1 repeat, 2 folds, subset of buildings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute dry-run mode: deterministic mock predictions without full training.",
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


def run_gbdt_hpo(
    family: str,
    arm: str,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path | None = None,
    smoke: bool = False,
    dry_run: bool = False,
    n_trials: int | None = None,
) -> list[Path]:
    """Execute GBDT HPO cross-validation across families and arms.

    Returns
    -------
    list[Path]:
        List of generated OOF parquet artifact paths.
    """
    models_cfg = config.get("models", {})
    families_to_run = (
        list(SUPPORTED_FAMILIES) if family == "all" else [normalize_family(family)]
    )

    if arm in ("both", "all"):
        arms_to_run = list(SUPPORTED_ARMS)
    else:
        arms_to_run = [normalize_arm(arm)]

    target_col = config.get("target", "anomaly")
    split_path = Path(
        config.get(
            "split_manifest", "outputs/splits/building_folds_repeated.parquet"
        )
    )
    data_path = Path(
        config.get("data_path", "outputs/prepared/train_joined.parquet")
    )

    if not split_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    split_sha = _file_sha256(split_path)
    data_sha = _file_sha256(data_path)
    config_sha = _file_sha256(config_path)

    split_df = pl.read_parquet(split_path)
    train_joined = pl.read_parquet(data_path)

    features_cfg = config.get("features", {})
    if features_cfg.get("predictors"):
        declared_predictors = list(features_cfg["predictors"])
    elif "numeric" in features_cfg or "categorical" in features_cfg:
        declared_predictors = list(features_cfg.get("numeric", [])) + list(
            features_cfg.get("categorical", [])
        )
    else:
        raise ValueError(
            "Configuration must explicitly declare 'features.predictors' (or 'numeric'/'categorical') allowlist"
        )

    predictor_cols = validate_feature_pool(declared_predictors)

    # Ensure all declared predictors exist in train_joined
    missing_cols = [c for c in predictor_cols if c not in train_joined.columns]
    if missing_cols:
        raise ValueError(
            f"Declared predictors not found in training dataset: {missing_cols}"
        )

    # Assign row_index strictly to development population
    dev_splits = split_df.filter(pl.col("split") == "dev")
    dev_building_ids = dev_splits["building_id"].unique().to_list()

    train_dev = train_joined.filter(pl.col("building_id").is_in(dev_building_ids))
    train_dev = (
        train_dev.drop("row_index", strict=False)
        .sort(["building_id", "timestamp"])
        .with_row_index("row_index")
        .with_columns(pl.col("row_index").cast(pl.Int64))
    )

    is_synthetic = bool(smoke)
    execution_mode = "synthetic_smoke" if is_synthetic else "full"
    non_comparative = is_synthetic
    n03_eligible = not is_synthetic

    if is_synthetic:
        repeats = [0]
        folds = [0, 1]
        effective_n_trials = (
            n_trials if n_trials is not None else config.get("smoke_n_trials", 2)
        )
    else:
        repeats = sorted(dev_splits["repeat"].unique().to_list())
        folds = sorted(dev_splits["fold"].unique().to_list())
        effective_n_trials = (
            n_trials if n_trials is not None else config.get("n_trials", 15)
        )

    result_paths: list[Path] = []

    for fam in families_to_run:
        if fam not in models_cfg:
            raise ValueError(f"Family '{fam}' not configured in {config_path}")

        model_spec = models_cfg[fam]
        base_params = dict(model_spec.get("params", {}))
        recipe_id = model_spec.get("recipe_id", f"recipe_gbdt_hpo_{fam}")
        model_config_id = model_spec.get(
            "model_config_id", f"cfg_gbdt_hpo_{fam}_default"
        )

        for arm_name in arms_to_run:
            run_id = f"gbdt_hpo_{fam}_{arm_name}_{execution_mode}_{uuid.uuid4().hex[:8]}"
            canonical_control_fit_sha = _json_sha256(
                {"family": fam, "arm": "control", "params": base_params}
            )
            initial_fit_config_sha = _json_sha256(
                {"family": fam, "arm": arm_name, "params": base_params}
            )
            oof_dfs: list[pl.DataFrame] = []
            per_fold_params: dict[str, dict[str, Any]] = {}
            per_fold_fit_sha: dict[str, str] = {}

            logger.info(
                "Executing GBDT HPO: family=%s, arm=%s, mode=%s, trials=%d, repeats=%s, folds=%s",
                fam,
                arm_name,
                execution_mode,
                effective_n_trials,
                repeats,
                folds,
            )

            for rep in repeats:
                split_seed = 42 + rep
                for fld in folds:
                    fold_id = f"{rep}:{fld}"
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
                    partition_params["seed"] = split_seed

                    preds, final_fold_params = train_hpo_partition(
                        model_family=fam,
                        arm=arm_name,
                        train_df=train_fold_df,
                        val_df=val_fold_df,
                        feature_cols=predictor_cols,
                        target_col=target_col,
                        params=partition_params,
                        n_trials=effective_n_trials,
                        seed=split_seed,
                        dry_run=dry_run,
                    )

                    fold_fit_sha = _json_sha256(final_fold_params)
                    per_fold_params[fold_id] = final_fold_params
                    per_fold_fit_sha[fold_id] = fold_fit_sha
                    row_fit_sha = (
                        canonical_control_fit_sha
                        if arm_name == "control"
                        else fold_fit_sha
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
                        pl.lit(int(rep), dtype=pl.Int64).alias("repeat"),
                        pl.lit(int(fld), dtype=pl.Int64).alias("fold"),
                        pl.Series("prediction", preds, dtype=pl.Float64),
                        pl.lit(split_sha).alias("split_sha256"),
                        pl.lit(data_sha).alias("data_sha256"),
                        pl.lit(config_sha).alias("config_sha256"),
                        pl.lit(run_id).alias("run_id"),
                        pl.lit(int(split_seed), dtype=pl.Int64).alias("split_seed"),
                        pl.lit(config.get("experiment_id", HPO_EXPERIMENT)).alias("experiment_id"),
                        pl.lit(arm_name).alias("arm_id"),
                        pl.lit("positive_probability").alias("score_kind"),
                        pl.lit(1, dtype=pl.Int64).alias("positive_label"),
                        pl.lit(execution_mode).alias("execution_mode"),
                        pl.lit(non_comparative).alias("non_comparative"),
                        pl.lit(n03_eligible).alias("n03_eligible"),
                        pl.lit(row_fit_sha).alias("model_fit_config_sha256"),
                    ])
                    oof_dfs.append(fold_oof_df)

            if not oof_dfs:
                raise RuntimeError(
                    f"No predictions generated for {fam}_{arm_name}."
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
                base = (
                    DEFAULT_OUTPUT_DIR / "smoke"
                    if smoke
                    else DEFAULT_OUTPUT_DIR
                )
                resolved_output_dir = base.resolve()

            dest_dir = (
                resolved_output_dir / execution_mode / f"{fam}_{arm_name}_{run_id}"
            ).resolve()
            dest_dir.mkdir(parents=True, exist_ok=True)
            oof_path = (
                dest_dir / f"{fam}_{arm_name}_candidate_oof.parquet"
            ).resolve()

            # Stream out in batches conforming to MAX_BATCH_ROWS
            batches = [
                arrow_table.slice(
                    offset, min(MAX_BATCH_ROWS, arrow_table.num_rows - offset)
                )
                for offset in range(0, arrow_table.num_rows, MAX_BATCH_ROWS)
            ]
            write_candidate_oof_batches(oof_path, batches, CANDIDATE_OOF_SCHEMA)

            # Build canonical candidate v1 manifest and validate OOF
            cand_id = f"{fam}_{arm_name}"
            cand_meta: dict[str, Any] = {
                "recipe_id": f"{recipe_id}_{arm_name}",
                "model_family": fam,
                "model_config_id": f"{model_config_id}_{arm_name}",
                "experiment_id": config.get("experiment_id", HPO_EXPERIMENT),
                "arm_id": arm_name,
                "score_kind": "positive_probability",
                "positive_label": 1,
            }
            if arm_name == "challenger":
                cand_meta["fold_fit_config_sha256"] = per_fold_fit_sha
            else:
                cand_meta["model_fit_config_sha256"] = canonical_control_fit_sha

            canonical_manifest = {
                "schema_version": 1,
                "source_notebook": "notebooks/02_baseline_and_candidate_design.ipynb",
                "source_cell_id": "train_gbdt_hpo",
                "vault_row_indices": [],
                "vault_building_ids": [],
                "protocol": {
                    "repeat_ids": repeats,
                    "fold_ids": folds,
                    "split_seeds": {str(r): 42 + int(r) for r in repeats},
                },
                "candidates": {cand_id: cand_meta},
                "split_sha256": split_sha,
                "data_sha256": data_sha,
                "config_sha256": config_sha,
                "run_id": run_id,
                "execution_mode": execution_mode,
                "non_comparative": non_comparative,
                "n03_eligible": n03_eligible,
            }

            expected_rows = oof_df.select(list(ROW_COLUMNS)).with_columns([
                pl.col("row_index").cast(pl.Int64),
                pl.col("repeat").cast(pl.Int64),
                pl.col("fold").cast(pl.Int64),
                pl.col("split_seed").cast(pl.Int64),
                pl.col("target").cast(pl.Int64),
                pl.col("building_id").cast(pl.Utf8),
                pl.col("timestamp").cast(pl.Datetime("ns")),
            ])
            validation_report = validate_candidate_oof(
                pl.read_parquet(oof_path),
                expected_rows,
                canonical_manifest,
            )

            manifest_data = {
                "schema_version": 1,
                "run_id": run_id,
                "family": fam,
                "arm": arm_name,
                "execution_mode": execution_mode,
                "non_comparative": non_comparative,
                "n03_eligible": n03_eligible,
                "records_count": arrow_table.num_rows,
                "split_sha256": split_sha,
                "data_sha256": data_sha,
                "config_sha256": config_sha,
                "model_fit_config_sha256": (
                    canonical_control_fit_sha
                    if arm_name == "control"
                    else initial_fit_config_sha
                ),
                "fold_fit_config_sha256": per_fold_fit_sha,
                "best_params_per_fold": per_fold_params,
                "n_trials": effective_n_trials,
                "validation_status": "passed",
                "oof_parquet": str(oof_path),
                "canonical_manifest": canonical_manifest,
                "validation_report": validation_report,
            }

            manifest_path = dest_dir / f"{fam}_{arm_name}_manifest.json"
            manifest_path.write_text(json.dumps(manifest_data, indent=2))
            logger.info("Successfully wrote manifest JSON to %s", manifest_path)

            transport_path = (
                dest_dir / f"{fam}_{arm_name}_transport_validation.json"
            )
            transport_path.write_text(json.dumps(manifest_data, indent=2))
            logger.info(
                "Successfully wrote transport validation JSON to %s", transport_path
            )

            result_paths.append(oof_path)

    return result_paths


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    results = run_gbdt_hpo(
        family=args.family,
        arm=args.arm,
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        smoke=args.smoke,
        dry_run=args.dry_run,
        n_trials=args.n_trials,
    )
    logger.info(
        "GBDT HPO completed successfully (%d OOF files generated).",
        len(results),
    )
    for p in results:
        logger.info("Artifact: %s", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
