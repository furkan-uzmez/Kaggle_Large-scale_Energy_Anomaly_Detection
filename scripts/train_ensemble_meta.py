#!/usr/bin/env python3
"""Level-2 Meta-Learner Tournament Training Runner.

Combines 6 diverse base model out-of-fold predictions using nested cross-validation
across 5 distinct meta-model architectures:
1. control: Caruana Greedy Forward Selection with Replacement
2. challenger_lr: L2-Regularized Logistic Regression on logits
3. challenger_nnls: Non-Negative Least Squares constrained weights
4. challenger_auc: Direct ROC-AUC rank optimizer via Nelder-Mead
5. challenger_shallow_lgb: Shallow LightGBM meta-learner

Nested CV Fitting Rules:
- For repeat r and outer fold k, the meta-model trains ONLY on repeat r rows where fold != k.
- Predicts strictly on repeat r fold k rows.
- No cross-repeat pooling or cross-fold leakage is permitted.
- OOF predictions are validated against the authoritative canonical candidate v1 schema.
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
    MAX_BATCH_ROWS,
    ROW_COLUMNS,
    validate_candidate_oof,
    write_candidate_oof_batches,
)
from project.models.ensemble_meta_learning import (
    SUPPORTED_META_ARMS,
    load_base_candidate_predictions,
    run_meta_nested_cv,
)

logger = logging.getLogger("train_ensemble_meta")

DEFAULT_CONFIG_PATH = Path("configs/experiments/ensemble_meta_tournament.yaml")
DEFAULT_SMOKE_OUTPUT_DIR = Path("outputs/models/ensemble_meta/smoke")
DEFAULT_FULL_OUTPUT_DIR = Path("outputs/models/ensemble_meta/full")


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
    """Parse command line arguments for the meta-learner tournament runner."""
    parser = argparse.ArgumentParser(
        description="Train Level-2 meta-learner models over base candidate predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="all",
        choices=[*SUPPORTED_META_ARMS, "all"],
        help="Meta-model arm to evaluate or 'all' to evaluate all 5 arms.",
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
        help="Custom output directory for predictions and manifests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute in fast dry-run synthetic smoke mode.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Execute in smoke mode.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
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


def run_tournament(
    arm: str,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
    smoke: bool = False,
) -> list[Path]:
    """Execute the Level-2 meta-learner tournament across requested arms."""
    is_synthetic = bool(dry_run or smoke)
    execution_mode = "synthetic_smoke" if is_synthetic else "full"
    non_comparative = is_synthetic
    n03_eligible = not is_synthetic

    if output_dir is not None:
        base_output_dir = Path(output_dir).resolve()
    else:
        base_output_dir = (
            DEFAULT_SMOKE_OUTPUT_DIR if is_synthetic else DEFAULT_FULL_OUTPUT_DIR
        ).resolve()

    base_output_dir.mkdir(parents=True, exist_ok=True)
    config_sha = _file_sha256(config_path)

    # Determine repeats and folds
    if is_synthetic:
        repeats = [0]
        folds = [0, 1]
        split_seeds = {"0": 42}
    else:
        repeats = config.get("repeats", [0, 1, 2])
        folds = config.get("folds", [0, 1, 2, 3])
        split_seeds = {
            str(r): config.get("split_seeds", {}).get(str(r), 42 + int(r))
            for r in repeats
        }

    # Load and horizontally merge all base candidate predictions
    candidate_specs = config.get("base_candidates")
    logger.info(
        "Loading base candidate predictions (%d candidates)...",
        len(candidate_specs or []),
    )
    merged_data, prediction_cols = load_base_candidate_predictions(candidate_specs)
    logger.info(
        "Successfully assembled base predictions: %d total rows, features: %s",
        merged_data.height,
        prediction_cols,
    )

    # Filter data for fast dry-run / smoke if requested
    if is_synthetic:
        # Select repeat 0, folds 0 and 1, taking first 2 buildings from each fold
        rep0_df = merged_data.filter(
            (pl.col("repeat") == 0) & (pl.col("fold").is_in([0, 1]))
        )
        b_f0 = (
            rep0_df.filter(pl.col("fold") == 0)["building_id"]
            .unique()
            .sort()
            .slice(0, 2)
            .to_list()
        )
        b_f1 = (
            rep0_df.filter(pl.col("fold") == 1)["building_id"]
            .unique()
            .sort()
            .slice(0, 2)
            .to_list()
        )
        data_to_fit = rep0_df.filter(pl.col("building_id").is_in(b_f0 + b_f1)).sort(
            ["building_id", "timestamp"]
        )
        logger.info(
            "Synthetic smoke mode enabled: %d rows selected across 4 buildings (folds %s)",
            data_to_fit.height,
            folds,
        )
    else:
        data_to_fit = merged_data

    split_sha = str(data_to_fit["split_sha256"][0])
    data_sha = str(data_to_fit["data_sha256"][0])

    # Determine arms to run
    if arm == "all":
        arms_to_run = list(SUPPORTED_META_ARMS)
    else:
        arm_norm = arm.strip().lower()
        if arm_norm not in SUPPORTED_META_ARMS:
            raise ValueError(
                f"Unsupported arm '{arm}'. Supported: {list(SUPPORTED_META_ARMS)}"
            )
        arms_to_run = [arm_norm]

    result_paths: list[Path] = []
    meta_arms_config = config.get("meta_arms", {})

    for arm_name in arms_to_run:
        arm_meta_cfg = meta_arms_config.get(arm_name, {})
        recipe_id = arm_meta_cfg.get("recipe_id", f"recipe_meta_{arm_name}")
        model_config_id = arm_meta_cfg.get("model_config_id", f"cfg_meta_{arm_name}")
        params = arm_meta_cfg.get("params", {})
        cand_id = f"meta_{arm_name}"
        run_id = f"ensemble_{arm_name}_{execution_mode}_{uuid.uuid4().hex[:8]}"

        logger.info(
            "Fitting Level-2 meta-learner: arm=%s, recipe=%s, run_id=%s, mode=%s",
            arm_name,
            recipe_id,
            run_id,
            execution_mode,
        )

        oof_df = run_meta_nested_cv(
            data=data_to_fit,
            prediction_cols=prediction_cols,
            arm_id=arm_name,
            meta_learner_params=params,
            repeats=repeats,
            folds=folds,
            run_id=run_id,
            execution_mode=execution_mode,
            non_comparative=non_comparative,
            n03_eligible=n03_eligible,
            config_sha256=config_sha,
            recipe_id=recipe_id,
            model_config_id=model_config_id,
            experiment_id=config.get("experiment_id", "ensemble_meta_tournament"),
        )

        dest_dir = (base_output_dir / f"{arm_name}_{run_id}").resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
        oof_path = (dest_dir / f"{arm_name}_candidate_oof.parquet").resolve()

        # Stream out in Arrow batches conforming to MAX_BATCH_ROWS
        arrow_table = (
            oof_df.select(CANDIDATE_OOF_SCHEMA.names)
            .to_arrow()
            .cast(CANDIDATE_OOF_SCHEMA)
        )
        batches = [
            arrow_table.slice(
                offset, min(MAX_BATCH_ROWS, arrow_table.num_rows - offset)
            )
            for offset in range(0, arrow_table.num_rows, MAX_BATCH_ROWS)
        ]
        write_candidate_oof_batches(oof_path, batches, CANDIDATE_OOF_SCHEMA)
        logger.info(
            "Successfully wrote OOF parquet (%d rows) to %s",
            arrow_table.num_rows,
            oof_path,
        )

        # Build canonical manifest and validate OOF artifact
        fit_hash = str(oof_df["model_fit_config_sha256"][0])
        canonical_manifest = {
            "schema_version": 1,
            "source_notebook": "notebooks/02_baseline_and_candidate_design.ipynb",
            "source_cell_id": "train_ensemble_meta",
            "vault_row_indices": [],
            "vault_building_ids": [],
            "protocol": {
                "repeat_ids": repeats,
                "fold_ids": folds,
                "split_seeds": split_seeds,
            },
            "candidates": {
                cand_id: {
                    "recipe_id": recipe_id,
                    "model_family": "ensemble_meta",
                    "model_config_id": model_config_id,
                    "experiment_id": config.get(
                        "experiment_id", "ensemble_meta_tournament"
                    ),
                    "arm_id": arm_name,
                    "score_kind": "positive_probability",
                    "positive_label": 1,
                    "model_fit_config_sha256": fit_hash,
                }
            },
            "split_sha256": split_sha,
            "data_sha256": data_sha,
            "config_sha256": config_sha,
            "run_id": run_id,
            "execution_mode": execution_mode,
            "non_comparative": non_comparative,
            "n03_eligible": n03_eligible,
        }

        expected_rows = oof_df.select(list(ROW_COLUMNS)).with_columns(
            [
                pl.col("row_index").cast(pl.Int64),
                pl.col("repeat").cast(pl.Int64),
                pl.col("fold").cast(pl.Int64),
                pl.col("split_seed").cast(pl.Int64),
                pl.col("target").cast(pl.Int64),
                pl.col("building_id").cast(pl.Utf8),
                pl.col("timestamp").cast(pl.Datetime("ns")),
            ]
        )

        validation_report = validate_candidate_oof(
            pl.read_parquet(oof_path),
            expected_rows,
            canonical_manifest,
        )
        logger.info(
            "OOF validation passed for %s: status=%s, mean_roc_auc=%.4f",
            cand_id,
            validation_report["status"],
            validation_report["candidate_metrics"][cand_id]["mean_repeat_metrics"][
                "roc_auc"
            ]
            or 0.0,
        )

        # Write manifest file
        manifest_data = {
            "schema_version": 1,
            "run_id": run_id,
            "candidate_id": cand_id,
            "arm": arm_name,
            "recipe_id": recipe_id,
            "model_family": "ensemble_meta",
            "execution_mode": execution_mode,
            "non_comparative": non_comparative,
            "n03_eligible": n03_eligible,
            "record_count": arrow_table.num_rows,
            "split_sha256": split_sha,
            "data_sha256": data_sha,
            "config_sha256": config_sha,
            "model_fit_config_sha256": fit_hash,
            "oof_parquet": str(oof_path),
            "canonical_manifest": canonical_manifest,
            "validation_report": validation_report,
        }
        manifest_path = dest_dir / f"{arm_name}_manifest.json"
        manifest_path.write_text(json.dumps(manifest_data, indent=2))
        logger.info("Successfully wrote manifest to %s", manifest_path)
        result_paths.append(oof_path)

    return result_paths


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config(args.config)
    run_tournament(
        arm=args.arm,
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        smoke=args.smoke,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
