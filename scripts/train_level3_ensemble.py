#!/usr/bin/env python3
"""Level-3 Meta-Ensemble Training and Submission Runner.

Combines:
- Arm A (Meta-2A): Probability-based Meta-Learner (using existing challenger_lr OOF).
- Arm B (Meta-2B): Representation-based Tree Leaf Embeddings (LightGBM leaf extractor -> OneHotEncoder -> sparse SGDClassifier log-loss).
- Level-3 Meta-Learner: Nested direct ROC-AUC bounded combination of Meta-2A and Meta-2B scores:
  S_L3 = w * S_2A + (1 - w) * S_2B, where w in [0.0, 1.0].

Supports:
- --smoke: 1 repeat, 2 folds on a subset of buildings.
- --dry-run: instantaneous mock training without heavy compute.
- full mode: 3 repeats, 4 folds over all development data.
- --submit: upload final validated submission to Kaggle via CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from sklearn.metrics import roc_auc_score

from project.models.candidate_artifacts import (
    CANDIDATE_OOF_SCHEMA,
    MAX_BATCH_ROWS,
    ROW_COLUMNS,
    validate_candidate_oof,
    write_candidate_oof_batches,
)
from project.models.level3_ensemble import (
    Meta2BLeafEmbeddingEstimator,
    build_level3_submission,
    run_level3_nested_cv,
)
from project.submission import save_submission

logger = logging.getLogger("train_level3_ensemble")

DEFAULT_CONFIG_PATH = Path("configs/experiments/level3_ensemble.yaml")
DEFAULT_FULL_OUTPUT_DIR = Path("outputs/models/level3_ensemble/full")
DEFAULT_SMOKE_OUTPUT_DIR = Path("outputs/models/level3_ensemble/smoke")


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


def submit_to_kaggle(file_path: Path, message: str) -> bool:
    """Submit a validated CSV to Kaggle via the CLI."""
    kaggle_bin = shutil.which("kaggle") or "kaggle"
    cmd = [
        str(kaggle_bin),
        "competitions",
        "submit",
        "-c",
        "energy-anomaly-detection",
        "-f",
        str(file_path),
        "-m",
        message,
    ]
    logger.info("Executing Kaggle CLI submit: %s", " ".join(cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            logger.info("Kaggle CLI success: %s", res.stdout.strip())
            return True
        logger.warning("Kaggle CLI warning/error: %s | stderr: %s", res.stdout, res.stderr)
        return False
    except OSError as e:
        logger.error("Failed to run kaggle CLI: %s", e)
        return False


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the Level-3 ensemble runner."""
    parser = argparse.ArgumentParser(
        description="Train Level-3 Meta-Ensemble and generate Kaggle submissions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="Custom output directory for models and OOF parquets.",
    )
    parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=Path("outputs/submissions"),
        help="Directory to save submission CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute in fast dry-run mode without intensive fitting.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Execute in fast smoke mode on 1 repeat, 2 folds, subset of buildings.",
    )
    parser.add_argument(
        "--skip-submission",
        action="store_true",
        help="Skip test prediction and Kaggle submission generation.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Automatically submit generated submission CSV to Kaggle via CLI.",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="Level-3 Meta-Ensemble (Arm A + Arm B Leaf Embeddings)",
        help="Kaggle submission description message.",
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


def run_training_pipeline(
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path | None = None,
    submissions_dir: Path = Path("outputs/submissions"),
    dry_run: bool = False,
    smoke: bool = False,
    skip_submission: bool = False,
    submit: bool = False,
    submission_message: str = "Level-3 Meta-Ensemble",
) -> tuple[Path, Path | None]:
    """Execute the Level-3 nested cross-validation and optional test submission pipeline."""
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

    # 1. Load data and split manifests
    split_manifest_path = Path(config.get("split_manifest", "outputs/splits/building_folds_repeated.parquet"))
    data_path = Path(config.get("data_path", "outputs/prepared/train_joined.parquet"))

    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    split_sha = _file_sha256(split_manifest_path)
    data_sha = _file_sha256(data_path)

    split_df = pl.read_parquet(split_manifest_path)
    dev_splits = split_df.filter(pl.col("split") == "dev")

    # Determine repeats and folds
    if is_synthetic:
        repeats = [0]
        folds = [0, 1]
        split_seeds = {"0": 42}
        # In smoke/dry-run mode, restrict to a small subset of buildings (2 per fold)
        # so that independent expected_rows and oof_df have identical full coverage
        smoke_b_f0 = dev_splits.filter((pl.col("repeat") == 0) & (pl.col("fold") == 0))["building_id"].unique().to_list()[:2]
        smoke_b_f1 = dev_splits.filter((pl.col("repeat") == 0) & (pl.col("fold") == 1))["building_id"].unique().to_list()[:2]
        smoke_buildings = smoke_b_f0 + smoke_b_f1
        dev_splits = dev_splits.filter(
            (pl.col("repeat") == 0)
            & (pl.col("fold").is_in([0, 1]))
            & pl.col("building_id").is_in(smoke_buildings)
        )
        dev_building_ids = smoke_buildings
    else:
        repeats = config.get("repeats", [0, 1, 2])
        folds = config.get("folds", [0, 1, 2, 3])
        split_seeds = {
            str(r): config.get("split_seeds", {}).get(str(r), 42 + int(r))
            for r in repeats
        }
        dev_building_ids = dev_splits["building_id"].unique().to_list()

    logger.info("Loading training data from %s...", data_path)
    train_joined = pl.read_parquet(data_path)

    # Strictly assign canonical row_index to development partition
    train_dev = (
        train_joined.filter(pl.col("building_id").is_in(dev_building_ids))
        .drop("row_index", strict=False)
        .sort(["building_id", "timestamp"])
        .with_row_index("row_index")
        .with_columns(pl.col("row_index").cast(pl.Int64))
    )
    logger.info("Prepared development partition: %d rows across %d buildings.", train_dev.height, len(dev_building_ids))

    # 2. Load Arm A (Meta-2A) out-of-fold predictions
    meta_2a_cfg = config.get("meta_2a", {})
    meta_2a_path = Path(meta_2a_cfg.get("oof_path", ""))
    if not meta_2a_path.is_file():
        raise FileNotFoundError(f"Arm A (Meta-2A) OOF artifact not found: {meta_2a_path}")

    logger.info("Loading Meta-2A OOF predictions from %s...", meta_2a_path)
    meta_2a_oof_df = pl.read_parquet(meta_2a_path)
    # Extract feature configurations
    features_cfg = config.get("features", {})
    numeric_features = list(features_cfg.get("numeric", []))
    categorical_features = list(features_cfg.get("categorical", ["primary_use"]))
    feature_cols = numeric_features + categorical_features
    target_col = config.get("target", "anomaly")

    meta_2b_cfg = config.get("meta_2b", {})
    level3_cfg = config.get("level3", {})

    recipe_id = level3_cfg.get("recipe_id", "recipe_level3_auc_combiner")
    model_config_id = level3_cfg.get("model_config_id", "cfg_level3_auc_bounded")
    run_id = f"level3_meta_{execution_mode}_{uuid.uuid4().hex[:8]}"

    logger.info(
        "Starting Level-3 Nested CV: mode=%s, repeats=%s, folds=%s, run_id=%s",
        execution_mode,
        repeats,
        folds,
        run_id,
    )

    # 3. Execute Nested Cross-Validation
    oof_df, metadata = run_level3_nested_cv(
        train_dev=train_dev,
        dev_splits=dev_splits,
        meta_2a_oof_df=meta_2a_oof_df,
        feature_cols=feature_cols,
        target_col=target_col,
        categorical_cols=categorical_features,
        repeats=repeats,
        folds=folds,
        meta_2b_params=meta_2b_cfg,
        level3_params=level3_cfg.get("params", {}),
        execution_mode=execution_mode,
        recipe_id=recipe_id,
        model_config_id=model_config_id,
        experiment_id=config.get("experiment_id", "level3_ensemble"),
        run_id=run_id,
        config_sha256=config_sha,
        dry_run=dry_run,
        smoke=smoke,
    )

    overall_auc = roc_auc_score(oof_df["target"].to_numpy(), oof_df["prediction"].to_numpy())
    logger.info("Level-3 CV Completed. Out-of-Fold ROC-AUC: %.5f | Mean Weight: %.4f", overall_auc, metadata["mean_weight"])
    logger.info("Per-fold Level-3 weights: %s", metadata["fold_weights"])

    # 4. Save OOF Parquet Artifact
    dest_dir = (base_output_dir / f"level3_{run_id}").resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    oof_path = (dest_dir / "level3_candidate_oof.parquet").resolve()

    arrow_table = (
        oof_df.select(CANDIDATE_OOF_SCHEMA.names)
        .to_arrow()
        .cast(CANDIDATE_OOF_SCHEMA)
    )
    batches = [
        arrow_table.slice(offset, min(MAX_BATCH_ROWS, arrow_table.num_rows - offset))
        for offset in range(0, arrow_table.num_rows, MAX_BATCH_ROWS)
    ]
    write_candidate_oof_batches(oof_path, batches, CANDIDATE_OOF_SCHEMA)
    logger.info("Successfully wrote Level-3 OOF parquet (%d rows) to %s", arrow_table.num_rows, oof_path)

    # 5. Build and Save Manifest
    fit_hash = metadata["fit_config_sha256"]
    protocol_repeats = [int(r) for r in repeats]
    protocol_folds = [int(f) for f in folds]
    manifest = {
        "schema_version": 1,
        "source_notebook": "notebooks/02_baseline_and_candidate_design.ipynb",
        "source_cell_id": "Level-3 Meta-Ensemble Orchestrator",
        "vault_row_indices": [],
        "vault_building_ids": [],
        "protocol": {
            "repeat_ids": protocol_repeats,
            "fold_ids": protocol_folds,
            "split_seeds": split_seeds,
        },
        "candidates": {
            "level3_ensemble": {
                "recipe_id": recipe_id,
                "model_family": "ensemble_level3",
                "model_config_id": model_config_id,
                "experiment_id": config.get("experiment_id", "level3_ensemble"),
                "arm_id": "level3_meta",
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

    manifest_path = dest_dir / "level3_candidate_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 6. Finding 4: Independently construct expected_rows by joining train_dev with repeated split manifest
    active_splits = dev_splits.filter(
        pl.col("repeat").is_in(repeats) & pl.col("fold").is_in(folds)
    ).select([
        pl.col("building_id").cast(pl.String),
        pl.col("repeat").cast(pl.Int64),
        pl.col("fold").cast(pl.Int64),
    ])

    expected_base = (
        train_dev.select([
            pl.col("row_index").cast(pl.Int64),
            pl.col("building_id").cast(pl.String),
            pl.col("timestamp"),
            pl.col(target_col).cast(pl.Int64).alias("target"),
        ])
        .join(active_splits, on="building_id", how="inner")
    )

    seed_map = {int(r): int(split_seeds[str(r)]) for r in repeats}
    seed_expr = pl.col("repeat").replace(seed_map).cast(pl.Int64)

    expected_rows = (
        expected_base.with_columns([
            seed_expr.alias("split_seed"),
            pl.lit(split_sha).alias("split_sha256"),
            pl.lit(data_sha).alias("data_sha256"),
            pl.lit(config_sha).alias("config_sha256"),
            pl.lit(run_id).alias("run_id"),
        ])
        .select(list(ROW_COLUMNS))
        .sort(["repeat", "fold", "building_id", "timestamp"])
    )
    validated = validate_candidate_oof(oof_df, expected_rows, manifest)
    logger.info("OOF artifact successfully validated: %s records", validated["oof_record_count"])

    # 7. Generate Submission if not skipped
    sub_path: Path | None = None
    if not skip_submission:
        sub_dir = Path(submissions_dir).resolve()
        if is_synthetic:
            sub_dir = sub_dir / "smoke"
        sub_dir.mkdir(parents=True, exist_ok=True)

        sample_sub_path = Path(config.get("sample_submission_path", "data/energy-anomaly-detection/sample_submission.csv"))
        test_path = Path(config.get("test_data_path", "outputs/prepared/test_joined.parquet"))

        logger.info("Loading test features from %s...", test_path)
        test_df = pl.read_parquet(test_path)
        if is_synthetic:
            test_df = test_df.head(1000)

        logger.info("Loading sample submission from %s...", sample_sub_path)
        sample_sub = pl.read_csv(sample_sub_path)
        if is_synthetic:
            sample_sub = sample_sub.head(test_df.height)

        # Arm A (Meta-2A) test probabilities (Finding 3: Fail-closed if missing in full mode)
        s2a_test_path = Path(meta_2a_cfg.get("test_submission_path", "outputs/submissions/submission_01_ensemble_logit_lr_0.9008.csv"))
        if not s2a_test_path.is_file():
            if not is_synthetic:
                raise FileNotFoundError(
                    f"Meta-2A test submission not found: {s2a_test_path}. "
                    "Full execution requires genuine Meta-2A test probabilities."
                )
            logger.warning("Meta-2A test submission not found at %s. Using synthetic baseline.", s2a_test_path)
            rng_test = np.random.default_rng(42)
            s2a_sub = sample_sub.select([pl.col("row_id").cast(pl.Int64)]).with_columns(
                pl.Series("s_2a", rng_test.uniform(0.01, 0.05, size=sample_sub.height), dtype=pl.Float64)
            )
        else:
            logger.info("Loading Meta-2A test probabilities from %s...", s2a_test_path)
            s2a_sub = pl.read_csv(s2a_test_path).select([
                pl.col("row_id").cast(pl.Int64),
                pl.col("anomaly").cast(pl.Float64).alias("s_2a"),
            ])

        # Arm B (Meta-2B) test probabilities
        logger.info("Generating Meta-2B leaf embedding predictions for test set (%d rows)...", test_df.height)
        X_train_full = train_dev.select(list(feature_cols)).to_pandas()
        y_train_full = train_dev[target_col].to_numpy()
        X_test = test_df.select(list(feature_cols)).to_pandas()

        if dry_run:
            m2b_estimator = Meta2BLeafEmbeddingEstimator(dry_run=True)
            m2b_estimator.fit(X_train_full, y_train_full)
            s_2b_test = m2b_estimator.predict_proba(X_test)
        else:
            lgb_p = dict(meta_2b_cfg.get("lgb_params", {}))
            if smoke:
                lgb_p["n_estimators"] = 20
            m2b_estimator = Meta2BLeafEmbeddingEstimator(
                lgb_params=lgb_p,
                encoder_params=meta_2b_cfg.get("encoder_params"),
                linear_params=meta_2b_cfg.get("linear_params"),
                categorical_cols=categorical_features,
            )
            m2b_estimator.fit(X_train_full, y_train_full)
            s_2b_test = m2b_estimator.predict_proba(X_test)

        m2b_pred_df = pl.DataFrame({
            "row_id": test_df["row_id"].cast(pl.Int64),
            "s_2b": pl.Series("s_2b", s_2b_test, dtype=pl.Float64),
        })

        # Finding 3: Join both Meta-2A and Meta-2B explicitly on row_id to sample_submission.csv
        aligned_test = (
            sample_sub.select([pl.col("row_id").cast(pl.Int64)])
            .join(s2a_sub, on="row_id", how="left")
            .join(m2b_pred_df, on="row_id", how="left")
        )

        if not is_synthetic:
            if aligned_test["s_2a"].null_count() > 0:
                raise ValueError(
                    f"Meta-2A test submission missing {aligned_test['s_2a'].null_count()} row_ids present in sample_submission.csv"
                )
            if aligned_test["s_2b"].null_count() > 0:
                raise ValueError(
                    f"Meta-2B test predictions missing {aligned_test['s_2b'].null_count()} row_ids present in sample_submission.csv"
                )
        else:
            aligned_test = aligned_test.with_columns([
                pl.col("s_2a").fill_null(0.03),
                pl.col("s_2b").fill_null(0.03),
            ])

        row_ids = aligned_test["row_id"].to_numpy()
        s_2a_aligned = aligned_test["s_2a"].to_numpy()
        s_2b_aligned = aligned_test["s_2b"].to_numpy()

        # Level-3 Blend using learned mean weight
        mean_w = metadata["mean_weight"]
        logger.info("Blending Arm A and Arm B test predictions with weight w=%.4f...", mean_w)
        sub_df = build_level3_submission(
            row_ids=row_ids,
            s_2a=s_2a_aligned,
            s_2b=s_2b_aligned,
            weight=mean_w,
        )

        sub_filename = f"submission_38_level3_meta_ensemble_{overall_auc:.4f}.csv"
        target_csv = sub_dir / sub_filename
        save_submission(sub_df, target_csv, expected_row_count=len(row_ids))
        sub_path = target_csv
        logger.info("Successfully generated Kaggle submission CSV at %s", target_csv)
        if submit and not is_synthetic:
            msg = f"{submission_message} (CV ROC-AUC: {overall_auc:.4f}, w: {mean_w:.3f})"
            logger.info("Submitting %s to Kaggle: %s", target_csv, msg)
            submit_to_kaggle(target_csv, msg)

    return oof_path, sub_path


def main() -> None:
    """CLI entrypoint for training the Level-3 Meta-Ensemble."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Loading Level-3 configuration from %s...", args.config)
    config = load_config(args.config)

    oof_path, sub_path = run_training_pipeline(
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        submissions_dir=args.submissions_dir,
        dry_run=args.dry_run,
        smoke=args.smoke,
        skip_submission=args.skip_submission,
        submit=args.submit,
        submission_message=args.message,
    )

    logger.info("Execution completed successfully.")
    logger.info("OOF Parquet: %s", oof_path)
    if sub_path:
        logger.info("Submission CSV: %s", sub_path)


if __name__ == "__main__":
    main()
