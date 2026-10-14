#!/usr/bin/env python3
"""Execute the three follow-up experiments recommended in Notebook 04 Section 8.

Experiments:
- EXP-F01: Diurnal-Weighted Manifold Mixup (concentrates synthetic interpolation on 00:00-04:00 UTC and evening transitions)
- EXP-F02: Platt-Calibrated Champion (nested zero-leakage 1D logistic calibration on inner-OOF logits of champion)
- EXP-F03: Skip-Connection Leaf Stacking (combines standardized raw continuous features with GBDT leaf embeddings)

Outputs canonical OOF parquets conforming to CANDIDATE_OOF_SCHEMA (24 columns),
transport validation JSON records, and competition test submission CSV files when requested.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from project.models.candidate_artifacts import (
    CANDIDATE_OOF_SCHEMA,
    write_candidate_oof_batches,
)
from project.models.followup_experiments import (
    DEFAULT_ALPHA_BASE,
    DEFAULT_ALPHA_TRANS,
    DEFAULT_TARGET_RATIO,
    DEFAULT_TRANS_WEIGHT,
    TRANSITION_HOURS_DEFAULT,
    run_nested_platt_calibration_partition,
    train_diurnal_mixup_partition,
    train_skip_leaf_partition,
)
from project.submission import (
    build_submission,
    save_submission,
    validate_submission_contract,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_followup_experiments")

DEFAULT_CONFIG_PATH = Path("configs/experiments/followup_experiments.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/models/followup_experiments")
SUBMISSIONS_OUTPUT_DIR = Path("outputs/submissions/all_experiments")
SMOKE_SUBMISSIONS_OUTPUT_DIR = Path("outputs/submissions/smoke")
MAX_BATCH_ROWS = 64_000

VALID_EXPERIMENTS = ("all", "exp_f01", "exp_f02", "exp_f03")


def _file_sha256(path: Path) -> str:
    """Hex SHA-256 of file content, or hashed path string if missing."""
    if not path.is_file():
        return hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(obj: Any) -> str:
    """Hex SHA-256 of canonical JSON."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Execute post-ensemble follow-up experiments (EXP-F01, EXP-F02, EXP-F03).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=list(VALID_EXPERIMENTS),
        help="Experiment to run: all, exp_f01, exp_f02, or exp_f03.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke mode: 1 repeat, 2 folds, subset of buildings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with deterministic mock scoring (fast dry-run verification).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Generate test submission CSV files conforming to competition contract.",
    )
    parser.add_argument(
        "--kaggle-submit",
        action="store_true",
        help="Upload generated submission CSV files to Kaggle via CLI.",
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
        help="Custom output directory for model predictions and manifests.",
    )
    parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=None,
        help="Custom output directory for generated submission files.",
    )
    return parser.parse_args(args)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError("Configuration must be a dictionary")
    return cfg


def run_followup_experiments(
    experiment_key: str,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path | None = None,
    submissions_dir: Path | None = None,
    smoke: bool = False,
    dry_run: bool = False,
    submit: bool = False,
    kaggle_submit: bool = False,
    split_df: pl.DataFrame | None = None,
    train_df: pl.DataFrame | None = None,
    test_df: pl.DataFrame | None = None,
    sample_sub_df: pl.DataFrame | None = None,
) -> dict[str, list[Path]]:
    """Execute follow-up cross-validation and optional test submission generation.

    Returns mapping with 'oof_paths' and 'submission_paths'.
    """
    selected_keys = (
        ["exp_f01", "exp_f02", "exp_f03"]
        if experiment_key == "all"
        else [experiment_key]
    )

    split_path = Path(
        config.get("split_manifest", "outputs/splits/building_folds_repeated.parquet")
    )
    data_path = Path(config.get("data_path", "outputs/prepared/train_joined.parquet"))
    test_path = Path(config.get("test_path", "outputs/prepared/test_joined.parquet"))
    sample_sub_path = Path(
        config.get(
            "sample_submission_path",
            "data/energy-anomaly-detection/sample_submission.csv",
        )
    )

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

    features_cfg = config.get("features", {})
    predictor_cols = list(features_cfg.get("predictors", []))
    categorical_cols = list(features_cfg.get("categorical", []))
    target_col = config.get("target", "anomaly")

    if not predictor_cols:
        raise ValueError("Configuration must declare features.predictors allowlist")

    dev_splits = split_df.filter(pl.col("split") == "dev")
    dev_building_ids = dev_splits["building_id"].unique().to_list()

    train_dev = (
        train_joined.filter(pl.col("building_id").is_in(dev_building_ids))
        .drop("row_index", strict=False)
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

    if output_dir is not None:
        output_root = Path(output_dir).resolve()
    else:
        output_root = (
            DEFAULT_OUTPUT_DIR / "smoke" if smoke else DEFAULT_OUTPUT_DIR
        ).resolve()

    if submissions_dir is not None:
        submissions_root = Path(submissions_dir).resolve()
    else:
        submissions_root = (
            SMOKE_SUBMISSIONS_OUTPUT_DIR if smoke else SUBMISSIONS_OUTPUT_DIR
        ).resolve()
    experiments_cfg = config.get("experiments", {})
    oof_paths: list[Path] = []
    submission_paths: list[Path] = []

    for exp_k in selected_keys:
        if exp_k not in experiments_cfg:
            raise KeyError(f"Experiment '{exp_k}' not defined in {config_path}")
        exp_spec = experiments_cfg[exp_k]

        exp_id = exp_spec.get("experiment_id", exp_k.upper())
        candidate_id = exp_spec.get("candidate_id", f"cand_{exp_k}")
        recipe_id = exp_spec.get("recipe_id", f"recipe_{exp_k}")
        model_family = exp_spec.get("model_family", "lightgbm")
        model_config_id = exp_spec.get("model_config_id", f"cfg_{exp_k}_default")
        score_kind = exp_spec.get("score_kind", "positive_probability")
        sub_filename = exp_spec.get("submission_filename", f"submission_{exp_k}.csv")

        run_id = f"{candidate_id}_{execution_mode}_{uuid.uuid4().hex[:8]}"
        fit_config_sha = _json_sha256(
            {
                "experiment": exp_k,
                "spec": exp_spec,
                "predictors": predictor_cols,
                "categorical": categorical_cols,
            }
        )

        logger.info(
            "Running follow-up experiment: id=%s candidate=%s mode=%s repeats=%s folds=%s",
            exp_id,
            candidate_id,
            execution_mode,
            repeats,
            folds,
        )

        dest_dir = (
            output_root / execution_mode / f"{candidate_id}_{run_id}"
        ).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
        oof_path = (dest_dir / f"{candidate_id}_candidate_oof.parquet").resolve()

        def _generate_oof_batches(
            current_k: str = exp_k,
            current_spec: dict[str, Any] = exp_spec,
            cand_id: str = candidate_id,
            rec_id: str = recipe_id,
            family: str = model_family,
            cfg_id: str = model_config_id,
            score_k: str = score_kind,
            e_id: str = exp_id,
            fit_sha: str = fit_config_sha,
            r_id: str = run_id,
        ) -> Iterator[Any]:
            total_records = 0
            for rep in repeats:
                split_seed = 42 + rep
                rep_manifest = dev_splits.filter(pl.col("repeat") == rep)

                for fld in folds:
                    train_b_ids = rep_manifest.filter(pl.col("fold") != fld)[
                        "building_id"
                    ].to_list()
                    val_b_ids = rep_manifest.filter(pl.col("fold") == fld)[
                        "building_id"
                    ].to_list()

                    if is_synthetic:
                        train_b_ids = train_b_ids[:4]
                        val_b_ids = val_b_ids[:2]

                    train_fold_df = train_dev.filter(
                        pl.col("building_id").is_in(train_b_ids)
                    )
                    val_fold_df = train_dev.filter(pl.col("building_id").is_in(val_b_ids))

                    if val_fold_df.height == 0:
                        continue

                    if current_k == "exp_f01":
                        sampling_cfg = current_spec.get("sampling", {})
                        f01_params = {
                            **current_spec.get("base_params", {}),
                            "target_ratio": float(
                                sampling_cfg.get("target_ratio", DEFAULT_TARGET_RATIO)
                            ),
                            "alpha_base": float(
                                sampling_cfg.get("alpha_base", DEFAULT_ALPHA_BASE)
                            ),
                            "alpha_trans": float(
                                sampling_cfg.get("alpha_trans", DEFAULT_ALPHA_TRANS)
                            ),
                            "trans_weight": float(
                                sampling_cfg.get("trans_weight", DEFAULT_TRANS_WEIGHT)
                            ),
                            "trans_hours": tuple(
                                sampling_cfg.get(
                                    "trans_hours", TRANSITION_HOURS_DEFAULT
                                )
                            ),
                        }
                        preds = train_diurnal_mixup_partition(
                            train_df=train_fold_df,
                            val_df=val_fold_df,
                            feature_cols=predictor_cols,
                            categorical_cols=categorical_cols,
                            target_col=target_col,
                            params=f01_params,
                            dry_run=dry_run,
                            seed=split_seed,
                        )

                    elif current_k == "exp_f02":
                        cal_cfg = current_spec.get("calibration", {})
                        f02_params = {
                            **current_spec.get("base_params", {}),
                            "c_regularization": float(
                                cal_cfg.get("c_regularization", 1.0)
                            ),
                        }
                        inner_k = int(cal_cfg.get("inner_folds", 3))
                        preds = run_nested_platt_calibration_partition(
                            train_df=train_fold_df,
                            val_df=val_fold_df,
                            feature_cols=predictor_cols,
                            categorical_cols=categorical_cols,
                            target_col=target_col,
                            inner_folds=inner_k,
                            params=f02_params,
                            dry_run=dry_run,
                            seed=split_seed,
                        )

                    elif current_k == "exp_f03":
                        preds = train_skip_leaf_partition(
                            train_df=train_fold_df,
                            val_df=val_fold_df,
                            feature_cols=predictor_cols,
                            categorical_cols=categorical_cols,
                            target_col=target_col,
                            gbdt_params=current_spec.get("gbdt_params", {}),
                            linear_params=current_spec.get("linear_params", {}),
                            dry_run=dry_run,
                            seed=split_seed,
                        )
                    else:
                        raise ValueError(f"Unknown experiment: {current_k}")

                    fold_oof = val_fold_df.select(
                        [
                            pl.col("row_index").cast(pl.Int64),
                            pl.col("building_id").cast(pl.Utf8),
                            pl.col("timestamp").cast(pl.Datetime("ns")),
                            pl.col(target_col).cast(pl.Int64).alias("target"),
                        ]
                    ).with_columns(
                        [
                            pl.lit(cand_id).alias("candidate_id"),
                            pl.lit(rec_id).alias("recipe_id"),
                            pl.lit(family).alias("model_family"),
                            pl.lit(cfg_id).alias("model_config_id"),
                            pl.lit(int(rep)).alias("repeat"),
                            pl.lit(int(fld)).alias("fold"),
                            pl.Series("prediction", preds, dtype=pl.Float64),
                            pl.lit(split_sha).alias("split_sha256"),
                            pl.lit(data_sha).alias("data_sha256"),
                            pl.lit(config_sha).alias("config_sha256"),
                            pl.lit(r_id).alias("run_id"),
                            pl.lit(int(split_seed)).alias("split_seed"),
                            pl.lit(e_id).alias("experiment_id"),
                            pl.lit(current_k).alias("arm_id"),
                            pl.lit(score_k).alias("score_kind"),
                            pl.lit(1).alias("positive_label"),
                            pl.lit(execution_mode).alias("execution_mode"),
                            pl.lit(non_comparative).alias("non_comparative"),
                            pl.lit(n03_eligible).alias("n03_eligible"),
                            pl.lit(fit_sha).alias("model_fit_config_sha256"),
                        ]
                    )

                    if any(fold_oof[n].null_count() > 0 for n in CANDIDATE_OOF_SCHEMA.names):
                        raise ValueError(
                            f"OOF records for {cand_id} (rep={rep}, fold={fld}) contain null values"
                        )

                    arrow_table = (
                        fold_oof.select(CANDIDATE_OOF_SCHEMA.names)
                        .to_arrow()
                        .cast(CANDIDATE_OOF_SCHEMA)
                    )

                    # Explicitly free per-fold arrays and dataframes immediately
                    del val_fold_df, train_fold_df, fold_oof, preds
                    gc.collect()

                    for offset in range(0, arrow_table.num_rows, MAX_BATCH_ROWS):
                        batch = arrow_table.slice(
                            offset, min(MAX_BATCH_ROWS, arrow_table.num_rows - offset)
                        )
                        total_records += batch.num_rows
                        yield batch

                    del arrow_table
                    gc.collect()

            if total_records == 0:
                raise RuntimeError(
                    f"No out-of-fold predictions produced for {cand_id}."
                )

        write_result = write_candidate_oof_batches(
            oof_path, _generate_oof_batches(), CANDIDATE_OOF_SCHEMA
        )
        oof_paths.append(oof_path)
        records_count = int(write_result.get("oof_record_count", 0))

        manifest_data = {
            "schema_version": 1,
            "run_id": run_id,
            "experiment_id": exp_id,
            "candidate_id": candidate_id,
            "recipe_id": recipe_id,
            "execution_mode": execution_mode,
            "non_comparative": non_comparative,
            "n03_eligible": n03_eligible,
            "records_count": records_count,
            "split_sha256": split_sha,
            "data_sha256": data_sha,
            "config_sha256": config_sha,
            "model_fit_config_sha256": fit_config_sha,
            "oof_parquet": str(oof_path),
            "spec": exp_spec,
        }
        manifest_path = dest_dir / f"{candidate_id}_transport_validation.json"
        manifest_path.write_text(json.dumps(manifest_data, indent=2))
        logger.info("Wrote transport validation JSON to %s", manifest_path)

        # ----------------------------------------------------------------------
        # Test Submission Generation
        # ----------------------------------------------------------------------
        if submit:
            logger.info(
                "Generating submission for %s -> %s under %s",
                exp_id,
                sub_filename,
                submissions_root,
            )
            submissions_root.mkdir(parents=True, exist_ok=True)
            sub_path = submissions_root / sub_filename

            # Load test and sample submission if not supplied
            if test_df is None:
                if not test_path.is_file():
                    raise FileNotFoundError(
                        f"Test data file not found: {test_path}"
                    )
                test_joined = pl.read_parquet(test_path)
            else:
                test_joined = test_df

            if sample_sub_df is None:
                if not sample_sub_path.is_file():
                    raise FileNotFoundError(
                        f"Sample submission file not found: {sample_sub_path}"
                    )
                sample_sub = pl.read_csv(sample_sub_path)
            else:
                sample_sub = sample_sub_df

            if smoke:
                sample_sub = sample_sub.head(1000)
                test_joined = test_joined.head(1000)
                train_sub_df = train_dev.head(2000)
            else:
                train_sub_df = train_dev

            if exp_k == "exp_f01":
                sampling_cfg = exp_spec.get("sampling", {})
                f01_params = {
                    **exp_spec.get("base_params", {}),
                    "target_ratio": float(
                        sampling_cfg.get("target_ratio", DEFAULT_TARGET_RATIO)
                    ),
                    "alpha_base": float(
                        sampling_cfg.get("alpha_base", DEFAULT_ALPHA_BASE)
                    ),
                    "alpha_trans": float(
                        sampling_cfg.get("alpha_trans", DEFAULT_ALPHA_TRANS)
                    ),
                    "trans_weight": float(
                        sampling_cfg.get("trans_weight", DEFAULT_TRANS_WEIGHT)
                    ),
                    "trans_hours": tuple(
                        sampling_cfg.get("trans_hours", TRANSITION_HOURS_DEFAULT)
                    ),
                }
                test_preds = train_diurnal_mixup_partition(
                    train_df=train_sub_df,
                    val_df=test_joined,
                    feature_cols=predictor_cols,
                    categorical_cols=categorical_cols,
                    target_col=target_col,
                    params=f01_params,
                    dry_run=dry_run,
                    seed=42,
                )
            elif exp_k == "exp_f02":
                cal_cfg = exp_spec.get("calibration", {})
                f02_params = {
                    **exp_spec.get("base_params", {}),
                    "c_regularization": float(
                        cal_cfg.get("c_regularization", 1.0)
                    ),
                }
                inner_k = int(cal_cfg.get("inner_folds", 3))
                test_preds = run_nested_platt_calibration_partition(
                    train_df=train_sub_df,
                    val_df=test_joined,
                    feature_cols=predictor_cols,
                    categorical_cols=categorical_cols,
                    target_col=target_col,
                    inner_folds=inner_k,
                    params=f02_params,
                    dry_run=dry_run,
                    seed=42,
                )
            elif exp_k == "exp_f03":
                test_preds = train_skip_leaf_partition(
                    train_df=train_sub_df,
                    val_df=test_joined,
                    feature_cols=predictor_cols,
                    categorical_cols=categorical_cols,
                    target_col=target_col,
                    gbdt_params=exp_spec.get("gbdt_params", {}),
                    linear_params=exp_spec.get("linear_params", {}),
                    dry_run=dry_run,
                    seed=42,
                )

            # Explicit row_id alignment with sample_submission
            if "row_id" in test_joined.columns:
                test_pred_df = pl.DataFrame(
                    {
                        "row_id": test_joined["row_id"].cast(pl.Int64),
                        "anomaly": pl.Series("anomaly", test_preds, dtype=pl.Float64),
                    }
                )
                submission_df = (
                    sample_sub.select(["row_id"])
                    .cast({"row_id": pl.Int64})
                    .join(test_pred_df, on="row_id", how="left")
                )
                if submission_df["anomaly"].null_count() > 0:
                    raise ValueError(
                        f"Missing test predictions after row_id alignment with sample_submission for {exp_id}"
                    )
            else:
                row_ids = sample_sub["row_id"].to_numpy()
                submission_df = build_submission(
                    row_ids=row_ids, probabilities=test_preds
                )

            validate_submission_contract(
                submission_df,
                expected_row_count=sample_sub.height,
                expected_row_ids=sample_sub["row_id"].to_list(),
            )
            saved_csv = save_submission(
                submission_df=submission_df,
                output_path=sub_path,
                expected_row_count=sample_sub.height,
            )
            submission_paths.append(saved_csv)
            logger.info("Successfully generated submission: %s", saved_csv)
            if kaggle_submit and not smoke and not dry_run:
                msg = f"Kaggle LEAD Experiment: {exp_id} ({exp_spec.get('title', exp_id)})"
                cmd = [
                    "kaggle",
                    "competitions",
                    "submit",
                    "-c",
                    "energy-anomaly-detection",
                    "-f",
                    str(saved_csv),
                    "-m",
                    msg,
                ]
                logger.info("Submitting %s to Kaggle: %s", saved_csv.name, msg)
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode == 0:
                    logger.info("Kaggle CLI success: %s", res.stdout.strip())
                else:
                    logger.warning("Kaggle CLI warning/error: %s | stderr: %s", res.stdout, res.stderr)
                time.sleep(3)

    return {"oof_paths": oof_paths, "submission_paths": submission_paths}


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    results = run_followup_experiments(
        experiment_key=args.experiment,
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        submissions_dir=args.submissions_dir,
        smoke=args.smoke,
        dry_run=args.dry_run,
        submit=args.submit or args.kaggle_submit,
        kaggle_submit=args.kaggle_submit,
    )
    logger.info("Follow-up execution complete. Generated OOFs: %s", results["oof_paths"])
    if args.submit:
        logger.info("Generated submissions: %s", results["submission_paths"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
