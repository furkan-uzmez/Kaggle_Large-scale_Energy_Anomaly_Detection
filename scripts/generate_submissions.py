#!/usr/bin/env python3
"""CLI script to generate champion Kaggle submissions for energy anomaly detection.

Generates the 8 champion submission files under outputs/submissions/ conforming
strictly to the competition submission contract:
1. submission_01_ensemble_logit_lr_0.9008.csv
2. submission_02_lightgbm_champion_0.8946.csv
3. submission_03_ensemble_caruana_0.8908.csv
4. submission_04_xgboost_champion_0.8937.csv
5. submission_05_lightgbm_focal_0.8937.csv
6. submission_06_catboost_weighted_0.8746.csv
7. submission_07_extratrees_0.8740.csv
8. submission_08_tabular_mlp_0.8302.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

from project.submission import (
    SUBMISSION_SPECS,
    audit_submission_file,
    generate_all_submissions,
    validate_submission_contract,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("generate_submissions")

DEFAULT_OUTPUT_DIR = Path("outputs/submissions")
DEFAULT_SMOKE_OUTPUT_DIR = Path("outputs/submissions/smoke")
DEFAULT_TRAIN_PATH = Path("outputs/prepared/train_joined.parquet")
DEFAULT_TEST_PATH = Path("outputs/prepared/test_joined.parquet")
DEFAULT_SAMPLE_SUB_PATH = Path("data/energy-anomaly-detection/sample_submission.csv")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Kaggle anomaly detection test submission files."
    )
    parser.add_argument(
        "--models",
        choices=["all", "ensemble", "single"],
        default="all",
        help="Model subset to generate: 'all' (8 submissions), 'ensemble' (2 submissions), 'single' (6 submissions).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute in dry-run mode on a tiny sample using deterministic mock predictors without fitting full models.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Execute smoke run with truncated iterations and epochs on a small training subset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of test rows to evaluate (default: 500 for --dry-run if omitted, or full 1,800,567 rows).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for generated CSV submissions.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Path to prepared training features Parquet.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=DEFAULT_TEST_PATH,
        help="Path to prepared test features Parquet.",
    )
    parser.add_argument(
        "--sample-submission-path",
        type=Path,
        default=DEFAULT_SAMPLE_SUB_PATH,
        help="Path to official sample_submission.csv for contract order verification.",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Main CLI execution routine."""
    parsed = parse_args(args)

    logger.info("=" * 70)
    logger.info("Kaggle Energy Anomaly Detection Submission Pipeline")
    logger.info(
        "Models mode: %s | Dry-run: %s | Smoke: %s",
        parsed.models,
        parsed.dry_run,
        parsed.smoke,
    )
    output_dir = parsed.output_dir
    if (parsed.smoke or parsed.dry_run) and output_dir == DEFAULT_OUTPUT_DIR:
        output_dir = DEFAULT_SMOKE_OUTPUT_DIR
        logger.info(
            "Smoke or dry-run active: redirecting output directory to %s to protect champion files",
            output_dir,
        )
    logger.info("Target output directory: %s", output_dir)
    logger.info("=" * 70)

    # In dry-run mode, default sample_size to 500 if not explicitly given
    sample_size = parsed.sample_size
    if parsed.dry_run and sample_size is None:
        sample_size = 500
        logger.info(
            "Dry-run mode active: defaulting test sample size to %d rows", sample_size
        )

    try:
        generated = generate_all_submissions(
            output_dir=output_dir,
            models_mode=parsed.models,
            smoke=parsed.smoke,
            dry_run=parsed.dry_run,
            sample_size=sample_size,
            train_path=parsed.train_path,
            test_path=parsed.test_path,
            sample_sub_path=parsed.sample_submission_path,
        )

        logger.info("-" * 70)
        logger.info("Generated Submissions Summary Verification:")
        logger.info("-" * 70)

        # Load reference sample_submission.csv for post-generation audit if available
        ref_row_ids: list[int] | None = None
        expected_row_count: int | None = None
        if parsed.sample_submission_path.is_file():
            logger.info(
                "Loading reference sample submission from %s for contract audit",
                parsed.sample_submission_path,
            )
            ref_df = pl.read_csv(parsed.sample_submission_path)
            if sample_size is not None and sample_size > 0:
                ref_df = ref_df.head(sample_size)
            ref_row_ids = ref_df["row_id"].to_list()
            expected_row_count = len(ref_row_ids)

        # Post-generation audit of all files
        for key, path in sorted(generated.items()):
            df = pl.read_csv(path)
            # Validate contract and match against sample_submission.csv row_id sequence
            validate_submission_contract(
                df,
                expected_row_count=expected_row_count if expected_row_count is not None else df.height,
                expected_row_ids=ref_row_ids,
            )

            if ref_row_ids is not None:
                actual_row_ids = df["row_id"].to_list()
                if actual_row_ids != ref_row_ids:
                    raise ValueError(
                        f"Post-generation audit failed for {path.name}: "
                        f"row_id sequence does not match sample_submission.csv! "
                        f"(actual length={len(actual_row_ids)}, expected length={len(ref_row_ids)})"
                    )

            # Run audit_submission_file check if reference exists
            if parsed.sample_submission_path.is_file():
                audit_submission_file(
                    path,
                    parsed.sample_submission_path,
                    sample_size=sample_size,
                )
            anom = df["anomaly"].to_numpy()
            min_val = float(anom.min())
            max_val = float(anom.max())
            mean_val = float(anom.mean())
            size_mb = path.stat().st_size / (1024 * 1024)

            spec = SUBMISSION_SPECS.get(key, {})
            exp_cv = spec.get("expected_cv_score", "N/A")

            logger.info(
                "[%s] (Expected CV: %s) -> %s\n"
                "  Rows: %d | Size: %.2f MB | Min: %.4f | Max: %.4f | Mean: %.4f | Contract & sample_sub audit: VERIFIED",
                key,
                exp_cv,
                path.name,
                df.height,
                size_mb,
                min_val,
                max_val,
                mean_val,
            )

        logger.info("=" * 70)
        logger.info(
            "SUCCESS: All %d requested submissions generated and verified.",
            len(generated),
        )
        logger.info("=" * 70)
        return 0

    except Exception:
        logger.exception("Submission pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
