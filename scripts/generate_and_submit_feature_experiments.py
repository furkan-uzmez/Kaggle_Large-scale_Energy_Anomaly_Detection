"""Train and submit all feature engineering and ablation experiment models to Kaggle.

Covers the 10 core feature hypothesis and ablation candidates from candidate_full_v2:
1. feat_missingness_indicators
2. feat_missingness_bitmask
3. feat_categorical_interactions
4. feat_frequency_encoding
5. feat_meter_deviation_ratio
6. feat_histogram_quantile
7. feat_safe_calendar_numeric_transforms
8. dim_semantic_block_drop_calendar
9. dim_semantic_block_drop_measurement
10. dim_semantic_block_drop_building_metadata

Fits HistGradientBoostingClassifier on full train_joined.parquet, predicts on
test_joined.parquet, validates against sample_submission.csv, and submits to Kaggle.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

from project.submission import (
    CORE_GBDT_CATEGORICAL,
    CORE_GBDT_PREDICTOR_COLUMNS,
    _to_clean_pandas,
    build_submission,
    save_submission,
    validate_submission_contract,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_feature_experiments")


def submit_to_kaggle(file_path: Path, message: str) -> bool:
    """Submit a validated CSV to Kaggle via the CLI."""
    cmd = [
        "/home/furkan/.local/bin/kaggle",
        "competitions",
        "submit",
        "-c",
        "energy-anomaly-detection",
        "-f",
        str(file_path),
        "-m",
        message,
    ]
    logger.info("Submitting to Kaggle CLI: %s", " ".join(cmd))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and submit feature engineering and ablation models.")
    parser.add_argument("--smoke", action="store_true", help="Run in fast smoke mode on tiny sample")
    args = parser.parse_args()

    smoke = args.smoke
    out_dir = Path("outputs/submissions/feature_experiments")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_sub_path = Path("data/energy-anomaly-detection/sample_submission.csv")
    train_path = Path("outputs/prepared/train_joined.parquet")
    test_path = Path("outputs/prepared/test_joined.parquet")

    logger.info("Loading reference sample submission...")
    sample_sub = pl.read_csv(sample_sub_path)
    if smoke:
        sample_sub = sample_sub.head(1000)
    row_ids = sample_sub["row_id"].to_numpy()

    logger.info("Loading train data from %s...", train_path)
    train_df = pl.read_parquet(train_path)
    logger.info("Loading test data from %s...", test_path)
    test_df = pl.read_parquet(test_path)
    if smoke:
        train_df = train_df.head(2000)
        test_df = test_df.head(1000)

    y_train = np.asarray(train_df["anomaly"], dtype=np.int64)

    # Base 9 core features
    X_tr_base = _to_clean_pandas(train_df, CORE_GBDT_PREDICTOR_COLUMNS)
    X_te_base = _to_clean_pandas(test_df, CORE_GBDT_PREDICTOR_COLUMNS)

    for cat in CORE_GBDT_CATEGORICAL:
        X_tr_base[cat] = X_tr_base[cat].astype("category")
        X_te_base[cat] = pd.Categorical(X_te_base[cat], categories=X_tr_base[cat].cat.categories)

    # 10 Experiment Definitions
    experiments = []

    # 1. Missingness Indicators
    def exp_missingness_indicators():
        X_tr = X_tr_base.copy()
        X_te = X_te_base.copy()
        for col in ["year_built", "floor_count"]:
            X_tr[f"{col}_isnan"] = X_tr[col].isna().astype(float)
            X_te[f"{col}_isnan"] = X_te[col].isna().astype(float)
        return X_tr, X_te

    experiments.append(("submission_22_feat_missingness_indicators.csv", "Feat: Missingness Indicators", exp_missingness_indicators))

    # 2. Missingness Bitmask
    def exp_missingness_bitmask():
        X_tr = X_tr_base.copy()
        X_te = X_te_base.copy()
        mask_tr = (X_tr["year_built"].isna().astype(int)) | (X_tr["floor_count"].isna().astype(int) * 2)
        mask_te = (X_te["year_built"].isna().astype(int)) | (X_te["floor_count"].isna().astype(int) * 2)
        X_tr["missing_bitmask"] = mask_tr
        X_te["missing_bitmask"] = mask_te
        return X_tr, X_te

    experiments.append(("submission_23_feat_missingness_bitmask.csv", "Feat: Missingness Bitmask", exp_missingness_bitmask))

    # 3. Categorical Interactions
    def exp_cat_interactions():
        X_tr = X_tr_base.copy()
        X_te = X_te_base.copy()
        inter_tr = X_tr["primary_use"].astype(str) + "_" + (X_tr["hour"] // 6).astype(str)
        inter_te = X_te["primary_use"].astype(str) + "_" + (X_te["hour"] // 6).astype(str)
        X_tr["use_daypart"] = inter_tr.astype("category")
        X_te["use_daypart"] = pd.Categorical(inter_te, categories=X_tr["use_daypart"].cat.categories)
        return X_tr, X_te

    experiments.append(("submission_24_feat_categorical_interactions.csv", "Feat: Categorical Interactions", exp_cat_interactions))

    # 4. Frequency Encoding
    def exp_freq_encoding():
        X_tr = X_tr_base.copy()
        X_te = X_te_base.copy()
        freq = X_tr["primary_use"].value_counts(normalize=True).to_dict()
        X_tr["use_freq"] = X_tr["primary_use"].map(freq).fillna(0.0)
        X_te["use_freq"] = X_te["primary_use"].map(freq).fillna(0.0)
        return X_tr, X_te

    experiments.append(("submission_25_feat_frequency_encoding.csv", "Feat: Frequency Encoding", exp_freq_encoding))

    # 5. Meter Deviation Ratio
    def exp_meter_deviation():
        X_tr = X_tr_base.copy()
        X_te = X_te_base.copy()
        med = float(X_tr["meter_reading"].median()) if float(X_tr["meter_reading"].median()) > 0 else 1.0
        X_tr["meter_ratio"] = X_tr["meter_reading"] / med
        X_te["meter_ratio"] = X_te["meter_reading"] / med
        return X_tr, X_te

    experiments.append(("submission_26_feat_meter_deviation_ratio.csv", "Feat: Meter Deviation Ratio", exp_meter_deviation))

    # 6. Histogram Quantile
    def exp_histogram_quantile():
        X_tr = X_tr_base.copy()
        X_te = X_te_base.copy()
        bins = np.quantile(X_tr["meter_reading"].dropna(), np.linspace(0, 1, 11))
        bins[0] -= 1e-5
        bins[-1] += 1e-5
        X_tr["meter_qbin"] = pd.cut(X_tr["meter_reading"], bins=bins, labels=False).fillna(-1)
        X_te["meter_qbin"] = pd.cut(X_te["meter_reading"], bins=bins, labels=False).fillna(-1)
        return X_tr, X_te

    experiments.append(("submission_27_feat_histogram_quantile.csv", "Feat: Histogram Quantile Bins", exp_histogram_quantile))

    # 7. Safe Calendar Numeric Transforms
    def exp_calendar_transforms():
        X_tr = X_tr_base.copy()
        X_te = X_te_base.copy()
        X_tr["sin_hour"] = np.sin(2 * np.pi * X_tr["hour"] / 24.0)
        X_tr["cos_hour"] = np.cos(2 * np.pi * X_tr["hour"] / 24.0)
        X_te["sin_hour"] = np.sin(2 * np.pi * X_te["hour"] / 24.0)
        X_te["cos_hour"] = np.cos(2 * np.pi * X_te["hour"] / 24.0)
        return X_tr, X_te

    experiments.append(("submission_28_feat_calendar_transforms.csv", "Feat: Calendar Numeric Sin/Cos", exp_calendar_transforms))

    # 8. Drop Calendar Block (Ablation)
    def exp_drop_calendar():
        cols = [c for c in CORE_GBDT_PREDICTOR_COLUMNS if c not in ["hour", "weekday", "month", "year"]]
        X_tr = _to_clean_pandas(train_df, cols)
        X_te = _to_clean_pandas(test_df, cols)
        X_tr["primary_use"] = X_tr["primary_use"].astype("category")
        X_te["primary_use"] = pd.Categorical(X_te["primary_use"], categories=X_tr["primary_use"].cat.categories)
        return X_tr, X_te

    experiments.append(("submission_29_dim_drop_calendar.csv", "Ablation: Drop Calendar Block", exp_drop_calendar))

    # 9. Drop Measurement Block (Ablation - Significant Regression)
    def exp_drop_measurement():
        cols = [c for c in CORE_GBDT_PREDICTOR_COLUMNS if c != "meter_reading"]
        X_tr = _to_clean_pandas(train_df, cols)
        X_te = _to_clean_pandas(test_df, cols)
        X_tr["primary_use"] = X_tr["primary_use"].astype("category")
        X_te["primary_use"] = pd.Categorical(X_te["primary_use"], categories=X_tr["primary_use"].cat.categories)
        return X_tr, X_te

    experiments.append(("submission_30_dim_drop_measurement.csv", "Ablation: Drop Measurement Block", exp_drop_measurement))

    # 10. Drop Building Metadata (Ablation)
    def exp_drop_metadata():
        cols = [c for c in CORE_GBDT_PREDICTOR_COLUMNS if c not in ["square_feet", "year_built", "floor_count", "primary_use"]]
        X_tr = _to_clean_pandas(train_df, cols)
        X_te = _to_clean_pandas(test_df, cols)
        return X_tr, X_te

    experiments.append(("submission_31_dim_drop_building_metadata.csv", "Ablation: Drop Building Metadata", exp_drop_metadata))

    logger.info("Executing and submitting %d feature and ablation experiments...", len(experiments))
    success_count = 0
    for idx, (filename, desc, prep_fn) in enumerate(experiments, start=1):
        target_csv = out_dir / filename
        logger.info("[%d/%d] Training and evaluating: %s -> %s", idx, len(experiments), desc, filename)
        t0 = time.time()
        try:
            X_tr, X_te = prep_fn()
            clf = HistGradientBoostingClassifier(
                max_iter=30 if smoke else 100,
                random_state=42,
                early_stopping=False,
            )
            clf.fit(X_tr, y_train)
            probs = clf.predict_proba(X_te)[:, 1]
            sub = build_submission(row_ids, probs)
            validate_submission_contract(sub, expected_row_count=len(row_ids), expected_row_ids=sample_sub["row_id"])
            save_submission(sub, target_csv)
            elapsed = time.time() - t0
            logger.info("[%d/%d] Successfully generated %s in %.1f seconds", idx, len(experiments), target_csv.name, elapsed)

            if not smoke:
                msg = f"Kaggle LEAD Experiment: {desc}"
                ok = submit_to_kaggle(target_csv, msg)
                if ok:
                    success_count += 1
                time.sleep(3)
            else:
                success_count += 1
        except Exception:
            logger.exception("[%d/%d] Failed experiment %s", idx, len(experiments), desc)

    logger.info("Feature experiments completed: %d/%d successfully processed.", success_count, len(experiments))


if __name__ == "__main__":
    main()
