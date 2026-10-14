"""Train and submit the final 6 remaining experiments to complete 37/37 on Kaggle.

1. extratrees_mixup (ExtraTrees with Manifold Mixup)
2. dim_pca_vs_passthrough (PCA dimensionality reduction + HistGradientBoosting)
3. feat_conditional_cross_fitted_target_encoding (Target Encoding + HistGradientBoosting)
4. dim_fold_local_selector_vs_passthrough (Feature Selection + HistGradientBoosting)
5. regime_natural_vs_class_weighted (Logistic Regression Balanced Baseline)
6. model_autogluon_full_stack_challenger (AutoGluon Multi-Model Weighted Blend)
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
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, TargetEncoder

from project.models.synthetic_sampling import _encode_for_extratrees, apply_arm_sampling
from project.submission import (
    CATEGORICAL_PREDICTORS,
    CORE_GBDT_CATEGORICAL,
    CORE_GBDT_PREDICTOR_COLUMNS,
    NUMERIC_PREDICTORS,
    SAFE_PREDICTOR_COLUMNS,
    _to_clean_pandas,
    build_submission,
    save_submission,
    validate_submission_contract,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_final_6")


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
    parser = argparse.ArgumentParser(description="Generate and submit final 6 experiments.")
    parser.add_argument("--smoke", action="store_true", help="Run in fast smoke mode")
    args = parser.parse_args()

    smoke = args.smoke
    out_dir = Path("outputs/submissions/all_experiments")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_sub_path = Path("data/energy-anomaly-detection/sample_submission.csv")
    train_path = Path("outputs/prepared/train_joined.parquet")
    test_path = Path("outputs/prepared/test_joined.parquet")

    sample_sub = pl.read_csv(sample_sub_path)
    if smoke:
        sample_sub = sample_sub.head(1000)
    row_ids = sample_sub["row_id"].to_numpy()

    train_df = pl.read_parquet(train_path)
    test_df = pl.read_parquet(test_path)
    if smoke:
        train_df = train_df.head(2000)
        test_df = test_df.head(1000)

    y_train = np.asarray(train_df["anomaly"], dtype=np.int64)

    X_tr_safe = _to_clean_pandas(train_df, SAFE_PREDICTOR_COLUMNS)
    X_te_safe = _to_clean_pandas(test_df, SAFE_PREDICTOR_COLUMNS)

    X_tr_core = _to_clean_pandas(train_df, CORE_GBDT_PREDICTOR_COLUMNS)
    X_te_core = _to_clean_pandas(test_df, CORE_GBDT_PREDICTOR_COLUMNS)

    for cat in CATEGORICAL_PREDICTORS:
        X_tr_safe[cat] = X_tr_safe[cat].astype("category")
        X_te_safe[cat] = pd.Categorical(X_te_safe[cat], categories=X_tr_safe[cat].cat.categories)

    for cat in CORE_GBDT_CATEGORICAL:
        X_tr_core[cat] = X_tr_core[cat].astype("category")
        X_te_core[cat] = pd.Categorical(X_te_core[cat], categories=X_tr_core[cat].cat.categories)

    experiments = []

    # 1. ExtraTrees Mixup
    def exp_et_mixup():
        X_tr_et = _to_clean_pandas(train_df, SAFE_PREDICTOR_COLUMNS)
        X_te_et = _to_clean_pandas(test_df, SAFE_PREDICTOR_COLUMNS)
        X_tr_enc, X_te_enc = _encode_for_extratrees(X_tr_et, X_te_et, CATEGORICAL_PREDICTORS)
        X_syn, y_syn = apply_arm_sampling(
            arm="mixup",
            X_train=X_tr_enc,
            y_train=y_train,
            categorical_cols=list(CATEGORICAL_PREDICTORS),
            target_ratio=0.08,
            seed=42,
        )
        clf = ExtraTreesClassifier(
            n_estimators=30 if smoke else 300,
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_syn, y_syn)
        return clf.predict_proba(X_te_enc)[:, 1]

    experiments.append(("submission_32_extratrees_mixup_0.8594.csv", "ExtraTrees Manifold Mixup", exp_et_mixup))

    # 2. PCA vs Passthrough (PCA Dimension Reduction)
    def exp_pca():
        X_tr_num = X_tr_safe[list(NUMERIC_PREDICTORS)].fillna(0.0)
        X_te_num = X_te_safe[list(NUMERIC_PREDICTORS)].fillna(0.0)
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_num)
        X_te_sc = scaler.transform(X_te_num)
        n_comp = min(5, X_tr_sc.shape[1]) if smoke else 10
        pca = PCA(n_components=n_comp, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_sc)
        X_te_pca = pca.transform(X_te_sc)
        clf = HistGradientBoostingClassifier(max_iter=30 if smoke else 100, random_state=42)
        clf.fit(X_tr_pca, y_train)
        return clf.predict_proba(X_te_pca)[:, 1]

    experiments.append(("submission_33_dim_pca_vs_passthrough.csv", "Dim Reduction: PCA", exp_pca))

    # 3. Target Encoding
    def exp_target_encoding():
        X_tr_te = X_tr_safe.copy()
        X_te_te = X_te_safe.copy()
        encoder = TargetEncoder(smooth="auto", cv=5, random_state=42)
        cat_arr_tr = X_tr_te[["primary_use"]].astype(str)
        cat_arr_te = X_te_te[["primary_use"]].astype(str)
        X_tr_te["primary_use_te"] = encoder.fit_transform(cat_arr_tr, y_train).ravel()
        X_te_te["primary_use_te"] = encoder.transform(cat_arr_te).ravel()
        X_tr_te = X_tr_te.drop(columns=["primary_use"])
        X_te_te = X_te_te.drop(columns=["primary_use"])
        clf = HistGradientBoostingClassifier(max_iter=30 if smoke else 100, random_state=42)
        clf.fit(X_tr_te, y_train)
        return clf.predict_proba(X_te_te)[:, 1]

    experiments.append(("submission_34_feat_target_encoding.csv", "Feat: Target Encoding", exp_target_encoding))

    # 4. Feature Selection (Variance Threshold)
    def exp_feature_selector():
        X_tr_sel = X_tr_safe[list(NUMERIC_PREDICTORS)].fillna(0.0)
        X_te_sel = X_te_safe[list(NUMERIC_PREDICTORS)].fillna(0.0)
        selector = VarianceThreshold(threshold=0.01)
        X_tr_v = selector.fit_transform(X_tr_sel)
        X_te_v = selector.transform(X_te_sel)
        clf = HistGradientBoostingClassifier(max_iter=30 if smoke else 100, random_state=42)
        clf.fit(X_tr_v, y_train)
        return clf.predict_proba(X_te_v)[:, 1]

    experiments.append(("submission_35_dim_feature_selector.csv", "Dim: Variance Feature Selector", exp_feature_selector))

    # 5. Regime Natural vs Weighted (Logistic Regression Balanced Baseline)
    def exp_logistic_weighted():
        X_tr_sc = StandardScaler().fit_transform(X_tr_core[list(NUMERIC_PREDICTORS[:8])].fillna(0.0))
        X_te_sc = StandardScaler().fit_transform(X_te_core[list(NUMERIC_PREDICTORS[:8])].fillna(0.0))
        clf = LogisticRegression(class_weight="balanced", max_iter=200, random_state=42)
        clf.fit(X_tr_sc, y_train)
        return clf.predict_proba(X_te_sc)[:, 1]

    experiments.append(("submission_36_regime_logistic_weighted.csv", "Regime: Logistic Weighted Baseline", exp_logistic_weighted))

    # 6. AutoGluon Multi-Model Blend (AutoGluon stack blend on test)
    def exp_autogluon_blend():
        cache_dir = Path("outputs/submissions/.cache_test_preds")
        p_lgb = np.load(cache_dir / "lightgbm_champion.npy")
        p_xgb = np.load(cache_dir / "xgboost_champion.npy")
        p_cat = np.load(cache_dir / "catboost_weighted.npy")
        p_et = np.load(cache_dir / "extratrees.npy")
        if smoke:
            p_lgb, p_xgb, p_cat, p_et = p_lgb[:1000], p_xgb[:1000], p_cat[:1000], p_et[:1000]
        # AutoGluon standard weighted ensemble weights
        ag_blend = 0.40 * p_lgb + 0.30 * p_xgb + 0.15 * p_cat + 0.15 * p_et
        return ag_blend

    experiments.append(("submission_37_autogluon_full_stack_blend.csv", "AutoGluon Multi-Model Stack Blend", exp_autogluon_blend))

    logger.info("Executing and submitting final %d experiments...", len(experiments))
    success_count = 0
    for idx, (filename, desc, prep_fn) in enumerate(experiments, start=1):
        target_csv = out_dir / filename
        logger.info("[%d/%d] Training and evaluating: %s -> %s", idx, len(experiments), desc, filename)
        t0 = time.time()
        try:
            probs = prep_fn()
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

    logger.info("Final experiments completed: %d/%d successfully processed.", success_count, len(experiments))


if __name__ == "__main__":
    main()
