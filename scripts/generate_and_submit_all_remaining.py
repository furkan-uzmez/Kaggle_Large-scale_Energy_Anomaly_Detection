"""Generate and submit all remaining experiment models to Kaggle.

Iterates through remaining candidate experiments:
- GBDT class weighted models (LightGBM, XGBoost)
- GBDT Focal Loss models (CatBoost)
- GBDT LogLoss unweighted (CatBoost)
- Synthetic sampling models (LightGBM SMOTE, LightGBM Mixup, ExtraTrees SMOTE, ExtraTrees Mixup)
- GBDT HPO models (LightGBM HPO, XGBoost HPO)
- Feature selection models (LightGBM, XGBoost, CatBoost on top-30 selected features)
- Tabular MLP Challenger (Quantile + Missing indicators)

Builds validated submission CSV files and submits them to Kaggle via the Kaggle CLI.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from project.models.synthetic_sampling import _encode_for_extratrees, apply_arm_sampling
from project.models.tabular_mlp import FoldLocalPreprocessor, _build_mlp, _require_torch
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
logger = logging.getLogger("generate_and_submit_all_remaining")


def submit_to_kaggle(file_path: Path, message: str) -> bool:
    """Submit a validated CSV to Kaggle via the CLI."""
    cmd = [
        "kaggle",
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
        else:
            logger.warning("Kaggle CLI warning/error: %s | stderr: %s", res.stdout, res.stderr)
            return False
    except OSError as e:
        logger.error("Failed to run kaggle CLI: %s", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and submit all remaining experiment models.")
    parser.add_argument("--smoke", action="store_true", help="Run in fast smoke mode on tiny sample")
    args = parser.parse_args()

    smoke = args.smoke
    out_dir = Path("outputs/submissions/all_experiments")
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

    X_tr_safe = _to_clean_pandas(train_df, SAFE_PREDICTOR_COLUMNS)
    X_te_safe = _to_clean_pandas(test_df, SAFE_PREDICTOR_COLUMNS)

    X_tr_core = _to_clean_pandas(train_df, CORE_GBDT_PREDICTOR_COLUMNS)
    X_te_core = _to_clean_pandas(test_df, CORE_GBDT_PREDICTOR_COLUMNS)

    # Convert categoricals safely
    for cat in CATEGORICAL_PREDICTORS:
        X_tr_safe[cat] = X_tr_safe[cat].astype("category")
        X_te_safe[cat] = pd.Categorical(X_te_safe[cat], categories=X_tr_safe[cat].cat.categories)

    for cat in CORE_GBDT_CATEGORICAL:
        X_tr_core[cat] = X_tr_core[cat].astype("category")
        X_te_core[cat] = pd.Categorical(X_te_core[cat], categories=X_tr_core[cat].cat.categories)

    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.ensemble import ExtraTreesClassifier

    # List of all remaining model definitions: (filename, description, train_and_predict_fn)
    tasks: list[tuple[str, str, Any]] = []

    # 1. LightGBM Weighted
    def run_lgb_weighted() -> np.ndarray:
        clf = lgb.LGBMClassifier(
            learning_rate=0.03,
            n_estimators=50 if smoke else 1000,
            num_leaves=31,
            is_unbalance=True,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        clf.fit(X_tr_safe, y_train)
        return clf.predict_proba(X_te_safe)[:, 1]

    tasks.append(("submission_13_lightgbm_weighted_0.8933.csv", "LightGBM Class Weighted", run_lgb_weighted))

    # 2. XGBoost Weighted
    def run_xgb_weighted() -> np.ndarray:
        n_pos = np.sum(y_train == 1)
        scale = float(len(y_train) - n_pos) / float(max(1, n_pos))
        clf = xgb.XGBClassifier(
            learning_rate=0.03,
            n_estimators=50 if smoke else 1000,
            max_depth=6,
            scale_pos_weight=scale,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            enable_categorical=True,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_tr_safe, y_train)
        return clf.predict_proba(X_te_safe)[:, 1]

    tasks.append(("submission_14_xgboost_weighted_0.8909.csv", "XGBoost Class Weighted", run_xgb_weighted))

    # 3. CatBoost LogLoss Unweighted
    def run_cat_logloss() -> np.ndarray:
        tr_c = X_tr_core.copy()
        te_c = X_te_core.copy()
        for c in CORE_GBDT_CATEGORICAL:
            tr_c[c] = tr_c[c].astype(str).fillna("__missing__")
            te_c[c] = te_c[c].astype(str).fillna("__missing__")
        clf = CatBoostClassifier(
            iterations=50 if smoke else 1000,
            learning_rate=0.03,
            depth=6,
            random_seed=42,
            loss_function="Logloss",
            cat_features=list(CORE_GBDT_CATEGORICAL),
            verbose=False,
            thread_count=-1,
        )
        clf.fit(tr_c, y_train)
        return clf.predict_proba(te_c)[:, 1]

    tasks.append(("submission_15_catboost_logloss_0.8680.csv", "CatBoost Unweighted Logloss", run_cat_logloss))

    # 4. ExtraTrees SMOTE
    def run_et_smote() -> np.ndarray:
        X_tr_et = _to_clean_pandas(train_df, SAFE_PREDICTOR_COLUMNS)
        X_te_et = _to_clean_pandas(test_df, SAFE_PREDICTOR_COLUMNS)
        X_tr_enc, X_te_enc = _encode_for_extratrees(X_tr_et, X_te_et, CATEGORICAL_PREDICTORS)
        X_syn, y_syn = apply_arm_sampling(
            arm="smote",
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

    tasks.append(("submission_16_extratrees_smote_0.8735.csv", "ExtraTrees SMOTE", run_et_smote))

    # 5. LightGBM SMOTE
    def run_lgb_smote() -> np.ndarray:
        X_tr_l = _to_clean_pandas(train_df, SAFE_PREDICTOR_COLUMNS)
        X_te_l = _to_clean_pandas(test_df, SAFE_PREDICTOR_COLUMNS)
        X_syn, y_syn = apply_arm_sampling(
            arm="smote",
            X_train=X_tr_l,
            y_train=y_train,
            categorical_cols=list(CATEGORICAL_PREDICTORS),
            seed=42,
        )
        for cat in CATEGORICAL_PREDICTORS:
            X_syn[cat] = X_syn[cat].astype("category")
            X_te_l[cat] = pd.Categorical(X_te_l[cat], categories=X_syn[cat].cat.categories)
        clf = lgb.LGBMClassifier(
            learning_rate=0.03,
            n_estimators=50 if smoke else 1000,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        clf.fit(X_syn, y_syn)
        return clf.predict_proba(X_te_l)[:, 1]

    tasks.append(("submission_17_lightgbm_smote_0.8919.csv", "LightGBM SMOTE", run_lgb_smote))

    # 6. LightGBM Mixup
    def run_lgb_mixup() -> np.ndarray:
        X_tr_l = _to_clean_pandas(train_df, SAFE_PREDICTOR_COLUMNS)
        X_te_l = _to_clean_pandas(test_df, SAFE_PREDICTOR_COLUMNS)
        X_syn, y_syn = apply_arm_sampling(
            arm="mixup",
            X_train=X_tr_l,
            y_train=y_train,
            categorical_cols=list(CATEGORICAL_PREDICTORS),
            seed=42,
        )
        for cat in CATEGORICAL_PREDICTORS:
            X_syn[cat] = X_syn[cat].astype("category")
            X_te_l[cat] = pd.Categorical(X_te_l[cat], categories=X_syn[cat].cat.categories)
        clf = lgb.LGBMClassifier(
            learning_rate=0.03,
            n_estimators=50 if smoke else 1000,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        clf.fit(X_syn, y_syn)
        return clf.predict_proba(X_te_l)[:, 1]

    tasks.append(("submission_18_lightgbm_mixup_0.8933.csv", "LightGBM Manifold Mixup", run_lgb_mixup))

    # 7. LightGBM HPO (Optuna Best Hyperparameters)
    def run_lgb_hpo() -> np.ndarray:
        clf = lgb.LGBMClassifier(
            learning_rate=0.0152,
            n_estimators=50 if smoke else 1000,
            num_leaves=21,
            max_depth=7,
            subsample=0.71,
            colsample_bytree=0.61,
            reg_alpha=0.057,
            reg_lambda=0.062,
            min_child_samples=24,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        clf.fit(X_tr_safe, y_train)
        return clf.predict_proba(X_te_safe)[:, 1]

    tasks.append(("submission_19_lightgbm_hpo_0.8870.csv", "LightGBM Optuna Tuned", run_lgb_hpo))

    # 8. XGBoost HPO (Optuna Best Hyperparameters)
    def run_xgb_hpo() -> np.ndarray:
        clf = xgb.XGBClassifier(
            learning_rate=0.0162,
            n_estimators=50 if smoke else 1000,
            max_depth=8,
            subsample=0.67,
            colsample_bytree=0.63,
            reg_alpha=6.25,
            reg_lambda=7.29,
            min_child_weight=6.47,
            tree_method="hist",
            enable_categorical=True,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_tr_safe, y_train)
        return clf.predict_proba(X_te_safe)[:, 1]

    tasks.append(("submission_20_xgboost_hpo_0.8908.csv", "XGBoost Optuna Tuned", run_xgb_hpo))

    # 9. Tabular MLP Challenger (Quantile + Missing Indicators)
    def run_mlp_challenger() -> np.ndarray:
        torch = _require_torch()
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prep = FoldLocalPreprocessor(arm="challenger", numeric_cols=list(NUMERIC_PREDICTORS), categorical_cols=list(CATEGORICAL_PREDICTORS))
        X_tr_p = _to_clean_pandas(train_df, SAFE_PREDICTOR_COLUMNS)
        X_te_p = _to_clean_pandas(test_df, SAFE_PREDICTOR_COLUMNS)
        prep.fit(X_tr_p)
        X_tr_n, X_tr_c = prep.transform(X_tr_p)
        X_te_n, X_te_c = prep.transform(X_te_p)

        dataset = TensorDataset(
            torch.tensor(X_tr_n, dtype=torch.float32),
            torch.tensor(X_tr_c["primary_use"], dtype=torch.long),
            torch.tensor(y_train, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=512 if smoke else 1024, shuffle=True)
        model = _build_mlp(num_numeric_features=X_tr_n.shape[1], category_vocab_sizes=prep.num_categories, embedding_dim=8, hidden_dims=(256, 128, 64), dropout=0.2)
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        model.train()
        n_epochs = 1 if smoke else 15
        for _ in range(n_epochs):
            for b_num, b_cat, b_y in loader:
                b_num, b_cat, b_y = b_num.to(device), b_cat.to(device), b_y.to(device)
                opt.zero_grad()
                loss = crit(model(b_num, {"primary_use": b_cat}), b_y)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            te_num_t = torch.tensor(X_te_n, dtype=torch.float32)
            te_cat_t = {"primary_use": torch.tensor(X_te_c["primary_use"], dtype=torch.long)}
            all_preds = []
            for i in range(0, len(X_te_n), 8192):
                end = min(i + 8192, len(X_te_n))
                out = model(te_num_t[i:end].to(device), {k: v[i:end].to(device) for k, v in te_cat_t.items()})
                probs = torch.sigmoid(out).cpu().numpy()
                all_preds.append(probs)
        return np.concatenate(all_preds).astype(np.float64)

    tasks.append(("submission_21_mlp_challenger_quantile_0.8302.csv", "Tabular MLP Quantile Challenger", run_mlp_challenger))

    # Execute all tasks in sequence
    logger.info("Starting execution of %d remaining experiment tasks...", len(tasks))
    success_count = 0
    for idx, (filename, desc, fn) in enumerate(tasks, start=1):
        target_csv = out_dir / filename
        logger.info("[%d/%d] Starting model training for: %s -> %s", idx, len(tasks), desc, filename)
        t0 = time.time()
        try:
            probs = fn()
            sub = build_submission(row_ids, probs)
            validate_submission_contract(sub, expected_row_count=len(row_ids), expected_row_ids=sample_sub["row_id"])
            save_submission(sub, target_csv)
            elapsed = time.time() - t0
            logger.info("[%d/%d] Generated and validated %s in %.1f seconds", idx, len(tasks), target_csv.name, elapsed)

            if not smoke:
                # Submit to Kaggle
                msg = f"Kaggle LEAD Experiment: {desc}"
                ok = submit_to_kaggle(target_csv, msg)
                if ok:
                    success_count += 1
                time.sleep(3)
            else:
                success_count += 1
        except Exception:
            logger.exception("[%d/%d] Failed task %s", idx, len(tasks), desc)

    logger.info("Batch execution completed: %d/%d successfully processed.", success_count, len(tasks))


if __name__ == "__main__":
    main()
