#!/usr/bin/env python3
"""Run GBDT feature selection experiments (Passthrough vs. Selected).

Evaluates LightGBM, CatBoost, and XGBoost across the frozen 4x3 repeated split manifest.
Feature selection is strictly fold-local (computed on training partitions only).
Generates canonical OOF parquets conforming to CANDIDATE_OOF_SCHEMA and transport validation
JSON files recording fold-local selected features and stability metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from project.models.candidate_artifacts import (
    CANDIDATE_OOF_SCHEMA,
    write_candidate_oof_batches,
)
from project.models.gbdt_feature_selection import (
    SUPPORTED_FAMILIES,
    SUPPORTED_REGIMES,
    normalize_family,
    normalize_regime,
    train_fs_partition,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/experiments/gbdt_feature_selection_campaign.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/models/gbdt_feature_selection")


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


FORBIDDEN_EXACT_NAMES: frozenset[str] = frozenset({
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
})


def is_forbidden_predictor(name: str) -> bool:
    """Check if a column name is forbidden by anti-leakage or ID/metadata rules."""
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


def compute_selection_stability(
    fold_features: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Compute Jaccard similarity and selection frequency across folds.

    Parameters
    ----------
    fold_features:
        Mapping of fold identifier (e.g. '0:0') to list of selected feature names.

    Returns
    -------
    dict[str, Any]:
        Stability metrics including mean pairwise Jaccard similarity,
        pairwise Jaccard matrix, and per-feature selection frequencies.
    """
    if not fold_features:
        return {
            "mean_pairwise_jaccard": 1.0,
            "jaccard_matrix": {},
            "selection_frequency": {},
        }

    fold_keys = list(fold_features.keys())
    sets = {k: set(fold_features[k]) for k in fold_keys}
    all_features = sorted({feat for s in sets.values() for feat in s})

    # Selection frequency per feature
    n_folds = len(fold_keys)
    selection_frequency: dict[str, float] = {}
    for feat in all_features:
        count = sum(1 for s in sets.values() if feat in s)
        selection_frequency[feat] = float(count / n_folds)

    # Pairwise Jaccard matrix
    jaccard_matrix: dict[str, dict[str, float]] = {k: {} for k in fold_keys}
    pairwise_values: list[float] = []

    for i, k1 in enumerate(fold_keys):
        for j, k2 in enumerate(fold_keys):
            s1 = sets[k1]
            s2 = sets[k2]
            union_len = len(s1 | s2)
            sim = float(len(s1 & s2) / union_len) if union_len > 0 else 1.0
            jaccard_matrix[k1][k2] = round(sim, 4)
            if i < j:
                pairwise_values.append(sim)

    mean_jaccard = float(np.mean(pairwise_values)) if pairwise_values else 1.0

    return {
        "mean_pairwise_jaccard": round(mean_jaccard, 4),
        "jaccard_matrix": jaccard_matrix,
        "selection_frequency": {k: round(v, 4) for k, v in selection_frequency.items()},
    }


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GBDT Feature Selection (Passthrough vs. Selected) cross-validation."
    )
    parser.add_argument(
        "--family",
        default="all",
        choices=[*SUPPORTED_FAMILIES, "all"],
        help="GBDT family to train: lightgbm, catboost, xgboost, or all.",
    )
    parser.add_argument(
        "--regime",
        default="both",
        choices=[*SUPPORTED_REGIMES, "both"],
        help="Feature regime: passthrough (Control), selected (Challenger), or both.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=30,
        help="Number of top features to select under the 'selected' regime.",
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


def run_gbdt_feature_selection(
    family: str,
    regime: str,
    k: int,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path | None = None,
    smoke: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    """Execute GBDT feature selection cross-validation.

    Returns
    -------
    list[Path]:
        List of generated OOF parquet artifact paths.
    """
    models_cfg = config.get("models", {})
    families_to_run = list(SUPPORTED_FAMILIES) if family == "all" else [normalize_family(family)]
    regimes_to_run = list(SUPPORTED_REGIMES) if regime == "both" else [normalize_regime(regime)]

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

    split_df = pl.read_parquet(split_path)
    train_joined = pl.read_parquet(data_path)

    # Resolve predictor columns strictly from declared allowlist
    features_cfg = config.get("features", {})
    if features_cfg.get("predictors"):
        declared_predictors = list(features_cfg["predictors"])
    elif "numeric" in features_cfg or "categorical" in features_cfg:
        declared_predictors = list(features_cfg.get("numeric", [])) + list(features_cfg.get("categorical", []))
    else:
        raise ValueError(
            "Configuration must explicitly declare 'features.predictors' (or 'numeric'/'categorical') allowlist"
        )

    predictor_cols = validate_predictor_cols(declared_predictors)

    # Ensure all declared predictors exist in train_joined
    missing_cols = [c for c in predictor_cols if c not in train_joined.columns]
    if missing_cols:
        raise ValueError(f"Declared predictors not found in training dataset: {missing_cols}")

    # Guard: assert zero gte_* or *_id columns entered predictor pool
    assert not any(c.startswith("gte_") or c.endswith("_id") for c in predictor_cols), (
        "Predictor pool contains forbidden gte_* or *_id columns"
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

    for fam in families_to_run:
        if fam not in models_cfg:
            raise ValueError(f"Family '{fam}' not configured in {config_path}")

        model_spec = models_cfg[fam]
        base_params = dict(model_spec.get("params", {}))
        recipe_id = model_spec.get("recipe_id", f"recipe_gbdt_fs_{fam}")
        model_config_id = model_spec.get("model_config_id", f"cfg_gbdt_fs_{fam}_default")

        for reg_name in regimes_to_run:
            run_id = f"gbdt_fs_{fam}_{reg_name}_{execution_mode}_{uuid.uuid4().hex[:8]}"
            fit_config_sha = _json_sha256({"family": fam, "regime": reg_name, "k": k, "params": base_params})

            oof_dfs: list[pl.DataFrame] = []
            per_fold_selected_features: dict[str, list[str]] = {}

            logger.info(
                "Executing GBDT Feature Selection: family=%s, regime=%s, k=%d, mode=%s, repeats=%s, folds=%s",
                fam,
                reg_name,
                k,
                execution_mode,
                repeats,
                folds,
            )

            for rep in repeats:
                split_seed = 42 + rep
                for fld in folds:
                    fold_id = f"{rep}:{fld}"
                    fold_manifest = dev_splits.filter(pl.col("repeat") == rep)
                    train_b_ids = fold_manifest.filter(pl.col("fold") != fld)["building_id"].to_list()
                    val_b_ids = fold_manifest.filter(pl.col("fold") == fld)["building_id"].to_list()

                    if is_synthetic:
                        train_b_ids = train_b_ids[:4]
                        val_b_ids = val_b_ids[:2]

                    train_fold_df = train_dev.filter(pl.col("building_id").is_in(train_b_ids))
                    val_fold_df = train_dev.filter(pl.col("building_id").is_in(val_b_ids))

                    if val_fold_df.height == 0:
                        continue

                    partition_params = dict(base_params)
                    partition_params["seed"] = split_seed

                    preds, selected_feats = train_fs_partition(
                        model_family=fam,
                        regime=reg_name,
                        train_df=train_fold_df,
                        val_df=val_fold_df,
                        all_features=predictor_cols,
                        target_col=target_col,
                        k=k,
                        params=partition_params,
                        dry_run=dry_run,
                    )

                    per_fold_selected_features[fold_id] = selected_feats

                    fold_oof_df = val_fold_df.select([
                        pl.col("row_index").cast(pl.Int64),
                        pl.col("building_id").cast(pl.Utf8),
                        pl.col("timestamp").cast(pl.Datetime("ns")),
                        pl.col(target_col).cast(pl.Int64).alias("target"),
                    ]).with_columns([
                        pl.lit(f"{fam}_{reg_name}").alias("candidate_id"),
                        pl.lit(f"{recipe_id}_{reg_name}").alias("recipe_id"),
                        pl.lit(fam).alias("model_family"),
                        pl.lit(f"{model_config_id}_{reg_name}").alias("model_config_id"),
                        pl.lit(int(rep)).alias("repeat"),
                        pl.lit(int(fld)).alias("fold"),
                        pl.Series("prediction", preds, dtype=pl.Float64),
                        pl.lit(split_sha).alias("split_sha256"),
                        pl.lit(data_sha).alias("data_sha256"),
                        pl.lit(config_sha).alias("config_sha256"),
                        pl.lit(run_id).alias("run_id"),
                        pl.lit(int(split_seed)).alias("split_seed"),
                        pl.lit("regime_gbdt_feature_selection").alias("experiment_id"),
                        pl.lit("control" if reg_name == "passthrough" else "challenger").alias("arm_id"),
                        pl.lit("positive_probability").alias("score_kind"),
                        pl.lit(1).alias("positive_label"),
                        pl.lit(execution_mode).alias("execution_mode"),
                        pl.lit(non_comparative).alias("non_comparative"),
                        pl.lit(n03_eligible).alias("n03_eligible"),
                        pl.lit(fit_config_sha).alias("model_fit_config_sha256"),
                    ])
                    oof_dfs.append(fold_oof_df)

            if not oof_dfs:
                raise RuntimeError(f"No predictions generated for {fam}_{reg_name}.")

            oof_df = pl.concat(oof_dfs, how="vertical")
            arrow_table = oof_df.select(CANDIDATE_OOF_SCHEMA.names).to_arrow().cast(CANDIDATE_OOF_SCHEMA)

            # Ensure schema conformity and null-free invariant
            if arrow_table.schema.names != CANDIDATE_OOF_SCHEMA.names:
                raise ValueError("Generated OOF schema does not match CANDIDATE_OOF_SCHEMA")
            if any(oof_df[n].null_count() > 0 for n in CANDIDATE_OOF_SCHEMA.names):
                raise ValueError("Generated OOF records contain null values")
            if output_dir is not None:
                resolved_output_dir = Path(output_dir).resolve()
            else:
                base = DEFAULT_OUTPUT_DIR / "smoke" if smoke else DEFAULT_OUTPUT_DIR
                resolved_output_dir = base.resolve()

            dest_dir = (resolved_output_dir / execution_mode / f"{fam}_{reg_name}_{run_id}").resolve()
            dest_dir.mkdir(parents=True, exist_ok=True)
            oof_path = (dest_dir / f"{fam}_{reg_name}_candidate_oof.parquet").resolve()

            # Stream out in max 64,000 batches
            max_batch_size = 64_000
            batches = [
                arrow_table.slice(offset, min(max_batch_size, arrow_table.num_rows - offset))
                for offset in range(0, arrow_table.num_rows, max_batch_size)
            ]
            write_candidate_oof_batches(oof_path, batches, CANDIDATE_OOF_SCHEMA)

            stability_metrics = compute_selection_stability(per_fold_selected_features)

            manifest_data = {
                "schema_version": 1,
                "run_id": run_id,
                "family": fam,
                "regime": reg_name,
                "k": k,
                "arm": "control" if reg_name == "passthrough" else "challenger",
                "execution_mode": execution_mode,
                "non_comparative": non_comparative,
                "n03_eligible": n03_eligible,
                "records_count": arrow_table.num_rows,
                "split_sha256": split_sha,
                "data_sha256": data_sha,
                "config_sha256": config_sha,
                "model_fit_config_sha256": fit_config_sha,
                "oof_parquet": str(oof_path),
                "stability": stability_metrics,
                "per_fold_selected_features": per_fold_selected_features,
            }

            manifest_path = dest_dir / f"{fam}_{reg_name}_transport_validation.json"
            manifest_path.write_text(json.dumps(manifest_data, indent=2))
            logger.info("Successfully wrote transport validation JSON to %s", manifest_path)

            result_paths.append(oof_path)

    return result_paths


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    results = run_gbdt_feature_selection(
        family=args.family,
        regime=args.regime,
        k=args.k,
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        smoke=args.smoke,
        dry_run=args.dry_run,
    )
    for p in results:
        print(f"Generated OOF artifact: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
