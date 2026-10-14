#!/usr/bin/env python3
"""Run the 3-arm synthetic sampling campaign (natural vs. SMOTENC vs. Tabular Mixup).

Evaluates LightGBM and ExtraTrees across the frozen 4x3 repeated Stratified
Group K-Fold split manifest (grouped on ``building_id``) for 3 arms:
- natural: control, original training fold (~2.08% positives)
- smote:   fold-local SMOTENC sampling of the positive class to ~8%
- mixup:   fold-local Tabular Mixup sampling of the positive class to ~8%

Synthetic rows are generated strictly from each outer-training fold; validation
folds are never resampled. Writes canonical OOF parquets conforming to
CANDIDATE_OOF_SCHEMA and transport validation JSON files recording run lineage,
sampling configuration and SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from project.models.candidate_artifacts import (
    CANDIDATE_OOF_SCHEMA,
    write_candidate_oof_batches,
)
from project.models.synthetic_sampling import (
    DEFAULT_K_NEIGHBORS,
    DEFAULT_MIXUP_ALPHA,
    DEFAULT_TARGET_RATIO,
    SUPPORTED_ARMS,
    SUPPORTED_FAMILIES,
    normalize_arm,
    normalize_family,
    train_synthetic_sampling_partition,
    validate_predictor_cols,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/experiments/synthetic_sampling_campaign.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/models/synthetic_sampling")
EXPERIMENT_ID = "regime_synthetic_sampling"
MAX_BATCH_ROWS = 64_000


def _file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file; of the path string when the file does not exist."""
    if not path.is_file():
        return hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(obj: Any) -> str:
    """Hex SHA-256 of a canonical JSON serialisation."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the 3-arm synthetic sampling cross-validation campaign.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--family",
        type=str,
        default="all",
        choices=["all", *SUPPORTED_FAMILIES],
        help="Model family to evaluate, or 'all'.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="all",
        choices=["all", *SUPPORTED_ARMS],
        help="Sampling arm to evaluate, or 'all'.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke mode: 1 repeat, 2 folds, subset of buildings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Real fold-local sampling but deterministic mock scoring (no model fits).",
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
    """Load the campaign configuration mapping."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a dictionary")
    return config


def _resolve_predictors(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return validated (predictors, categorical) from the config allowlist."""
    features_cfg = config.get("features", {})
    declared = features_cfg.get("predictors")
    if not declared:
        raise ValueError(
            "Configuration must explicitly declare 'features.predictors' allowlist"
        )
    predictors = validate_predictor_cols(list(declared))
    categorical = list(features_cfg.get("categorical", []))
    outside = [c for c in categorical if c not in predictors]
    if outside:
        raise ValueError(f"Categorical columns not in predictors allowlist: {outside}")
    return predictors, categorical


def _arm_sampling_params(config: dict[str, Any], arm: str) -> dict[str, Any]:
    """Sampling params for an arm (global ``sampling`` overridden by ``arm_definitions``)."""
    sampling = config.get("sampling", {})
    arm_def = config.get("arm_definitions", {}).get(arm, {})
    return {
        "target_ratio": float(
            arm_def.get("target_ratio", sampling.get("target_ratio", DEFAULT_TARGET_RATIO))
        ),
        "k_neighbors": int(
            arm_def.get("k_neighbors", sampling.get("k_neighbors", DEFAULT_K_NEIGHBORS))
        ),
        "alpha": float(
            arm_def.get(
                "alpha", sampling.get("mixup_alpha", DEFAULT_MIXUP_ALPHA)
            )
        ),
    }


def run_synthetic_sampling(
    family: str,
    arm: str,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path | None = None,
    smoke: bool = False,
    dry_run: bool = False,
    split_df: pl.DataFrame | None = None,
    train_df: pl.DataFrame | None = None,
) -> list[Path]:
    """Execute the synthetic sampling cross-validation.

    Parameters
    ----------
    family:
        ``lightgbm``, ``extratrees`` or ``all``.
    arm:
        ``natural``, ``smote``, ``mixup`` or ``all``.
    config:
        Parsed campaign YAML mapping.
    config_path:
        Path of the YAML config (hashed into lineage when it exists).
    output_dir:
        Optional output root for OOF parquets and transport JSONs.
    smoke:
        Restrict to repeat 0, folds 0-1 and 4 train / 2 validation buildings.
    dry_run:
        Real fold-local sampling, mock scoring.
    split_df, train_df:
        Optional preloaded split manifest / joined training data (for testing).

    Returns
    -------
    list[Path]:
        Generated OOF parquet paths (one per family x arm).
    """
    models_cfg = config.get("models", {})
    families = (
        list(SUPPORTED_FAMILIES) if family == "all" else [normalize_family(family)]
    )
    arms = list(SUPPORTED_ARMS) if arm == "all" else [normalize_arm(arm)]

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

    predictor_cols, categorical_cols = _resolve_predictors(config)
    missing_cols = [c for c in predictor_cols if c not in train_joined.columns]
    if missing_cols:
        raise ValueError(
            f"Declared predictors not found in training dataset: {missing_cols}"
        )

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
        output_root = (DEFAULT_OUTPUT_DIR / "smoke" if smoke else DEFAULT_OUTPUT_DIR).resolve()

    result_paths: list[Path] = []
    for fam in families:
        if fam not in models_cfg:
            raise ValueError(f"Family '{fam}' not configured in {config_path}")
        model_spec = models_cfg[fam]
        base_params = dict(model_spec.get("base_params", {}))
        recipe_id = model_spec.get("recipe_id", f"recipe_synthetic_sampling_{fam}")
        model_config_id = model_spec.get(
            "model_config_id", f"cfg_synthetic_sampling_{fam}_default"
        )

        for arm_id in arms:
            run_id = f"synthetic_sampling_{fam}_{arm_id}_{execution_mode}_{uuid.uuid4().hex[:8]}"
            sampling_params = _arm_sampling_params(config, arm_id)
            fit_config_sha = _json_sha256(
                {
                    "family": fam,
                    "arm": arm_id,
                    "base_params": base_params,
                    "sampling": sampling_params,
                    "categorical": categorical_cols,
                }
            )

            logger.info(
                "Executing synthetic sampling: family=%s arm=%s mode=%s repeats=%s folds=%s",
                fam,
                arm_id,
                execution_mode,
                repeats,
                folds,
            )

            oof_dfs: list[pl.DataFrame] = []
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
                    val_fold_df = train_dev.filter(
                        pl.col("building_id").is_in(val_b_ids)
                    )
                    if val_fold_df.height == 0:
                        continue

                    partition_params = {
                        **base_params,
                        **sampling_params,
                        "seed": split_seed,
                    }
                    preds = train_synthetic_sampling_partition(
                        model_family=fam,
                        arm=arm_id,
                        train_df=train_fold_df,
                        val_df=val_fold_df,
                        feature_cols=predictor_cols,
                        categorical_cols=categorical_cols,
                        target_col=target_col,
                        params=partition_params,
                        dry_run=dry_run,
                    )

                    oof_dfs.append(
                        val_fold_df.select(
                            [
                                pl.col("row_index").cast(pl.Int64),
                                pl.col("building_id").cast(pl.Utf8),
                                pl.col("timestamp").cast(pl.Datetime("ns")),
                                pl.col(target_col).cast(pl.Int64).alias("target"),
                            ]
                        ).with_columns(
                            [
                                pl.lit(f"{fam}_{arm_id}").alias("candidate_id"),
                                pl.lit(f"{recipe_id}_{arm_id}").alias("recipe_id"),
                                pl.lit(fam).alias("model_family"),
                                pl.lit(f"{model_config_id}_{arm_id}").alias(
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
                                pl.lit(EXPERIMENT_ID).alias("experiment_id"),
                                pl.lit(arm_id).alias("arm_id"),
                                pl.lit("positive_probability").alias("score_kind"),
                                pl.lit(1).alias("positive_label"),
                                pl.lit(execution_mode).alias("execution_mode"),
                                pl.lit(non_comparative).alias("non_comparative"),
                                pl.lit(n03_eligible).alias("n03_eligible"),
                                pl.lit(fit_config_sha).alias("model_fit_config_sha256"),
                            ]
                        )
                    )

            if not oof_dfs:
                raise RuntimeError(
                    f"No out-of-fold predictions generated for {fam}_{arm_id}."
                )

            oof_df = pl.concat(oof_dfs, how="vertical")
            arrow_table = (
                oof_df.select(CANDIDATE_OOF_SCHEMA.names)
                .to_arrow()
                .cast(CANDIDATE_OOF_SCHEMA)
            )
            if any(oof_df[n].null_count() > 0 for n in CANDIDATE_OOF_SCHEMA.names):
                raise ValueError("Generated OOF records contain null values")

            dest_dir = (output_root / execution_mode / f"{fam}_{arm_id}_{run_id}").resolve()
            dest_dir.mkdir(parents=True, exist_ok=True)
            oof_path = (dest_dir / f"{fam}_{arm_id}_candidate_oof.parquet").resolve()
            batches = [
                arrow_table.slice(
                    offset, min(MAX_BATCH_ROWS, arrow_table.num_rows - offset)
                )
                for offset in range(0, arrow_table.num_rows, MAX_BATCH_ROWS)
            ]
            write_candidate_oof_batches(oof_path, batches, CANDIDATE_OOF_SCHEMA)

            manifest_data = {
                "schema_version": 1,
                "run_id": run_id,
                "family": fam,
                "arm": arm_id,
                "execution_mode": execution_mode,
                "non_comparative": non_comparative,
                "n03_eligible": n03_eligible,
                "records_count": arrow_table.num_rows,
                "split_sha256": split_sha,
                "data_sha256": data_sha,
                "config_sha256": config_sha,
                "model_fit_config_sha256": fit_config_sha,
                "oof_parquet": str(oof_path),
                "arm_definition": config.get("arm_definitions", {}).get(arm_id, {}),
                "sampling_params": sampling_params,
                "categorical_cols": categorical_cols,
                "fold_local_sampling": arm_id != "natural",
                "validation_synthetic_rows": 0,
            }
            manifest_path = dest_dir / f"{fam}_{arm_id}_transport_validation.json"
            manifest_path.write_text(json.dumps(manifest_data, indent=2))
            logger.info("Wrote transport validation JSON to %s", manifest_path)
            result_paths.append(oof_path)

    return result_paths


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    results = run_synthetic_sampling(
        family=args.family,
        arm=args.arm,
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        smoke=args.smoke,
        dry_run=args.dry_run,
    )
    logger.info(
        "Synthetic sampling completed successfully (%d OOF files generated).",
        len(results),
    )
    for p in results:
        logger.info("Artifact: %s", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
