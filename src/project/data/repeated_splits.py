"""Repeated development folds with an immutable, previously assigned vault."""

from __future__ import annotations

import json
import os
import tempfile
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

from project.data.splits import (
    STRATIFICATION_METHOD,
    _assign_support_strata,
    _summarize_buildings,
    _validate_seed,
    build_split_diagnostics,
)
from project.data.validate import compute_file_sha256

SUPPORT_COLUMNS = [
    "row_count", "anomaly_count", "normal_count", "anomaly_prevalence",
]


def _validate_original(original: pl.DataFrame, train: pl.LazyFrame) -> None:
    required = {"building_id", "split", "fold", *SUPPORT_COLUMNS}
    if missing := required.difference(original.columns):
        raise ValueError(f"Original manifest missing columns: {sorted(missing)}")
    ids = original.get_column("building_id")
    if ids.null_count() or ids.n_unique() != original.height:
        raise ValueError("Original manifest requires unique, non-null building IDs")
    if original["split"].null_count() or set(original["split"]) != {"dev", "vault"}:
        raise ValueError("Original manifest must contain only dev and vault splits")
    vault = original.filter(pl.col("split") == "vault")
    if vault["fold"].null_count() != vault.height:
        raise ValueError("Original vault folds must be null")
    # This coverage scan projects IDs only, including vault IDs but no vault labels.
    current_ids = train.select("building_id").unique().collect(engine="streaming")
    if current_ids["building_id"].null_count():
        raise ValueError("Training building_id contains null IDs")
    if set(current_ids["building_id"]) != set(ids):
        raise ValueError("Original manifest coverage differs from current training IDs")


def _partition_signature(dev: pl.DataFrame, n_folds: int) -> frozenset:
    """Represent membership independently of arbitrary fold labels."""
    return frozenset(
        frozenset(dev.filter(pl.col("fold") == fold)["building_id"].to_list())
        for fold in range(n_folds)
    )


def build_repeated_split_manifest(
    train: pl.DataFrame | pl.LazyFrame,
    original_manifest: pl.DataFrame,
    split_seeds: list[int],
    n_dev_folds: int = 4,
) -> pl.DataFrame:
    """Repartition only original DEV buildings, copying vault support verbatim.

    Sorted building summaries and splitter-local RNG seeds make assignments stable
    across input ordering. Seeds are used exactly as supplied, without retrying or
    selecting seeds on scores. Infeasible or equivalent partitions fail closed.
    """
    if not isinstance(split_seeds, list) or not split_seeds:
        raise ValueError("split_seeds must be a non-empty list of seeds")
    for seed in split_seeds:
        _validate_seed(seed)
    if len(set(split_seeds)) != len(split_seeds):
        raise ValueError("split_seeds must not contain duplicate seeds")
    if isinstance(n_dev_folds, bool) or not isinstance(n_dev_folds, Integral):
        raise TypeError("n_dev_folds must be an integer >= 2")
    if n_dev_folds < 2:
        raise ValueError("n_dev_folds must be at least 2")
    lazy_train = train.lazy() if isinstance(train, pl.DataFrame) else train
    _validate_original(original_manifest, lazy_train)
    old_dev = original_manifest.filter(pl.col("split") == "dev").sort("building_id")
    dev_ids = old_dev.select("building_id")
    # Filter/join BEFORE target validation or aggregation. Vault support is historical.
    summary = _summarize_buildings(
        lazy_train.join(dev_ids.lazy(), on="building_id", how="semi"),
        "building_id", "anomaly",
    )
    if summary.height < n_dev_folds or summary.height % n_dev_folds:
        raise ValueError("DEV buildings must divide evenly across n_dev_folds")
    for column in SUPPORT_COLUMNS:
        if not summary[column].equals(old_dev[column]):
            raise ValueError(f"Original DEV {column} differs from current prepared support")
    for column in ("anomaly_count", "normal_count"):
        if int((summary[column] > 0).sum()) < n_dev_folds:
            raise ValueError(f"Insufficient building-level {column} support for all folds")
    summary, _strata = _assign_support_strata(summary, n_dev_folds)
    support_columns = SUPPORT_COLUMNS + (
        ["support_stratum"] if "support_stratum" in original_manifest.columns else []
    )
    columns = ["building_id", "repeat", "seed", "split", "fold", *support_columns]
    vault = original_manifest.filter(pl.col("split") == "vault")
    manifests = []
    signatures = set()
    for repeat, seed in enumerate(split_seeds):
        folds = np.full(summary.height, -1, dtype=np.int64)
        splitter = StratifiedKFold(n_splits=n_dev_folds, shuffle=True, random_state=seed)
        for fold, (_, indices) in enumerate(
            splitter.split(np.zeros(summary.height), summary["support_stratum"].to_numpy())
        ):
            folds[indices] = fold
        dev = summary.with_columns(
            pl.Series("fold", folds, dtype=pl.Int64), pl.lit("dev").alias("split"),
        )
        for fold in range(n_dev_folds):
            part = dev.filter(pl.col("fold") == fold)
            if part.height != summary.height // n_dev_folds:
                raise ValueError("DEV folds must have equal building counts")
            if any(int(part[c].sum()) == 0 for c in ("anomaly_count", "normal_count")):
                raise ValueError(f"DEV repeat {repeat} fold {fold} requires both classes")
        signature = _partition_signature(dev, n_dev_folds)
        if signature in signatures:
            raise ValueError("Repeated partitions must be genuinely distinct, ignoring fold labels")
        signatures.add(signature)
        parts = [
            part.with_columns(
                pl.lit(repeat, dtype=pl.Int64).alias("repeat"),
                pl.lit(int(seed), dtype=pl.Int64).alias("seed"),
            ).select(columns)
            for part in (dev, vault)
        ]
        manifests.append(pl.concat(parts, how="vertical_relaxed"))
    return pl.concat(manifests).sort("repeat", "building_id")


def _repeat_diagnostics(part: pl.DataFrame, seed: int) -> dict[str, Any]:
    # Reuse diagnostics over compact manifest support only, never scan vault labels.
    if "support_stratum" not in part.columns:
        part = part.with_columns(pl.lit(None, dtype=pl.Int64).alias("support_stratum"))
    diagnostics = build_split_diagnostics(part, seed)
    dev_diagnostics = build_split_diagnostics(part.filter(pl.col("split") == "dev"), seed)
    return {
        "class_support": diagnostics["class_support"],
        "support_strata": dev_diagnostics["support_strata"],
        "per_building_support": diagnostics["per_building_support"],
    }


def write_repeated_split_artifacts(
    train: pl.DataFrame | pl.LazyFrame,
    source_manifest: Path,
    project_root: Path,
    split_seeds: list[int],
    n_dev_folds: int = 4,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Stage and publish new split artifacts, metadata last, without overwrites.

    Identical existing bytes are verified and left untouched. Atomic hard-link
    publication uses no-clobber filesystem semantics even with concurrent writers.
    On failure only this call's newly published links are rolled back.
    """
    root = project_root.resolve()
    destination = output_dir or root / "outputs" / "splits"
    destination = destination.resolve()
    # This lane writes only the live split directory, never archived run trees.
    if destination != root / "outputs" / "splits":
        raise ValueError("Repeated output_dir must be the project outputs/splits directory")
    source_manifest = source_manifest.resolve()
    source_hash = compute_file_sha256(source_manifest)
    original = pl.read_parquet(source_manifest)
    manifest = build_repeated_split_manifest(train, original, split_seeds, n_dev_folds)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination, prefix=".staging_repeated_") as staging:
        stage = Path(staging)
        master_name = "building_folds_repeated.parquet"
        metadata_name = "split_metadata_repeated.json"
        master = stage / master_name
        manifest.write_parquet(master)
        family_hash = compute_file_sha256(master)
        first = manifest.filter(pl.col("repeat") == 0)
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "repeat_count": len(split_seeds),
            "repeats": len(split_seeds),
            "n_dev_folds": int(n_dev_folds),
            "split_seeds": [int(seed) for seed in split_seeds],
            "stratification_method": STRATIFICATION_METHOD,
            "stratification_scope": "original DEV buildings only; minimum stratum size n_dev_folds",
            "vault_support_origin": "copied verbatim from source manifest; no new vault target analysis",
            "seed_policy": "predeclared seeds; no score-dependent selection or retries",
            "family_sha256": family_hash,
            "source_vault_manifest_path": source_manifest.relative_to(root).as_posix(),
            "source_vault_manifest_sha256": source_hash,
            "source_manifest": {
                "path": source_manifest.relative_to(root).as_posix(),
                "sha256": source_hash,
            },
            "vault_building_ids": original.filter(pl.col("split") == "vault")
            .sort("building_id")["building_id"].to_list(),
            "summary": {
                "building_count": first.height,
                "row_count": int(first["row_count"].sum()),
                "dev_building_count": first.filter(pl.col("split") == "dev").height,
                "vault_building_count": first.filter(pl.col("split") == "vault").height,
                "manifest_row_count": manifest.height,
            },
            "artifacts": {
                "building_folds_path": (destination / master_name).relative_to(root).as_posix(),
                "building_folds_sha256": family_hash,
            },
            "partitions_genuinely_distinct": True,
            "per_repeat": [],
        }
        names = [master_name]
        for repeat, seed in enumerate(split_seeds):
            part = manifest.filter(pl.col("repeat") == repeat)
            name = f"building_folds_r{repeat}_seed{seed}.parquet"
            part.write_parquet(stage / name)
            names.append(name)
            metadata["per_repeat"].append({
                "repeat": repeat, "seed": int(seed),
                "path": (destination / name).relative_to(root).as_posix(),
                "sha256": compute_file_sha256(stage / name),
                **_repeat_diagnostics(part, int(seed)),
            })
        (stage / metadata_name).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        names.append(metadata_name)
        if compute_file_sha256(source_manifest) != source_hash:
            raise ValueError("Source vault manifest changed during repeated split generation")
        # Preflight the entire family before publishing even a single new artifact.
        for name in names:
            final = destination / name
            if final.is_symlink() or final == source_manifest:
                raise ValueError(f"Unsafe repeated artifact destination: {name}")
            if final.exists() and compute_file_sha256(final) != compute_file_sha256(stage / name):
                raise FileExistsError(f"Refusing to overwrite different repeated artifact: {name}")
        published = []
        try:
            for name in names:
                final = destination / name
                if final.exists():
                    if final.is_symlink() or compute_file_sha256(final) != compute_file_sha256(stage / name):
                        raise FileExistsError(f"Refusing to overwrite different concurrent artifact: {name}")
                    continue
                try:
                    os.link(stage / name, final)
                except FileExistsError:
                    if final.is_symlink() or compute_file_sha256(final) != compute_file_sha256(stage / name):
                        raise FileExistsError(f"Refusing to overwrite different concurrent artifact: {name}") from None
                    continue
                published.append(final)
        except Exception:
            for final in reversed(published):
                if final.exists() and os.path.samestat(final.stat(), (stage / final.name).stat()):
                    final.unlink()
            raise
    return metadata
