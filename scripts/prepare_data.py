#!/usr/bin/env python3
"""Data preparation script for LEAD competition.

Validates raw CSVs, checks schemas, hashes, keys, duplicate copies,
and joins features into Parquet artifacts using Polars streaming sinks
under outputs/prepared/ with comprehensive lineage under outputs/lineage/row_lineage.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import yaml

from project.data.join import (
    DEFAULT_TEST_DUPLICATE_COPIES,
    DEFAULT_TRAIN_DUPLICATE_COPIES,
    build_feature_joins,
)
from project.data.load import (
    load_data_contract,
    scan_raw_table,
)
from project.data.splits import (
    MAX_RANDOM_SEED,
    build_split_diagnostics,
    build_split_manifest,
)
from project.data.validate import (
    compute_file_sha256,
    validate_raw_file_metadata,
    validate_schema,
    validate_schema_types,
)
from project.paths import (
    RAW_CSV_FILENAMES,
    SAMPLE_SUBMISSION_CSV,
    TEST_CSV,
    TEST_FEATURES_CSV,
    TRAIN_CSV,
    TRAIN_FEATURES_CSV,
    ProjectPaths,
)


class PublicationRecoveryError(RuntimeError):
    """Raised when atomic publication fails and rollback also encounters errors."""

    def __init__(self, message: str, recovery_paths: list[str]):
        super().__init__(message)
        self.recovery_paths = recovery_paths


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LEAD data and validate contracts.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke mode on a bounded sample without writing full Parquet.",
    )
    group.add_argument(
        "--materialize",
        action="store_true",
        help="Run full validation and materialize Parquet under outputs/prepared/.",
    )
    group.add_argument(
        "--build-splits",
        action="store_true",
        help="Build a building-isolated development/vault manifest under outputs/splits/.",
    )
    group.add_argument(
        "--build-repeated-splits", action="store_true",
        help="Build repeated DEV folds from prepared data, preserving the original vault.",
    )
    parser.add_argument(
        "--split-config", type=Path, default=None,
        help="Repeated split YAML configuration (default: configs/split_repeated.yaml).",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=10000,
        help="Number of rows for bounded smoke sample (default: 10000, must be positive).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Root directory of the project.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing the raw CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output Parquet and lineage files (must be within outputs/).",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=None,
        help="Path to data_contract.yaml file.",
    )
    parsed = parser.parse_args(args)
    if parsed.split_config is not None and not parsed.build_repeated_splits:
        parser.error("--split-config is supported only with --build-repeated-splits")
    if parsed.smoke and parsed.sample_rows <= 0:
        parser.error(f"--sample-rows must be a positive integer, got {parsed.sample_rows}")
    return parsed


def publish_atomic_artifacts(
    staged_to_final: Mapping[Path, Path],
    staged_lineage: Path,
    final_lineage: Path,
) -> None:
    """Atomically publish staged artifacts to final destination paths with backup and rollback.

    Lineage is published last as the completion marker.
    """
    backups: dict[Path, Path] = {}
    promoted: list[tuple[Path, Path]] = []
    timestamp_tag = int(time.time() * 1_000_000)

    targets_in_order = list(staged_to_final.items()) + [(staged_lineage, final_lineage)]
    try:
        # 1. Back up existing final targets
        for staged_file, final_target in targets_in_order:
            if not staged_file.is_file():
                raise FileNotFoundError(f"Staged file missing before publication: {staged_file}")
            if final_target.exists():
                bak_path = final_target.with_name(f"{final_target.name}.bak_{timestamp_tag}")
                os.replace(final_target, bak_path)
                backups[final_target] = bak_path

        # 2. Promote staged files to final targets
        for staged_file, final_target in targets_in_order:
            os.replace(staged_file, final_target)
            promoted.append((final_target, staged_file))

    except Exception as primary_exc:
        rollback_errors: list[str] = []
        # Roll back: restore any backed-up files and remove promoted staged files
        for final_target, _staged in promoted:
            if final_target in backups:
                try:
                    os.replace(backups[final_target], final_target)
                    del backups[final_target]
                except OSError as rollback_exc:
                    rollback_errors.append(
                        f"Failed to restore {final_target} from backup {backups[final_target]}: {rollback_exc}"
                    )
            elif final_target.exists():
                try:
                    final_target.unlink()
                except OSError as rollback_exc:
                    rollback_errors.append(
                        f"Failed to clean up partially promoted {final_target}: {rollback_exc}"
                    )

        for final_target, bak_path in list(backups.items()):
            if bak_path.exists():
                try:
                    os.replace(bak_path, final_target)
                    del backups[final_target]
                except OSError as rollback_exc:
                    rollback_errors.append(
                        f"Failed to restore {final_target} from backup {bak_path}: {rollback_exc}"
                    )

        if rollback_errors:
            recovery_paths = [str(b) for b in backups.values()] + [str(s) for s in staged_to_final]
            error_msg = (
                f"Atomic publication failed ({primary_exc}) and rollback encountered errors: "
                f"{'; '.join(rollback_errors)}. "
                f"Recovery artifacts preserved at: {recovery_paths}"
            )
            raise PublicationRecoveryError(error_msg, recovery_paths) from primary_exc

        raise

    else:
        # Success: remove all backups
        for bak_path in backups.values():
            if bak_path.exists():
                try:
                    bak_path.unlink()
                except OSError:
                    pass


def _validated_split_config(
    split_config: Mapping[str, object],
) -> tuple[str, str, float, int, int]:
    allowed_fields = {
        "building_col",
        "target_col",
        "vault_fraction",
        "n_dev_folds",
        "seed",
    }
    if not all(isinstance(field, str) for field in split_config):
        raise TypeError("Split config field names must be strings")
    unknown_fields = sorted(set(split_config).difference(allowed_fields))
    if unknown_fields:
        raise ValueError(f"Unknown split config field(s): {unknown_fields}")

    building_col = split_config.get("building_col", "building_id")
    target_col = split_config.get("target_col", "anomaly")
    if not isinstance(building_col, str):
        raise TypeError("Split config 'building_col' must be a string")
    if not building_col.strip():
        raise ValueError("Split config 'building_col' must be a non-empty string")
    if not isinstance(target_col, str):
        raise TypeError("Split config 'target_col' must be a string")
    if not target_col.strip():
        raise ValueError("Split config 'target_col' must be a non-empty string")

    vault_fraction = split_config.get("vault_fraction", 0.2)
    if isinstance(vault_fraction, bool) or not isinstance(vault_fraction, (int, float)):
        raise TypeError("Split config 'vault_fraction' must be numeric")

    n_dev_folds = split_config.get("n_dev_folds", 4)
    if isinstance(n_dev_folds, bool) or not isinstance(n_dev_folds, int):
        raise TypeError("Split config 'n_dev_folds' must be an integer")

    seed = split_config.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("Split config 'seed' must be an integer")
    if not 0 <= seed <= MAX_RANDOM_SEED:
        raise ValueError(
            f"Split config 'seed' must be between 0 and {MAX_RANDOM_SEED}"
        )

    return (
        building_col.strip(),
        target_col.strip(),
        float(vault_fraction),
        n_dev_folds,
        seed,
    )


def _run_build_splits(
    project_paths: ProjectPaths,
    raw_data_dir: Path,
    outputs_dir: Path,
) -> int:
    allowed_outputs_base = project_paths.outputs_dir.resolve()
    resolved_outputs_dir = outputs_dir.resolve()
    try:
        resolved_outputs_dir.relative_to(allowed_outputs_base)
    except ValueError:
        print(
            f"Error: Output directory {resolved_outputs_dir} must be within {allowed_outputs_base}",
            file=sys.stderr,
        )
        return 1

    split_dir = outputs_dir / "splits"
    if split_dir.is_symlink():
        resolved_split_dir = split_dir.resolve()
        try:
            resolved_split_dir.relative_to(allowed_outputs_base)
        except ValueError:
            print(
                f"Error: Split directory {split_dir} is a symlink resolving to "
                f"{resolved_split_dir}, which is outside {allowed_outputs_base}",
                file=sys.stderr,
            )
            return 1

    train_csv = raw_data_dir / TRAIN_CSV
    split_config_path = project_paths.root / "configs" / "split.yaml"
    if not train_csv.is_file():
        print(f"Error: Required training CSV not found at {train_csv}", file=sys.stderr)
        return 1
    if not split_config_path.is_file():
        print(f"Error: Split config not found at {split_config_path}", file=sys.stderr)
        return 1

    with split_config_path.open("r", encoding="utf-8") as config_file:
        split_config = yaml.safe_load(config_file)
    if not isinstance(split_config, Mapping):
        raise TypeError(
            f"Split config must contain a YAML mapping: {split_config_path}"
        )

    building_col, target_col, vault_fraction, n_dev_folds, seed = (
        _validated_split_config(split_config)
    )

    print(f"Building split manifest from {train_csv}...")
    manifest = build_split_manifest(
        scan_raw_table(train_csv),
        building_col=building_col,
        target_col=target_col,
        vault_fraction=vault_fraction,
        n_dev_folds=n_dev_folds,
        seed=seed,
    )
    diagnostics = build_split_diagnostics(manifest, seed, building_col=building_col)
    diagnostics["hashes"] = {
        "split_config_sha256": compute_file_sha256(split_config_path),
        "train_csv_sha256": compute_file_sha256(train_csv),
    }
    diagnostics["split_config"] = dict(split_config)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    resolved_outputs_dir = outputs_dir.resolve()
    try:
        resolved_outputs_dir.relative_to(allowed_outputs_base)
    except ValueError:
        print(
            f"Error: Output directory {outputs_dir} resolves to {resolved_outputs_dir}, "
            f"which is outside {allowed_outputs_base}",
            file=sys.stderr,
        )
        return 1

    split_dir.mkdir(parents=True, exist_ok=True)
    resolved_split_dir = split_dir.resolve()
    try:
        resolved_split_dir.relative_to(allowed_outputs_base)
    except ValueError:
        print(
            f"Error: Split directory {split_dir} resolves to {resolved_split_dir}, "
            f"which is outside {allowed_outputs_base}",
            file=sys.stderr,
        )
        return 1

    staging_dir = Path(tempfile.mkdtemp(dir=outputs_dir, prefix=".staging_splits_"))
    preserve_staging = False
    try:
        staged_manifest = staging_dir / "building_folds.parquet"
        staged_metadata = staging_dir / "split_metadata.json"
        final_manifest = split_dir / staged_manifest.name
        final_metadata = split_dir / staged_metadata.name

        manifest.write_parquet(staged_manifest)
        diagnostics["artifacts"] = {
            "building_folds_path": str(final_manifest),
            "building_folds_sha256": compute_file_sha256(staged_manifest),
        }
        with staged_metadata.open("w", encoding="utf-8") as metadata_file:
            json.dump(diagnostics, metadata_file, indent=2)
            metadata_file.write("\n")

        publish_atomic_artifacts(
            staged_to_final={staged_manifest: final_manifest},
            staged_lineage=staged_metadata,
            final_lineage=final_metadata,
        )
    except PublicationRecoveryError:
        preserve_staging = True
        raise
    finally:
        if not preserve_staging and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"Split manifest: {final_manifest}")
    print(f"Split diagnostics: {final_metadata}")
    print(json.dumps(diagnostics["summary"], sort_keys=True))
    for partition in [
        diagnostics["class_support"]["vault"],
        *diagnostics["class_support"]["development_folds"],
    ]:
        label = (
            "vault" if partition["fold"] is None else f"dev fold {partition['fold']}"
        )
        print(
            f"  {label}: buildings={partition['building_count']}, rows={partition['row_count']}, "
            f"positive={partition['positive_rows']}, negative={partition['negative_rows']}, "
            f"prevalence={partition['anomaly_prevalence']:.6f}"
        )
    return 0


def _run_build_repeated_splits(project_paths: ProjectPaths, parsed: argparse.Namespace) -> int:
    from project.data.repeated_splits import write_repeated_split_artifacts

    root = project_paths.root.resolve()
    config_path = parsed.split_config or root / "configs" / "split_repeated.yaml"
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    fields = {"source_manifest", "prepared_train", "output_dir", "n_dev_folds", "split_seeds"}
    if not isinstance(config, Mapping) or set(config) != fields:
        raise ValueError(f"Repeated split config must contain exactly {sorted(fields)}")
    paths = {}
    for field in ("source_manifest", "prepared_train", "output_dir"):
        value = config[field]
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise ValueError(f"Repeated config {field} must be a relative path")
        path = (root / value).resolve()
        path.relative_to(root)
        paths[field] = path
    if parsed.output_dir is not None or parsed.data_dir is not None:
        raise ValueError("Repeated splits use prepared_train and output_dir from --split-config")
    metadata = write_repeated_split_artifacts(
        pl.scan_parquet(paths["prepared_train"]), paths["source_manifest"], root,
        config["split_seeds"], config["n_dev_folds"], output_dir=paths["output_dir"],
    )
    print(f"Repeated split manifest: {metadata['artifacts']['building_folds_path']}")
    print(f"Family SHA256: {metadata['family_sha256']}")
    print(json.dumps(metadata["summary"], sort_keys=True))
    return 0


def _run_prepare_data(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    project_paths = (
        ProjectPaths.from_root(parsed.project_root)
        if parsed.project_root
        else ProjectPaths.resolve()
    )
    raw_data_dir = parsed.data_dir or project_paths.raw_data_dir
    outputs_dir = parsed.output_dir or project_paths.outputs_dir

    if parsed.build_repeated_splits:
        return _run_build_repeated_splits(project_paths, parsed)
    if parsed.build_splits:
        return _run_build_splits(project_paths, raw_data_dir, outputs_dir)

    tracemalloc.start()
    start_time = time.perf_counter()

    # Constrain output directory to be within outputs_dir
    allowed_outputs_base = project_paths.outputs_dir.resolve()
    resolved_output_dir = outputs_dir.resolve()
    try:
        resolved_output_dir.relative_to(allowed_outputs_base)
    except ValueError:
        print(
            f"Error: Output directory {resolved_output_dir} must be within {allowed_outputs_base}",
            file=sys.stderr,
        )
        return 1

    lineage_dir = outputs_dir / "lineage"
    prepared_dir = outputs_dir / "prepared"

    # Validate that neither child directory is an external symlink or resolves outside allowed_outputs_base
    for child_dir in (prepared_dir, lineage_dir):
        if child_dir.is_symlink():
            resolved_child = child_dir.resolve()
            try:
                resolved_child.relative_to(allowed_outputs_base)
            except ValueError:
                print(
                    f"Error: Child directory {child_dir} is a symlink resolving to {resolved_child}, which is outside {allowed_outputs_base}",
                    file=sys.stderr,
                )
                return 1

        child_dir.mkdir(parents=True, exist_ok=True)
        resolved_child = child_dir.resolve()
        try:
            resolved_child.relative_to(allowed_outputs_base)
        except ValueError:
            print(
                f"Error: Child directory {child_dir} resolves to {resolved_child}, which is outside {allowed_outputs_base}",
                file=sys.stderr,
            )
            return 1

    contract = load_data_contract(parsed.contract)

    # Validate exact contract raw-file coverage against the five expected source CSVs
    contract_raw_files = set(contract.get("raw_files", {}).keys())
    expected_raw_files = set(RAW_CSV_FILENAMES)
    if contract_raw_files != expected_raw_files:
        raise ValueError(
            f"Contract raw_files must exactly match expected files {sorted(expected_raw_files)}, "
            f"got {sorted(contract_raw_files)}"
        )

    file_paths = {
        "train": raw_data_dir / TRAIN_CSV,
        "test": raw_data_dir / TEST_CSV,
        "sample_submission": raw_data_dir / SAMPLE_SUBMISSION_CSV,
        "train_features": raw_data_dir / TRAIN_FEATURES_CSV,
        "test_features": raw_data_dir / TEST_FEATURES_CSV,
    }

    # Verify all raw files exist
    for name, path in file_paths.items():
        if not path.is_file():
            print(f"Error: Required file '{name}' not found at {path}", file=sys.stderr)
            return 1

    mode = "smoke" if parsed.smoke else "materialize"
    print(f"=== Starting LEAD Data Preparation (mode: {mode}) ===")
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Outputs directory:  {outputs_dir}")

    raw_file_metadata: dict[str, dict] = {}
    hashes_validated = False
    row_counts_validated = False

    if parsed.materialize:
        print("Validating raw file hashes and row counts against contract...")
        for filename, spec in contract["raw_files"].items():
            file_key = filename.replace(".csv", "")
            p = file_paths[file_key]
            meta = validate_raw_file_metadata(
                path=p,
                expected_sha256=spec["sha256"],
                expected_rows=spec["row_count"],
            )
            raw_file_metadata[filename] = meta
            print(f"  [OK] {filename}: sha256 matches, rows={meta['rows']}")
        hashes_validated = True
        row_counts_validated = True
    else:
        for filename in contract["raw_files"]:
            file_key = filename.replace(".csv", "")
            p = file_paths[file_key]
            raw_file_metadata[filename] = {
                "filename": filename,
                "path": str(p),
                "bytes": p.stat().st_size,
            }

    print("Scanning data tables and validating schemas against contract...")
    source_scans: dict[str, pl.LazyFrame] = {}
    source_schemas: dict[str, dict[str, str]] = {}

    for name, p in file_paths.items():
        lf = scan_raw_table(p)
        filename = f"{name}.csv"
        file_spec = contract["raw_files"][filename]
        req_cols = file_spec.get("required_columns", [])
        table_schema_types = file_spec.get("schema_types", {})

        # Validate schema columns and table-specific dtypes
        schema_snapshot = validate_schema(lf, required_columns=req_cols, table_name=name)
        validate_schema_types(lf, schema_types=table_schema_types, table_name=name)

        source_schemas[filename] = schema_snapshot
        source_scans[name] = lf

    schemas_validated = True

    tables: dict[str, pl.LazyFrame] = {}
    if parsed.smoke:
        sample_rows = parsed.sample_rows
        print(f"Smoke mode: taking first {sample_rows} rows from each table scan...")
        for name, lf in source_scans.items():
            tables[name] = lf.head(sample_rows)
    else:
        tables = source_scans

    print("Building lazy feature joins and validating integrity...")
    train_joined_lf, test_joined_lf = build_feature_joins(tables)

    # Use a staging directory under outputs_dir to guarantee same-filesystem atomic promotion
    staging_dir = Path(tempfile.mkdtemp(dir=outputs_dir, prefix=".staging_"))
    preserve_staging = False

    try:
        train_out_path = None
        test_out_path = None
        train_sha256 = None
        test_sha256 = None
        train_shape = None
        test_shape = None

        staged_to_final: dict[Path, Path] = {}

        if parsed.materialize:
            final_train_out = prepared_dir / "train_joined.parquet"
            final_test_out = prepared_dir / "test_joined.parquet"

            staged_train = staging_dir / "train_joined.parquet"
            staged_test = staging_dir / "test_joined.parquet"

            print(f"Streaming train_joined Parquet directly to staging {staged_train}...")
            train_joined_lf.sink_parquet(staged_train, engine="streaming")
            print(f"Streaming test_joined Parquet directly to staging {staged_test}...")
            test_joined_lf.sink_parquet(staged_test, engine="streaming", maintain_order=True)

            train_sha256 = compute_file_sha256(staged_train)
            test_sha256 = compute_file_sha256(staged_test)

            staged_to_final[staged_train] = final_train_out
            staged_to_final[staged_test] = final_test_out

            train_out_path = str(final_train_out)
            test_out_path = str(final_test_out)
            train_shape = [contract["raw_files"]["train.csv"]["row_count"], len(train_joined_lf.collect_schema())]
            test_shape = [contract["raw_files"]["test.csv"]["row_count"], len(test_joined_lf.collect_schema())]
        else:
            # In smoke mode, collect the small sample to verify shapes
            smoke_train = train_joined_lf.collect()
            smoke_test = test_joined_lf.collect()
            train_shape = list(smoke_train.shape)
            test_shape = list(smoke_test.shape)
            print(f"Smoke join successful! Sample train shape: {train_shape}, sample test shape: {test_shape}")

        elapsed_sec = time.perf_counter() - start_time
        _curr_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        python_traced_peak_mb = peak_mem / (1024 * 1024)

        lineage_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "mode": mode,
            "sample_rows": parsed.sample_rows if parsed.smoke else None,
            "raw_file_metadata": raw_file_metadata,
            "source_schemas": source_schemas,
            "dropped_duplicate_columns": {
                "train_features": list(DEFAULT_TRAIN_DUPLICATE_COPIES),
                "test_features": list(DEFAULT_TEST_DUPLICATE_COPIES),
            },
            "validations": {
                "hashes_validated": hashes_validated,
                "row_counts_validated": row_counts_validated,
                "schemas_validated": schemas_validated,
                "unique_keys_validated": True,
                "one_to_one_joins_validated": True,
                "duplicate_copy_equality_validated": True,
                "train_test_building_overlap": 0,
                "sample_submission_order_preserved": True,
            },
            "joined_tables": {
                "train": {
                    "shape": train_shape,
                    "columns": train_joined_lf.collect_schema().names(),
                    "output_path": train_out_path,
                    "sha256": train_sha256,
                },
                "test": {
                    "shape": test_shape,
                    "columns": test_joined_lf.collect_schema().names(),
                    "output_path": test_out_path,
                    "sha256": test_sha256,
                },
            },
            "resource_metrics": {
                "elapsed_seconds": round(elapsed_sec, 3),
                "python_traced_peak_memory_mb": round(python_traced_peak_mb, 2),
                "memory_note": (
                    "Streaming sink avoids full in-memory table collection during materialize, "
                    "but build-side hash tables and buffers can consume memory proportional to unique keys. "
                    "Measure process peak RSS externally before full materialization."
                ),
            },
        }

        # Distinct lineage files: row_lineage.json for materialize, row_lineage_smoke.json for smoke
        lineage_filename = "row_lineage_smoke.json" if parsed.smoke else "row_lineage.json"
        staged_lineage = staging_dir / lineage_filename
        final_lineage = lineage_dir / lineage_filename

        print(f"Writing lineage to staging {staged_lineage}...")
        with open(staged_lineage, "w", encoding="utf-8") as f:
            json.dump(lineage_record, f, indent=2)

        # Atomically promote staged outputs to final paths with backup and rollback
        publish_atomic_artifacts(
            staged_to_final=staged_to_final,
            staged_lineage=staged_lineage,
            final_lineage=final_lineage,
        )

        if parsed.smoke:
            print(f"Smoke completed successfully. Lineage published to {final_lineage}.")
            print(f"Resource metrics: elapsed = {elapsed_sec:.2f}s, Python-traced peak memory = {python_traced_peak_mb:.2f} MB")
            print(
                "Note: Streaming sink avoids full in-memory table collection, but join hash tables and buffers "
                "can consume memory proportional to unique keys. Full-run RSS must be measured externally before full execution."
            )
        else:
            print(f"Materialization completed successfully via streaming sinks. Lineage published to {final_lineage}.")
            print(f"Resource metrics: elapsed = {elapsed_sec:.2f}s, Python-traced peak memory = {python_traced_peak_mb:.2f} MB")

    except PublicationRecoveryError:
        preserve_staging = True
        raise
    finally:
        if not preserve_staging and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    return 0


def main(args: list[str] | None = None) -> int:
    try:
        return _run_prepare_data(args)
    except PublicationRecoveryError as e:
        print(f"CRITICAL: Publication recovery error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"Error during data preparation: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
