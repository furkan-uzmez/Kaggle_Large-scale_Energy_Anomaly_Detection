"""Temporary projected full-campaign fixtures; never a production fallback.

Builds a confined ``outputs/prepared/train_joined.parquet`` plus a frozen
four-fold/three-repeat split manifest with deliberately non-parity development
labels, opaque string building identities, exact nanosecond timestamps, real
notebook cell ids and vault poison rows. Fixture data is test/diagnostic only:
it proves I/O behavior and must never substitute for authenticated real data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl

from project.experiments.candidate_contracts import CandidateSpec
from project.experiments.full_run_contracts import (
    FullCandidateSnapshot,
    FullResourcePolicy,
    FullRunSpec,
    SourceBinding,
)
from project.features.registry import build_lead_baseline_registry

DATA_RELATIVE = "outputs/prepared/train_joined.parquet"
SPLIT_RELATIVE = "outputs/splits/building_folds_repeated.parquet"
MATRIX_RELATIVE = "configs/experiments/feature_campaign.yaml"
PRODUCER_RELATIVE = "src/project/features/registry.py"
NOTEBOOK_RELATIVE = "notebooks/02_baseline_and_candidate_design.ipynb"

TARGET_COLUMN = "anomaly"
CATEGORICAL_COLUMN = "primary_use"
BUILDING_COLUMN = "building_id"
TIMESTAMP_COLUMN = "timestamp"

DEV_BUILDINGS = (
    "bldg_amber", "bldg_birch", "bldg_cedar", "bldg_dune",
    "bldg_elm", "bldg_fir", "bldg_garnet", "bldg_hazel",
)
# Integer building identities chosen so lexicographic order differs from numeric
# order: numeric sort is [1, 2, 9, 10, 11, 12, 19, 20]; text sort is
# [1, 10, 11, 12, 19, 2, 20, 9].
INT_BUILDINGS = (1, 2, 9, 10, 11, 12, 19, 20)
# Disjoint integer vault identity; a nonempty vault is required by FullRunSpec.
INT_VAULT_BUILDING = 200
VAULT_BUILDINGS = ("bldg_ivy", "bldg_jade")
SPLIT_SEEDS = (42, 43, 44)
PRIMARY_USES = ("Office", "Education", "Retail", "Healthcare")
BASE_NS = 1_700_000_000_000_000_000
VAULT_METER_READING = -9_999_999.0


def _base_ticks(timestamp_unit: str) -> int:
    """Return the fixture epoch in the requested physical time unit.

    The authoritative index stores physical integer ticks in the source unit
    (``us`` for the real development data), so a microsecond fixture must carry
    microsecond ticks rather than the nanosecond fixture default.
    """
    if timestamp_unit == "ns":
        return BASE_NS
    if timestamp_unit == "us":
        return BASE_NS // 1_000
    raise ValueError("fixture supports only ns/us timestamps")

# Three frozen, genuinely different building partitions (not folded permutations)
# so each repeat still covers all eight development buildings exactly once.
FOLD_ASSIGNMENTS = {
    0: ((0, 1), (2, 3), (4, 5), (6, 7)),
    1: ((0, 2), (1, 3), (4, 6), (5, 7)),
    2: ((0, 3), (1, 2), (4, 7), (5, 6)),
}

PREDICTOR_COLUMNS = build_lead_baseline_registry().names


@dataclass(frozen=True)
class FullFixture:
    root: Path
    data_path: Path
    split_path: Path
    spec: FullRunSpec
    full_frame: pl.DataFrame


def development_frame(timestamp_unit: str = "ns") -> pl.DataFrame:
    """Return the 52 development rows in canonical fixture order."""
    records: list[dict[str, object]] = []
    for ordinal, building in enumerate(DEV_BUILDINGS):
        for offset in range(ordinal + 3):
            records.append(_row(ordinal, building, offset, target=(ordinal + offset) % 2,
                                timestamp_unit=timestamp_unit))
    return pl.DataFrame(records, schema=_schema(timestamp_unit)).sort([BUILDING_COLUMN, TIMESTAMP_COLUMN])


def vault_frame(timestamp_unit: str = "ns") -> pl.DataFrame:
    """Return valid-looking vault poison rows that must never be retained."""
    records = [
        _row(100 + ordinal, building, offset, target=(ordinal + offset) % 2, poison=True,
             timestamp_unit=timestamp_unit)
        for ordinal, building in enumerate(VAULT_BUILDINGS)
        for offset in range(2)
    ]
    return pl.DataFrame(records, schema=_schema(timestamp_unit)).sort([BUILDING_COLUMN, TIMESTAMP_COLUMN])


def int_development_frame(timestamp_unit: str = "ns") -> pl.DataFrame:
    """Integer-building development rows whose numeric order differs from text order."""
    records: list[dict[str, object]] = []
    for ordinal, building in enumerate(INT_BUILDINGS):
        for offset in range(ordinal + 3):
            records.append(_int_row(ordinal, building, offset, target=(ordinal + offset) % 2,
                                    timestamp_unit=timestamp_unit))
    return pl.DataFrame(records, schema=_int_schema(timestamp_unit)).sort([BUILDING_COLUMN, TIMESTAMP_COLUMN])


def int_vault_frame(timestamp_unit: str = "ns") -> pl.DataFrame:
    """Return valid-looking integer vault rows that must never be retained."""
    records = [
        _int_row(100 + offset, INT_VAULT_BUILDING, offset, target=offset % 2,
                 timestamp_unit=timestamp_unit)
        for offset in range(4)
    ]
    return pl.DataFrame(records, schema=_int_schema(timestamp_unit)).sort([BUILDING_COLUMN, TIMESTAMP_COLUMN])


def int_split_manifest(*, include_vault: bool = False) -> pl.DataFrame:
    """Four-fold/three-repeat assignments for integer building identities."""
    records: list[dict[str, object]] = []
    for repeat in range(3):
        for position, building in enumerate(INT_BUILDINGS):
            records.append({"building_id": building, "repeat": repeat,
                            "seed": SPLIT_SEEDS[repeat], "split": "dev",
                            "fold": _int_fold_for(repeat, position)})
        if include_vault:
            records.append({"building_id": INT_VAULT_BUILDING, "repeat": repeat,
                            "seed": SPLIT_SEEDS[repeat], "split": "vault", "fold": None})
    return pl.DataFrame(records, schema={
        "building_id": pl.Int64, "repeat": pl.Int64, "seed": pl.Int64,
        "split": pl.String, "fold": pl.Int64,
    })


def write_int_fixture(root: Path, *, vault: bool = False, timestamp_unit: str = "ns") -> FullFixture:
    """Write a confined integer-building prepared/split fixture.

    ``vault`` appends a disjoint integer vault partition so the fixture can
    satisfy the ``FullRunSpec`` nonempty-vault contract.
    """
    root = Path(root)
    data_path = _write_parent(root) / DATA_RELATIVE
    split_path = _write_parent(root) / SPLIT_RELATIVE
    full_frame = int_development_frame(timestamp_unit)
    if vault:
        full_frame = pl.concat([full_frame, int_vault_frame(timestamp_unit)])
    full_frame.write_parquet(data_path)
    int_split_manifest(include_vault=vault).write_parquet(split_path)
    (root / MATRIX_RELATIVE).parent.mkdir(parents=True, exist_ok=True)
    (root / MATRIX_RELATIVE).write_text("experiment_matrix: []\n")
    (root / PRODUCER_RELATIVE).parent.mkdir(parents=True, exist_ok=True)
    (root / PRODUCER_RELATIVE).write_text("# fixture producer\n")
    (root / NOTEBOOK_RELATIVE).parent.mkdir(parents=True, exist_ok=True)
    (root / NOTEBOOK_RELATIVE).write_text(json.dumps({"cells": [{"id": "real-cell", "cell_type": "code"}]}))
    (root / "test_joined.parquet").write_bytes(b"not-preparation-data")
    return FullFixture(root, data_path, split_path, make_spec(root), full_frame)


def split_manifest() -> pl.DataFrame:
    """Return the frozen dev/vault split assignments for seeds 42/43/44."""
    records: list[dict[str, object]] = []
    for repeat in range(3):
        for building in DEV_BUILDINGS:
            records.append({"building_id": building, "repeat": repeat,
                            "seed": SPLIT_SEEDS[repeat], "split": "dev",
                            "fold": _fold_for(repeat, building)})
        for building in VAULT_BUILDINGS:
            records.append({"building_id": building, "repeat": repeat,
                            "seed": SPLIT_SEEDS[repeat], "split": "vault", "fold": None})
    return pl.DataFrame(records, schema={
        "building_id": pl.String, "repeat": pl.Int64, "seed": pl.Int64,
        "split": pl.String, "fold": pl.Int64,
    })


def write_fixture(root: Path, *, timestamp_unit: str = "ns") -> FullFixture:
    """Write the confined prepared/split/source files and return the layout."""
    root = Path(root)
    data_path = _write_parent(root) / DATA_RELATIVE
    split_path = _write_parent(root) / SPLIT_RELATIVE
    full_frame = pl.concat([development_frame(timestamp_unit), vault_frame(timestamp_unit)])
    full_frame.write_parquet(data_path)
    split_manifest().write_parquet(split_path)
    (root / "configs/experiments").mkdir(parents=True, exist_ok=True)
    (root / MATRIX_RELATIVE).write_text("experiment_matrix: []\n")
    (root / PRODUCER_RELATIVE).parent.mkdir(parents=True, exist_ok=True)
    (root / PRODUCER_RELATIVE).write_text("# fixture producer\n")
    (root / NOTEBOOK_RELATIVE).parent.mkdir(parents=True, exist_ok=True)
    (root / NOTEBOOK_RELATIVE).write_text(json.dumps({"cells": [{"id": "real-cell", "cell_type": "code"}]}))
    # A decoy competition test file must never be opened by full development I/O.
    (root / "test_joined.parquet").write_bytes(b"not-preparation-data")
    return FullFixture(root, data_path, split_path, make_spec(root), full_frame)


def overwrite_data(fixture: FullFixture, frame: pl.DataFrame) -> Path:
    """Rewrite the prepared Parquet; callers rebuild the spec afterwards."""
    frame.write_parquet(fixture.data_path)
    return fixture.data_path


def write_admission_fixture(root: Path, *, int_buildings: bool = False,
                            timestamp_unit: str = "ns") -> FullFixture:
    """Copy only code/config/source evidence; all dataset bytes are generated.

    ``int_buildings`` selects an integer-building development frame whose numeric
    order differs from its text order; both variants carry a disjoint, nonempty
    vault partition required by ``FullRunSpec``.
    """
    from project.experiments.full_campaign_admission import CORE
    from project.experiments.full_campaign_preflight import current_environment
    from project.experiments.full_run_contracts import _file_hash

    repository = Path(__file__).resolve().parents[1]
    fixture = (write_int_fixture(root, vault=True, timestamp_unit=timestamp_unit)
               if int_buildings else write_fixture(root, timestamp_unit=timestamp_unit))
    frame = fixture.full_frame.with_columns(pl.lit(100).alias("year_built"))
    frame.write_parquet(fixture.data_path)
    evidence_paths = (
        MATRIX_RELATIVE, "configs/experiments/baselines.yaml", NOTEBOOK_RELATIVE,
        "docs/evidence/year_built_encoding_v1.json", "scripts/train.py", "scripts/campaign.py",
        "scripts/audit_full_development.py",
        "docs/superpowers/specs/2026-10-04-full-campaign-decisions.md",
    )
    for relative in evidence_paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repository / relative).read_bytes())
    for original in (repository / "src/project").rglob("*.py"):
        target = root / original.relative_to(repository)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
    sources = tuple(replace(source, sha256=_file_hash(root / source.path))
                    for source in fixture.spec.sources)
    sources += tuple(SourceBinding(role, relative, _file_hash(root / relative)) for role, relative in (
        ("baseline", "configs/experiments/baselines.yaml"),
        ("encoding", "docs/evidence/year_built_encoding_v1.json"),
    ))
    candidate = replace(fixture.spec.candidates[0], source_cell_id="cand-matrix-render",
                        spec=replace(fixture.spec.candidates[0].spec, feature_pool=CORE))
    development_rows = int_development_frame(timestamp_unit).height if int_buildings else 52
    vault_rows = 4
    spec = replace(fixture.spec, sources=sources, candidates=(candidate,),
                   environment=current_environment(), development_rows=development_rows,
                   vault_rows=vault_rows)
    return replace(fixture, spec=spec, full_frame=frame)


def overwrite_split(fixture: FullFixture, frame: pl.DataFrame) -> Path:
    frame.write_parquet(fixture.split_path)
    return fixture.split_path


def make_spec(root: Path) -> FullRunSpec:
    """Build a full-run snapshot whose data/split bindings match current bytes."""
    root = Path(root)
    pseudo = CandidateSpec("feat_missingness_indicators", PREDICTOR_COLUMNS,
                           "hist_gradient_boosting", ())
    snapshot = FullCandidateSnapshot(pseudo, ("control", "challenger"),
                                     NOTEBOOK_RELATIVE, "real-cell")
    sources = tuple(
        SourceBinding(role, path, hashlib.sha256((root / path).read_bytes()).hexdigest())
        for role, path in (
            ("matrix", MATRIX_RELATIVE), ("data", DATA_RELATIVE),
            ("split", SPLIT_RELATIVE), ("producer", PRODUCER_RELATIVE),
            ("notebook", NOTEBOOK_RELATIVE),
        )
    )
    return FullRunSpec("fixture-run", (snapshot,), sources,
                       (("python", "fixture"),), FullResourcePolicy(informed_opt_out=True))


def _row(ordinal: int, building: str, offset: int, *, target: int, poison: bool = False,
         timestamp_unit: str = "ns") -> dict[str, object]:
    timestamp = _base_ticks(timestamp_unit) + ordinal * 10 + offset
    return {
        BUILDING_COLUMN: building,
        TIMESTAMP_COLUMN: timestamp,
        TARGET_COLUMN: target,
        "meter_reading": VAULT_METER_READING if poison else 10.0 + ordinal + offset,
        CATEGORICAL_COLUMN: PRIMARY_USES[ordinal % len(PRIMARY_USES)],
        "square_feet": 1_000.0 + 7.0 * ordinal,
        "year_built": 1_960 + ordinal,
        "floor_count": 1 + ordinal % 5,
        "hour": offset % 24,
        "weekday": ordinal % 7,
        "month": 1 + offset % 12,
        "year": 2_024,
    }


def _schema(timestamp_unit: str = "ns") -> dict[str, pl.DataType]:
    return {
        BUILDING_COLUMN: pl.String,
        TIMESTAMP_COLUMN: pl.Datetime(timestamp_unit),
        TARGET_COLUMN: pl.Int64,
        "meter_reading": pl.Float64,
        CATEGORICAL_COLUMN: pl.String,
        "square_feet": pl.Float64,
        "year_built": pl.Int64,
        "floor_count": pl.Int64,
        "hour": pl.Int64,
        "weekday": pl.Int64,
        "month": pl.Int64,
        "year": pl.Int64,
    }


def _int_schema(timestamp_unit: str = "ns") -> dict[str, pl.DataType]:
    schema = _schema(timestamp_unit)
    schema[BUILDING_COLUMN] = pl.Int64
    return schema


def _int_row(ordinal: int, building: int, offset: int, *, target: int,
             timestamp_unit: str = "ns") -> dict[str, object]:
    row = _row(ordinal, str(building), offset, target=target, timestamp_unit=timestamp_unit)
    row[BUILDING_COLUMN] = building
    return row


def _int_fold_for(repeat: int, position: int) -> int:
    for fold, members in enumerate(FOLD_ASSIGNMENTS[repeat]):
        if position in members:
            return fold
    raise ValueError("fixture building has no frozen fold")


def _fold_for(repeat: int, building: str) -> int:
    position = DEV_BUILDINGS.index(building)
    for fold, members in enumerate(FOLD_ASSIGNMENTS[repeat]):
        if position in members:
            return fold
    raise ValueError("fixture building has no frozen fold")


def _write_parent(root: Path) -> Path:
    (root / "outputs/prepared").mkdir(parents=True, exist_ok=True)
    (root / "outputs/splits").mkdir(parents=True, exist_ok=True)
    return root


def labels_are_not_row_parity() -> bool:
    """Diagnostic guard: fixture labels deliberately differ from index parity."""
    labels = development_frame().get_column(TARGET_COLUMN).to_numpy()
    return not np.array_equal(labels, np.arange(labels.size) % 2)
