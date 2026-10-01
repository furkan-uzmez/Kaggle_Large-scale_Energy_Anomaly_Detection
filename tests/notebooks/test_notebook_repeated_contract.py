"""Read-only repeated split contract exercised through both notebook validators."""
from __future__ import annotations

import ast
from pathlib import Path

import nbformat
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ["01_data_understanding.ipynb", "02_baseline_and_candidate_design.ipynb"]


def validator(name):
    notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    functions = [node for cell in notebook.cells if cell.cell_type == "code"
                 for node in ast.parse(cell.source).body
                 if isinstance(node, ast.FunctionDef) and node.name == "validate_repeated_manifest"]
    assert functions, "Notebook must validate the repeated manifest before diagnostics"
    namespace = {"pl": pl}
    exec(compile(ast.Module(body=functions, type_ignores=[]), name, "exec"), namespace)  # noqa: S102
    return namespace["validate_repeated_manifest"]


def manifests():
    old = pl.DataFrame([
        {"building_id": i, "split": "dev" if i < 160 else "vault",
         "fold": i % 4 if i < 160 else None,
         "row_count": 10, "anomaly_count": 1, "normal_count": 9}
        for i in range(200)
    ])
    repeated = pl.concat([
        old.with_columns(pl.lit(r).alias("repeat"), pl.lit(seed).alias("seed"),
                         pl.when(pl.col("split") == "dev")
                         .then((r * (pl.col("building_id") // 4) + pl.col("building_id")) % 4)
                         .otherwise(None).alias("fold"))
        for r, seed in enumerate([42, 43, 44])
    ])
    metadata = {"repeats": 3, "n_dev_folds": 4, "split_seeds": [42, 43, 44],
                "vault_building_ids": list(range(160, 200))}
    return old, repeated, metadata


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_repeated_group_key_fixed_vault_and_population(name):
    old, repeated, metadata = manifests()
    summary = validator(name)(repeated, metadata, old)
    assert summary.height == 15
    assert summary.filter(pl.col("split") == "dev")["buildings"].to_list() == [40] * 12
    assert summary.filter(pl.col("split") == "vault")["buildings"].to_list() == [40] * 3


@pytest.mark.parametrize("name", NOTEBOOKS)
@pytest.mark.parametrize("corruption", ["duplicate", "seed", "vault", "copied", "relabeled", "support"])
def test_invalid_repeated_lineage_fails_closed(name, corruption):
    old, repeated, metadata = manifests()
    if corruption == "duplicate":
        repeated = pl.concat([repeated, repeated.slice(0, 1)])
    elif corruption == "seed":
        repeated = repeated.with_columns(pl.lit(42).alias("seed"))
    elif corruption == "vault":
        repeated = repeated.with_columns(
            pl.when((pl.col("repeat") == 1) & pl.col("building_id").is_in([0, 160]))
            .then(pl.when(pl.col("building_id") == 0).then(pl.lit("vault")).otherwise(pl.lit("dev")))
            .otherwise(pl.col("split")).alias("split"))
    elif corruption in {"copied", "relabeled"}:
        repeated = repeated.with_columns(
            pl.when((pl.col("repeat") == 1) & (pl.col("split") == "dev"))
            .then((pl.col("building_id") + (1 if corruption == "relabeled" else 0)) % 4)
            .otherwise(pl.col("fold")).alias("fold"))
    else:
        repeated = repeated.with_columns(
            pl.when((pl.col("repeat") == 2) & (pl.col("building_id") == 0))
            .then(11).otherwise(pl.col("row_count")).alias("row_count"))
    with pytest.raises(ValueError):
        validator(name)(repeated, metadata, old)
