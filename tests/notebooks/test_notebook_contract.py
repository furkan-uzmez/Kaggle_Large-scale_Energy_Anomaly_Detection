from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import nbformat
import pytest

from project.data.validate import compute_file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_data_understanding.ipynb"
REQUIRED_SECTIONS = (
    "Problem contract",
    "Reproducibility and raw data",
    "Join and data integrity",
    "Target and data quality",
    "Train/test shift",
    "Feature availability and leakage",
    "Persisted split decision",
    "Decisions for Notebook 02",
)
RAW_TABLE_NAMES = {
    "raw_train",
    "raw_test",
    "raw_train_features",
    "raw_test_features",
    "train_joined",
    "test_joined",
}
ABSOLUTE_USER_PATH = re.compile(
    r"(?:/home/[^/\s]+|/Users/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)",
    re.IGNORECASE,
)
PROCESS_METADATA = re.compile(
    r"(?:\.agent(?:/|\\|\b)|\bsubagents?\b|\btask\s*#?\s*\d+\b|"
    r"\bagent[ _-]?(?:id|workflow|ownership)\b|\btools?[- ]called\b|"
    r"\bprompts?[- ]used\b|\bimplementation plan\b)",
    re.IGNORECASE,
)
FORBIDDEN_CODE = re.compile(
    r"(?:\.fit\s*\(|\b(?:GridSearchCV|RandomizedSearchCV|Optuna|"
    r"RandomizedSearch|HalvingGridSearchCV|adversarial_validate)\b|"
    r"\b(?:subprocess|runpy|threshold|upload|submit|to_csv)\b|"
    r"\bos\.system\s*\(|\.to_csv\s*\()",
    re.IGNORECASE,
)


def _markdown_headings(notebook: nbformat.NotebookNode) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        for line in cell.source.splitlines():
            match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if match:
                headings.append((len(match.group(1)), match.group(2)))
    return headings


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Call, ast.Subscript)):
        node = node.value if isinstance(node, (ast.Attribute, ast.Subscript)) else node.func
    return node.id if isinstance(node, ast.Name) else None


def _load_notebook_function(function_name: str):
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        function_node = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == function_name
            ),
            None,
        )
        if function_node is None:
            continue

        function_module = ast.Module(body=[function_node], type_ignores=[])
        namespace = {
            "Path": Path,
            "compute_file_sha256": compute_file_sha256,
        }
        # Execute only this repository-owned function definition, not notebook cells.
        exec(compile(function_module, str(NOTEBOOK_PATH), "exec"), namespace)  # noqa: S102
        return namespace[function_name]

    raise AssertionError(
        f"Notebook 01 is missing the testable '{function_name}' contract"
    )


def _notebook_text_and_metadata(
    notebook: nbformat.NotebookNode,
) -> tuple[str, str]:
    content_parts: list[str] = []
    cell_metadata: list[dict[str, object]] = []

    for cell in notebook.cells:
        content_parts.append(cell.source)
        cell_metadata.append(dict(cell.metadata))
        for output in cell.get("outputs", []):
            content_parts.append(json.dumps(output.get("data", {}), default=str))
            content_parts.append(str(output.get("text", "")))
            if output.get("ename"):
                content_parts.append(str(output["ename"]))
            if output.get("evalue"):
                content_parts.append(str(output["evalue"]))
            content_parts.extend(str(line) for line in output.get("traceback", []))

    notebook_metadata = {
        key: value
        for key, value in dict(notebook.metadata).items()
        if key not in {"kernelspec", "language_info"}
    }
    return "\n".join(content_parts), json.dumps(
        {"notebook": notebook_metadata, "cells": cell_metadata}, default=str
    )


def test_notebook_01_contract() -> None:
    assert NOTEBOOK_PATH.is_file(), "Notebook 01 does not exist yet"

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(notebook)

    markdown_cells = [cell for cell in notebook.cells if cell.cell_type == "markdown"]
    assert markdown_cells, "Notebook 01 must include reader-facing Markdown"
    assert markdown_cells[0].source.splitlines()[0] == "# 01 — Data Understanding"

    headings = _markdown_headings(notebook)
    level_two_headings = [title for level, title in headings if level == 2]
    assert level_two_headings == list(REQUIRED_SECTIONS)
    assert any(level == 3 for level, _title in headings), (
        "Notebook sections must include nested Markdown headings"
    )

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs, (
                f"Code cell {index} must remain unexecuted and output-free in source"
            )
            assert index > 0 and notebook.cells[index - 1].cell_type == "markdown", (
                f"Code cell {index} must immediately follow a Markdown purpose cell"
            )
            purpose = notebook.cells[index - 1].source.strip().casefold()
            assert purpose.startswith("**purpose:**"), (
                f"Markdown cell {index - 1} must state the code cell's purpose"
            )

    all_text, metadata_text = _notebook_text_and_metadata(notebook)
    assert not ABSOLUTE_USER_PATH.search(all_text), "Notebook content contains a user-specific absolute path"
    assert not ABSOLUTE_USER_PATH.search(metadata_text), "Notebook metadata contains a user-specific absolute path"
    assert not PROCESS_METADATA.search(all_text), "Notebook content contains workflow or agent metadata"
    assert not PROCESS_METADATA.search(metadata_text), "Notebook metadata contains workflow or agent metadata"

    code_source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert not FORBIDDEN_CODE.search(code_source), (
        "Notebook 01 must not fit models, tune parameters, build submissions, or run scripts"
    )

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function_name = _root_name(node.func)
                if function_name in {"display", "print"}:
                    assert all(
                        not (isinstance(argument, ast.Name) and argument.id in RAW_TABLE_NAMES)
                        for argument in node.args
                    ), "Notebook 01 must not display a full raw or joined table"
                if isinstance(node.func, ast.Attribute) and node.func.attr == "show":
                    assert _root_name(node.func.value) not in RAW_TABLE_NAMES, (
                        "Notebook 01 must not display a full raw or joined table"
                    )
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Name):
                assert node.value.id not in RAW_TABLE_NAMES, (
                    "Notebook 01 must not implicitly display a full raw or joined table"
                )

    full_source = "\n".join(cell.source for cell in notebook.cells)
    required_contract_phrases = (
        "(building_id, timestamp)",
        "ROC-AUC",
        "building_folds.parquet",
        "pending/blocked",
        "<=0.55",
        "0.55–0.60",
        ">0.60",
        "Observation",
        "Interpretation",
        "Action",
        "true constant",
        "quasi-constant",
        "training-fold-fitted mask",
        "cross-validation ablation",
        "building row count is diagnostic metadata only",
        "categorical fields by mode",
        "not classifier predictors",
        "manual",
        "Cleaning",
        "Missingness",
        "Preprocessing",
        "Outlier policy",
        "Imbalance",
        "Deferred experiments for Notebook 02",
        "PCA",
    )
    for phrase in required_contract_phrases:
        assert phrase.casefold() in full_source.casefold(), (
            f"Notebook 01 is missing required contract language: {phrase}"
        )


def test_domain_auc_below_chance_is_flagged_as_inverse_ranking() -> None:
    interpret_domain_auc = _load_notebook_function("interpret_domain_auc")

    interpretation = interpret_domain_auc(0.435)

    assert "inverse ranking" in interpretation.casefold()
    assert "do not invert" in interpretation.casefold()
    assert "weak-separation" not in interpretation.casefold()


@pytest.mark.parametrize(
    ("auc", "expected_band"),
    [
        (0.50, "weak-separation"),
        (0.55, "weak-separation"),
        (0.550001, "moderate-separation"),
        (0.60, "moderate-separation"),
        (0.600001, "material domain separation"),
    ],
)
def test_domain_auc_heuristic_bands_apply_only_at_or_above_chance(
    auc: float,
    expected_band: str,
) -> None:
    interpret_domain_auc = _load_notebook_function("interpret_domain_auc")

    assert expected_band in interpret_domain_auc(auc).casefold()


def test_shift_report_rejects_stale_artifact_hashes(tmp_path: Path) -> None:
    validate_domain_artifact_hashes = _load_notebook_function(
        "validate_domain_artifact_hashes"
    )
    artifact_path = tmp_path / "prepared.parquet"
    artifact_path.write_bytes(b"current prepared data")
    report = {
        "artifacts": {
            "input_train_parquet": {
                "sha256": compute_file_sha256(artifact_path),
            }
        }
    }
    artifact_paths = {"input_train_parquet": artifact_path}

    validate_domain_artifact_hashes(report, artifact_paths)

    artifact_path.write_bytes(b"changed prepared data")
    with pytest.raises(ValueError, match="stale"):
        validate_domain_artifact_hashes(report, artifact_paths)
