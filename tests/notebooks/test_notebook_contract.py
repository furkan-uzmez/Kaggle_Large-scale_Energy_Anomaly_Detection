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
NOTEBOOK_02_PATH = PROJECT_ROOT / "notebooks" / "02_baseline_and_candidate_design.ipynb"
REQUIRED_SECTIONS = (
    "Problem contract",
    "Reproducibility and raw data",
    "Join and data integrity",
    "Target and data quality",
    "Outlier analysis",
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

NOTEBOOK_02_TITLE = "# 02 — Baseline and Candidate Design"
NOTEBOOK_02_REQUIRED_SECTIONS = (
    "Purpose and execution status",
    "Inherited decisions and deferred experiments",
    "Persisted split verification",
    "Baseline portfolio",
    "Model-family audit",
    "Finite experiment matrix",
    "Resource budget and stop rules",
    "Smoke-test requirements",
)
# Notebook 02 reads artifact metrics but never fits, tunes, selects, or scores.
NOTEBOOK_02_FORBIDDEN_CODE = re.compile(
    r"(?:\.fit\s*\(|\bfit_transform\s*\(|\bfit_predict\s*\(|\bpartial_fit\s*\(|"
    r"\.predict\w*\s*\(|\bpredict_proba\b|\bdecision_function\s*\(|"
    r"\b(?:GridSearchCV|RandomizedSearchCV|HalvingGridSearchCV|HalvingRandomSearchCV|"
    r"BayesSearchCV|optuna|hyperopt|cross_val_score|cross_val_predict|cross_validate|"
    r"RFECV|RFE|SelectKBest|SelectFromModel|SequentialFeatureSelector|"
    r"roc_auc_score|average_precision_score|brier_score_loss|"
    r"precision_recall_curve|roc_curve|train_test_split|GroupKFold|StratifiedKFold|"
    r"StratifiedGroupKFold|TimeSeriesSplit)\b|"
    r"\b(?:subprocess|runpy|joblib|pickle|upload|submit)\b|threshold|"
    r"\blog_loss\s*\(|\bos\.system\s*\(|\.to_csv\s*\(|\.write_parquet\s*\(|\.write_csv\s*\()",
    re.IGNORECASE,
)
# Reject any claim of a measured score, ranking, or winner.
NOTEBOOK_02_SCORE_CLAIM = re.compile(
    r"(?:\b(?:roc[ _-]?auc|auc|average precision|log ?loss|brier|accuracy|f1)\b"
    r"[^.\n]{0,40}?(?:\bis\b|\bwas\b|\bof\b|[:=])\s*[01]?\.\d+|"
    r"\b(?:best|winning|winner|top[- ]performing|strongest|outperform\w*|"
    r"beats|best[- ]scoring|highest[- ]scoring)\b[^.\n]{0,40}?"
    r"\b(?:model|baseline|candidate|family|experiment)\b|"
    r"\bachieve[sd]?\b[^.\n]{0,25}\b(?:roc[ _-]?auc|auc|score)\b)",
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

    code_cells = [
        (index, cell)
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
    ]
    assert code_cells, "Notebook 01 must include executable code cells"
    source_is_clean = all(
        cell.execution_count is None and not cell.outputs for _index, cell in code_cells
    )
    if not source_is_clean:
        assert [cell.execution_count for _index, cell in code_cells] == list(
            range(1, len(code_cells) + 1)
        ), "An outputful Notebook 01 must be executed top-to-bottom from a fresh kernel"
        for index, cell in code_cells:
            assert cell.outputs, f"Code cell {index} must have embedded execution output"
            assert not any(output.output_type == "error" for output in cell.outputs), (
                f"Code cell {index} contains an execution error"
            )

    for index, cell in code_cells:
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
        "building_folds",
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


def test_notebook_01_outlier_audit_contract() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    full_source = "\n".join(cell.source for cell in notebook.cells)

    # 1. Required Outlier analysis section
    headings = _markdown_headings(notebook)
    level_two_headings = [title for level, title in headings if level == 2]
    assert "Outlier analysis" in level_two_headings
    assert (
        level_two_headings.index("Target and data quality")
        < level_two_headings.index("Outlier analysis")
        < level_two_headings.index("Train/test shift")
    )

    # 2. Helper call dev manifest
    assert "from project.data.outlier_audit import build_outlier_audit" in code_source
    assert "build_outlier_audit(" in code_source
    assert "train_joined" in code_source
    assert "split_path" in code_source or "split_manifest" in code_source
    assert "building_folds_sha256" in code_source

    # Feature exclusions
    for col in ("building_id", "row_id", "anomaly", "timestamp"):
        assert col in code_source
    assert "gte_" in code_source

    # 3. Charts included and scaled
    assert "plt." in code_source
    lowered_code = code_source.casefold()
    assert "hist" in lowered_code
    assert "boxplot" in lowered_code
    assert "log1p" in lowered_code or "signed" in lowered_code
    assert "meter_by_building" in code_source
    assert "meter_by_month" in code_source

    # 4. OIA data-sourced and guardrails
    assert "Observation" in full_source
    assert "Interpretation" in full_source
    assert "Action" in full_source
    lowered_full = full_source.casefold()
    assert "iqr" in lowered_full and "mad" in lowered_full
    assert "degenerate" in lowered_full or "unavailable" in lowered_full
    assert "exploratory" in lowered_full
    assert "scale" in lowered_full or "season" in lowered_full

    # 5. Handoff note in decisions
    assert "handoff" in lowered_full
    assert "notebook 02" in lowered_full
    assert "raw" in lowered_full and "transform" in lowered_full


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


def _read_notebook_02() -> nbformat.NotebookNode:
    assert NOTEBOOK_02_PATH.is_file(), "Notebook 02 does not exist yet"
    return nbformat.read(NOTEBOOK_02_PATH, as_version=4)


def test_notebook_02_contract() -> None:
    notebook = _read_notebook_02()
    nbformat.validate(notebook)

    markdown_cells = [cell for cell in notebook.cells if cell.cell_type == "markdown"]
    assert markdown_cells, "Notebook 02 must include reader-facing Markdown"
    assert markdown_cells[0].source.splitlines()[0] == NOTEBOOK_02_TITLE

    headings = _markdown_headings(notebook)
    level_two_headings = [title for level, title in headings if level == 2]
    assert level_two_headings == list(NOTEBOOK_02_REQUIRED_SECTIONS)
    assert any(level == 3 for level, _title in headings), (
        "Notebook 02 sections must include nested Markdown headings"
    )

    code_cells = [
        (index, cell)
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
    ]
    assert code_cells, "Notebook 02 must include executable code cells"
    assert [cell.execution_count for _index, cell in code_cells] == list(
        range(1, len(code_cells) + 1)
    ), "Notebook 02 must be executed top-to-bottom from a fresh kernel"

    for index, cell in code_cells:
        assert cell.outputs, f"Code cell {index} must have embedded execution output"
        assert not any(output.output_type == "error" for output in cell.outputs), (
            f"Code cell {index} contains an execution error"
        )
        assert not dict(cell.metadata), (
            f"Code cell {index} must carry empty per-cell metadata"
        )
        assert index > 0 and notebook.cells[index - 1].cell_type == "markdown", (
            f"Code cell {index} must immediately follow a Markdown purpose cell"
        )
        purpose = notebook.cells[index - 1].source.strip().casefold()
        assert purpose.startswith("**purpose:**"), (
            f"Markdown cell {index - 1} must state the code cell's purpose"
        )

    all_text, metadata_text = _notebook_text_and_metadata(notebook)
    assert not ABSOLUTE_USER_PATH.search(all_text), (
        "Notebook 02 content contains a user-specific absolute path"
    )
    assert not ABSOLUTE_USER_PATH.search(metadata_text), (
        "Notebook 02 metadata contains a user-specific absolute path"
    )
    assert not PROCESS_METADATA.search(all_text), (
        "Notebook 02 content contains workflow or agent metadata"
    )
    assert not PROCESS_METADATA.search(metadata_text), (
        "Notebook 02 metadata contains workflow or agent metadata"
    )


def test_notebook_02_cell_ids_are_stable_and_distinct_from_notebook_01() -> None:
    notebook_01 = nbformat.read(NOTEBOOK_PATH, as_version=4)
    notebook_02 = _read_notebook_02()

    ids_01 = {cell.get("id") for cell in notebook_01.cells}
    ids_02 = [cell.get("id") for cell in notebook_02.cells]

    assert all(isinstance(cell_id, str) and cell_id for cell_id in ids_02), (
        "Every Notebook 02 cell must carry a stable string id"
    )
    assert len(set(ids_02)) == len(ids_02), "Notebook 02 cell ids must be unique"
    assert not set(ids_02).intersection(ids_01), (
        "Notebook 02 cell ids must not collide with Notebook 01"
    )


def test_notebook_02_performs_no_fitting_tuning_or_selection() -> None:
    notebook = _read_notebook_02()

    code_source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    match = NOTEBOOK_02_FORBIDDEN_CODE.search(code_source)
    assert match is None, (
        "Notebook 02 is design-only and must not fit, tune, select features, "
        f"score, re-split, or persist artifacts (found: {match.group(0) if match else ''})"
    )

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                assert root not in {"sklearn", "optuna", "xgboost", "lightgbm", "catboost"}, (
                    f"Notebook 02 must not import the modelling library '{root}'"
                )


def test_notebook_02_makes_no_score_or_winner_claims() -> None:
    notebook = _read_notebook_02()

    # Validated artifact outputs may contain measured numbers; static prose may not invent them.
    all_text = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    match = NOTEBOOK_02_SCORE_CLAIM.search(all_text)
    assert match is None, (
        "Notebook 02 must not claim a measured score, ranking, or winner "
        f"(found: {match.group(0) if match else ''})"
    )

    lowered = all_text.casefold()
    for phrase in ("not run", "pending"):
        assert phrase in lowered, (
            f"Notebook 02 must state that results are '{phrase}' until execution"
        )


def test_notebook_02_discloses_task6_smoke_as_noncomparative() -> None:
    notebook = _read_notebook_02()
    all_text, _metadata_text = _notebook_text_and_metadata(notebook)
    lowered = all_text.casefold()

    assert "non-comparative smoke" in lowered
    assert "workflow-continuity" in lowered or "workflow continuity" in lowered
    assert "no comparative" in lowered and "oof" in lowered
    assert "no comparative campaign run has taken place" in lowered
    assert "until a comparative oof artifact exists" in lowered
    assert "because nothing has run" not in lowered
    assert "no run has taken place" not in lowered
    assert "no baseline, candidate, or model family has been run" not in lowered
    # Reader-facing prose must disclose the smoke without internal task vocabulary.
    assert not PROCESS_METADATA.search(all_text)


def test_notebook_02_references_required_design_artifacts() -> None:
    notebook = _read_notebook_02()

    all_text, _metadata_text = _notebook_text_and_metadata(notebook)
    lowered = all_text.casefold()

    required_references = (
        # Baseline portfolio and its three universal baselines.
        "configs/experiments/baselines.yaml",
        "dummy_prior",
        "logistic_l2",
        "hist_gradient_boosting",
        # Context-baseline audit.
        "business_incumbent",
        "temporal_naive",
        "grouped_hierarchical",
        "current_champion",
        # Deferred hand-off inherited from Notebook 01.
        "deferred experiments",
        # Model-family audit and its blocked-status explanation.
        "model_family_audit",
        "blocked_unapproved_compute",
        # Finite experiment matrix and its two status axes.
        "configs/experiments/feature_campaign.yaml",
        "experiment_matrix",
        "execution_status",
        # Resource budget and stop rules.
        "configs/resource_budget.yaml",
        "stop_rules",
        "host_preflight_status",
        "user_approval_status",
        # Persisted split, never re-split.
        "outputs/splits/building_folds",
        "vault",
        # Design-only framing.
        "design_only",
        "execution_authorized",
        # Finding structure inherited from Notebook 01.
        "observation",
        "interpretation",
        "action",
    )
    for phrase in required_references:
        assert phrase.casefold() in lowered, (
            f"Notebook 02 is missing required design reference: {phrase}"
        )


def test_notebook_02_explains_blocked_status_is_not_a_false_na() -> None:
    notebook = _read_notebook_02()

    all_text, _metadata_text = _notebook_text_and_metadata(notebook)
    lowered = all_text.casefold()

    assert "not a false n/a" in lowered or "not an n/a" in lowered, (
        "Notebook 02 must state that blocked_unapproved_compute is not a false N/A"
    )
    assert "architectur" in lowered, (
        "Notebook 02 must distinguish architectural inapplicability from resource blocking"
    )


def test_notebook_02_reads_configs_rather_than_hardcoding_matrix_rows() -> None:
    notebook = _read_notebook_02()

    code_source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert "yaml.safe_load" in code_source, (
        "Notebook 02 must derive design tables from the persisted YAML configs"
    )

    hardcoded_ids = [
        experiment_id
        for experiment_id in ("feat_core_vs_provided", "dim_pca_vs_passthrough")
        if experiment_id in code_source
    ]
    assert not hardcoded_ids, (
        "Notebook 02 must render experiment rows from the campaign config, "
        f"not hardcode them (found: {hardcoded_ids})"
    )


def test_notebook_02_counts_pending_approval_gates_as_open() -> None:
    notebook = _read_notebook_02()
    function_node = None
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        function_node = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "pending_gate_names"
            ),
            None,
        )
        if function_node is not None:
            break

    assert function_node is not None, (
        "Notebook 02 must define a testable pending_gate_names helper"
    )
    function_module = ast.Module(body=[function_node], type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(function_module, str(NOTEBOOK_02_PATH), "exec"), namespace)  # noqa: S102
    pending_gate_names = namespace["pending_gate_names"]
    assert callable(pending_gate_names)
    assert set(pending_gate_names({
        "host_preflight_status": "pending",
        "user_approval_status": "pending",
    })) == {"host_preflight_status", "user_approval_status"}
    assert pending_gate_names({
        "host_preflight_status": "approved",
        "user_approval_status": "approved",
    }) == []
