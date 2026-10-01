from __future__ import annotations

import re
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_02_PATH = PROJECT_ROOT / "notebooks" / "02_baseline_and_candidate_design.ipynb"

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
NOTEBOOK_02_SCORE_CLAIM = re.compile(
    r"(?:\b(?:roc[ _-]?auc|auc|average precision|log ?loss|brier|accuracy|f1)\b"
    r"[^.\n]{0,40}?(?:\bis\b|\bwas\b|\bof\b|[:=])\s*[01]?\.\d+|"
    r"\b(?:best|winning|winner|top[- ]performing|strongest|outperform\w*|"
    r"beats|best[- ]scoring|highest[- ]scoring)\b[^.\n]{0,40}?"
    r"\b(?:model|baseline|candidate|family|experiment)\b|"
    r"\bachieve[sd]?\b[^.\n]{0,25}\b(?:roc[ _-]?auc|auc|score)\b)",
    re.IGNORECASE,
)


def _load_notebook_02() -> nbformat.NotebookNode:
    assert NOTEBOOK_02_PATH.is_file(), "Notebook 02 does not exist"
    notebook = nbformat.read(NOTEBOOK_02_PATH, as_version=4)
    nbformat.validate(notebook)
    return notebook


def _extract_section_cells(
    notebook: nbformat.NotebookNode, section_title: str
) -> list[nbformat.NotebookNode]:
    cells: list[nbformat.NotebookNode] = []
    in_section = False

    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            lines = cell.source.splitlines()
            first_line = lines[0].strip() if lines else ""
            if first_line.startswith(f"## {section_title}"):
                in_section = True
            elif first_line.startswith("## ") and in_section:
                in_section = False
        if in_section:
            cells.append(cell)
    return cells


def test_notebook_02_visual_sections_contain_plots() -> None:
    notebook = _load_notebook_02()
    required_visual_sections = (
        "Persisted split verification",
        "Baseline portfolio",
        "Model-family audit",
        "Finite experiment matrix",
        "Resource budget and stop rules",
    )

    for section_name in required_visual_sections:
        cells = _extract_section_cells(notebook, section_name)
        assert cells, f"Section '{section_name}' must exist"
        code_cells = [c for c in cells if c.cell_type == "code"]
        assert code_cells, f"Section '{section_name}' must contain code cells"

        section_code = "\n".join(c.source for c in code_cells)
        assert re.search(
            r"\b(?:show_figure|plot_\w+|render_\w+)\s*\(|plt\.|matplotlib", section_code
        ), f"Section '{section_name}' must render a chart from the persisted design"

        has_image = any(
            output.output_type in ("display_data", "execute_result")
            and "image/png" in output.get("data", {})
            for c in code_cells
            for output in c.outputs
        )
        assert has_image, (
            f"Section '{section_name}' must contain at least one embedded 'image/png' chart"
        )


def test_notebook_02_visual_cell_ordering_and_purpose_markdown() -> None:
    notebook = _load_notebook_02()
    code_indices = [i for i, c in enumerate(notebook.cells) if c.cell_type == "code"]

    for idx in code_indices:
        assert idx > 0, f"Code cell at index {idx} must follow a purpose Markdown cell"
        preceding = notebook.cells[idx - 1]
        assert preceding.cell_type == "markdown", (
            f"Cell at index {idx - 1} before code cell must be a Markdown cell"
        )
        purpose = preceding.source.strip().casefold()
        assert purpose.startswith("**purpose:**"), (
            f"Markdown cell at index {idx - 1} must state purpose with '**Purpose:**'"
        )


def test_notebook_02_split_chart_semantics() -> None:
    notebook = _load_notebook_02()
    cells = _extract_section_cells(notebook, "Persisted split verification")
    split_code = "\n".join(c.source for c in cells if c.cell_type == "code").casefold()

    assert "fold" in split_code, "Split chart must include development folds"
    assert "vault" in split_code, "Split chart must include the vault holdout"


def test_notebook_02_baseline_chart_semantics() -> None:
    notebook = _load_notebook_02()
    cells = _extract_section_cells(notebook, "Baseline portfolio")
    baseline_code = "\n".join(c.source for c in cells if c.cell_type == "code").casefold()

    assert "universal_baselines" in baseline_code, (
        "Baseline chart must be built from the persisted universal baseline portfolio"
    )
    assert "not run" in baseline_code, (
        "Baseline review must report not run when no eligible artifact exists"
    )
    assert "load_baseline_review(root)" in baseline_code
    assert "review['status'] != 'completed'" in baseline_code
    assert any(
        phrase in baseline_code
        for phrase in ("mean repeat-level development oof roc-auc", "baseline pooled development oof roc-auc")
    )
    assert any(
        phrase in baseline_code
        for phrase in ("mean repeat-level development oof average precision", "baseline pooled development oof average precision")
    )

    rendered = "\n".join(
        str(output.get("data", {}).get("text/html", ""))
        for cell in cells
        if cell.cell_type == "code"
        for output in cell.outputs
    ).casefold()
    for baseline in ("dummy_prior", "logistic_l2", "hist_gradient_boosting"):
        assert baseline in rendered, (
            f"Executed baseline evidence must include {baseline}"
        )


def test_notebook_02_family_audit_chart_semantics() -> None:
    notebook = _load_notebook_02()
    cells = _extract_section_cells(notebook, "Model-family audit")
    family_code = "\n".join(c.source for c in cells if c.cell_type == "code").casefold()
    family_text = "\n".join(c.source for c in cells).casefold()

    assert "blocked_unapproved_compute" in family_code, (
        "Family chart must visualize blocked_unapproved_compute status"
    )
    assert "not a false n/a" in family_text or "not an n/a" in family_text, (
        "Family narrative must state that blocked compute is not a false N/A"
    )


def test_notebook_02_experiment_matrix_chart_semantics() -> None:
    notebook = _load_notebook_02()
    cells = _extract_section_cells(notebook, "Finite experiment matrix")
    matrix_code = "\n".join(c.source for c in cells if c.cell_type == "code").casefold()
    matrix_text = "\n".join(c.source for c in cells).casefold()

    assert "category" in matrix_code, "Matrix chart must group by experiment category"
    assert "execution_status" in matrix_code or "status" in matrix_code, (
        "Matrix chart must display execution status"
    )
    assert "target" in matrix_text, (
        "Matrix narrative must clarify that candidate 100 is a target, not a quota"
    )


def test_notebook_02_resource_budget_chart_semantics() -> None:
    notebook = _load_notebook_02()
    cells = _extract_section_cells(notebook, "Resource budget and stop rules")
    budget_code = "\n".join(c.source for c in cells if c.cell_type == "code").casefold()

    assert "pending" in budget_code or "null" in budget_code, (
        "Budget chart must show unmeasured / null caps and pending gate status"
    )


def test_notebook_02_no_performance_or_score_claims() -> None:
    notebook = _load_notebook_02()
    all_text = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    match = NOTEBOOK_02_SCORE_CLAIM.search(all_text)
    assert match is None, (
        f"Found forbidden performance or score claim: {match.group(0) if match else ''}"
    )


def test_notebook_02_code_hygiene_and_forbidden_code() -> None:
    notebook = _load_notebook_02()
    code_source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    match = NOTEBOOK_02_FORBIDDEN_CODE.search(code_source)
    assert match is None, (
        f"Forbidden pattern found in code: {match.group(0) if match else ''}"
    )

    all_text = "\n".join(c.source for c in notebook.cells)
    assert not ABSOLUTE_USER_PATH.search(all_text), "Absolute user path found in notebook 02"
    assert not PROCESS_METADATA.search(all_text), "Process metadata found in notebook 02"


def test_notebook_02_metadata_is_empty() -> None:
    notebook = _load_notebook_02()
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    for idx, cell in enumerate(code_cells):
        assert not dict(cell.metadata), f"Code cell {idx} must have empty metadata"
