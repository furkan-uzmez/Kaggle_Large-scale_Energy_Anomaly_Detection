from __future__ import annotations

import re
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_data_understanding.ipynb"

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


def _load_notebook() -> nbformat.NotebookNode:
    assert NOTEBOOK_PATH.is_file(), "Notebook 01 does not exist"
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
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


def test_visual_first_major_evidence_sections_contain_plots() -> None:
    notebook = _load_notebook()
    required_visual_sections = (
        "Reproducibility and raw data",
        "Join and data integrity",
        "Target and data quality",
        "Outlier analysis",
        "Train/test shift",
        "Feature availability and leakage",
        "Persisted split decision",
    )

    for section_name in required_visual_sections:
        cells = _extract_section_cells(notebook, section_name)
        assert cells, f"Section '{section_name}' must exist"
        code_cells = [c for c in cells if c.cell_type == "code"]
        assert code_cells, f"Section '{section_name}' must contain code cells"

        # Check that the section contains plotting code and embedded image/png in outputs
        section_code = "\n".join(c.source for c in code_cells)
        assert re.search(
            r"\b(?:show_figure|plot_\w+|render_\w+)\s*\(|plt\.|matplotlib", section_code
        ), f"Section '{section_name}' must render a chart from executed evidence"

        has_image = any(
            output.output_type in ("display_data", "execute_result")
            and "image/png" in output.get("data", {})
            for c in code_cells
            for output in c.outputs
        )
        assert has_image, (
            f"Section '{section_name}' must contain at least one embedded 'image/png' chart"
        )


def test_visual_first_cell_ordering_and_purpose_markdown() -> None:
    notebook = _load_notebook()
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


def test_domain_shift_chart_semantics() -> None:
    notebook = _load_notebook()
    cells = _extract_section_cells(notebook, "Train/test shift")
    shift_code = "\n".join(c.source for c in cells if c.cell_type == "code").casefold()
    shift_text = "\n".join(c.source for c in cells).casefold()

    # Domain AUC chart must have honest 0.50 reference line
    assert "0.5" in shift_code or "0.50" in shift_code, (
        "Domain shift chart code must reference the 0.50 chance baseline"
    )
    # Must explicitly state inverse ranking / adversarial diagnostic
    assert "inverse ranking" in shift_text, (
        "Shift narrative or chart must state the inverse ranking finding"
    )
    assert "adversarial" in shift_text or "shift diagnostic" in shift_text, (
        "Shift section must clarify this is a domain shift diagnostic, not an anomaly model score"
    )


def test_target_and_quality_chart_semantics() -> None:
    notebook = _load_notebook()
    cells = _extract_section_cells(notebook, "Target and data quality")
    quality_code = "\n".join(c.source for c in cells if c.cell_type == "code").casefold()
    quality_text = "\n".join(c.source for c in cells).casefold()

    # Must include class imbalance chart
    assert "target_summary" in quality_code or "imbalance" in quality_code, (
        "Quality plotting must visualize class distribution from target_summary"
    )
    # Missingness bounds must be qualified as max range / group bounds, not group means
    assert "bound" in quality_text or "max" in quality_text or "range" in quality_text, (
        "Missingness chart annotations must distinguish max bounds from averages"
    )


def test_persisted_split_chart_semantics() -> None:
    notebook = _load_notebook()
    cells = _extract_section_cells(notebook, "Persisted split decision")
    split_code = "\n".join(c.source for c in cells if c.cell_type == "code").casefold()

    # Must visualize 4 dev folds + vault
    assert "fold" in split_code, "Split visualization must include fold partitions"
    assert "vault" in split_code, "Split visualization must include the vault partition"


def test_feature_governance_chart_semantics() -> None:
    notebook = _load_notebook()
    cells = _extract_section_cells(notebook, "Feature availability and leakage")
    feat_code = "\n".join(c.source for c in cells if c.cell_type == "code").casefold()

    # Must reflect pending gte_* policy
    assert "gte_" in feat_code or "pending" in feat_code, (
        "Feature governance chart must show pending gte_* status"
    )


def test_visual_notebook_hygiene_and_no_forbidden_code() -> None:
    notebook = _load_notebook()
    code_source = "\n".join(
        c.source for c in notebook.cells if c.cell_type == "code"
    )
    match = FORBIDDEN_CODE.search(code_source)
    assert match is None, (
        f"Forbidden pattern found in code: {match.group(0) if match else ''}"
    )

    all_text = "\n".join(c.source for c in notebook.cells)
    assert not ABSOLUTE_USER_PATH.search(all_text), "Found absolute user path in notebook text"
    assert not PROCESS_METADATA.search(all_text), "Found process metadata in notebook text"
