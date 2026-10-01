from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat
import pytest

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


def _extract_outlier_cells(notebook: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    cells: list[nbformat.NotebookNode] = []
    in_outlier_section = False

    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            for line in cell.source.splitlines():
                if line.strip().startswith("## Outlier analysis"):
                    in_outlier_section = True
                    break
                elif line.strip().startswith("## ") and in_outlier_section:
                    in_outlier_section = False
                    break
        if in_outlier_section:
            cells.append(cell)
    return cells


def test_outlier_section_heading_and_placement() -> None:
    notebook = _load_notebook()
    headings: list[tuple[int, str]] = []
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        for line in cell.source.splitlines():
            match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if match:
                headings.append((len(match.group(1)), match.group(2)))

    level_two = [title for level, title in headings if level == 2]
    assert "Outlier analysis" in level_two, "Notebook 01 must include an 'Outlier analysis' section"

    target_idx = level_two.index("Target and data quality")
    outlier_idx = level_two.index("Outlier analysis")
    shift_idx = level_two.index("Train/test shift")
    assert target_idx < outlier_idx < shift_idx, (
        "'Outlier analysis' must be placed after 'Target and data quality' and before 'Train/test shift'"
    )

    outlier_cells = _extract_outlier_cells(notebook)
    assert outlier_cells, "Outlier analysis section must contain content cells"

    # Check for nested level-3 headings
    outlier_text = "\n".join(c.source for c in outlier_cells if c.cell_type == "markdown")
    level_three = [
        match.group(1)
        for line in outlier_text.splitlines()
        if (match := re.match(r"^###\s+(.+?)\s*#*$", line))
    ]
    assert len(level_three) >= 2, "Outlier analysis section must contain nested level-3 subheadings"


def test_outlier_code_cells_have_purpose_markdown() -> None:
    notebook = _load_notebook()
    code_indices = [i for i, c in enumerate(notebook.cells) if c.cell_type == "code"]

    for idx in code_indices:
        assert idx > 0, f"Code cell at index {idx} cannot be the first cell"
        preceding = notebook.cells[idx - 1]
        assert preceding.cell_type == "markdown", (
            f"Code cell at index {idx} must be preceded by a Markdown purpose cell"
        )
        assert preceding.source.strip().casefold().startswith("**purpose:**"), (
            f"Markdown cell at index {idx - 1} must start with '**Purpose:**'"
        )


def test_outlier_backend_call_contract() -> None:
    notebook = _load_notebook()
    outlier_cells = _extract_outlier_cells(notebook)
    outlier_code = "\n".join(c.source for c in outlier_cells if c.cell_type == "code")

    assert "from project.data.outlier_audit import build_outlier_audit" in outlier_code, (
        "Outlier code must import 'build_outlier_audit' from 'project.data.outlier_audit'"
    )
    assert "build_outlier_audit(" in outlier_code, (
        "Outlier code must call 'build_outlier_audit'"
    )

    # Must call on train_joined and dev manifest
    assert "train_joined" in outlier_code, "Outlier audit must be called on 'train_joined'"
    assert "split_path" in outlier_code or "split_manifest" in outlier_code, (
        "Outlier audit must supply the split manifest from split_path"
    )

    # Must validate manifest hash before audit
    assert "building_folds_sha256" in outlier_code, (
        "Outlier code must validate split manifest hash against split_metadata.json artifacts.building_folds_sha256"
    )

    # Exclusions check
    for excluded in ("building_id", "row_id", "anomaly", "timestamp"):
        assert excluded in outlier_code, (
            f"Outlier feature selection must explicitly exclude '{excluded}'"
        )
    assert "gte_" in outlier_code, (
        "Outlier feature selection must explicitly exclude 'gte_*' features"
    )

    # Must not use entire-train feature_quality for new findings
    tree = ast.parse(outlier_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "feature_quality":
            pytest.fail("Outlier analysis must not rely on entire-train 'feature_quality' table")


def test_outlier_charts_included_and_annotated() -> None:
    notebook = _load_notebook()
    outlier_cells = _extract_outlier_cells(notebook)
    outlier_code = "\n".join(c.source for c in outlier_cells if c.cell_type == "code")
    outlier_text = "\n".join(c.source for c in outlier_cells)

    # Verify matplotlib/seaborn presence
    assert "plt." in outlier_code or "matplotlib" in outlier_code, (
        "Outlier analysis must generate visual plots via matplotlib"
    )

    # Verify plot types
    lowered = outlier_code.casefold()
    assert "hist" in lowered, "Outlier analysis must produce a histogram comparison"
    assert "boxplot" in lowered or "box" in lowered, "Outlier analysis must produce a boxplot comparison"

    # Verify scaling for zeros/negatives (e.g. signed log1p)
    assert "log1p" in lowered or "symlog" in lowered or "signed" in lowered, (
        "Meter reading distribution must use appropriate scaling (e.g. signed log1p) for zero/negative values"
    )

    # Verify annotation about bounded stratified sample
    all_outlier_text = outlier_text.casefold()
    assert (
        "sample" in all_outlier_text
        and ("descriptive" in all_outlier_text or "stratified" in all_outlier_text or "bounded" in all_outlier_text)
    ), "Outlier plots must annotate that the sample is a bounded descriptive sample, not exact full prevalence"

    # Verify building and month / time overview charts
    assert "meter_by_building" in outlier_code or "building" in lowered, (
        "Outlier analysis must visualize building-level flag distributions or overviews"
    )
    assert "meter_by_month" in outlier_code or "month" in lowered, (
        "Outlier analysis must visualize monthly or temporal flag patterns"
    )


def test_outlier_methodological_guardrails() -> None:
    notebook = _load_notebook()
    outlier_cells = _extract_outlier_cells(notebook)
    outlier_text = "\n".join(c.source for c in outlier_cells).casefold()

    # Zero IQR / MAD caveat
    assert "iqr" in outlier_text and "mad" in outlier_text, (
        "Outlier narrative must discuss IQR and MAD spread metrics"
    )
    assert "zero" in outlier_text or "unavailable" in outlier_text or "degenerate" in outlier_text, (
        "Narrative must state that zero IQR/MAD is explicitly unavailable / degenerate"
    )

    # Exploratory vs confirmed error
    assert "exploratory" in outlier_text or "not confirmed" in outlier_text or "candidate signal" in outlier_text, (
        "Narrative must state flags are exploratory indicators, not confirmed data errors"
    )

    # Scale / seasonality / regime caveat
    assert (
        "scale" in outlier_text
        or "season" in outlier_text
        or "regime" in outlier_text
        or "weather" in outlier_text
    ), "Narrative must note scale, seasonality, or operational regime caveats"


def test_outlier_oia_and_decision_ledger() -> None:
    notebook = _load_notebook()
    outlier_cells = _extract_outlier_cells(notebook)
    outlier_text = "\n".join(c.source for c in outlier_cells)

    # Observation, Interpretation, Action structure
    assert "observation" in outlier_text.casefold(), "Outlier analysis must include Observation"
    assert "interpretation" in outlier_text.casefold(), "Outlier analysis must include Interpretation"
    assert "action" in outlier_text.casefold(), "Outlier analysis must include Action"

    # Decision ledger under Outlier section or Decisions section
    assert "ledger" in outlier_text.casefold() or "decision" in outlier_text.casefold(), (
        "Outlier analysis must include a decision ledger or summary"
    )


def test_handoff_note_in_decisions_section() -> None:
    notebook = _load_notebook()
    full_text = "\n".join(c.source for c in notebook.cells).casefold()

    assert "handoff" in full_text or "notebook 02" in full_text, (
        "Notebook 01 must include a handoff note under decisions for Notebook 02"
    )
    assert "raw" in full_text and ("transform" in full_text or "indicator" in full_text), (
        "Handoff note must state hypotheses regarding retaining raw vs indicator/transform features"
    )


def test_outlier_code_hygiene_and_no_forbidden_tokens() -> None:
    notebook = _load_notebook()
    outlier_cells = _extract_outlier_cells(notebook)
    outlier_code = "\n".join(c.source for c in outlier_cells if c.cell_type == "code")

    # Critical: FORBIDDEN_CODE contains 'threshold'! Code must NOT contain 'threshold'.
    forbidden_match = FORBIDDEN_CODE.search(outlier_code)
    assert forbidden_match is None, (
        f"Outlier code contains forbidden pattern: {forbidden_match.group(0) if forbidden_match else ''}. "
        "Remember: the word 'threshold' is forbidden in code; use 'bound', 'cutoff', 'fence', or 'limit' instead."
    )

    all_outlier_text = "\n".join(c.source for c in outlier_cells)
    assert not ABSOLUTE_USER_PATH.search(all_outlier_text), (
        "Outlier section contains a user-specific absolute path"
    )
    assert not PROCESS_METADATA.search(all_outlier_text), (
        "Outlier section contains process or agent metadata"
    )


def test_outlier_plots_contain_embedded_png() -> None:
    notebook = _load_notebook()
    plot_cell_ids = {"lead-outlier-dist-plots", "lead-outlier-pattern-plots"}
    plot_cells = [c for c in notebook.cells if c.id in plot_cell_ids]
    assert len(plot_cells) == 2, "Both outlier plot cells must exist"

    for cell in plot_cells:
        assert cell.outputs, f"Plot cell {cell.id} must have execution outputs"
        png_outputs = [
            output
            for output in cell.outputs
            if output.output_type in ("display_data", "execute_result")
            and "image/png" in output.get("data", {})
        ]
        assert len(png_outputs) >= 1, (
            f"Plot cell {cell.id} must contain embedded 'image/png' data in outputs"
        )


def test_outlier_narrative_avoids_unsupported_causal_claims() -> None:
    notebook = _load_notebook()
    outlier_cells = _extract_outlier_cells(notebook)
    outlier_text = "\n".join(c.source for c in outlier_cells).casefold()

    # Must NOT claim proven/genuine causes
    assert "genuine" not in outlier_text, (
        "Outlier analysis must avoid claiming unverified 'genuine' causes"
    )
    assert "rather than pure sensor error" not in outlier_text, (
        "Outlier analysis must avoid asserting root-cause absence of sensor errors"
    )

    # Scientific blocker: must not claim zeros are proven shutdowns/inactivity
    assert "zero values represent temporary meter shutdowns" not in outlier_text, (
        "Outlier analysis must not claim zero readings are proven shutdowns or inactivity"
    )
    assert "shutdowns, or other causes remain unverified" in outlier_text, (
        "Outlier analysis must qualify zero readings: 'inactivity, shutdowns, or other causes remain unverified'"
    )


def test_outlier_plots_do_not_fabricate_synthetic_data() -> None:
    notebook = _load_notebook()
    outlier_cells = _extract_outlier_cells(notebook)
    dist_cell = next(c for c in outlier_cells if c.id == "lead-outlier-dist-plots")
    assert "np.array([0.0])" not in dist_cell.source, (
        "Plot code must not fabricate synthetic [0.0] data for absent classes"
    )


def test_outlier_synthetic_isolated_execution() -> None:
    notebook = _load_notebook()
    outlier_cells = _extract_outlier_cells(notebook)
    dist_cell = next(c for c in outlier_cells if c.id == "lead-outlier-dist-plots")
    findings_cell = next(c for c in outlier_cells if c.id == "lead-outlier-findings")

    import numpy as np
    import polars as pl

    # Test 1: Single class in plot_sample (anomaly=1 only)
    single_class_sample = pl.DataFrame({
        "building_id": [1, 1],
        "timestamp": [1000, 2000],
        "meter_reading": [10.0, 20.0],
        "anomaly": [1, 1],
    })
    ns = {
        "plot_sample": single_class_sample,
        "plt": __import__("matplotlib.pyplot", fromlist=[""]),
        "np": np,
        "get_ipython": lambda: None,
    }
    # Execute code in namespace
    exec(compile(ast.parse(dist_cell.source), "<dist-test>", "exec"), ns)  # noqa: S102
    ns["plt"].close("all")

    # Test 2: Degenerate meter_by_building with 0 evaluable counts and missing anomaly class in meter_by_class
    synth_scope = pl.DataFrame({
        "development_rows": [100],
        "development_buildings": [1],
        "sample_rows": [2],
        "vault_buildings_excluded": [0],
    })
    synth_class = pl.DataFrame({
        "anomaly": [0],
        "rows": [100],
        "finite_count": [100],
        "min": [0.0],
        "q01": [0.0],
        "median": [0.0],
        "q99": [0.0],
        "max": [0.0],
        "zero_count": [100],
        "negative_count": [0],
    })
    synth_bldg = pl.DataFrame({
        "building_id": [1],
        "rows": [100],
        "finite_count": [100],
        "anomaly_rows": [0],
        "q25": [0.0],
        "median": [0.0],
        "q75": [0.0],
        "iqr": [0.0],
        "mad": [0.0],
        "iqr_available": [False],
        "mad_available": [False],
        "iqr_flag_count": [None],
        "mad_flag_count": [None],
        "iqr_evaluable_count": [0],
        "mad_evaluable_count": [0],
    })
    findings_ns = {
        "scope_df": synth_scope,
        "meter_by_class": synth_class,
        "meter_by_building": synth_bldg,
        "pl": pl,
        "Markdown": lambda x: x,
        "display": lambda x: None,
    }
    exec(compile(ast.parse(findings_cell.source), "<findings-test>", "exec"), findings_ns)  # noqa: S102
    report = str(findings_ns["oia_report"])
    assert "0.00%" not in report, (
        "Rate must not be reported as 0.00% when evaluable count is 0"
    )

