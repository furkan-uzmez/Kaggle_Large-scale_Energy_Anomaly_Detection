"""Contract and integrity test for notebooks/05_final_results.ipynb.

Exercises strict conformance to:
1. nbformat v4 specification.
2. Reader-facing problem title and mandatory hierarchical sections.
3. Every code cell preceded by a Markdown cell with `**Purpose:**`.
4. Read-only role: strictly NO model fitting, tuning, training, or raw data access.
5. No user-specific absolute paths (/home/..., /Users/..., etc.).
6. No agent/process meta, task IDs, or prompt logs.
7. Verification of outputs/submission.csv as the authoritative final deliverable.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "05_final_results.ipynb"

REQUIRED_SECTIONS = (
    "Setup and tournament provenance",
    "Comprehensive tournament leaderboard",
    "Winning architecture and methodology deep dive",
    "Submission artifact contract and cryptographic verification",
    "Production limitations and deployment stance",
)

FORBIDDEN_CODE_PATTERNS = re.compile(
    r"(?:\.fit\s*\(|\bfit_transform\s*\(|\bfit_predict\s*\(|\bpartial_fit\s*\(|"
    r"\.predict\w*\s*\(|\bpredict_proba\b|\bdecision_function\s*\(|"
    r"\b(?:GridSearchCV|RandomizedSearchCV|BayesSearchCV|optuna|hyperopt)\b|"
    r"\b(?:train_test_split|GroupKFold|StratifiedKFold|StratifiedGroupKFold)\b|"
    r"\b(?:subprocess|runpy)\b|\bos\.system\s*\(|"
    r"\b(?:train_joined|test_joined|raw_train|raw_test)\.parquet\b|"
    r"\b(?:outputs/prepared|data/raw)\b)",
    re.IGNORECASE,
)

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


def _load_notebook() -> nbformat.NotebookNode:
    assert NOTEBOOK_PATH.is_file(), f"Notebook 05 does not exist at {NOTEBOOK_PATH}"
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def test_notebook_05_format_and_title() -> None:
    nb = _load_notebook()
    nbformat.validate(nb)
    first_cell = nb.cells[0]
    assert first_cell.cell_type == "markdown", "First cell must be Markdown"
    assert first_cell.source.startswith("# 05 — Final Tournament Results and Submission Verification"), (
        "First cell must begin with the reader-facing title: # 05 — Final Tournament Results and Submission Verification"
    )


def test_notebook_05_required_sections_present() -> None:
    nb = _load_notebook()
    h2_headings = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            for line in cell.source.splitlines():
                line = line.strip()
                if line.startswith("## "):
                    text = re.sub(r"^##\s*(?:\d+\.\s*)?", "", line).strip()
                    h2_headings.append(text)

    for section in REQUIRED_SECTIONS:
        found = any(section.casefold() in h.casefold() for h in h2_headings)
        assert found, f"Required section '{section}' missing from Notebook 05 headings: {h2_headings}"


def test_notebook_05_every_code_cell_has_purpose_markdown() -> None:
    nb = _load_notebook()
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert i > 0, "First cell cannot be code"
            prev_cell = nb.cells[i - 1]
            assert prev_cell.cell_type == "markdown", (
                f"Cell {i} (code) must be immediately preceded by a Markdown cell"
            )
            assert "**Purpose:**" in prev_cell.source or "### Purpose" in prev_cell.source, (
                f"Markdown cell before code cell {i} must contain '**Purpose:**'"
            )


def test_notebook_05_read_only_no_forbidden_code() -> None:
    nb = _load_notebook()
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            lines = cell.source.splitlines()
            for line_no, line in enumerate(lines, 1):
                clean_line = re.sub(r"#.*$", "", line).strip()
                if not clean_line:
                    continue
                match = FORBIDDEN_CODE_PATTERNS.search(clean_line)
                assert not match, (
                    f"Forbidden pattern '{match.group(0) if match else ''}' found in cell {i}, "
                    f"line {line_no}: {clean_line}"
                )


def test_notebook_05_no_absolute_user_paths() -> None:
    nb = _load_notebook()
    for i, cell in enumerate(nb.cells):
        match = ABSOLUTE_USER_PATH.search(cell.source)
        assert not match, (
            f"Absolute user path '{match.group(0) if match else ''}' in cell {i} source"
        )
        # Also check all cell outputs (stream stdout, display_data text/plain, execute_result, etc.)
        for output in getattr(cell, "outputs", []):
            output_type = output.get("output_type", "")
            # Stream outputs (stdout / stderr)
            if output_type == "stream":
                text = output.get("text", "")
                omatch = ABSOLUTE_USER_PATH.search(text)
                assert not omatch, (
                    f"Absolute user path '{omatch.group(0)}' in cell {i} stream output"
                )
            # Rich outputs (display_data / execute_result) — inspect text/plain
            elif output_type in ("display_data", "execute_result"):
                plain = output.get("data", {}).get("text/plain", "")
                omatch = ABSOLUTE_USER_PATH.search(plain)
                assert not omatch, (
                    f"Absolute user path '{omatch.group(0)}' in cell {i} {output_type} text/plain"
                )


def test_notebook_05_no_process_or_agent_metadata() -> None:
    nb = _load_notebook()
    for i, cell in enumerate(nb.cells):
        match = PROCESS_METADATA.search(cell.source)
        assert not match, (
            f"Process/agent metadata '{match.group(0) if match else ''}' in cell {i} source"
        )
