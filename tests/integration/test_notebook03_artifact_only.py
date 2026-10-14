from __future__ import annotations

import ast
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nbformat
import pyarrow as pa
import pyarrow.parquet as pq

from project.models.candidate_artifacts import CANDIDATE_OOF_SCHEMA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "03_model_comparison_and_error_analysis.ipynb"

# Expected Level 2 Markdown headings in Notebook 03
REQUIRED_SECTIONS = (
    "Protocol and artifact lineage",
    "Campaign eligibility and missing work",
    "Model comparison and statistical significance",
    "Guardrails and probability calibration",
    "Paired error slices",
    "Family and error diversity",
    "Limitations and risks",
    "Ensemble plan and recommendations",
)

# Forbidden code patterns (model fitting, parameter tuning, prediction, calibration, table access)
FORBIDDEN_CODE_PATTERNS = re.compile(
    r"(?:\.fit\s*\(|\bfit_transform\s*\(|\bfit_predict\s*\(|\bpartial_fit\s*\(|"
    r"\.predict\w*\s*\(|\bpredict_proba\b|\bdecision_function\s*\(|"
    r"\b(?:GridSearchCV|RandomizedSearchCV|HalvingGridSearchCV|HalvingRandomSearchCV|"
    r"BayesSearchCV|optuna|hyperopt|cross_val_score|cross_val_predict|cross_validate|"
    r"RFECV|RFE|SelectKBest|SelectFromModel|SequentialFeatureSelector|"
    r"CalibratedClassifierCV|IsotonicRegression|"
    r"train_test_split|GroupKFold|StratifiedKFold|StratifiedGroupKFold|TimeSeriesSplit)\b|"
    r"\b(?:subprocess|runpy)\b|\bos\.system\s*\(|"
    r"\b(?:train_joined|test_joined|raw_train|raw_test)\.parquet\b|"
    r"\b(?:outputs/prepared|data/raw)\b)",
    re.IGNORECASE,
)

# User-specific absolute path patterns
ABSOLUTE_USER_PATH = re.compile(
    r"(?:/home/[^/\s]+|/Users/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)",
    re.IGNORECASE,
)

# Process, workflow, and agent metadata patterns
PROCESS_METADATA = re.compile(
    r"(?:\.agent(?:/|\\|\b)|\bsubagents?\b|\btask\s*#?\s*\d+\b|"
    r"\bagent[ _-]?(?:id|workflow|ownership)\b|\btools?[- ]called\b|"
    r"\bprompts?[- ]used\b|\bimplementation plan\b)",
    re.IGNORECASE,
)

REQUIRED_ENSEMBLE_KEYS = {
    "provisional_level1_candidates",
    "provisional_rejections_with_reasons",
    "level1_eligibility_policy",
    "simple_blend_candidates",
    "level2_meta_learners",
    "level3_attempt_policy",
    "nested_cv_contract",
    "promotion_thresholds",
}


def _load_notebook() -> nbformat.NotebookNode:
    """Load Notebook 03; fails with FileNotFoundError if notebook has not been created."""
    if not NOTEBOOK_PATH.is_file():
        raise FileNotFoundError(
            f"Notebook 03 does not exist yet at {NOTEBOOK_PATH.relative_to(PROJECT_ROOT)}"
        )
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def _markdown_headings(notebook: nbformat.NotebookNode) -> list[tuple[int, str]]:
    """Extract markdown headings (level, title) from notebook cells."""
    headings: list[tuple[int, str]] = []
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        for line in cell.source.splitlines():
            match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line.strip())
            if match:
                headings.append((len(match.group(1)), match.group(2)))
    return headings


# =====================================================================
# Contract Tests: Notebook 03 Existence and Format
# =====================================================================


def test_notebook03_exists() -> None:
    """Verifies that 03_model_comparison_and_error_analysis.ipynb exists on disk."""
    _load_notebook()


def test_notebook03_valid_nbformat() -> None:
    """Verifies that notebook conforms to nbformat version 4 schema."""
    notebook = _load_notebook()
    nbformat.validate(notebook)
    assert len(notebook.cells) > 0, "Notebook 03 cannot be empty"


# =====================================================================
# Contract Tests: Artifact-Only Analysis Role (No Training / Tuning)
# =====================================================================


def test_notebook03_artifact_only_ast_and_source() -> None:
    """Ensures notebook strictly analyzes artifacts and never fits or tunes models.

    Validates both raw code regex and parsed AST call nodes.
    """
    notebook = _load_notebook()
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert code_cells, "Notebook 03 must contain executable code cells"

    for idx, cell in enumerate(code_cells):
        source = cell.source
        assert not FORBIDDEN_CODE_PATTERNS.search(source), (
            f"Code cell {idx} violates artifact-only contract by invoking training/tuning/system calls"
        )

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Forbid estimator fitting, predicting, and scoring calls
                if isinstance(func, ast.Attribute) and func.attr in {
                    "fit",
                    "fit_transform",
                    "fit_predict",
                    "partial_fit",
                    "predict",
                    "predict_proba",
                    "predict_log_proba",
                    "decision_function",
                    "score",
                }:
                    raise AssertionError(
                        f"Code cell {idx} invokes prohibited estimator method '{func.attr}()'"
                    )
                # Forbid function calls like os.system()
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "system"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "os"
                ):
                    raise AssertionError(f"Code cell {idx} invokes prohibited 'os.system()'")


# =====================================================================
# Contract Tests: Cell Structure and Formatting Standards
# =====================================================================


def test_notebook03_purpose_markdown_before_every_code_cell() -> None:
    """Enforces that every code cell is immediately preceded by a Markdown purpose cell."""
    notebook = _load_notebook()
    code_indices = [i for i, c in enumerate(notebook.cells) if c.cell_type == "code"]

    for idx in code_indices:
        assert idx > 0, f"Code cell at index {idx} cannot be the first cell in the notebook"
        preceding = notebook.cells[idx - 1]
        assert preceding.cell_type == "markdown", (
            f"Cell at index {idx - 1} before code cell {idx} must be a Markdown purpose cell"
        )
        purpose_text = preceding.source.strip().casefold()
        assert purpose_text.startswith("**purpose:**"), (
            f"Markdown cell at index {idx - 1} must begin with '**Purpose:**'"
        )


def test_notebook03_hygiene_no_process_metadata_or_user_absolute_paths() -> None:
    """Ensures repo-relative paths only and no agent process metadata exist."""
    notebook = _load_notebook()
    content_parts = []
    metadata_parts = []

    for cell in notebook.cells:
        content_parts.append(cell.source)
        metadata_parts.append(json.dumps(cell.metadata))
        for output in cell.get("outputs", []):
            content_parts.append(str(output.get("text", "")))

    full_text = "\n".join(content_parts)
    full_metadata = "\n".join(metadata_parts)

    assert not ABSOLUTE_USER_PATH.search(full_text), (
        "Notebook 03 contains user-specific absolute filesystem paths"
    )
    assert not ABSOLUTE_USER_PATH.search(full_metadata), (
        "Notebook 03 metadata contains user-specific absolute filesystem paths"
    )
    assert not PROCESS_METADATA.search(full_text), (
        "Notebook 03 content contains agent, subagent, or process metadata"
    )
    assert not PROCESS_METADATA.search(full_metadata), (
        "Notebook 03 metadata contains agent, subagent, or process metadata"
    )
    for idx, cell in enumerate(c for c in notebook.cells if c.cell_type == "code"):
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value.strip()
                if val.startswith(("http://", "https://", "ftp://")):
                    continue
                # Reject any string pointing to an absolute POSIX or Windows root
                if (val.startswith("/") and len(val) > 1 and not val.startswith("//")) or re.match(r"^[A-Za-z]:[/\\]", val):
                    raise AssertionError(
                        f"Code cell {idx} contains prohibited absolute filesystem path '{val}'"
                    )
# =====================================================================
# Contract Tests: Heading Hierarchy and O/I/A/V Structure
# =====================================================================


def test_notebook03_headings_and_sections() -> None:
    """Verifies that all required analytical sections are present in Markdown headings."""
    notebook = _load_notebook()
    headings = _markdown_headings(notebook)

    # Notebook must have Level 1 Title
    assert any(level == 1 for level, _title in headings), (
        "Notebook 03 must have a Level 1 title heading"
    )

    level_two_headings = [title for level, title in headings if level == 2]
    for required in REQUIRED_SECTIONS:
        assert any(required.lower() in h.lower() for h in level_two_headings), (
            f"Notebook 03 is missing required section: '{required}'"
        )


def test_notebook03_oiav_narrative_structure() -> None:
    """Verifies presence of Observation, Intent/Interpretation, Action, Verification narrative."""
    notebook = _load_notebook()
    markdown_text = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
    lowered = markdown_text.lower()

    assert "observation" in lowered, "Notebook 03 narrative must include Observation statements"
    assert ("intent" in lowered) or ("interpretation" in lowered), (
        "Notebook 03 narrative must include Intent or Interpretation statements"
    )
    assert "action" in lowered, "Notebook 03 narrative must include Action statements"
    assert "verification" in lowered, "Notebook 03 narrative must include Verification statements"


# =====================================================================
# Integration Test: Clean-Kernel Execution and ensemble_plan.json
# =====================================================================
# Fixture Generator for Isolated Clean-Kernel Execution
# =====================================================================


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_sealed_fixture_campaign(campaign_dir: Path) -> Path:
    """Build a tiny, hermetic, canonical v1 finalized campaign fixture inside campaign_dir."""
    campaign_dir.mkdir(parents=True, exist_ok=True)
    completion_dir = campaign_dir / "completion"
    completion_dir.mkdir(parents=True, exist_ok=True)

    run_id = campaign_dir.name
    buildings = [f"bldg_{i:02d}" for i in range(1, 9)]
    n_rows_per_partition = 4

    exp_id = "feat_core_vs_weather"
    receipt_artifact_sha: dict[str, str] = {}

    exp_records: dict[str, list[Any]] = {
        "candidate_id": [],
        "recipe_id": [],
        "model_family": [],
        "model_config_id": [],
        "row_index": [],
        "repeat": [],
        "fold": [],
        "building_id": [],
        "timestamp": [],
        "prediction": [],
        "target": [],
        "split_sha256": [],
        "data_sha256": [],
        "config_sha256": [],
        "run_id": [],
        "split_seed": [],
        "experiment_id": [],
        "arm_id": [],
        "score_kind": [],
        "positive_label": [],
        "execution_mode": [],
        "non_comparative": [],
        "n03_eligible": [],
        "model_fit_config_sha256": [],
    }

    declaration_candidates = {}

    for arm in ("control", "challenger"):
        cand_id = f"{exp_id}_{arm}"
        fit_hash = hashlib.sha256(f"{exp_id}_{arm}".encode()).hexdigest()
        declaration_candidates[cand_id] = {
            "arm_id": arm,
            "experiment_id": exp_id,
            "model_config_id": f"{cand_id}_config",
            "model_family": "hist_gradient_boosting",
            "model_fit_config_sha256": fit_hash,
            "positive_label": 1,
            "recipe_id": exp_id,
            "score_kind": "positive_probability",
        }

        for repeat in range(3):
            seed = 42 + repeat
            for fold in range(4):
                fold_bldgs = buildings[fold * 2 : (fold + 1) * 2]
                for r_idx in range(n_rows_per_partition):
                    global_row_index = fold * n_rows_per_partition + r_idx
                    bldg = fold_bldgs[r_idx % len(fold_bldgs)]
                    ts_ns = int(datetime(2022, 1, 3, r_idx, 0, 0, tzinfo=UTC).timestamp() * 1_000_000_000)
                    target = 1 if (r_idx % 2 == 0) else 0
                    pred = 0.35 + 0.30 * target if arm == "control" else 0.20 + 0.60 * target

                    exp_records["candidate_id"].append(cand_id)
                    exp_records["recipe_id"].append(exp_id)
                    exp_records["model_family"].append("hist_gradient_boosting")
                    exp_records["model_config_id"].append(f"{cand_id}_config")
                    exp_records["row_index"].append(global_row_index)
                    exp_records["repeat"].append(repeat)
                    exp_records["fold"].append(fold)
                    exp_records["building_id"].append(bldg)
                    exp_records["timestamp"].append(ts_ns)
                    exp_records["prediction"].append(pred)
                    exp_records["target"].append(target)
                    exp_records["split_sha256"].append("b" * 64)
                    exp_records["data_sha256"].append("a" * 64)
                    exp_records["config_sha256"].append("c" * 64)
                    exp_records["run_id"].append(run_id)
                    exp_records["split_seed"].append(seed)
                    exp_records["experiment_id"].append(exp_id)
                    exp_records["arm_id"].append(arm)
                    exp_records["score_kind"].append("positive_probability")
                    exp_records["positive_label"].append(1)
                    exp_records["execution_mode"].append("full")
                    exp_records["non_comparative"].append(False)
                    exp_records["n03_eligible"].append(True)
                    exp_records["model_fit_config_sha256"].append(fit_hash)

    arrow_dict = dict(exp_records)
    arrow_dict["timestamp"] = pa.array(exp_records["timestamp"], type=pa.timestamp("ns"))
    arrow_table = pa.Table.from_pydict(arrow_dict, schema=CANDIDATE_OOF_SCHEMA)

    oof_filename = f"{exp_id}_oof.parquet"
    oof_path = campaign_dir / oof_filename
    pq.write_table(arrow_table, oof_path)
    receipt_artifact_sha[oof_filename] = _compute_sha256(oof_path)

    decl_filename = f"{exp_id}_declaration.json"
    decl_path = campaign_dir / decl_filename
    _write_json(decl_path, {
        "candidates": declaration_candidates,
        "config_sha256": "c" * 64,
        "experiment_id": exp_id,
        "recipe_id": exp_id,
    })
    receipt_artifact_sha[decl_filename] = _compute_sha256(decl_path)

    lineage_filename = f"{exp_id}_source_lineage.json"
    lineage_path = campaign_dir / lineage_filename
    _write_json(lineage_path, {
        "data_sha256": "a" * 64,
        "split_sha256": "b" * 64,
        "experiment_id": exp_id,
    })
    receipt_artifact_sha[lineage_filename] = _compute_sha256(lineage_path)

    trans_filename = f"{exp_id}_transport_validation.json"
    trans_path = campaign_dir / trans_filename
    _write_json(trans_path, {
        "experiment_id": exp_id,
        "record_count": len(exp_records["candidate_id"]),
        "validation_status": "passed",
    })
    receipt_artifact_sha[trans_filename] = _compute_sha256(trans_path)

    # Sealed feature metadata
    metadata_records = {
        "building_id": buildings,
        "primary_use": ["Office"] * len(buildings),
        "hour": [12] * len(buildings),
        "weather_missing": [False] * len(buildings),
    }
    meta_path = campaign_dir / "feature_metadata.parquet"
    pq.write_table(pa.Table.from_pydict(metadata_records), meta_path)
    receipt_artifact_sha["feature_metadata.parquet"] = _compute_sha256(meta_path)

    status_path = campaign_dir / "run_status.json"
    _write_json(status_path, {"status": "completed", "run_id": run_id, "exit_code": 0})
    receipt_artifact_sha["run_status.json"] = _compute_sha256(status_path)

    spec_path = campaign_dir / "spec.json"
    _write_json(spec_path, {"run_id": run_id, "execution_mode": "full"})
    receipt_artifact_sha["spec.json"] = _compute_sha256(spec_path)

    manifest_path = completion_dir / "manifest.json"
    _write_json(manifest_path, {
        "candidates": [{
            "arms": ["control", "challenger"],
            "experiment_id": exp_id,
            "n03_eligible": True,
            "oof_path": str(oof_path),
            "oof_sha256": receipt_artifact_sha[oof_filename],
            "status": "completed",
        }],
        "execution_mode": "full",
        "feature_metadata_sha256": receipt_artifact_sha["feature_metadata.parquet"],
        "run_id": run_id,
        "schema_version": 1,
    })

    receipt_path = completion_dir / "receipt.json"
    _write_json(receipt_path, {
        "artifact_sha256": receipt_artifact_sha,
        "campaign_id": run_id,
        "status": "completed",
    })

    return campaign_dir


# =====================================================================
# Integration Test: Clean-Kernel Execution and ensemble_plan.json
# =====================================================================


def test_notebook03_clean_kernel_execution(tmp_path: Path) -> None:
    """Executes Notebook 03 in an isolated clean kernel sandbox.

    Ensures top-to-bottom execution completes without errors in an isolated tmp_path
    environment, respects runtime boundary guards (no prepared/raw data access),
    and emits a newly created outputs/ensemble_plan.json containing all 8 required keys.
    """
    from nbclient import NotebookClient

    notebook = _load_notebook()

    # 1. Create isolated execution sandbox
    sandbox_root = tmp_path / "sandbox"
    sandbox_root.mkdir(parents=True)
    outputs_dir = sandbox_root / "outputs"
    outputs_dir.mkdir(parents=True)

    target_plan_path = outputs_dir / "ensemble_plan.json"
    # Pre-execution invariant: ensemble plan must NOT exist before notebook execution
    assert not target_plan_path.exists(), "Target ensemble_plan.json must not exist before execution"

    # 2. Build hermetic sealed campaign fixture inside sandbox (never copy ambient repo outputs)
    campaign_sandbox = outputs_dir / "campaigns" / "campaign_full_v1"
    _create_sealed_fixture_campaign(campaign_sandbox)

    # 3. Runtime boundary guards:
    # Ensure raw data and prepared datasets are strictly absent from sandbox
    assert not (sandbox_root / "outputs" / "prepared").exists()
    assert not (sandbox_root / "data" / "raw").exists()

    # 4. Inject in-memory cross-kernel runtime boundary guard cell as the first cell
    guard_code = (
        "import os, builtins, pathlib\n"
        f"os.environ['N03_CAMPAIGN_DIR'] = {str(campaign_sandbox)!r}\n"
        "_orig_open = open\n"
        "def _guarded_open(file, *args, **kwargs):\n"
        "    file_str = str(file).replace(chr(92), '/')\n"
        "    for forbidden in ('prepared', 'data/raw', 'train_joined', 'test_joined', 'vault'):\n"
        "        if forbidden in file_str:\n"
        "            raise PermissionError(f'Runtime guard: Prohibited access to {file_str}')\n"
        "    return _orig_open(file, *args, **kwargs)\n"
        "builtins.open = _guarded_open\n"
        "ip = get_ipython()\n"
        "if ip:\n"
        "    ip.user_ns['open'] = _guarded_open\n"
        "    ip.user_global_ns['open'] = _guarded_open\n"
        "_orig_path_open = pathlib.Path.open\n"
        "def _guarded_path_open(self, *args, **kwargs):\n"
        "    file_str = str(self).replace(chr(92), '/')\n"
        "    for forbidden in ('prepared', 'data/raw', 'train_joined', 'test_joined', 'vault'):\n"
        "        if forbidden in file_str:\n"
        "            raise PermissionError(f'Runtime guard: Prohibited access to {file_str}')\n"
        "    return _orig_path_open(self, *args, **kwargs)\n"
        "pathlib.Path.open = _guarded_path_open\n"
    )
    compile(guard_code, "<guard_cell>", "exec")
    guard_cell = nbformat.v4.new_code_cell(guard_code)
    notebook.cells.insert(0, guard_cell)

    # 5. Isolated clean-kernel execution
    client = NotebookClient(
        nb=notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(sandbox_root)}},
    )

    executed_nb = client.execute()

    # 6. Post-execution runtime assertions:
    for idx, cell in enumerate(executed_nb.cells):
        if cell.cell_type == "code":
            for out in cell.get("outputs", []):
                assert out.get("output_type") != "error", (
                    f"Cell {idx} failed during clean-kernel execution: {out.get('evalue')}"
                )

    # Verify ensemble_plan.json was freshly created inside sandbox outputs
    assert target_plan_path.is_file(), (
        f"Execution failed to create required {target_plan_path.relative_to(sandbox_root)}"
    )

    plan_payload = json.loads(target_plan_path.read_text())
    assert REQUIRED_ENSEMBLE_KEYS.issubset(set(plan_payload.keys())), (
        f"ensemble_plan.json missing required keys: {REQUIRED_ENSEMBLE_KEYS - set(plan_payload.keys())}"
    )
