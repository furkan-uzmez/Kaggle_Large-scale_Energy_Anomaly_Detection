"""Tests for the append-only experiment inventory (Task 6 RED phase).

All imports of new production modules are deferred inside test bodies via
importlib so that pytest collection succeeds before the modules are created.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import pathlib
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_append_inventory_records():
    """Return append_inventory_records, asserting the module and function exist."""
    pkg_spec = importlib.util.find_spec("project.experiments")
    assert pkg_spec is not None, "project.experiments package must exist"

    mod_spec = importlib.util.find_spec("project.experiments.inventory")
    assert mod_spec is not None, (
        "project.experiments.inventory must be created for Task 6"
    )

    mod = importlib.import_module("project.experiments.inventory")
    fn = getattr(mod, "append_inventory_records", None)
    assert fn is not None, (
        "append_inventory_records must be defined in project.experiments.inventory"
    )
    return fn


def _minimal_record(**overrides) -> dict:
    """Return one valid inventory record, optionally overriding individual fields."""
    base = {
        "experiment_id": "exp-baseline-001",
        "hypothesis_id": "h-baseline-prior",
        "varied_factor_id": "vf-model-family",
        "baseline_id": "baseline-dummy",
        "candidate_id": "dummy_prior",
        "recipe_id": "recipe_dummy_prior",
        "model_config_id": "cfg_dummy_prior",
        "data_sha256": "aabbcc" * 10 + "aabb",
        "split_sha256": "11223344" * 8,
        "config_sha256": "deadbeef" * 8,
        "primary_metric": "roc_auc",
        "oof_path": "outputs/runs/run-smoke-001/oof_predictions.parquet",
        "wall_time_seconds": 1.23,
        "outcome": "smoke_complete",
        "rejection_reason": None,
        "run_id": "run-smoke-001",
        "timestamp": "2026-09-29T00:00:00Z",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_append_inventory_records_function_exists():
    """append_inventory_records is importable and callable."""
    fn = _get_append_inventory_records()
    assert callable(fn)


def test_append_inventory_records_writes_valid_jsonl():
    """A single record is written as valid JSONL (one JSON object per line)."""
    fn = _get_append_inventory_records()
    record = _minimal_record()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "experiment_inventory.jsonl"

        fn(path, [record])

        assert path.exists(), "inventory file must be created"
        lines = path.read_text().splitlines()
        assert len(lines) == 1, "one record must produce one line"
        parsed = json.loads(lines[0])
        assert isinstance(parsed, dict), "each line must be a JSON object"
        assert parsed["candidate_id"] == "dummy_prior"
        assert parsed["run_id"] == "run-smoke-001"
        assert parsed["primary_metric"] == "roc_auc"


def test_append_inventory_records_second_append_preserves_first_line():
    """Second call appends a new line; the original first line is byte-for-byte unchanged."""
    fn = _get_append_inventory_records()
    record_a = _minimal_record(candidate_id="dummy_prior", run_id="run-01")
    record_b = _minimal_record(candidate_id="logistic_l2", run_id="run-01")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "experiment_inventory.jsonl"

        fn(path, [record_a])
        first_line_bytes = path.read_bytes().splitlines()[0]

        fn(path, [record_b])
        lines = path.read_bytes().splitlines()

        assert len(lines) == 2, "two appends must produce two lines"
        assert lines[0] == first_line_bytes, (
            "first serialized line must be preserved byte-for-byte after second append"
        )

        parsed_b = json.loads(lines[1])
        assert parsed_b["candidate_id"] == "logistic_l2"


def test_append_inventory_records_multiple_records_in_one_call():
    """Passing two records in one call writes two lines."""
    fn = _get_append_inventory_records()
    records = [
        _minimal_record(candidate_id="dummy_prior"),
        _minimal_record(candidate_id="logistic_l2"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "experiment_inventory.jsonl"

        fn(path, records)

        lines = path.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["candidate_id"] == "dummy_prior"
        assert json.loads(lines[1])["candidate_id"] == "logistic_l2"


def test_append_inventory_records_required_fields_are_validated():
    """Omitting a required field must raise an error without modifying the file."""
    fn = _get_append_inventory_records()

    required_fields = [
        "experiment_id",
        "hypothesis_id",
        "varied_factor_id",
        "baseline_id",
        "candidate_id",
        "recipe_id",
        "model_config_id",
        "data_sha256",
        "split_sha256",
        "config_sha256",
        "primary_metric",
        "oof_path",
        "wall_time_seconds",
        "outcome",
        "rejection_reason",
        "run_id",
        "timestamp",
    ]

    for field in required_fields:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "experiment_inventory.jsonl"
            # Write one valid record first so we can check file is not modified
            fn(path, [_minimal_record()])
            original_content = path.read_bytes()

            incomplete = _minimal_record()
            del incomplete[field]

            with pytest.raises(ValueError):
                fn(path, [incomplete])

            assert path.read_bytes() == original_content, (
                f"file must not be modified when field '{field}' is missing"
            )


def test_append_inventory_records_does_not_truncate_or_rewrite_prior_lines():
    """Multiple sequential appends accumulate all records; no prior line ever disappears."""
    fn = _get_append_inventory_records()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "experiment_inventory.jsonl"
        candidate_ids = ["dummy_prior", "logistic_l2", "hist_gradient_boosting"]

        for cid in candidate_ids:
            fn(path, [_minimal_record(candidate_id=cid)])

        lines = path.read_text().splitlines()
        assert len(lines) == len(candidate_ids), (
            "All appended records must be present"
        )
        for i, cid in enumerate(candidate_ids):
            assert json.loads(lines[i])["candidate_id"] == cid, (
                f"line {i} must contain candidate_id={cid!r}"
            )


def test_append_inventory_records_creates_parent_directories_if_absent():
    """append_inventory_records must create parent directories as needed."""
    fn = _get_append_inventory_records()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "nested" / "dir" / "experiment_inventory.jsonl"
        assert not path.parent.exists()

        fn(path, [_minimal_record()])

        assert path.exists(), "file must be created under auto-created directories"
        lines = path.read_text().splitlines()
        assert len(lines) == 1


def test_append_inventory_records_all_task6_fields_round_trip():
    """All Task 6 design fields survive a JSON round-trip faithfully."""
    fn = _get_append_inventory_records()
    record = _minimal_record(
        varied_factor_id="vf-model-family",
        baseline_id="baseline-dummy",
        wall_time_seconds=3.14,
        outcome="smoke_complete",
        rejection_reason=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "experiment_inventory.jsonl"
        fn(path, [record])

        parsed = json.loads(path.read_text().splitlines()[0])
        for key, value in record.items():
            assert key in parsed, f"field '{key}' must survive round-trip"
            assert parsed[key] == value, (
                f"field '{key}' mismatch: expected {value!r}, got {parsed[key]!r}"
            )


def test_append_inventory_records_rejects_non_mapping_record(tmp_path: pathlib.Path):
    """A value that is not a record mapping fails before creating the file."""
    fn = _get_append_inventory_records()
    path = tmp_path / "inventory.jsonl"

    with pytest.raises(TypeError, match="mapping"):
        fn(path, [object()])

    assert not path.exists()
