"""Tests for OOF artifact validation (Task 6 RED phase).

All imports of new production modules are deferred inside test bodies using
importlib so that pytest collection succeeds before the modules are created.
"""

from __future__ import annotations

import importlib
import importlib.util
import math

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_validate_oof_artifact():
    """Return validate_oof_artifact, asserting the module and function exist."""
    pkg_spec = importlib.util.find_spec("project.models")
    assert pkg_spec is not None, "project.models package must exist"

    mod_spec = importlib.util.find_spec("project.models.artifacts")
    assert mod_spec is not None, (
        "project.models.artifacts must be created for Task 6"
    )

    mod = importlib.import_module("project.models.artifacts")
    fn = getattr(mod, "validate_oof_artifact", None)
    assert fn is not None, (
        "validate_oof_artifact must be defined in project.models.artifacts"
    )
    return fn


def _polars():
    """Return polars, asserting it is importable."""
    spec = importlib.util.find_spec("polars")
    assert spec is not None, "polars must be installed"
    return importlib.import_module("polars")


def _sklearn_metrics():
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    return roc_auc_score, average_precision_score, log_loss, brier_score_loss


# ---------------------------------------------------------------------------
# Synthetic fixture builders
# ---------------------------------------------------------------------------

_N_BUILDINGS = 8
_N_FOLDS = 4
_ROWS_PER_BUILDING = 2  # 2 observations per building → 16 total dev rows

# Fixed hashes – arbitrary but deterministic strings for testing
_SPLIT_SHA = "aabbccdd" * 8
_DATA_SHA = "11223344" * 8
_CONFIG_SHA = "deadbeef" * 8

# Candidate IDs used in valid OOF
_CANDIDATES = ["dummy_prior", "logistic_l2", "hist_gradient_boosting"]


def _make_expected_rows(pl):
    """Build the expected development manifest rows (no predictions)."""
    import pandas as pd

    rows = []
    row_index = 0
    for building_id in range(_N_BUILDINGS):
        fold = building_id % _N_FOLDS  # assign buildings to 4 folds
        for obs in range(_ROWS_PER_BUILDING):
            ts = pd.Timestamp("2022-01-01") + pd.Timedelta(hours=obs)
            rows.append(
                {
                    "row_index": row_index,
                    "building_id": building_id,
                    "timestamp": ts,
                    "fold": fold,
                    "target": obs % 2,  # alternates 0,1 so both classes present
                    "split_sha256": _SPLIT_SHA,
                    "data_sha256": _DATA_SHA,
                    "config_sha256": _CONFIG_SHA,
                    "run_id": "run-smoke-001",
                }
            )
            row_index += 1

    return pl.from_pandas(pd.DataFrame(rows))


def _make_oof(pl, candidates=None, extra_cols=None, bad_prediction=None):
    """
    Build a valid OOF DataFrame for the given candidate IDs.

    Parameters
    ----------
    candidates:
        List of candidate_id strings; defaults to _CANDIDATES.
    extra_cols:
        Dict of extra column name → list to append.
    bad_prediction:
        If a float (incl. NaN / inf), overwrite all prediction values.
    """
    import pandas as pd

    if candidates is None:
        candidates = _CANDIDATES

    rows = []
    row_index = 0
    for building_id in range(_N_BUILDINGS):
        fold = building_id % _N_FOLDS
        for obs in range(_ROWS_PER_BUILDING):
            ts = pd.Timestamp("2022-01-01") + pd.Timedelta(hours=obs)
            target = obs % 2
            for candidate_id in candidates:
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "recipe_id": f"recipe_{candidate_id}",
                        "model_family": candidate_id,
                        "model_config_id": f"cfg_{candidate_id}",
                        "row_index": row_index,
                        "repeat": 0,
                        "fold": fold,
                        "building_id": building_id,
                        "timestamp": ts,
                        "prediction": 0.5 if bad_prediction is None else bad_prediction,
                        "target": target,
                        "split_sha256": _SPLIT_SHA,
                        "data_sha256": _DATA_SHA,
                        "config_sha256": _CONFIG_SHA,
                        "run_id": "run-smoke-001",
                    }
                )
            row_index += 1

    df = pd.DataFrame(rows)
    if extra_cols:
        for col, values in extra_cols.items():
            df[col] = values[: len(df)] if len(values) >= len(df) else values

    return pl.from_pandas(df)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_validate_oof_artifact_function_exists():
    """validate_oof_artifact is importable and callable."""
    fn = _get_validate_oof_artifact()
    assert callable(fn)


def test_validate_oof_artifact_returns_metric_dict_for_valid_input():
    """Valid OOF yields per-candidate metric dicts with the four required keys."""
    pl = _polars()
    fn = _get_validate_oof_artifact()

    expected_rows = _make_expected_rows(pl)
    oof = _make_oof(pl)

    result = fn(oof, expected_rows)

    assert isinstance(result, dict), "validate_oof_artifact must return a dict"
    for candidate_id in _CANDIDATES:
        assert candidate_id in result, f"{candidate_id} must appear in result"
        metrics = result[candidate_id]
        assert isinstance(metrics, dict), "Per-candidate result must be a dict"
        assert "roc_auc" in metrics, "roc_auc must be reported"
        assert "average_precision" in metrics, "average_precision must be reported"
        assert "log_loss" in metrics, "log_loss must be reported"
        assert "brier_score" in metrics, "brier_score must be reported"


def test_validate_oof_artifact_metrics_match_sklearn_recomputation():
    """Pooled metrics in the result must match sklearn recalculation."""
    pl = _polars()
    fn = _get_validate_oof_artifact()
    roc_auc_score, average_precision_score, log_loss_fn, brier_score_loss = (
        _sklearn_metrics()
    )

    expected_rows = _make_expected_rows(pl)
    oof = _make_oof(pl)

    result = fn(oof, expected_rows)

    for candidate_id in _CANDIDATES:
        # Collect pooled targets and predictions for this candidate
        oof_candidate = oof.filter(pl.col("candidate_id") == candidate_id).sort(
            ["building_id", "timestamp"]
        )
        targets = oof_candidate["target"].to_numpy()
        preds = oof_candidate["prediction"].to_numpy()

        # Only compute if both classes present in targets
        if targets.sum() == 0 or targets.sum() == len(targets):
            continue  # degenerate; skip recompute check

        expected_roc = roc_auc_score(targets, preds)
        expected_ap = average_precision_score(targets, preds)
        expected_ll = log_loss_fn(targets, preds)
        expected_bs = brier_score_loss(targets, preds)

        metrics = result[candidate_id]
        assert math.isclose(metrics["roc_auc"], expected_roc, rel_tol=1e-6), (
            f"roc_auc mismatch for {candidate_id}"
        )
        assert math.isclose(metrics["average_precision"], expected_ap, rel_tol=1e-6), (
            f"average_precision mismatch for {candidate_id}"
        )
        assert math.isclose(metrics["log_loss"], expected_ll, rel_tol=1e-6), (
            f"log_loss mismatch for {candidate_id}"
        )
        assert math.isclose(metrics["brier_score"], expected_bs, rel_tol=1e-6), (
            f"brier_score mismatch for {candidate_id}"
        )


def test_validate_oof_artifact_rejects_missing_required_column():
    """OOF missing any required field must raise a ValueError."""
    pl = _polars()
    fn = _get_validate_oof_artifact()

    required_fields = [
        "candidate_id",
        "recipe_id",
        "model_family",
        "model_config_id",
        "row_index",
        "repeat",
        "fold",
        "building_id",
        "timestamp",
        "prediction",
        "target",
        "split_sha256",
        "data_sha256",
        "config_sha256",
        "run_id",
    ]

    expected_rows = _make_expected_rows(pl)
    oof_full = _make_oof(pl)

    for field in required_fields:
        oof_missing = oof_full.drop(field)
        with pytest.raises(ValueError, match=r".+"):
            fn(oof_missing, expected_rows)


def test_validate_oof_artifact_rejects_nan_predictions():
    """OOF with NaN predictions must raise a ValueError."""
    pl = _polars()
    fn = _get_validate_oof_artifact()

    expected_rows = _make_expected_rows(pl)
    oof = _make_oof(pl, bad_prediction=float("nan"))

    with pytest.raises(ValueError, match=r"nan|finite|invalid|out.of.range"):
        fn(oof, expected_rows)


def test_validate_oof_artifact_rejects_out_of_range_predictions():
    """OOF with prediction > 1 must raise a ValueError."""
    pl = _polars()
    fn = _get_validate_oof_artifact()

    expected_rows = _make_expected_rows(pl)
    oof = _make_oof(pl, bad_prediction=1.5)

    with pytest.raises(ValueError, match=r"range|\[0.*1\]|invalid|out.of.range"):
        fn(oof, expected_rows)


def test_validate_oof_artifact_rejects_negative_predictions():
    """OOF with prediction < 0 must raise a ValueError."""
    pl = _polars()
    fn = _get_validate_oof_artifact()

    expected_rows = _make_expected_rows(pl)
    oof = _make_oof(pl, bad_prediction=-0.1)

    with pytest.raises(ValueError, match=r"range|\[0.*1\]|invalid|out.of.range"):
        fn(oof, expected_rows)


def test_validate_oof_artifact_rejects_duplicate_task_key():
    """OOF with a duplicated (candidate_id, row_index, repeat) must be rejected."""
    pl = _polars()
    fn = _get_validate_oof_artifact()

    expected_rows = _make_expected_rows(pl)
    oof = _make_oof(pl)

    # Duplicate the entire OOF (all rows appear twice → all keys duplicated)
    oof_duplicated = pl.concat([oof, oof])

    with pytest.raises(ValueError, match=r"duplicate|uniqueness|task.key"):
        fn(oof_duplicated, expected_rows)


def test_validate_oof_artifact_rejects_vault_rows():
    """OOF containing vault-building rows (not in expected_rows) must be rejected."""
    pl = _polars()
    fn = _get_validate_oof_artifact()

    expected_rows = _make_expected_rows(pl)
    oof = _make_oof(pl)

    import pandas as pd

    # Add a row for a building_id NOT in expected_rows → vault leakage
    vault_row = pl.from_pandas(
        pd.DataFrame(
            [
                {
                    "candidate_id": "dummy_prior",
                    "recipe_id": "recipe_dummy_prior",
                    "model_family": "dummy_prior",
                    "model_config_id": "cfg_dummy_prior",
                    "row_index": 9999,
                    "repeat": 0,
                    "fold": 0,
                    "building_id": 999,  # vault / unknown building
                    "timestamp": pd.Timestamp("2022-01-01"),
                    "prediction": 0.5,
                    "target": 0,
                    "split_sha256": _SPLIT_SHA,
                    "data_sha256": _DATA_SHA,
                    "config_sha256": _CONFIG_SHA,
                    "run_id": "run-smoke-001",
                }
            ]
        )
    )
    oof_with_vault = pl.concat([oof, vault_row])

    with pytest.raises(ValueError, match=r"vault|extra|unexpected|coverage"):
        fn(oof_with_vault, expected_rows)


def test_validate_oof_artifact_rejects_missing_dev_rows():
    """OOF with fewer rows than expected (some dev rows missing) must be rejected."""
    pl = _polars()
    fn = _get_validate_oof_artifact()

    expected_rows = _make_expected_rows(pl)
    oof_full = _make_oof(pl)

    # Remove rows for one candidate entirely for one building
    oof_partial = oof_full.filter(
        ~((pl.col("building_id") == 0) & (pl.col("candidate_id") == "dummy_prior"))
    )

    with pytest.raises(ValueError, match=r"missing|coverage|incomplete"):
        fn(oof_partial, expected_rows)


def test_validate_oof_artifact_rejects_target_mismatch():
    """OOF with target values that differ from expected_rows must be rejected."""
    pl = _polars()
    fn = _get_validate_oof_artifact()

    expected_rows = _make_expected_rows(pl)
    oof = _make_oof(pl)

    # Overwrite all targets in the OOF to a wrong value
    oof_wrong_target = oof.with_columns(pl.lit(99).cast(pl.Int64).alias("target"))

    with pytest.raises(ValueError, match=r"target|mismatch|align"):
        fn(oof_wrong_target, expected_rows)


def test_validate_oof_artifact_rejects_hash_mismatch():
    """OOF with a wrong split_sha256 must be rejected."""
    pl = _polars()
    fn = _get_validate_oof_artifact()

    expected_rows = _make_expected_rows(pl)
    oof = _make_oof(pl)

    # Change the split hash in expected rows to one that won't match OOF
    expected_rows_wrong_hash = expected_rows.with_columns(
        pl.lit("wronghash" * 7).alias("split_sha256")
    )

    with pytest.raises(ValueError, match=r"hash|sha256|mismatch"):
        fn(oof, expected_rows_wrong_hash)


def test_validate_oof_artifact_coverage_is_exactly_per_candidate():
    """Each candidate must have exactly one prediction per development row."""
    pl = _polars()
    fn = _get_validate_oof_artifact()

    expected_rows = _make_expected_rows(pl)
    oof = _make_oof(pl)

    result = fn(oof, expected_rows)

    total_dev_rows = _N_BUILDINGS * _ROWS_PER_BUILDING
    for candidate_id in _CANDIDATES:
        oof_candidate = oof.filter(pl.col("candidate_id") == candidate_id)
        assert len(oof_candidate) == total_dev_rows, (
            f"{candidate_id} must have exactly {total_dev_rows} OOF rows, "
            f"got {len(oof_candidate)}"
        )

    # result must contain exactly the candidate IDs present in the OOF
    assert set(result.keys()) == set(_CANDIDATES)
