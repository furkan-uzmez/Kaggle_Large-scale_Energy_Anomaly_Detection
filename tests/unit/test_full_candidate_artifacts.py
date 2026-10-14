"""Disk-backed full-OOF coverage and exact-metric behavior tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from project.experiments.candidate_contracts import CandidateScore
from project.models import candidate_artifacts as artifacts
from project.models import full_candidate_artifacts as full
from project.models.candidate_scores import evaluate_candidate_metrics

NOTEBOOK = "notebooks/02_baseline_and_candidate_design.ipynb"
SPLIT_SHA = "a" * 64
DATA_SHA = "b" * 64
CONFIG_SHA = "c" * 64
RUN_ID = "fixture-run"
BASE_NS = 1_700_000_000_000_000_000
BUILDINGS = [f"b{i}" for i in range(8)]
ROWS_PER_BUILDING = 3
SEEDS = {0: 42, 1: 43, 2: 44}
FOLDS = {building: index // 2 for index, building in enumerate(BUILDINGS)}
REPEATED_PAIRS = (
    ((0, 1), (2, 3), (4, 5), (6, 7)),
    ((0, 2), (1, 3), (4, 6), (5, 7)),
    ((0, 3), (1, 2), (4, 7), (5, 6)),
)


def _spec(kind: str, wait_hash: str = "d" * 64) -> dict:
    return {
        "recipe_id": "feat_missingness_indicators",
        "model_family": "hist_gradient_boosting",
        "model_config_id": "cfg",
        "experiment_id": "feat_missingness_indicators",
        "arm_id": "challenger",
        "score_kind": kind,
        "positive_label": 1,
        "model_fit_config_sha256": wait_hash,
    }


def manifest(candidates: dict | None = None) -> dict:
    return {
        "schema_version": 1,
        "source_notebook": NOTEBOOK,
        "source_cell_id": "cell",
        "vault_row_indices": [],
        "vault_building_ids": [],
        "protocol": {"repeat_ids": [0, 1, 2], "fold_ids": [0, 1, 2, 3],
                     "split_seeds": {str(k): v for k, v in SEEDS.items()}},
        "candidates": candidates or {"prob": _spec("positive_probability")},
        "split_sha256": SPLIT_SHA,
        "data_sha256": DATA_SHA,
        "config_sha256": CONFIG_SHA,
        "run_id": RUN_ID,
        "execution_mode": "full",
        "non_comparative": False,
        "n03_eligible": False,
    }


def _dev_rows(repeat: int, distinct=False):
    rows = []
    for index, building in enumerate(BUILDINGS):
        for j in range(ROWS_PER_BUILDING):
            row_index = index * ROWS_PER_BUILDING + j
            fold = next(f for f, pair in enumerate(REPEATED_PAIRS[repeat]) if index in pair) if distinct else FOLDS[building]
            rows.append({
                "row_index": row_index, "repeat": repeat, "fold": fold,
                "building_id": building, "timestamp_ns": BASE_NS + index * 10 + j,
                "target": (index + j) % 2, "split_seed": SEEDS[repeat],
                "split_sha256": SPLIT_SHA, "data_sha256": DATA_SHA,
                "config_sha256": CONFIG_SHA, "run_id": RUN_ID,
            })
    return rows


def expected_batches(distinct=False):
    for repeat in (0, 1, 2):
        frame = pl.DataFrame(_dev_rows(repeat, distinct)).with_columns(
            pl.col("timestamp_ns").cast(pl.Int64).cast(pl.Datetime("ns")).alias("timestamp")
        ).drop("timestamp_ns")
        yield frame


def _probability(target: int, row_index: int, repeat: int) -> float:
    value = 0.5 + 0.1 * (1 if target == 1 else -1) + 0.01 * ((row_index + repeat) % 3)
    return float(min(max(value, 0.001), 0.999))


def _candidate_rows(kind: str, candidate_id: str, meta: dict):
    rows = []
    for repeat in (0, 1, 2):
        for row in _dev_rows(repeat):
            if kind == "positive_probability":
                prediction = _probability(row["target"], row["row_index"], repeat)
            else:
                prediction = (1.0 if row["target"] == 1 else -1.0) + 0.01 * row["row_index"]
            rows.append({
                "candidate_id": candidate_id, "recipe_id": meta["recipe_id"],
                "model_family": meta["model_family"], "model_config_id": meta["model_config_id"],
                "row_index": row["row_index"], "repeat": repeat, "fold": row["fold"],
                "building_id": row["building_id"],
                "timestamp": BASE_NS + (row["row_index"] // ROWS_PER_BUILDING) * 10 + row["row_index"] % ROWS_PER_BUILDING,
                "prediction": prediction, "target": row["target"], "split_sha256": SPLIT_SHA,
                "data_sha256": DATA_SHA, "config_sha256": CONFIG_SHA, "run_id": RUN_ID,
                "split_seed": SEEDS[repeat], "experiment_id": meta["experiment_id"],
                "arm_id": meta["arm_id"], "score_kind": kind, "positive_label": 1,
                "execution_mode": "full", "non_comparative": False, "n03_eligible": False,
                "model_fit_config_sha256": meta["model_fit_config_sha256"],
            })
    return rows


def write_oof(path: Path, rows) -> Path:
    schema = artifacts.CANDIDATE_OOF_SCHEMA
    table = pa.Table.from_pylist(rows, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def probability_fixture(tmp_path: Path, meta=None):
    meta = meta or _spec("positive_probability")
    # Rows carry the schema natural-key timestamp independently; use integer ns in Arrow.
    rows = _candidate_rows("positive_probability", "prob", meta)
    oof = write_oof(tmp_path / "oof.parquet", rows)
    return oof, manifest({"prob": meta})


def test_valid_probability_artifact_matches_sklearn_repeat_metrics(tmp_path):
    oof, declaration = probability_fixture(tmp_path)
    payload = full.validate_full_candidate_oof(oof, list(expected_batches()), declaration)
    assert payload["status"] == "validated"
    assert payload["summary_basis"] == "equal_mean_of_repeats_not_pooled_rows"
    assert payload["expected_row_repeat_count"] == len(BUILDINGS) * ROWS_PER_BUILDING * 3

    # Oracle: compare each repeat metric to the installed project metric function.
    for repeat in (0, 1, 2):
        rows = [r for r in _candidate_rows("positive_probability", "prob", _spec("positive_probability"))
                if r["repeat"] == repeat]
        y = np.asarray([r["target"] for r in rows])
        p = np.asarray([r["prediction"] for r in rows])
        expected = evaluate_candidate_metrics(y, CandidateScore("positive_probability", p))
        recorded = next(r for r in payload["repeat_metrics"] if r["repeat"] == repeat)
        assert recorded["roc_auc"] == pytest.approx(expected["roc_auc"], abs=1e-9)
        assert recorded["average_precision"] == pytest.approx(expected["average_precision"], abs=1e-9)
        assert recorded["log_loss"] == pytest.approx(expected["log_loss"], rel=1e-9, abs=1e-12)
        assert recorded["brier_score"] == pytest.approx(expected["brier_score"], rel=1e-9, abs=1e-12)


def test_repeat_metrics_are_equal_mean_not_pooled(tmp_path):
    meta = _spec("positive_probability")
    # Distinct per-repeat quality: repeats 0/1 separate well, repeat 2 is noisy.
    rows = []
    for repeat in (0, 1, 2):
        for row in _dev_rows(repeat):
            if repeat in (0, 1):
                prediction = _probability(row["target"], row["row_index"], repeat)
            else:
                prediction = 0.5 + 0.001 * ((row["row_index"] * 11) % 7)
            rows.append({
                "candidate_id": "prob", "row_index": row["row_index"], "repeat": repeat,
                "fold": row["fold"], "building_id": row["building_id"],
                "timestamp": BASE_NS + (row["row_index"] // ROWS_PER_BUILDING) * 10 + row["row_index"] % ROWS_PER_BUILDING,
                "prediction": prediction, "target": row["target"], "split_seed": row["split_seed"],
                "split_sha256": SPLIT_SHA, "data_sha256": DATA_SHA, "config_sha256": CONFIG_SHA,
                "run_id": RUN_ID, "recipe_id": meta["recipe_id"], "model_family": meta["model_family"],
                "model_config_id": meta["model_config_id"], "experiment_id": meta["experiment_id"],
                "arm_id": meta["arm_id"], "score_kind": "positive_probability", "positive_label": 1,
                "execution_mode": "full", "non_comparative": False, "n03_eligible": False,
                "model_fit_config_sha256": meta["model_fit_config_sha256"],
            })
    oof = write_oof(tmp_path / "varied.parquet", rows)
    declaration = manifest({"prob": meta})
    payload = full.validate_full_candidate_oof(oof, list(expected_batches()), declaration)
    repeats = [r["roc_auc"] for r in payload["repeat_metrics"]]
    mean = payload["candidate_metrics"]["prob"]["mean_repeat_metrics"]["roc_auc"]
    assert mean == pytest.approx(float(np.mean(repeats)), abs=1e-12)
    assert len({round(value, 6) for value in repeats}) >= 2
    pooled = CandidateScore("positive_probability", np.asarray([r["prediction"] for r in rows]))
    pooled_auc = evaluate_candidate_metrics(np.asarray([r["target"] for r in rows]), pooled)["roc_auc"]
    assert abs(mean - pooled_auc) > 0.01
    assert payload["summary_basis"] == "equal_mean_of_repeats_not_pooled_rows"


def test_margin_candidate_has_no_probability_guardrails(tmp_path):
    meta = _spec("decision_margin")
    rows = _candidate_rows("decision_margin", "margin", meta)
    oof = write_oof(tmp_path / "margin.parquet", rows)
    payload = full.validate_full_candidate_oof(oof, list(expected_batches()), manifest({"margin": meta}))
    for repeat in payload["repeat_metrics"]:
        assert repeat["roc_auc"] is not None and repeat["average_precision"] is not None
        assert repeat["log_loss"] is None and repeat["brier_score"] is None
        assert repeat["guardrail_status"] == "requires_calibrated_probability"
    assert payload["candidate_metrics"]["margin"]["probability_pool_eligible"] is False


def _mutate(rows, predicate, **changes):
    return [{**row, **changes} if predicate(row) else row for row in rows]


def test_missing_expected_row_is_rejected(tmp_path):
    _oof, declaration = probability_fixture(tmp_path)
    dropped = [r for r in _candidate_rows("positive_probability", "prob", _spec("positive_probability"))
               if not (r["repeat"] == 0 and r["row_index"] == 0)]
    write_oof(tmp_path / "missing.parquet", dropped)
    with pytest.raises(ValueError):
        full.validate_full_candidate_oof(tmp_path / "missing.parquet", list(expected_batches()), declaration)


def test_duplicate_key_across_batches_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(full, "MAX_BATCH_ROWS", 5)
    meta = _spec("positive_probability")
    rows = _candidate_rows("positive_probability", "prob", meta)
    rows.insert(6, dict(rows[0]))
    oof = write_oof(tmp_path / "dup.parquet", rows)
    with pytest.raises(ValueError):
        full.validate_full_candidate_oof(oof, list(expected_batches()), manifest({"prob": meta}))


def test_extra_arm_is_rejected(tmp_path):
    _oof, declaration = probability_fixture(tmp_path)
    rows = _candidate_rows("positive_probability", "prob", _spec("positive_probability"))
    rows += _candidate_rows("positive_probability", "surprise", _spec("positive_probability"))
    write_oof(tmp_path / "extra.parquet", rows)
    with pytest.raises(ValueError):
        full.validate_full_candidate_oof(tmp_path / "extra.parquet", list(expected_batches()), declaration)


@pytest.mark.parametrize("field,value", [
    ("target", 1),
    ("model_fit_config_sha256", "e" * 64),
    ("split_seed", 999),
    ("fold", 3),
    ("data_sha256", "e" * 64),
    ("split_sha256", "e" * 64),
    ("config_sha256", "e" * 64),
])
def test_wrong_alignment_fields_are_rejected(tmp_path, field, value):
    _oof, declaration = probability_fixture(tmp_path)
    meta = _spec("positive_probability")
    rows = _candidate_rows("positive_probability", "prob", meta)
    victim = next(r for r in rows if r["repeat"] == 0 and r["row_index"] == 0 and r[field] != value)
    if field == "target":
        victim["target"] = 1 - victim["target"]
    else:
        victim[field] = value
    write_oof(tmp_path / "wrong.parquet", rows)
    with pytest.raises(ValueError):
        full.validate_full_candidate_oof(tmp_path / "wrong.parquet", list(expected_batches()), declaration)


def test_vault_row_is_rejected(tmp_path):
    oof, declaration = probability_fixture(tmp_path)
    declaration = dict(declaration)
    declaration["vault_row_indices"] = [5]
    with pytest.raises(ValueError):
        full.validate_full_candidate_oof(oof, list(expected_batches()), declaration)


def test_inconsistent_population_across_repeats_is_rejected(tmp_path):
    oof, declaration = probability_fixture(tmp_path)
    batches = list(expected_batches())
    poisoned = batches[2].with_columns(
        pl.when(pl.col("row_index") == 0).then(1 - pl.col("target")).otherwise(pl.col("target")).alias("target")
    )
    batches[2] = poisoned
    with pytest.raises(ValueError):
        full.validate_full_candidate_oof(oof, batches, declaration)


def test_metrics_not_computed_before_full_coverage(tmp_path):
    _oof, declaration = probability_fixture(tmp_path)
    rows = [r for r in _candidate_rows("positive_probability", "prob", _spec("positive_probability"))
            if not (r["repeat"] == 1 and r["row_index"] == 0)]
    write_oof(tmp_path / "partial.parquet", rows)
    with pytest.raises(ValueError):
        full.validate_full_candidate_oof(tmp_path / "partial.parquet", list(expected_batches()), declaration)


def test_distinct_repeat_partitions_allow_building_to_change_folds(tmp_path):
    rows = _candidate_rows("positive_probability", "prob", _spec("positive_probability"))
    for row in rows:
        position = BUILDINGS.index(row["building_id"])
        row["fold"] = next(f for f, pair in enumerate(REPEATED_PAIRS[row["repeat"]]) if position in pair)
    path = write_oof(tmp_path / "distinct.parquet", rows)
    payload = full.validate_full_candidate_oof(path, expected_batches(distinct=True), manifest())
    assert payload["status"] == "validated"
    assert len(payload["fold_metrics"]) == 12


def test_source_microseconds_match_observed_nanoseconds_losslessly(tmp_path):
    rows = _candidate_rows("positive_probability", "prob", _spec("positive_probability"))
    for row in rows:
        row["timestamp"] = BASE_NS + row["row_index"] * 1000
    path = write_oof(tmp_path / "canonical.parquet", rows)
    batches = [frame.with_columns(
        ((pl.lit(BASE_NS // 1000) + pl.col("row_index"))).cast(pl.Datetime("us")).alias("timestamp")
    ) for frame in expected_batches()]
    assert full.validate_full_candidate_oof(path, batches, manifest())["status"] == "validated"


@pytest.mark.parametrize("change", ["timezone", "building_dtype", "unit", "overflow"])
def test_expected_key_policy_does_not_silently_erase_schema(tmp_path, change):
    path, declaration = probability_fixture(tmp_path)
    batches = list(expected_batches())
    if change == "timezone":
        batches = [frame.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC")) for frame in batches]
    elif change == "building_dtype":
        batches = [frame.with_columns(pl.col("building_id").str.strip_prefix("b").cast(pl.Int64)) for frame in batches]
    elif change == "unit":
        batches = [frame.with_columns(pl.col("timestamp").cast(pl.Datetime("ms"))) for frame in batches]
    else:
        batches = [frame.with_columns(pl.lit(2**63 - 1).cast(pl.Datetime("us")).alias("timestamp")) for frame in batches]
    with pytest.raises(ValueError):
        full.validate_full_candidate_oof(path, batches, declaration)


def test_population_verification_never_materializes_sql_row_sets(tmp_path, monkeypatch):
    import sqlite3
    original = full.sqlite3.connect

    class StreamingCursor:
        def __init__(self, cursor):
            self.cursor = cursor

        def __getattr__(self, name):
            return getattr(self.cursor, name)

        def fetchall(self):
            pytest.fail("population/fit hash queries must stream or aggregate")

        def __iter__(self):
            return iter(self.cursor)

    class StreamingConnection(sqlite3.Connection):
        def execute(self, *args, **kwargs):
            return StreamingCursor(super().execute(*args, **kwargs))

    monkeypatch.setattr(full.sqlite3, "connect", lambda *a, **kw: original(*a, factory=StreamingConnection, **kw))
    path, declaration = probability_fixture(tmp_path)
    assert full.validate_full_candidate_oof(path, expected_batches(), declaration)["status"] == "validated"


def test_authoritative_natural_duplicate_across_batches_is_rejected(tmp_path):
    path, declaration = probability_fixture(tmp_path)
    batches = list(expected_batches())
    batches.insert(1, batches[0].head(1).with_columns(pl.lit(100, dtype=pl.Int64).alias("row_index")))
    with pytest.raises(ValueError, match="duplicate"):
        full.validate_full_candidate_oof(path, batches, declaration)


def test_hpo_hashes_are_fold_scoped_and_verified(tmp_path):
    meta = _spec("positive_probability")
    meta.update(experiment_id="regime_same_family_hpo_vs_default", recipe_id="regime_same_family_hpo_vs_default")
    meta.pop("model_fit_config_sha256")
    meta["fold_fit_config_sha256"] = {f"{r}:{f}": f"{r * 4 + f + 1:064x}" for r in range(3) for f in range(4)}
    rows = _candidate_rows("positive_probability", "prob", _spec("positive_probability"))
    for row in rows:
        row.update(experiment_id=meta["experiment_id"], recipe_id=meta["recipe_id"],
                   model_fit_config_sha256=meta["fold_fit_config_sha256"][f"{row['repeat']}:{row['fold']}"])
    declaration = manifest({"prob": meta})
    path = write_oof(tmp_path / "hpo.parquet", rows)
    assert full.validate_full_candidate_oof(path, expected_batches(), declaration)["status"] == "validated"
    rows[-1]["model_fit_config_sha256"] = "f" * 64
    path = write_oof(tmp_path / "hpo-wrong.parquet", rows)
    with pytest.raises(ValueError, match="fit hash"):
        full.validate_full_candidate_oof(path, expected_batches(), declaration)


@pytest.mark.parametrize("scores", [[0., 1., .5, .5], [.5] * 4, [1., 0., 1., 0.]])
def test_exact_grouped_ties_and_probability_extremes_match_installed_oracle(tmp_path, scores):
    rows = _candidate_rows("positive_probability", "prob", _spec("positive_probability"))
    for position, row in enumerate(rows):
        row["prediction"] = scores[position % len(scores)]
    path = write_oof(tmp_path / "ties.parquet", rows)
    payload = full.validate_full_candidate_oof(path, expected_batches(), manifest())
    for repeat in range(3):
        part = [row for row in rows if row["repeat"] == repeat]
        oracle = evaluate_candidate_metrics(np.array([row["target"] for row in part]),
            CandidateScore("positive_probability", np.array([row["prediction"] for row in part])))
        actual = next(row for row in payload["repeat_metrics"] if row["repeat"] == repeat)
        for metric in ("roc_auc", "average_precision", "log_loss", "brier_score"):
            assert actual[metric] == pytest.approx(oracle[metric], rel=1e-12, abs=1e-12)


def test_jointly_poisoned_repeat_labels_cannot_redefine_expected_population(tmp_path):
    rows = _candidate_rows("positive_probability", "prob", _spec("positive_probability"))
    rows = _mutate(rows, lambda row: row["repeat"] == 2 and row["row_index"] == 0, target=1)
    batches = list(expected_batches())
    batches[2] = batches[2].with_columns(
        pl.when(pl.col("row_index") == 0).then(1).otherwise(pl.col("target")).alias("target")
    )
    path = write_oof(tmp_path / "poisoned-population.parquet", rows)
    with pytest.raises(ValueError, match="population"):
        full.validate_full_candidate_oof(path, batches, manifest())


def test_opaque_integer_building_policy_is_supported_without_string_coercion(tmp_path):
    rows = _candidate_rows("positive_probability", "prob", _spec("positive_probability"))
    for row in rows:
        row["building_id"] = int(row["building_id"][1:])
    schema = artifacts.candidate_oof_schema(pa.int64(), pa.timestamp("ns"))
    path = tmp_path / "integer.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    batches = [frame.with_columns(pl.col("building_id").str.strip_prefix("b").cast(pl.Int64))
               for frame in expected_batches()]
    assert full.validate_full_candidate_oof(path, batches, manifest())["status"] == "validated"


def test_one_building_cannot_span_folds_within_the_same_repeat(tmp_path):
    path, declaration = probability_fixture(tmp_path)
    batches = list(expected_batches())
    batches[0] = batches[0].with_columns(
        pl.when(pl.col("row_index") == 0).then(1).otherwise(pl.col("fold")).alias("fold")
    )
    with pytest.raises(ValueError, match="within a repeat"):
        full.validate_full_candidate_oof(path, batches, declaration)


def test_extra_natural_key_cannot_replace_an_expected_key_at_equal_count(tmp_path):
    rows = _candidate_rows("positive_probability", "prob", _spec("positive_probability"))
    rows[0]["row_index"] = 999
    path = write_oof(tmp_path / "replacement.parquet", rows)
    with pytest.raises(ValueError, match="coverage|absent"):
        full.validate_full_candidate_oof(path, expected_batches(), manifest())


def test_matching_timezone_policy_and_microsecond_source_are_supported(tmp_path):
    rows = _candidate_rows("positive_probability", "prob", _spec("positive_probability"))
    for row in rows:
        row["timestamp"] = BASE_NS + row["row_index"] * 1000
    schema = artifacts.candidate_oof_schema(pa.string(), pa.timestamp("ns", "UTC"))
    path = tmp_path / "timezone.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    batches = [frame.with_columns(
        (pl.lit(BASE_NS // 1000) + pl.col("row_index")).cast(pl.Datetime("us", "UTC")).alias("timestamp")
    ) for frame in expected_batches()]
    assert full.validate_full_candidate_oof(path, batches, manifest())["status"] == "validated"
