"""Full sequential runner: shared accounting protocol and batch schema tests."""

from __future__ import annotations

import hashlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from full_campaign_fixtures import write_admission_fixture, write_fixture
from project.experiments import full_candidate_runner as runner
from project.experiments.candidate_contracts import FitBudget, FitCounter
from project.experiments.fit_accounting import FitAccounting
from project.experiments.full_run_contracts import FullFitJournal
from project.models.candidate_artifacts import (
    CANDIDATE_OOF_SCHEMA,
    candidate_oof_schema,
)


def test_fitcounter_satisfies_shared_accounting_protocol():
    counter = FitCounter(FitBudget(20, 4, 30.0, 256 * 1024**2, 1))
    assert isinstance(counter, FitAccounting)


def test_accountant_charges_exactly_one_unit_per_fit_and_rejects_unknown(tmp_path):
    fixture = write_fixture(tmp_path / "campaign")
    journal = FullFitJournal(tmp_path / "journal", fixture.spec, source_root=fixture.root)
    try:
        plan = fixture.spec.planned_units
        accountant = runner.FullFitAccountant(journal, plan)
        assert isinstance(accountant, FitAccounting)
        assert accountant.reserved_model_fits == len(plan)
        with pytest.raises(ValueError):
            accountant.open_units(("not/a/planned/unit",))
        first = plan[0]
        accountant.open_units((first,))
        accountant.before_fit()
        # A second before_fit without finishing the active attempt is illegal.
        with pytest.raises(ValueError):
            accountant.before_fit()
        with pytest.raises(ValueError):
            accountant.finish_active("completed")  # no evidence
        (runner.ArtifactEvidence("x.json", hashlib.sha256(b"{}").hexdigest()),)
        accountant.finish_active("failed", reason="test", evidence=())
        assert accountant.used_model_fits == 1
    finally:
        journal.close()


def test_run_full_campaign_rejects_unadmitted_spec_before_journal_or_oof(tmp_path):
    fixture = write_fixture(tmp_path / "campaign")
    run_root = tmp_path / "run"
    with pytest.raises(TypeError, match="FullDataPreflight"):
        runner.run_full_campaign(fixture.spec, fixture.root, run_root)
    assert not run_root.exists()


def test_batch_table_matches_canonical_schema(tmp_path):
    fixture = write_fixture(tmp_path / "campaign")

    batch = {
        "row_index": np.array([0, 1]), "building_id": ["b0", "b0"],
        "timestamp": np.array([1_700_000_000_000_000_000, 1_700_000_000_000_000_001], dtype="int64"),
        "target": np.array([0, 1]), "split_seed": 42, "repeat": 0, "fold": 0,
    }
    table = runner._batch_table(
        batch, np.array([0.2, 0.8]), fixture.spec.candidates[0].spec,
        type("Arm", (), {"arm_id": "challenger", "score_kind": "positive_probability"})(),
        fixture.spec, 0, 0, "d" * 64,
    )
    assert table.schema.equals(CANDIDATE_OOF_SCHEMA, check_metadata=True)


def test_batch_table_accepts_explicit_int64_schema(tmp_path):
    """Regression: an int64 row batch must build an int64 OOF table."""
    fixture = write_fixture(tmp_path / "campaign")
    batch = {
        "row_index": np.array([0, 1]), "building_id": np.array([1, 2], dtype="int64"),
        "timestamp": np.array([1_700_000_000_000_000_000, 1_700_000_000_000_000_001], dtype="int64"),
        "target": np.array([0, 1]), "split_seed": 42, "repeat": 0, "fold": 0,
    }
    schema = candidate_oof_schema(pa.int64(), pa.timestamp("ns"))
    table = runner._batch_table(
        batch, np.array([0.2, 0.8]), fixture.spec.candidates[0].spec,
        type("Arm", (), {"arm_id": "challenger", "score_kind": "positive_probability"})(),
        fixture.spec, 0, 0, "d" * 64, schema=schema,
    )
    assert table.schema.equals(schema, check_metadata=True)
    assert table.column("building_id").to_pylist() == [1, 2]


def test_oof_schema_preserves_source_microsecond_unit():
    """Regression: the OOF schema must follow the index time unit, not hardcode ns.

    The real development index stores microsecond physical ticks. Hardcoding a
    nanosecond Arrow type scaled every shard timestamp by 1000 and made the
    parent reject otherwise-correct shards as a row/key mismatch.
    """
    from project.experiments import full_candidate_io as io

    meta = io._IndexMeta(
        run_binding="0" * 64, data_sha256="0" * 64, split_sha256="0" * 64,
        data_path="outputs/prepared/train_joined.parquet", timestamp_unit="us",
        timestamp_time_zone=None, building_dtype="int64", predictor_types={},
        dev_row_count=1, identity_sha256="0" * 64,
    )
    assert runner._oof_schema(meta).field("timestamp").type == pa.timestamp("us")


def test_batch_table_honors_microsecond_schema_without_ns_scaling(tmp_path):
    """Regression: a microsecond batch must keep its physical ticks."""
    fixture = write_fixture(tmp_path / "campaign")
    ticks = np.array([1_700_000_000_000_000, 1_700_000_000_000_001], dtype="int64")
    batch = {
        "row_index": np.array([0, 1]), "building_id": np.array([1, 2], dtype="int64"),
        "timestamp": ticks.view("datetime64[us]"),
        "target": np.array([0, 1]), "split_seed": 42, "repeat": 0, "fold": 0,
    }
    schema = candidate_oof_schema(pa.int64(), pa.timestamp("us"))
    table = runner._batch_table(
        batch, np.array([0.2, 0.8]), fixture.spec.candidates[0].spec,
        type("Arm", (), {"arm_id": "challenger", "score_kind": "positive_probability"})(),
        fixture.spec, 0, 0, "d" * 64, schema=schema,
    )
    assert table.schema.equals(schema, check_metadata=True)
    import polars as pl
    assert pl.from_arrow(table.column("timestamp")).cast(pl.Int64).to_list() == ticks.tolist()


def test_runner_rejects_diagnostic_spec_with_typeerror(tmp_path):
    from project.experiments.candidate_contracts import (
        CandidateSpec,
        FitBudget,
        RunSpec,
    )
    run_spec = RunSpec(
        1,
        (CandidateSpec("feat_missingness_indicators", ("meter_reading",), "hist_gradient_boosting", (), ()),),
        FitBudget(1, 1, 1.0, 1024, 1),
        "synthetic_smoke",
        "notebooks/02_baseline_and_candidate_design.ipynb",
        "cell",
    )
    with pytest.raises(TypeError, match="typed FullRunSpec"):
        runner.run_full_campaign(run_spec, tmp_path, tmp_path / "run")


def test_runner_retains_blocked_source_record_without_fake_execution(tmp_path):
    from dataclasses import replace

    from project.experiments.full_campaign_admission import (
        FullCampaignIntent,
        inspect_full_campaign,
    )
    from project.experiments.full_campaign_preflight import preflight_full_campaign

    fixture = write_admission_fixture(tmp_path / "campaign")
    blocked_cand = runner.CandidateSpec(
        "feat_group_aggregates_blocked",
        fixture.spec.candidates[0].spec.feature_pool,
        "hist_gradient_boosting",
        (),
        (),
    )
    from project.experiments.full_run_contracts import FullCandidateSnapshot
    blocked_snapshot = FullCandidateSnapshot(
        blocked_cand,
        ("control", "challenger"),
        "notebooks/02_baseline_and_candidate_design.ipynb",
        "cand-matrix-render",
    )
    spec = replace(
        fixture.spec,
        candidates=(blocked_snapshot,),
    )
    intent = FullCampaignIntent(spec, True, resource_opt_out_record="user: fixture only")
    source_intent = replace(intent, spec=replace(spec, sources=tuple(
        item for item in spec.sources if item.role != "encoding")))
    report = inspect_full_campaign(fixture.root, source_intent)
    record = next(row.record for row in report.rows if row.record.experiment_id == blocked_cand.experiment_id)
    assert record.status == "blocked"
    assert "blocked_group_aggregates" in record.blocker_codes
    with pytest.raises(ValueError, match="ordinary_route_not_reviewed"):
        preflight_full_campaign(fixture.root, intent)
    assert not (tmp_path / "run").exists()


def test_runner_executes_integer_building_candidate_end_to_end(tmp_path):
    """Regression: the parent runner must validate and union int64 OOF shards.

    The first real campaign died inside the worker before the runner's
    ``_validate_shard`` and OOF-union paths ran. This drives the full parent
    handshake with an integer-building admission fixture so those int64 paths
    are exercised end-to-end against a real subprocess.
    """
    import json
    from pathlib import Path

    import pyarrow.parquet as pq

    from project.experiments.full_campaign_admission import FullCampaignIntent
    from project.experiments.full_campaign_capabilities import (
        resolve_full_campaign_capability,
    )
    from project.experiments.full_campaign_preflight import preflight_full_campaign

    fixture = write_admission_fixture(tmp_path / "campaign", int_buildings=True)
    spec = fixture.spec
    admission = preflight_full_campaign(fixture.root, FullCampaignIntent(
        spec, True, resource_opt_out_record="user: fixture only"))
    capability = resolve_full_campaign_capability(fixture.root)

    run_dir = tmp_path / "run"
    res = runner.run_full_campaign(
        spec,
        fixture.root,
        run_dir,
        data_preflight=admission,
        capability=capability,
    )
    assert res["status"] == "finished_with_blockers"
    assert len(res["experiments"]) == 1
    assert res["experiments"][0]["status"] == "completed"
    oof_path = Path(res["experiments"][0]["oof_path"])
    table = pq.read_table(oof_path)
    assert table.schema.field("building_id").type == pa.int64()
    assert table.num_rows == 52 * 3 * 2
    validation = json.loads((run_dir / "feat_missingness_indicators_transport_validation.json").read_text())
    assert validation["status"] == "validated"
    assert res["comparative_completion_available"] is True
    assert (run_dir / "completion/manifest.json").is_file()


def test_runner_executes_microsecond_integer_candidate_end_to_end(tmp_path):
    """Regression: a microsecond development index must survive the full handshake.

    The second real campaign died because the worker rescaled microsecond
    timestamps to nanoseconds, so the parent's authoritative tick comparison
    rejected every row. This drives the whole parent/worker/validation path with
    a microsecond, integer-building admission fixture.
    """
    import json
    from pathlib import Path

    import pyarrow.parquet as pq

    from project.experiments.full_campaign_admission import FullCampaignIntent
    from project.experiments.full_campaign_capabilities import (
        resolve_full_campaign_capability,
    )
    from project.experiments.full_campaign_preflight import preflight_full_campaign

    fixture = write_admission_fixture(
        tmp_path / "campaign", int_buildings=True, timestamp_unit="us")
    spec = fixture.spec
    admission = preflight_full_campaign(fixture.root, FullCampaignIntent(
        spec, True, resource_opt_out_record="user: fixture only"))
    capability = resolve_full_campaign_capability(fixture.root)

    run_dir = tmp_path / "run"
    res = runner.run_full_campaign(
        spec,
        fixture.root,
        run_dir,
        data_preflight=admission,
        capability=capability,
    )
    assert res["status"] == "finished_with_blockers"
    assert len(res["experiments"]) == 1
    assert res["experiments"][0]["status"] == "completed"
    oof_path = Path(res["experiments"][0]["oof_path"])
    table = pq.read_table(oof_path)
    assert table.schema.field("timestamp").type == pa.timestamp("us")
    assert table.schema.field("building_id").type == pa.int64()
    assert table.num_rows == 52 * 3 * 2
    validation = json.loads((run_dir / "feat_missingness_indicators_transport_validation.json").read_text())
    assert validation["status"] == "validated"
    assert res["comparative_completion_available"] is True
    assert (run_dir / "completion/manifest.json").is_file()


def test_runner_executes_genuinely_admitted_candidate_with_worker_subprocesses(tmp_path):
    import json
    from pathlib import Path

    import pyarrow.parquet as pq

    from project.experiments.full_campaign_admission import FullCampaignIntent
    from project.experiments.full_campaign_capabilities import (
        resolve_full_campaign_capability,
    )
    from project.experiments.full_campaign_preflight import preflight_full_campaign

    fixture = write_admission_fixture(tmp_path / "campaign")
    spec = fixture.spec
    admission = preflight_full_campaign(fixture.root, FullCampaignIntent(
        spec, True, resource_opt_out_record="user: fixture only"))
    capability = resolve_full_campaign_capability(fixture.root)

    run_dir = tmp_path / "run"
    res = runner.run_full_campaign(
        spec,
        fixture.root,
        run_dir,
        data_preflight=admission,
        capability=capability,
    )
    assert res["status"] == "finished_with_blockers"
    assert len(res["experiments"]) == 1
    assert res["experiments"][0]["status"] == "completed"
    oof_path = Path(res["experiments"][0]["oof_path"])
    assert oof_path.is_file()
    assert pq.ParquetFile(oof_path).metadata.num_rows == 52 * 3 * 2
    validation = json.loads((run_dir / "feat_missingness_indicators_transport_validation.json").read_text())
    assert validation["status"] == "validated"
    manifest_path = run_dir / "feat_missingness_indicators" / "manifest.json"
    assert not manifest_path.exists()
    assert res["comparative_completion_available"] is True
    assert (run_dir / "completion/manifest.json").is_file()


def test_stream_shards_stays_within_writer_batch_bound(tmp_path):
    """Regression: the OOF union must stream each shard within the writer cap.

    ``_stream_shards`` previously read shards with pyarrow's default 65536-row
    batch size, while ``write_candidate_oof_batches`` rejects any batch above
    ``MAX_BATCH_ROWS`` (64000). Every real development shard is far larger than a
    single batch, so the first union batch was rejected and the campaign died
    after all fits had already validated. Tiny fixtures never exceeded 64000
    rows, so the flaw stayed hidden until a real union ran.
    """
    from project.models.candidate_artifacts import MAX_BATCH_ROWS

    rows = 70_000
    table = pa.table({
        "row_index": np.arange(rows, dtype=np.int64),
        "value": np.arange(rows, dtype=np.float64),
    })
    shard = tmp_path / "shard.parquet"
    with pq.ParquetWriter(shard, table.schema, compression="zstd") as writer:
        writer.write_table(table, row_group_size=MAX_BATCH_ROWS)

    batches = list(runner._stream_shards([shard]))
    assert sum(batch.num_rows for batch in batches) == rows
    assert all(0 < batch.num_rows <= MAX_BATCH_ROWS for batch in batches)
