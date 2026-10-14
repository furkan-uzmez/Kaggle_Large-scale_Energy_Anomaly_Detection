"""Real locks and authenticated ordinary fixture integration; no prepared repo data."""

from dataclasses import replace

import pytest

from full_campaign_fixtures import write_admission_fixture
from full_campaign_workers import write_result
from project.experiments import full_campaign_runtime as runtime
from project.experiments.full_campaign_admission import FullCampaignIntent
from project.experiments.full_campaign_capabilities import (
    resolve_full_campaign_capability,
)
from project.experiments.full_campaign_preflight import (
    preflight_full_campaign,
    verify_admitted_full_run,
)
from project.experiments.full_candidate_runner import run_full_campaign


def test_campaign_lock_keeps_same_inode_after_release(tmp_path):
    lock = runtime.CampaignLock(tmp_path)
    with lock:
        inode = lock.path.stat().st_ino
    assert lock.path.exists(), "unlinking a flock inode permits concurrent lock domains"
    with runtime.CampaignLock(tmp_path):
        assert lock.path.stat().st_ino == inode


def test_source_lock_is_shared_across_different_run_roots(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with (runtime.CampaignLock(tmp_path / "run-a", source_root=source),
          pytest.raises(runtime.CampaignLockError),
          runtime.CampaignLock(tmp_path / "run-b", source_root=source)):
        pytest.fail("second source-wide campaign must not acquire lock")
    assert not (tmp_path / "run-a").exists()
    assert not (tmp_path / "run-b").exists()


def test_exit_zero_with_artifact_still_requires_parent_validation(tmp_path):
    result = runtime.run_work_unit("unit", write_result, tmp_path,
                                   expected_artifacts=("result.json",))
    assert result.status == "awaiting_validation"
    assert result.exit_code == 0
    assert result.artifacts
    assert not list(tmp_path.glob("completion_*.json"))
    assert not runtime.verify_resume(tmp_path, "unit", ("result.json",))[0]


def fixture_intent(spec):
    return FullCampaignIntent(spec, True, resource_opt_out_record="user: fixture-only ordinary route acceptance")


def test_genuine_fixture_admission_has_explicit_ready_rows_without_writes(tmp_path):
    fixture = write_admission_fixture(tmp_path / "source")
    before = sorted(str(path) for path in fixture.root.rglob("*"))
    admitted = preflight_full_campaign(fixture.root, fixture_intent(fixture.spec))
    assert admitted.ready_experiment_ids == ("feat_missingness_indicators",)
    ready = [row.record for row in admitted.rows if row.record.status == "ready"]
    assert len(ready) == 1
    assert ready[0].blocker_codes == ()
    assert len(admitted.rows) == 21
    verify_admitted_full_run(fixture.root, admitted, spec=fixture.spec)
    assert sorted(str(path) for path in fixture.root.rglob("*")) == before
    with pytest.raises(ValueError, match="comparative"):
        verify_admitted_full_run(fixture.root, admitted, spec=fixture.spec, phase="completion")


def test_missing_encoding_binding_cannot_issue_admission(tmp_path):
    fixture = write_admission_fixture(tmp_path / "source")
    spec = replace(fixture.spec, sources=tuple(item for item in fixture.spec.sources if item.role != "encoding"))
    with pytest.raises(ValueError, match="encoding"):
        preflight_full_campaign(fixture.root, fixture_intent(spec))


def test_declared_encoding_cannot_admit_absolute_calendar_year_values(tmp_path):
    import polars as pl

    from project.experiments.full_run_contracts import _file_hash

    fixture = write_admission_fixture(tmp_path / "source")
    fixture.full_frame.with_columns(pl.lit(2000).alias("year_built")).write_parquet(fixture.data_path)
    spec = replace(fixture.spec, sources=tuple(
        replace(item, sha256=_file_hash(fixture.data_path)) if item.role == "data" else item
        for item in fixture.spec.sources))
    with pytest.raises(ValueError, match="encoding"):
        preflight_full_campaign(fixture.root, fixture_intent(spec))


def test_genuinely_admitted_campaign_rejects_occupied_destination(tmp_path):
    fixture = write_admission_fixture(tmp_path / "source")
    admitted = preflight_full_campaign(fixture.root, fixture_intent(fixture.spec))
    destination = tmp_path / "run"
    destination.mkdir()
    marker = destination / "retained-user-file"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        run_full_campaign(fixture.spec, fixture.root, destination, data_preflight=admitted,
                          capability=resolve_full_campaign_capability(fixture.root))
    assert list(destination.iterdir()) == [marker]


def test_changed_encoding_invalidates_genuine_admission_before_writes(tmp_path):
    fixture = write_admission_fixture(tmp_path / "source")
    admitted = preflight_full_campaign(fixture.root, fixture_intent(fixture.spec))
    (fixture.root / "docs/evidence/year_built_encoding_v1.json").write_text('{}')
    destination = tmp_path / "run"
    with pytest.raises(ValueError, match="hash|encoding"):
        run_full_campaign(fixture.spec, fixture.root, destination, data_preflight=admitted,
                          capability=resolve_full_campaign_capability(fixture.root))
    assert not destination.exists()


def test_genuine_ordinary_campaign_completes_selected_work_with_authenticated_proof(tmp_path):
    import json

    import pyarrow.parquet as pq

    from project.experiments.full_run_contracts import FullFitJournal

    fixture = write_admission_fixture(tmp_path / "source")
    admitted = preflight_full_campaign(fixture.root, fixture_intent(fixture.spec))
    root = tmp_path / "run"
    result = run_full_campaign(fixture.spec, fixture.root, root, data_preflight=admitted,
                               capability=resolve_full_campaign_capability(fixture.root))
    assert result["status"] == "finished_with_blockers"
    assert result["n03_eligible"] is False
    assert result["comparative_completion_available"] is True
    assert len(result["experiments"]) == 1
    assert len(result["inventory"]) == 21
    selected = next(row for row in result["inventory"] if row["experiment_id"] == "feat_missingness_indicators")
    assert selected["status"] == "completed"
    assert selected["n03_eligible"] is True
    assert json.loads((root / "completion/run_status.json").read_text()) == result
    assert json.loads((root / "run_status.json").read_text())["status"] == "awaiting_validation"
    oof = pq.read_table(root / "feat_missingness_indicators_oof.parquet")
    assert oof.num_rows == 52 * 3 * 2
    assert set(oof["fold"].to_pylist()) == {0, 1, 2, 3}
    assert set(oof["repeat"].to_pylist()) == {0, 1, 2}
    assert set(oof["n03_eligible"].to_pylist()) == {False}
    with FullFitJournal(root / "journal", fixture.spec, source_root=fixture.root) as journal:
        assert len(journal.attempts()) == 24
        assert all(attempt.status == "completed" for attempt in journal.attempts())
    assert len(list((root / "attempts").glob("*/worker-request.json"))) == 24
    assert not list((root / "attempts").glob("*/completion_*.json"))
    assert not (root / "feat_missingness_indicators/manifest.json").exists()


def test_runner_honors_an_existing_source_wide_lock_before_destination_writes(tmp_path):
    fixture = write_admission_fixture(tmp_path / "source")
    admitted = preflight_full_campaign(fixture.root, fixture_intent(fixture.spec))
    destination = tmp_path / "another-run"
    with (runtime.CampaignLock(tmp_path / "first-run", source_root=fixture.root),
          pytest.raises(runtime.CampaignLockError)):
        run_full_campaign(fixture.spec, fixture.root, destination, data_preflight=admitted,
                          capability=resolve_full_campaign_capability(fixture.root))
    assert not destination.exists()


def test_corrupt_real_worker_shard_terminalizes_active_journal_before_next_fit(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from project.experiments import full_candidate_runner as runner
    from project.experiments.full_run_contracts import FullFitJournal

    fixture = write_admission_fixture(tmp_path / "source")
    admitted = preflight_full_campaign(fixture.root, fixture_intent(fixture.spec))
    original = runner.run_work_unit
    calls = []

    def corrupt_after_real_exit(*args, **kwargs):
        outcome = original(*args, **kwargs)
        calls.append(outcome)
        shard = kwargs["work_dir"] / "shard.parquet"
        table = pq.read_table(shard)
        target = table["target"].to_pylist()
        target[0] = 1 - target[0]
        field = table.schema.field("target")
        table = table.set_column(table.schema.get_field_index("target"), field,
                                 pa.array(target, type=field.type))
        pq.write_table(table, shard)
        return outcome

    monkeypatch.setattr(runner, "run_work_unit", corrupt_after_real_exit)
    root = tmp_path / "run"
    with pytest.raises(ValueError, match="authoritative label/key"):
        runner.run_full_campaign(fixture.spec, fixture.root, root, data_preflight=admitted,
                                 capability=resolve_full_campaign_capability(fixture.root))
    assert len(calls) == 1
    assert calls[0].status == "awaiting_validation"
    with FullFitJournal(root / "journal", fixture.spec, source_root=fixture.root) as journal:
        attempts = journal.attempts()
        assert len(attempts) == 1
        assert attempts[0].status == "failed"
        assert attempts[0].evidence == ()
    assert not (root / "feat_missingness_indicators_oof.parquet").exists()
    import json

    status = json.loads((root / "run_status.json").read_text())
    assert status["status"] == "failed"
    assert len(status["inventory"]) == 21
    assert status["n03_eligible"] is False


def test_public_capability_constructor_never_grants_reviewed_route(tmp_path):
    from project.experiments.full_campaign_capabilities import (
        FullCampaignCapability,
        verify_full_campaign_capability,
    )

    fixture = write_admission_fixture(tmp_path / "source")
    genuine = resolve_full_campaign_capability(fixture.root)
    assert genuine.available
    verify_full_campaign_capability(fixture.root, genuine)
    with pytest.raises(ValueError, match="forged"):
        verify_full_campaign_capability(fixture.root, FullCampaignCapability(str(fixture.root), True, ()))
    with pytest.raises(ValueError, match="hash"):
        (fixture.root / "src/project/experiments/full_snapshot_codec.py").write_text("# not reviewed")
        verify_full_campaign_capability(fixture.root, genuine)
