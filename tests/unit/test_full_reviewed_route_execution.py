"""Real synthetic subprocess dispatch and independent route receipt boundaries."""
import json
from dataclasses import replace

import pytest

from full_campaign_fixtures import write_admission_fixture
from project.experiments.candidate_contracts import CandidateSpec
from project.experiments.full_campaign_admission import FullCampaignIntent
from project.experiments.full_campaign_capabilities import resolve_full_campaign_capability
from project.experiments.full_campaign_preflight import preflight_full_campaign
from project.experiments.full_candidate_runner import run_full_campaign
from project.experiments.full_campaign_evidence import verify_ordinary_campaign_completion
from project.experiments.full_run_contracts import FullFitJournal


@pytest.mark.slow
def test_synthetic_exact_pca_dispatch_independent_completion_and_tampering(tmp_path):
    fixture = write_admission_fixture(tmp_path / "source")
    original = fixture.spec.candidates[0]
    snapshot = replace(original, spec=CandidateSpec("dim_pca_vs_passthrough",
        original.spec.feature_pool, "svm", (), (("pca_components", .95),)),
        arms=("control", "challenger"), selector_arms=())
    spec = replace(fixture.spec, candidates=(snapshot,))
    admitted = preflight_full_campaign(fixture.root, FullCampaignIntent(
        spec, True, resource_opt_out_record="synthetic reviewed exact route test only"))
    capability = resolve_full_campaign_capability(fixture.root,
        experiment_ids=(snapshot.spec.experiment_id,))
    run = tmp_path / "run"
    completed = run_full_campaign(spec, fixture.root, run,
        data_preflight=admitted, capability=capability)
    assert completed["status"] == "finished_with_blockers"
    selected = next(row for row in completed["inventory"] if row["experiment_id"] == "dim_pca_vs_passthrough")
    assert selected["status"] == "completed"
    receipt = verify_ordinary_campaign_completion(fixture.root, run, admitted)
    assert receipt["status"] == "finished_with_blockers"
    assert any(exp["experiment_id"] == "dim_pca_vs_passthrough" and exp["status"] == "completed"
               for exp in receipt["experiments"])
    with FullFitJournal.open_readonly(run / "journal", spec, source_root=fixture.root) as journal:
        attempts = journal.attempts()
        assert len(attempts) == 24
        assert all(attempt.status == "completed" for attempt in attempts)
    declaration = json.loads((run / "dim_pca_vs_passthrough_declaration.json").read_text())
    assert {item["score_kind"] for item in declaration["candidates"].values()} == {"decision_margin"}
    validation = json.loads((run / "dim_pca_vs_passthrough_transport_validation.json").read_text())
    assert "requires_calibrated_probability" in json.dumps(validation)
    shard = run / "journal" / attempts[0].artifact_directory / "shard.parquet"
    with shard.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash|changed|mismatch"):
        verify_ordinary_campaign_completion(fixture.root, run, admitted)
