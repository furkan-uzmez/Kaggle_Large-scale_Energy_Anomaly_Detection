"""Launch permission must not be inferred from filesystem or duck typing."""

from dataclasses import replace

import numpy as np
import pytest

from full_campaign_fixtures import write_fixture
from project.experiments import full_campaign_worker as worker
from project.experiments import full_candidate_runner as runner
from project.experiments.full_campaign_capabilities import (
    FullCampaignCapability,
    resolve_full_campaign_capability,
    verify_full_campaign_capability,
)
from project.experiments.full_candidate_runner import run_full_campaign
from project.experiments.full_run_contracts import SourceBinding, _file_hash
from project.models.candidate_artifacts import CANDIDATE_OOF_SCHEMA


def test_module_presence_never_grants_launch(tmp_path):
    for name in ("pyproject.toml", ".git", "src/project/experiments/full_campaign_worker.py",
                 "src/project/experiments/full_campaign_runtime.py",
                 "src/project/experiments/full_candidate_runner.py",
                 "src/project/experiments/full_campaign_evidence.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert not resolve_full_campaign_capability(tmp_path).available
    with pytest.raises(ValueError):
        verify_full_campaign_capability(
            tmp_path, FullCampaignCapability(str(tmp_path), True, ())
        )


def test_fake_data_evidence_rejected_without_fallback_or_writes(tmp_path):
    fixture = write_fixture(tmp_path / "fixture")
    destination = tmp_path / "run"
    with pytest.raises(TypeError, match="AdmittedFullRun"):
        run_full_campaign(fixture.spec, fixture.root, destination,
                          data_preflight=object())
    assert not destination.exists()


def test_encoding_requires_authoritative_spec_bound_provenance(tmp_path):
    fixture = write_fixture(tmp_path / "fixture")
    # An invented root provenance file is not an encoding authority.
    (fixture.root / "provenance.json").write_text('{"encoding_id":"absolute_calendar_year"}')
    with pytest.raises(ValueError, match="encoding provenance"):
        runner._encoding_provenance(fixture.root, fixture.spec)
    path = fixture.root / "docs/evidence/year_built_encoding_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"encoding_id":"lead_year_offset_1900_uint8_v1"}')
    binding = SourceBinding("encoding", str(path.relative_to(fixture.root)), _file_hash(path))
    spec = replace(fixture.spec, sources=fixture.spec.sources + (binding,))
    encoding, relative, digest = runner._encoding_provenance(fixture.root, spec)
    assert encoding == "lead_year_offset_1900_uint8_v1"
    assert relative == binding.path
    assert digest == binding.sha256
    path.write_text('{}')
    with pytest.raises(ValueError):
        runner._encoding_provenance(fixture.root, spec)


def test_worker_writes_batches_before_requesting_next_batch(tmp_path):
    output = tmp_path / "shard.parquet"
    fixture = write_fixture(tmp_path / "fixture")
    arm = type("Arm", (), {"arm_id": "challenger", "score_kind": "positive_probability"})()

    def table(row):
        batch = {"row_index": [row], "building_id": ["b0"],
                 "timestamp": np.array([1_700_000_000_000_000_000 + row]),
                 "target": [row], "split_seed": 42}
        return runner._batch_table(batch, np.array([0.2]), fixture.spec.candidates[0].spec,
                                   arm, fixture.spec, 0, 0, "d" * 64)

    def batches():
        yield table(0)
        assert output.exists(), "worker accumulated batches instead of writing them"
        yield table(1)

    digest, rows = worker._write_oof_shard(output, batches(), CANDIDATE_OOF_SCHEMA)
    assert rows == 2
    assert digest == _file_hash(output)
