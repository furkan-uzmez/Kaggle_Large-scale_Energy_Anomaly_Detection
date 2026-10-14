"""Frozen campaign transport contracts; real smoke is explicitly parent-owned."""
from __future__ import annotations

import importlib
import os
import secrets
import shutil
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from project.experiments.candidate_contracts import CandidateSpec
from project.experiments.candidate_dispatch import _configuration
from project.experiments.full_campaign_preflight import current_environment
from project.experiments.full_campaign_worker import ParentLifecycleService, prepare_worker_request
from project.experiments.full_input_schema import CORE
from project.experiments.full_run_contracts import FullCandidateSnapshot, SourceBinding, _file_hash
from project.experiments.full_snapshot_codec import PROVENANCE_PATH, encode_snapshot, semantic_digest, configuration_digest
from project.features.input_encoding import LEAD_YEAR_OFFSET_1900_UINT8_V1 as ENCODING

RECIPE = "model_autogluon_core_features"
ROOT = Path(__file__).resolve().parents[2]


def module(name):
    qualified = f"project.experiments.full_autogluon_{name}"
    return importlib.import_module(qualified)


def snapshot():
    return FullCandidateSnapshot(CandidateSpec(RECIPE, CORE, "autogluon", ()),
                                 ("challenger",), "notebooks/source.ipynb", "cell")


def test_frozen_arm_declares_stable_full_profile_not_bounded_lease():
    contract = module("contract")
    arms = contract.build_reviewed_autogluon_arms(snapshot(), year_built_encoding=ENCODING)
    assert tuple(arms) == ("challenger",)
    arm = arms["challenger"]
    assert arm.raw_pool == CORE and arm.score_kind == "positive_probability"
    assert not hasattr(arm.transformer_or_pipeline, "fit")
    first = configuration_digest(_configuration(arm.transformer_or_pipeline))
    again = contract.build_reviewed_autogluon_arms(snapshot(), year_built_encoding=ENCODING)
    assert first == configuration_digest(_configuration(again["challenger"].transformer_or_pipeline))
    assert len(snapshot().planned_units) == 12


@pytest.mark.parametrize("fault", ["params", "methods", "core", "arms", "encoding"])
def test_rejects_unreviewed_recipe(fault):
    contract = module("contract")
    value = snapshot()
    encoding = ENCODING
    if fault == "params":
        value = replace(value, spec=replace(value.spec, model_params=(("n_estimators", 4),)))
    elif fault == "methods":
        value = replace(value, spec=replace(value.spec, method_params=(("limit", 1),)))
    elif fault == "core":
        value = replace(value, spec=replace(value.spec, feature_pool=CORE[::-1]))
    elif fault == "arms":
        value = replace(value, arms=("control", "challenger"))
    else:
        encoding = "unknown"
    with pytest.raises(ValueError):
        contract.build_reviewed_autogluon_arms(value, year_built_encoding=encoding)


def campaign_fixture(tmp_path):
    from full_campaign_fixtures import write_fixture
    from project.experiments import full_candidate_io as io
    fixture = write_fixture(tmp_path / "source")
    import polars as pl
    fixture.full_frame.with_columns((pl.col("year_built") - 1900).cast(pl.UInt8)).write_parquet(fixture.data_path)
    data_source = next(item for item in fixture.spec.sources if item.role == "data")
    sources = tuple(replace(item, sha256=_file_hash(fixture.data_path)) if item == data_source else item
                    for item in fixture.spec.sources)
    profile = "docs/evidence/autogluon_full_profile_v1.json"
    (fixture.root / profile).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / profile, fixture.root / profile)
    (fixture.root / PROVENANCE_PATH).write_text('{"encoding_id":"' + ENCODING + '"}')
    candidate = replace(snapshot(), source_notebook=fixture.spec.candidates[0].source_notebook,
                        source_cell_id=fixture.spec.candidates[0].source_cell_id)
    extra = tuple(SourceBinding(role, name, _file_hash(fixture.root / name)) for role, name in
                  (("encoding", PROVENANCE_PATH), ("autogluon_profile", profile)))
    spec = replace(fixture.spec, candidates=(candidate,), sources=sources + extra,
                   environment=current_environment(), development_rows=52, vault_rows=4)
    run_root = tmp_path / "run"
    run_root.mkdir(mode=0o700)
    index = io.build_development_index(fixture.root, spec, run_root / "index.sqlite3")
    directory = run_root / "attempts" / secrets.token_hex(16)
    provenance = extra[0]
    payload = {"version": 1, "project_root": str(fixture.root), "spec": encode_snapshot(spec),
        "spec_sha256": spec.binding_sha256, "index_path": str(index), "index_sha256": _file_hash(index),
        "experiment_id": RECIPE, "arm_id": "challenger", "repeat": 0, "fold": 0,
        "year_built_encoding": ENCODING, "provenance_path": PROVENANCE_PATH,
        "provenance_sha256": provenance.sha256, "semantic_sha256": semantic_digest(ENCODING, provenance.sha256),
        "work_id": f"{RECIPE}/challenger/r0/f0/model", "attempt_id": directory.name,
        "output_path": "shard.parquet", "parent_pid": os.getpid(), "ack_timeout_seconds": 60.,
        "nonce": secrets.token_hex(32)}
    return directory, payload, spec


def test_real_tiny_outer_worker_preserves_backend_evidence_and_rejects_score_substitution(tmp_path):
    """Run explicitly alone by parent: one real XT fit, never production data."""
    worker = module("svc_worker")
    from project.experiments.full_campaign_runtime import run_work_unit
    directory, payload, spec = campaign_fixture(tmp_path)
    key = secrets.token_hex(32)
    payload = prepare_worker_request(directory, payload, key)
    directory.chmod(0o700)
    debits = []
    def debit(phase, event):
        assert phase == "model" and not (directory / "backend" / "backend-models").exists()
        debits.append(event)
    service = ParentLifecycleService(directory, key, on_debit=debit)
    outcome = run_work_unit(payload["work_id"], worker.BACKEND_TARGET, directory,
        expected_artifacts=("shard.parquet",), poll_callback=service.service_once,
        environment_binding={"FULL_CAMPAIGN_WORKER_KEY": key},
        attempt_binding={"attempt_id": directory.name, "nonce": payload["nonce"], "spec_sha256": spec.binding_sha256})
    assert outcome.exit_code == 0, (directory / "stderr.log").read_text()
    assert outcome.observation["cleanup_confirmed"] is True
    assert len(debits) == 1 and service.shard_evidence is not None
    proof = worker.verify_autogluon_unit(directory, key, expected_request=payload)
    assert proof["backend"]["models"] == ["ExtraTrees"]
    assert proof["backend"]["evidence_kind"] == "runtime_instrumented"
    assert proof["num_rows"] == 7
    assert proof == worker.verify_autogluon_artifacts(directory, expected_request=payload)
    scores = pq.read_table(directory / "backend" / "scores.parquet")
    shard = pq.read_table(directory / "shard.parquet")
    assert shard["prediction"].to_pylist() == scores["probability"].to_pylist()
    assert shard["row_index"].to_pylist() == scores["row_index"].to_pylist()
    assert set(shard["score_kind"].to_pylist()) == {"positive_probability"}
    import pyarrow as pa
    substituted = shard.set_column(shard.schema.get_field_index("prediction"), "prediction",
                                   pa.array([.123456789] * shard.num_rows))
    pq.write_table(substituted, directory / "shard.parquet")
    with pytest.raises(ValueError, match="score|canonical|mapping"):
        worker.verify_autogluon_unit(directory, key, expected_request=payload)


@pytest.mark.parametrize("fault", ["request_signature", "index_hash", "approval_hash", "parent_identity"])
def test_ordinary_authentication_tampering_never_debits_or_constructs_backend(tmp_path, fault):
    worker = module("svc_worker")
    directory, payload, _ = campaign_fixture(tmp_path)
    payload["parent_pid"] = os.getppid()
    key = secrets.token_hex(32)
    payload = prepare_worker_request(directory, payload, key)
    if fault == "request_signature":
        (directory / "worker-request.json").write_text('{"payload":{},"signature":"bad"}')
    elif fault == "index_hash":
        with Path(payload["index_path"]).open("ab") as stream:
            stream.write(b"tamper")
    elif fault == "approval_hash":
        with (Path(payload["project_root"]) / "docs/evidence/autogluon_full_profile_v1.json").open("ab") as stream:
            stream.write(b"tamper")
    else:
        from project.experiments.full_campaign_worker import _publish_message
        from project.models.full_autogluon_bridge import authenticate
        payload["parent_pid"] = -1
        (directory / "worker-request.json").unlink()
        _publish_message(directory / "worker-request.json",
                         {"payload": payload, "signature": authenticate(payload, key.encode())})
    with pytest.raises(ValueError):
        worker.execute_autogluon_request(directory, key, exchange=lambda _: pytest.fail("debit forbidden"))
    assert not (directory / "backend").exists()
    assert not (directory / "shard.parquet").exists()
