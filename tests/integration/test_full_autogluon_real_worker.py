"""AUTHORIZED real tiny fits, strictly sequential; never production/comparative data.

Run alone after parent grants the actual-fit lane. Other suites must not invoke
this test while fitting. Artifacts persist under /tmp/opencode for parent review.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from project.experiments import full_campaign_runtime as runtime
from project.experiments.candidate_contracts import CandidateSpec
from project.experiments.full_campaign_worker import (
    ParentLifecycleService,
    prepare_worker_request,
)
from project.experiments.full_run_contracts import (
    ArtifactEvidence,
    FullCandidateSnapshot,
    FullFitJournal,
    FullResourcePolicy,
    FullRunSpec,
    SourceBinding,
)
from project.experiments.full_snapshot_codec import encode_snapshot, semantic_digest
from project.features.registry import build_lead_baseline_registry
from project.models import full_autogluon_bridge as bridge
from project.models import full_autogluon_worker as backend_worker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENCODING = "lead_year_offset_1900_uint8_v1"
RECIPE = "model_autogluon_core_features"


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def _fixture_table(rows: int, rng: np.random.Generator, labels: np.ndarray | None = None) -> pa.Table:
    signal = rng.normal(4, 1, rows) if labels is None else 2 + 4 * labels + rng.normal(0, .8, rows)
    values = {
        "meter_reading": pa.array([None if i % 17 == 0 else float(value) for i, value in enumerate(signal)], type=pa.float64()),
        "primary_use": pa.array([None if i % 13 == 0 else ("Office", "Education", "Retail")[i % 3] for i in range(rows)]),
        "square_feet": pa.array([None if i % 19 == 0 else float(800 + i * 31) for i in range(rows)]),
        "year_built": pa.array([None if i % 23 == 0 else 90 + i % 25 for i in range(rows)], type=pa.int64()),
        "floor_count": pa.array([None if i % 29 == 0 else 1 + i % 8 for i in range(rows)], type=pa.int64()),
        "hour": pa.array(rng.integers(0, 24, rows)),
        "weekday": pa.array(rng.integers(0, 7, rows)),
        "month": pa.array(rng.integers(1, 13, rows)),
        "year": pa.array([2016] * rows, type=pa.int64()),
    }
    names = build_lead_baseline_registry().names
    table = pa.table({name: values[name] for name in names})
    if labels is not None:
        table = table.append_column("anomaly", pa.array(labels, type=pa.int64()))
    else:
        table = table.add_column(0, "row_index", pa.array(1001 + np.arange(rows) * 7, type=pa.int64()))
    return table


def _make_fixture(root: Path, environment: dict) -> tuple[FullRunSpec, Path, Path]:
    rng = np.random.default_rng(20261005)
    labels = np.repeat([0, 1], 64)
    rng.shuffle(labels)
    train = root / "synthetic-train.parquet"
    predict = root / "synthetic-predict.parquet"
    pq.write_table(_fixture_table(128, rng, labels), train, row_group_size=32)
    pq.write_table(_fixture_table(32, rng), predict, row_group_size=7)
    _json(root / "synthetic-splits.json", {"scope": "tiny transport smoke, not outer-CV proof",
          "units": [0, 1], "reserved_excluded_membership_only": ["synthetic-reserved"],
          "vault_features_or_labels": None})
    _json(root / "synthetic-matrix.json", {"experiment": RECIPE, "scope": "two tiny backend units only"})
    _json(root / "synthetic-notebook.ipynb", {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {}, "cells": [{"id": "synthetic-core-fixture", "cell_type": "markdown",
                                     "metadata": {}, "source": ["Synthetic-only backend smoke" ]}]})
    provenance = root / "docs/evidence/year_built_encoding_v1.json"
    _json(provenance, {"encoding_id": ENCODING, "scope": "synthetic raw construction-year codes only"})
    approval_relative = "docs/evidence/autogluon_full_profile_v1.json"
    shutil.copyfile(PROJECT_ROOT / approval_relative, root / approval_relative)
    shutil.copyfile(Path(__file__), root / "synthetic-producer.py")
    sources = tuple(SourceBinding(role, name, bridge._file_hash(root / name)) for role, name in (
        ("data", train.name), ("split", "synthetic-splits.json"), ("matrix", "synthetic-matrix.json"),
        ("producer", "synthetic-producer.py"), ("notebook", "synthetic-notebook.ipynb"),
        ("encoding", "docs/evidence/year_built_encoding_v1.json"), ("autogluon_profile", approval_relative)))
    snapshot = FullCandidateSnapshot(CandidateSpec(RECIPE, build_lead_baseline_registry().names,
                                    "autogluon", ()), ("challenger",), "synthetic-notebook.ipynb", "synthetic-core-fixture")
    versions = tuple(sorted((name, str(value)) for name, value in environment.items()))
    # One reserved, excluded membership declaration satisfies the transport
    # contract's positive vault count. No reserved-row features/labels exist.
    spec = FullRunSpec("autogluon-real-tiny-v1", (snapshot,), sources, versions,
                       FullResourcePolicy(True), development_rows=160, vault_rows=1)
    spec.verify_bindings(root)
    return spec, train, predict


def test_two_real_autogluon_units_are_supervised_and_sequential(monkeypatch):
    root = Path(tempfile.mkdtemp(prefix="autogluon-real-worker-v1-", dir="/tmp/opencode"))
    source = root / "synthetic-source"
    source.mkdir(mode=0o700)
    run_root = root / "run"
    run_root.mkdir(mode=0o700)
    interpreter = str(PROJECT_ROOT / ".venv-autogluon/bin/python")
    environment = bridge.inspect_environment(interpreter)
    bridge.validate_environment(environment)
    spec, train_path, predict_path = _make_fixture(source, environment)
    key = secrets.token_hex(32)
    reports = []
    predictions = []
    previous_cleanup = None
    monkeypatch.setenv("FULL_CAMPAIGN_WORKER_KEY", key)
    with runtime.CampaignLock(source_root=source, path_or_root=run_root), FullFitJournal(run_root, spec, source_root=source) as journal:
        for fold in (0, 1):
            unit = f"{RECIPE}/challenger/r0/f{fold}/model"
            attempt = journal.begin(unit)
            directory = run_root / attempt.artifact_directory
            directory.chmod(0o700)
            provenance = next(item for item in spec.sources if item.role == "encoding")
            payload = {"version": 1, "project_root": str(source), "spec": encode_snapshot(spec),
                "spec_sha256": spec.binding_sha256, "index_path": str(source / "synthetic-splits.json"),
                "index_sha256": bridge._file_hash(source / "synthetic-splits.json"),
                "experiment_id": RECIPE, "arm_id": "challenger", "repeat": 0, "fold": fold,
                "year_built_encoding": ENCODING, "provenance_path": provenance.path,
                "provenance_sha256": provenance.sha256, "semantic_sha256": semantic_digest(ENCODING, provenance.sha256),
                "work_id": unit, "attempt_id": attempt.attempt_id, "output_path": "shard.parquet",
                "parent_pid": os.getpid(), "ack_timeout_seconds": 60., "nonce": secrets.token_hex(32)}
            lifecycle = prepare_worker_request(directory, payload, key)
            shutil.copyfile(train_path, directory / "train.parquet")
            shutil.copyfile(predict_path, directory / "predict.parquet")
            binding = {"run_id": spec.run_id, "attempt_id": attempt.attempt_id, "config_hash": spec.binding_sha256,
                       "data_hash": bridge._file_hash(train_path), "split_hash": bridge._file_hash(source / "synthetic-splits.json")}
            request = bridge.prepare_request(directory, binding, environment)
            request["prediction_batch_rows"] = 7
            request = backend_worker.bind_worker_request(directory, request)
            envelope = {"payload": request, "signature": bridge.authenticate(request, key.encode())}
            backend_worker._publish(directory, "backend-request.json", envelope)
            acknowledged = []
            def debit(phase, event, attempt=attempt, unit=unit, directory=directory, acknowledged=acknowledged):
                records = [item for item in journal.attempts() if item.attempt_id == attempt.attempt_id]
                assert len(records) == 1 and records[0].status == "running"
                assert phase == "model" and event["unit"] == unit
                assert not (directory / "backend-models").exists()
                acknowledged.append({"unit": unit, "attempt_id": attempt.attempt_id,
                                     "journal_model_fits": journal.used_model_fits,
                                     "before_predictor_construction": True})
            service = ParentLifecycleService(directory, key, on_debit=debit)
            start = time.monotonic()
            assert previous_cleanup is None or start > previous_cleanup
            # Existing supervisor forwards sys.path. Scope launch inputs to trusted
            # repository source only, never Python3.13 stdlib/site-packages.
            with monkeypatch.context() as isolated_launch:
                isolated_launch.setattr(sys, "path", [str(PROJECT_ROOT / "src")])
                isolated_launch.setenv("PYTHONPATH", str(PROJECT_ROOT / "src"))
                outcome = runtime.run_work_unit(unit, "project.models.full_autogluon_worker:run", directory,
                    journal=journal, attempt_id=attempt.attempt_id, python_executable=interpreter,
                    expected_artifacts=("scores.parquet", "response.json", "backend-status.json", "backend-started.json"),
                    poll_callback=service.service_once, spec_binding=spec.binding_sha256,
                    attempt_binding={"request_hash": hashlib.sha256(bridge._canonical(request)).hexdigest()})
            assert outcome.exit_code == 0, (directory / "stderr.log").read_text()
            assert outcome.status == "awaiting_validation" and outcome.observation["cleanup_confirmed"] is True
            identity = json.loads((directory / "worker-identity.json").read_text())
            runtime.confirm_owned_attempt_exit(identity)
            previous_cleanup = time.monotonic()
            assert len(acknowledged) == 1
            response_envelope = json.loads((directory / "response.json").read_text())
            response = response_envelope["payload"]
            assert hmac.compare_digest(response_envelope["signature"], bridge.authenticate(response, key.encode()))
            bridge.verify_response(directory, request, response)
            scores = pq.read_table(directory / "scores.parquet")
            expected_ids = pq.read_table(predict_path, columns=["row_index"]).column(0).to_pylist()
            assert scores.column("row_index").to_pylist() == expected_ids
            values = scores.column("probability").to_numpy()
            assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
            backend = response["backend"]
            assert backend["models"] == ["ExtraTrees"] and backend["evidence_kind"] == "runtime_instrumented"
            assert backend["learner_random_state"] == 42
            assert backend["effective_model_params"]["random_state"] == 42
            assert backend["prediction_batches"] == 5
            calls = backend["forest_fit_calls"]
            assert [item["actual_tree_count"] for item in calls] in ([64], [4, 64])
            assert all(item["random_state"] == 42 and item["outcome"] == "success" for item in calls)
            for name in ("trainer_attempts", "abstract_fit_attempts"):
                records = backend["model_fit_attempts"][name]
                assert len(records) == 1 and records[0]["outcome"] == "success"
                assert not records[0]["rejected_attempts"]
            assert response["score_semantics"] == bridge.SCORE_SEMANTICS
            assert json.loads((directory / "backend-status.json").read_text())["status"] == "awaiting_parent_validation"
            sealed_names = ("scores.parquet", "response.json", "backend-status.json", "backend-started.json",
                            "backend-request.json", "worker-request.json", "event-0001.json", "ack-0001.json",
                            "stdout.log", "stderr.log", "worker-identity.json")
            evidence = tuple(ArtifactEvidence(str((directory / name).relative_to(run_root)), bridge._file_hash(directory / name))
                             for name in sealed_names)
            journal.seal(attempt.attempt_id, evidence)
            journal.finish(attempt.attempt_id, "completed")
            reports.append({"unit": unit, "attempt_id": attempt.attempt_id, "directory": str(directory),
                            "started_monotonic": start, "cleanup_confirmed_monotonic": previous_cleanup,
                            "supervisor": asdict(outcome), "parent_debit": acknowledged,
                            "backend": backend, "signed_response_sha256": bridge._file_hash(directory / "response.json"),
                            "sealed_artifacts": [asdict(item) for item in evidence],
                            "nonce": lifecycle["nonce"]})
            predictions.append(values)
        assert journal.used_model_fits == 2 and len(journal.verify_completed()) == 2
        assert np.allclose(predictions[0], predictions[1], rtol=1e-13, atol=1e-13)
        attempts = [asdict(item) for item in journal.attempts()]
    with sqlite3.connect(run_root / "full_fit.sqlite3") as connection:
        head, count = connection.execute("SELECT head,count FROM binding").fetchone()
    report = {"version": 1, "record_kind": "real_tiny_synthetic_backend_execution",
              "comparative_full_completion": False, "production_data_accessed": False,
              "full_campaign_promoted": False, "environment": environment,
              "fixture_train_rows_per_unit": 128, "fixture_train_class_counts": {"0": 64, "1": 64},
              "fixture_predict_rows_per_unit": 32, "prediction_batch_rows": 7,
              "source_search_path": [str(PROJECT_ROOT / "src")],
              "source_root": str(source), "run_root": str(run_root), "spec_sha256": spec.binding_sha256,
              "sources": [asdict(item) for item in spec.sources], "units": reports,
              "parent_journal": {"path": str(run_root / "full_fit.sqlite3"),
                                 "sha256": bridge._file_hash(run_root / "full_fit.sqlite3"), "head": head,
                                 "event_count": count, "model_fit_count": 2, "attempts": attempts},
              "scope_note": "Two tiny unit completions only. Reserved excluded membership is metadata only; no vault features/labels exist or were accessed."}
    _json(root / "real-smoke-report.json", report)
    print(f"REAL_SMOKE_REPORT={root / 'real-smoke-report.json'}")
