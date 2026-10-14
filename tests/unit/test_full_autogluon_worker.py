"""Recording-fake backend tests, never actual AutoGluon fitting."""
import importlib
import json
import os
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from project.experiments.candidate_contracts import CandidateSpec
from project.experiments.full_run_contracts import (
    FullCandidateSnapshot,
    FullResourcePolicy,
    FullRunSpec,
    SourceBinding,
)
from project.experiments.full_snapshot_codec import encode_snapshot, semantic_digest
from project.models import full_autogluon_bridge as bridge


def worker():
    try:
        return importlib.import_module("project.models.full_autogluon_worker")
    except ModuleNotFoundError:
        pytest.fail("authenticated AutoGluon worker is not implemented")


def fixture_request(tmp_path, monkeypatch):
    root = tmp_path / "attempt"
    root.mkdir(mode=0o700)
    pq.write_table(pa.table({"meter_reading": [1., 2.], "anomaly": [0, 1]}), root / "train.parquet")
    pq.write_table(pa.table({"row_index": [10, 11], "meter_reading": [3., 4.]}), root / "predict.parquet")
    interpreter = str(Path(__file__).resolve().parents[2] / ".venv-autogluon/bin/python")
    environment = bridge.inspect_environment(interpreter)
    sources = tuple(SourceBinding(role, role + ".json", digest * 64) for role, digest in
                    (("matrix", "a"), ("data", "b"), ("split", "c"), ("producer", "d"), ("notebook", "e")))
    encoding_hash = "f" * 64
    sources += (SourceBinding("encoding", "docs/evidence/year_built_encoding_v1.json", encoding_hash),)
    approval_path = Path(__file__).resolve().parents[2] / "docs/evidence/autogluon_full_profile_v1.json"
    sources += (SourceBinding("autogluon_profile", "docs/evidence/autogluon_full_profile_v1.json",
                              bridge._file_hash(approval_path)),)
    snapshot = FullCandidateSnapshot(CandidateSpec("model_autogluon_core_features", ("meter_reading",), "autogluon", ()),
                                     ("challenger",), "notebook.json", "fixture-cell")
    spec = FullRunSpec("run", (snapshot,), sources, (("fixture", "recording_fake"),), FullResourcePolicy(True))
    lifecycle = {"work_id": "model_autogluon_core_features/challenger/r0/f0/model",
                 "attempt_id": "attempt", "nonce": "d" * 64, "parent_pid": os.getppid(),
                 "ack_timeout_seconds": 1., "spec_sha256": spec.binding_sha256,
                 "experiment_id": "model_autogluon_core_features", "spec": encode_snapshot(spec),
                 "version": 1, "project_root": str(tmp_path), "index_path": str(root / "index.sqlite3"),
                 "index_sha256": "a" * 64, "arm_id": "challenger", "repeat": 0, "fold": 0,
                 "year_built_encoding": "lead_year_offset_1900_uint8_v1",
                 "provenance_path": "docs/evidence/year_built_encoding_v1.json", "provenance_sha256": encoding_hash,
                 "semantic_sha256": semantic_digest("lead_year_offset_1900_uint8_v1", encoding_hash),
                 "output_path": "shard.parquet"}
    key = b"fixture-secret"
    envelope = {"payload": lifecycle, "signature": bridge.authenticate(lifecycle, key)}
    (root / "worker-request.json").write_bytes(bridge._canonical(envelope))
    binding = {"run_id": "run", "attempt_id": "attempt", "config_hash": spec.binding_sha256,
               "data_hash": "b" * 64, "split_hash": "c" * 64}
    request = bridge.prepare_request(root, binding, environment)
    module = worker()
    request = module.bind_worker_request(root, request)
    monkeypatch.setattr(module, "_live_environment", lambda: environment)
    return module, root, request, key, lifecycle


class RecordingBackend:
    events: ClassVar[list] = []
    failure = None
    inventory: ClassVar[list[str]] = ["ExtraTrees"]
    classes_: ClassVar[list[int]] = [0, 1]
    positive_class = 1

    def __init__(self, **kwargs):
        self.events.append(("construct", kwargs))
        self._learner = SimpleNamespace(random_state=kwargs.get("learner_kwargs", {}).get("random_state", 0))
        if self.failure == "construct":
            raise RuntimeError("construction fixture failure")

    def _validate_fit_kwargs(self, kwargs):
        return kwargs

    def fit(self, *, train_data, **kwargs):
        self.events.append(("fit", kwargs, train_data.copy()))
        if self.failure == "fit":
            raise RuntimeError("fit fixture failure")
        return self

    def model_names(self):
        return self.inventory

    def info(self):
        return {"model_info": {"ExtraTrees": {"model_type": "XTModel",
                "hyperparameters": {"n_estimators": 64, "max_depth": 12, "min_samples_leaf": 2, "random_state": 42},
                "hyperparameters_fit": {"n_estimators": 64}}}}

    def predict_proba(self, *, data, as_pandas, as_multiclass):
        self.events.append(("predict", list(data.columns)))
        return pd.DataFrame({0: [.75] * len(data), 1: [.25] * len(data)}, index=data.index)


@pytest.fixture(autouse=True)
def reset_backend():
    RecordingBackend.events = []
    RecordingBackend.failure = None
    RecordingBackend.inventory = ["ExtraTrees"]


def authorizer(key, lifecycle, events):
    def exchange(event):
        events.append(event["event"])
        assert event["nonce"] == lifecycle["nonce"]
        payload = dict(event, status="authorized")
        return {"payload": payload, "signature": bridge.authenticate(payload, key)}
    return exchange


def test_debit_precedes_constructor_and_exact_profile_streams_scores(tmp_path, monkeypatch):
    module, root, request, key, lifecycle = fixture_request(tmp_path, monkeypatch)
    events = []
    def exchange(event):
        assert not RecordingBackend.events
        return authorizer(key, lifecycle, events)(event)
    response = module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                                  exchange, backend_type=RecordingBackend)
    assert events == ["debit"]
    assert [item[0] for item in RecordingBackend.events] == ["construct", "fit", "predict"]
    params = RecordingBackend.events[1][1]
    expected = bridge.full_xt_profile()
    expected["memory_limit"] = "auto"
    assert params == expected
    assert params["refit_full"] is False and params["feature_prune_kwargs"] is None
    assert params["raise_on_model_failure"] is True
    assert RecordingBackend.events[1][2].columns.tolist() == ["meter_reading", "anomaly"]
    assert RecordingBackend.events[2][1] == ["meter_reading"]
    assert pq.read_table(root / "scores.parquet").to_pydict() == {"row_index": [10, 11], "probability": [.25, .25]}
    assert response["backend"]["models"] == ["ExtraTrees"]
    assert json.loads((root / "backend-status.json").read_text())["status"] == "awaiting_parent_validation"


def test_forged_ack_never_constructs(tmp_path, monkeypatch):
    module, root, request, key, _ = fixture_request(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="ack"):
        module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                           lambda event: {"payload": dict(event, status="authorized"), "signature": "bad"},
                           backend_type=RecordingBackend)
    assert not RecordingBackend.events
    assert not (root / "scores.parquet").exists()


@pytest.mark.parametrize("phase", ["construct", "fit"])
def test_failures_remain_failed_no_response(tmp_path, monkeypatch, phase):
    module, root, request, key, lifecycle = fixture_request(tmp_path, monkeypatch)
    RecordingBackend.failure = phase
    with pytest.raises(RuntimeError):
        module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                           authorizer(key, lifecycle, []), backend_type=RecordingBackend)
    assert json.loads((root / "backend-status.json").read_text())["status"] == "failed"
    assert not (root / "response.json").exists()


def test_extra_inventory_rejected(tmp_path, monkeypatch):
    module, root, request, key, lifecycle = fixture_request(tmp_path, monkeypatch)
    RecordingBackend.inventory = ["ExtraTrees", "WeightedEnsemble"]
    with pytest.raises(ValueError, match="inventory"):
        module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                           authorizer(key, lifecycle, []), backend_type=RecordingBackend)
    assert not (root / "scores.parquet").exists()


def test_cancel_does_not_claim_success(tmp_path, monkeypatch):
    module, root, request, key, _ = fixture_request(tmp_path, monkeypatch)
    def cancel(_event):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                           cancel, backend_type=RecordingBackend)
    assert json.loads((root / "backend-status.json").read_text())["status"] == "cancelled"
    assert not RecordingBackend.events


def test_inherited_autogluon_memory_cap_is_rejected_before_debit(tmp_path, monkeypatch):
    module, root, request, key, lifecycle = fixture_request(tmp_path, monkeypatch)
    monkeypatch.setenv("AG_MEMORY_LIMIT_IN_GB", "2")
    events = []
    with pytest.raises(ValueError, match="memory"):
        module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                           authorizer(key, lifecycle, events), backend_type=RecordingBackend)
    assert not events and not RecordingBackend.events


def test_live_source_change_fails_before_constructor(tmp_path, monkeypatch):
    module, root, request, key, lifecycle = fixture_request(tmp_path, monkeypatch)
    request["backend_binding"]["worker_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="hash"):
        module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                           authorizer(key, lifecycle, []), backend_type=RecordingBackend)
    assert not RecordingBackend.events


def test_reduced_forest_rejected(tmp_path, monkeypatch):
    module, root, request, key, lifecycle = fixture_request(tmp_path, monkeypatch)
    class ReducedBackend(RecordingBackend):
        def info(self):
            info = super().info()
            info["model_info"]["ExtraTrees"]["hyperparameters_fit"]["n_estimators"] = 40
            return info
    with pytest.raises(ValueError, match="reduced"):
        module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                           authorizer(key, lifecycle, []), backend_type=ReducedBackend)
    assert not (root / "scores.parquet").exists()


def test_transport_binding_must_match_authenticated_snapshot(tmp_path, monkeypatch):
    module, root, request, key, lifecycle = fixture_request(tmp_path, monkeypatch)
    request["binding"]["config_hash"] = "a" * 64
    with pytest.raises(ValueError, match="snapshot"):
        module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                           authorizer(key, lifecycle, []), backend_type=RecordingBackend)
    assert not RecordingBackend.events


def test_prediction_tamper_during_fit_blocks_response(tmp_path, monkeypatch):
    module, root, request, key, lifecycle = fixture_request(tmp_path, monkeypatch)
    class ChangingBackend(RecordingBackend):
        def fit(self, **kwargs):
            super().fit(**kwargs)
            pq.write_table(pa.table({"row_index": [10, 11], "meter_reading": [30., 40.]}), root / "predict.parquet")
            return self
    with pytest.raises(ValueError, match="hash"):
        module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                           authorizer(key, lifecycle, []), backend_type=ChangingBackend)
    assert not (root / "response.json").exists()


def test_existing_parent_lifecycle_service_authorizes_fake_backend(tmp_path, monkeypatch):
    from project.experiments.full_campaign_worker import ParentLifecycleService
    module, root, request, key, lifecycle = fixture_request(tmp_path, monkeypatch)
    debits = []
    service = ParentLifecycleService(root, key.decode(), on_debit=lambda phase, event: debits.append((phase, event)))
    failures = []
    def service_debit():
        try:
            deadline = time.monotonic() + 2
            while not debits and time.monotonic() < deadline:
                service.service_once()
                time.sleep(.005)
        except (OSError, ValueError, TypeError) as error:
            failures.append(error)
    thread = threading.Thread(target=service_debit)
    thread.start()
    try:
        response = module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                                      module._file_exchange(root, lifecycle, key), backend_type=RecordingBackend)
    finally:
        thread.join(timeout=3)
    assert not failures and not thread.is_alive()
    assert len(debits) == 1 and debits[0][0] == "model"
    assert response["backend"]["evidence_kind"] == "recording_fake"


def test_secondary_entrypoint_and_installed_memory_validator_without_fit():
    module = worker()
    interpreter = str(Path(__file__).resolve().parents[2] / ".venv-autogluon/bin/python")
    result = subprocess.run([interpreter, "-I", str(Path(module.__file__).resolve()), "--help"],
                            check=True, capture_output=True, text=True)
    assert "--attempt-directory" in result.stdout
    code = ("from autogluon.tabular import TabularPredictor as P; "
            "P._validate_and_set_memory_limit(None, 'auto'); "
            "exec('try:\\n P._validate_and_set_memory_limit(None, None)\\nexcept ValueError:\\n print(\"None rejected; auto accepted\")')")
    result = subprocess.run([interpreter, "-I", "-c", code], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "None rejected; auto accepted"


@pytest.mark.parametrize("kind", ["boolean", "object", "reversed_index"])
def test_invalid_backend_probabilities_fail_closed(tmp_path, monkeypatch, kind):
    module, root, request, key, lifecycle = fixture_request(tmp_path, monkeypatch)
    class InvalidBackend(RecordingBackend):
        def predict_proba(self, **kwargs):
            if kind == "boolean":
                return pd.DataFrame({False: [False, False], True: [True, True]})
            if kind == "object":
                return pd.DataFrame({0: ["0.75", "0.75"], 1: ["0.25", "0.25"]})
            return pd.DataFrame({0: [.75, .75], 1: [.25, .25]}, index=[1, 0])
    with pytest.raises(ValueError, match="probability"):
        module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                           authorizer(key, lifecycle, []), backend_type=InvalidBackend)
    assert not (root / "response.json").exists()


def test_forest_audit_records_growth_on_one_object_no_actual_estimator_fit(monkeypatch):
    from sklearn.ensemble import ExtraTreesClassifier
    module = worker()
    monkeypatch.setattr(ExtraTreesClassifier, "fit", lambda model, *args, **kwargs: model)
    model = SimpleNamespace(n_estimators=4, warm_start=True, n_jobs=-1,
                            random_state=42, max_leaf_nodes=15000, estimators_=[object()] * 4,
                            bootstrap=True, criterion="gini", oob_score=False,
                            max_depth=12, min_samples_leaf=2)
    with module._forest_audit() as calls:
        ExtraTreesClassifier.fit(model, [], [])
        model.n_estimators = 64
        model.estimators_ = [object()] * 64
        ExtraTreesClassifier.fit(model, [], [])
    assert [item["n_estimators"] for item in calls] == [4, 64]
    assert calls[0]["n_jobs"] == -1


def test_forest_audit_rejects_concurrent_recording_calls_without_real_fits(monkeypatch):
    from sklearn.ensemble import ExtraTreesClassifier
    module = worker()
    entered = threading.Event()
    release = threading.Event()
    def recording_fit(model, *args, **kwargs):
        entered.set()
        assert release.wait(2)
        return model
    monkeypatch.setattr(ExtraTreesClassifier, "fit", recording_fit)
    model = SimpleNamespace(n_estimators=64, warm_start=False, n_jobs=-1,
                            random_state=42, max_leaf_nodes=15000, estimators_=[object()] * 64,
                            bootstrap=True, criterion="gini", oob_score=False,
                            max_depth=12, min_samples_leaf=2)
    with pytest.raises(ValueError, match="audit"), module._forest_audit() as calls:
        thread = threading.Thread(target=lambda: ExtraTreesClassifier.fit(model, [], []))
        thread.start()
        try:
            assert entered.wait(2)
            with pytest.raises(ValueError, match="concurrent"):
                ExtraTreesClassifier.fit(model, [], [])
        finally:
            release.set()
            thread.join(timeout=3)
        assert not thread.is_alive()
    assert len(calls) == 1


def test_seeds_and_approval_are_bound_and_learner_is_seeded(tmp_path, monkeypatch):
    module, root, request, key, lifecycle = fixture_request(tmp_path, monkeypatch)
    module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                       authorizer(key, lifecycle, []), backend_type=RecordingBackend)
    constructor = RecordingBackend.events[0][1]
    assert constructor["learner_kwargs"] == {"random_state": 42}
    assert request["backend_binding"]["resolved_profile"]["constructor"]["learner_kwargs"] == {"random_state": 42}
    assert request["backend_binding"]["resolved_profile"]["fit"]["hyperparameters"]["XT"]["random_state"] == 42
    approval_path = Path(__file__).resolve().parents[2] / "docs/evidence/autogluon_full_profile_v1.json"
    assert request["backend_binding"]["approval_sha256"] == bridge._file_hash(approval_path)
    approval = json.loads(approval_path.read_text())
    assert approval["record_kind"] == "approval_not_execution_proof"
    assert approval["native_memory_protection"] == "keep"


@pytest.mark.parametrize("kind", ["learner", "forest", "positive_float"])
def test_seed_or_positive_class_drift_rejected(tmp_path, monkeypatch, kind):
    module, root, request, key, lifecycle = fixture_request(tmp_path, monkeypatch)
    class DriftingBackend(RecordingBackend):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if kind == "learner":
                self._learner.random_state = 0
            if kind == "positive_float":
                self.positive_class = 1.0

        def info(self):
            info = super().info()
            if kind == "forest":
                info["model_info"]["ExtraTrees"]["hyperparameters"]["random_state"] = 0
            return info
    with pytest.raises(ValueError):
        module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                           authorizer(key, lifecycle, []), backend_type=DriftingBackend)
    assert not (root / "response.json").exists()


def test_method_params_rejected_before_debit(tmp_path, monkeypatch):
    from project.experiments.full_snapshot_codec import decode_snapshot
    module, root, request, key, lifecycle = fixture_request(tmp_path, monkeypatch)
    spec = decode_snapshot(lifecycle["spec"])
    candidate = replace(spec.candidates[0].spec, method_params=(("ignored", True),))
    spec = replace(spec, candidates=(replace(spec.candidates[0], spec=candidate),))
    lifecycle.update(spec=encode_snapshot(spec), spec_sha256=spec.binding_sha256)
    request["binding"]["config_hash"] = spec.binding_sha256
    (root / "worker-request.json").write_bytes(bridge._canonical({"payload": lifecycle, "signature": bridge.authenticate(lifecycle, key)}))
    request["backend_binding"]["lifecycle_sha256"] = bridge._file_hash(root / "worker-request.json")
    events = []
    with pytest.raises(ValueError):
        module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                           authorizer(key, lifecycle, events), backend_type=RecordingBackend)
    assert not events and not RecordingBackend.events


def test_model_entrypoint_rejects_hidden_extra_and_refit_with_recording_fake():
    module = worker()
    calls = []
    class XT:
        pass
    class Trainer:
        def _train_single(self, X, y, model, **kwargs):
            calls.append(model)
            return model
    first = XT()
    with pytest.raises(ValueError, match="attempt"), module._model_audit(Trainer, XT) as records:
        Trainer()._train_single(None, None, first)
        with pytest.raises(ValueError, match="attempt"):
            Trainer()._train_single(None, None, first)
        with pytest.raises(ValueError, match="attempt"):
            Trainer()._train_single(None, None, XT())
    assert calls == [first]
    assert len(records) == 1


def test_failed_hidden_model_attempt_remains_counted_before_retry():
    module = worker()
    class XT:
        pass
    class Trainer:
        def _train_single(self, X, y, model, **kwargs):
            raise RuntimeError("nonsurviving fixture model")
    with pytest.raises(ValueError, match="attempt"), module._model_audit(Trainer, XT) as records:
        with pytest.raises(RuntimeError):
            Trainer()._train_single(None, None, XT())
        with pytest.raises(ValueError, match="attempt"):
            Trainer()._train_single(None, None, XT())
    assert records[0]["outcome"] == "failed"


def test_forest_actual_tree_count_and_seed_verified_without_real_fits(monkeypatch):
    from sklearn.ensemble import ExtraTreesClassifier
    module = worker()
    monkeypatch.setattr(ExtraTreesClassifier, "fit", lambda model, *args, **kwargs: model)
    for seed, tree_count in ((0, 64), (42, 63)):
        model = SimpleNamespace(n_estimators=64, warm_start=False, n_jobs=-1,
                                random_state=seed, max_leaf_nodes=15000,
                                estimators_=[object()] * tree_count, bootstrap=True,
                                criterion="gini", oob_score=False, max_depth=12, min_samples_leaf=2)
        with pytest.raises(ValueError), module._forest_audit():
            ExtraTreesClassifier.fit(model, [], [])


def test_different_forest_rejection_is_latched_even_if_backend_catches_it(monkeypatch):
    from sklearn.ensemble import ExtraTreesClassifier
    module = worker()
    monkeypatch.setattr(ExtraTreesClassifier, "fit", lambda model, *args, **kwargs: model)
    params = {"n_estimators": 64, "warm_start": False, "n_jobs": -1, "random_state": 42,
              "max_leaf_nodes": 15000, "estimators_": [object()] * 64,
              "bootstrap": True, "criterion": "gini", "oob_score": False, "max_depth": 12, "min_samples_leaf": 2}
    with pytest.raises(ValueError, match="audit"), module._forest_audit():
        ExtraTreesClassifier.fit(SimpleNamespace(**params), [], [])
        with pytest.raises(ValueError, match="extra"):
            ExtraTreesClassifier.fit(SimpleNamespace(**params), [], [])


def test_approval_must_be_in_configuration_snapshot_before_debit(tmp_path, monkeypatch):
    from project.experiments.full_snapshot_codec import decode_snapshot
    module, root, request, key, lifecycle = fixture_request(tmp_path, monkeypatch)
    spec = decode_snapshot(lifecycle["spec"])
    spec = replace(spec, sources=tuple(source for source in spec.sources if source.role != "autogluon_profile"))
    lifecycle.update(spec=encode_snapshot(spec), spec_sha256=spec.binding_sha256)
    request["binding"]["config_hash"] = spec.binding_sha256
    (root / "worker-request.json").write_bytes(bridge._canonical({"payload": lifecycle, "signature": bridge.authenticate(lifecycle, key)}))
    request["backend_binding"]["lifecycle_sha256"] = bridge._file_hash(root / "worker-request.json")
    events = []
    with pytest.raises(ValueError, match="approval"):
        module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                           authorizer(key, lifecycle, events), backend_type=RecordingBackend)
    assert not events and not RecordingBackend.events


def test_trainer_and_abstract_model_hook_share_exact_identity_with_fakes():
    module = worker()
    class Abstract:
        def fit(self, **kwargs):
            return self
    class XT(Abstract):
        pass
    class Trainer:
        def _train_single(self, X, y, model, **kwargs):
            return model.fit()
    owner = []
    with (module._model_audit(Trainer, XT, owner) as trainer_records,
          module._abstract_fit_audit(Abstract, XT, owner) as abstract_records):
        model = XT()
        assert Trainer()._train_single(None, None, model) is model
    assert owner == [model]
    assert trainer_records[0]["outcome"] == abstract_records[0]["outcome"] == "success"


def test_unrouted_abstract_fit_is_rejected_with_fake():
    module = worker()
    class Abstract:
        def fit(self, **kwargs):
            pytest.fail("unrouted actual method must never run")
    class XT(Abstract):
        pass
    with pytest.raises(ValueError, match="unrouted"), module._abstract_fit_audit(Abstract, XT, []):
        XT().fit()


def test_study_method_override_is_rejected_before_debit(tmp_path, monkeypatch):
    module, root, request, key, lifecycle = fixture_request(tmp_path, monkeypatch)
    lifecycle["spec"]["spec"]["candidates"]["$tuple"][0]["study_method"] = "ignored_bayesian_method"
    (root / "worker-request.json").write_bytes(bridge._canonical({"payload": lifecycle, "signature": bridge.authenticate(lifecycle, key)}))
    request["backend_binding"]["lifecycle_sha256"] = bridge._file_hash(root / "worker-request.json")
    events = []
    with pytest.raises(ValueError, match="study method"):
        module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                           authorizer(key, lifecycle, events), backend_type=RecordingBackend)
    assert not events and not RecordingBackend.events


def test_authenticated_resolved_profile_override_rejected_before_debit(tmp_path, monkeypatch):
    module, root, request, key, lifecycle = fixture_request(tmp_path, monkeypatch)
    request["backend_binding"]["resolved_profile"]["constructor"]["learner_kwargs"]["random_state"] = 0
    events = []
    with pytest.raises(ValueError, match="resolved"):
        module.run_backend(root, (request, key, bridge.authenticate(request, key)),
                           authorizer(key, lifecycle, events), backend_type=RecordingBackend)
    assert not events and not RecordingBackend.events


def test_installed_model_entrypoint_signatures_without_fitting():
    root = Path(__file__).resolve().parents[2]
    interpreter = str(root / ".venv-autogluon/bin/python")
    code = (f"import sys;sys.path.insert(0,{str(root / 'src')!r}); "
            "import inspect; from project.models import full_autogluon_worker as W; "
            "from autogluon.core.models.abstract.abstract_model import AbstractModel; "
            "from autogluon.tabular.models.xt.xt_model import XTModel; "
            "from autogluon.tabular.trainer.abstract_trainer import AbstractTabularTrainer as T; "
            "assert W._backend_type().__name__ == 'TabularPredictor'; "
            "assert XTModel.fit is AbstractModel.fit; "
            "assert 'model' in inspect.signature(T._train_single).parameters; "
            "print('reviewed model hooks available; no construction or fit')")
    result = subprocess.run([interpreter, "-I", "-c", code], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "reviewed model hooks available; no construction or fit"


def test_failed_forest_call_prevents_retry_even_when_fake_backend_catches_error(monkeypatch):
    from sklearn.ensemble import ExtraTreesClassifier
    module = worker()
    attempted = []
    def failed(model, *args, **kwargs):
        attempted.append(model)
        raise RuntimeError("recording-only failed forest")
    monkeypatch.setattr(ExtraTreesClassifier, "fit", failed)
    model = SimpleNamespace(n_estimators=4, warm_start=True, n_jobs=-1, random_state=42,
                            max_leaf_nodes=15000, bootstrap=True, criterion="gini", oob_score=False,
                            max_depth=12, min_samples_leaf=2)
    with pytest.raises(ValueError, match="audit"), module._forest_audit() as records:
        with pytest.raises(RuntimeError):
            ExtraTreesClassifier.fit(model, [], [])
        model.n_estimators = 64
        with pytest.raises(ValueError, match="growth"):
            ExtraTreesClassifier.fit(model, [], [])
    assert attempted == [model]
    assert records[0]["outcome"] == "failed"
