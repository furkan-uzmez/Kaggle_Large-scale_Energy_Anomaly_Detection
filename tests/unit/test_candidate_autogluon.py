"""Task 8 guards with generated frames and fake external boundaries only."""

import importlib
import inspect
import os
import time
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from candidate_fixtures import binary_target, core_frame
from project.experiments.candidate_contracts import BlockedExperimentError
from project.features.registry import build_lead_baseline_registry


def api():
    return importlib.import_module("project.models.autogluon_adapter")


class FeatureMetadata:
    """Backend-faithful representation of autogluon.common FeatureMetadata."""

    __module__ = "autogluon.common.features.feature_metadata"

    def __init__(self, type_map_raw=None, type_group_map_special=None):
        self.type_map_raw = dict(type_map_raw) if type_map_raw is not None else {}
        self.type_group_map_special = (
            {k: list(v) for k, v in type_group_map_special.items()}
            if type_group_map_special is not None
            else {}
        )


FakeFeatureMetadata = FeatureMetadata


class FakePredictor:
    """Expose the reviewed 1.4.0 public signatures, never fit a real model."""

    instances: ClassVar[list["FakePredictor"]] = []

    def __init__(self, label, problem_type=None, eval_metric=None, path=None,
                 positive_class=None):
        self.init_params = {"label": label, "problem_type": problem_type,
                            "eval_metric": eval_metric, "path": path,
                            "positive_class": positive_class}
        self.classes_ = [1, 0]
        self.positive_class = positive_class
        self.fit_calls = []
        self.prediction_calls = []
        self.__class__.instances.append(self)

    def fit(self, train_data, tuning_data=None, time_limit=None, presets=None,
            hyperparameters=None, fit_weighted_ensemble=True,
            dynamic_stacking=False, calibrate_decision_threshold="auto",
            num_cpus="auto", num_gpus="auto", fit_strategy="sequential",
            memory_limit="auto", **kwargs):
        self.fit_calls.append(dict(train_data=train_data.copy(deep=True),
                                   tuning_data=tuning_data, time_limit=time_limit,
                                   presets=presets, hyperparameters=hyperparameters,
                                   fit_weighted_ensemble=fit_weighted_ensemble,
                                   dynamic_stacking=dynamic_stacking,
                                   calibrate_decision_threshold=calibrate_decision_threshold,
                                   num_cpus=num_cpus, num_gpus=num_gpus,
                                   fit_strategy=fit_strategy, memory_limit=memory_limit,
                                   **kwargs))
        train_data.iloc[0, 0] = 999.0
        return self

    def predict_proba(self, data, model=None, as_pandas=True, as_multiclass=True,
                      transform_features=True):
        self.prediction_calls.append(data.copy(deep=True))
        return pd.DataFrame({1: np.full(len(data), .8), 0: np.full(len(data), .2)},
                            index=data.index)

    def info(self):
        return {
            "model_info": {
                "ExtraTrees": {
                    "hyperparameters": {
                        "n_estimators": np.int64(64),
                        "max_depth": np.int64(12),
                        "min_samples_leaf": np.int64(2),
                    },
                    "feature_metadata": FakeFeatureMetadata(
                        type_map_raw={"meter_reading": "float", "square_feet": "float"},
                        type_group_map_special={"binned": []},
                    ),
                    "val_score": np.float64(0.95),
                    "can_infer": np.bool_(True),
                }
            },
            "feature_metadata_in": FakeFeatureMetadata(
                type_map_raw={"meter_reading": "float", "square_feet": "float"},
                type_group_map_special={"binned": []},
            ),
        }

    def model_names(self):
        return ["ExtraTrees"]

    def _validate_fit_kwargs(self, *, kwargs):
        return kwargs.copy()


@pytest.fixture
def environment(monkeypatch, tmp_path):
    module = api()
    settings = module.AutoGluonSettings(10., 128 * 1024**2)
    frame = core_frame()
    frame.attrs["outer_validation"] = frame.iloc[:2].copy()
    labels = binary_target(frame)
    labels.attrs["outer_validation_labels"] = [1, 0]
    FakePredictor.instances = []
    monkeypatch.setattr(module, "_import_tabular", lambda: SimpleNamespace(
        __version__="1.4.0", TabularPredictor=FakePredictor, FeatureMetadata=FeatureMetadata))
    metadata_module = "autogluon.common.features.feature_metadata"
    original_import = module.importlib.import_module

    def import_boundary(name, package=None):
        if name == metadata_module:
            return SimpleNamespace(FeatureMetadata=FeatureMetadata)
        return original_import(name, package)

    monkeypatch.setattr(module.importlib, 'import_module', import_boundary)
    binding = module._SupervisedRuntime(
        worker_pid=os.getpid(), supervisor_pid=os.getppid(),
        time_limit=settings.time_limit, max_rss_bytes=settings.max_rss_bytes,
        num_cpus=1, deadline_monotonic=time.monotonic() + 10.,
        artifact_path=str(tmp_path), lease_id="synthetic-worker-lease",
    )
    return module, settings, frame, labels, binding


def admit(monkeypatch, environment):
    module, _, _, _, binding = environment
    monkeypatch.setattr(module, "_verify_supervised_runtime", lambda settings: binding)


def test_unsupervised_fit_blocks_before_backend_creation(environment):
    module, settings, frame, labels, _ = environment
    with pytest.raises(BlockedExperimentError) as error:
        module.AutoGluonCandidate(settings).fit(frame, labels)
    assert error.value.code == "supervisor_unavailable"
    assert FakePredictor.instances == []


@pytest.mark.parametrize("change", [
    {"worker_pid": -1}, {"supervisor_pid": -1}, {"time_limit": 11.},
    {"max_rss_bytes": 1}, {"num_cpus": 2}, {"deadline_monotonic": 0.},
    {"deadline_monotonic": float("inf")}, {"artifact_path": "relative"},
    {"lease_id": ""}, {"worker_pid": True},
])
def test_falsified_or_mismatched_runtime_blocks_before_backend_fit(environment, monkeypatch, change):
    module, settings, frame, labels, binding = environment
    monkeypatch.setattr(module, "_verify_supervised_runtime", lambda settings: replace(binding, **change))
    with pytest.raises(BlockedExperimentError):
        module.AutoGluonCandidate(settings).fit(frame, labels)
    assert FakePredictor.instances == []


@pytest.mark.parametrize("binding", [None, True, {}, SimpleNamespace(worker_pid=1)])
def test_runtime_requires_typed_worker_binding(environment, monkeypatch, binding):
    module, settings, frame, labels, _ = environment
    monkeypatch.setattr(module, "_verify_supervised_runtime", lambda settings: binding)
    with pytest.raises(BlockedExperimentError):
        module.AutoGluonCandidate(settings).fit(frame, labels)
    assert FakePredictor.instances == []


def test_runtime_is_rechecked_immediately_before_fit(environment, monkeypatch):
    module, settings, frame, labels, binding = environment
    calls = []

    def verify(settings):
        calls.append(settings)
        if len(calls) == 1:
            return binding
        raise BlockedExperimentError("supervisor_unavailable", "lease has ended")

    monkeypatch.setattr(module, "_verify_supervised_runtime", verify)
    with pytest.raises(BlockedExperimentError, match="lease"):
        module.AutoGluonCandidate(settings).fit(frame, labels)
    assert len(calls) == 2
    assert len(FakePredictor.instances) == 1
    assert FakePredictor.instances[0].fit_calls == []


def test_fixed_profile_is_train_only_and_owns_inputs_and_evidence(environment, monkeypatch):
    module, settings, frame, labels, binding = environment
    admit(monkeypatch, environment)
    original, original_labels = frame.copy(deep=True), labels.copy(deep=True)
    estimator = module.AutoGluonCandidate(settings)
    assert estimator.fit(frame, labels) is estimator
    backend = FakePredictor.instances[0]
    captured = backend.fit_calls[0]
    table = captured.pop("train_data")
    assert tuple(table.columns) == (*tuple(frame.columns), module._LABEL_COLUMN)
    assert table.index.equals(frame.index)
    assert table.attrs == {}
    pd.testing.assert_frame_equal(table[list(frame.columns)], original, check_flags=False,
                                  check_names=True)
    np.testing.assert_array_equal(table[module._LABEL_COLUMN], labels)
    assert captured == {
        "tuning_data": None, "time_limit": 10., "presets": None,
        "hyperparameters": {"XT": {"n_estimators": 64, "max_depth": 12,
                                    "min_samples_leaf": 2}},
        "num_bag_folds": 0, "num_stack_levels": 0, "auto_stack": False,
        "dynamic_stacking": False, "fit_weighted_ensemble": False,
        "calibrate_decision_threshold": False, "calibrate": False,
        "hyperparameter_tune_kwargs": None, "num_cpus": 1, "num_gpus": 0,
        "fit_strategy": "sequential", "memory_limit": .125,
    }
    assert backend.init_params == {"label": module._LABEL_COLUMN, "problem_type": "binary",
                                   "eval_metric": "roc_auc", "path": binding.artifact_path,
                                   "positive_class": 1}
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_series_equal(labels, original_labels)
    assert estimator.backend_version_ == "1.4.0"
    assert estimator.resolved_fit_params_["hyperparameters"] == captured["hyperparameters"]
    assert estimator.backend_model_names_ == ("ExtraTrees",)
    assert estimator.backend_info_["model_info"]["ExtraTrees"]["hyperparameters"]["n_estimators"] == 64
    assert type(estimator.backend_info_["model_info"]["ExtraTrees"]["hyperparameters"]["n_estimators"]) is int
    assert estimator.backend_info_["model_info"]["ExtraTrees"]["val_score"] == 0.95
    assert type(estimator.backend_info_["model_info"]["ExtraTrees"]["val_score"]) is float
    assert estimator.backend_info_["model_info"]["ExtraTrees"]["can_infer"] is True
    assert type(estimator.backend_info_["model_info"]["ExtraTrees"]["can_infer"]) is bool
    assert estimator.backend_info_["feature_metadata_in"] == {
        "type_map_raw": {"meter_reading": "float", "square_feet": "float"},
        "type_group_map_special": {"binned": []},
    }
    assert estimator.backend_info_["model_info"]["ExtraTrees"]["feature_metadata"] == {
        "type_map_raw": {"meter_reading": "float", "square_feet": "float"},
        "type_group_map_special": {"binned": []},
    }
    assert estimator.get_feature_names_out().tolist() == frame.columns.tolist()


@pytest.mark.parametrize("override", ["tuning_data", "unlabeled_data", "validation_data",
                                      "runtime_authorized", "presets", "num_gpus",
                                      "hyperparameters", "groups"])
def test_fit_has_no_arbitrary_override_or_validation_input(environment, override):
    module, settings, frame, labels, _ = environment
    with pytest.raises(TypeError):
        module.AutoGluonCandidate(settings).fit(frame, labels, **{override: True})
    assert FakePredictor.instances == []


@pytest.mark.parametrize("field,value", [
    ("time_limit", True), ("time_limit", 0), ("time_limit", float("nan")),
    ("time_limit", float("inf")), ("time_limit", "10"),
    ("max_rss_bytes", True), ("max_rss_bytes", 0), ("max_rss_bytes", 1.5),
    ("num_cpus", True), ("num_cpus", 0), ("num_cpus", 2),
])
def test_settings_require_explicit_positive_limits_and_cpu_one(field, value):
    with pytest.raises(ValueError, match=field):
        api().AutoGluonSettings(**{"time_limit": 10., "max_rss_bytes": 1024,
                                  field: value})


def test_settings_and_estimator_are_cloneable_without_fitted_state(environment, monkeypatch):
    module, settings, frame, labels, _ = environment
    estimator = module.AutoGluonCandidate(settings)
    assert estimator.get_params() == {"settings": settings}
    assert estimator.settings is settings
    with pytest.raises(FrozenInstanceError):
        settings.num_cpus = 2
    with pytest.raises(NotFittedError):
        estimator.predict_proba(frame)
    admit(monkeypatch, environment)
    estimator.fit(frame, labels)
    copied = clone(estimator)
    assert copied.settings == settings
    assert not hasattr(copied, "classes_")
    estimator.set_params(settings=replace(settings, time_limit=5.))
    with pytest.raises(ValueError, match="settings"):
        estimator.predict_proba(frame)


def test_lazy_import_missing_dependency_and_capability_are_explicit(environment, monkeypatch):
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)

    def missing():
        raise ModuleNotFoundError("synthetic missing autogluon")

    monkeypatch.setattr(module, "_import_tabular", missing)
    for operation in (module.autogluon_capability,
                      lambda: module.AutoGluonCandidate(settings).fit(frame, labels)):
        with pytest.raises(BlockedExperimentError) as error:
            operation()
        assert error.value.code == "optional_dependency_unavailable"
    assert FakePredictor.instances == []


def test_capability_checks_without_constructing_or_fitting_backend(environment):
    module, _, _, _, _ = environment
    assert module.autogluon_capability() == "1.4.0"
    assert FakePredictor.instances == []


@pytest.mark.parametrize("version", [None, "1.3.0", "1.5.0", "garbage"])
def test_unreviewed_version_blocks_without_backend_construction(environment, monkeypatch, version):
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)
    monkeypatch.setattr(module, "_import_tabular", lambda: SimpleNamespace(
        __version__=version, TabularPredictor=FakePredictor))
    with pytest.raises(BlockedExperimentError) as error:
        module.AutoGluonCandidate(settings).fit(frame, labels)
    assert error.value.code == "optional_dependency_unsupported"
    assert FakePredictor.instances == []


def test_kwargs_alone_do_not_prove_support_for_explicit_safety_guards(environment, monkeypatch):
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)

    class Unsupported(FakePredictor):
        def fit(self, train_data, **kwargs):
            raise AssertionError("unsupported fit must not run")

    monkeypatch.setattr(module, "_import_tabular", lambda: SimpleNamespace(
        __version__="1.4.0", TabularPredictor=Unsupported))
    with pytest.raises(BlockedExperimentError) as error:
        module.AutoGluonCandidate(settings).fit(frame, labels)
    assert error.value.code == "optional_dependency_unsupported"
    assert FakePredictor.instances == []


@pytest.mark.parametrize("mode", ["reject", "drop", "change"])
def test_backend_kwarg_validation_must_accept_and_preserve_every_guard(environment, monkeypatch, mode):
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)

    class UnsupportedKwargs(FakePredictor):
        def _validate_fit_kwargs(self, *, kwargs):
            if mode == "reject":
                raise ValueError("auto_stack unsupported")
            checked = kwargs.copy()
            if mode == "drop":
                checked.pop("auto_stack")
            else:
                checked["auto_stack"] = True
            return checked

    monkeypatch.setattr(module, "_import_tabular", lambda: SimpleNamespace(
        __version__="1.4.0", TabularPredictor=UnsupportedKwargs))
    with pytest.raises(BlockedExperimentError) as error:
        module.AutoGluonCandidate(settings).fit(frame, labels)
    assert error.value.code == "optional_dependency_unsupported"
    assert FakePredictor.instances[0].fit_calls == []


@pytest.mark.parametrize("mutation", ["empty", "duplicate_index", "null_index", "repeated_column",
                                      "target", "collision", "numeric_bool", "numeric_complex",
                                      "infinity", "numeric_object", "category_number", "hour",
                                      "year_built", "unused_numeric_category"])
def test_strict_fit_frame_rejects_invalid_domains_before_backend(environment, monkeypatch, mutation):
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)
    frame = frame.copy(deep=True)
    if mutation == "empty":
        frame, labels = frame.iloc[:0], labels.iloc[:0]
    elif mutation == "duplicate_index":
        frame.index = [0] * len(frame)
    elif mutation == "null_index":
        frame.index = [None, *range(1, len(frame))]
    elif mutation == "repeated_column":
        frame.columns = ["meter_reading"] * len(frame.columns)
    elif mutation == "unused_numeric_category":
        frame["primary_use"] = pd.Categorical(["office"] * len(frame), categories=["office", 1])
    else:
        column, value = {
            "target": ("anomaly", 1), "collision": (module._LABEL_COLUMN, 1),
            "numeric_bool": ("square_feet", True), "numeric_complex": ("floor_count", 1j),
            "infinity": ("meter_reading", np.inf), "numeric_object": ("year", "2024"),
            "category_number": ("primary_use", 7), "hour": ("hour", 24),
            "year_built": ("year_built", 3000),
        }[mutation]
        frame[column] = value
    with pytest.raises((ValueError, TypeError)):
        module.AutoGluonCandidate(settings).fit(frame, labels)
    assert FakePredictor.instances == []


@pytest.mark.parametrize("mutation", ["reverse_index", "boolean", "float", "one_class", "null", "array"])
def test_fit_target_is_aligned_integer_binary_with_both_classes(environment, monkeypatch, mutation):
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)
    labels = {"reverse_index": lambda: labels.iloc[::-1], "boolean": lambda: labels.astype(bool),
              "float": lambda: labels.astype(float), "one_class": lambda: labels * 0,
              "null": lambda: labels.astype("Int8").mask(labels.index == 0),
              "array": lambda: labels.to_numpy()}[mutation]()
    with pytest.raises((TypeError, ValueError)):
        module.AutoGluonCandidate(settings).fit(frame, labels)
    assert FakePredictor.instances == []


def test_probability_mapping_row_alignment_empty_schema_and_immutability(environment, monkeypatch):
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)
    estimator = module.AutoGluonCandidate(settings).fit(frame, labels)
    np.testing.assert_array_equal(estimator.classes_, [0, 1])
    probe = frame.iloc[[7, 2, 10]].copy(deep=True)
    before = probe.copy(deep=True)
    probabilities = estimator.predict_proba(probe)
    np.testing.assert_allclose(probabilities, [[.2, .8]] * 3)
    assert probabilities.dtype == np.float64
    pd.testing.assert_frame_equal(probe, before)
    backend = FakePredictor.instances[0]
    assert backend.prediction_calls[0].index.equals(probe.index)
    assert backend.prediction_calls[0].attrs == {}
    assert estimator.predict_proba(probe.iloc[:0]).shape == (0, 2)
    assert len(backend.prediction_calls) == 1
    with pytest.raises(ValueError, match="schema"):
        estimator.predict_proba(probe.iloc[:, ::-1])
    with pytest.raises(ValueError, match="input_features"):
        estimator.get_feature_names_out(["meter_reading"])


@pytest.mark.parametrize("failure", ["nan", "infinity", "range", "row_count", "index",
                                     "class_names", "boolean", "normalization", "array",
                                     "nan_both_normalized", "out_of_range_normalized"])
def test_invalid_backend_probabilities_fail_without_clipping_or_reordering_rows(environment, monkeypatch, failure):
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)
    estimator = module.AutoGluonCandidate(settings).fit(frame, labels)
    result = pd.DataFrame({1: [.8] * len(frame), 0: [.2] * len(frame)}, index=frame.index)
    if failure in {"nan", "infinity", "range", "normalization", "boolean"}:
        result[1] = {"nan": np.nan, "infinity": np.inf, "range": 1.1,
                     "normalization": .9, "boolean": True}[failure]
    elif failure == "nan_both_normalized":
        # Nonfinite probabilities that a clip/nan_to_num repair would hide behind
        # an apparently valid row sum must still be rejected, never imputed.
        result[1], result[0] = np.nan, np.nan
    elif failure == "out_of_range_normalized":
        # Out-of-range probabilities whose raw sum is already 1.0 must be rejected
        # on range, not silently clipped into a plausible-looking distribution.
        result[1], result[0] = 1.2, -.2
    elif failure == "row_count":
        result = result.iloc[:-1]
    elif failure == "index":
        result = result.iloc[::-1]
    elif failure == "class_names":
        result.columns = ["1", "0"]
    else:
        result = result.to_numpy()
    monkeypatch.setattr(FakePredictor.instances[0], "predict_proba", lambda *args, **kwargs: result)
    with pytest.raises((ValueError, TypeError)):
        estimator.predict_proba(frame)


@pytest.mark.parametrize("tamper", [{"classes_": [0, 1]}, {"positive_class": 0},
                                    {"classes_": [1, 1]}, {"positive_class": True}])
def test_backend_class_identity_is_reverified_on_every_prediction(environment, monkeypatch, tamper):
    """A backend whose fitted class identity drifts after fit must block, not remap."""
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)
    estimator = module.AutoGluonCandidate(settings).fit(frame, labels)
    backend = FakePredictor.instances[0]
    for name, value in tamper.items():
        monkeypatch.setattr(backend, name, value)
    with pytest.raises(ValueError):
        estimator.predict_proba(frame)


def test_fit_revalidates_declared_limits_even_when_settings_were_tampered(environment, monkeypatch):
    """Limits are re-checked at fit, so a bypassed frozen record cannot widen them.

    The tampered CPU count is mirrored into the supervisor binding so the
    runtime bounds comparison would accept it; only the fit-time settings
    revalidation can reject this, which isolates that guard from the lease check.
    """
    module, settings, frame, labels, binding = environment
    tampered = module.AutoGluonSettings(settings.time_limit, settings.max_rss_bytes)
    object.__setattr__(tampered, "num_cpus", 8)
    monkeypatch.setattr(module, "_verify_supervised_runtime",
                        lambda _settings: replace(binding, num_cpus=8))
    estimator = module.AutoGluonCandidate(settings)
    estimator.set_params(settings=tampered)
    with pytest.raises(ValueError, match="num_cpus"):
        estimator.fit(frame, labels)
    assert FakePredictor.instances == []


def test_failed_refit_clears_fitted_state_instead_of_serving_stale_backend(environment, monkeypatch):
    """A blocked refit must leave the estimator unfitted, never reuse the old fit."""
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)
    estimator = module.AutoGluonCandidate(settings).fit(frame, labels)
    assert estimator.predict_proba(frame).shape == (len(frame), 2)
    monkeypatch.setattr(module, "_verify_supervised_runtime",
                        lambda settings: (_ for _ in ()).throw(
                            BlockedExperimentError("supervisor_unavailable", "lease revoked")))
    with pytest.raises(BlockedExperimentError):
        estimator.fit(frame, labels)
    for attribute in ("backend_", "classes_", "settings_", "backend_version_"):
        assert not hasattr(estimator, attribute)
    with pytest.raises(NotFittedError):
        estimator.predict_proba(frame)


@pytest.mark.parametrize("attribute,value,expected", [
    ("model_names", list, ValueError),
    ("model_names", lambda: ["ExtraTrees", ""], ValueError),
    ("model_names", lambda: ["ExtraTrees", 7], ValueError),
    ("model_names", lambda: ("ExtraTrees",), ValueError),
    ("info", lambda: ["not-a-mapping"], TypeError),
    ("info", lambda: None, TypeError),
    ("info", lambda: {"loss": float("nan")}, ValueError),
    ("info", lambda: {"loss": float("inf")}, ValueError),
    ("info", lambda: {"loss": float("-inf")}, ValueError),
    ("info", lambda: {"loss": np.nan}, ValueError),
    ("info", lambda: {"loss": np.inf}, ValueError),
    ("info", lambda: {123: "numeric_key"}, TypeError),
    ("info", lambda: {"unknown_obj": object()}, TypeError),
    ("info", lambda: {"arr": np.array([1, 2, 3])}, TypeError),
    ("info", lambda: {"df": pd.DataFrame({"a": [1]})}, TypeError),
    ("info", lambda: {"series": pd.Series([1])}, TypeError),
])
def test_unusable_backend_evidence_fails_instead_of_being_recorded(
        environment, monkeypatch, attribute, value, expected):
    """Diagnostic inventory must be real and serializable, never a silent blank."""
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)
    monkeypatch.setattr(FakePredictor, attribute, lambda self: value())
    with pytest.raises(expected):
        module.AutoGluonCandidate(settings).fit(frame, labels)


def test_backend_info_truthful_serialization_and_immutability(environment, monkeypatch):
    """Evidence is truthful JSON-native record; input backend info remains immutable."""
    import json
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)

    raw_info = {
        "model_info": {
            "ExtraTrees": {
                "hyperparameters": {
                    "n_estimators": np.int64(64),
                    "learning_rate": np.float64(0.05),
                    "enabled": np.bool_(True),
                },
                "feature_metadata": FeatureMetadata(
                    type_map_raw={"meter_reading": "float"},
                    type_group_map_special={"binned": ["meter_reading"]},
                ),
            }
        },
        "feature_metadata_in": FeatureMetadata(
            type_map_raw={"meter_reading": "float"},
            type_group_map_special={"binned": ["meter_reading"]},
        ),
        "scores": [np.float64(0.8), np.float64(0.9)],
    }
    raw_info_copy = {
        "model_info": {
            "ExtraTrees": {
                "hyperparameters": {
                    "n_estimators": np.int64(64),
                    "learning_rate": np.float64(0.05),
                    "enabled": np.bool_(True),
                },
                "feature_metadata": FeatureMetadata(
                    type_map_raw={"meter_reading": "float"},
                    type_group_map_special={"binned": ["meter_reading"]},
                ),
            }
        },
        "feature_metadata_in": FeatureMetadata(
            type_map_raw={"meter_reading": "float"},
            type_group_map_special={"binned": ["meter_reading"]},
        ),
        "scores": [np.float64(0.8), np.float64(0.9)],
    }
    monkeypatch.setattr(FakePredictor, "info", lambda self: raw_info)

    estimator = module.AutoGluonCandidate(settings).fit(frame, labels)
    evidence = estimator.backend_info_

    # Verify input immutability
    assert raw_info["scores"] == raw_info_copy["scores"]
    assert raw_info["model_info"]["ExtraTrees"]["hyperparameters"] == raw_info_copy["model_info"]["ExtraTrees"]["hyperparameters"]

    # Verify JSON round-trip compliance
    serialized = json.dumps(evidence, allow_nan=False)
    round_tripped = json.loads(serialized)
    assert round_tripped == evidence

    # Verify projection types
    assert isinstance(evidence["model_info"]["ExtraTrees"]["hyperparameters"]["n_estimators"], int)
    assert not isinstance(evidence["model_info"]["ExtraTrees"]["hyperparameters"]["n_estimators"], np.integer)
    assert isinstance(evidence["model_info"]["ExtraTrees"]["hyperparameters"]["learning_rate"], float)
    assert not isinstance(evidence["model_info"]["ExtraTrees"]["hyperparameters"]["learning_rate"], np.floating)
    assert isinstance(evidence["model_info"]["ExtraTrees"]["hyperparameters"]["enabled"], bool)
    assert not isinstance(evidence["model_info"]["ExtraTrees"]["hyperparameters"]["enabled"], np.bool_)
    assert evidence["scores"] == [0.8, 0.9]
    assert all(type(s) is float for s in evidence["scores"])


@pytest.mark.parametrize('module_name', ['caller.untrusted',
                                       'autogluon.common.features.feature_metadata'])
def test_caller_classname_forge_rejected(environment, monkeypatch, module_name):
    """Class name and AutoGluon module prefix cannot substitute for class identity."""
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)

    class FeatureMetadata:
        __module__ = module_name

        def __init__(self):
            self.type_map_raw = {"col": "float"}
            self.type_group_map_special = {}

    monkeypatch.setattr(FakePredictor, "info", lambda self: {
        "model_info": {"ExtraTrees": {"hyperparameters": {}}},
        "feature_metadata_in": FeatureMetadata(),
    })
    with pytest.raises(TypeError, match="unsupported diagnostic object"):
        module.AutoGluonCandidate(settings).fit(frame, labels)


@pytest.mark.parametrize("pathological_case", ["circular_dict", "circular_list", "deep_nesting"])
def test_pathological_recursive_and_deep_nesting_fails(environment, monkeypatch, pathological_case):
    """Circular references and excessively deep structures must raise ValueError."""
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)

    if pathological_case == "circular_dict":
        info = {"model_info": {"ExtraTrees": {"hyperparameters": {}}}}
        info["cycle"] = info
    elif pathological_case == "circular_list":
        cycle_list = []
        cycle_list.append(cycle_list)
        info = {"model_info": {"ExtraTrees": {"hyperparameters": {}}}, "cycle": cycle_list}
    else:
        deep = {}
        cur = deep
        for _ in range(20):
            cur["nested"] = {}
            cur = cur["nested"]
        info = {"model_info": {"ExtraTrees": {"hyperparameters": {}}}, "deep": deep}

    monkeypatch.setattr(FakePredictor, "info", lambda self: info)
    with pytest.raises(ValueError):
        module.AutoGluonCandidate(settings).fit(frame, labels)


@pytest.mark.parametrize("malformed", [
    {"type_map_raw": None},
    {"type_map_raw": {1: "float"}},
    {"type_map_raw": {"col": 123}},
    {"type_group_map_special": "not_a_mapping"},
    {"type_group_map_special": {1: ["col"]}},
    {"type_group_map_special": {"binned": "not_a_sequence"}},
    {"type_group_map_special": {"binned": [123]}},
])
def test_malformed_feature_metadata_fails(environment, monkeypatch, malformed):
    """FeatureMetadata with invalid type_map_raw or type_group_map_special must raise TypeError."""
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)

    class BrokenMetadata(module._DiagnosticFeatureMetadataBoundary):
        def __init__(self):
            self.type_map_raw = malformed.get("type_map_raw", {"meter_reading": "float"})
            self.type_group_map_special = malformed.get("type_group_map_special", {})

    monkeypatch.setattr(FakePredictor, "info", lambda self: {
        "model_info": {"ExtraTrees": {"hyperparameters": {}}},
        "feature_metadata_in": BrokenMetadata(),
    })
    with pytest.raises(TypeError, match='FeatureMetadata type_'):
        module.AutoGluonCandidate(settings).fit(frame, labels)


def test_boundary_fake_feature_metadata_supported(environment, monkeypatch):
    """The private boundary fake class is accepted and accurately projected."""
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)

    boundary_meta = module._DiagnosticFeatureMetadataBoundary(
        type_map_raw={"square_feet": "float"},
        type_group_map_special={"binned": ["square_feet"]},
    )
    monkeypatch.setattr(FakePredictor, "info", lambda self: {
        "model_info": {"ExtraTrees": {"hyperparameters": {}}},
        "feature_metadata_in": boundary_meta,
    })

    estimator = module.AutoGluonCandidate(settings).fit(frame, labels)
    assert estimator.backend_info_["feature_metadata_in"] == {
        "type_map_raw": {"square_feet": "float"},
        "type_group_map_special": {"binned": ["square_feet"]},
    }


def test_imported_metadata_subclass_is_recognized(environment, monkeypatch):
    module, settings, frame, labels, _ = environment
    admit(monkeypatch, environment)

    class SpecializedMetadata(FakeFeatureMetadata):
        pass

    monkeypatch.setattr(FakePredictor, 'info', lambda self: {
        'model_info': {'ExtraTrees': {'hyperparameters': {}}},
        'feature_metadata_in': SpecializedMetadata({'meter_reading': 'float'}),
    })
    estimator = module.AutoGluonCandidate(settings).fit(frame, labels)
    assert estimator.backend_info_['feature_metadata_in'] == {
        'type_map_raw': {'meter_reading': 'float'}, 'type_group_map_special': {},
    }


def test_reserved_label_column_cannot_collide_with_any_approved_predictor(environment):
    """No approved predictor may claim the reserved label name, and a frame
    carrying it is rejected at the documented collision-verification point."""
    module, _, frame, _, _ = environment
    approved = build_lead_baseline_registry().names
    assert module._LABEL_COLUMN not in approved
    assert module._LABEL_COLUMN not in frame.columns
    colliding = frame.copy(deep=True)
    colliding[module._LABEL_COLUMN] = 1
    with pytest.raises(ValueError, match="collision"):
        module._validate_frame(colliding)


def test_module_import_does_not_import_optional_backend(monkeypatch):
    module = api()
    original = importlib.import_module

    def deny_optional(name, *args, **kwargs):
        if name.startswith("autogluon"):
            raise AssertionError("importing adapter must not import optional backend")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", deny_optional)
    importlib.reload(module)
    assert tuple(inspect.signature(module.AutoGluonCandidate).parameters) == ("settings",)
