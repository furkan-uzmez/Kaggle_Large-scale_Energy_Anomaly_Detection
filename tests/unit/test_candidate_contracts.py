"""Task 1 behavioral guards and generated-only protocol tests; no model fitting."""

import fcntl
import importlib
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import FrozenInstanceError, fields, replace
from threading import Barrier

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from project.features.registry import (
    FeatureRegistry,
    ProvenanceStatus,
    TargetDependence,
    build_lead_baseline_registry,
)


def api():
    return importlib.import_module("project.experiments.candidate_contracts")


def budget():
    return api().FitBudget(12, 2, 10.0, 128 * 1024**2)


def candidate():
    return api().CandidateSpec(
        "feat_meter_deviation_ratio",
        ("meter_reading", "square_feet", "year_built", "floor_count", "hour",
         "weekday", "month", "year", "primary_use"),
        "hist_gradient_boosting", (("early_stopping", False), ("random_state", 42)),
    )


def run_spec():
    return api().RunSpec(1, (candidate(),), budget(), "synthetic_smoke",
                         "notebooks/02_baseline_and_candidate_design.ipynb", "cell-18")


@pytest.mark.parametrize("name,value", [
    ("max_model_fits", True), ("max_model_fits", 0), ("max_model_fits", -1),
    ("max_model_fits", 1.5), ("max_trials", True), ("max_trials", -1),
    ("max_wall_seconds", True), ("max_wall_seconds", 0),
    ("max_wall_seconds", float("nan")), ("max_wall_seconds", float("inf")),
    ("max_rss_bytes", True), ("max_rss_bytes", 0), ("max_rss_bytes", -1),
    ("cpu_threads", True), ("cpu_threads", 0),
])
def test_budget_requires_exact_positive_finite_caps(name, value):
    with pytest.raises(ValueError, match=name):
        replace(budget(), **{name: value})


def test_fit_counter_fails_before_unbudgeted_fit():
    counter = api().FitCounter(api().FitBudget(2, 0, 5.0, 128 * 1024**2))
    counter.reserve(2)
    counter.before_fit()
    counter.before_fit()
    with pytest.raises(ValueError, match="fit"):
        counter.before_fit()
    assert counter.used_model_fits == 2
    assert counter.reserved_model_fits == 2


def test_owned_counter_api_is_additive_and_requires_pristine_binding(tmp_path):
    counter = api().FitCounter(budget())
    assert callable(getattr(counter, "bind_campaign", None)), "persistent campaign binding is required"
    counter.reserve(1)
    with pytest.raises(api().BlockedExperimentError, match="pristine"):
        counter.bind_campaign(tmp_path)


def test_owned_reservations_are_isolated_and_restored(tmp_path):
    counter = api().FitCounter(budget())
    assert callable(getattr(counter, "bind_campaign", None)), "persistent campaign binding is required"
    counter.bind_campaign(tmp_path)
    counter.claim_owner("owner-a", "manifest-a", ("inner:0:0", "outer:default"), 1)
    counter.claim_owner("owner-b", "manifest-b", ("inner:0:0",), 1)
    attempt = counter.before_owned_fit("owner-b", "inner:0:0")
    counter.finish_owned_fit(attempt, "failed")
    with pytest.raises(api().BlockedExperimentError, match="owned"):
        counter.before_owned_fit("owner-b", "inner:0:0")
    with pytest.raises(api().BlockedExperimentError, match="owner"):
        counter.before_fit()
    fresh = api().FitCounter(budget())
    fresh.bind_campaign(tmp_path)
    assert (fresh.used_model_fits, fresh.reserved_model_fits, fresh.reserved_trials) == (1, 3, 2)
    counter.claim_owner("owner-a", "manifest-a", ("inner:0:0", "outer:default"), 1)
    assert counter.reserved_model_fits == 3


def test_campaign_journal_truncation_cannot_reset_budget(tmp_path):
    counter = api().FitCounter(budget())
    counter.bind_campaign(tmp_path)
    counter.claim_owner("owner", "manifest", ("unit",), 1)
    attempt = counter.before_owned_fit("owner", "unit")
    counter.finish_owned_fit(attempt, "failed")
    with closing(sqlite3.connect(tmp_path / "fit_budget.sqlite3")) as conn, conn:
        conn.execute("DELETE FROM accounting_event WHERE sequence = 3")
    fresh = api().FitCounter(budget())
    with pytest.raises(api().BlockedExperimentError, match="journal"):
        fresh.bind_campaign(tmp_path)


def test_campaign_binding_is_exact_persistent_and_single_root(tmp_path):
    root, other = tmp_path / "root", tmp_path / "other"
    root.mkdir()
    other.mkdir()
    counter = api().FitCounter(budget())
    counter.bind_campaign(root)
    counter.claim_owner("owner", "manifest", ("unit",), 1)
    with pytest.raises(api().BlockedExperimentError, match="one campaign"):
        counter.bind_campaign(other)
    with pytest.raises(api().BlockedExperimentError, match="exact fit budget"):
        api().FitCounter(replace(budget(), max_model_fits=13)).bind_campaign(root)
    with pytest.raises(api().BlockedExperimentError, match="manifest"):
        counter.claim_owner("owner", "changed", ("unit",), 1)
    assert (counter.reserved_model_fits, counter.reserved_trials) == (1, 1)
    assert (root / ".fit_budget.lock").is_file()


def test_campaign_persistent_flock_blocks_accounting_mutation(tmp_path):
    counter = api().FitCounter(budget())
    counter.bind_campaign(tmp_path)
    fd = os.open(tmp_path / ".fit_budget.lock", os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(api().BlockedExperimentError, match="lock"):
            counter.claim_owner("owner", "manifest", ("unit",), 1)
        assert counter.reserved_model_fits == counter.used_model_fits == 0
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    counter.claim_owner("owner", "manifest", ("unit",), 1)
    assert counter.reserved_model_fits == 1


def test_fit_counter_cumulative_atomic_reservations_failed_attempts_and_readonly_state():
    counter = api().FitCounter(api().FitBudget(3, 1, 5.0, 128 * 1024**2))
    with pytest.raises(api().BlockedExperimentError, match="fit") as error:
        counter.before_fit()
    assert error.value.code == "fit_not_reserved"
    counter.reserve(2, trials=1)
    with pytest.raises(RuntimeError):
        counter.before_fit()
        raise RuntimeError("synthetic failed fit after debit")
    assert counter.used_model_fits == 1
    with pytest.raises(api().BlockedExperimentError, match="trial"):
        counter.reserve(1, trials=1)
    assert (counter.reserved_model_fits, counter.reserved_trials) == (2, 1)
    counter.reserve(1)
    counter.before_fit()
    counter.before_fit()
    with pytest.raises(api().BlockedExperimentError, match="fit"):
        counter.reserve(1)
    for name in ("used_model_fits", "reserved_model_fits", "reserved_trials", "budget"):
        with pytest.raises(AttributeError):
            setattr(counter, name, 999)
    assert counter.used_model_fits == 3


@pytest.mark.parametrize("fits,trials", [(True, 0), (0, 0), (-1, 0), (1.5, 0), (1, True), (1, -1)])
def test_fit_counter_invalid_requests_do_not_change_accounting(fits, trials):
    counter = api().FitCounter(budget())
    with pytest.raises(ValueError):
        counter.reserve(fits, trials)
    assert counter.used_model_fits == counter.reserved_model_fits == counter.reserved_trials == 0


def test_fit_counter_parallel_reservation_and_debits_have_no_overspend():
    counter = api().FitCounter(api().FitBudget(3, 0, 5.0, 128 * 1024**2))
    barrier = Barrier(4)

    def reserve_one(_):
        barrier.wait()
        try:
            counter.reserve(1)
            return True
        except api().BlockedExperimentError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(reserve_one, range(4))) == 3

    def debit_one(_):
        try:
            counter.before_fit()
            return True
        except api().BlockedExperimentError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(debit_one, range(8))) == 3
    assert counter.used_model_fits == counter.reserved_model_fits == 3


@pytest.mark.parametrize("kind,values,label", [
    ("invented", np.array([.1, .9]), 1),
    ("positive_probability", np.array([-.1, 1.1]), 1),
    ("decision_margin", np.array([np.nan]), 1),
    ("decision_margin", np.array([np.inf]), 1),
    ("decision_margin", np.array([[1.]]), 1),
    ("decision_margin", np.array([True]), 1),
    ("decision_margin", np.array([1j]), 1),
    ("decision_margin", np.array(["1"]), 1),
    ("decision_margin", np.array([1.]), True),
    ("decision_margin", np.array([1.]), 0),
])
def test_scores_require_explicit_kind_finite_vector_and_class_identity(kind, values, label):
    with pytest.raises(ValueError, match="score|probability|positive_label"):
        api().CandidateScore(kind, values, label)


def test_scores_copy_float64_values_and_cannot_mutate_source_or_packet():
    original = np.array([-2, 2], dtype=np.int32)
    score = api().CandidateScore("decision_margin", original)
    original[0] = 100
    np.testing.assert_array_equal(score.values, [-2., 2.])
    assert score.values.dtype == np.float64
    with pytest.raises(ValueError):
        score.values[0] = 0
    with pytest.raises(ValueError):
        score.values.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        score.kind = "positive_probability"
    probability = api().CandidateScore("positive_probability", np.array([0., 1.]))
    np.testing.assert_array_equal(probability.values, [0., 1.])


@pytest.mark.parametrize("change,expected_error", [
    ({"feature_pool": ("meter_reading", "meter_reading")}, ValueError),
    ({"feature_pool": ("building_id",)}, ValueError),
    ({"feature_pool": ("gte_probe",)}, ValueError),
    ({"feature_pool": ["meter_reading"]}, ValueError),
    ({"model_params": {"C": 1}}, TypeError),
    ({"model_params": (("C", 1.), ("C", 2.))}, ValueError),
    ({"method_params": (("bins", [1, 2]),)}, ValueError),
    ({"method_params": (("factory", lambda: None),)}, ValueError),
    ({"model_params": (("C", float("nan")),)}, ValueError),
])
def test_specs_reject_mutable_duplicate_reserved_or_nonfinite_parameters(change, expected_error):
    with pytest.raises(expected_error, match="pool|parameter|params|predictor"):
        replace(candidate(), **change)


def test_run_configuration_is_frozen_cloneable_and_not_a_permission_grant():
    spec = run_spec()
    assert clone(spec, safe=False) == spec
    with pytest.raises(FrozenInstanceError):
        spec.version = 2
    full = replace(spec, execution_mode="full")
    assert full.execution_mode == "full"
    assert not hasattr(full, "execution_authorized")
    with pytest.raises(TypeError):
        api().RunSpec(1, (candidate(),), budget(), "full", spec.source_notebook,
                      spec.source_cell_id, execution_authorized=True)
    for change in ({"version": True}, {"source_cell_id": " "},
                   {"execution_mode": "trust_me_synthetic"},
                   {"candidates": [candidate()]}, {"candidates": (candidate(), candidate())},
                   {"source_notebook": "../outside.ipynb"},
                   {"budget": replace(budget(), cpu_threads=2)}):
        with pytest.raises(ValueError, match="version|cell|mode|candidate|notebook|CPU|cpu"):
            replace(spec, **change)


def test_study_space_is_explicit_immutable_bounded_and_unique():
    valid = api().StudySpec(2, 2, 42, (("learning_rate", (.03, .1)), ("max_iter", (100,))))
    assert clone(valid, safe=False) == valid
    for change in ({"n_trials": True}, {"n_trials": 0}, {"n_trials": 3},
                   {"inner_folds": 1}, {"inner_folds": 6}, {"seed": True},
                   {"parameter_lists": (("early_stopping", (True,)),)},
                   {"parameter_lists": (("max_iter", (99,)),)},
                   {"parameter_lists": (("learning_rate", (.1, .1)),)},
                   {"parameter_lists": (("max_iter", (100,)), ("max_iter", (200,)))},
                   {"parameter_lists": (("learning_rate", [.03, .1]),)}):
        with pytest.raises(ValueError, match="trial|fold|seed|parameter|space"):
            replace(valid, **change)


def test_blocked_error_and_result_truthfulness():
    error = api().BlockedExperimentError("optional_dependency_unavailable", "backend unavailable")
    assert error.code == "optional_dependency_unavailable" and str(error) == "backend unavailable"
    with pytest.raises(ValueError):
        api().BlockedExperimentError("", "")
    result = api().RunResult("diagnostic-run", "completed", ("bundle/metrics.json",), True, False)
    assert result.non_comparative and not result.n03_eligible
    for change in ({"n03_eligible": True}, {"non_comparative": 1},
                   {"status": "invented"}, {"artifact_paths": ()},
                   {"artifact_paths": ["bundle/metrics.json"]}):
        with pytest.raises(ValueError, match="eligible|comparative|status|artifact"):
            replace(result, **change)


def fixture_api():
    return importlib.import_module("candidate_fixtures")


def test_generated_protocol_fresh_copies_keys_groups_and_inner_class_support():
    fixture = fixture_api()
    frame = fixture.core_frame()
    assert len(frame) == 16
    assert len(frame) <= 128
    assert frame.columns.tolist() == list(candidate().feature_pool)
    groups = frame.attrs["groups"]
    assert groups.index.equals(frame.index) and groups.nunique() == 8
    target = fixture.binary_target(frame)
    for group in groups.unique():
        assert set(target[groups == group]) == {0, 1}
    partitions = fixture.synthetic_partitions()
    assert len(partitions) == 2
    assert [(p.repeat, p.fold) for p in partitions] == [(0, 0), (0, 1)]
    validation_ids = []
    for partition in partitions:
        api().validate_partition(partition, build_lead_baseline_registry())
        assert not set(partition.groups_train) & set(partition.validation_keys.building_id)
        assert partition.groups_train.nunique() == 4
        validation_ids.extend(partition.validation_keys.row_index.tolist())
        assert not partition.validation_keys.duplicated(["building_id", "timestamp"]).any()
        for count in (2, 3):
            folds = StratifiedGroupKFold(count, shuffle=True, random_state=42).split(
                partition.X_train, partition.y_train, partition.groups_train)
            for train, valid in folds:
                assert not set(partition.groups_train.iloc[train]) & set(partition.groups_train.iloc[valid])
                assert set(partition.y_train.iloc[train]) == set(partition.y_train.iloc[valid]) == {0, 1}
    assert sorted(validation_ids) == list(range(16))
    again = fixture.synthetic_partitions()
    for first, second in zip(partitions, again, strict=True):
        pd.testing.assert_frame_equal(first.X_train, second.X_train)
        pd.testing.assert_frame_equal(first.validation_keys, second.validation_keys)
        assert first.X_train is not second.X_train
    frame.iloc[0, 0] = 999
    groups.iloc[0] = "changed"
    assert fixture.core_frame().iloc[0, 0] != 999
    assert fixture.core_frame().attrs["groups"].iloc[0] != "changed"


@pytest.fixture
def partition():
    return fixture_api().synthetic_partitions()[0]


def test_partition_and_training_view_never_expose_validation_targets_or_aliases(partition):
    contract = api()
    assert "y_validation" not in {field.name for field in fields(partition)}
    partition.X_train.attrs["outer_validation_targets"] = fixture_api().binary_target(partition.X_validation)
    training = contract.training_view(partition)
    assert {field.name for field in fields(training)} == {"repeat", "fold", "X_train", "y_train", "groups_train"}
    assert training.X_train.attrs == training.y_train.attrs == training.groups_train.attrs == {}
    assert not hasattr(training, "X_validation")
    before = partition.X_train.copy(deep=True)
    training.X_train.iloc[0, 0] = 999
    pd.testing.assert_frame_equal(partition.X_train, before)
    with pytest.raises(FrozenInstanceError):
        training.fold = 3


@pytest.mark.parametrize("field", ["y_train", "groups_train"])
def test_partition_rejects_misaligned_training_metadata(partition, field):
    with pytest.raises(ValueError, match="index|align"):
        replace(partition, **{field: getattr(partition, field).iloc[::-1]})


@pytest.mark.parametrize("mutation", ["overlap", "natural_key", "row_id", "fold", "target_column", "timestamp"])
def test_partition_validation_keys_reject_contamination_and_bad_identity(partition, mutation):
    keys = partition.validation_keys.copy()
    if mutation == "overlap":
        keys["building_id"] = partition.groups_train.iloc[0]
    elif mutation == "natural_key":
        keys.iloc[1, keys.columns.get_loc("building_id")] = keys.building_id.iloc[0]
        keys.iloc[1, keys.columns.get_loc("timestamp")] = keys.timestamp.iloc[0]
    elif mutation == "row_id":
        keys["row_index"] = 1
    elif mutation == "fold":
        keys["fold"] = 3
    elif mutation == "target_column":
        keys["anomaly"] = 0
    else:
        keys["timestamp"] = "not-a-timestamp"
    with pytest.raises(ValueError, match="group|overlap|disjoint|key|row|fold|column|timestamp"):
        replace(partition, validation_keys=keys)


@pytest.mark.parametrize("column", ["building_id", "anomaly", "gte_probe"])
def test_group_metadata_is_not_a_predictor_even_when_constructor_receives_it(partition, column):
    with pytest.raises(ValueError, match="predictor|schema|column"):
        replace(partition, X_train=partition.X_train.assign(**{column: 1}),
                X_validation=partition.X_validation.assign(**{column: 1}))
    api().validate_partition(partition, build_lead_baseline_registry())


def test_raw_source_validation_cannot_accept_forged_or_derived_metadata(partition):
    baseline = build_lead_baseline_registry()
    for change in ({"provenance_status": ProvenanceStatus.PENDING},
                   {"target_dependence": TargetDependence.TARGET_DERIVED},
                   {"source_columns": ("anomaly",)}):
        registry = FeatureRegistry(replace(baseline.get(n), **change) if n == "meter_reading" else baseline.get(n)
                                   for n in baseline.names)
        with pytest.raises(ValueError, match="source|provenance|target|verified"):
            api().validate_partition(partition, registry)
    forged = replace(baseline.get("meter_reading"), name="additional_raw", source_columns=("additional_raw",))
    registry = FeatureRegistry([baseline.get(n) for n in baseline.names] + [forged])
    added = replace(partition, X_train=partition.X_train.assign(additional_raw=1.),
                    X_validation=partition.X_validation.assign(additional_raw=1.))
    with pytest.raises(ValueError, match="source|producer|verified|registered"):
        api().validate_partition(added, registry)
    with pytest.raises(TypeError, match="FeatureRegistry"):
        api().validate_partition(partition, {"meter_reading": {"verified": True}})


@pytest.mark.parametrize("mutation", ["empty", "duplicate_index", "schema", "one_class", "bool_target", "null_group", "infinity", "bool_numeric", "category"])
def test_partition_rejects_invalid_fit_frames_and_targets(partition, mutation):
    changes = {}
    if mutation == "empty":
        changes = {"X_train": partition.X_train.iloc[:0], "y_train": partition.y_train.iloc[:0],
                   "groups_train": partition.groups_train.iloc[:0]}
    elif mutation == "duplicate_index":
        for field in ("X_train", "y_train", "groups_train"):
            changes[field] = getattr(partition, field).copy()
            changes[field].index = [0] * len(changes[field])
    elif mutation == "schema":
        changes["X_validation"] = partition.X_validation.iloc[:, ::-1]
    elif mutation == "one_class":
        changes["y_train"] = partition.y_train * 0
    elif mutation == "bool_target":
        changes["y_train"] = partition.y_train.astype(bool)
    elif mutation == "null_group":
        changes["groups_train"] = partition.groups_train.copy()
        changes["groups_train"].iloc[0] = None
    else:
        column, value = {"infinity": ("meter_reading", np.inf), "bool_numeric": ("hour", True),
                         "category": ("primary_use", 7)}[mutation]
        changes["X_train"] = partition.X_train.assign(**{column: value})
    with pytest.raises(ValueError, match="row|index|schema|class|target|group|finite|numeric|primary_use"):
        replace(partition, **changes)


def test_run_budget_requires_typed_contract_not_mapping():
    with pytest.raises(TypeError, match="FitBudget"):
        replace(run_spec(), budget={"max_model_fits": 12})


@pytest.mark.parametrize("field", ["X_train", "X_validation"])
def test_partition_requires_dataframe_types(partition, field):
    with pytest.raises(TypeError, match="DataFrame"):
        replace(partition, **{field: getattr(partition, field).to_numpy()})


def test_partition_accepts_nullable_widths_missing_predictors_and_new_category_vocabularies(partition):
    train, valid = partition.X_train.copy(), partition.X_validation.copy()
    for name in train.columns[:-1]:
        train[name] = pd.array(train[name], dtype="Float64")
        valid[name] = pd.array(valid[name], dtype="Float32")
    train.iloc[0, train.columns.get_loc("meter_reading")] = pd.NA
    train["primary_use"] = pd.array([pd.NA] * len(train), dtype="string")
    valid["primary_use"] = pd.Categorical(["heldout_only"] * len(valid))
    labels = partition.y_train.astype("Int8")
    packet = replace(partition, X_train=train, X_validation=valid, y_train=labels)
    api().validate_partition(packet, build_lead_baseline_registry())
    assert packet.X_train.meter_reading.isna().iloc[0]
    assert packet.y_train.dtype == pd.Int8Dtype()
    assert set(packet.X_validation.primary_use) == {"heldout_only"}


def test_partition_owns_snapshots_without_mutating_inputs(partition):
    train, labels, groups = partition.X_train.copy(), partition.y_train.copy(), partition.groups_train.copy()
    train.attrs["caller_metadata"] = "preserve on caller only"
    packet = replace(partition, X_train=train, y_train=labels, groups_train=groups)
    assert train.attrs["caller_metadata"] == "preserve on caller only"
    assert packet.X_train.attrs == {}
    expected_train, expected_labels, expected_groups = packet.X_train.copy(), packet.y_train.copy(), packet.groups_train.copy()
    train.iloc[0, 0] = 999
    labels.iloc[0] = 1 - labels.iloc[0]
    groups.iloc[0] = "changed"
    pd.testing.assert_frame_equal(packet.X_train, expected_train)
    pd.testing.assert_series_equal(packet.y_train, expected_labels)
    pd.testing.assert_series_equal(packet.groups_train, expected_groups)
