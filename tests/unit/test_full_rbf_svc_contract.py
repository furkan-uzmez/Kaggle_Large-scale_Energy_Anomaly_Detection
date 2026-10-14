"""Pure route tests: no data files and no real estimator fitting."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from project.experiments import full_rbf_svc_contract as contract
from project.experiments.candidate_contracts import CandidateSpec
from project.experiments.candidate_dispatch import _configuration
from project.experiments.full_input_schema import CORE
from project.experiments.full_run_contracts import (
    FullCandidateSnapshot,
    FullResourcePolicy,
    FullRunSpec,
    SourceBinding,
)
from project.experiments.full_snapshot_codec import (
    configuration_digest,
    decode_snapshot,
    encode_snapshot,
)
from project.features.input_encoding import LEAD_YEAR_OFFSET_1900_UINT8_V1
from project.models.candidate_scores import (
    evaluate_candidate_metrics,
    extract_candidate_score,
)


def api(name):
    value = getattr(contract, name, None)
    assert callable(value), f"missing RBF contract behavior: {name}"
    return value


def snapshot():
    return FullCandidateSnapshot(
        CandidateSpec("dim_pca_vs_passthrough", CORE, "svm", ()),
        ("control", "challenger"), "notebooks/02_feature_engineering.ipynb", "rbf-source",
    )


def run():
    sources = tuple(SourceBinding(role, f"evidence/{role}", str(i) * 64)
                    for i, role in enumerate(("matrix", "data", "split", "producer", "notebook"), 1))
    return FullRunSpec("rbf-fixture", (snapshot(),), sources, (("python", "fixture"),),
                       FullResourcePolicy(True), development_rows=100, vault_rows=20)


def cardinality(spec):
    return api("RBFDevelopmentCardinality")(
        spec.binding_sha256, spec.sources[1], spec.sources[2], spec.development_rows,
        ((20, 25, 25, 30), (25, 25, 25, 25), (30, 20, 30, 20)),
    )


def build(item=None):
    return api("build_full_rbf_svc_arms")(
        snapshot() if item is None else item,
        year_built_encoding=LEAD_YEAR_OFFSET_1900_UINT8_V1,
    )


def test_exact_graph_identity_score_and_snapshot_transport():
    arms = build()
    assert tuple(arms) == ("control", "challenger")
    assert len(snapshot().planned_units) == 24
    assert decode_snapshot(encode_snapshot(run())) == run()
    for name, arm in arms.items():
        assert arm.candidate_id == f"dim_pca_vs_passthrough_{name}"
        assert arm.score_kind == "decision_margin"
        assert arm.raw_pool == CORE
        pipe = arm.transformer_or_pipeline
        model = pipe.named_steps["model"]
        params = model.get_params()
        assert params["kernel"] == "rbf" and params["C"] == 1.0
        assert params["gamma"] == "scale" and params["max_iter"] == -1
        assert params["random_state"] == 42 and params["class_weight"] is None
        # Installed sklearn 1.9 disables probabilities via this default sentinel;
        # do not rewrite the reviewed factory to the deprecated boolean setting.
        assert params["probability"] == "deprecated"
        assert not hasattr(model, "predict_proba")
        assert params["cache_size"] == 200.0
        assert pipe.named_steps["selector"] == "passthrough"
        pre = pipe.named_steps["preprocessor"]
        assert pre.use_pca is (name == "challenger")
        assert pre.pca_components == .95
        assert not hasattr(pre, "column_transformer_")
        assert configuration_digest(_configuration(arm)) == configuration_digest(_configuration(build()[name]))
    assert configuration_digest(_configuration(arms["control"])) != configuration_digest(_configuration(arms["challenger"]))


@pytest.mark.parametrize("changes", [
    {"model_family": "logistic_l2"}, {"feature_pool": CORE[::-1]},
    {"model_params": (("kernel", "linear"),)}, {"model_params": (("probability", True),)},
    {"model_params": (("C", 1),)}, {"model_params": (("max_iter", 100),)},
    {"method_params": (("pca_components", .9),)}, {"method_params": (("sample_rows", 10),)},
])
def test_rejects_scientific_drift(changes):
    with pytest.raises(ValueError):
        build(replace(snapshot(), spec=replace(snapshot().spec, **changes)))


@pytest.mark.parametrize("changes", [
    {"arms": ("passthrough", "pca")}, {"selector_arms": ("challenger",)},
    {"arms": ("challenger", "control")},
])
def test_rejects_identity_and_accounting_drift(changes):
    with pytest.raises(ValueError):
        build(replace(snapshot(), **changes))


def test_source_bound_feasibility_is_not_admission_or_allocation_claim():
    spec = run()
    report = api("estimate_full_rbf_svc")(spec, cardinality(spec))
    assert report.planned_model_fits == 24
    assert report.planned_selector_fits == 0
    assert report.oof_rows_per_arm == 300
    assert report.largest_training_rows == 80
    assert report.conceptual_dense_gram_bytes == 80 * 80 * 8
    assert report.kernel_cache_mib == 200.0
    assert report.execution_authorized is False
    assert report.wall_seconds_estimate is None
    assert report.risk == "severe_compute_risk"
    assert report.resource_policy == FullResourcePolicy(True)
    assert report.spec_sha256 == spec.binding_sha256
    assert report == api("estimate_full_rbf_svc")(spec, cardinality(spec))
    changed = replace(cardinality(spec), data_source=replace(spec.sources[1], sha256="a" * 64))
    with pytest.raises(ValueError, match="source|binding"):
        api("estimate_full_rbf_svc")(spec, changed)
    with pytest.raises(ValueError, match="binding"):
        api("estimate_full_rbf_svc")(replace(spec, run_id="changed"), cardinality(spec))


@pytest.mark.parametrize("counts", [
    ((25, 25, 25, 25),), ((25, 25, 25, 24),) * 3,
    ((True, 25, 25, 49),) * 3, ((0, 25, 25, 50),) * 3,
])
def test_cardinality_requires_all_actual_partition_counts(counts):
    spec = run()
    with pytest.raises(ValueError):
        replace(cardinality(spec), validation_rows=counts)


def test_training_pipeline_routes_only_supplied_training_rows_to_recording_fakes(monkeypatch):
    from project.features.selection import FoldLocalPreprocessor, _SparseFrameSVC

    events = []
    def record_pre(self, X, y=None):
        events.append(("pre", self.use_pca, tuple(X.index), tuple(y)))
        return self
    def transform(self, X):
        events.append(("transform", self.use_pca, tuple(X.index)))
        return X
    def record_model(self, X, y, sample_weight=None):
        events.append(("model", tuple(X.index), tuple(y)))
        return self
    monkeypatch.setattr(FoldLocalPreprocessor, "fit", record_pre)
    monkeypatch.setattr(FoldLocalPreprocessor, "transform", transform)
    monkeypatch.setattr(_SparseFrameSVC, "fit", record_model)
    train = pd.DataFrame({name: [1., 2.] for name in CORE}, index=[11, 12])
    labels = pd.Series([0, 1], index=train.index)
    for arm in build().values():
        arm.transformer_or_pipeline.fit(train, labels)
    assert events == [
        ("pre", False, (11, 12), (0, 1)), ("transform", False, (11, 12)),
        ("model", (11, 12), (0, 1)),
        ("pre", True, (11, 12), (0, 1)), ("transform", True, (11, 12)),
        ("model", (11, 12), (0, 1)),
    ]


def test_margin_scores_never_become_probabilities():
    build()
    class RecordingMargin:
        classes_ = np.array([0, 1])
        def decision_function(self, X):
            return np.array([-2., 3.])
        def predict_proba(self, X):
            raise AssertionError("probability path forbidden")
    score = extract_candidate_score(RecordingMargin(), np.ones((2, 1)), "decision_margin")
    np.testing.assert_array_equal(score.values, [-2., 3.])
    metrics = evaluate_candidate_metrics(np.array([0, 1]), score)
    assert metrics["log_loss"] is None and metrics["brier_score"] is None
    assert metrics["guardrail_status"] == "requires_calibrated_probability"


def test_pca_numeric_only_construction_and_inference_never_refits(monkeypatch):
    from project.features import selection

    events = []
    class RecordingColumnTransformer:
        def __init__(self, branches, **kwargs):
            self.branches = branches
            self.named_transformers_ = {name: pipe for name, pipe, _ in branches}
        def fit(self, X, y):
            events.append(("fit", tuple(X.index)))
            numeric = self.named_transformers_["numeric"]
            pca = numeric.named_steps["pca"]
            if pca != "passthrough":
                assert pca.n_components == .95 and pca.svd_solver == "full"
                assert pca.whiten is False and pca.random_state == 42
                pca.n_components_ = 1
            assert numeric.named_steps["imputer"].strategy == "median"
            assert type(numeric.named_steps["scaler"]).__name__ == "StandardScaler"
            categorical = self.named_transformers_["categorical"]
            encoder = categorical.named_steps["one_hot"]
            assert encoder.handle_unknown == "ignore" and encoder.sparse_output is True
            encoder.categories_ = [np.array(["office"])]
            return self
        def get_feature_names_out(self):
            return np.array(["pca0", "primary_use_office"])
        def transform(self, X):
            events.append(("transform", tuple(X.index)))
            return np.zeros((len(X), 2))
    monkeypatch.setattr(selection, "ColumnTransformer", RecordingColumnTransformer)
    train = pd.DataFrame({name: [1., 2.] for name in CORE}, index=[11, 12])
    train["primary_use"] = "office"
    validation = train.copy()
    validation.index = [21, 22]
    pre = build()["challenger"].named_steps["preprocessor"]
    pre.fit(train, pd.Series([0, 1], index=train.index))
    pre.transform(validation)
    assert events == [("fit", (11, 12)), ("transform", (21, 22))]
    assert pre.registry_.get("pca0").source_columns == CORE[:-1]
    assert pre.registry_.get("primary_use_office").source_columns == ("primary_use",)


def test_predebit_validator_accepts_actual_typed_frozen_svc_defaults():
    from project.features.selection import _SparseFrameSVC
    defaults = tuple(_SparseFrameSVC(C=1., kernel="rbf", gamma="scale", random_state=42).get_params().items())
    item = replace(snapshot(), spec=replace(snapshot().spec, model_params=defaults))
    api("validate_full_rbf_svc_snapshot")(item, year_built_encoding=LEAD_YEAR_OFFSET_1900_UINT8_V1)
    assert tuple(build(item)) == ("control", "challenger")
