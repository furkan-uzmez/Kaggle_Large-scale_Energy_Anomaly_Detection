"""Independent scientific references and authenticated all-partition fixture runs."""

import json
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from test_full_campaign_ordinary_routes import route_fixture
from threadpoolctl import threadpool_limits

from project.experiments.full_campaign_admission import FullCampaignIntent
from project.experiments.full_campaign_capabilities import (
    build_reviewed_ordinary_arms,
    resolve_full_campaign_capability,
)
from project.experiments.full_campaign_preflight import preflight_full_campaign
from project.experiments.full_candidate_runner import (
    resume_full_campaign,
    run_full_campaign,
)
from project.experiments.full_run_contracts import FullFitJournal, _file_hash

ROUTES = (
    "feat_histogram_quantile", "regime_natural_vs_class_weighted",
    "feat_missingness_passthrough_comparison", "model_extratrees_core_features",
)
ENCODING = "lead_year_offset_1900_uint8_v1"
TREE_PARAMS = {"n_estimators": 64, "max_depth": 12, "min_samples_leaf": 2,
               "max_features": "sqrt", "bootstrap": False, "class_weight": None, "n_jobs": 1, "random_state": 42}


def partition(fixture):
    raw = fixture.full_frame.head(52).to_pandas()
    manifest = pl.read_parquet(fixture.split_path).filter(pl.col("repeat") == 0)
    buildings = manifest.filter(pl.col("fold") == 0)["building_id"].to_list()
    mask = raw["building_id"].isin(buildings)
    assert not set(raw.loc[mask, "building_id"]) & set(raw.loc[~mask, "building_id"])
    pool = list(fixture.spec.candidates[0].spec.feature_pool)
    return raw.loc[~mask, pool], raw.loc[mask, pool], raw.loc[~mask, "anomaly"], raw.loc[mask, "anomaly"]


def distribution(train, probe, method):
    """Independent numpy definition, not the production transformer."""
    values = train["meter_reading"].dropna().to_numpy()
    output = probe.copy()
    query = probe["meter_reading"].to_numpy()
    if method == "bin":
        edges = np.unique(np.quantile(values, np.arange(1, 8) / 8))
        result = np.searchsorted(edges, query, side="right").astype(float)
    else:
        knots = np.unique(np.quantile(values, np.linspace(0, 1, 64)))
        cdf = np.array([(np.count_nonzero(values < k) + .5 * np.count_nonzero(values == k))
                        / len(values) for k in knots])
        result = np.interp(query, knots, cdf)
    result[np.isnan(query)] = np.nan
    output[f"meter_reading_{method}"] = result
    return output


def reference_pipeline(experiment, arm, train):
    numeric = tuple(n for n in train if n != "primary_use")
    native = experiment == "feat_missingness_passthrough_comparison" and arm == "control"
    model = ExtraTreesClassifier(**TREE_PARAMS) if (
        experiment == "model_extratrees_core_features" and arm == "challenger") else (
            HistGradientBoostingClassifier(early_stopping=False, random_state=42))
    return Pipeline([
        ("prepare", ColumnTransformer([
            ("numeric", "passthrough" if native else SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), ("primary_use",)),
        ], remainder="drop", sparse_threshold=0.0)),
        ("model", model),
    ])


@pytest.mark.parametrize("experiment", ROUTES)
def test_exact_graph_train_only_state_and_independent_scores(tmp_path, experiment):
    fixture = route_fixture(tmp_path / "source", experiment)
    arms = build_reviewed_ordinary_arms(fixture.spec.candidates[0], year_built_encoding=ENCODING)
    assert tuple(arms) == fixture.spec.candidates[0].arms
    train, valid, y, _ = partition(fixture)
    with threadpool_limits(limits=1):
        for name, arm in arms.items():
            pipeline = arm.transformer_or_pipeline
            X_train, X_valid = train, valid
            if name in ("bin", "quantile"):
                X_train, X_valid = distribution(train, train, name), distribution(train, valid, name)
                feature = pipeline.named_steps["features"]
                actual = feature.fit_transform(train, y)
                np.testing.assert_allclose(actual.iloc[:, -1], X_train.iloc[:, -1], atol=1e-14)
                np.testing.assert_allclose(feature.transform(valid).iloc[:, -1], X_valid.iloc[:, -1], atol=1e-14)
                state = {k: v.copy() for k, v in vars(feature).items() if isinstance(v, np.ndarray)}
                poisoned = valid.copy()
                poisoned["meter_reading"] = 1e12
                feature.transform(poisoned)
                for key, value in state.items():
                    np.testing.assert_array_equal(vars(feature)[key], value)
                feature.fit(train, 1 - y)
                for key, value in state.items():
                    np.testing.assert_array_equal(vars(feature)[key], value)
            expected = reference_pipeline(experiment, name, X_train)
            weights = None
            if experiment == "regime_natural_vs_class_weighted" and name == "challenger":
                weights = np.array([len(y) / (2 * np.count_nonzero(y == label)) for label in y])
                assert pipeline.weight_mode == "balanced"
            if hasattr(pipeline, "weight_mode") and weights is None:
                assert pipeline.weight_mode == "natural"
            expected.fit(X_train, y, **({"model__sample_weight": weights} if weights is not None else {}))
            pipeline.fit(train, y)
            np.testing.assert_allclose(pipeline.predict_proba(valid), expected.predict_proba(X_valid), atol=1e-14, rtol=0)
            model = pipeline.named_steps["model"]
            assert model.get_params() == expected.named_steps["model"].get_params()
            preprocess = pipeline.named_steps["preprocess"]
            if experiment == "feat_missingness_passthrough_comparison" and name == "control":
                assert preprocess.transformers[0][1] == "passthrough"
                transformed = preprocess.named_transformers_["numeric"].transform(valid.drop(columns="primary_use"))
                np.testing.assert_array_equal(np.isnan(transformed), valid.drop(columns="primary_use").isna())
            else:
                imputer = preprocess.named_transformers_["numeric"]
                np.testing.assert_allclose(imputer.statistics_, X_train.drop(columns="primary_use").median(), atol=1e-14)
                frozen = imputer.statistics_.copy()
                pipeline.predict_proba(valid)
                np.testing.assert_array_equal(imputer.statistics_, frozen)


@pytest.mark.parametrize("experiment", ROUTES)
def test_all12_real_transport_receipt_accounting_and_resume(tmp_path, experiment):
    fixture = route_fixture(tmp_path / "source", experiment)
    admitted = preflight_full_campaign(fixture.root, FullCampaignIntent(
        fixture.spec, True, resource_opt_out_record="user: fixture comparison only"))
    assert admitted.ready_experiment_ids == (experiment,)
    assert len(admitted.rows) == 21
    root = tmp_path / "run"
    capability = resolve_full_campaign_capability(fixture.root)
    result = run_full_campaign(fixture.spec, fixture.root, root, data_preflight=admitted, capability=capability)
    row = next(r for r in result["inventory"] if r["experiment_id"] == experiment)
    assert row["status"] == "completed" and row["n03_eligible"] is True
    oof = pl.read_parquet(root / f"{experiment}_oof.parquet")
    arms = fixture.spec.candidates[0].arms
    assert oof.height == 52 * 3 * len(arms)
    assert oof.select("arm_id", "repeat", "fold").unique().height == 12 * len(arms)
    assert oof["score_kind"].unique().to_list() == ["positive_probability"]
    assert oof["n03_eligible"].unique().to_list() == [False]
    train, valid, labels, targets = partition(fixture)
    with threadpool_limits(limits=1):
        for name in arms:
            X_train, X_valid = (distribution(train, train, name), distribution(train, valid, name)) if name in (
                "bin", "quantile") else (train, valid)
            reference = reference_pipeline(experiment, name, X_train)
            kwargs = {}
            if experiment == "regime_natural_vs_class_weighted" and name == "challenger":
                kwargs["model__sample_weight"] = np.array([len(labels) / (2 * np.count_nonzero(labels == value)) for value in labels])
            reference.fit(X_train, labels, **kwargs)
            observed = oof.filter((pl.col("arm_id") == name) & (pl.col("repeat") == 0) & (pl.col("fold") == 0)).sort("row_index")
            assert observed["row_index"].to_list() == valid.index.to_list()
            assert observed["target"].to_list() == targets.to_list()
            np.testing.assert_allclose(observed["prediction"], reference.predict_proba(X_valid)[:, 1], atol=1e-14, rtol=0)
    with FullFitJournal.open_readonly(root / "journal", fixture.spec, source_root=fixture.root) as journal:
        attempts = journal.attempts()
    assert len(attempts) == 12 * len(arms)
    assert all(a.status == "completed" and a.unit.endswith("/model") for a in attempts)
    assert json.loads((root / "completion/receipt.json").read_text())["verified_model_fits"] == len(attempts)
    assert resume_full_campaign(fixture.spec, fixture.root, root, data_preflight=admitted, capability=capability) == result
    with FullFitJournal.open_readonly(root / "journal", fixture.spec, source_root=fixture.root) as journal:
        assert journal.attempts() == attempts


@pytest.mark.parametrize("experiment", ROUTES)
def test_altered_config_never_admitted(tmp_path, experiment):
    fixture = route_fixture(tmp_path / "source", experiment)
    snapshot = fixture.spec.candidates[0]
    params = (("n_estimators", 65),) if experiment == "model_extratrees_core_features" else (("early_stopping", True),)
    snapshot = replace(snapshot, spec=replace(snapshot.spec, model_params=params))
    with pytest.raises(ValueError):
        preflight_full_campaign(fixture.root, FullCampaignIntent(replace(fixture.spec, candidates=(snapshot,)), True,
            resource_opt_out_record="user: fixture rejection only"))


@pytest.mark.parametrize("experiment,column,value", (
    ("feat_histogram_quantile", "meter_reading", 1.0),
    ("feat_missingness_passthrough_comparison", "floor_count", None),
))
def test_degenerate_outer_training_support_blocks_admission(tmp_path, experiment, column, value):
    fixture = route_fixture(tmp_path / "source", experiment)
    frame = fixture.full_frame.with_columns(pl.lit(value).cast(pl.Float64).alias(column))
    frame.write_parquet(fixture.data_path)
    spec = replace(fixture.spec, sources=tuple(replace(s, sha256=_file_hash(fixture.data_path))
        if s.role == "data" else s for s in fixture.spec.sources))
    with pytest.raises(ValueError, match="training.*support"):
        preflight_full_campaign(fixture.root, FullCampaignIntent(spec, True,
            resource_opt_out_record="user: fixture support rejection only"))
