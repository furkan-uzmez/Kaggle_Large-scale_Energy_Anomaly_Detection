"""Genuine fixture transport and scientific route contracts; no repository data."""

import json
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from full_campaign_fixtures import write_admission_fixture
from project.experiments.candidate_contracts import CandidateSpec
from project.experiments.full_campaign_admission import FullCampaignIntent
from project.experiments.full_campaign_capabilities import (
    resolve_full_campaign_capability,
)
from project.experiments.full_campaign_preflight import preflight_full_campaign
from project.experiments.full_candidate_runner import (
    resume_full_campaign,
    run_full_campaign,
)
from project.experiments.full_run_contracts import FullFitJournal, _file_hash

ROUTES = (
    "feat_categorical_interactions", "feat_frequency_encoding",
    "feat_meter_deviation_ratio",
)


def route_fixture(root, experiment):
    fixture = write_admission_fixture(root)
    frame = fixture.full_frame.with_columns(
        pl.when(pl.col("building_id") == "bldg_amber").then(pl.lit("ValidationOnly"))
        .otherwise(pl.col("primary_use")).alias("primary_use"),
        pl.when(pl.col("building_id") == "bldg_birch").then(None)
        .otherwise(pl.col("floor_count")).alias("floor_count"),
        pl.when(pl.col("building_id") == "bldg_cedar").then(0.0)
        .otherwise(pl.col("square_feet")).alias("square_feet"))
    frame.write_parquet(fixture.data_path)
    sources = tuple(replace(source, sha256=_file_hash(fixture.data_path))
                    if source.role == "data" else source for source in fixture.spec.sources)
    family = "extra_trees" if experiment == "model_extratrees_core_features" else "hist_gradient_boosting"
    params = (("n_estimators", 64), ("max_depth", 12), ("min_samples_leaf", 2),
              ("max_features", "sqrt"), ("bootstrap", False), ("class_weight", None),
              ("n_jobs", 1), ("random_state", 42)) if family == "extra_trees" else (
                  ("early_stopping", False), ("random_state", 42))
    methods = (("n_bins", 8), ("n_quantiles", 64)) if experiment == "feat_histogram_quantile" else ()
    spec = CandidateSpec(experiment, fixture.spec.candidates[0].spec.feature_pool, family, params, methods)
    arms = ("control", "bin", "quantile") if experiment == "feat_histogram_quantile" else ("control", "challenger")
    snapshot = replace(fixture.spec.candidates[0], spec=spec, arms=arms)
    return replace(fixture, full_frame=frame, spec=replace(fixture.spec, candidates=(snapshot,), sources=sources))


def reference_features(train, valid, experiment):
    """Independent arithmetic; no production candidate transformer is called."""
    train, valid = train.copy(), valid.copy()
    if experiment == "feat_frequency_encoding":
        frequencies = train["primary_use"].value_counts(dropna=True) / len(train)
        for frame in (train, valid):
            frame["primary_use_frequency"] = frame["primary_use"].map(frequencies).fillna(0.0)
    elif experiment == "feat_categorical_interactions":
        for frame in (train, valid):
            for calendar in ("hour", "weekday", "month"):
                frame[f"primary_use_x_{calendar}"] = [json.dumps([category, int(value)],
                    ensure_ascii=False, separators=(",", ":")) for category, value
                    in zip(frame["primary_use"], frame[calendar], strict=True)]
    else:
        for frame in (train, valid):
            valid_area = frame["square_feet"].notna() & (frame["square_feet"] > 0)
            ratio = (frame["meter_reading"] / frame["square_feet"]).where(valid_area)
            frame["meter_reading_per_square_foot"] = ratio
            frame["meter_reading_per_square_foot_invalid"] = ratio.isna().astype(np.uint8)
    return train, valid


@pytest.mark.parametrize("experiment", ROUTES)
def test_exact_reviewed_transform_and_train_only_state(tmp_path, experiment):
    import pandas as pd

    from project.experiments.full_campaign_capabilities import (
        build_reviewed_ordinary_arms,
    )

    fixture = route_fixture(tmp_path / "source", experiment)
    raw = fixture.full_frame.head(52).to_pandas()
    pool = list(fixture.spec.candidates[0].spec.feature_pool)
    train_mask = ~raw["building_id"].isin(["bldg_amber", "bldg_birch"])
    train, valid = raw.loc[train_mask, pool], raw.loc[~train_mask, pool]
    labels = raw.loc[train.index, "anomaly"]
    arms = build_reviewed_ordinary_arms(fixture.spec.candidates[0],
                                      year_built_encoding="lead_year_offset_1900_uint8_v1")
    assert tuple(arms) == ("control", "challenger")
    assert "features" not in arms["control"].named_steps
    feature = arms["challenger"].named_steps["features"]
    expected_train, expected_valid = reference_features(train, valid, experiment)
    actual_train = feature.fit_transform(train, labels)
    pd.testing.assert_frame_equal(actual_train, expected_train, check_dtype=False)
    actual_valid = feature.transform(valid)
    pd.testing.assert_frame_equal(actual_valid, expected_valid, check_dtype=False)
    # Fitting features cannot use labels; prediction batches never update state.
    feature.fit(train, 1 - labels)
    pd.testing.assert_frame_equal(feature.transform(valid), actual_valid)
    if experiment == "feat_frequency_encoding":
        frozen = dict(feature.frequencies_)
        poisoned = valid.copy()
        poisoned["primary_use"] = "ValidationOnly"
        assert np.all(feature.transform(poisoned)["primary_use_frequency"] == 0.0)
        poisoned["primary_use"] = None
        assert np.all(feature.transform(poisoned)["primary_use_frequency"] == 0.0)
        assert feature.frequencies_ == frozen
    for arm in arms.values():
        assert arm.raw_pool == tuple(pool)
        assert arm.score_kind == "positive_probability"
        model = arm.named_steps["model"]
        assert model.early_stopping is False and model.random_state == 42
        assert not hasattr(model, "classes_")


@pytest.mark.parametrize("experiment", ROUTES)
@pytest.mark.parametrize("violation", ("model", "labels", "group", "predictors"))
def test_genuine_admission_rejects_scientific_drift_before_run_writes(tmp_path, experiment, violation):
    fixture = route_fixture(tmp_path / "source", experiment)
    spec = fixture.spec
    snapshot = spec.candidates[0]
    if violation == "model":
        snapshot = replace(snapshot, spec=replace(snapshot.spec, model_params=(("early_stopping", True),)))
    elif violation == "predictors":
        snapshot = replace(snapshot, spec=replace(snapshot.spec,
            feature_pool=snapshot.spec.feature_pool[:-1] + ("unreviewed_weather",)))
    else:
        if violation == "labels":
            fixture.full_frame.with_columns(pl.lit(2).alias("anomaly")).write_parquet(fixture.data_path)
            role, path = "data", fixture.data_path
        else:
            manifest = pl.read_parquet(fixture.split_path)
            pl.concat([manifest, manifest.head(1)]).write_parquet(fixture.split_path)
            role, path = "split", fixture.split_path
        spec = replace(spec, sources=tuple(replace(item, sha256=_file_hash(path))
            if item.role == role else item for item in spec.sources))
    spec = replace(spec, candidates=(snapshot,))
    with pytest.raises(ValueError):
        preflight_full_campaign(fixture.root, FullCampaignIntent(
            spec, True, resource_opt_out_record="user: fixture rejection only"))
    assert not (tmp_path / "run").exists()
    assert not (fixture.root / "campaign.lock").exists()


@pytest.mark.parametrize("name", ("building_id", "anomaly", "gte_probe"))
def test_identifier_target_and_gte_cannot_enter_candidate_schema(tmp_path, name):
    fixture = route_fixture(tmp_path / "source", "feat_frequency_encoding")
    with pytest.raises(ValueError, match="ID/target/gte"):
        replace(fixture.spec.candidates[0].spec,
                feature_pool=fixture.spec.candidates[0].spec.feature_pool[:-1] + (name,))


def assert_reference_fold(fixture, experiment, frame):
    """Compare real worker scores/labels with independently fitted sklearn math."""
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from threadpoolctl import threadpool_limits

    raw = fixture.full_frame.head(52).to_pandas()
    raw.index = pd.RangeIndex(len(raw))
    assignments = pl.read_parquet(fixture.split_path).filter(pl.col("repeat") == 0)
    validation_buildings = assignments.filter(pl.col("fold") == 0)["building_id"].to_list()
    valid_mask = raw["building_id"].isin(validation_buildings)
    assert not set(raw.loc[valid_mask, "building_id"]) & set(raw.loc[~valid_mask, "building_id"])
    pool = list(fixture.spec.candidates[0].spec.feature_pool)
    train, valid = raw.loc[~valid_mask, pool], raw.loc[valid_mask, pool]
    derived_train, derived_valid = reference_features(train, valid, experiment)
    for arm, X_train, X_valid in (("control", train, valid), ("challenger", derived_train, derived_valid)):
        categorical = [name for name in X_train if name == "primary_use" or name.startswith("primary_use_x_")]
        numeric = [name for name in X_train if name not in categorical]
        category = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        reference = Pipeline([
            ("prepare", ColumnTransformer([
                ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
                ("categorical", category, categorical),
            ], remainder="drop", sparse_threshold=0.0)),
            ("model", HistGradientBoostingClassifier(early_stopping=False, random_state=42)),
        ])
        with threadpool_limits(limits=1):
            reference.fit(X_train, raw.loc[X_train.index, "anomaly"])
            expected = reference.predict_proba(X_valid)[:, 1]
        observed = frame.filter((pl.col("arm_id") == arm) & (pl.col("repeat") == 0)
                                & (pl.col("fold") == 0)).sort("row_index")
        assert observed["row_index"].to_list() == X_valid.index.to_list()
        assert observed["target"].to_list() == raw.loc[X_valid.index, "anomaly"].to_list()
        np.testing.assert_allclose(observed["prediction"].to_numpy(), expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("experiment", ROUTES)
def test_actual_admission_and_all12_subprocess_completion_and_resume(tmp_path, experiment):
    fixture = route_fixture(tmp_path / "source", experiment)
    intent = FullCampaignIntent(fixture.spec, True, resource_opt_out_record="user: fixture route only")
    admitted = preflight_full_campaign(fixture.root, intent)
    assert admitted.ready_experiment_ids == (experiment,)
    assert len(admitted.rows) == 21
    root = tmp_path / "run"
    capability = resolve_full_campaign_capability(fixture.root)
    result = run_full_campaign(fixture.spec, fixture.root, root, data_preflight=admitted, capability=capability)
    assert result["status"] == "finished_with_blockers"
    selected = next(row for row in result["inventory"] if row["experiment_id"] == experiment)
    assert selected["status"] == "completed" and selected["n03_eligible"] is True
    frame = pl.read_parquet(root / f"{experiment}_oof.parquet")
    arms = fixture.spec.candidates[0].arms
    assert frame.height == 52 * 3 * len(arms)
    assert frame.select("arm_id", "repeat", "fold").unique().height == len(arms) * 12
    assert frame["n03_eligible"].unique().to_list() == [False]
    assert set(frame["building_id"]) <= set(fixture.full_frame.head(52)["building_id"])
    assert frame["score_kind"].unique().to_list() == ["positive_probability"]
    assert_reference_fold(fixture, experiment, frame)
    with FullFitJournal.open_readonly(root / "journal", fixture.spec, source_root=fixture.root) as journal:
        attempts = journal.attempts()
    assert len(attempts) == 12 * len(arms)
    assert all(attempt.status == "completed" for attempt in attempts)
    receipt = json.loads((root / "completion/receipt.json").read_text())
    assert receipt["verified_model_fits"] == len(attempts)
    assert resume_full_campaign(fixture.spec, fixture.root, root, data_preflight=admitted,
                                capability=capability) == result
    with FullFitJournal.open_readonly(root / "journal", fixture.spec, source_root=fixture.root) as journal:
        assert journal.attempts() == attempts


@pytest.mark.parametrize("experiment", (
    "feat_conditional_cross_fitted_target_encoding",
    "dim_semantic_block_drop",
    "dim_pca_vs_passthrough", "regime_same_family_hpo_vs_default", "model_autogluon_core_features",
))
def test_deferred_routes_are_not_silently_promoted(tmp_path, experiment):
    fixture = route_fixture(tmp_path / "source", "feat_meter_deviation_ratio")
    # HPO requires a finite study even for a rejected declarative snapshot.
    if experiment == "regime_same_family_hpo_vs_default":
        from project.experiments.candidate_contracts import StudySpec
        snapshot = replace(fixture.spec.candidates[0], spec=replace(
            fixture.spec.candidates[0].spec, experiment_id=experiment),
            study=StudySpec(2, 3, 42, (("max_iter", (100, 200)),)), study_method="random_search")
    else:
        snapshot = replace(fixture.spec.candidates[0], spec=replace(
            fixture.spec.candidates[0].spec, experiment_id=experiment))
    spec = replace(fixture.spec, candidates=(snapshot,))
    with pytest.raises(ValueError, match="ordinary_route_not_reviewed"):
        preflight_full_campaign(fixture.root, FullCampaignIntent(
            spec, True, resource_opt_out_record="user: fixture only"))
