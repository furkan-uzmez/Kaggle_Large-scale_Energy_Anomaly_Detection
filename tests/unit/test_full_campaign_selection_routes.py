"""Real selector/semantic fixtures: identities, accounting and sealed completion."""

import json
from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from test_full_campaign_comparison_routes import partition
from test_full_campaign_ordinary_routes import route_fixture

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
from project.experiments.full_run_contracts import FullFitJournal

VARIANTS = tuple("dim_semantic_block_drop_" + block for block in ("calendar", "measurement", "building_metadata"))
SELECTOR = "dim_fold_local_selector_vs_passthrough"
BLOCKS = {"calendar": ("hour", "weekday", "month", "year"), "measurement": ("meter_reading",),
          "building_metadata": ("primary_use", "square_feet", "year_built", "floor_count")}


def independent_fit(train, labels, identity, arm):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_selection import SelectFromModel
    from sklearn.impute import MissingIndicator, SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    selector = identity == SELECTOR
    drop = () if selector or arm == "control" else BLOCKS[identity.removeprefix("dim_semantic_block_drop_")]
    retained = [n for n in train if n not in drop]
    numeric = [n for n in retained if n != "primary_use"]
    branches = []
    if numeric:
        numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler())]) if selector else FeatureUnion([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("indicator", MissingIndicator(features="all", error_on_new=False))])
        branches.append(("numeric", numeric_pipe, numeric))
    if "primary_use" in retained:
        branches.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=selector,
                feature_name_combiner=lambda feature, category: json.dumps([feature, category], separators=(",", ":"))))]),
            ["primary_use"]))
    pre = ColumnTransformer(branches, remainder="drop", sparse_threshold=1. if selector else 0.)
    select = SelectFromModel(LogisticRegression(C=1., l1_ratio=1., solver="liblinear", max_iter=1000,
        class_weight=None, random_state=42), threshold=1e-5) if selector and arm == "challenger" else "passthrough"
    model = LogisticRegression(C=1., l1_ratio=0., solver="lbfgs", class_weight=None,
        max_iter=1000, random_state=42) if selector else HistGradientBoostingClassifier(early_stopping=False, random_state=42)
    pipeline = Pipeline([("pre", pre), ("select", select), ("model", model)])
    pipeline.fit(train, labels)
    return pipeline, tuple(retained)


def selection_fixture(root, selector=False):
    fixture = route_fixture(root, "feat_meter_deviation_ratio")
    base = fixture.spec.candidates[0]
    ids = (SELECTOR,) if selector else VARIANTS
    snapshots = tuple(replace(base, spec=CandidateSpec(identity, base.spec.feature_pool,
        "logistic_l2" if selector else "hist_gradient_boosting", (),
        () if selector else (("drop_block", identity.removeprefix("dim_semantic_block_drop_")),)),
        selector_arms=("challenger",) if selector else ()) for identity in ids)
    return replace(fixture, spec=replace(fixture.spec, candidates=snapshots))


@pytest.mark.parametrize("selector", (False, True))
def test_real_all12_selection_completion_and_verified_resume(tmp_path, selector):
    fixture = selection_fixture(tmp_path / "source", selector)
    admitted = preflight_full_campaign(fixture.root, FullCampaignIntent(fixture.spec, True,
        resource_opt_out_record="user: selection fixture only"))
    ids = (SELECTOR,) if selector else VARIANTS
    assert admitted.ready_experiment_ids == ids
    assert len(admitted.rows) == 21
    root = tmp_path / "run"
    capability = resolve_full_campaign_capability(fixture.root)
    result = run_full_campaign(fixture.spec, fixture.root, root, data_preflight=admitted, capability=capability)
    assert [c["experiment_id"] for c in result["experiments"]] == list(ids)
    assert all(c["status"] == "completed" for c in result["experiments"])
    with FullFitJournal.open_readonly(root / "journal", fixture.spec, source_root=fixture.root) as journal:
        attempts = journal.attempts()
    assert all(a.status == "completed" for a in attempts)
    assert sum(a.unit.endswith("/model") for a in attempts) == (24 if selector else 72)
    assert sum(a.unit.endswith("/selector") for a in attempts) == (12 if selector else 0)
    receipt = json.loads((root / "completion/receipt.json").read_text())
    assert receipt["verified_model_fits"] == (24 if selector else 72)
    assert receipt["verified_selector_fits"] == (12 if selector else 0)
    for identity in ids:
        frame = pl.read_parquet(root / f"{identity}_oof.parquet")
        assert frame.height == 52 * 3 * 2
        assert frame.select("arm_id", "repeat", "fold").unique().height == 24
        assert frame["score_kind"].unique().to_list() == ["positive_probability"]
        assert frame["n03_eligible"].unique().to_list() == [False]
        from threadpoolctl import threadpool_limits
        train, valid, labels, targets = partition(fixture)
        for arm in ("control", "challenger"):
            with threadpool_limits(limits=1):
                reference, retained = independent_fit(train, labels, identity, arm)
                prediction = reference.predict_proba(valid)[:, 1]
            observed = frame.filter((pl.col("arm_id") == arm) & (pl.col("repeat") == 0) & (pl.col("fold") == 0)).sort("row_index")
            assert observed["row_index"].to_list() == valid.index.to_list()
            assert observed["target"].to_list() == targets.to_list()
            np.testing.assert_allclose(observed["prediction"], prediction, rtol=0, atol=1e-14)
            if not selector:
                from project.experiments.full_campaign_capabilities import (
                    build_reviewed_ordinary_arms,
                )
                snapshot = next(s for s in fixture.spec.candidates if s.spec.experiment_id == identity)
                actual = build_reviewed_ordinary_arms(snapshot, year_built_encoding="lead_year_offset_1900_uint8_v1")[arm]
                pre = actual.named_steps["preprocessor"]
                pre.fit(train, labels)
                assert pre.retained_features_ == retained
                for name in pre.registry_.names:
                    assert set(pre.registry_.get(name).source_columns) <= set(retained)
                assert actual.named_steps["model"].get_params() == reference.named_steps["model"].get_params()
            elif arm == "challenger":
                selected = next(a for a in attempts if a.unit == f"{SELECTOR}/challenger/r0/f0/selector")
                model_attempt = next(a for a in attempts if a.unit == f"{SELECTOR}/challenger/r0/f0/model")
                assert attempts.index(selected) < attempts.index(model_attempt)
                outcome = json.loads((root / "journal" / selected.artifact_directory / "selector-outcome.json").read_text())
                np.testing.assert_allclose(outcome["coef"], reference.named_steps["select"].estimator_.coef_, atol=1e-14)
                assert outcome["support"] == reference.named_steps["select"].get_support().tolist()
                assert outcome["training"]["rows"] == len(train)
                assert outcome["training"]["names"] == reference.named_steps["pre"].get_feature_names_out().tolist()
    if not selector:
        row = next(r for r in result["inventory"] if r["experiment_id"] == "dim_semantic_block_drop")
        assert row["status"] == "completed" and row["variant_coverage"] == {v: "completed" for v in VARIANTS}
        hashes = []
        for identity in ids:
            declaration = json.loads((root / f"{identity}_declaration.json").read_text())
            hashes.append(declaration["candidates"][f"{identity}_challenger"]["model_fit_config_sha256"])
            lineage = json.loads((root / f"{identity}_source_lineage.json").read_text())
            assert lineage["source_matrix_id"] == "dim_semantic_block_drop"
            assert lineage["source_cell_id"] == "cand-matrix-render"
            assert lineage["retained_raw_columns"]["challenger"] == [n for n in fixture.spec.candidates[0].spec.feature_pool
                if n not in BLOCKS[identity.removeprefix("dim_semantic_block_drop_")]]
        assert len(set(hashes)) == 3
    else:
        for selected in (a for a in attempts if a.unit.endswith("/selector")):
            directory = root / "journal" / selected.artifact_directory
            request = json.loads((directory / "worker-request.json").read_text())["payload"]
            worker_dir = root / "attempts" / request["attempt_id"]
            names = [json.loads((worker_dir / f"event-{seq:04d}.json").read_text())["payload"]["event"] for seq in range(1, 5)]
            assert names == ["debit", "selector_success", "debit", "shard_closed"]
            ack = json.loads((worker_dir / "ack-0002.json").read_text())["payload"]
            assert ack["status"] == "authorized" and ack["phase"] == "selector"
            assert (worker_dir / "ack-0002.json").stat().st_mtime_ns <= (worker_dir / "event-0003.json").stat().st_mtime_ns
    assert resume_full_campaign(fixture.spec, fixture.root, root, data_preflight=admitted, capability=capability) == result
    with FullFitJournal.open_readonly(root / "journal", fixture.spec, source_root=fixture.root) as journal:
        assert journal.attempts() == attempts
    if selector:
        from project.experiments.full_campaign_evidence import verify_selector_outcome
        directory = root / "journal" / next(a for a in attempts if a.unit.endswith("/selector")).artifact_directory
        request = json.loads((directory / "worker-request.json").read_text())["payload"]
        outcome = json.loads((directory / "selector-outcome.json").read_text())
        corrupt = tmp_path / "corrupt-outcome.json"
        outcome["training"]["labels_sha256"] = "0" * 64
        corrupt.write_text(json.dumps(outcome))
        with pytest.raises(ValueError, match="training/preprocessing"):
            verify_selector_outcome(corrupt, root / "dev_index.sqlite3", fixture.spec, request)
        payload = json.loads((directory / "selector-outcome.json").read_text())
        payload["support"] = [False] * len(payload["support"])
        corrupt.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="coefficient/support"):
            verify_selector_outcome(corrupt, root / "dev_index.sqlite3", fixture.spec, request)


def test_partial_or_misnamed_semantic_family_fails_closed(tmp_path):
    fixture = selection_fixture(tmp_path / "source")
    spec = replace(fixture.spec, candidates=fixture.spec.candidates[:1])
    with pytest.raises(ValueError):
        preflight_full_campaign(fixture.root, FullCampaignIntent(spec, True, resource_opt_out_record="user: fixture only"))


@pytest.mark.parametrize("violation", ("model", "method", "selector_arms", "wrong_block"))
def test_selection_admission_rejects_unreviewed_requests(tmp_path, violation):
    fixture = selection_fixture(tmp_path / "source", selector=violation != "wrong_block")
    snapshot = fixture.spec.candidates[0]
    if violation == "model":
        snapshot = replace(snapshot, spec=replace(snapshot.spec, model_params=(("C", 2.),)))
    elif violation == "method":
        snapshot = replace(snapshot, spec=replace(snapshot.spec, method_params=(("threshold", 0.),)))
    elif violation == "selector_arms":
        snapshot = replace(snapshot, selector_arms=())
    else:
        snapshot = replace(snapshot, spec=replace(snapshot.spec, method_params=(("drop_block", "measurement"),)))
    spec = replace(fixture.spec, candidates=(snapshot, *fixture.spec.candidates[1:]))
    with pytest.raises(ValueError):
        preflight_full_campaign(fixture.root, FullCampaignIntent(spec, True, resource_opt_out_record="user: fixture rejection"))
