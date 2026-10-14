"""New reviewed intent configs use real frozen factories; old intent stays ordinary."""
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from project.experiments.full_campaign_cli import _candidate
from project.experiments.full_campaign_capabilities import build_reviewed_route_arms, route_for_experiments
from project.experiments.full_nested_tuning import full_hpo_study_spec
from project.experiments.full_snapshot_codec import configuration_digest
from project.experiments.candidate_dispatch import _configuration

ROOT = Path(__file__).resolve().parents[2]
ENCODING = "lead_year_offset_1900_uint8_v1"


@pytest.mark.parametrize("slug,identity,score,fits", [
    ("target_encoding", "feat_conditional_cross_fitted_target_encoding", "positive_probability", 24),
    ("pca_rbf", "dim_pca_vs_passthrough", "decision_margin", 24),
    ("hpo", "regime_same_family_hpo_vs_default", "positive_probability", 312),
    ("autogluon", "model_autogluon_core_features", "positive_probability", 12),
    ("provided", "feat_core_vs_provided", "positive_probability", 24),
])
def test_reviewed_config_resolves_real_exact_factory(slug, identity, score, fits):
    document = yaml.safe_load((ROOT / f"configs/experiments/{slug}_full_preflight_v2.yaml").read_text())
    assert document["sources"]["notebook"] == "notebooks/02_baseline_and_candidate_design.ipynb"
    assert len(document["candidates"]) == 1
    snapshot = _candidate(document["candidates"][0])
    assert snapshot.spec.experiment_id == identity
    assert snapshot.source_cell_id == "cand-matrix-render"
    arms = build_reviewed_route_arms(snapshot, year_built_encoding=ENCODING)
    assert tuple(arms) == snapshot.arms
    assert {arm.score_kind for arm in arms.values()} == {score}
    assert len(snapshot.planned_units) == fits
    assert all(len(configuration_digest(_configuration(arm.transformer_or_pipeline))) == 64 for arm in arms.values())
    if slug == "hpo":
        assert snapshot.study == full_hpo_study_spec()
        assert snapshot.study.n_trials * 12 == 96
    if slug == "autogluon":
        assert snapshot.arms == ("challenger",)
        assert snapshot.spec.model_params == ()
        assert document["sources"]["autogluon_profile"] == "docs/evidence/autogluon_full_profile_v1.json"


@pytest.mark.parametrize("changes", [{"n_trials": 7}, {"seed": 43}, {"inner_folds": 4}])
def test_hpo_changed_finite_study_rejected_before_dispatch(changes):
    document = yaml.safe_load((ROOT / "configs/experiments/hpo_full_preflight_v2.yaml").read_text())
    snapshot = _candidate(document["candidates"][0])
    with pytest.raises(ValueError, match="exact|frozen|approved"):
        build_reviewed_route_arms(replace(snapshot, study=replace(snapshot.study, **changes)), year_built_encoding=ENCODING)


def test_new_routes_never_extend_historical_ordinary_intent():
    document = yaml.safe_load((ROOT / "configs/experiments/candidate_full_v2.yaml").read_text())
    identities = tuple(row["experiment_id"] for row in document["candidates"])
    assert len(identities) == 14
    assert route_for_experiments(identities) == "reviewed_ordinary_transport_codec_v1"
