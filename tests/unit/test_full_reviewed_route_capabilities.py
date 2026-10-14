"""Reviewed routes are selected explicitly, never widened ordinary permission."""
from dataclasses import replace
from pathlib import Path

import pytest

from project.experiments import full_campaign_capabilities as capabilities
from project.experiments.candidate_contracts import CandidateSpec
from project.experiments.full_run_contracts import FullCandidateSnapshot
from project.experiments.full_input_schema import CORE

ROOT = Path(__file__).resolve().parents[2]
TE = "feat_conditional_cross_fitted_target_encoding"
PCA = "dim_pca_vs_passthrough"
ENCODING = "lead_year_offset_1900_uint8_v1"


def snapshot(identity):
    methods = (("smoothing", 10.0), ("n_splits", 3), ("seed", 42), ("max_categories", 4096)) if identity == TE else (("pca_components", .95),)
    return FullCandidateSnapshot(CandidateSpec(identity, CORE,
        "hist_gradient_boosting" if identity == TE else "svm", (), methods),
        ("control", "challenger"), "notebooks/02_baseline_and_candidate_design.ipynb", "cand-matrix-render")


@pytest.mark.parametrize("identity", [TE, PCA, "regime_same_family_hpo_vs_default", "model_autogluon_core_features"])
def test_explicit_recipe_selects_separate_authenticated_capability(identity):
    selected = capabilities.resolve_full_campaign_capability(ROOT, experiment_ids=(identity,))
    ordinary = capabilities.resolve_full_campaign_capability(ROOT)
    assert selected.available
    assert selected.route_id != ordinary.route_id
    assert selected.supported_experiment_ids == (identity,)
    capabilities.verify_full_campaign_capability(ROOT, selected)
    assert ordinary.supported_experiment_ids == capabilities.ORDINARY_EXPERIMENTS


def test_mixed_new_routes_fail_before_route_authority():
    with pytest.raises(ValueError, match="mixed|unsupported"):
        capabilities.resolve_full_campaign_capability(ROOT, experiment_ids=(TE, PCA))


def test_exact_pca_graph_retains_margin_semantics():
    builder = capabilities.build_reviewed_route_arms
    arms = builder(snapshot(PCA), year_built_encoding=ENCODING)
    assert tuple(arms) == ("control", "challenger")
    assert all(arm.score_kind == "decision_margin" for arm in arms.values())
    assert not any(hasattr(arm.named_steps["model"], "classes_") for arm in arms.values())
    with pytest.raises(ValueError, match="frozen|exact|reviewed"):
        builder(replace(snapshot(PCA), spec=replace(snapshot(PCA).spec,
            method_params=(("pca_components", .9),))), year_built_encoding=ENCODING)


def test_exact_te_graph_rejects_numeric_type_coercion():
    builder = capabilities.build_reviewed_route_arms
    source = snapshot(TE)
    arms = builder(source, year_built_encoding=ENCODING)
    assert all(arm.score_kind == "positive_probability" for arm in arms.values())
    with pytest.raises(ValueError, match="reviewed|frozen"):
        builder(replace(source, spec=replace(source.spec,
            method_params=(("smoothing", 10),) + source.spec.method_params[1:])), year_built_encoding=ENCODING)


def test_selected_route_source_mutation_invalidates_capability(tmp_path):
    import shutil
    for binding in capabilities.resolve_full_campaign_capability(ROOT).implementation_sources:
        target = tmp_path / binding.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / binding.path, target)
    selected = capabilities.resolve_full_campaign_capability(tmp_path, experiment_ids=(PCA,))
    assert selected.available
    source = tmp_path / "src/project/experiments/full_rbf_svc_worker.py"
    source.write_text(source.read_text() + "\n# changed source\n")
    with pytest.raises(ValueError, match="source|binding|hash"):
        capabilities.verify_full_campaign_capability(tmp_path, selected)
