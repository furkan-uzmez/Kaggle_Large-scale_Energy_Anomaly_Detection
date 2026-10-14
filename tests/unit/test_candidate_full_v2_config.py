"""Static snapshot contracts only; population binding/admission belongs to parent."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from project.experiments import full_campaign_cli
from project.experiments.full_campaign_admission import (
    MATRIX_IDS,
    FullCampaignIntent,
    inspect_full_campaign,
)
from project.experiments.full_campaign_capabilities import (
    ORDINARY_EXPERIMENTS,
    build_reviewed_ordinary_arms,
)
from project.experiments.full_campaign_preflight import current_environment
from project.experiments.full_run_contracts import (
    SEMANTIC_VARIANTS,
    SourceBinding,
    source_matrix_identity,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/candidate_full_v2.yaml"
SELECTED = (
    "feat_missingness_indicators",
    "feat_categorical_interactions",
    "feat_frequency_encoding",
    "feat_meter_deviation_ratio",
    "feat_histogram_quantile",
    "feat_missingness_bitmask",
    "feat_safe_calendar_numeric_transforms",
    "dim_semantic_block_drop_calendar",
    "dim_semantic_block_drop_measurement",
    "dim_semantic_block_drop_building_metadata",
    "dim_fold_local_selector_vs_passthrough",
    "regime_natural_vs_class_weighted",
    "feat_missingness_passthrough_comparison",
    "model_extratrees_core_features",
)
CORE = (
    "meter_reading", "square_feet", "year_built", "floor_count",
    "hour", "weekday", "month", "year", "primary_use",
)
NOTEBOOK = "notebooks/02_baseline_and_candidate_design.ipynb"
ENCODING = "docs/evidence/year_built_encoding_v1.json"


@pytest.fixture
def document():
    return yaml.safe_load(CONFIG.read_text())


def test_reviewed_v2_snapshot_exists():
    assert CONFIG.is_file(), "The reviewed ordinary-route v2 snapshot is missing"


@pytest.fixture
def static_spec(document, monkeypatch):
    """Real schema/source audit; explicitly unverified population test bindings."""
    original_open = Path.open

    def refuse_population_open(path, *args, **kwargs):
        relative = str(path)
        assert "outputs/prepared/" not in relative
        assert "outputs/splits/" not in relative
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refuse_population_open)

    def bind_static_sources(entries, root):
        assert set(entries) == set(full_campaign_cli.SOURCE_ROLES)
        bindings = []
        for role in full_campaign_cli.SOURCE_ROLES:
            relative = entries[role]
            if role in ("data", "split"):
                # Test-only digest of a string, not claimed population evidence.
                payload = ("unverified-test-binding:" + relative).encode()
            else:
                payload = (root / relative).read_bytes()
            bindings.append(SourceBinding(role, relative, hashlib.sha256(payload).hexdigest()))
        return tuple(bindings)

    monkeypatch.setattr(full_campaign_cli, "_bind_sources", bind_static_sources)
    return full_campaign_cli.load_full_spec(CONFIG, ROOT)


def test_snapshot_selects_only_current_reviewed_routes_in_matrix_order(document, static_spec):
    assert tuple(entry["experiment_id"] for entry in document["candidates"]) == SELECTED
    assert set(SELECTED) <= set(ORDINARY_EXPERIMENTS)
    assert tuple(identity for name in MATRIX_IDS for identity in (
        SEMANTIC_VARIANTS if name == "dim_semantic_block_drop" else (name,)) if identity in SELECTED) == SELECTED
    assert static_spec.run_id == "candidate_full_v2"
    assert document["run_directory"] == "outputs/campaigns/candidate_full_v2"
    assert len(static_spec.planned_units) == 360
    assert sum(unit.endswith("/selector") for unit in static_spec.planned_units) == 12
    assert sum(unit.endswith("/model") for unit in static_spec.planned_units) == 348
    assert static_spec.planned_trials == 0
    assert (static_spec.folds, static_spec.repeats, static_spec.split_seeds) == (4, 3, (42, 43, 44))
    assert (static_spec.estimator_seed, static_spec.positive_label) == (42, 1)
    assert (static_spec.development_rows, static_spec.vault_rows) == (1399631, 349863)


def test_exact_source_paths_environment_and_full_campaign_opt_out(document, static_spec):
    assert document["sources"] == {
        "matrix": "configs/experiments/feature_campaign.yaml",
        "baseline": "configs/experiments/baselines.yaml",
        "data": "outputs/prepared/train_joined.parquet",
        "split": "outputs/splits/building_folds_repeated.parquet",
        "producer": "src/project/features/registry.py",
        "notebook": NOTEBOOK,
        "encoding": ENCODING,
    }
    assert static_spec.environment == current_environment()
    record = document["resource_opt_out_record"]
    assert record == (
        "docs/superpowers/specs/2026-10-04-sequential-full-campaign-design.md"
        "#scope-and-informed-resource-opt-out"
    )
    authorization = (ROOT / record.split("#")[0]).read_text()
    assert "The user explicitly opts out" in authorization
    assert "RAM/RSS, CPU-thread and wall-time caps" in authorization
    assert asdict(static_spec.resource_policy) == {
        "informed_opt_out": True, "rss_limit_enforced": False, "max_rss_bytes": None,
        "cpu_limit_enforced": False, "cpu_threads": None,
        "wall_limit_enforced": False, "max_wall_seconds": None,
    }


def test_candidates_have_authentic_provenance_and_frozen_unfitted_arms(static_spec):
    notebook = json.loads((ROOT / NOTEBOOK).read_text())
    cells = [cell for cell in notebook["cells"] if cell.get("id") == "cand-matrix-render"]
    assert len(cells) == 1
    assert "campaign['experiment_matrix']" in "".join(cells[0]["source"])
    encoding = json.loads((ROOT / ENCODING).read_text())["encoding_id"]
    assert encoding == "lead_year_offset_1900_uint8_v1"
    for snapshot in static_spec.candidates:
        assert snapshot.source_notebook == NOTEBOOK
        assert snapshot.source_cell_id == "cand-matrix-render"
        assert snapshot.spec.feature_pool == CORE
        trees = snapshot.spec.experiment_id == "model_extratrees_core_features"
        histogram = snapshot.spec.experiment_id == "feat_histogram_quantile"
        selector = snapshot.spec.experiment_id == "dim_fold_local_selector_vs_passthrough"
        semantic = snapshot.spec.experiment_id in SEMANTIC_VARIANTS
        assert snapshot.spec.model_family == ("logistic_l2" if selector else "extra_trees" if trees else "hist_gradient_boosting")
        assert dict(snapshot.spec.model_params) == ({"n_estimators": 64, "max_depth": 12,
            "min_samples_leaf": 2, "max_features": "sqrt", "bootstrap": False, "class_weight": None,
            "n_jobs": 1, "random_state": 42} if trees else {"C": 1., "l1_ratio": 0., "solver": "lbfgs",
                "class_weight": None, "max_iter": 1000, "random_state": 42} if selector else {"early_stopping": False, "random_state": 42})
        assert dict(snapshot.spec.method_params) == ({"n_bins": 8, "n_quantiles": 64} if histogram else {
            "drop_block": snapshot.spec.experiment_id.removeprefix("dim_semantic_block_drop_")} if semantic else {})
        assert snapshot.arms == (("control", "bin", "quantile") if histogram else ("control", "challenger"))
        assert snapshot.selector_arms == (("challenger",) if selector else ())
        assert snapshot.study is None and snapshot.study_method is None
        arms = build_reviewed_ordinary_arms(snapshot, year_built_encoding=encoding)
        assert tuple(arms) == snapshot.arms
        for arm in arms.values():
            model = arm.named_steps["model"]
            if trees and arm.arm_id == "challenger":
                assert model.n_estimators == 64 and model.max_depth == 12 and model.n_jobs == 1
            elif not selector:
                assert model.early_stopping is False
            assert model.random_state == 42
            assert not hasattr(model, "classes_")


def test_source_only_inventory_accounts_for_all_rows_without_claiming_readiness(static_spec, document):
    # The legacy source inspector deliberately excludes the encoding extension.
    inspection_spec = replace(static_spec, sources=tuple(
        source for source in static_spec.sources if source.role != "encoding"))
    report = inspect_full_campaign(ROOT, FullCampaignIntent(
        inspection_spec, True, resource_opt_out_record=document["resource_opt_out_record"]))
    assert report.blocker_codes == ()
    assert tuple(row.record.experiment_id for row in report.rows) == MATRIX_IDS
    assert len(report.rows) == 21
    assert report.execution_authorized is False and report.n03_eligible is False
    for row in report.rows:
        assert row.record.n03_eligible is False
        if row.record.experiment_id in {source_matrix_identity(identity) for identity in SELECTED}:
            assert row.record.status == "pending_implementation"
            assert row.record.blocker_codes == ("full_dispatch_unavailable", "real_preflight_pending")
            assert row.planned_model_fits == (72 if row.record.experiment_id == "dim_semantic_block_drop" else
                36 if row.record.experiment_id == "feat_histogram_quantile" else 24)
            assert row.planned_selector_fits == (12 if row.record.experiment_id == "dim_fold_local_selector_vs_passthrough" else 0)
            assert row.planned_trials == 0
        elif row.record.experiment_id == "prohibited_opaque_building_id_digit_extraction":
            assert row.record.status == "prohibited"
        else:
            assert row.record.status == "blocked"
            assert "candidate_configuration_pending" in row.record.blocker_codes
