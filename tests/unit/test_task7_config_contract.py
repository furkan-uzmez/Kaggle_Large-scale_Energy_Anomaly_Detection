"""Task 7 Config Contract Tests.

TDD unit tests for the design-only YAML schema files introduced in Task 7:
  - configs/resource_budget.yaml
  - configs/experiments/feature_campaign.yaml

These tests DO NOT execute training, HPO, model fits, or feature generation.
They only verify structural invariants and cross-references.

Run:
    ./.venv/bin/python -m pytest tests/unit/test_task7_config_contract.py -q
"""

from __future__ import annotations

import pathlib
import tomllib
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).parents[2]
RESOURCE_BUDGET_PATH = PROJECT_ROOT / "configs" / "resource_budget.yaml"
FEATURE_CAMPAIGN_PATH = PROJECT_ROOT / "configs" / "experiments" / "feature_campaign.yaml"
BASELINES_PATH = PROJECT_ROOT / "configs" / "experiments" / "baselines.yaml"

# Task 6 contract constants (frozen).
TASK6_TARGET = "anomaly"
TASK6_POSITIVE_LABEL = 1
TASK6_PRIMARY_METRIC = "roc_auc"
TASK6_GUARDRAILS = {"average_precision", "log_loss", "brier_score"}
TASK6_FOLDS = 4
TASK6_SEED = 42
TASK6_SPLIT_MANIFEST = "outputs/splits/building_folds.parquet"


def test_pyyaml_is_declared_as_direct_runtime_dependency() -> None:
    """Project code imports YAML directly, so PyYAML must not be transitive-only."""
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    direct_names = {
        dependency.split(">", 1)[0].split("<", 1)[0].split("=", 1)[0].strip().lower()
        for dependency in dependencies
    }
    assert "pyyaml" in direct_names, (
        "PyYAML must be declared directly because scripts and notebooks import yaml"
    )

# Required model families for audit (exactly these 8).
REQUIRED_MODEL_FAMILIES = {
    "extra_trees",
    "random_forest",
    "hist_gradient_boosting",
    "ebm_gam",
    "svm",
    "knn",
    "tabular_neural_net",
    "autogluon",
}

# Required keys for each model_family_audit entry.
MODEL_AUDIT_REQUIRED_KEYS = {
    "applicable",
    "decision_reason",
    "resource_estimate",
    "compatible_recipes",
    "planned_config",
    "execution_status",
    "oof_artifact",
    "rejection_reason",
}

# Required categories for experiment_matrix entries.
EXPERIMENT_MATRIX_REQUIRED_KEYS = {
    "id",
    "status",
    "execution_status",
}


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    """Load a YAML file and assert it is a mapping."""
    assert path.exists(), f"YAML file must exist at {path}"
    with path.open() as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), f"{path.name} must parse as a YAML mapping, got {type(data)}"
    return data


# ===========================================================================
# Resource Budget Tests
# ===========================================================================


class TestResourceBudgetExists:
    """configs/resource_budget.yaml structural invariants."""

    def test_file_exists(self):
        assert RESOURCE_BUDGET_PATH.exists(), (
            f"configs/resource_budget.yaml must exist at {RESOURCE_BUDGET_PATH}"
        )

    def test_parses_as_mapping(self):
        _load_yaml(RESOURCE_BUDGET_PATH)

    def test_schema_version_is_1(self):
        cfg = _load_yaml(RESOURCE_BUDGET_PATH)
        assert cfg.get("schema_version") == 1, (
            f"schema_version must be 1, got {cfg.get('schema_version')!r}"
        )

    def test_campaign_id_is_task7_candidate_design(self):
        cfg = _load_yaml(RESOURCE_BUDGET_PATH)
        assert cfg.get("campaign_id") == "task7_candidate_design", (
            f"campaign_id must be 'task7_candidate_design', got {cfg.get('campaign_id')!r}"
        )

    def test_mode_is_design_only(self):
        cfg = _load_yaml(RESOURCE_BUDGET_PATH)
        assert cfg.get("mode") == "design_only", (
            f"mode must be 'design_only', got {cfg.get('mode')!r}"
        )

    def test_execution_not_authorized(self):
        cfg = _load_yaml(RESOURCE_BUDGET_PATH)
        assert cfg.get("execution_authorized") is False, (
            f"execution_authorized must be false, got {cfg.get('execution_authorized')!r}"
        )

    def test_candidate_target_is_finite(self):
        cfg = _load_yaml(RESOURCE_BUDGET_PATH)
        target = cfg.get("candidate_target")
        assert isinstance(target, int) and target > 0, (
            f"candidate_target must be a positive integer, got {target!r}"
        )

    def test_candidate_cap_is_finite(self):
        cfg = _load_yaml(RESOURCE_BUDGET_PATH)
        cap = cfg.get("candidate_cap")
        assert isinstance(cap, int) and cap > 0, (
            f"candidate_cap must be a positive integer, got {cap!r}"
        )

    def test_candidate_target_equals_candidate_cap(self):
        cfg = _load_yaml(RESOURCE_BUDGET_PATH)
        assert cfg.get("candidate_target") == cfg.get("candidate_cap"), (
            "candidate_target and candidate_cap must be equal per Task 7 spec (both 100)"
        )

    def test_candidate_cap_is_100(self):
        cfg = _load_yaml(RESOURCE_BUDGET_PATH)
        assert cfg.get("candidate_cap") == 100, (
            f"candidate_cap must be 100, got {cfg.get('candidate_cap')!r}"
        )

    @pytest.mark.parametrize("cap_key", [
        "maximum_cardinality",
        "max_ram_gb",
        "max_wall_time_seconds",
        "max_model_fits",
        "max_hpo_trials",
        "max_seed_count",
    ])
    def test_resource_cap_is_present_but_null(self, cap_key: str):
        cfg = _load_yaml(RESOURCE_BUDGET_PATH)
        assert cap_key in cfg, (
            f"resource cap key '{cap_key}' must be present in resource_budget.yaml"
        )
        assert cfg[cap_key] is None, (
            f"resource cap '{cap_key}' must be null until measurement/approval, "
            f"got {cfg[cap_key]!r}"
        )

    def test_host_preflight_status_is_pending(self):
        cfg = _load_yaml(RESOURCE_BUDGET_PATH)
        assert cfg.get("host_preflight_status") == "pending", (
            f"host_preflight_status must be 'pending', got {cfg.get('host_preflight_status')!r}"
        )

    def test_user_approval_status_is_pending(self):
        cfg = _load_yaml(RESOURCE_BUDGET_PATH)
        assert cfg.get("user_approval_status") == "pending", (
            f"user_approval_status must be 'pending', got {cfg.get('user_approval_status')!r}"
        )

    def test_stop_rules_present(self):
        cfg = _load_yaml(RESOURCE_BUDGET_PATH)
        assert "stop_rules" in cfg, "stop_rules section must be present in resource_budget.yaml"
        stop_rules = cfg["stop_rules"]
        assert isinstance(stop_rules, (list, dict)), (
            "stop_rules must be a list or mapping"
        )

    def test_stop_rules_prohibit_execution_while_pending(self):
        """Stop rules must contain at least one rule referencing pending approval/caps."""
        cfg = _load_yaml(RESOURCE_BUDGET_PATH)
        stop_rules = cfg.get("stop_rules", [])
        # Convert to string for simple substring search regardless of structure.
        text = yaml.safe_dump(stop_rules)
        assert any(
            kw in text for kw in ("pending", "blocked", "caps", "approval")
        ), "stop_rules must reference pending approval or caps blocking execution"

    def test_no_padding_requirement_in_stop_rules(self):
        """Stop rules must explicitly prohibit padding the candidate count."""
        cfg = _load_yaml(RESOURCE_BUDGET_PATH)
        stop_rules = cfg.get("stop_rules", [])
        text = yaml.safe_dump(stop_rules)
        assert "pad" in text or "padding" in text or "never pad" in text.lower(), (
            "stop_rules must explicitly prohibit padding the candidate count"
        )

    def test_preserve_grouped_folds_vault_in_stop_rules(self):
        """Stop rules must reference preserving grouped folds/vault."""
        cfg = _load_yaml(RESOURCE_BUDGET_PATH)
        stop_rules = cfg.get("stop_rules", [])
        text = yaml.safe_dump(stop_rules)
        assert any(kw in text for kw in ("fold", "vault", "group")), (
            "stop_rules must reference preserving grouped folds or vault"
        )


# ===========================================================================
# Feature Campaign Tests
# ===========================================================================


class TestFeatureCampaignExists:
    """configs/experiments/feature_campaign.yaml structural invariants."""

    def test_file_exists(self):
        assert FEATURE_CAMPAIGN_PATH.exists(), (
            f"configs/experiments/feature_campaign.yaml must exist at {FEATURE_CAMPAIGN_PATH}"
        )

    def test_parses_as_mapping(self):
        _load_yaml(FEATURE_CAMPAIGN_PATH)

    def test_schema_version_present(self):
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        assert "schema_version" in cfg, "schema_version must be present"
        assert isinstance(cfg["schema_version"], int), "schema_version must be an integer"

    def test_campaign_id_present(self):
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        assert "campaign_id" in cfg, "campaign_id must be present"
        assert isinstance(cfg["campaign_id"], str), "campaign_id must be a string"

    def test_mode_is_design_only(self):
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        assert cfg.get("mode") == "design_only", (
            f"mode must be 'design_only', got {cfg.get('mode')!r}"
        )

    def test_execution_not_authorized(self):
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        assert cfg.get("execution_authorized") is False, (
            f"execution_authorized must be false, got {cfg.get('execution_authorized')!r}"
        )

    def test_baseline_config_reference_present(self):
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        assert "baseline_config" in cfg, "baseline_config reference must be present"
        assert isinstance(cfg["baseline_config"], str), "baseline_config must be a string path"

    def test_baseline_config_points_to_baselines_yaml(self):
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        ref = cfg.get("baseline_config", "")
        assert "baselines.yaml" in ref, (
            f"baseline_config must reference baselines.yaml, got {ref!r}"
        )

    def test_resource_budget_reference_present(self):
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        assert "resource_budget" in cfg, "resource_budget reference must be present"

    def test_resource_budget_reference_points_to_resource_budget_yaml(self):
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        ref = cfg.get("resource_budget", "")
        assert "resource_budget.yaml" in str(ref), (
            f"resource_budget must reference resource_budget.yaml, got {ref!r}"
        )


class TestFeatureCampaignPredictionContract:
    """prediction_contract must match the Task 6 frozen contract."""

    def _contract(self) -> dict[str, Any]:
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        assert "prediction_contract" in cfg, (
            "prediction_contract section must be present in feature_campaign.yaml"
        )
        return cfg["prediction_contract"]

    def test_target_matches_task6(self):
        contract = self._contract()
        assert contract.get("target") == TASK6_TARGET, (
            f"prediction_contract.target must be '{TASK6_TARGET}', got {contract.get('target')!r}"
        )

    def test_positive_label_matches_task6(self):
        contract = self._contract()
        assert contract.get("positive_label") == TASK6_POSITIVE_LABEL, (
            f"prediction_contract.positive_label must be {TASK6_POSITIVE_LABEL}, "
            f"got {contract.get('positive_label')!r}"
        )

    def test_primary_metric_matches_task6(self):
        contract = self._contract()
        assert contract.get("primary_metric") == TASK6_PRIMARY_METRIC, (
            f"prediction_contract.primary_metric must be '{TASK6_PRIMARY_METRIC}', "
            f"got {contract.get('primary_metric')!r}"
        )

    def test_guardrails_match_task6(self):
        contract = self._contract()
        guardrails = set(contract.get("guardrails", []))
        assert guardrails == TASK6_GUARDRAILS, (
            f"prediction_contract.guardrails must be {TASK6_GUARDRAILS}, got {guardrails}"
        )

    def test_split_manifest_present(self):
        contract = self._contract()
        assert "split_manifest" in contract, (
            "prediction_contract.split_manifest must be present"
        )
        assert TASK6_SPLIT_MANIFEST in str(contract["split_manifest"]), (
            f"split_manifest must reference '{TASK6_SPLIT_MANIFEST}', "
            f"got {contract.get('split_manifest')!r}"
        )

    def test_folds_match_task6(self):
        contract = self._contract()
        assert contract.get("folds") == TASK6_FOLDS, (
            f"prediction_contract.folds must be {TASK6_FOLDS}, got {contract.get('folds')!r}"
        )

    def test_no_duplicate_seed_in_campaign(self):
        """Seed must not be redefined in feature_campaign.yaml (reference baseline_config instead)."""
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        # Seed at top level is forbidden; must not appear directly under campaign root
        # (it belongs exclusively to baselines.yaml).
        assert "seed" not in cfg, (
            "feature_campaign.yaml must not define 'seed' at the top level; "
            "it is inherited from the baseline_config reference"
        )

    def test_no_duplicate_feature_list_in_campaign(self):
        """Core feature lists must not be duplicated; rely on baseline_config reference."""
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        # Top-level 'features' key is forbidden (avoid duplication).
        assert "features" not in cfg, (
            "feature_campaign.yaml must not define 'features' at the top level; "
            "core predictors are referenced from baseline_config"
        )

    def test_no_seed_in_prediction_contract(self):
        """Seed must not appear inside prediction_contract."""
        contract = self._contract()
        assert "seed" not in contract, (
            "prediction_contract must not contain a 'seed' key; "
            "seed is inherited from the baseline_config"
        )


class TestFeatureCampaignFeatureSets:
    """feature_sets referencing correct data sources."""

    def _feature_sets(self) -> dict[str, Any]:
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        assert "feature_sets" in cfg, "feature_sets section must be present"
        return cfg["feature_sets"]

    def test_feature_sets_present(self):
        fs = self._feature_sets()
        assert isinstance(fs, (dict, list)), "feature_sets must be a mapping or list"

    def test_core_predictors_reference_baseline_config(self):
        """Core predictors must reference baseline_config, not duplicate the list."""
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        fs = cfg.get("feature_sets", {})
        text = yaml.safe_dump(fs)
        assert "baseline_config" in text or "baselines.yaml" in text, (
            "feature_sets must reference baseline_config for core predictors"
        )

    def test_provided_features_reference_train_joined_parquet(self):
        """Provided-feature comparison must reference the correct data path."""
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        fs = cfg.get("feature_sets", {})
        text = yaml.safe_dump(fs)
        assert "train_joined.parquet" in text, (
            "feature_sets must reference 'outputs/prepared/train_joined.parquet' "
            "for provided-feature comparison"
        )


class TestFeatureCampaignModelFamilyAudit:
    """model_family_audit must have all required entries with correct fields."""

    def _audit(self) -> dict[str, Any]:
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        assert "model_family_audit" in cfg, "model_family_audit must be present"
        audit = cfg["model_family_audit"]
        assert isinstance(audit, dict), "model_family_audit must be a mapping"
        return audit

    def test_all_required_families_present(self):
        audit = self._audit()
        present = set(audit.keys())
        missing = REQUIRED_MODEL_FAMILIES - present
        assert not missing, (
            f"model_family_audit is missing required families: {missing}"
        )

    @pytest.mark.parametrize("family", sorted(REQUIRED_MODEL_FAMILIES))
    def test_family_has_all_required_keys(self, family: str):
        audit = self._audit()
        entry = audit.get(family, {})
        assert isinstance(entry, dict), f"model_family_audit[{family!r}] must be a mapping"
        missing_keys = MODEL_AUDIT_REQUIRED_KEYS - set(entry.keys())
        assert not missing_keys, (
            f"model_family_audit[{family!r}] is missing required keys: {missing_keys}"
        )

    def test_hist_gradient_boosting_references_baseline_config(self):
        """HGB must reference baseline_config but have null OOF artifact (no comparative OOF yet)."""
        audit = self._audit()
        hgb = audit.get("hist_gradient_boosting", {})
        text = yaml.safe_dump(hgb)
        assert "baseline_config" in text or "baselines.yaml" in text, (
            "hist_gradient_boosting audit entry must reference the Task 6 baseline config"
        )
        assert hgb.get("oof_artifact") is None, (
            "hist_gradient_boosting.oof_artifact must be null (no comparative OOF artifact yet)"
        )

    def test_runnable_but_unapproved_families_are_blocked_unapproved_compute(self):
        """Families that are runnable but lack resource approval must be blocked_unapproved_compute."""
        audit = self._audit()
        for family, entry in audit.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("applicable") is True:
                status = entry.get("execution_status")
                # Runnable families with null resource estimates must be blocked.
                if entry.get("resource_estimate") is None:
                    assert status == "blocked_unapproved_compute", (
                        f"model_family_audit[{family!r}]: applicable=true with null "
                        f"resource_estimate must have execution_status='blocked_unapproved_compute', "
                        f"got {status!r}"
                    )

    def test_na_families_have_rejection_reason(self):
        """Families marked applicable=false (N/A) must provide a rejection_reason."""
        audit = self._audit()
        for family, entry in audit.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("applicable") is False:
                reason = entry.get("rejection_reason")
                assert reason is not None and str(reason).strip(), (
                    f"model_family_audit[{family!r}]: applicable=false requires a non-null "
                    f"rejection_reason, got {reason!r}"
                )

    def test_no_runnable_family_marked_na_only_for_resource_reasons(self):
        """No runnable model family may be discarded simply because resources/deps are unapproved."""
        audit = self._audit()
        for family, entry in audit.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("applicable") is False:
                reason = str(entry.get("rejection_reason", "")).lower()
                # These are architectural/compatibility rejections, not resource ones.
                # If rejection_reason mentions only resource/approval, flag it.
                resource_only = (
                    reason
                    and all(
                        kw in reason
                        for kw in ("resource",)
                    )
                    and not any(
                        kw in reason
                        for kw in (
                            "incompatible",
                            "not supported",
                            "architecture",
                            "scalability",
                            "not applicable",
                            "memory",
                            "runtime",
                            "cost",
                            "dependency",
                        )
                    )
                )
                # Pure resource-only rejections are disallowed; should be blocked instead.
                assert not resource_only, (
                    f"model_family_audit[{family!r}]: rejection_reason appears to be purely "
                    f"resource-based. Use applicable=true + execution_status='blocked_unapproved_compute' "
                    f"instead of applicable=false for resource constraints."
                )

    def test_applicable_runnable_families_have_null_oof_prior_to_execution(self):
        """Unexecuted runnable families must have null oof_artifact."""
        audit = self._audit()
        for family, entry in audit.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("applicable") is True and entry.get("execution_status") in (
                "blocked_unapproved_compute",
                "planned",
                "design_only",
            ):
                assert entry.get("oof_artifact") is None, (
                    f"model_family_audit[{family!r}]: has execution_status="
                    f"{entry.get('execution_status')!r} but non-null oof_artifact "
                    f"= {entry.get('oof_artifact')!r}"
                )

    def test_applicable_runnable_families_have_null_resource_estimate(self):
        """All blocked_unapproved_compute families must have null resource_estimate."""
        audit = self._audit()
        for family, entry in audit.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("execution_status") == "blocked_unapproved_compute":
                assert entry.get("resource_estimate") is None, (
                    f"model_family_audit[{family!r}]: blocked_unapproved_compute families "
                    f"must have null resource_estimate, got {entry.get('resource_estimate')!r}"
                )


class TestFeatureCampaignContextBaselines:
    """context_baselines must include the required baseline categories with correct rationale."""

    def _baselines(self) -> dict[str, Any]:
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        assert "context_baselines" in cfg, "context_baselines must be present"
        cb = cfg["context_baselines"]
        assert isinstance(cb, dict), "context_baselines must be a mapping"
        return cb

    def test_business_incumbent_present(self):
        cb = self._baselines()
        assert "business_incumbent" in cb or "incumbent" in cb, (
            "context_baselines must include business/incumbent baseline"
        )

    def test_temporal_naive_present(self):
        cb = self._baselines()
        assert "temporal_naive" in cb, (
            "context_baselines must include temporal_naive"
        )

    def test_temporal_naive_na_has_justified_reason(self):
        """temporal_naive N/A must be justified by unseen-building/no-label-history rationale."""
        cb = self._baselines()
        tn = cb.get("temporal_naive", {})
        if not isinstance(tn, dict):
            return
        if tn.get("applicable") is False or tn.get("status") in ("N/A", "na", "not_applicable"):
            reason = str(tn.get("rationale", tn.get("reason", ""))).lower()
            assert any(
                kw in reason
                for kw in ("unseen", "building", "no label history", "no-label", "label history")
            ), (
                "temporal_naive N/A must be justified by unseen-building/no-label-history rationale, "
                f"got: {reason!r}"
            )

    def test_grouped_hierarchical_present(self):
        cb = self._baselines()
        found = any(
            k in cb
            for k in ("grouped_hierarchical", "hierarchical", "grouped")
        )
        assert found, "context_baselines must include grouped/hierarchical baseline"

    def test_current_champion_present(self):
        cb = self._baselines()
        assert "current_champion" in cb or "champion" in cb, (
            "context_baselines must include current champion"
        )

    def test_smoke_run_is_not_described_as_current_champion(self):
        """A non-comparative Task 6 smoke cannot establish a model champion."""
        champion = self._baselines()["current_champion"]
        description = str(champion.get("description", "")).lower()
        rationale = str(champion.get("rationale", "")).lower()

        assert "current best model" not in description
        assert "no model is currently designated" in description
        assert "non-comparative smoke" in rationale
        assert "comparative" in rationale and "oof" in rationale

    def test_grouped_hierarchical_not_asserting_unsupported_na(self):
        """grouped/hierarchical rationale must remain pending or justified, not assert unsupported N/A."""
        cb = self._baselines()
        gh = cb.get("grouped_hierarchical", cb.get("hierarchical", cb.get("grouped")))
        if gh is None or not isinstance(gh, dict):
            return
        status = str(gh.get("status", gh.get("applicable", ""))).lower()
        if "na" in status or "not_applicable" in status or status == "false":
            reason = str(gh.get("rationale", gh.get("reason", ""))).lower()
            # Must not assert outright N/A without justification from data evidence.
            assert "pending" in reason or "uncertain" in reason or len(reason) > 20, (
                f"grouped/hierarchical context_baseline must not assert unsupported N/A; "
                f"rationale must be pending or well-justified, got: {reason!r}"
            )


class TestFeatureCampaignExperimentMatrix:
    """experiment_matrix must be finite, have unique IDs, required categories, null results."""

    def _matrix(self) -> list[dict[str, Any]]:
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        assert "experiment_matrix" in cfg, "experiment_matrix must be present"
        matrix = cfg["experiment_matrix"]
        assert isinstance(matrix, list), "experiment_matrix must be a list"
        assert len(matrix) > 0, "experiment_matrix must have at least one entry"
        return matrix

    def test_experiment_matrix_present_and_non_empty(self):
        matrix = self._matrix()
        assert len(matrix) >= 10, (
            f"experiment_matrix should have at least 10 entries for adequate coverage, "
            f"got {len(matrix)}"
        )

    def test_all_ids_unique(self):
        matrix = self._matrix()
        ids = [entry.get("id") for entry in matrix if isinstance(entry, dict)]
        assert all(id_ is not None for id_ in ids), "All experiment_matrix entries must have an 'id'"
        assert len(ids) == len(set(ids)), (
            f"Experiment matrix IDs must be unique; duplicates found: "
            f"{[id_ for id_ in ids if ids.count(id_) > 1]}"
        )

    @pytest.mark.parametrize("key", sorted(EXPERIMENT_MATRIX_REQUIRED_KEYS))
    def test_all_entries_have_required_keys(self, key: str):
        matrix = self._matrix()
        for entry in matrix:
            if not isinstance(entry, dict):
                continue
            assert key in entry, (
                f"experiment_matrix entry {entry.get('id', '<unknown>')!r} "
                f"is missing required key '{key}'"
            )

    def test_na_entries_have_reason(self):
        """Entries with status/execution_status N/A must have a rationale or reason."""
        matrix = self._matrix()
        for entry in matrix:
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", "")).lower()
            exec_status = str(entry.get("execution_status", "")).lower()
            if "n/a" in status or "not_applicable" in status or "na" in exec_status:
                reason = entry.get("rationale", entry.get("reason", entry.get("hypothesis")))
                assert reason is not None and str(reason).strip(), (
                    f"experiment_matrix entry {entry.get('id', '<unknown>')!r}: "
                    f"N/A status requires a non-null rationale/reason, got {reason!r}"
                )

    def test_result_fields_are_null(self):
        """No experiment_matrix entry may have non-null result data (design-only phase)."""
        matrix = self._matrix()
        result_keys = {"result", "oof_artifact", "score", "metric_value", "improvement"}
        for entry in matrix:
            if not isinstance(entry, dict):
                continue
            for key in result_keys:
                val = entry.get(key)
                assert val is None, (
                    f"experiment_matrix entry {entry.get('id', '<unknown>')!r}: "
                    f"result field '{key}' must be null in design-only phase, got {val!r}"
                )

    def test_no_threshold_tuning_entries(self):
        """Threshold tuning must not appear in the experiment matrix (forbidden in Task 7)."""
        matrix = self._matrix()
        for entry in matrix:
            if not isinstance(entry, dict):
                continue
            text = yaml.safe_dump(entry).lower()
            assert "threshold tuning" not in text and "threshold_tuning" not in text, (
                f"experiment_matrix entry {entry.get('id', '<unknown>')!r}: "
                "threshold tuning is forbidden in Task 7"
            )

    def test_opaque_building_id_digit_extraction_prohibited(self):
        """Opaque building-ID digit extraction must be N/A or explicitly prohibited."""
        matrix = self._matrix()
        # Look for the building-ID digit extraction entry.
        found = False
        for entry in matrix:
            if not isinstance(entry, dict):
                continue
            id_ = str(entry.get("id", "")).lower()
            hypothesis = str(entry.get("hypothesis", "")).lower()
            if "building" in id_ and "digit" in id_ or "building" in hypothesis and "digit" in hypothesis:
                found = True
                status = str(entry.get("status", "")).lower()
                exec_status = str(entry.get("execution_status", "")).lower()
                assert "n/a" in status or "prohibit" in status or "blocked" in exec_status or "prohibited" in exec_status, (
                    f"experiment_matrix entry {entry.get('id', '<unknown>')!r}: "
                    "opaque building-ID digit extraction must be N/A or explicitly prohibited"
                )
        assert found, (
            "experiment_matrix must include an entry for opaque building-ID digit extraction "
            "(status: N/A + explicit prohibition)"
        )

    def test_unverified_gte_family_explicitly_blocked(self):
        """Unverified gte-family features must be explicitly blocked in the matrix."""
        matrix = self._matrix()
        found = False
        for entry in matrix:
            if not isinstance(entry, dict):
                continue
            id_ = str(entry.get("id", "")).lower()
            hypothesis = str(entry.get("hypothesis", "")).lower()
            if "gte" in id_ or "gte" in hypothesis:
                found = True
                exec_status = str(entry.get("execution_status", "")).lower()
                assert "block" in exec_status, (
                    f"experiment_matrix entry {entry.get('id', '<unknown>')!r}: "
                    "unverified gte-family features must be explicitly blocked"
                )
        assert found, (
            "experiment_matrix must include at least one entry for unverified gte-family features "
            "(execution_status: blocked)"
        )

    def test_matrix_execution_status_matches_family_gate(self):
        """A matrix entry may not claim execution readiness while its family is compute-blocked."""
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        matrix = self._matrix()
        audit = cfg.get("model_family_audit", {})
        blocked_families = {
            family
            for family, entry in audit.items()
            if isinstance(entry, dict)
            and entry.get("execution_status") == "blocked_unapproved_compute"
        }

        offenders = [
            (entry.get("id"), entry.get("model_family"))
            for entry in matrix
            if isinstance(entry, dict)
            and entry.get("model_family") in blocked_families
            and str(entry.get("execution_status", "")).lower() == "planned"
        ]
        assert not offenders, (
            "experiment_matrix entries must not use execution_status='planned' while their "
            "model_family is blocked_unapproved_compute; design intent belongs in 'status'. "
            f"Offending (id, model_family): {offenders}"
        )

    def test_matrix_model_family_refs_resolve(self):
        """Every referenced model_family must exist in the audit or the frozen baseline config."""
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        baselines = _load_yaml(BASELINES_PATH)
        known = set(cfg.get("model_family_audit", {})) | set(baselines.get("models", {}))

        unknown = sorted(
            {
                str(entry.get("model_family"))
                for entry in self._matrix()
                if isinstance(entry, dict) and entry.get("model_family") is not None
            }
            - known
        )
        assert not unknown, (
            "experiment_matrix references unknown model_family values "
            f"(not in model_family_audit or baselines.yaml models): {unknown}"
        )

    def test_no_entry_is_executable_while_campaign_gate_closed(self):
        """With execution_authorized false, no entry may advertise a runnable execution status."""
        cfg = _load_yaml(FEATURE_CAMPAIGN_PATH)
        if cfg.get("execution_authorized") is not False:
            pytest.skip("campaign execution is authorized; gate check does not apply")

        closed_statuses = {
            "blocked_unapproved_compute",
            "blocked",
            "prohibited",
        }
        offenders = [
            (entry.get("id"), entry.get("execution_status"))
            for entry in self._matrix()
            if isinstance(entry, dict)
            and str(entry.get("execution_status", "")).lower() not in closed_statuses
        ]
        assert not offenders, (
            "feature_campaign.yaml has execution_authorized=false, so every matrix row must use an "
            "explicit closed execution_status (blocked_unapproved_compute, blocked, prohibited). "
            f"Offending (id, execution_status): {offenders}"
        )

    def test_required_experiment_categories_covered(self):
        """The matrix must cover all required experiment categories."""
        matrix = self._matrix()
        all_text = yaml.safe_dump(matrix).lower()

        required_categories = {
            "core vs provided features": ["core", "provided"],
            "missingness indicators": ["missingness", "indicator"],
            "categorical interactions": ["categorical", "interaction"],
            "frequency encoding": ["frequency", "encoding"],
            "group aggregates": ["group aggregate", "grouped aggregate"],
            "pca vs passthrough": ["pca", "passthrough"],
            "class-weighted": ["class_weight", "class weighted", "class-weight"],
            "calendar transforms": ["calendar", "transform"],
        }

        for category, keywords in required_categories.items():
            covered = any(kw in all_text for kw in keywords)
            assert covered, (
                f"experiment_matrix must cover the '{category}' category "
                f"(keywords checked: {keywords})"
            )


class TestTask6ContractFrozenInBaselines:
    """Baseline config must still match the frozen Task 6 contract."""

    def test_baselines_yaml_exists(self):
        assert BASELINES_PATH.exists(), (
            f"configs/experiments/baselines.yaml must exist at {BASELINES_PATH}"
        )

    def test_baselines_target_is_anomaly(self):
        cfg = _load_yaml(BASELINES_PATH)
        assert cfg.get("target") == TASK6_TARGET

    def test_baselines_positive_label_is_1(self):
        cfg = _load_yaml(BASELINES_PATH)
        assert cfg.get("positive_label") == TASK6_POSITIVE_LABEL

    def test_baselines_primary_metric_is_roc_auc(self):
        cfg = _load_yaml(BASELINES_PATH)
        assert cfg.get("primary_metric") == TASK6_PRIMARY_METRIC

    def test_baselines_guardrails_match(self):
        cfg = _load_yaml(BASELINES_PATH)
        guardrails = set(cfg.get("guardrails", []))
        assert guardrails == TASK6_GUARDRAILS

    def test_baselines_seed_is_42(self):
        cfg = _load_yaml(BASELINES_PATH)
        assert cfg.get("seed") == TASK6_SEED

    def test_baselines_folds_is_4(self):
        cfg = _load_yaml(BASELINES_PATH)
        assert cfg.get("folds") == TASK6_FOLDS

    def test_baselines_split_manifest_correct(self):
        cfg = _load_yaml(BASELINES_PATH)
        assert cfg.get("split_manifest") == TASK6_SPLIT_MANIFEST
