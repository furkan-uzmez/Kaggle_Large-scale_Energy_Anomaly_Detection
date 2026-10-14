"""Authoritative candidate recipe dispatch and supervised fit routing.

This module is the single owner of recipe dispatch and pipeline assembly for
all candidate experiments (the 10 new candidate experiments and the 8 existing
candidate/recipe implementations).

Fitting views receive only TrainingPartition (no held-out validation fields).
All fittings route through FitCounter before any fit attempts.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.metaestimators import available_if

from project.experiments.candidate_contracts import (
    BlockedExperimentError,
    CandidateSpec,
    FitCounter,
    FoldPartition,
    ScoreKind,
    TrainingPartition,
    validate_partition,
)
from project.experiments.fit_accounting import FitAccounting
from project.experiments.inner_cv import validate_inner_inputs
from project.features.candidates import (
    build_candidate_registry,
    make_candidate_transformer,
)
from project.features.distributions import (
    MeterDistributionTransformer,
    build_distribution_registry,
)
from project.features.encodings import (
    build_encoding_registry,
    make_encoding_transformer,
)
from project.features.input_encoding import (
    ABSOLUTE_CALENDAR_YEAR,
    require_year_built_encoding,
)
from project.features.provided import resolve_provided_pool
from project.features.registry import (
    FeatureRegistry,
    build_lead_baseline_registry,
)
from project.features.selection import make_selection_arms
from project.features.target_encoding import (
    GroupCrossFittedTargetEncoder,
    build_target_encoding_registry,
)
from project.features.value_recipes import (
    EvidencedSentinelTransformer,
    MeterAreaRatioTransformer,
    SentinelRule,
    build_value_recipe_registry,
)
from project.models.autogluon_adapter import AutoGluonCandidate, AutoGluonSettings
from project.models.candidate_factories import (
    _CandidateArm,
    _PredictorSchema,
    fit_candidate_arm,
)
from project.models.candidate_factories import (
    build_candidate_arms as build_task7_candidate_arms,
)

NEW_EXPERIMENT_IDS: tuple[str, ...] = (
    "feat_core_vs_provided",
    "feat_causal_group_aggregates",
    "feat_preserve_raw_vs_invalid_sentinel",
    "feat_meter_deviation_ratio",
    "feat_histogram_quantile",
    "feat_conditional_cross_fitted_target_encoding",
    "feat_missingness_passthrough_comparison",
    "regime_natural_vs_class_weighted",
    "model_extratrees_core_features",
    "model_autogluon_core_features",
    "regime_same_family_hpo_vs_default",
)

EXISTING_EXPERIMENT_IDS: tuple[str, ...] = (
    "feat_missingness_indicators",
    "feat_safe_calendar_numeric_transforms",
    "feat_categorical_interactions",
    "feat_missingness_bitmask",
    "feat_frequency_encoding",
    "dim_pca_vs_passthrough",
    "dim_fold_local_selector_vs_passthrough",
    "dim_semantic_block_drop",
)

ALL_SUPPORTED_EXPERIMENT_IDS: tuple[str, ...] = NEW_EXPERIMENT_IDS + EXISTING_EXPERIMENT_IDS

_PROHIBITED_OR_BLOCKED: Mapping[str, tuple[str, str]] = {
    "prohibited_opaque_building_id_digit_extraction": (
        "prohibited_experiment",
        "opaque building ID digit extraction is strictly prohibited",
    ),
    "blocked_unverified_gte_family": (
        "blocked_unverified_gte",
        "unverified GTE feature family is blocked pending provenance review",
    ),
    "feat_group_aggregates_blocked": (
        "blocked_group_aggregates",
        "group aggregates without historical split verification are blocked",
    ),
}

_TASK7_RECIPES = {
    "feat_missingness_passthrough_comparison",
    "regime_natural_vs_class_weighted",
    "model_extratrees_core_features",
}

_SELECTION_RECIPES = {
    "dim_pca_vs_passthrough",
    "dim_fold_local_selector_vs_passthrough",
    "dim_semantic_block_drop",
}


def _configuration(value):
    """Type-sensitive, clone-stable graph of declared parameters, never fit state."""
    if isinstance(value, BaseEstimator):
        return (type(value), _configuration(value.get_params(deep=False)))
    if isinstance(value, FeatureRegistry):
        return (type(value), tuple(_configuration(value.get(n)) for n in value.names))
    if is_dataclass(value):
        return (type(value), tuple((f.name, _configuration(getattr(value, f.name))) for f in fields(value)))
    if isinstance(value, Mapping):
        return tuple((k, _configuration(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(_configuration(v) for v in value))
    return (type(value), value)


@dataclass(frozen=True)
class CandidateArm(BaseEstimator):
    """Execution arm encapsulation with immutable lineage and estimator bindings."""

    candidate_id: str
    arm_id: str
    raw_pool: tuple[str, ...]
    transformer_or_pipeline: Any
    authoritative_registry: FeatureRegistry
    score_kind: ScoreKind

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a nonempty string")
        if not isinstance(self.arm_id, str) or not self.arm_id.strip():
            raise ValueError("arm_id must be a nonempty string")
        if not isinstance(self.raw_pool, tuple) or not self.raw_pool:
            raise ValueError("raw_pool must be a nonempty tuple of feature names")
        if self.transformer_or_pipeline is None:
            raise ValueError("transformer_or_pipeline must not be None")
        if not isinstance(self.authoritative_registry, FeatureRegistry):
            raise TypeError("authoritative_registry must be an authoritative FeatureRegistry")
        if self.score_kind not in ("positive_probability", "decision_margin"):
            raise ValueError("unsupported explicit score_kind")

    @property
    def classes_(self) -> Any:
        return getattr(self.transformer_or_pipeline, "classes_", None)

    @property
    def feature_names_in_(self) -> Any:
        return getattr(self.transformer_or_pipeline, "feature_names_in_", None)

    @property
    def n_features_in_(self) -> Any:
        return getattr(self.transformer_or_pipeline, "n_features_in_", None)

    @property
    def named_steps(self) -> Any:
        return getattr(self.transformer_or_pipeline, "named_steps", None)

    def _check_configuration(self):
        if hasattr(self, "fitted_configuration_") and self.fitted_configuration_ != _configuration(self.transformer_or_pipeline):
            raise ValueError("fitted configuration changed; fit again before use")

    @available_if(lambda self: hasattr(self.transformer_or_pipeline, "predict_proba"))
    def predict_proba(self, X: Any, **kwargs: Any) -> Any:
        self._check_configuration()
        method = getattr(self.transformer_or_pipeline, "predict_proba", None)
        if callable(method):
            return method(X, **kwargs)
        raise AttributeError(f"{type(self.transformer_or_pipeline).__name__} has no predict_proba")

    @available_if(lambda self: hasattr(self.transformer_or_pipeline, "predict"))
    def predict(self, X: Any, **kwargs: Any) -> Any:
        self._check_configuration()
        method = getattr(self.transformer_or_pipeline, "predict", None)
        if callable(method):
            return method(X, **kwargs)
        raise AttributeError(f"{type(self.transformer_or_pipeline).__name__} has no predict")

    @available_if(lambda self: hasattr(self.transformer_or_pipeline, "decision_function"))
    def decision_function(self, X: Any, **kwargs: Any) -> Any:
        self._check_configuration()
        method = getattr(self.transformer_or_pipeline, "decision_function", None)
        if callable(method):
            return method(X, **kwargs)
        raise AttributeError(f"{type(self.transformer_or_pipeline).__name__} has no decision_function")

    def __getitem__(self, item: Any) -> Any:
        return self.transformer_or_pipeline[item]

    @property
    def fit_cost_metadata(self):
        pipe = self.transformer_or_pipeline
        steps = pipe.named_steps if isinstance(pipe, Pipeline) else {}
        selector = steps.get("selector")
        encoder = steps.get("features")
        return {
            "supervised_model_fits": 0 if isinstance(pipe, HGBTuningStrategy) else (2 if isinstance(selector, BaseEstimator) else 1),
            "learned_preprocessing_boundary": "outer_train",
            "encoder_boundary": "group_disjoint_inner_train" if isinstance(encoder, GroupCrossFittedTargetEncoder) else "outer_train_or_fixed",
            "encoder_fitted_objects": encoder.n_splits + 1 if isinstance(encoder, GroupCrossFittedTargetEncoder) else 0,
            "feature_fit_boundaries": tuple(
                (name, self.authoritative_registry.get(name).transform_fit_boundary.value)
                for name in self.authoritative_registry.names
            ),
            "hpo_ready": False,
        }


@dataclass(frozen=True)
class HGBTuningStrategy:
    """Strategy interface for nested HGB random search tuning."""

    study_spec: Any | None
    study_dir: str | Path | None
    spec: CandidateSpec

    def tune(self, training: TrainingPartition, counter: FitCounter) -> Any:
        raise BlockedExperimentError(
            "hpo_not_ready", "Task 9 accounting/resume integration requires parent approval",
        )


@dataclass(frozen=True)
class ResolvedCandidate:
    """Container describing resolved arms and candidate experiment metadata."""

    spec: CandidateSpec
    arms: dict[str, CandidateArm]
    authoritative_registry: FeatureRegistry
    score_kind: ScoreKind
    metadata: tuple[tuple[str, object], ...] = ()

    @property
    def experiment_id(self) -> str:
        return self.spec.experiment_id

    @property
    def model_family(self) -> str:
        return self.spec.model_family

    @property
    def raw_pool(self) -> tuple[str, ...]:
        return self.spec.feature_pool


def _build_feature_pipeline(
    raw_pool: tuple[str, ...],
    transformer: Any | None,
    output_registry: FeatureRegistry,
    model: Any,
) -> Pipeline:
    raw_registry = build_lead_baseline_registry()
    raw_categorical = tuple(
        n for n in raw_pool
        if raw_registry.get(n) is not None and raw_registry.get(n).family == "categorical"
    )
    out_categorical = tuple(
        n for n in output_registry.names
        if output_registry.get(n) is not None and output_registry.get(n).family == "categorical"
    )
    out_numeric = tuple(n for n in output_registry.names if n not in out_categorical)

    category_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transforms = []
    if out_numeric:
        transforms.append(("numeric", SimpleImputer(strategy="median", keep_empty_features=True), out_numeric))
    if out_categorical:
        transforms.append(("categorical", category_pipeline, out_categorical))
    preprocess = ColumnTransformer(transforms, remainder="drop", sparse_threshold=0.0)

    steps = [("schema", _PredictorSchema(raw_pool, raw_categorical))]
    if transformer is not None:
        steps.append(("features", transformer))
    steps.append(("preprocess", preprocess))
    steps.append(("model", model))
    return Pipeline(steps)


def _default_hgb_model(spec: CandidateSpec) -> HistGradientBoostingClassifier:
    model_kwargs: dict[str, Any] = {"early_stopping": False, "random_state": 42}
    if spec.model_params and spec.model_family != "autogluon":
        model_kwargs.update(dict(spec.model_params))
    return HistGradientBoostingClassifier(**model_kwargs)


def _validate_parameters(spec):
    methods = dict(spec.method_params)
    allowed = {
        "feat_histogram_quantile": {"n_bins", "n_quantiles"},
        "feat_conditional_cross_fitted_target_encoding": {"smoothing", "n_splits", "cv_splits", "seed", "max_categories"},
        "dim_pca_vs_passthrough": {"pca_components"},
        "dim_semantic_block_drop": {"drop_block"},
        "model_autogluon_core_features": {"time_limit", "max_rss_bytes", "num_cpus"},
        "regime_same_family_hpo_vs_default": {"study_spec", "study_dir"},
        "feat_preserve_raw_vs_invalid_sentinel": {"rule", "sentinel_rule", "column", "invalid_values", "replacement_value", "evidence_reference", "evidence_sha256", "evidence_root"},
        "feat_causal_group_aggregates": {"windows"},
    }.get(spec.experiment_id, set())
    if set(methods) - allowed:
        raise ValueError("unknown method parameters for recipe")
    bounds = {"n_bins": (2, 64), "n_quantiles": (2, 256), "n_splits": (2, 5), "cv_splits": (2, 5), "seed": (0, 2**32-1), "max_categories": (1, 4096)}
    for key, value in methods.items():
        if key in bounds and (type(value) is not int or not bounds[key][0] <= value <= bounds[key][1]):
            raise ValueError(f"invalid typed/bounded method parameter: {key}")
        if key == "smoothing" and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1000):
            raise ValueError("invalid smoothing")
        if key == "pca_components" and (type(value) is not float or not 0 < value < 1):
            raise ValueError("pca_components must be a float in (0,1)")
        if key == "drop_block" and value not in ("calendar", "measurement", "building_metadata"):
            raise ValueError("invalid drop_block")
    if "cv_splits" in methods and "n_splits" in methods:
        raise ValueError("duplicate split aliases")
    for key in ("study_dir", "evidence_root"):
        if key in methods and (type(methods[key]) is not str or not methods[key].strip()):
            raise ValueError(f"{key} must be a nonempty path string")
    if spec.experiment_id == "feat_preserve_raw_vs_invalid_sentinel":
        aliases = set(methods) & {"rule", "sentinel_rule"}
        scalar_keys = set(methods) & {"column", "invalid_values", "replacement_value", "evidence_reference", "evidence_sha256"}
        if len(aliases) > 1 or aliases and scalar_keys:
            raise ValueError("ambiguous sentinel rule configuration")
        _extract_sentinel_rule(methods)
    params = dict(spec.model_params)
    if spec.model_family == "autogluon":
        if set(params) - {"time_limit", "max_rss_bytes", "num_cpus"} or set(params) & set(methods):
            raise ValueError("unknown or duplicate AutoGluon settings")
        AutoGluonSettings(**{"time_limit": 10., "max_rss_bytes": 1024**3, **methods, **params})
    elif spec.experiment_id not in _TASK7_RECIPES:
        if spec.model_family == "svm":
            from sklearn.svm import SVC
            defaults = SVC(C=1., kernel="rbf", gamma="scale", random_state=42).get_params()
        elif spec.model_family == "logistic_l2":
            from sklearn.linear_model import LogisticRegression
            defaults = LogisticRegression(C=1., l1_ratio=0., solver="lbfgs", max_iter=1000, random_state=42).get_params()
        else:
            defaults = HistGradientBoostingClassifier(early_stopping=False, random_state=42).get_params()
        if any(k not in defaults or _configuration(v) != _configuration(defaults[k]) for k, v in params.items()):
            raise ValueError("model parameters must match frozen defaults")


def _extract_sentinel_rule(method_dict: Mapping[str, Any]) -> SentinelRule:
    for key in ("sentinel_rule", "rule"):
        if key in method_dict:
            val = method_dict[key]
            if isinstance(val, SentinelRule):
                return val
            if isinstance(val, tuple) and len(val) == 5:
                return SentinelRule(
                    column=val[0],
                    invalid_values=val[1],
                    replacement_value=val[2],
                    evidence_reference=val[3],
                    evidence_sha256=val[4],
                )
    if all(k in method_dict for k in ("column", "invalid_values", "replacement_value", "evidence_reference", "evidence_sha256")):
        return SentinelRule(
            column=method_dict["column"],
            invalid_values=method_dict["invalid_values"],
            replacement_value=method_dict["replacement_value"],
            evidence_reference=method_dict["evidence_reference"],
            evidence_sha256=method_dict["evidence_sha256"],
        )
    raise BlockedExperimentError(
        "missing_evidence_rule",
        "feat_preserve_raw_vs_invalid_sentinel requires an explicit evidenced SentinelRule",
    )


def _validate_spec_family(spec: CandidateSpec) -> None:
    family = spec.model_family
    exp = spec.experiment_id
    if exp == "dim_pca_vs_passthrough":
        if family != "svm":
            raise ValueError(f"mismatched or unsupported model family {family!r} for {exp!r}; expected 'svm'")
    elif exp == "dim_fold_local_selector_vs_passthrough":
        if family != "logistic_l2":
            raise ValueError(f"mismatched or unsupported model family {family!r} for {exp!r}; expected 'logistic_l2'")
    elif exp == "model_extratrees_core_features":
        if family not in ("extratrees", "extra_trees"):
            raise ValueError(f"mismatched or unsupported model family {family!r} for {exp!r}; expected 'extratrees'")
    elif exp == "model_autogluon_core_features":
        if family != "autogluon":
            raise ValueError(f"mismatched or unsupported model family {family!r} for {exp!r}; expected 'autogluon'")
    else:
        if family != "hist_gradient_boosting":
            raise ValueError(
                f"mismatched or unsupported model family {family!r} for {exp!r}; "
                f"expected 'hist_gradient_boosting'"
            )


def build_experiment_arms(
    spec: CandidateSpec,
    registry: FeatureRegistry | None = None,
    *,
    year_built_encoding: str = ABSOLUTE_CALENDAR_YEAR,
) -> dict[str, CandidateArm]:
    """Assemble arms with explicit input semantics, without fitting or run approval.

    The keyword-only encoding context is not a CandidateSpec method parameter.
    It does not rewrite the frozen raw predictors or their preprocessing.
    """
    require_year_built_encoding(year_built_encoding)
    if not isinstance(spec, CandidateSpec):
        raise TypeError("spec must be a typed CandidateSpec")

    exp_id = spec.experiment_id
    if exp_id in _PROHIBITED_OR_BLOCKED:
        code, reason = _PROHIBITED_OR_BLOCKED[exp_id]
        raise BlockedExperimentError(code, reason)

    if exp_id not in ALL_SUPPORTED_EXPERIMENT_IDS:
        raise ValueError(f"unsupported or unregistered experiment ID: {exp_id!r}")

    _validate_spec_family(spec)
    _validate_parameters(spec)

    base_reg = registry if registry is not None else build_lead_baseline_registry(year_built_encoding)
    method_dict = dict(spec.method_params)
    arms: dict[str, CandidateArm] = {}

    # Case 1: Task 7 candidate factories (native, weighted, extratrees)
    if exp_id in _TASK7_RECIPES:
        norm_spec = spec
        if spec.model_family == "extra_trees":
            norm_spec = replace(spec, model_family="extratrees")
        task7_arms = build_task7_candidate_arms(
            norm_spec, base_reg, year_built_encoding=year_built_encoding,
        )
        for arm_id, pipe in task7_arms.items():
            arms[arm_id] = CandidateArm(
                candidate_id=f"{exp_id}_{arm_id}",
                arm_id=arm_id,
                raw_pool=spec.feature_pool,
                transformer_or_pipeline=pipe,
                authoritative_registry=base_reg,
                score_kind="positive_probability",
            )
        return arms

    # Case 2: Selection / dimensionality recipes
    if exp_id in _SELECTION_RECIPES:
        drop_block = method_dict.get("drop_block")
        if exp_id == "dim_semantic_block_drop" and drop_block is None:
            raise ValueError("semantic recipe requires an explicit drop_block")
        pca_components = float(method_dict.get("pca_components", 0.95))
        selection_arms = make_selection_arms(
            exp_id,
            spec.feature_pool,
            registry=base_reg,
            drop_block=str(drop_block) if drop_block is not None else None,
            pca_components=pca_components,
        )
        score_kind: ScoreKind = "decision_margin" if exp_id == "dim_pca_vs_passthrough" else "positive_probability"
        for arm_id, pipe in selection_arms.items():
            arms[arm_id] = CandidateArm(
                candidate_id=f"{exp_id}_{arm_id}",
                arm_id=arm_id,
                raw_pool=spec.feature_pool,
                transformer_or_pipeline=pipe,
                authoritative_registry=base_reg,
                score_kind=score_kind,
            )
        return arms

    # Case 3: feat_core_vs_provided
    if exp_id == "feat_core_vs_provided":
        lead_core_reg = build_lead_baseline_registry(year_built_encoding=year_built_encoding)
        core_pool = lead_core_reg.names
        resolved_pool = resolve_provided_pool(core_pool, spec.feature_pool, base_reg)
        control_pipe = _build_feature_pipeline(core_pool, None, lead_core_reg, _default_hgb_model(spec))
        challenger_pipe = _build_feature_pipeline(resolved_pool, None, base_reg, _default_hgb_model(spec))
        arms["control"] = CandidateArm(
            candidate_id=f"{exp_id}_control",
            arm_id="control",
            raw_pool=core_pool,
            transformer_or_pipeline=control_pipe,
            authoritative_registry=lead_core_reg,
            score_kind="positive_probability",
        )
        arms["challenger"] = CandidateArm(
            candidate_id=f"{exp_id}_challenger",
            arm_id="challenger",
            raw_pool=resolved_pool,
            transformer_or_pipeline=challenger_pipe,
            authoritative_registry=base_reg,
            score_kind="positive_probability",
        )
        return arms

    # Case 3b: feat_causal_group_aggregates
    if exp_id == "feat_causal_group_aggregates":
        from project.features.causal_aggregates import (
            CAUSAL_ROLLING_FEATURES,
            CausalRollingWindowTransformer,
            build_causal_aggregates_registry,
        )
        windows = method_dict.get("windows", (24, 168))
        causal_reg = build_causal_aggregates_registry(year_built_encoding=year_built_encoding)
        transformer = CausalRollingWindowTransformer(target_column="meter_reading", windows=windows)
        control_pipe = _build_feature_pipeline(spec.feature_pool, None, base_reg, _default_hgb_model(spec))
        challenger_pipe = _build_feature_pipeline(spec.feature_pool, transformer, causal_reg, _default_hgb_model(spec))
        arms["control"] = CandidateArm(
            candidate_id=f"{exp_id}_control",
            arm_id="control",
            raw_pool=spec.feature_pool,
            transformer_or_pipeline=control_pipe,
            authoritative_registry=base_reg,
            score_kind="positive_probability",
        )
        arms["challenger"] = CandidateArm(
            candidate_id=f"{exp_id}_challenger",
            arm_id="challenger",
            raw_pool=spec.feature_pool,
            transformer_or_pipeline=challenger_pipe,
            authoritative_registry=causal_reg,
            score_kind="positive_probability",
        )
        return arms

    # Case 4: feat_preserve_raw_vs_invalid_sentinel
    if exp_id == "feat_preserve_raw_vs_invalid_sentinel":
        rule = _extract_sentinel_rule(method_dict)
        evidence_root = method_dict.get("evidence_root")
        root_path = Path(evidence_root) if evidence_root is not None else None
        sentinel_reg = build_value_recipe_registry(exp_id, rule=rule)
        transformer = EvidencedSentinelTransformer(rule, sentinel_reg, evidence_root=root_path)
        control_pipe = _build_feature_pipeline(spec.feature_pool, None, base_reg, _default_hgb_model(spec))
        challenger_pipe = _build_feature_pipeline(spec.feature_pool, transformer, sentinel_reg, _default_hgb_model(spec))
        arms["control"] = CandidateArm(
            candidate_id=f"{exp_id}_control",
            arm_id="control",
            raw_pool=spec.feature_pool,
            transformer_or_pipeline=control_pipe,
            authoritative_registry=base_reg,
            score_kind="positive_probability",
        )
        arms["challenger"] = CandidateArm(
            candidate_id=f"{exp_id}_challenger",
            arm_id="challenger",
            raw_pool=spec.feature_pool,
            transformer_or_pipeline=challenger_pipe,
            authoritative_registry=sentinel_reg,
            score_kind="positive_probability",
        )
        return arms

    # Case 5: feat_meter_deviation_ratio
    if exp_id == "feat_meter_deviation_ratio":
        ratio_reg = build_value_recipe_registry(exp_id)
        transformer = MeterAreaRatioTransformer()
        control_pipe = _build_feature_pipeline(spec.feature_pool, None, base_reg, _default_hgb_model(spec))
        challenger_pipe = _build_feature_pipeline(spec.feature_pool, transformer, ratio_reg, _default_hgb_model(spec))
        arms["control"] = CandidateArm(
            candidate_id=f"{exp_id}_control",
            arm_id="control",
            raw_pool=spec.feature_pool,
            transformer_or_pipeline=control_pipe,
            authoritative_registry=base_reg,
            score_kind="positive_probability",
        )
        arms["challenger"] = CandidateArm(
            candidate_id=f"{exp_id}_challenger",
            arm_id="challenger",
            raw_pool=spec.feature_pool,
            transformer_or_pipeline=challenger_pipe,
            authoritative_registry=ratio_reg,
            score_kind="positive_probability",
        )
        return arms

    # Case 6: feat_histogram_quantile (3 arms)
    if exp_id == "feat_histogram_quantile":
        n_bins = int(method_dict.get("n_bins", 8))
        n_quantiles = int(method_dict.get("n_quantiles", 64))
        bin_reg = build_distribution_registry("bin")
        q_reg = build_distribution_registry("quantile")
        bin_tr = MeterDistributionTransformer("bin", n_bins=n_bins, n_quantiles=n_quantiles)
        q_tr = MeterDistributionTransformer("quantile", n_bins=n_bins, n_quantiles=n_quantiles)

        control_pipe = _build_feature_pipeline(spec.feature_pool, None, base_reg, _default_hgb_model(spec))
        bin_pipe = _build_feature_pipeline(spec.feature_pool, bin_tr, bin_reg, _default_hgb_model(spec))
        q_pipe = _build_feature_pipeline(spec.feature_pool, q_tr, q_reg, _default_hgb_model(spec))

        arms["control"] = CandidateArm(
            candidate_id=f"{exp_id}_control",
            arm_id="control",
            raw_pool=spec.feature_pool,
            transformer_or_pipeline=control_pipe,
            authoritative_registry=base_reg,
            score_kind="positive_probability",
        )
        arms["bin"] = CandidateArm(
            candidate_id=f"{exp_id}_bin",
            arm_id="bin",
            raw_pool=spec.feature_pool,
            transformer_or_pipeline=bin_pipe,
            authoritative_registry=bin_reg,
            score_kind="positive_probability",
        )
        arms["quantile"] = CandidateArm(
            candidate_id=f"{exp_id}_quantile",
            arm_id="quantile",
            raw_pool=spec.feature_pool,
            transformer_or_pipeline=q_pipe,
            authoritative_registry=q_reg,
            score_kind="positive_probability",
        )
        return arms

    # Case 7: feat_conditional_cross_fitted_target_encoding
    if exp_id == "feat_conditional_cross_fitted_target_encoding":
        te_reg = build_target_encoding_registry()
        smoothing = float(method_dict.get("smoothing", 10.0))
        n_splits = int(method_dict.get("cv_splits", method_dict.get("n_splits", 3)))
        seed = int(method_dict.get("seed", 42))
        max_cat = int(method_dict.get("max_categories", 4096))
        te_tr = GroupCrossFittedTargetEncoder(
            smoothing=smoothing,
            n_splits=n_splits,
            seed=seed,
            max_categories=max_cat,
        )
        control_pipe = _build_feature_pipeline(spec.feature_pool, None, base_reg, _default_hgb_model(spec))
        challenger_pipe = _build_feature_pipeline(spec.feature_pool, te_tr, te_reg, _default_hgb_model(spec))
        arms["control"] = CandidateArm(
            candidate_id=f"{exp_id}_control",
            arm_id="control",
            raw_pool=spec.feature_pool,
            transformer_or_pipeline=control_pipe,
            authoritative_registry=base_reg,
            score_kind="positive_probability",
        )
        arms["challenger"] = CandidateArm(
            candidate_id=f"{exp_id}_challenger",
            arm_id="challenger",
            raw_pool=spec.feature_pool,
            transformer_or_pipeline=challenger_pipe,
            authoritative_registry=te_reg,
            score_kind="positive_probability",
        )
        return arms

    # Case 8: model_autogluon_core_features
    if exp_id == "model_autogluon_core_features":
        model_params_dict = dict(spec.model_params)
        time_limit = float(model_params_dict.get("time_limit", method_dict.get("time_limit", 10.0)))
        max_rss = int(model_params_dict.get("max_rss_bytes", method_dict.get("max_rss_bytes", 1024 * 1024 * 1024)))
        num_cpus = int(model_params_dict.get("num_cpus", method_dict.get("num_cpus", 1)))
        settings = AutoGluonSettings(time_limit=time_limit, max_rss_bytes=max_rss, num_cpus=num_cpus)
        ag_candidate = AutoGluonCandidate(settings)
        control_pipe = _build_feature_pipeline(spec.feature_pool, None, base_reg, _default_hgb_model(spec))
        arms["control"] = CandidateArm(
            candidate_id=f"{exp_id}_control",
            arm_id="control",
            raw_pool=spec.feature_pool,
            transformer_or_pipeline=control_pipe,
            authoritative_registry=base_reg,
            score_kind="positive_probability",
        )
        arms["challenger"] = CandidateArm(
            candidate_id=f"{exp_id}_challenger",
            arm_id="challenger",
            raw_pool=spec.feature_pool,
            transformer_or_pipeline=ag_candidate,
            authoritative_registry=base_reg,
            score_kind="positive_probability",
        )
        return arms

    # Case 9: regime_same_family_hpo_vs_default
    if exp_id == "regime_same_family_hpo_vs_default":
        study_spec = method_dict.get("study_spec")
        study_dir = method_dict.get("study_dir")
        hpo_strategy = HGBTuningStrategy(study_spec=study_spec, study_dir=study_dir, spec=spec)
        control_pipe = _build_feature_pipeline(spec.feature_pool, None, base_reg, _default_hgb_model(spec))
        arms["control"] = CandidateArm(
            candidate_id=f"{exp_id}_control",
            arm_id="control",
            raw_pool=spec.feature_pool,
            transformer_or_pipeline=control_pipe,
            authoritative_registry=base_reg,
            score_kind="positive_probability",
        )
        arms["challenger"] = CandidateArm(
            candidate_id=f"{exp_id}_challenger",
            arm_id="challenger",
            raw_pool=spec.feature_pool,
            transformer_or_pipeline=hpo_strategy,
            authoritative_registry=base_reg,
            score_kind="positive_probability",
        )
        return arms

    # Case 10: Existing candidates and encodings recipes
    if exp_id in ("feat_missingness_indicators", "feat_safe_calendar_numeric_transforms", "feat_categorical_interactions"):
        feat_reg = build_candidate_registry(exp_id, year_built_encoding=year_built_encoding)
        tr = make_candidate_transformer(exp_id, year_built_encoding=year_built_encoding)
    elif exp_id in ("feat_missingness_bitmask", "feat_frequency_encoding"):
        feat_reg = build_encoding_registry(exp_id, year_built_encoding=year_built_encoding)
        tr = make_encoding_transformer(exp_id, year_built_encoding=year_built_encoding)
    else:
        raise ValueError(f"unhandled experiment ID: {exp_id}")

    control_pipe = _build_feature_pipeline(spec.feature_pool, None, base_reg, _default_hgb_model(spec))
    challenger_pipe = _build_feature_pipeline(spec.feature_pool, tr, feat_reg, _default_hgb_model(spec))
    arms["control"] = CandidateArm(
        candidate_id=f"{exp_id}_control",
        arm_id="control",
        raw_pool=spec.feature_pool,
        transformer_or_pipeline=control_pipe,
        authoritative_registry=base_reg,
        score_kind="positive_probability",
    )
    arms["challenger"] = CandidateArm(
        candidate_id=f"{exp_id}_challenger",
        arm_id="challenger",
        raw_pool=spec.feature_pool,
        transformer_or_pipeline=challenger_pipe,
        authoritative_registry=feat_reg,
        score_kind="positive_probability",
    )
    return arms


def resolve_candidate_spec(
    spec: CandidateSpec,
    registry: FeatureRegistry | None = None,
    *,
    year_built_encoding: str = ABSOLUTE_CALENDAR_YEAR,
) -> ResolvedCandidate:
    """Resolve a candidate specification into verified arms and metadata."""
    if not isinstance(spec, CandidateSpec):
        raise TypeError("spec must be a typed CandidateSpec")
    arms = build_experiment_arms(spec, registry=registry, year_built_encoding=year_built_encoding)
    primary_arm = arms.get("challenger") or next(iter(arms.values()))
    return ResolvedCandidate(
        spec=spec,
        arms=arms,
        authoritative_registry=primary_arm.authoritative_registry,
        score_kind=primary_arm.score_kind,
        metadata=spec.method_params,
    )


def fit_arm(
    arm: CandidateArm,
    partition: TrainingPartition,
    counter: FitCounter,
) -> CandidateArm:
    """Fit a candidate arm against a restricted TrainingPartition.

    Routing guarantees:
    - Counter debits before fit attempt.
    - Partition is verified and restricted: no validation fields accessible.
    - Supervised group routing occurs for steps requiring group context.
    """
    if not isinstance(arm, CandidateArm):
        raise TypeError("arm must be a typed CandidateArm")
    if not isinstance(partition, TrainingPartition) or isinstance(partition, FoldPartition):
        raise TypeError("partition must be a TrainingPartition; FoldPartition is prohibited")
    if not isinstance(counter, FitCounter) and not isinstance(counter, FitAccounting):
        raise TypeError("counter must satisfy the shared FitAccounting protocol")

    validate_partition(partition, build_lead_baseline_registry())

    underlying = arm.transformer_or_pipeline

    if isinstance(underlying, HGBTuningStrategy):
        fitted_estimator = underlying.tune(partition, counter)
        return replace(arm, transformer_or_pipeline=fitted_estimator)

    underlying = clone(underlying)

    def completed(estimator):
        result = replace(arm, transformer_or_pipeline=estimator)
        object.__setattr__(result, "fitted_configuration_", _configuration(estimator))
        return result

    if isinstance(underlying, AutoGluonCandidate):
        counter.before_fit()
        underlying.fit(partition.X_train, partition.y_train)
        return completed(underlying)

    if isinstance(underlying, _CandidateArm):
        fitted_pipe = fit_candidate_arm(
            underlying,
            partition.X_train,
            partition.y_train,
            groups=partition.groups_train,
            fit_counter=counter,
        )
        return completed(fitted_pipe)

    if isinstance(underlying, Pipeline):
        selector = underlying.named_steps.get("selector")
        if isinstance(selector, BaseEstimator):
            if counter.reserved_model_fits - counter.used_model_fits < 2:
                raise BlockedExperimentError("fit_not_reserved", "selector and final model require two reserved fits")
            X = partition.X_train
            for name, step in underlying.steps[:-1]:
                if step is None or isinstance(step, str):
                    continue
                if name == "selector":
                    counter.before_fit()
                X = step.fit_transform(X, partition.y_train)
            counter.before_fit()
            underlying.steps[-1][1].fit(X, partition.y_train)
            return completed(underlying)
        params: dict[str, Any] = {}
        if partition.groups_train is not None:
            validate_inner_inputs(partition.y_train, partition.groups_train)
            for name, step in underlying.steps[:-1]:
                if step == "passthrough" or step is None:
                    continue
                method = getattr(step, "fit_transform", getattr(step, "fit", None))
                if method is not None and "groups" in inspect.signature(method).parameters:
                    params[f"{name}__groups"] = partition.groups_train
        counter.before_fit()
        underlying.fit(partition.X_train, partition.y_train, **params)
        return completed(underlying)

    # General scikit-learn estimator fallback
    counter.before_fit()
    fitted_est = underlying.fit(partition.X_train, partition.y_train)
    return completed(fitted_est)
