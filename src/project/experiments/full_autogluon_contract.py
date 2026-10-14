"""Frozen XT campaign declarations, without estimators or launch permission."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from project.experiments.candidate_contracts import CandidateSpec, _validate_managed_path
from project.experiments.candidate_dispatch import CandidateArm
from project.experiments.full_input_schema import CORE
from project.experiments.full_run_contracts import FullCandidateSnapshot, FullRunSpec, _file_hash
from project.experiments.full_snapshot_codec import canonical_json, parse_json
from project.features.input_encoding import LEAD_YEAR_OFFSET_1900_UINT8_V1
from project.features.registry import build_lead_baseline_registry
from project.models import full_autogluon_bridge as bridge
from project.models import full_autogluon_worker as backend

EXPERIMENT_ID = "model_autogluon_core_features"
ROUTE_ID = "reviewed_autogluon_xt_transport_v1"
APPROVAL_PATH = "docs/evidence/autogluon_full_profile_v1.json"
APPROVAL_SHA256 = "2234b2a0a863b8a33d21da415dfa647eb9c10fad918c7d26a6f22799624bd1c0"
IMPLEMENTATION_ROOT = Path(__file__).absolute().parents[3]


@dataclass(frozen=True)
class FrozenAutoGluonProfile:
    """Configuration-only marker; never the bounded AutoGluon adapter lease."""

    resolved_profile_json: str
    approval_sha256: str = APPROVAL_SHA256


def validate_full_autogluon_snapshot(snapshot: FullCandidateSnapshot, *, year_built_encoding: str) -> None:
    if type(snapshot) is not FullCandidateSnapshot or type(snapshot.spec) is not CandidateSpec:
        raise TypeError("exact full AutoGluon snapshot required")
    requested = snapshot.spec
    if (requested.experiment_id != EXPERIMENT_ID or requested.model_family != "autogluon"
            or requested.feature_pool != CORE or requested.model_params or requested.method_params
            or snapshot.arms != ("challenger",) or snapshot.selector_arms
            or snapshot.study is not None or snapshot.study_method is not None
            or year_built_encoding != LEAD_YEAR_OFFSET_1900_UINT8_V1):
        raise ValueError("AutoGluon frozen challenger, canonical core and empty parameters required")


def _reviewed_profile() -> dict:
    approval = _validate_managed_path(IMPLEMENTATION_ROOT / APPROVAL_PATH, "AutoGluon approval",
                                     root=IMPLEMENTATION_ROOT)
    if _file_hash(approval) != APPROVAL_SHA256:
        raise ValueError("AutoGluon authoritative approval hash mismatch")
    backend._verify_approval()
    reviewed = parse_json(approval.read_text())
    fit = {**reviewed["fit_policy"], "hyperparameters": {"XT": reviewed["xt_hyperparameters"]},
           "memory_limit": "auto", "num_cpus": 1, "num_gpus": 0}
    expected = {"constructor": reviewed["constructor"], "fit": fit}
    actual = backend.resolved_profile()
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError("AutoGluon resolved profile differs from authoritative approval")
    bridge.validate_profile(bridge.full_xt_profile())
    return actual


def build_reviewed_autogluon_arms(snapshot: FullCandidateSnapshot, *, year_built_encoding: str) -> dict[str, CandidateArm]:
    validate_full_autogluon_snapshot(snapshot, year_built_encoding=year_built_encoding)
    marker = FrozenAutoGluonProfile(canonical_json(_reviewed_profile()))
    registry = build_lead_baseline_registry(year_built_encoding=year_built_encoding)
    return {"challenger": CandidateArm(EXPERIMENT_ID + "_challenger", "challenger", CORE,
                                       marker, registry, "positive_probability")}


def verify_autogluon_profile_source(spec: FullRunSpec, source: str | Path) -> None:
    bindings = tuple(item for item in spec.sources if item.role == "autogluon_profile")
    if len(bindings) != 1 or (bindings[0].path, bindings[0].sha256) != (APPROVAL_PATH, APPROVAL_SHA256):
        raise ValueError("exact hash-bound AutoGluon approval source required")
    bindings[0].verify(Path(source))
    _reviewed_profile()


def verify_autogluon_environment(source: str | Path) -> dict:
    """Probe only the executing reviewed repository's fixed Python3.12 environment."""
    source = _validate_managed_path(source, "AutoGluon source", directory=True)
    approval = _validate_managed_path(source / APPROVAL_PATH, "AutoGluon source approval", root=source)
    if _file_hash(approval) != APPROVAL_SHA256:
        raise ValueError("AutoGluon source approval hash mismatch")
    interpreter = IMPLEMENTATION_ROOT / ".venv-autogluon/bin/python"
    facts = bridge.inspect_environment(str(interpreter))
    bridge.validate_environment(facts)
    if (facts["executable"] != str(interpreter)
            or facts["prefix"] != str(IMPLEMENTATION_ROOT / ".venv-autogluon")):
        raise ValueError("AutoGluon approved environment root mismatch")
    return facts
