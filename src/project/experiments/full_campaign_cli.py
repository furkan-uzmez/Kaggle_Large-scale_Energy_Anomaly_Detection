"""Full-campaign CLI support: load a concrete full-intent snapshot.

The full modes require an explicit ``--config`` snapshot. This module builds a
typed ``FullRunSpec``/``FullCampaignIntent`` from that YAML, computing source byte
hashes live from the repository so a stale digest cannot be smuggled in. It never
defaults a sentinel rule, HPO tuning, or a model family: those must be explicit in
the config. Validate-only performs no writes and creates no run directory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from project.experiments.candidate_contracts import (
    CandidateSpec,
    StudySpec,
    _validate_managed_path,
)
from project.experiments.full_campaign_admission import (
    FullCampaignIntent,
)
from project.experiments.full_campaign_capabilities import (
    resolve_full_campaign_capability,
)
from project.experiments.full_campaign_preflight import (
    current_environment,
    preflight_full_campaign,
)
from project.experiments.full_run_contracts import (
    FullCandidateSnapshot,
    FullResourcePolicy,
    FullRunSpec,
    SourceBinding,
    _file_hash,
    _relative,
)

SOURCE_ROLES = ("matrix", "baseline", "data", "split", "producer", "notebook", "encoding")
OPTIONAL_ROLES = ("autogluon_profile",)

def load_full_spec(config_path: str | Path, project_root: str | Path) -> FullRunSpec:
    path = Path(config_path)
    if not path.is_file():
        raise ValueError("full-intent config does not exist")
    document = yaml.safe_load(path.read_text())
    return _spec_from_document(document, Path(project_root))


def _spec_from_document(document: Any, root: Path) -> FullRunSpec:
    if not isinstance(document, dict):
        raise TypeError("full-intent config must be a mapping")
    record = document.get("resource_opt_out_record")
    if not isinstance(record, str) or not record.strip():
        raise ValueError("explicit existing resource_opt_out_record required; consent is never inferred")
    required = {"run_id", "candidates", "sources", "environment"}
    if not required <= set(document):
        raise ValueError(f"full-intent config is missing keys: {sorted(required - set(document))}")
    sources = _bind_sources(document["sources"], root)
    candidates = tuple(_candidate(entry) for entry in document["candidates"])
    environment = tuple((str(name), str(value)) for name, value in document["environment"].items())
    if tuple(sorted(environment)) != current_environment():
        raise ValueError("declared environment differs from actual environment")
    resource_policy = FullResourcePolicy(informed_opt_out=True)
    return FullRunSpec(
        run_id=document["run_id"],
        candidates=candidates,
        sources=sources,
        environment=environment,
        resource_policy=resource_policy,
        development_rows=document.get("development_rows"),
        vault_rows=document.get("vault_rows"),
    )


def _bind_sources(entries: Any, root: Path) -> tuple[SourceBinding, ...]:
    if not isinstance(entries, dict):
        raise TypeError("sources must map roles to repo-relative paths")
    allowed = set(SOURCE_ROLES) | set(OPTIONAL_ROLES)
    if not set(SOURCE_ROLES) <= set(entries) or not set(entries) <= allowed:
        raise ValueError(f"sources must declare {SOURCE_ROLES} and optionally {OPTIONAL_ROLES}")
    bindings = []
    for role in entries:
        relative = entries[role]
        if not isinstance(relative, str):
            raise TypeError("source paths must be strings")
        _relative(relative)
        target = _validate_managed_path(root / relative, "full source", root=root)
        if not target.is_file():
            raise ValueError(f"source file is missing: {relative}")
        digest = _file_hash(target)
        bindings.append(SourceBinding(role, relative, digest))
    return tuple(bindings)


def _candidate(entry: Any) -> FullCandidateSnapshot:
    if not isinstance(entry, dict):
        raise TypeError("each candidate must be a mapping")
    experiment_id = entry["experiment_id"]
    arms = tuple(entry.get("arms", ("control", "challenger")))
    selector_arms = tuple(entry.get("selector_arms", ()))
    study = None
    study_method = None
    if "study" in entry:
        study_document = entry["study"]
        study = StudySpec(
            n_trials=study_document["n_trials"],
            inner_folds=study_document["inner_folds"],
            seed=study_document["seed"],
            parameter_lists=tuple(
                (name, tuple(values)) for name, values in study_document["parameter_lists"].items()
            ),
        )
        study_method = study_document.get("method", "random_search")
    spec = CandidateSpec(
        experiment_id=experiment_id,
        feature_pool=tuple(entry["feature_pool"]),
        model_family=entry["model_family"],
        model_params=tuple((k, v) for k, v in entry.get("model_params", {}).items()),
        method_params=tuple((k, tuple(v) if isinstance(v, list) else v)
                            for k, v in entry.get("method_params", {}).items()),
    )
    return FullCandidateSnapshot(
        spec=spec, arms=arms, source_notebook=entry["source_notebook"],
        source_cell_id=entry["source_cell_id"], study=study,
        study_method=study_method, selector_arms=selector_arms,
    )


def inspect(config_path: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    encoded = Path(config_path).read_bytes()
    document = yaml.safe_load(encoded)
    spec = _spec_from_document(document, root)
    report = preflight_full_campaign(root, FullCampaignIntent(
        spec, True, resource_opt_out_record=document["resource_opt_out_record"]))
    capability = resolve_full_campaign_capability(root, experiment_ids=tuple(
        snapshot.spec.experiment_id for snapshot in spec.candidates))
    return {
        "status": ("ready_for_ordinary_transport" if capability.route_id == "reviewed_ordinary_transport_codec_v1"
                   else "ready_for_reviewed_transport"),
        "route_id": capability.route_id,
        "intent_sha256": hashlib.sha256(encoded).hexdigest(),
        "snapshot_encoding": "full_snapshot_codec_v1",
        "run_id": spec.run_id,
        "blocker_codes": list(capability.blocker_codes),
        "data_ready_experiment_ids": list(report.ready_experiment_ids),
        "rows": [
            {"experiment_id": row.record.experiment_id, "status": row.record.status,
             "blockers": list(row.record.blocker_codes)}
            for row in report.rows
        ],
        "execution_authorized": False,
        "n03_eligible": False,
    }


def launch(config_path: str | Path, project_root: str | Path, *, resume: bool = False,
           retry_reason: str | None = None) -> dict[str, Any]:
    """Launch only the genuinely admitted ordinary route, never a readiness report."""
    from project.experiments.full_candidate_runner import run_full_campaign

    root = _validate_managed_path(project_root, "full project root", directory=True)
    config = _validate_managed_path(config_path, "full intent config")
    document = yaml.safe_load(config.read_text())
    spec = _spec_from_document(document, root)
    admitted = preflight_full_campaign(root, FullCampaignIntent(
        spec, True, resource_opt_out_record=document["resource_opt_out_record"]))
    relative = document.get("run_directory")
    if not isinstance(relative, str) or not relative.strip() or Path(relative).is_absolute():
        raise ValueError("explicit repo-relative run_directory required")
    destination = _validate_managed_path(root / relative, "full campaign destination", root=root, directory=True)
    if resume:
        from project.experiments.full_candidate_runner import resume_full_campaign
        return resume_full_campaign(spec, root, destination, data_preflight=admitted,
                                    capability=resolve_full_campaign_capability(root, experiment_ids=tuple(
                                        snapshot.spec.experiment_id for snapshot in spec.candidates)), retry_reason=retry_reason)
    if retry_reason is not None:
        raise ValueError("retry reason only valid for explicit resume")
    return run_full_campaign(spec, root, destination, data_preflight=admitted,
                             capability=resolve_full_campaign_capability(root, experiment_ids=tuple(
                                 snapshot.spec.experiment_id for snapshot in spec.candidates)))
