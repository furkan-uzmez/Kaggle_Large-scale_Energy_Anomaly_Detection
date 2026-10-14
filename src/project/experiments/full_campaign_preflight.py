"""Read-only projected full admission, separate from source-only inspection.

No run/index/journal directories, fits or pilot artifacts are created. Dataset
scans use bounded Arrow batches, never a retained Python set of natural keys.
Native n_unique still retains population-scale engine state: a scalar result
does not imply constant-memory uniqueness or a total-RAM guarantee.
Data evidence is process-local and is never launch authorization. Separate
admission authenticates the fixed ordinary transport route. Comparative
completion additionally authenticates durable parent authority and independently
rechecks the full receipt; readiness alone never supplies that authority.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import os
import platform
import secrets
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from project.experiments.candidate_contracts import _validate_managed_path
from project.experiments.full_campaign_admission import (
    FullAdmissionRow,
    FullCampaignIntent,
    inspect_full_campaign,
)
from project.experiments.full_input_schema import validate_input_dtypes
from project.experiments.full_run_contracts import (
    AdmittedFullRun,
    FullRunSpec,
    FullStatusRecord,
    OrdinaryCompletionAdmission,
    SourceBinding,
    _file_hash,
    _json,
)

MAX_BATCH_ROWS = 64_000
_SEAL_KEY = secrets.token_bytes(32)
_PACKAGES = ("numpy", "pandas", "polars", "pyarrow", "PyYAML", "scikit-learn",
             "scipy", "joblib", "threadpoolctl")
_THREAD_SETTINGS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS")
_SCIENTIFIC_FILES = (
    "scripts/train.py",
    "src/project/experiments/candidate_contracts.py",
    "src/project/experiments/candidate_dispatch.py",
    "src/project/experiments/fit_accounting.py",
    "src/project/experiments/inner_cv.py",
    "src/project/experiments/nested_tuning.py",
    "src/project/experiments/full_run_contracts.py",
    "src/project/experiments/full_campaign_admission.py",
    "src/project/experiments/full_campaign_preflight.py",
    "src/project/experiments/full_input_schema.py",
    "src/project/experiments/full_candidate_runner.py",
    "src/project/experiments/full_campaign_runtime.py",
    "src/project/experiments/full_candidate_io.py",
    "src/project/experiments/full_campaign_cli.py",
    "scripts/campaign.py",
)


@dataclass(frozen=True)
class FullDataPreflight:
    """Authenticated data-stage evidence only, deliberately not AdmittedFullRun.

    Parent may verify this before index preparation, but must not pass it to a
    launch boundary. The inventory retains full_dispatch_unavailable even for
    scientifically/data-ready selected rows.
    """

    spec: FullRunSpec
    source_root: str
    rows: tuple[FullAdmissionRow, ...]
    implementation_sources: tuple[SourceBinding, ...]
    actual_environment: tuple[tuple[str, str], ...]
    resource_opt_out_record: str
    train_rows: int
    timestamp_unit: str
    timestamp_time_zone: str | None
    data_ready_experiment_ids: tuple[str, ...]
    seal: str
    canonical_timestamp_unit: str = field(default="ns", init=False)
    execution_authorized: bool = field(default=False, init=False)


def current_environment() -> tuple[tuple[str, str], ...]:
    """Versions and effective thread settings; never impose thread caps."""
    values = [("python", platform.python_version())]
    values.extend((name, importlib.metadata.version(name)) for name in _PACKAGES)
    values.extend((name, os.environ.get(name, "unset")) for name in _THREAD_SETTINGS)
    return tuple(sorted(values))


def _implementation_bindings(root: Path) -> tuple[SourceBinding, ...]:
    # Discover the scientific package inventory independently of caller input.
    paths = set(_SCIENTIFIC_FILES)
    # Bind the whole execution closure, including future worker/evidence modules
    # when present. Presence/hashes alone never prove an executable capability.
    from project.experiments.full_campaign_capabilities import _UNPROMOTED_MODULES
    paths.update(str(p.relative_to(root)) for p in (root / "src/project").rglob("*.py")
                 if str(p.relative_to(root)) not in _UNPROMOTED_MODULES)
    paths.update(str(p.relative_to(root)) for p in (root / "scripts").rglob("*.py"))
    bindings = []
    for relative in sorted(paths):
        path = _validate_managed_path(root / relative, "scientific source", root=root)
        if not path.is_file():
            raise ValueError(f"scientific implementation missing: {relative}")
        bindings.append(SourceBinding("implementation", relative, _file_hash(path)))
    return tuple(bindings)


def _seal(evidence: FullDataPreflight) -> str:
    payload = asdict(evidence)
    payload.pop("seal")
    domain = type(evidence).__name__ + ":"
    return hmac.new(_SEAL_KEY, (domain + _json(payload)).encode(), hashlib.sha256).hexdigest()


def _seal_admitted(evidence: AdmittedFullRun) -> str:
    payload = asdict(evidence)
    payload.pop("seal")
    domain = type(evidence).__name__ + ":"
    return hmac.new(_SEAL_KEY, (domain + _json(payload)).encode(), hashlib.sha256).hexdigest()


def verify_admitted_full_run(
    root: str | Path, admitted: AdmittedFullRun, *, spec: FullRunSpec,
    phase: str = "launch",
    completion_admission: OrdinaryCompletionAdmission | None = None,
) -> None:
    """Authenticate fresh bindings; completion additionally requires durable parent proof."""
    if type(admitted) is not AdmittedFullRun or type(spec) is not FullRunSpec:
        raise TypeError("authentic AdmittedFullRun and exact FullRunSpec required")
    if phase not in ("launch", "resume", "completion"):
        raise ValueError("invalid admission verification phase")
    from project.experiments.full_campaign_capabilities import (
        build_reviewed_route_arms,
        resolve_full_campaign_capability,
        verify_full_campaign_capability,
    )
    capability = resolve_full_campaign_capability(root, experiment_ids=tuple(
        snapshot.spec.experiment_id for snapshot in spec.candidates))
    if not capability.available:
        raise ValueError("full_dispatch_unavailable: integrated full worker route required")
    verify_full_campaign_capability(root, capability)
    if not isinstance(admitted.seal, str) or not hmac.compare_digest(admitted.seal, _seal_admitted(admitted)):
        raise ValueError("forged or mutated admitted full run")
    source_root = _validate_managed_path(root, "source root", directory=True)
    if str(source_root) != admitted.source_root or spec.binding_sha256 != admitted.spec.binding_sha256:
        raise ValueError("admission binding mismatch")
    if current_environment() != admitted.actual_environment:
        raise ValueError("admission environment changed")
    spec.verify_bindings(source_root)
    if _implementation_bindings(source_root) != admitted.implementation_sources:
        raise ValueError("scientific implementation binding changed")
    selected = tuple(candidate.spec.experiment_id for candidate in spec.candidates)
    if (admitted.ready_experiment_ids != selected
            or not set(selected) <= set(capability.supported_experiment_ids)):
        raise ValueError("selected route readiness mismatch")
    from project.experiments.full_run_contracts import source_matrix_identity
    source_selected = {source_matrix_identity(identity) for identity in selected}
    for row in admitted.rows:
        if row.record.experiment_id in source_selected and (row.record.status != "ready" or row.record.blocker_codes):
            raise ValueError("admitted readiness retains execution blockers")
    _verify_encoding_source(source_root, spec)
    _verify_route_extensions(source_root, spec)
    from project.experiments.full_candidate_runner import _encoding_provenance
    encoding, _, _ = _encoding_provenance(source_root, spec)
    for snapshot in spec.candidates:
        build_reviewed_route_arms(snapshot, year_built_encoding=encoding)
    if phase == "completion":
        if type(completion_admission) is not OrdinaryCompletionAdmission:
            raise ValueError("comparative_evidence_unavailable: authentic durable parent completion admission required")
        from project.experiments.full_campaign_evidence import (
            verify_completion_admission,
        )
        verify_completion_admission(source_root, admitted, completion_admission)


def _verify_encoding_source(root: Path, spec: FullRunSpec) -> SourceBinding:
    from project.experiments.full_candidate_runner import _encoding_provenance
    from project.experiments.full_snapshot_codec import PROVENANCE_PATH

    _encoding_provenance(root, spec)
    binding = next(source for source in spec.sources if source.role == "encoding")
    reference = Path(__file__).resolve().parents[3] / PROVENANCE_PATH
    if binding.sha256 != _file_hash(reference):
        raise ValueError("encoding provenance differs from authoritative reviewed evidence")
    return binding


def _verify_route_extensions(root: Path, spec: FullRunSpec) -> None:
    """Authenticate isolated profile/environment independently of parent imports."""
    profiles = tuple(source for source in spec.sources if source.role == "autogluon_profile")
    selected = tuple(snapshot.spec.experiment_id for snapshot in spec.candidates)
    if selected == ("model_autogluon_core_features",):
        from project.experiments.full_autogluon_contract import (
            verify_autogluon_environment, verify_autogluon_profile_source,
        )
        verify_autogluon_profile_source(spec, root)
        verify_autogluon_environment(root)
    elif profiles:
        raise ValueError("autogluon profile source is unsupported for selected route")


def verify_full_data_preflight(
    root: str | Path, evidence: FullDataPreflight, *, spec: FullRunSpec,
) -> None:
    """Recheck exact data-stage bindings; this never verifies launch permission."""
    if type(evidence) is not FullDataPreflight or type(spec) is not FullRunSpec:
        raise TypeError("authentic FullDataPreflight and exact FullRunSpec required")
    if not isinstance(evidence.seal, str) or not hmac.compare_digest(evidence.seal, _seal(evidence)):
        raise ValueError("forged or mutated data preflight")
    source_root = _validate_managed_path(root, "source root", directory=True)
    if str(source_root) != evidence.source_root or spec.binding_sha256 != evidence.spec.binding_sha256:
        raise ValueError("admission binding mismatch")
    if current_environment() != evidence.actual_environment:
        raise ValueError("admission environment changed")
    spec.verify_bindings(source_root)
    if _implementation_bindings(source_root) != evidence.implementation_sources:
        raise ValueError("scientific implementation binding changed")


def _assignments(path: Path) -> pl.DataFrame:
    schema = pl.read_parquet_schema(path)
    required = ("building_id", "repeat", "seed", "split", "fold")
    if any(name not in schema for name in required):
        raise ValueError("split schema incomplete")
    if schema["building_id"] not in (pl.String, pl.Int64):
        raise ValueError("opaque building dtype unsupported")
    if any(not schema[name].is_integer() for name in ("repeat", "seed", "fold")):
        raise ValueError("split protocol must use integer columns")
    manifest = pl.scan_parquet(path).select(required).collect(engine="streaming")
    if manifest.is_empty() or any(manifest[name].null_count() for name in required[:-1]):
        raise ValueError("null or empty split assignments")
    if manifest.select(pl.struct("building_id", "repeat").n_unique()).item() != manifest.height:
        raise ValueError("duplicate building/repeat assignment")
    if sorted(manifest["repeat"].unique().to_list()) != [0, 1, 2]:
        raise ValueError("split repeat contract mismatch")
    reference = manifest.filter(pl.col("repeat") == 0).select("building_id", "split").sort("building_id")
    signatures = set()
    for repeat, seed in enumerate((42, 43, 44)):
        part = manifest.filter(pl.col("repeat") == repeat)
        if part["seed"].unique().to_list() != [seed]:
            raise ValueError("split seed mismatch")
        if not part.select("building_id", "split").sort("building_id").equals(reference):
            raise ValueError("population membership differs across repeats")
        if part.filter(~pl.col("split").is_in(["dev", "vault"])).height:
            raise ValueError("unknown split membership")
        dev = part.filter(pl.col("split") == "dev")
        vault = part.filter(pl.col("split") == "vault")
        if dev["fold"].null_count() or sorted(dev["fold"].unique().to_list()) != [0, 1, 2, 3]:
            raise ValueError("outer fold contract mismatch")
        if vault["fold"].null_count() != vault.height:
            raise ValueError("vault folds must be null")
        # Only building-scale sets: compare partition membership independently
        # of arbitrary fold labels, including nonadjacent repeated partitions.
        signature = frozenset(frozenset(dev.filter(pl.col("fold") == fold)["building_id"])
                              for fold in range(4))
        if signature in signatures:
            raise ValueError("repeated partitions must be genuinely distinct")
        signatures.add(signature)
    return manifest


def _data_checks(root: Path, spec: FullRunSpec) -> tuple[int, str, str | None]:
    sources = {s.role: s for s in spec.sources}
    data_path = sources["data"].verify(root)
    manifest = _assignments(sources["split"].verify(root))
    parquet = pq.ParquetFile(data_path)
    schema = pl.read_parquet_schema(data_path)
    required = {"building_id", "timestamp", "anomaly"} | {
        feature for candidate in spec.candidates for feature in candidate.spec.feature_pool
    }
    if required - set(schema):
        raise ValueError("prepared schema missing required features or keys")
    validate_input_dtypes(schema)
    if schema["building_id"] != manifest.schema["building_id"]:
        raise ValueError("building dtype differs between data and split")
    timestamp = schema["timestamp"]
    if not isinstance(timestamp, pl.Datetime) or timestamp.time_unit not in ("ns", "us"):
        raise ValueError("timestamp must have lossless ns/us canonical policy")
    if not schema["anomaly"].is_integer():
        raise ValueError("target must be integer binary")
    membership = manifest.filter(pl.col("repeat") == 0).select("building_id", "split")
    counts = {"dev": 0, "vault": 0}
    support = {}
    # Building-scale sufficient statistics prove all outer training partitions
    # can fit these recipes; no held-out values enter fitted transformer state.
    native_columns = tuple(dict.fromkeys(feature for candidate in spec.candidates
        if candidate.spec.experiment_id == "feat_missingness_passthrough_comparison"
        for feature in candidate.spec.feature_pool if feature != "primary_use"))
    histogram = any(c.spec.experiment_id == "feat_histogram_quantile" for c in spec.candidates)
    recipe_columns = tuple(dict.fromkeys((*native_columns, *(("meter_reading",) if histogram else ()))))
    recipe_support = {}
    seen_buildings = set()  # Building-scale metadata only, never row keys.
    check_encoding = "encoding" in sources
    projected = ["building_id", "timestamp", "anomaly"] + (["year_built"] if check_encoding else [])
    projected = list(dict.fromkeys((*projected, *recipe_columns)))
    for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS, columns=projected):
        frame = pl.from_arrow(batch).join(membership, on="building_id", how="left")
        if frame["building_id"].null_count() or frame["timestamp"].null_count() or frame["split"].null_count():
            raise ValueError("null natural key or unassigned building")
        seen_buildings.update(frame["building_id"].unique().to_list())
        physical = frame["timestamp"].cast(pl.Int64)
        if timestamp.time_unit == "us" and (physical.min() < -(2**63) // 1000 + 1 or physical.max() > (2**63 - 1) // 1000):
            raise ValueError("timestamp ns canonicalization would overflow")
        for split in counts:
            counts[split] += frame.filter(pl.col("split") == split).height
        dev = frame.filter(pl.col("split") == "dev")
        if check_encoding:
            year = pl.col("year_built")
            if dev.filter(year.is_not_null() & (
                    ~year.is_finite() | (year < 0) | (year > 255) | (year != year.floor()))).height:
                raise ValueError("year_built values violate immutable offset encoding contract")
        if dev["anomaly"].null_count() or dev.filter(~pl.col("anomaly").is_in([0, 1])).height:
            raise ValueError("development target must be binary 0/1")
        # Only aggregated class counts leave a batch; vault labels are discarded.
        for building, target, count in dev.group_by("building_id", "anomaly").len().iter_rows():
            support[building, target] = support.get((building, target), 0) + count
        for column in recipe_columns:
            finite = dev.filter(pl.col(column).is_not_null() & pl.col(column).is_finite())
            for building, count, low, high in finite.group_by("building_id").agg(
                    pl.len(), pl.col(column).min().alias("low"), pl.col(column).max().alias("high")).iter_rows():
                previous = recipe_support.get((building, column), (0, low, high))
                recipe_support[building, column] = (previous[0] + count, min(previous[1], low), max(previous[2], high))
    if seen_buildings != set(membership["building_id"].to_list()):
        raise ValueError("declared building has no prepared rows")
    if counts != {"dev": spec.development_rows, "vault": spec.vault_rows}:
        raise ValueError("declared development/vault row counts mismatch")
    train_rows = spec.development_rows + spec.vault_rows
    if parquet.metadata.num_rows != train_rows:
        raise ValueError("declared train row count mismatch")
    # No Python key set, but exact native uniqueness may retain O(train_rows)
    # state. Batching this scan does not promise constant overall RAM usage.
    unique = pl.scan_parquet(data_path).select(
        pl.struct("building_id", "timestamp").n_unique()
    ).collect(engine="streaming").item()
    if unique != train_rows:
        raise ValueError("duplicate natural key")
    for repeat in range(3):
        part = manifest.filter((pl.col("repeat") == repeat) & (pl.col("split") == "dev"))
        for fold in range(4):
            for validation in (True, False):
                buildings = part.filter((pl.col("fold") == fold) if validation else (pl.col("fold") != fold))["building_id"]
                if any(sum(support.get((building, label), 0) for building in buildings) == 0 for label in (0, 1)):
                    raise ValueError("outer training/validation class support missing")
                if not validation:
                    for column in recipe_columns:
                        stats = [recipe_support[b, column] for b in buildings if (b, column) in recipe_support]
                        if column in native_columns and not stats:
                            raise ValueError(f"native numeric outer training support missing: {column}")
                        if histogram and column == "meter_reading" and (
                                sum(s[0] for s in stats) < 2 or min(s[1] for s in stats) == max(s[2] for s in stats)):
                            raise ValueError("histogram outer training distribution support missing")
    return train_rows, timestamp.time_unit, timestamp.time_zone


def preflight_full_campaign_data(root: str | Path, intent: FullCampaignIntent) -> FullDataPreflight:
    """Authenticate sources/config/data/classes without authorizing execution.

    All 21 dispositions remain visible. Scientific unresolved rows are never
    authorized; unrequested rows stay blocked with configuration-pending reasons.
    Full dispatch remains blocked even after the data checks pass.
    """
    # The legacy source-only inspector owns matrix/baseline/notebook checks and
    # intentionally has no encoding role. Authenticate that extension here,
    # then inspect its unchanged scientific inputs; retain the original full
    # binding in all issued evidence.
    encoding_sources = tuple(source for source in intent.spec.sources if source.role == "encoding")
    inspection_intent = intent
    if encoding_sources:
        _verify_encoding_source(Path(root), intent.spec)
        inspection_intent = replace(intent, spec=replace(intent.spec, sources=tuple(
            source for source in intent.spec.sources if source.role != "encoding")))
    _verify_route_extensions(Path(root), intent.spec)
    report = inspect_full_campaign(root, inspection_intent)
    if report.blocker_codes:
        raise ValueError("source admission blocked: " + ",".join(report.blocker_codes))
    record = intent.resource_opt_out_record
    if not isinstance(record, str) or not record.strip():
        raise ValueError("explicit user resource opt-out record required")
    actual_environment = current_environment()
    if intent.spec.environment != actual_environment:
        raise ValueError("declared environment does not match actual versions/settings")
    source_root = _validate_managed_path(root, "source root", directory=True)
    intent.spec.verify_bindings(source_root)
    implementation = _implementation_bindings(source_root)
    train_rows, unit, zone = _data_checks(source_root, intent.spec)
    rows = []
    for row in report.rows:
        codes = tuple(code for code in row.record.blocker_codes
                      if code != "real_preflight_pending")
        status = "prohibited" if row.record.status == "prohibited" else "blocked"
        rows.append(replace(row, record=FullStatusRecord(row.record.experiment_id, status, codes)))
    ready = tuple(row.record.experiment_id for row in rows
                  if row.record.status != "prohibited"
                  and row.record.blocker_codes == ("full_dispatch_unavailable",))
    from project.experiments.full_run_contracts import source_matrix_identity
    selected = tuple(candidate.spec.experiment_id for candidate in intent.spec.candidates)
    if not {source_matrix_identity(identity) for identity in selected} <= set(ready):
        raise ValueError("requested experiment retains scientific preflight blockers")
    ready = selected
    evidence = FullDataPreflight(intent.spec, str(source_root), tuple(rows), implementation,
                            actual_environment, record, train_rows, unit, zone, ready, "")
    evidence = replace(evidence, seal=_seal(evidence))
    # Catch mutation across read boundaries before returning data-stage evidence.
    verify_full_data_preflight(source_root, evidence, spec=intent.spec)
    return evidence


def preflight_full_campaign(root: str | Path, intent: FullCampaignIntent) -> AdmittedFullRun:
    """Preflight full campaign and return authentic AdmittedFullRun if route is integrated."""
    from project.experiments.full_campaign_capabilities import (
        build_reviewed_route_arms,
        resolve_full_campaign_capability,
        verify_full_campaign_capability,
    )
    try:
        capability = resolve_full_campaign_capability(root, experiment_ids=tuple(
            snapshot.spec.experiment_id for snapshot in intent.spec.candidates))
    except ValueError as exc:
        raise ValueError(f"ordinary_route_not_reviewed: {exc}") from exc
    if not capability.available:
        raise ValueError("full_dispatch_unavailable: integrated full worker route required")
    verify_full_campaign_capability(root, capability)
    selected = tuple(candidate.spec.experiment_id for candidate in intent.spec.candidates)
    if not set(selected) <= set(capability.supported_experiment_ids):
        raise ValueError("ordinary_route_not_reviewed: selected route not reviewed")
    _verify_encoding_source(Path(root), intent.spec)
    from project.experiments.full_candidate_runner import _encoding_provenance
    encoding, _, _ = _encoding_provenance(Path(root), intent.spec)
    for snapshot in intent.spec.candidates:
        build_reviewed_route_arms(snapshot, year_built_encoding=encoding)
    data_evidence = preflight_full_campaign_data(root, intent)
    from project.experiments.full_run_contracts import source_matrix_identity
    source_selected = {source_matrix_identity(identity) for identity in selected}
    rows = tuple(replace(row, record=FullStatusRecord(row.record.experiment_id, "ready", ()))
                 if row.record.experiment_id in source_selected else row for row in data_evidence.rows)
    admitted = AdmittedFullRun(
        spec=data_evidence.spec,
        source_root=data_evidence.source_root,
        rows=rows,
        implementation_sources=data_evidence.implementation_sources,
        actual_environment=data_evidence.actual_environment,
        resource_opt_out_record=data_evidence.resource_opt_out_record,
        train_rows=data_evidence.train_rows,
        timestamp_unit=data_evidence.timestamp_unit,
        timestamp_time_zone=data_evidence.timestamp_time_zone,
        ready_experiment_ids=data_evidence.data_ready_experiment_ids,
        seal="",
        canonical_timestamp_unit=data_evidence.canonical_timestamp_unit,
    )
    admitted = replace(admitted, seal=_seal_admitted(admitted))
    verify_admitted_full_run(root, admitted, spec=intent.spec, phase="launch")
    return admitted
