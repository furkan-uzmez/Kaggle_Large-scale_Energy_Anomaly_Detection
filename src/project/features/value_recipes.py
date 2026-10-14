"""FIXED-boundary value recipes: declared ratio arithmetic and evidenced sentinels.

Both recipes are ``FIXED``: nothing is learned from data, so ``fit`` only
validates the frame, verifies lineage/evidence and snapshots the schema. Neither
recipe reads labels, group metadata, history or held-out rows.

Ratio appends ``meter_reading_per_square_foot`` (float64) and
``meter_reading_per_square_foot_invalid`` (uint8) and leaves every raw predictor
unchanged. A finite meter divided by a finite area > 0 yields the quotient;
negative and zero meters are legitimate measurements and stay valid. A missing
meter, a missing or nonpositive area, or an arithmetic overflow yields NaN with
flag 1 rather than an invented number, and nonmissing raw infinities fail
validation instead of being silently converted. Invalid ratios are never
infilled here: a later imputer remains the single owner of infill.

The sentinel recipe replaces *only* exact finite numeric equality matches that a
``SentinelRule`` declares, and only after the rule's confined repo-relative
evidence file verifies against its declared SHA-256. It performs no range
inference, never treats negatives, zeros or generic missing values as invalid,
deletes no rows and emits no learned outlier flag. A missing rule, missing
evidence, unconfined evidence or a digest mismatch raises
``BlockedExperimentError`` instead of guessing a sentinel. The matched raw
control arm keeps the original values, so the two arms differ only by the
declared factor.
"""

from __future__ import annotations

import errno
import hashlib
import math
import os
import re
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from project.experiments.candidate_contracts import BlockedExperimentError
from project.features.registry import (
    AvailabilityStatus,
    FeatureMetadata,
    FeatureRegistry,
    ProvenanceStatus,
    ResourceCost,
    TargetDependence,
    TransformFitBoundary,
    build_lead_baseline_registry,
)

RATIO_RECIPE = "feat_meter_deviation_ratio"
SENTINEL_RECIPE = "feat_preserve_raw_vs_invalid_sentinel"
RECIPE_IDS = (RATIO_RECIPE, SENTINEL_RECIPE)

METER_COLUMN = "meter_reading"
AREA_COLUMN = "square_feet"
RATIO_NAME = "meter_reading_per_square_foot"
FLAG_NAME = "meter_reading_per_square_foot_invalid"
RATIO_SOURCES = (METER_COLUMN, AREA_COLUMN)

MAX_EVIDENCE_BYTES = 1024 * 1024
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_RATIO_FALLBACK = (
    "Missing meter, missing or nonpositive area, and arithmetic overflow yield a missing "
    "ratio with invalid flag 1; invalid ratios are never infilled by this recipe."
)
_RATIO_SEMANTICS = {
    RATIO_NAME: (
        "Fixed meter_reading divided by a finite square_feet area greater than zero; "
        "negative and zero meter readings are valid measurements."
    ),
    FLAG_NAME: (
        "Fixed indicator that the declared ratio was not computable because the meter was "
        "missing, the area was missing or nonpositive, or the quotient overflowed float64."
    ),
}


# --- Shared local frame validation ---------------------------------------------


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Validate raw numeric values without changing their scalar identities."""
    column = frame[name]
    if not is_numeric_dtype(column.dtype) or is_bool_dtype(column.dtype) or is_complex_dtype(column.dtype):
        raise TypeError(f"{name} must have a real numeric, nonboolean dtype")
    if np.isinf(column).any():
        raise ValueError(f"{name} contains nonfinite nonmissing values")
    return column


def _numeric_values(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Convert validated ratio operands to the declared float64 arithmetic dtype."""
    return _numeric_column(frame, name).to_numpy(dtype=float, na_value=np.nan)


def _validate_frame(frame: object, required: tuple[str, ...], label: str) -> pd.DataFrame:
    """Accept any ordered subset of approved raw predictors covering ``required``."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} input must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError(f"{label} input columns must be unique")
    approved = set(build_lead_baseline_registry().names)
    if any(not isinstance(name, str) or name not in approved for name in frame.columns):
        raise ValueError(f"{label} schema columns must be approved baseline predictors only")
    for name in required:
        if name not in frame.columns:
            raise ValueError(f"required {label} column missing: {name}")
    if not frame.index.is_unique or frame.index.hasnans:
        raise ValueError(f"{label} index must be unique and nonnull")
    return frame


def _verified_baseline_source(name: str) -> FeatureMetadata:
    """Return a raw producer definition only when its lineage is still trusted."""
    metadata = build_lead_baseline_registry().get(name)
    if (metadata is None
            or metadata.provenance_status is not ProvenanceStatus.VERIFIED
            or metadata.prediction_time_availability is not AvailabilityStatus.AVAILABLE
            or metadata.target_dependence is not TargetDependence.INDEPENDENT
            or metadata.source_columns != (name,)):
        raise ValueError(f"unsafe baseline source lineage/provenance/availability/target: {name}")
    return metadata


# --- Recipe registries ---------------------------------------------------------


def build_value_recipe_registry(recipe_id: str, rule: SentinelRule | None = None) -> FeatureRegistry:
    """Return authoritative lineage for one declared FIXED value recipe.

    No caller flag can promote an unsafe raw producer. The sentinel recipe
    requires an explicit rule: an absent rule is an explicit blocker, never an
    inferred default sentinel.
    """
    if not isinstance(recipe_id, str) or recipe_id not in RECIPE_IDS:
        raise ValueError(f"unsupported value recipe identifier: {recipe_id!r}")
    if recipe_id == RATIO_RECIPE:
        if rule is not None:
            raise ValueError("the ratio recipe accepts no sentinel rule")
        return _ratio_registry()
    if rule is None:
        raise BlockedExperimentError(
            "missing_sentinel_rule",
            "the invalid-sentinel recipe requires an explicit evidenced SentinelRule",
        )
    if not isinstance(rule, SentinelRule):
        raise TypeError("rule must be a typed SentinelRule")
    return _sentinel_registry(rule)


def _ratio_registry() -> FeatureRegistry:
    baseline = build_lead_baseline_registry()
    for name in RATIO_SOURCES:
        _verified_baseline_source(name)
    dates = [baseline.get(name).available_at for name in RATIO_SOURCES
             if baseline.get(name).available_at is not None]
    definitions = [baseline.get(name) for name in baseline.names]
    definitions.extend(
        FeatureMetadata(
            name=name, family="candidate_value_recipe", source_columns=RATIO_SOURCES,
            semantic_meaning=_RATIO_SEMANTICS[name],
            prediction_time_availability=AvailabilityStatus.AVAILABLE,
            target_dependence=TargetDependence.INDEPENDENT,
            transform_fit_boundary=TransformFitBoundary.FIXED,
            fallback_behavior=_RATIO_FALLBACK,
            resource_cost=ResourceCost.LOW, provenance_status=ProvenanceStatus.VERIFIED,
            available_at=max(dates) if dates else None,
        )
        for name in (RATIO_NAME, FLAG_NAME)
    )
    return FeatureRegistry(definitions)


def _sentinel_registry(rule: SentinelRule) -> FeatureRegistry:
    """Return baseline lineage with the declared column marked FIXED and evidenced.

    No feature is added: the recipe rewrites declared values in place, so the
    schema is unchanged and the evidence is recorded on the raw producer itself.
    """
    baseline = build_lead_baseline_registry()
    source = _verified_baseline_source(rule.column)
    semantics = (
        f"{source.semantic_meaning} Fixed invalid-value sentinel recipe replaces exactly "
        f"{rule.invalid_values!r} with {rule.replacement_value!r}; evidence "
        f"{rule.evidence_reference} sha256 {rule.evidence_sha256}."
    )
    definitions = []
    for name in baseline.names:
        metadata = baseline.get(name)
        if name == rule.column:
            metadata = FeatureMetadata(
                name=metadata.name, family=metadata.family,
                source_columns=metadata.source_columns, semantic_meaning=semantics,
                prediction_time_availability=metadata.prediction_time_availability,
                target_dependence=metadata.target_dependence,
                transform_fit_boundary=TransformFitBoundary.FIXED,
                fallback_behavior=(
                    "Only exactly matched declared values are replaced; generic missing values, "
                    "negatives and zeros stay unchanged and no row is deleted."
                ),
                resource_cost=metadata.resource_cost,
                provenance_status=metadata.provenance_status,
                available_at=metadata.available_at,
            )
        definitions.append(metadata)
    return FeatureRegistry(definitions)


# --- Fixed meter/area ratio ----------------------------------------------------


class MeterAreaRatioTransformer(TransformerMixin, BaseEstimator):
    """Append the declared meter/area ratio and its invalid flag.

    The transformer is parameterless by design: the arithmetic is fixed, so
    there is no tunable threshold that could silently change the comparison.
    ``fit`` learns nothing and rejects empty training frames; ``transform``
    accepts zero rows and returns the correctly typed schema.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> MeterAreaRatioTransformer:
        frame = _validate_frame(X, RATIO_SOURCES, "ratio")
        if len(frame) == 0:
            raise ValueError("ratio fit requires nonempty rows")
        for name in RATIO_SOURCES:
            _numeric_values(frame, name)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(frame.columns)
        self.output_features_ = (*tuple(frame.columns), RATIO_NAME, FLAG_NAME)
        self.registry_ = _ratio_registry()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "output_features_")
        frame = _validate_frame(X, RATIO_SOURCES, "ratio")
        if tuple(frame.columns) != tuple(self.feature_names_in_):
            raise ValueError("ratio transform schema must match the ordered fit columns")
        meter = _numeric_values(frame, METER_COLUMN)
        area = _numeric_values(frame, AREA_COLUMN)
        computable = ~np.isnan(meter) & ~np.isnan(area) & (area > 0.0)
        ratio = np.full(meter.shape, np.nan, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            np.divide(meter, area, out=ratio, where=computable)
        overflowed = computable & ~np.isfinite(ratio)
        ratio[overflowed] = np.nan
        output = frame.copy(deep=True)
        output[RATIO_NAME] = pd.Series(ratio, index=frame.index, dtype=np.float64)
        output[FLAG_NAME] = pd.Series(
            (~computable | overflowed).astype(np.uint8), index=frame.index, dtype=np.uint8
        )
        return output

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "output_features_")
        if input_features is not None and tuple(input_features) != tuple(self.feature_names_in_):
            raise ValueError("input_features must match the ordered fit schema")
        return np.asarray(self.output_features_, dtype=object)


# --- Evidenced invalid-value sentinel ------------------------------------------


def _require_finite_scalar(value: object, label: str) -> int | float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a finite nonboolean real number")
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite nonboolean real number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must be a finite nonboolean real number")
    return value


def _validate_reference(reference: object) -> None:
    if not isinstance(reference, str):
        raise TypeError("evidence_reference must be a confined repo-relative string path")
    problem = (
        not reference
        or reference != reference.strip()
        or "\\" in reference
        or reference.startswith(("/", "~"))
        or any(not segment or segment in (".", "..") for segment in reference.split("/"))
    )
    if problem or Path(reference).is_absolute():
        raise ValueError(
            "evidence_reference must be a confined repo-relative path without absolute, "
            "home, parent or empty segments"
        )


@dataclass(frozen=True)
class SentinelRule:
    """Declare exact finite invalid values, their sentinel and hashed evidence.

    The rule is deliberately a closed enumeration. It cannot express a range, a
    quantile threshold or a learned outlier criterion, so no caller can turn
    this recipe into an undeclared outlier filter.
    """

    column: str
    invalid_values: tuple[float, ...]
    replacement_value: float
    evidence_reference: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column.strip():
            raise ValueError("column must be a nonempty authoritative raw column name")
        try:
            source = _verified_baseline_source(self.column)
        except ValueError as error:
            raise ValueError(f"column is not an authoritative verified raw column: {self.column}") from error
        if source.family == "categorical":
            raise ValueError(f"column must be numeric for exact sentinel equality: {self.column}")

        if not isinstance(self.invalid_values, tuple):
            raise TypeError("invalid_values must be an immutable tuple of finite numbers")
        if not self.invalid_values:
            raise ValueError("invalid_values must declare at least one exact invalid value")
        declared = []
        for value in self.invalid_values:
            try:
                declared.append(_require_finite_scalar(value, "invalid_values"))
            except TypeError as error:
                raise ValueError("invalid_values must contain finite nonboolean real numbers") from error
        if len(set(declared)) != len(declared):
            raise ValueError("invalid_values must be unique")

        replacement = _require_finite_scalar(self.replacement_value, "replacement_value")
        if replacement in set(declared):
            raise ValueError("replacement_value must be distinct from every declared invalid value")

        _validate_reference(self.evidence_reference)
        if not isinstance(self.evidence_sha256, str):
            raise TypeError("evidence_sha256 must be a lowercase hexadecimal SHA-256 string")
        if not _HEX64.fullmatch(self.evidence_sha256):
            raise ValueError("evidence_sha256 must be 64 lowercase hexadecimal characters")


def _same_registry(submitted: FeatureRegistry, expected: FeatureRegistry) -> bool:
    return (submitted.names == expected.names
            and all(submitted.get(name) == expected.get(name) for name in expected.names))


def _verify_evidence(rule: SentinelRule, root: Path) -> str:
    """Hash a bounded regular file opened beneath pinned, no-follow directories."""
    try:
        with ExitStack() as stack:
            resolved_root = Path(root).resolve(strict=True)
            directory = os.open(resolved_root.anchor, os.O_RDONLY | os.O_DIRECTORY)
            stack.callback(os.close, directory)
            segments = (*resolved_root.parts[1:], *rule.evidence_reference.split("/"))
            for segment in segments[:-1]:
                directory = os.open(segment, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=directory)
                stack.callback(os.close, directory)
            descriptor = os.open(segments[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=directory)
            stack.callback(os.close, descriptor)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise BlockedExperimentError("missing_sentinel_evidence", "evidence must be a regular file")
            if info.st_size > MAX_EVIDENCE_BYTES:
                raise BlockedExperimentError("oversized_sentinel_evidence", "evidence exceeds the byte limit")
            payload = bytearray()
            while len(payload) <= MAX_EVIDENCE_BYTES:
                chunk = os.read(descriptor, min(65536, MAX_EVIDENCE_BYTES + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
            if len(payload) > MAX_EVIDENCE_BYTES or os.fstat(descriptor).st_size > MAX_EVIDENCE_BYTES:
                raise BlockedExperimentError("oversized_sentinel_evidence", "evidence exceeds the byte limit")
            digest = hashlib.sha256(payload).hexdigest()
    except OSError as error:
        code = ("unconfined_sentinel_evidence" if error.errno in (errno.ELOOP, errno.ENOTDIR)
                else "missing_sentinel_evidence")
        raise BlockedExperimentError(code, f"cannot safely read evidence: {rule.evidence_reference}") from error
    if digest != rule.evidence_sha256:
        raise BlockedExperimentError(
            "sentinel_evidence_hash_mismatch",
            f"sentinel evidence digest disagrees with the declared SHA-256: {rule.evidence_reference}",
        )
    return digest


def _typed_replacement(column: pd.Series, rule: SentinelRule) -> Any:
    """Return the sentinel in the column's own dtype, or block on precision loss."""
    try:
        probe = pd.Series([rule.replacement_value], dtype=column.dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise BlockedExperimentError(
            "sentinel_dtype_conflict",
            f"sentinel {rule.replacement_value!r} is not storable in {column.dtype}",
        ) from error
    stored = probe.iloc[0]
    if _require_finite_scalar(stored, "stored replacement") != _require_finite_scalar(
        rule.replacement_value, "replacement_value"
    ):
        raise BlockedExperimentError(
            "sentinel_dtype_conflict",
            f"sentinel {rule.replacement_value!r} loses precision in {column.dtype}",
        )
    return stored


class EvidencedSentinelTransformer(TransformerMixin, BaseEstimator):
    """Replace exactly the declared, evidenced invalid values on a copy.

    ``registry`` must be the authoritative registry this exact rule produces;
    a caller-built lookalike cannot authorize a rewritten raw producer.
    ``evidence_root`` confines evidence resolution, so tests verify real hashed
    bytes in a sandbox without approving any real data defect.
    """

    def __init__(self, rule: SentinelRule | None, registry: FeatureRegistry,
                 evidence_root: Path | None = None) -> None:
        self.rule = rule
        self.registry = registry
        self.evidence_root = evidence_root

    def fit(self, X: pd.DataFrame, y: Any = None) -> EvidencedSentinelTransformer:
        rule = self._checked_rule()
        if not isinstance(self.registry, FeatureRegistry):
            raise TypeError("registry must be an authoritative FeatureRegistry")
        if not _same_registry(self.registry, _sentinel_registry(rule)):
            raise BlockedExperimentError(
                "unverified_sentinel_registry",
                "submitted registry does not match this rule's authoritative lineage",
            )
        frame = _validate_frame(X, (rule.column,), "sentinel")
        if len(frame) == 0:
            raise ValueError("sentinel fit requires nonempty rows")
        _numeric_column(frame, rule.column)
        _typed_replacement(frame[rule.column], rule)
        self.evidence_sha256_ = _verify_evidence(rule, self._checked_root())
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(frame.columns)
        self.output_features_ = tuple(frame.columns)
        self.rule_ = rule
        self.evidence_root_ = self.evidence_root
        self.registry_ = self.registry
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_unchanged_parameters()
        rule = self.rule_
        frame = _validate_frame(X, (rule.column,), "sentinel")
        if tuple(frame.columns) != tuple(self.feature_names_in_):
            raise ValueError("sentinel transform schema must match the ordered fit columns")
        column = frame[rule.column]
        _numeric_column(frame, rule.column)
        replacement = _typed_replacement(column, rule)
        declared = {_require_finite_scalar(value, "invalid_values") for value in rule.invalid_values}
        matched = np.asarray([
            False if pd.isna(value) else _require_finite_scalar(value, rule.column) in declared
            for value in column
        ], dtype=bool)
        output = frame.copy(deep=True)
        if matched.any():
            replaced = column.mask(pd.Series(matched, index=frame.index), replacement)
            if replaced.dtype != column.dtype:
                replaced = replaced.astype(column.dtype)
            output[rule.column] = replaced
        return output

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        self._check_unchanged_parameters()
        if input_features is not None and tuple(input_features) != tuple(self.feature_names_in_):
            raise ValueError("input_features must match the ordered fit schema")
        return np.asarray(self.output_features_, dtype=object)

    def _checked_rule(self) -> SentinelRule:
        if self.rule is None:
            raise BlockedExperimentError(
                "missing_sentinel_rule",
                "an explicit evidenced SentinelRule is required; no sentinel is inferred",
            )
        if not isinstance(self.rule, SentinelRule):
            raise TypeError("rule must be a typed SentinelRule")
        return self.rule

    def _checked_root(self) -> Path:
        root = self.evidence_root
        if root is None:
            from project.paths import find_project_root

            root = find_project_root()
        if not isinstance(root, Path):
            raise TypeError("evidence_root must be a filesystem Path confining the evidence")
        if not root.is_dir():
            raise BlockedExperimentError(
                "missing_sentinel_evidence", f"sentinel evidence root is absent: {root}"
            )
        return root

    def _check_unchanged_parameters(self) -> None:
        check_is_fitted(self, "output_features_")
        if (self.rule, self.evidence_root) != (self.rule_, self.evidence_root_):
            raise ValueError("sentinel parameter changed; fit again before use")
        if self.registry is not self.registry_:
            raise ValueError("sentinel parameter changed; fit again before use")
