"""Synthetic contracts for the fixed ratio and exact-evidence sentinel recipes.

Expectations are hand-written literals, never the transformer reused as its own
oracle. Both recipes are FIXED boundaries: nothing is learned from data, so the
tests assert declared arithmetic and declared evidence instead of fitted
statistics. Evidence files are real bytes hashed with hashlib inside ``tmp_path``;
no repository data, output directory, or real defect is read or approved here.
"""

import hashlib
import importlib
import importlib.util
import io
import os
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from candidate_fixtures import CORE_COLUMNS, core_frame
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
RATIO_NAME = "meter_reading_per_square_foot"
FLAG_NAME = "meter_reading_per_square_foot_invalid"


def api():
    name = "project.features.value_recipes"
    assert importlib.util.find_spec(name), "value recipe API is missing"
    return importlib.import_module(name)


def ratio_transformer():
    return api().MeterAreaRatioTransformer()


def sentinel_rule(**overrides):
    fields = {
        "column": "meter_reading",
        "invalid_values": (-999.0, -9999.0),
        "replacement_value": -1.0,
        "evidence_reference": "docs/evidence/meter_sentinel.json",
        "evidence_sha256": "0" * 64,
    }
    fields.update(overrides)
    return api().SentinelRule(**fields)


def write_evidence(root, reference, payload=b'{"invalid": [-999.0, -9999.0]}'):
    """Create real evidence bytes under a sandbox root and return its digest."""
    path = root.joinpath(*reference.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def evidenced_rule(root, **overrides):
    """Return a rule whose declared digest matches freshly written evidence."""
    reference = overrides.pop("evidence_reference", "docs/evidence/meter_sentinel.json")
    payload = overrides.pop("payload", b'{"invalid": [-999.0, -9999.0]}')
    digest = write_evidence(root, reference, payload)
    return sentinel_rule(evidence_reference=reference, evidence_sha256=digest, **overrides)


def sentinel_transformer(rule, root, registry=None):
    if registry is None:
        registry = api().build_value_recipe_registry(SENTINEL_RECIPE, rule)
    return api().EvidencedSentinelTransformer(rule, registry, evidence_root=root)


def frame_with(meter, area=None, index=None, meter_dtype=None, area_dtype=None):
    """Return a core-schema frame carrying the supplied meter/area values."""
    values = list(meter)
    template = core_frame()
    repeats = max(1, -(-len(values) // len(template)))
    output = pd.concat([template] * repeats, ignore_index=True).iloc[: len(values)].copy()
    output["meter_reading"] = pd.Series(values, index=output.index, dtype=meter_dtype or "float64")
    if area is not None:
        output["square_feet"] = pd.Series(list(area), index=output.index, dtype=area_dtype or "float64")
    if index is not None:
        output.index = pd.Index(index)
    return output


def fitted_names(transformer):
    return tuple(n for n in sorted(vars(transformer)) if n.endswith("_") and not n.startswith("_"))


# --- Ratio: declared arithmetic ------------------------------------------------


def test_ratio_preserves_negative_meter_and_flags_bad_area():
    """The plan's exact acceptance example, asserted as literals."""
    frame = core_frame().iloc[:4].copy()
    frame["meter_reading"] = [-6.0, 2.0, 2.0, np.nan]
    frame["square_feet"] = [3.0, 0.0, -1.0, 2.0]
    output = ratio_transformer().fit_transform(frame)
    np.testing.assert_allclose(
        output.meter_reading_per_square_foot, [-2.0, np.nan, np.nan, np.nan], equal_nan=True
    )
    assert output.meter_reading_per_square_foot_invalid.tolist() == [0, 1, 1, 1]


def test_zero_meter_over_positive_area_is_valid_zero_not_invalid():
    output = ratio_transformer().fit_transform(frame_with([0.0, -0.0], [4.0, 4.0]))
    np.testing.assert_allclose(output[RATIO_NAME], [0.0, 0.0])
    assert output[FLAG_NAME].tolist() == [0, 0]


def test_subnormal_area_overflow_becomes_missing_and_flagged():
    """A finite positive subnormal divisor must not invent an infinite ratio.

    1e308/5e-324 overflows float64, while 1e-300/5e-324 stays finite near 2e23,
    so the same subnormal area must flag only the row that actually overflowed.
    """
    output = ratio_transformer().fit_transform(frame_with([1e308, 1e-300], [5e-324, 5e-324]))
    assert np.isnan(output[RATIO_NAME].to_numpy(dtype=float)).tolist() == [True, False]
    assert output[FLAG_NAME].tolist() == [1, 0]
    np.testing.assert_allclose(output[RATIO_NAME].iloc[1], 2.0240225330731062e23)


def test_negative_overflow_is_also_rejected_rather_than_negative_infinity():
    output = ratio_transformer().fit_transform(frame_with([-1e308], [5e-324]))
    assert np.isnan(output[RATIO_NAME].iloc[0])
    assert output[FLAG_NAME].tolist() == [1]


def test_pandas_na_in_either_source_yields_missing_ratio_and_flag():
    frame = frame_with(
        [pd.NA, 8.0, 4.0], [2.0, None, 2.0], meter_dtype="Float64", area_dtype="Float64"
    )
    output = ratio_transformer().fit_transform(frame)
    np.testing.assert_allclose(
        output[RATIO_NAME].to_numpy(dtype=float), [np.nan, np.nan, 2.0], equal_nan=True
    )
    assert output[FLAG_NAME].tolist() == [1, 1, 0]


def test_nullable_integer_sources_are_divided_without_silent_infill():
    frame = frame_with([7, pd.NA], [2, 2], meter_dtype="Int64", area_dtype="Int64")
    output = ratio_transformer().fit_transform(frame)
    np.testing.assert_allclose(
        output[RATIO_NAME].to_numpy(dtype=float), [3.5, np.nan], equal_nan=True
    )
    assert output[FLAG_NAME].tolist() == [0, 1]


@pytest.mark.parametrize("column", ["meter_reading", "square_feet"])
@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_raw_infinities_fail_validation_rather_than_being_invented(column, bad):
    frame = frame_with([2.0, 4.0], [2.0, 4.0])
    frame[column] = [bad, 4.0]
    with pytest.raises(ValueError, match="nonfinite"):
        ratio_transformer().fit_transform(frame)


# --- Ratio: frame and schema contracts -----------------------------------------


def test_ratio_output_dtypes_are_exactly_float64_and_uint8():
    output = ratio_transformer().fit_transform(frame_with([4.0], [2.0]))
    assert output[RATIO_NAME].dtype == np.dtype("float64")
    assert output[FLAG_NAME].dtype == np.dtype("uint8")


def test_ratio_appends_named_columns_after_unchanged_raw_predictors():
    frame = frame_with([4.0], [2.0])
    output = ratio_transformer().fit_transform(frame)
    assert tuple(output.columns) == (*CORE_COLUMNS, RATIO_NAME, FLAG_NAME)
    pd.testing.assert_frame_equal(output[list(CORE_COLUMNS)], frame)


def test_ratio_does_not_mutate_its_input_frame():
    frame = frame_with([-6.0, np.nan], [3.0, 0.0])
    before = frame.copy(deep=True)
    ratio_transformer().fit(frame).transform(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert RATIO_NAME not in frame.columns and FLAG_NAME not in frame.columns


def test_ratio_preserves_index_identity_and_row_order():
    index = pd.Index([7, 3, 11], name="row_index")
    output = ratio_transformer().fit_transform(frame_with([6.0, 4.0, 2.0], [2.0, 2.0, 2.0], index=index))
    assert output.index.equals(index)
    np.testing.assert_allclose(output[RATIO_NAME], [3.0, 2.0, 1.0])


def test_ratio_rejects_duplicate_index_labels_on_fit_and_transform():
    frame = frame_with([4.0, 4.0], [2.0, 2.0], index=pd.Index([5, 5]))
    with pytest.raises(ValueError, match="index"):
        ratio_transformer().fit(frame)
    fitted = ratio_transformer().fit(frame_with([4.0, 4.0], [2.0, 2.0]))
    with pytest.raises(ValueError, match="index"):
        fitted.transform(frame)


def test_ratio_fit_on_zero_rows_fails_but_transform_returns_typed_schema():
    empty = frame_with([4.0], [2.0]).iloc[:0]
    with pytest.raises(ValueError, match="nonempty"):
        ratio_transformer().fit(empty)
    output = ratio_transformer().fit(frame_with([4.0], [2.0])).transform(empty)
    assert len(output) == 0
    assert tuple(output.columns) == (*CORE_COLUMNS, RATIO_NAME, FLAG_NAME)
    assert output[RATIO_NAME].dtype == np.dtype("float64")
    assert output[FLAG_NAME].dtype == np.dtype("uint8")


@pytest.mark.parametrize("missing", ["meter_reading", "square_feet"])
def test_ratio_requires_both_declared_raw_sources(missing):
    frame = frame_with([4.0], [2.0]).drop(columns=[missing])
    with pytest.raises(ValueError, match=missing):
        ratio_transformer().fit(frame)


def test_ratio_rejects_unapproved_or_precomputed_output_columns():
    frame = frame_with([4.0], [2.0])
    frame[RATIO_NAME] = [2.0]
    with pytest.raises(ValueError, match="approved"):
        ratio_transformer().fit(frame)


def test_ratio_rejects_boolean_sources_masquerading_as_numeric():
    frame = frame_with([4.0], [2.0])
    frame["meter_reading"] = [True]
    with pytest.raises(TypeError, match="nonboolean"):
        ratio_transformer().fit(frame)


def test_ratio_transform_requires_the_ordered_fit_schema():
    fitted = ratio_transformer().fit(frame_with([4.0], [2.0]))
    reordered = frame_with([4.0], [2.0])[list(reversed(CORE_COLUMNS))]
    with pytest.raises(ValueError, match="schema"):
        fitted.transform(reordered)


# --- Ratio: estimator protocol -------------------------------------------------


def test_ratio_transform_before_fit_raises_not_fitted():
    with pytest.raises(NotFittedError):
        ratio_transformer().transform(frame_with([4.0], [2.0]))


def test_ratio_clone_is_unfitted_and_reproduces_results():
    fitted = ratio_transformer().fit(frame_with([4.0], [2.0]))
    fresh = clone(fitted)
    assert fresh.get_params() == {}
    assert fitted_names(fresh) == ()
    np.testing.assert_allclose(
        fresh.fit_transform(frame_with([4.0], [2.0]))[RATIO_NAME],
        fitted.transform(frame_with([4.0], [2.0]))[RATIO_NAME],
    )


def test_ratio_feature_names_out_matches_appended_schema():
    fitted = ratio_transformer().fit(frame_with([4.0], [2.0]))
    assert tuple(fitted.get_feature_names_out()) == (*CORE_COLUMNS, RATIO_NAME, FLAG_NAME)
    with pytest.raises(ValueError, match="input_features"):
        fitted.get_feature_names_out(("meter_reading",))


def test_ratio_fit_ignores_supplied_labels():
    frame = frame_with([6.0, 4.0], [2.0, 2.0])
    labels = pd.Series([0, 1], index=frame.index)
    np.testing.assert_allclose(
        ratio_transformer().fit(frame, labels).transform(frame)[RATIO_NAME],
        ratio_transformer().fit(frame).transform(frame)[RATIO_NAME],
    )


# --- Ratio: registry lineage ---------------------------------------------------


def test_ratio_registry_records_fixed_boundary_and_verified_raw_lineage():
    registry = api().build_value_recipe_registry(RATIO_RECIPE)
    assert registry.names == (*build_lead_baseline_registry().names, RATIO_NAME, FLAG_NAME)
    for name in (RATIO_NAME, FLAG_NAME):
        metadata = registry.get(name)
        assert metadata.transform_fit_boundary is TransformFitBoundary.FIXED
        assert metadata.target_dependence is TargetDependence.INDEPENDENT
        assert metadata.provenance_status is ProvenanceStatus.VERIFIED
        assert metadata.prediction_time_availability is AvailabilityStatus.AVAILABLE
        assert metadata.source_columns == ("meter_reading", "square_feet")
        assert metadata.available_at is None


def test_ratio_registry_is_attached_to_the_fitted_transformer():
    fitted = ratio_transformer().fit(frame_with([4.0], [2.0]))
    assert isinstance(fitted.registry_, FeatureRegistry)
    assert fitted.registry_.get(FLAG_NAME) is not None


def test_unknown_recipe_identifier_is_rejected():
    with pytest.raises(ValueError, match="recipe"):
        api().build_value_recipe_registry("feat_not_a_declared_recipe")


def test_ratio_recipe_rejects_an_irrelevant_sentinel_rule():
    with pytest.raises(ValueError, match="rule"):
        api().build_value_recipe_registry(RATIO_RECIPE, sentinel_rule())


# --- SentinelRule validation ---------------------------------------------------


def test_sentinel_rule_fields_are_frozen():
    rule = sentinel_rule()
    with pytest.raises(FrozenInstanceError):
        rule.replacement_value = 5.0


@pytest.mark.parametrize("values", [(), (-999.0, -999.0), (np.nan,), (np.inf,), (True,), ("-999",)])
def test_sentinel_rule_requires_unique_finite_nonboolean_values(values):
    with pytest.raises(ValueError, match="invalid_values"):
        sentinel_rule(invalid_values=values)


def test_sentinel_rule_rejects_a_mutable_value_collection():
    with pytest.raises(TypeError, match="invalid_values"):
        sentinel_rule(invalid_values=[-999.0])


@pytest.mark.parametrize("replacement", [np.nan, np.inf, True, "x", None])
def test_sentinel_rule_requires_a_finite_nonboolean_replacement(replacement):
    with pytest.raises((ValueError, TypeError), match="replacement_value"):
        sentinel_rule(replacement_value=replacement)


def test_sentinel_rule_rejects_a_replacement_equal_to_a_declared_invalid_value():
    with pytest.raises(ValueError, match="distinct"):
        sentinel_rule(replacement_value=-9999.0)


@pytest.mark.parametrize("column", ["anomaly", "building_id", "primary_use", "not_a_column", ""])
def test_sentinel_rule_requires_a_numeric_authoritative_raw_column(column):
    with pytest.raises(ValueError, match="column"):
        sentinel_rule(column=column)


@pytest.mark.parametrize(
    "reference",
    [
        "/etc/passwd",
        "../outside.json",
        "docs/../../outside.json",
        "docs//evidence.json",
        "docs\\evidence.json",
        "",
        "docs/evidence/",
        "~/evidence.json",
    ],
)
def test_sentinel_rule_requires_a_confined_repo_relative_reference(reference):
    with pytest.raises(ValueError, match="evidence_reference"):
        sentinel_rule(evidence_reference=reference)


@pytest.mark.parametrize("digest", ["0" * 63, "0" * 65, "Z" * 64, "A" * 64, 0, ""])
def test_sentinel_rule_requires_lowercase_hex_sha256(digest):
    with pytest.raises((ValueError, TypeError), match="evidence_sha256"):
        sentinel_rule(evidence_sha256=digest)


# --- Sentinel: evidence verification -------------------------------------------


def test_sentinel_blocks_when_declared_evidence_is_absent(tmp_path):
    rule = sentinel_rule(evidence_sha256=hashlib.sha256(b"{}").hexdigest())
    registry = api().build_value_recipe_registry(SENTINEL_RECIPE, rule)
    transformer = api().EvidencedSentinelTransformer(rule, registry, evidence_root=tmp_path)
    with pytest.raises(BlockedExperimentError) as error:
        transformer.fit(frame_with([-999.0], [2.0]))
    assert error.value.code == "missing_sentinel_evidence"


def test_sentinel_blocks_when_evidence_bytes_are_tampered(tmp_path):
    rule = evidenced_rule(tmp_path)
    transformer = sentinel_transformer(rule, tmp_path)
    transformer.fit(frame_with([-999.0], [2.0]))
    tmp_path.joinpath(*rule.evidence_reference.split("/")).write_bytes(b'{"invalid": [0.0]}')
    with pytest.raises(BlockedExperimentError) as error:
        sentinel_transformer(rule, tmp_path).fit(frame_with([-999.0], [2.0]))
    assert error.value.code == "sentinel_evidence_hash_mismatch"


def test_sentinel_blocks_when_evidence_is_a_directory(tmp_path):
    rule = evidenced_rule(tmp_path)
    path = tmp_path.joinpath(*rule.evidence_reference.split("/"))
    path.unlink()
    path.mkdir()
    with pytest.raises(BlockedExperimentError) as error:
        sentinel_transformer(rule, tmp_path).fit(frame_with([-999.0], [2.0]))
    assert error.value.code == "missing_sentinel_evidence"


def test_sentinel_blocks_an_evidence_symlink_escaping_the_sandbox(tmp_path):
    outside = tmp_path.parent / "outside_evidence.json"
    outside.write_bytes(b"{}")
    root = tmp_path / "sandbox"
    link = root / "docs" / "evidence" / "meter_sentinel.json"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlink creation is unavailable on this platform")
    rule = sentinel_rule(evidence_sha256=hashlib.sha256(b"{}").hexdigest())
    registry = api().build_value_recipe_registry(SENTINEL_RECIPE, rule)
    transformer = api().EvidencedSentinelTransformer(rule, registry, evidence_root=root)
    with pytest.raises(BlockedExperimentError) as error:
        transformer.fit(frame_with([-999.0], [2.0]))
    assert error.value.code == "unconfined_sentinel_evidence"


def test_sentinel_blocks_oversized_evidence(tmp_path):
    rule = evidenced_rule(tmp_path, payload=b"x" * (1024 * 1024 + 1))
    with pytest.raises(BlockedExperimentError) as error:
        sentinel_transformer(rule, tmp_path).fit(frame_with([-999.0], [2.0]))
    assert error.value.code == "oversized_sentinel_evidence"


def test_sentinel_recipe_registry_requires_an_explicit_rule():
    with pytest.raises(BlockedExperimentError) as error:
        api().build_value_recipe_registry(SENTINEL_RECIPE)
    assert error.value.code == "missing_sentinel_rule"


def test_sentinel_transformer_requires_an_explicit_rule(tmp_path):
    registry = build_lead_baseline_registry()
    with pytest.raises(BlockedExperimentError) as error:
        api().EvidencedSentinelTransformer(None, registry, evidence_root=tmp_path).fit(
            frame_with([-999.0], [2.0])
        )
    assert error.value.code == "missing_sentinel_rule"


def test_sentinel_rejects_a_forged_registry_without_the_recipe_record(tmp_path):
    rule = evidenced_rule(tmp_path)
    transformer = sentinel_transformer(rule, tmp_path, registry=build_lead_baseline_registry())
    with pytest.raises(BlockedExperimentError) as error:
        transformer.fit(frame_with([-999.0], [2.0]))
    assert error.value.code == "unverified_sentinel_registry"


def test_sentinel_rejects_a_registry_built_for_a_different_rule(tmp_path):
    rule = evidenced_rule(tmp_path)
    other = api().build_value_recipe_registry(SENTINEL_RECIPE, replace(rule, invalid_values=(-77.0,)))
    with pytest.raises(BlockedExperimentError) as error:
        sentinel_transformer(rule, tmp_path, registry=other).fit(frame_with([-999.0], [2.0]))
    assert error.value.code == "unverified_sentinel_registry"


# --- Sentinel: exact-equality replacement --------------------------------------


def test_sentinel_replaces_only_exact_declared_matches(tmp_path):
    rule = evidenced_rule(tmp_path)
    frame = frame_with([-999.0, 0.0, -1.0, 50.0, np.nan, -9999.0], [2.0] * 6)
    output = sentinel_transformer(rule, tmp_path).fit_transform(frame)
    np.testing.assert_allclose(
        output.meter_reading.to_numpy(dtype=float),
        [-1.0, 0.0, -1.0, 50.0, np.nan, -1.0],
        equal_nan=True,
    )


def test_sentinel_does_not_replace_near_but_unequal_values(tmp_path):
    rule = evidenced_rule(tmp_path)
    frame = frame_with([-999.0000001, -998.9999999, -999.0], [2.0] * 3)
    output = sentinel_transformer(rule, tmp_path).fit_transform(frame)
    np.testing.assert_allclose(
        output.meter_reading.to_numpy(dtype=float), [-999.0000001, -998.9999999, -1.0]
    )


def test_sentinel_leaves_generic_missing_values_missing(tmp_path):
    rule = evidenced_rule(tmp_path)
    frame = frame_with([pd.NA, -999.0], [2.0, 2.0], meter_dtype="Float64")
    output = sentinel_transformer(rule, tmp_path).fit_transform(frame)
    assert output.meter_reading.isna().tolist() == [True, False]
    assert output.meter_reading.dropna().tolist() == [-1.0]


def test_sentinel_deletes_no_rows_and_adds_no_columns(tmp_path):
    rule = evidenced_rule(tmp_path)
    frame = frame_with([-999.0, 5.0, -9999.0], [2.0] * 3, index=pd.Index([4, 9, 1]))
    output = sentinel_transformer(rule, tmp_path).fit_transform(frame)
    assert tuple(output.columns) == CORE_COLUMNS
    assert output.index.equals(frame.index)
    assert not any("invalid" in c or "outlier" in c for c in output.columns)


def test_sentinel_preserves_the_original_column_dtype(tmp_path):
    rule = evidenced_rule(tmp_path, column="year_built", invalid_values=(0,), replacement_value=1900)
    frame = frame_with([5.0, 5.0], [2.0, 2.0])
    frame["year_built"] = pd.Series([0, 1995], index=frame.index, dtype="int64")
    output = sentinel_transformer(rule, tmp_path).fit_transform(frame)
    assert output.year_built.dtype == np.dtype("int64")
    assert output.year_built.tolist() == [1900, 1995]


def test_sentinel_blocks_a_replacement_that_cannot_be_stored_losslessly(tmp_path):
    rule = evidenced_rule(tmp_path, column="year_built", invalid_values=(0,), replacement_value=1900.5)
    frame = frame_with([5.0], [2.0])
    frame["year_built"] = pd.Series([0], index=frame.index, dtype="int64")
    with pytest.raises(BlockedExperimentError) as error:
        sentinel_transformer(rule, tmp_path).fit_transform(frame)
    assert error.value.code == "sentinel_dtype_conflict"


def test_sentinel_leaves_other_raw_columns_untouched(tmp_path):
    rule = evidenced_rule(tmp_path)
    frame = frame_with([-999.0, 5.0], [2.0, 3.0])
    output = sentinel_transformer(rule, tmp_path).fit_transform(frame)
    others = [c for c in CORE_COLUMNS if c != "meter_reading"]
    pd.testing.assert_frame_equal(output[others], frame[others])


def test_sentinel_does_not_mutate_its_input_frame(tmp_path):
    rule = evidenced_rule(tmp_path)
    frame = frame_with([-999.0, 5.0], [2.0, 2.0])
    before = frame.copy(deep=True)
    sentinel_transformer(rule, tmp_path).fit(frame).transform(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_sentinel_records_the_exact_rule_and_fixed_boundary(tmp_path):
    rule = evidenced_rule(tmp_path)
    fitted = sentinel_transformer(rule, tmp_path).fit(frame_with([-999.0], [2.0]))
    metadata = fitted.registry_.get("meter_reading")
    assert metadata.transform_fit_boundary is TransformFitBoundary.FIXED
    assert metadata.provenance_status is ProvenanceStatus.VERIFIED
    assert metadata.source_columns == ("meter_reading",)
    assert rule.evidence_sha256 in metadata.semantic_meaning
    assert "-999.0" in metadata.semantic_meaning and "-1.0" in metadata.semantic_meaning
    assert fitted.registry_.names == build_lead_baseline_registry().names
    assert fitted.evidence_sha256_ == rule.evidence_sha256


# --- Sentinel: frame, schema and estimator protocol ----------------------------


def test_sentinel_requires_the_declared_column_to_be_present(tmp_path):
    rule = evidenced_rule(tmp_path)
    frame = frame_with([-999.0], [2.0]).drop(columns=["meter_reading"])
    with pytest.raises(ValueError, match="meter_reading"):
        sentinel_transformer(rule, tmp_path).fit(frame)


def test_sentinel_rejects_duplicate_index_labels(tmp_path):
    rule = evidenced_rule(tmp_path)
    frame = frame_with([-999.0, -999.0], [2.0, 2.0], index=pd.Index([3, 3]))
    with pytest.raises(ValueError, match="index"):
        sentinel_transformer(rule, tmp_path).fit(frame)


def test_sentinel_rejects_raw_infinities_in_the_declared_column(tmp_path):
    rule = evidenced_rule(tmp_path)
    frame = frame_with([np.inf, -999.0], [2.0, 2.0])
    with pytest.raises(ValueError, match="nonfinite"):
        sentinel_transformer(rule, tmp_path).fit(frame)


def test_sentinel_fit_on_zero_rows_fails_but_transform_returns_typed_schema(tmp_path):
    rule = evidenced_rule(tmp_path)
    empty = frame_with([-999.0], [2.0]).iloc[:0]
    with pytest.raises(ValueError, match="nonempty"):
        sentinel_transformer(rule, tmp_path).fit(empty)
    fitted = sentinel_transformer(rule, tmp_path).fit(frame_with([-999.0], [2.0]))
    output = fitted.transform(empty)
    assert len(output) == 0
    assert tuple(output.columns) == CORE_COLUMNS
    assert output.meter_reading.dtype == np.dtype("float64")


def test_sentinel_transform_requires_the_ordered_fit_schema(tmp_path):
    rule = evidenced_rule(tmp_path)
    fitted = sentinel_transformer(rule, tmp_path).fit(frame_with([-999.0], [2.0]))
    reordered = frame_with([-999.0], [2.0])[list(reversed(CORE_COLUMNS))]
    with pytest.raises(ValueError, match="schema"):
        fitted.transform(reordered)


def test_sentinel_transform_before_fit_raises_not_fitted(tmp_path):
    rule = evidenced_rule(tmp_path)
    with pytest.raises(NotFittedError):
        sentinel_transformer(rule, tmp_path).transform(frame_with([-999.0], [2.0]))


def test_sentinel_clone_starts_unfitted_and_keeps_constructor_arguments(tmp_path):
    rule = evidenced_rule(tmp_path)
    fitted = sentinel_transformer(rule, tmp_path).fit(frame_with([-999.0], [2.0]))
    fresh = clone(fitted)
    assert fresh.get_params()["rule"] == rule
    assert fresh.get_params()["evidence_root"] == tmp_path
    assert fitted_names(fresh) == ()
    with pytest.raises(NotFittedError):
        fresh.transform(frame_with([-999.0], [2.0]))


def test_changing_the_rule_after_fit_requires_refitting(tmp_path):
    rule = evidenced_rule(tmp_path)
    fitted = sentinel_transformer(rule, tmp_path).fit(frame_with([-999.0], [2.0]))
    fitted.rule = replace(rule, replacement_value=-2.0)
    with pytest.raises(ValueError, match="parameter changed"):
        fitted.transform(frame_with([-999.0], [2.0]))


def test_changing_the_evidence_root_after_fit_requires_refitting(tmp_path):
    rule = evidenced_rule(tmp_path)
    fitted = sentinel_transformer(rule, tmp_path).fit(frame_with([-999.0], [2.0]))
    fitted.evidence_root = tmp_path / "other"
    with pytest.raises(ValueError, match="parameter changed"):
        fitted.transform(frame_with([-999.0], [2.0]))


def test_changing_the_registry_after_fit_requires_refitting(tmp_path):
    rule = evidenced_rule(tmp_path)
    fitted = sentinel_transformer(rule, tmp_path).fit(frame_with([-999.0], [2.0]))
    other_rule = replace(rule, replacement_value=-2.0)
    fitted.registry = api().build_value_recipe_registry(SENTINEL_RECIPE, other_rule)
    with pytest.raises(ValueError, match="parameter changed"):
        fitted.transform(frame_with([-999.0], [2.0]))


def test_sentinel_fit_ignores_supplied_labels(tmp_path):
    rule = evidenced_rule(tmp_path)
    frame = frame_with([-999.0, 5.0], [2.0, 2.0])
    labels = pd.Series([0, 1], index=frame.index)
    pd.testing.assert_frame_equal(
        sentinel_transformer(rule, tmp_path).fit(frame, labels).transform(frame),
        sentinel_transformer(rule, tmp_path).fit(frame).transform(frame),
    )


def test_sentinel_feature_names_out_preserves_the_raw_schema(tmp_path):
    rule = evidenced_rule(tmp_path)
    fitted = sentinel_transformer(rule, tmp_path).fit(frame_with([-999.0], [2.0]))
    assert tuple(fitted.get_feature_names_out()) == CORE_COLUMNS


def test_sentinel_rejects_a_non_registry_authoritative_argument(tmp_path):
    rule = evidenced_rule(tmp_path)
    with pytest.raises(TypeError, match="FeatureRegistry"):
        api().EvidencedSentinelTransformer(rule, {"meter_reading": 1}, evidence_root=tmp_path).fit(
            frame_with([-999.0], [2.0])
        )


def test_sentinel_rejects_a_registry_whose_raw_producer_was_rewritten(tmp_path):
    """A caller-built lookalike registry cannot authorize a new raw producer."""
    rule = evidenced_rule(tmp_path)
    honest = api().build_value_recipe_registry(SENTINEL_RECIPE, rule)
    forged = FeatureRegistry([
        FeatureMetadata(
            name=name,
            family="measurement",
            source_columns=("meter_reading", "square_feet") if name == "meter_reading" else (name,),
            semantic_meaning=honest.get(name).semantic_meaning,
            prediction_time_availability=AvailabilityStatus.AVAILABLE,
            target_dependence=TargetDependence.INDEPENDENT,
            transform_fit_boundary=TransformFitBoundary.FIXED,
            fallback_behavior=honest.get(name).fallback_behavior,
            resource_cost=ResourceCost.LOW,
            provenance_status=ProvenanceStatus.VERIFIED,
        )
        for name in honest.names
    ])
    with pytest.raises(BlockedExperimentError) as error:
        sentinel_transformer(rule, tmp_path, registry=forged).fit(frame_with([-999.0], [2.0]))
    assert error.value.code == "unverified_sentinel_registry"


@pytest.mark.parametrize("dtype", ["int64", "Int64", "uint64"])
def test_exact_large_integer_identity_is_preserved(tmp_path, dtype):
    rule = evidenced_rule(tmp_path, column="year_built", invalid_values=(2**53,), replacement_value=0)
    frame = pd.DataFrame({"year_built": pd.Series([2**53, 2**53 + 1], dtype=dtype)})
    output = sentinel_transformer(rule, tmp_path).fit_transform(frame)
    assert output.year_built.tolist() == [0, 2**53 + 1]
    assert output.year_built.dtype == frame.year_built.dtype


def test_exact_large_integer_rule_distinctness(tmp_path):
    rule = evidenced_rule(tmp_path, column="year_built", invalid_values=(2**53, 2**53 + 1),
                          replacement_value=2**53 + 2)
    assert rule.invalid_values == (2**53, 2**53 + 1)
    rule = replace(rule, invalid_values=(2**53,), replacement_value=2**53 + 1)
    frame = pd.DataFrame({"year_built": [2**53]})
    assert sentinel_transformer(rule, tmp_path).fit_transform(frame).year_built.tolist() == [2**53 + 1]


def test_exact_large_integer_replacement_rejects_float_rounding(tmp_path):
    rule = evidenced_rule(tmp_path, invalid_values=(-999,), replacement_value=2**53 + 1)
    with pytest.raises(BlockedExperimentError) as error:
        sentinel_transformer(rule, tmp_path).fit(frame_with([-999]))
    assert error.value.code == "sentinel_dtype_conflict"


@pytest.mark.parametrize("meter,area", [(2030, None), (None, 2031), (2030, 2031)])
def test_ratio_propagates_latest_source_availability(monkeypatch, meter, area):
    baseline = build_lead_baseline_registry()
    dates = {"meter_reading": meter, "square_feet": area}
    restricted = FeatureRegistry([
        replace(baseline.get(name), available_at=datetime(dates[name], 1, 1, tzinfo=UTC))
        if dates.get(name) else baseline.get(name) for name in baseline.names
    ])
    monkeypatch.setattr(api(), "build_lead_baseline_registry", lambda: restricted)
    registry = api().build_value_recipe_registry(RATIO_RECIPE)
    expected = datetime(max(value for value in (meter, area) if value), 1, 1, tzinfo=UTC)
    for name in (RATIO_NAME, FLAG_NAME):
        assert registry.get(name).available_at == expected


@pytest.mark.parametrize("mutation", ["escape", "disappear", "growth"])
def test_evidence_race_is_blocked_at_actual_read(tmp_path, monkeypatch, mutation):
    root = tmp_path / "root"
    rule = evidenced_rule(root, payload=b"{}")
    path = root / rule.evidence_reference
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    original_io_open, original_os_open, original_read = io.open, os.open, os.read
    triggered = False

    def mutate():
        nonlocal triggered
        if triggered:
            return
        triggered = True
        if mutation == "growth":
            path.write_bytes(b"x" * (api().MAX_EVIDENCE_BYTES + 1))
        else:
            path.unlink()
            path.symlink_to(outside if mutation == "escape" else tmp_path / "absent")

    def opening(file, *args, **kwargs):
        if Path(file) == path and args and args[0] == "rb":
            mutate()
        return original_io_open(file, *args, **kwargs)

    def descriptor_open(file, *args, **kwargs):
        if str(file) == path.name and mutation != "growth":
            mutate()
        return original_os_open(file, *args, **kwargs)

    def descriptor_read(fd, size):
        if mutation == "growth":
            mutate()
        return original_read(fd, size)

    monkeypatch.setattr(io, "open", opening)
    monkeypatch.setattr(os, "open", descriptor_open)
    monkeypatch.setattr(os, "read", descriptor_read)
    with pytest.raises(BlockedExperimentError) as error:
        sentinel_transformer(rule, root).fit(frame_with([-999]))
    assert triggered
    assert error.value.code == ("oversized_sentinel_evidence" if mutation == "growth"
                                else "unconfined_sentinel_evidence")


def test_evidence_disappearance_before_descriptor_open_is_a_blocker(tmp_path, monkeypatch):
    rule = evidenced_rule(tmp_path)
    path = tmp_path / rule.evidence_reference
    original_open = os.open

    def opening(file, *args, **kwargs):
        if str(file) == path.name:
            path.unlink()
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(os, "open", opening)
    with pytest.raises(BlockedExperimentError) as error:
        sentinel_transformer(rule, tmp_path).fit(frame_with([-999]))
    assert error.value.code == "missing_sentinel_evidence"


def test_evidence_parent_swap_to_symlink_is_blocked(tmp_path, monkeypatch):
    root = tmp_path / "root"
    rule = evidenced_rule(root)
    parent = root / "docs" / "evidence"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_open = os.open

    def opening(file, *args, **kwargs):
        if str(file) == "evidence":
            parent.rename(root / "docs" / "original")
            parent.symlink_to(outside, target_is_directory=True)
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(os, "open", opening)
    with pytest.raises(BlockedExperimentError) as error:
        sentinel_transformer(rule, root).fit(frame_with([-999]))
    assert error.value.code == "unconfined_sentinel_evidence"


def test_evidence_hash_uses_pinned_descriptor_after_path_swap(tmp_path, monkeypatch):
    rule = evidenced_rule(tmp_path)
    path = tmp_path / rule.evidence_reference
    original_read = os.read
    swapped = False

    def reading(fd, size):
        nonlocal swapped
        if not swapped:
            swapped = True
            path.rename(path.with_suffix(".original"))
            path.write_bytes(b"different evidence")
        return original_read(fd, size)

    monkeypatch.setattr(os, "read", reading)
    fitted = sentinel_transformer(rule, tmp_path).fit(frame_with([-999]))
    assert swapped
    assert fitted.evidence_sha256_ == rule.evidence_sha256
