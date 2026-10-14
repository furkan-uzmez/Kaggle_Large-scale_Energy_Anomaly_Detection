import numpy as np
import pandas as pd
import pytest

from project.features.candidates import (
    MISSINGNESS,
    SAFE_NUMERIC,
    CandidateFeatureTransformer,
)
from project.features.encodings import (
    BITMASK,
    BITMASK_COLUMNS,
    EncodingFeatureTransformer,
)
from project.features.input_encoding import construction_years, year_built_values

ENCODED = "lead_year_offset_1900_uint8_v1"


def frame(values=(0, 100, 255, None)):
    result = pd.DataFrame({name: [1.] * len(values) for name in BITMASK_COLUMNS})
    result["primary_use"] = "Office"
    result["year"] = 2020
    result["year_built"] = pd.array(values, dtype="Float64")
    result["floor_count"] = 0
    result.index = pd.Index(range(10, 10 + len(values)), name="original")
    return result


@pytest.mark.parametrize("recipe", [MISSINGNESS, SAFE_NUMERIC])
def test_encoded_raw_preserved_and_semantics(recipe):
    raw = frame()
    transformer = CandidateFeatureTransformer(recipe)
    transformer.set_params(year_built_encoding=ENCODED)
    output = transformer.fit_transform(raw)
    pd.testing.assert_frame_equal(output[list(raw.columns)], raw)
    if recipe == MISSINGNESS:
        assert output.year_built_missing.tolist() == [0, 0, 1, 1]
        assert output.floor_count_missing.tolist() == [0, 0, 0, 0]
    else:
        np.testing.assert_allclose(output.building_age, [120, 20, np.nan, np.nan])


def test_bitmask_uses_same_missingness_contract():
    raw = frame()
    transformer = EncodingFeatureTransformer(BITMASK)
    transformer.set_params(year_built_encoding=ENCODED)
    output = transformer.fit_transform(raw)
    pd.testing.assert_frame_equal(output[list(raw.columns)], raw)
    assert output.missingness_bitmask.tolist() == [0, 0, 4, 4]


@pytest.mark.parametrize("bad", [-1, 256, 1.5, np.inf, 2000])
def test_explicit_encoded_domain_rejects_invalid(bad):
    transformer = CandidateFeatureTransformer(MISSINGNESS)
    transformer.set_params(year_built_encoding=ENCODED)
    with pytest.raises(ValueError):
        transformer.fit(frame([bad]))


def test_unsigned_age_does_not_wrap_and_future_rejected_only_for_age():
    raw = frame([0, 100, 255])
    raw["year_built"] = raw.year_built.astype("uint8")
    transformer = CandidateFeatureTransformer(SAFE_NUMERIC)
    transformer.set_params(year_built_encoding=ENCODED)
    np.testing.assert_allclose(transformer.fit_transform(raw).building_age, [120, 20, np.nan])
    future = frame([121])
    CandidateFeatureTransformer(MISSINGNESS, year_built_encoding=ENCODED).fit(future)
    with pytest.raises(ValueError, match="observation year"):
        transformer.fit(future)


def test_default_absolute_and_parameter_mutation():
    with pytest.raises(ValueError):
        CandidateFeatureTransformer(MISSINGNESS).fit(frame())
    transformer = CandidateFeatureTransformer(MISSINGNESS, year_built_encoding=ENCODED).fit(frame())
    assert transformer.get_params()["year_built_encoding"] == ENCODED
    transformer.set_params(year_built_encoding="absolute_calendar_year")
    with pytest.raises(ValueError, match="parameter changed"):
        transformer.transform(frame())


@pytest.mark.parametrize("recipe", [BITMASK, "feat_frequency_encoding"])
def test_encoding_recipes_validate_explicit_domain_and_parameter_mutation(recipe):
    transformer = EncodingFeatureTransformer(recipe, year_built_encoding=ENCODED)
    with pytest.raises(ValueError):
        transformer.fit(frame([2000]))
    transformer.fit(frame())
    transformer.set_params(year_built_encoding="absolute_calendar_year")
    with pytest.raises(ValueError, match="parameter changed"):
        transformer.transform(frame())


def test_direct_input_contract_no_inference_and_non_numeric_rejected():
    column = pd.Series([0, 254, 255, None], name="year_built", dtype="Float64")
    np.testing.assert_allclose(construction_years(column, ENCODED), [1900, 2154, np.nan, np.nan])
    with pytest.raises(ValueError):
        year_built_values(column, "absolute_calendar_year")
    with pytest.raises(ValueError, match="unsupported"):
        year_built_values(column, "guess")
    with pytest.raises(TypeError):
        year_built_values(pd.Series([True], name="year_built"), ENCODED)
