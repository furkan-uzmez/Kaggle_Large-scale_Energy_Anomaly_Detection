"""Unit tests for standalone GBDT challenger models and runner."""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from scripts.train_gbdt_standalone import (
    DEFAULT_CONFIG_PATH,
    load_config,
    parse_args,
    run_standalone_gbdt,
)

from project.models.candidate_artifacts import CANDIDATE_OOF_SCHEMA
from project.models.gbdt_standalone import (
    SUPPORTED_FAMILIES,
    MissingGbdtDependencyError,
    get_installation_guidance,
    is_gbdt_available,
    normalize_family,
    train_standalone_gbdt_partition,
)


@pytest.fixture
def synthetic_train_and_val_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create synthetic training and validation DataFrames with missing values and categoricals."""
    # Training set with 50 rows
    rng = np.random.default_rng(42)
    n_train = 50
    train_dict = {
        "building_id": [1] * 25 + [2] * 25,
        "timestamp": [
            datetime.datetime(2016, 1, 1, tzinfo=datetime.UTC)
            + datetime.timedelta(hours=i)
            for i in range(n_train)
        ],
        "meter_reading": rng.normal(100.0, 20.0, size=n_train),
        "square_feet": [50000] * 25 + [75000] * 25,
        "year_built": [1975 if i % 5 != 0 else None for i in range(n_train)],  # missing values
        "floor_count": [4 if i % 3 != 0 else None for i in range(n_train)],  # missing values
        "hour": [i % 24 for i in range(n_train)],
        "weekday": [(i // 24) % 7 for i in range(n_train)],
        "month": [1] * n_train,
        "year": [2016] * n_train,
        "primary_use": ["Education" if i % 2 == 0 else "Office" for i in range(n_train)],
        "anomaly": [1 if i % 7 == 0 else 0 for i in range(n_train)],
    }
    train_df = pl.DataFrame(train_dict)

    # Validation set with 20 rows
    n_val = 20
    val_dict = {
        "building_id": [3] * n_val,
        "timestamp": [
            datetime.datetime(2016, 1, 1, tzinfo=datetime.UTC)
            + datetime.timedelta(hours=i)
            for i in range(n_val)
        ],
        "meter_reading": rng.normal(110.0, 25.0, size=n_val),
        "square_feet": [60000] * n_val,
        "year_built": [1980 if i % 4 != 0 else None for i in range(n_val)],
        "floor_count": [5 if i % 2 != 0 else None for i in range(n_val)],
        "hour": [i % 24 for i in range(n_val)],
        "weekday": [(i // 24) % 7 for i in range(n_val)],
        "month": [1] * n_val,
        "year": [2016] * n_val,
        "primary_use": ["Education" if i % 3 == 0 else "Office" for i in range(n_val)],
        "anomaly": [1 if i % 5 == 0 else 0 for i in range(n_val)],
    }
    val_df = pl.DataFrame(val_dict)
    return train_df, val_df


def test_normalize_family():
    """Verify family name normalization, aliases, and error rejection."""
    assert normalize_family("lightgbm") == "lightgbm"
    assert normalize_family("LightGBM") == "lightgbm"
    assert normalize_family("lgbm") == "lightgbm"
    assert normalize_family("catboost") == "catboost"
    assert normalize_family("cb") == "catboost"
    assert normalize_family("xgboost") == "xgboost"
    assert normalize_family("xgb") == "xgboost"

    with pytest.raises(ValueError, match="Unsupported GBDT family"):
        normalize_family("random_forest")

    with pytest.raises(TypeError, match="family must be a string"):
        normalize_family(123)  # type: ignore[arg-type]


def test_missing_dependency_and_guidance():
    """Verify availability inspection and clear installation guidance."""
    for fam in SUPPORTED_FAMILIES:
        guidance = get_installation_guidance(fam)
        assert fam in guidance.lower()
        assert "install" in guidance.lower()
        assert "pip install" in guidance or "uv pip install" in guidance

    # In active environment where packages are not installed:
    # is_gbdt_available should return False and dry_run=False should raise MissingGbdtDependencyError
    if not is_gbdt_available("lightgbm"):
        df = pl.DataFrame({"x": [1, 2], "y": [0, 1]})
        with pytest.raises(MissingGbdtDependencyError, match="LightGBM is not installed"):
            train_standalone_gbdt_partition(
                family="lightgbm",
                train_df=df,
                val_df=df,
                feature_cols=["x"],
                target_col="y",
                params={},
                dry_run=False,
            )


def test_dry_run_partition_training(synthetic_train_and_val_frames):
    """Verify mock/dry-run partition training across all families and input types."""
    train_df, val_df = synthetic_train_and_val_frames
    feature_cols = [
        "meter_reading",
        "square_feet",
        "year_built",
        "floor_count",
        "hour",
        "weekday",
        "month",
        "year",
        "primary_use",
    ]
    target_col = "anomaly"

    for fam in SUPPORTED_FAMILIES:
        preds = train_standalone_gbdt_partition(
            family=fam,
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            target_col=target_col,
            params={"seed": 42},
            dry_run=True,
        )
        assert isinstance(preds, np.ndarray)
        assert preds.dtype == np.float64
        assert preds.shape == (val_df.height,)
        assert np.all(np.isfinite(preds))
        assert np.all((preds >= 0.0) & (preds <= 1.0))

        # Check deterministic reproducibility
        preds_repeat = train_standalone_gbdt_partition(
            family=fam,
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            target_col=target_col,
            params={"seed": 42},
            dry_run=True,
        )
        np.testing.assert_array_equal(preds, preds_repeat)


def test_dry_run_with_pandas_frames(synthetic_train_and_val_frames):
    """Verify partition training operates seamlessly on Pandas DataFrames."""
    train_pl, val_pl = synthetic_train_and_val_frames
    train_pd = train_pl.to_pandas()
    val_pd = val_pl.to_pandas()
    feature_cols = ["meter_reading", "primary_use"]

    preds = train_standalone_gbdt_partition(
        family="xgboost",
        train_df=train_pd,
        val_df=val_pd,
        feature_cols=feature_cols,
        target_col="anomaly",
        params={"seed": 100},
        dry_run=True,
    )
    assert len(preds) == len(val_pd)
    assert isinstance(preds, np.ndarray)


def test_config_loading():
    """Verify loading and structure of gbdt_standalone_challengers.yaml."""
    assert DEFAULT_CONFIG_PATH.is_file(), f"Missing config file at {DEFAULT_CONFIG_PATH}"
    config = load_config(DEFAULT_CONFIG_PATH)

    assert config["target"] == "anomaly"
    assert config["positive_label"] == 1
    assert set(config["split_seeds"]) == {42, 43, 44}
    assert "models" in config

    for fam in ("lightgbm", "catboost", "xgboost"):
        assert fam in config["models"], f"Missing {fam} in models config"
        spec = config["models"][fam]
        assert spec["model_family"] == fam
        assert "params" in spec
        params = spec["params"]
        assert params.get("learning_rate") == 0.03
        assert "early_stopping_rounds" not in params
        assert "early_stopping" not in params
    assert config.get("arms") == ["control", "challenger"]
    for fam in ("lightgbm", "catboost", "xgboost"):
        assert config["models"][fam].get("arms") == ["control", "challenger"]

def test_cli_argument_parsing():
    """Verify CLI argument parsing defaults and overrides."""
    args = parse_args(["--family", "lightgbm", "--smoke", "--dry-run"])
    assert args.family == "lightgbm"
    assert args.smoke is True
    assert args.dry_run is True
    assert args.arm == "both"
    assert args.config == DEFAULT_CONFIG_PATH
    assert args.output_dir is None

    # Test explicit arm choices
    args_ctrl = parse_args(["--family", "lightgbm", "--arm", "control"])
    assert args_ctrl.arm == "control"
    args_chall = parse_args(["--family", "catboost", "--arm", "challenger"])
    assert args_chall.arm == "challenger"
    args_both = parse_args(["--family", "xgboost", "--arm", "both"])
    assert args_both.arm == "both"

    # Test invalid family
    with pytest.raises(SystemExit):
        parse_args(["--family", "unsupported_family"])
    # Test invalid arm
    with pytest.raises(SystemExit):
        parse_args(["--family", "lightgbm", "--arm", "invalid_arm"])


def test_candidate_oof_schema_conformity(tmp_path: Path, synthetic_train_and_val_frames):
    """Verify end-to-end OOF generation produces valid Parquet conforming to CANDIDATE_OOF_SCHEMA."""
    train_df, val_df = synthetic_train_and_val_frames

    # Prepare minimal synthetic split manifest covering the buildings
    # dev buildings: 1, 2, 3
    splits_records = []
    for rep in [0]:
        for b_id, fld in [(1, 0), (2, 1), (3, 1)]:
            splits_records.append(
                {
                    "building_id": b_id,
                    "repeat": rep,
                    "seed": 42 + rep,
                    "split": "dev",
                    "fold": fld,
                    "row_count": 25,
                    "anomaly_count": 5,
                    "normal_count": 20,
                    "anomaly_prevalence": 0.2,
                    "support_stratum": 0,
                }
            )
    split_path = tmp_path / "splits.parquet"
    pl.DataFrame(splits_records).write_parquet(split_path)

    # Combine train and val into single joined data file
    joined_df = pl.concat([train_df, val_df], how="vertical").with_row_index("row_index")
    data_path = tmp_path / "train_joined.parquet"
    joined_df.write_parquet(data_path)

    # Create synthetic config pointing to temporary data
    config_data = {
        "target": "anomaly",
        "positive_label": 1,
        "split_manifest": str(split_path),
        "data_path": str(data_path),
        "features": {
            "numeric": ["meter_reading", "square_feet"],
            "categorical": ["primary_use"],
        },
        "models": {
            "lightgbm": {
                "model_family": "lightgbm",
                "recipe_id": "recipe_gbdt_lightgbm",
                "model_config_id": "cfg_gbdt_lightgbm_default",
                "params": {"learning_rate": 0.03, "seed": 42},
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.dump(config_data))

    out_dir = tmp_path / "outputs"
    oof_parquet = run_standalone_gbdt(
        family="lightgbm",
        config=config_data,
        config_path=config_path,
        output_dir=out_dir,
        smoke=True,
        dry_run=True,
    )

    assert oof_parquet.is_file()
    persisted = pl.read_parquet(oof_parquet)

    # Verify column names and order
    expected_names = list(CANDIDATE_OOF_SCHEMA.names)
    assert list(persisted.columns) == expected_names

    # Verify schema and metadata via PyArrow Parquet table
    arrow_persisted = pq.read_table(oof_parquet)
    schema = arrow_persisted.schema
    assert schema.names == CANDIDATE_OOF_SCHEMA.names
    assert schema.metadata is not None
    assert schema.metadata.get(b"candidate_schema_version") == b"1"
    for col in expected_names:
        assert persisted[col].null_count() == 0

    # Verify predictions are within [0, 1]
    assert persisted["prediction"].min() >= 0.0
    assert persisted["prediction"].max() <= 1.0

    # Verify provenance
    assert persisted["non_comparative"].all()
    assert not persisted["n03_eligible"].any()
    assert set(persisted["execution_mode"]) == {"synthetic_smoke"}


def test_gbdt_leak_free_outer_validation(monkeypatch: pytest.MonkeyPatch, synthetic_train_and_val_frames):
    """Verify outer validation labels are strictly predict/score-only with zero leakage into fit."""
    train_df, val_df = synthetic_train_and_val_frames
    feature_cols = ["meter_reading", "square_feet", "primary_use"]
    target_col = "anomaly"

    # Drop target column entirely from val_df to prove outer fold labels are never accessed
    val_no_target = val_df.drop(target_col)
    assert target_col not in val_no_target.columns

    monkeypatch.setattr("project.models.gbdt_standalone.is_gbdt_available", lambda fam: True)

    # 1. LightGBM: verify fit kwargs do not contain eval_set or callbacks
    # 1. LightGBM: verify early_stopping_rounds is ignored in both init and fit
    lgb_fit_calls = []
    lgb_init_kwargs = []
    class FakeLGBMClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            lgb_init_kwargs.append(kwargs)
        def fit(self, X, y, **kwargs):
            lgb_fit_calls.append({"X": X, "y": y, "kwargs": kwargs})
            return self
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.42)])

    fake_lgb = type("FakeLGB", (), {
        "LGBMClassifier": FakeLGBMClassifier,
        "early_stopping": lambda *args, **kwargs: None,
    })
    monkeypatch.setitem(sys.modules, "lightgbm", fake_lgb)

    preds_lgb = train_standalone_gbdt_partition(
        family="lightgbm",
        train_df=train_df,
        val_df=val_no_target,
        feature_cols=feature_cols,
        target_col=target_col,
        params={"n_estimators": 50, "early_stopping_rounds": 50},
        dry_run=False,
    )
    assert len(preds_lgb) == val_no_target.height
    assert len(lgb_fit_calls) == 1
    call_lgb = lgb_fit_calls[0]
    assert "eval_set" not in call_lgb["kwargs"]
    assert "callbacks" not in call_lgb["kwargs"]
    assert "early_stopping_rounds" not in call_lgb["kwargs"]
    assert "early_stopping_rounds" not in lgb_init_kwargs[0]

    # 2. CatBoost: verify early_stopping_rounds is ignored in both init and fit
    cb_fit_calls = []
    cb_init_kwargs = []
    class FakeCatBoostClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            cb_init_kwargs.append(kwargs)
        def fit(self, X, y, **kwargs):
            cb_fit_calls.append({"X": X, "y": y, "kwargs": kwargs})
            return self
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.33)])

    fake_cb = type("FakeCB", (), {"CatBoostClassifier": FakeCatBoostClassifier})
    monkeypatch.setitem(sys.modules, "catboost", fake_cb)

    preds_cb = train_standalone_gbdt_partition(
        family="catboost",
        train_df=train_df,
        val_df=val_no_target,
        feature_cols=feature_cols,
        target_col=target_col,
        params={"iterations": 50, "early_stopping_rounds": 50},
        dry_run=False,
    )
    assert len(preds_cb) == val_no_target.height
    assert len(cb_fit_calls) == 1
    call_cb = cb_fit_calls[0]
    assert "eval_set" not in call_cb["kwargs"]
    assert "early_stopping_rounds" not in call_cb["kwargs"]
    assert "early_stopping_rounds" not in cb_init_kwargs[0]

    # 3. XGBoost: verify early_stopping_rounds is ignored in both init and fit
    xgb_fit_calls = []
    xgb_init_kwargs = []
    class FakeXGBClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            xgb_init_kwargs.append(kwargs)
        def fit(self, X, y, **kwargs):
            xgb_fit_calls.append({"X": X, "y": y, "kwargs": kwargs})
            return self
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.full(len(X), 0.55)])

    fake_xgb = type("FakeXGB", (), {"XGBClassifier": FakeXGBClassifier})
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgb)

    preds_xgb = train_standalone_gbdt_partition(
        family="xgboost",
        train_df=train_df,
        val_df=val_no_target,
        feature_cols=feature_cols,
        target_col=target_col,
        params={"n_estimators": 50, "early_stopping_rounds": 50},
        dry_run=False,
    )
    assert len(preds_xgb) == val_no_target.height
    assert len(xgb_fit_calls) == 1
    call_xgb = xgb_fit_calls[0]
    assert "eval_set" not in call_xgb["kwargs"]
    assert "early_stopping_rounds" not in call_xgb["kwargs"]
    assert "early_stopping_rounds" not in xgb_init_kwargs[0]

def test_categorical_sentinel_with_missing_values():
    """Verify handling of pandas Categorical with missing values without TypeError."""
    import pandas as pd

    from project.models.gbdt_standalone import (
        _prepare_matrices_for_catboost,
        _prepare_matrices_for_lightgbm,
        _prepare_matrices_for_xgboost,
    )

    # Create training and validation frames with pd.Categorical having missing values
    X_tr = pd.DataFrame({
        "cat_col": pd.Categorical(["Education", None, "Office", "Retail"]),
        "num_col": [1.0, 2.0, 3.0, 4.0],
    })
    X_va = pd.DataFrame({
        "cat_col": pd.Categorical(["Office", "Retail", None]),
        "num_col": [5.0, 6.0, 7.0],
    })

    # Test CatBoost matrix preparation does not raise "Cannot setitem on a Categorical"
    cb_tr, cb_va, cb_cats = _prepare_matrices_for_catboost(X_tr, X_va)
    assert "cat_col" in cb_cats
    assert "_missing_" in cb_tr["cat_col"].values
    assert "_missing_" in cb_va["cat_col"].values

    # Test LightGBM matrix preparation handles categoricals with nulls
    lgb_tr, _lgb_va, lgb_cats = _prepare_matrices_for_lightgbm(X_tr, X_va)
    assert "cat_col" in lgb_cats
    assert isinstance(lgb_tr["cat_col"].dtype, pd.CategoricalDtype)

    # Test XGBoost matrix preparation handles categoricals with nulls
    xgb_tr, _xgb_va = _prepare_matrices_for_xgboost(X_tr, X_va)
    assert isinstance(xgb_tr["cat_col"].dtype, pd.CategoricalDtype)


def test_string_extension_dtypes_recognized():
    """Verify that pandas string extension dtypes are uniformly treated as categoricals."""
    import pandas as pd

    from project.models.gbdt_standalone import (
        _is_categorical_like,
        _prepare_matrices_for_catboost,
        _prepare_matrices_for_lightgbm,
        _prepare_matrices_for_xgboost,
    )

    # Verify _is_categorical_like accepts string extension dtypes
    assert _is_categorical_like(pd.Series(["a", "b"], dtype="string").dtype)
    assert _is_categorical_like(pd.Series(["a", "b"], dtype=object).dtype)
    assert _is_categorical_like(pd.Series(pd.Categorical(["a", "b"])).dtype)
    assert not _is_categorical_like(pd.Series([1, 2]).dtype)
    assert not _is_categorical_like(pd.Series([1.5, 2.5]).dtype)

    # DataFrame with string extension dtype
    X_tr = pd.DataFrame({"text_feat": pd.Series(["apple", "banana"], dtype="string")})
    X_va = pd.DataFrame({"text_feat": pd.Series(["banana", "apple"], dtype="string")})

    _, _, cb_cats = _prepare_matrices_for_catboost(X_tr, X_va)
    assert "text_feat" in cb_cats

    _, _, lgb_cats = _prepare_matrices_for_lightgbm(X_tr, X_va)
    assert "text_feat" in lgb_cats

    xgb_tr, _xgb_va = _prepare_matrices_for_xgboost(X_tr, X_va)
    assert isinstance(xgb_tr["text_feat"].dtype, pd.CategoricalDtype)


def test_parameter_driven_mock_provenance(tmp_path: Path, synthetic_train_and_val_frames):
    """Verify mock parameter strictly enforces non_comparative=True and n03_eligible=False."""
    train_df, val_df = synthetic_train_and_val_frames
    split_path = tmp_path / "splits.parquet"
    pl.DataFrame([
        {"building_id": 1, "repeat": 0, "seed": 42, "split": "dev", "fold": 0, "row_count": 25,
         "anomaly_count": 5, "normal_count": 20, "anomaly_prevalence": 0.2, "support_stratum": 0},
        {"building_id": 2, "repeat": 0, "seed": 42, "split": "dev", "fold": 1, "row_count": 25,
         "anomaly_count": 5, "normal_count": 20, "anomaly_prevalence": 0.2, "support_stratum": 0},
    ]).write_parquet(split_path)

    data_path = tmp_path / "train_joined.parquet"
    pl.concat([train_df, val_df], how="vertical").write_parquet(data_path)

    # Config with mock: True in params
    config_data = {
        "target": "anomaly",
        "positive_label": 1,
        "split_manifest": str(split_path),
        "data_path": str(data_path),
        "features": {"numeric": ["meter_reading"], "categorical": []},
        "models": {
            "lightgbm": {
                "model_family": "lightgbm",
                "recipe_id": "mock_recipe",
                "model_config_id": "cfg_mock",
                "params": {"mock": True, "seed": 42},
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))

    # Call with smoke=False, dry_run=False -> mock is driven purely by params
    oof_parquet = run_standalone_gbdt(
        family="lightgbm",
        config=config_data,
        config_path=config_path,
        output_dir=tmp_path / "out",
        smoke=False,
        dry_run=False,
    )
    persisted = pl.read_parquet(oof_parquet)
    assert persisted["non_comparative"].all()
    assert not persisted["n03_eligible"].any()
    assert set(persisted["execution_mode"]) == {"synthetic_smoke"}

    # Verify manifest JSON also contains non_comparative=True and n03_eligible=False
    manifest_path = oof_parquet.parent / "lightgbm_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["non_comparative"] is True
    assert manifest["n03_eligible"] is False
    assert manifest["execution_mode"] == "synthetic_smoke"


def test_timestamp_nanosecond_preservation(tmp_path: Path):
    """Verify physical nanosecond timestamp precision is preserved without microsecond truncation."""
    ticks = [1451649600123456789, 1451653200987654321]
    ts_ns = pl.Series("timestamp", pa.array(ticks, type=pa.timestamp("ns")))
    train_df = pl.DataFrame({
        "building_id": [1, 1],
        "timestamp": ts_ns,
        "meter_reading": [10.0, 20.0],
        "anomaly": [0, 1],
    })
    val_df = pl.DataFrame({
        "building_id": [2, 2],
        "timestamp": ts_ns,
        "meter_reading": [15.0, 25.0],
        "anomaly": [1, 0],
    })

    split_path = tmp_path / "splits.parquet"
    pl.DataFrame([
        {"building_id": 1, "repeat": 0, "seed": 42, "split": "dev", "fold": 0, "row_count": 2,
         "anomaly_count": 1, "normal_count": 1, "anomaly_prevalence": 0.5, "support_stratum": 0},
        {"building_id": 2, "repeat": 0, "seed": 42, "split": "dev", "fold": 1, "row_count": 2,
         "anomaly_count": 1, "normal_count": 1, "anomaly_prevalence": 0.5, "support_stratum": 0},
    ]).write_parquet(split_path)

    data_path = tmp_path / "train_joined.parquet"
    pl.concat([train_df, val_df], how="vertical").write_parquet(data_path)

    config_data = {
        "target": "anomaly",
        "positive_label": 1,
        "split_manifest": str(split_path),
        "data_path": str(data_path),
        "features": {"numeric": ["meter_reading"], "categorical": []},
        "models": {
            "xgboost": {
                "model_family": "xgboost",
                "recipe_id": "ts_recipe",
                "model_config_id": "cfg_ts",
                "params": {"seed": 42},
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))

    oof_parquet = run_standalone_gbdt(
        family="xgboost",
        config=config_data,
        config_path=config_path,
        output_dir=tmp_path / "ts_out",
        smoke=True,
        dry_run=True,
    )
    persisted = pl.read_parquet(oof_parquet)
    assert persisted.schema["timestamp"] == pl.Datetime("ns")
    arrow_tab = pq.read_table(oof_parquet)
    assert arrow_tab.schema.field("timestamp").type == pa.timestamp("ns")
    # Verify integer nanosecond ticks match physical input without microsecond truncation
    # Verify integer nanosecond ticks match physical input across folds without microsecond truncation
    # 2 folds * 2 arms = 4 occurrences of each tick
    expected_ticks = ticks * 4
    assert persisted["timestamp"].cast(pl.Int64).to_list() == expected_ticks
    assert arrow_tab.column("timestamp").cast(pa.int64()).to_pylist() == expected_ticks


@pytest.mark.parametrize(
    ("arm_mode", "expected_arms"),
    [
        ("control", ["control"]),
        ("challenger", ["challenger"]),
        ("both", ["control", "challenger"]),
    ],
)
def test_arm_selection_oof_emission_parameterized(
    tmp_path: Path,
    synthetic_train_and_val_frames,
    arm_mode: str,
    expected_arms: list[str],
):
    """Verify runner behavior parameterized across control, challenger, and both."""
    train_df, val_df = synthetic_train_and_val_frames
    split_path = tmp_path / "splits.parquet"
    pl.DataFrame([
        {"building_id": 1, "repeat": 0, "seed": 42, "split": "dev", "fold": 0, "row_count": 25,
         "anomaly_count": 5, "normal_count": 20, "anomaly_prevalence": 0.2, "support_stratum": 0},
        {"building_id": 2, "repeat": 0, "seed": 42, "split": "dev", "fold": 1, "row_count": 25,
         "anomaly_count": 5, "normal_count": 20, "anomaly_prevalence": 0.2, "support_stratum": 0},
    ]).write_parquet(split_path)

    data_path = tmp_path / "train_joined.parquet"
    pl.concat([train_df, val_df], how="vertical").write_parquet(data_path)

    config_data = {
        "target": "anomaly",
        "positive_label": 1,
        "split_manifest": str(split_path),
        "data_path": str(data_path),
        "features": {"numeric": ["meter_reading"], "categorical": []},
        "arms": ["control", "challenger"],
        "models": {
            "lightgbm": {
                "model_family": "lightgbm",
                "recipe_id": "recipe_gbdt_lightgbm",
                "model_config_id": "cfg_gbdt_lightgbm_default",
                "arms": ["control", "challenger"],
                "params": {"learning_rate": 0.03, "seed": 42},
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))

    oof_path = run_standalone_gbdt(
        family="lightgbm",
        config=config_data,
        config_path=config_path,
        output_dir=tmp_path / f"out_{arm_mode}",
        smoke=True,
        dry_run=True,
        arm=arm_mode,
    )
    df = pl.read_parquet(oof_path)

    # Assert exact arm_id sets
    assert sorted(df["arm_id"].unique().to_list()) == sorted(expected_arms)

    # Assert unique candidate IDs matching arms
    expected_candidate_ids = [f"lightgbm_{a}" for a in expected_arms]
    assert sorted(df["candidate_id"].unique().to_list()) == sorted(expected_candidate_ids)

    # Assert per-arm row completeness and equal counts across arms
    per_arm_counts = {
        arm: df.filter(pl.col("arm_id") == arm).height
        for arm in expected_arms
    }
    assert len(set(per_arm_counts.values())) == 1
    base_count = next(iter(per_arm_counts.values()))
    assert base_count > 0
    assert df.height == base_count * len(expected_arms)

    for arm in expected_arms:
        arm_df = df.filter(pl.col("arm_id") == arm)
        keys = arm_df.select(["row_index", "repeat", "fold"]).unique()
        assert keys.height == arm_df.height

    # Assert manifest arms metadata
    manifest_path = oof_path.parent / "lightgbm_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["arm"] == arm_mode
    assert manifest["arms"] == expected_arms


def test_arm_selection_invalid_choice(tmp_path: Path, synthetic_train_and_val_frames):
    """Verify ValueError is raised for invalid arm choices."""
    train_df, _val_df = synthetic_train_and_val_frames
    split_path = tmp_path / "splits.parquet"
    pl.DataFrame([
        {"building_id": 1, "repeat": 0, "seed": 42, "split": "dev", "fold": 0, "row_count": 25,
         "anomaly_count": 5, "normal_count": 20, "anomaly_prevalence": 0.2, "support_stratum": 0},
    ]).write_parquet(split_path)
    data_path = tmp_path / "train_joined.parquet"
    train_df.write_parquet(data_path)

    config_data = {
        "target": "anomaly",
        "positive_label": 1,
        "split_manifest": str(split_path),
        "data_path": str(data_path),
        "features": {"numeric": ["meter_reading"], "categorical": []},
        "models": {
            "lightgbm": {
                "model_family": "lightgbm",
                "params": {"learning_rate": 0.03, "seed": 42},
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))

    with pytest.raises(ValueError, match="Unsupported arm"):
        run_standalone_gbdt(
            family="lightgbm",
            config=config_data,
            config_path=config_path,
            output_dir=tmp_path / "out_err",
            smoke=True,
            dry_run=True,
            arm="invalid_arm",
        )
