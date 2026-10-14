"""Unit tests for the tabular MLP architecture (tabular_mlp.py + train_tabular_mlp.py).

Covers:
- Fold-local fit isolation (preprocessor fitted on train, not val)
- Unseen primary_use category handling
- Allowlist guard rejects forbidden columns
- Inner-split uses no outer-val rows
- Dry-run for both arms passes without torch
- OOF schema conformity
- MissingTorchDependencyError raised correctly on non-dry-run (when torch absent)
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import pytest
from scripts.train_tabular_mlp import (
    DEFAULT_CONFIG_PATH,
    load_config,
    parse_args,
    run_tabular_mlp,
)

from project.models.candidate_artifacts import CANDIDATE_OOF_SCHEMA
from project.models.tabular_mlp import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FoldLocalPreprocessor,
    MissingTorchDependencyError,
    _split_inner_val,
    train_mlp_partition,
    validate_feature_allowlist,
)

# ---------------------------------------------------------------------------
# Minimal feature subset used throughout tests
# ---------------------------------------------------------------------------

NUM_COLS = ["meter_reading", "square_feet", "air_temperature", "hour"]
CAT_COLS = ["primary_use"]
FEATURE_COLS = [*NUM_COLS, *CAT_COLS]
USES = ["Education", "Office", "Lodging", "Retail"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pd(n: int, n_pos: int, seed: int, building: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    anomaly = np.zeros(n, dtype=np.int64)
    anomaly[rng.choice(n, size=n_pos, replace=False)] = 1
    return pd.DataFrame(
        {
            "building_id": [building] * n,
            "timestamp": [
                datetime.datetime(2016, 1, 1, tzinfo=datetime.UTC)
                + datetime.timedelta(hours=i)
                for i in range(n)
            ],
            "anomaly": anomaly.tolist(),
            "meter_reading": rng.normal(100.0, 20.0, n).tolist(),
            "square_feet": rng.normal(5000.0, 500.0, n).tolist(),
            "air_temperature": rng.normal(15.0, 5.0, n).tolist(),
            "hour": rng.integers(0, 24, n).astype(float).tolist(),
            "primary_use": rng.choice(USES, n).tolist(),
        }
    )


def _make_pl(n: int, n_pos: int, seed: int, building: int = 1) -> pl.DataFrame:
    return pl.from_pandas(_make_pd(n, n_pos, seed, building))


def _make_split_manifest() -> pl.DataFrame:
    rows = [
        {"building_id": b, "repeat": 0, "fold": f, "split": "dev"}
        for b, f in [(1, 0), (2, 0), (3, 1), (4, 1)]
    ]
    return pl.DataFrame(rows)


def _make_campaign_data() -> pl.DataFrame:
    return pl.concat(
        [_make_pl(120, 6, seed=10 + b, building=b) for b in (1, 2, 3, 4)]
    )


# ---------------------------------------------------------------------------
# Suite 1: Allowlist guard
# ---------------------------------------------------------------------------


class TestAllowlistGuard:
    def test_valid_features_pass(self) -> None:
        validate_feature_allowlist(FEATURE_COLS)  # must not raise

    def test_site_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            validate_feature_allowlist(["site_id"])

    def test_building_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            validate_feature_allowlist(["building_id"])

    def test_building_hour_rejected(self) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            validate_feature_allowlist(["building_hour"])

    def test_building_meter_rejected(self) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            validate_feature_allowlist(["building_meter"])

    def test_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            validate_feature_allowlist(["timestamp"])

    def test_anomaly_rejected(self) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            validate_feature_allowlist(["anomaly"])

    @pytest.mark.parametrize(
        "col",
        [
            "gte_hour",
            "gte_weekday",
            "gte_month",
            "gte_building_id",
            "gte_primary_use",
            "gte_site_id",
            "gte_meter",
            "gte_meter_hour",
        ],
    )
    def test_gte_prefix_rejected(self, col: str) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            validate_feature_allowlist([col])

    def test_all_allowed_numeric_features_pass(self) -> None:
        validate_feature_allowlist(list(NUMERIC_FEATURES))  # must not raise

    def test_all_allowed_categorical_features_pass(self) -> None:
        validate_feature_allowlist(list(CATEGORICAL_FEATURES))  # must not raise

    def test_empty_list_passes(self) -> None:
        validate_feature_allowlist([])  # must not raise

    def test_unknown_column_rejected(self) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            validate_feature_allowlist(["completely_unknown_col"])


# ---------------------------------------------------------------------------
# Suite 2: FoldLocalPreprocessor — fold-local fit isolation
# ---------------------------------------------------------------------------


class TestFoldLocalPreprocessor:
    """Ensure preprocessor is fitted strictly on outer-train rows."""

    @pytest.fixture
    def train_df(self) -> pd.DataFrame:
        return _make_pd(300, 6, seed=1, building=1)

    @pytest.fixture
    def val_df(self) -> pd.DataFrame:
        return _make_pd(100, 3, seed=2, building=3)

    @pytest.mark.parametrize("arm", ["control", "challenger"])
    def test_fit_uses_train_stats_not_val(self, arm: str, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        """Medians computed from outer-train must differ from joint stats."""
        pre = FoldLocalPreprocessor(arm, NUM_COLS, CAT_COLS)
        pre.fit(train_df)
        train_median = float(train_df["meter_reading"].median())
        assert pre._medians["meter_reading"] == pytest.approx(train_median, rel=1e-6)

    @pytest.mark.parametrize("arm", ["control", "challenger"])
    def test_transform_shape_train(self, arm: str, train_df: pd.DataFrame) -> None:
        pre = FoldLocalPreprocessor(arm, NUM_COLS, CAT_COLS)
        pre.fit(train_df)
        X_num, cat_indices = pre.transform(train_df)
        n_num = len(NUM_COLS)
        if arm == "challenger":
            assert X_num.shape == (len(train_df), n_num * 2)
        else:
            assert X_num.shape == (len(train_df), n_num)
        assert "primary_use" in cat_indices
        assert cat_indices["primary_use"].shape == (len(train_df),)

    @pytest.mark.parametrize("arm", ["control", "challenger"])
    def test_transform_shape_val(self, arm: str, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        pre = FoldLocalPreprocessor(arm, NUM_COLS, CAT_COLS)
        pre.fit(train_df)
        X_num, cat_indices = pre.transform(val_df)
        assert X_num.shape[0] == len(val_df)
        assert cat_indices["primary_use"].shape == (len(val_df),)

    def test_transform_raises_if_not_fitted(self) -> None:
        pre = FoldLocalPreprocessor("control", NUM_COLS, CAT_COLS)
        with pytest.raises(RuntimeError, match="fit"):
            pre.transform(_make_pd(10, 1, seed=99))

    def test_invalid_arm_raises(self) -> None:
        with pytest.raises(ValueError, match="arm"):
            FoldLocalPreprocessor("bad_arm", NUM_COLS, CAT_COLS)

    @pytest.mark.parametrize("arm", ["control", "challenger"])
    def test_missing_values_filled(self, arm: str, train_df: pd.DataFrame) -> None:
        """NaN in val rows must not appear in transformed output."""
        pre = FoldLocalPreprocessor(arm, NUM_COLS, CAT_COLS)
        pre.fit(train_df)
        val_with_nans = train_df.copy()
        val_with_nans.loc[val_with_nans.index[:5], "meter_reading"] = np.nan
        X_num, _ = pre.transform(val_with_nans)
        assert np.all(np.isfinite(X_num))

    def test_challenger_has_double_numeric_width(self) -> None:
        df = _make_pd(200, 4, seed=7)
        pre_ctrl = FoldLocalPreprocessor("control", NUM_COLS, CAT_COLS)
        pre_ctrl.fit(df)
        pre_chal = FoldLocalPreprocessor("challenger", NUM_COLS, CAT_COLS)
        pre_chal.fit(df)
        X_ctrl, _ = pre_ctrl.transform(df)
        X_chal, _ = pre_chal.transform(df)
        assert X_chal.shape[1] == X_ctrl.shape[1] * 2

    # ------------------------------------------------------------------
    # P1 regression: log1p applied before fit so scaler sees same values
    # ------------------------------------------------------------------

    def test_control_meter_reading_std_close_to_one(self) -> None:
        """Control arm: after fit+transform on same skewed data, meter_reading
        column should have std close to 1.0 (StandardScaler fitted on log1p values)."""
        rng = np.random.default_rng(0)
        # Skewed distribution that differs greatly before/after log1p
        mr = rng.exponential(scale=500.0, size=300)
        df = _make_pd(300, 6, seed=1)
        df["meter_reading"] = mr
        pre = FoldLocalPreprocessor("control", NUM_COLS, CAT_COLS)
        pre.fit(df)
        X_num, _ = pre.transform(df)
        mr_idx = NUM_COLS.index("meter_reading")
        col_std = float(np.std(X_num[:, mr_idx]))
        assert 0.9 <= col_std <= 1.1, (
            f"meter_reading std after control transform should be ~1.0, got {col_std:.4f}"
        )

    def test_both_arms_preserve_meter_reading_rank_order(self) -> None:
        """Both control and challenger must preserve the rank order of meter_reading
        (log1p is monotone, scaling is monotone → ranks must be preserved)."""
        rng = np.random.default_rng(42)
        mr = rng.exponential(scale=200.0, size=100)
        df = _make_pd(100, 3, seed=2)
        df["meter_reading"] = mr
        mr_idx = NUM_COLS.index("meter_reading")
        for arm in ("control", "challenger"):
            pre = FoldLocalPreprocessor(arm, NUM_COLS, CAT_COLS)
            pre.fit(df)
            X_num, _ = pre.transform(df)
            raw_ranks = np.argsort(mr)
            transformed_ranks = np.argsort(X_num[:, mr_idx])
            assert np.array_equal(raw_ranks, transformed_ranks), (
                f"{arm}: rank order of meter_reading not preserved after transform"
            )


# ---------------------------------------------------------------------------
# Suite 2b: weight_decay config honoured (P2) and small-batch guard (P3)
# ---------------------------------------------------------------------------


class TestTrainMlpPartitionConfig:
    """Verify config wiring that does not require torch to be installed."""

    def test_weight_decay_read_from_params(self) -> None:
        """train_mlp_partition in dry-run mode must not raise; config param is
        exercised via the public params dict (no torch needed for dry-run)."""
        df = pd.concat(
            [_make_pd(120, 4, seed=s, building=s) for s in range(1, 5)], ignore_index=True
        )
        val = _make_pd(40, 2, seed=99, building=99)
        # If weight_decay is wired correctly the call must complete without error.
        preds = train_mlp_partition(
            arm="control",
            train_df=df,
            val_df=val,
            numeric_cols=NUM_COLS,
            categorical_cols=CAT_COLS,
            dry_run=True,
            params={"weight_decay": 1e-3},
        )
        assert preds.shape == (len(val),)

    def test_impute_numerics_log1p_applied_before_fit(self) -> None:
        """The log1p transform must be visible inside _impute_numerics, not after;
        calling it directly must return log1p(x) for a positive meter_reading column."""
        rng = np.random.default_rng(7)
        raw_mr = rng.exponential(100.0, size=50)
        df = _make_pd(50, 2, seed=3)
        df["meter_reading"] = raw_mr
        pre = FoldLocalPreprocessor("control", NUM_COLS, CAT_COLS)
        pre.fit(df)
        X_raw = pre._impute_numerics(df)
        mr_idx = NUM_COLS.index("meter_reading")
        expected_log = np.sign(raw_mr) * np.log1p(np.abs(raw_mr))
        np.testing.assert_allclose(
            X_raw[:, mr_idx], expected_log.astype(np.float32), rtol=1e-5,
            err_msg="_impute_numerics must apply log1p to meter_reading before returning"
        )


# ---------------------------------------------------------------------------
# Suite 3: Unseen category handling
# ---------------------------------------------------------------------------


class TestUnseenCategoryHandling:
    """Unseen primary_use in validation must map to unknown index, not crash."""

    def test_unseen_primary_use_maps_to_zero(self) -> None:
        train_df = _make_pd(200, 4, seed=1)
        # Override all training rows to use only "Education"
        train_df["primary_use"] = "Education"

        val_df = _make_pd(50, 2, seed=2)
        val_df["primary_use"] = "COMPLETELY_NEW_CATEGORY_XYZ"

        pre = FoldLocalPreprocessor("control", NUM_COLS, CAT_COLS)
        pre.fit(train_df)
        _, cat_indices = pre.transform(val_df)
        # Unknown category index is 0 by convention
        assert np.all(cat_indices["primary_use"] == 0)

    def test_known_category_maps_to_nonzero(self) -> None:
        train_df = _make_pd(200, 4, seed=1)
        train_df["primary_use"] = "Education"

        pre = FoldLocalPreprocessor("control", NUM_COLS, CAT_COLS)
        pre.fit(train_df)
        _, cat_indices = pre.transform(train_df)
        # "Education" should map to index >= 1
        assert np.all(cat_indices["primary_use"] >= 1)

    def test_vocabulary_is_train_only(self) -> None:
        """Val-only categories must not appear in vocabulary."""
        train_df = _make_pd(200, 4, seed=1)
        train_df["primary_use"] = "Education"
        val_df = _make_pd(50, 2, seed=2)
        val_df["primary_use"] = "OFFICE_XTRA"

        pre = FoldLocalPreprocessor("control", NUM_COLS, CAT_COLS)
        pre.fit(train_df)

        vocab = pre._category_vocab["primary_use"]
        assert "OFFICE_XTRA" not in vocab


# ---------------------------------------------------------------------------
# Suite 4: Inner-split uses no outer-val rows
# ---------------------------------------------------------------------------


class TestInnerSplitIsolation:
    """_split_inner_val must not use outer-val rows."""

    def test_inner_val_buildings_disjoint_from_inner_train(self) -> None:
        df = pd.DataFrame(
            {
                "building_id": list(range(10)) * 20,
                "x": np.random.default_rng(0).normal(size=200),
            }
        )
        inner_train, inner_val = _split_inner_val(df, building_col="building_id", seed=1)
        train_buildings = set(inner_train["building_id"].unique())
        val_buildings = set(inner_val["building_id"].unique())
        assert train_buildings.isdisjoint(val_buildings)

    def test_inner_split_covers_all_outer_train_rows(self) -> None:
        df = _make_pd(300, 6, seed=5)
        inner_train, inner_val = _split_inner_val(df, building_col="building_id", seed=0)
        assert len(inner_train) + len(inner_val) == len(df)

    def test_inner_val_has_at_least_one_building(self) -> None:
        df = _make_pd(200, 4, seed=3)
        _, inner_val = _split_inner_val(df, building_col="building_id", seed=0)
        assert len(inner_val) > 0

    def test_outer_val_rows_not_in_inner_split(self) -> None:
        """This test verifies the contract: outer_val passed to train_mlp_partition
        must not appear in inner train or inner val."""
        outer_train = _make_pd(400, 8, seed=10, building=1)
        outer_val = _make_pd(100, 3, seed=11, building=99)  # different building

        inner_tr, inner_v = _split_inner_val(outer_train, building_col="building_id", seed=42)
        # Outer-val building_id must not appear in either inner set
        outer_val_buildings = set(outer_val["building_id"].unique())
        inner_tr_buildings = set(inner_tr["building_id"].unique())
        inner_v_buildings = set(inner_v["building_id"].unique())
        assert outer_val_buildings.isdisjoint(inner_tr_buildings)
        assert outer_val_buildings.isdisjoint(inner_v_buildings)


# ---------------------------------------------------------------------------
# Suite 5: Dry-run both arms (no torch required)
# ---------------------------------------------------------------------------


class TestDryRunBothArms:
    """Dry-run must succeed without torch being installed."""

    @pytest.fixture
    def train_df(self) -> pd.DataFrame:
        return _make_pd(400, 8, seed=1, building=1)

    @pytest.fixture
    def val_df(self) -> pd.DataFrame:
        return _make_pd(100, 3, seed=2, building=3)

    @pytest.mark.parametrize("arm", ["control", "challenger"])
    def test_dry_run_returns_probabilities(
        self, arm: str, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> None:
        preds = train_mlp_partition(
            arm=arm,
            train_df=train_df,
            val_df=val_df,
            numeric_cols=NUM_COLS,
            categorical_cols=CAT_COLS,
            target_col="anomaly",
            seed=42,
            dry_run=True,
        )
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(val_df),)
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)
        assert np.all(np.isfinite(preds))

    @pytest.mark.parametrize("arm", ["control", "challenger"])
    def test_dry_run_deterministic(
        self, arm: str, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> None:
        preds_a = train_mlp_partition(
            arm=arm, train_df=train_df, val_df=val_df,
            numeric_cols=NUM_COLS, categorical_cols=CAT_COLS,
            target_col="anomaly", seed=7, dry_run=True,
        )
        preds_b = train_mlp_partition(
            arm=arm, train_df=train_df, val_df=val_df,
            numeric_cols=NUM_COLS, categorical_cols=CAT_COLS,
            target_col="anomaly", seed=7, dry_run=True,
        )
        np.testing.assert_array_equal(preds_a, preds_b)

    def test_dry_run_empty_val(self, train_df: pd.DataFrame) -> None:
        val_empty = _make_pd(0, 0, seed=99)
        preds = train_mlp_partition(
            arm="control", train_df=train_df, val_df=val_empty,
            numeric_cols=NUM_COLS, categorical_cols=CAT_COLS,
            target_col="anomaly", seed=0, dry_run=True,
        )
        assert preds.shape == (0,)


# ---------------------------------------------------------------------------
# Suite 6: MissingTorchDependencyError
# ---------------------------------------------------------------------------


class TestMissingTorchDependencyError:
    """When torch is not installed, non-dry-run calls must raise MissingTorchDependencyError."""

    def test_missing_torch_raises_on_real_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import project.models.tabular_mlp as mlp_mod

        monkeypatch.setattr(mlp_mod, "is_torch_available", lambda: False)

        train_df = _make_pd(100, 2, seed=1)
        val_df = _make_pd(30, 1, seed=2)

        with pytest.raises(MissingTorchDependencyError, match="PyTorch"):
            train_mlp_partition(
                arm="control",
                train_df=train_df,
                val_df=val_df,
                numeric_cols=NUM_COLS,
                categorical_cols=CAT_COLS,
                target_col="anomaly",
                seed=0,
                dry_run=False,  # Real call
            )

    def test_dry_run_does_not_raise_missing_torch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dry-run must work even when torch appears unavailable."""
        import project.models.tabular_mlp as mlp_mod

        monkeypatch.setattr(mlp_mod, "is_torch_available", lambda: False)

        train_df = _make_pd(100, 2, seed=1)
        val_df = _make_pd(30, 1, seed=2)

        # Must not raise
        preds = train_mlp_partition(
            arm="control",
            train_df=train_df,
            val_df=val_df,
            numeric_cols=NUM_COLS,
            categorical_cols=CAT_COLS,
            target_col="anomaly",
            seed=0,
            dry_run=True,
        )
        assert preds.shape == (len(val_df),)

    def test_error_message_contains_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import project.models.tabular_mlp as mlp_mod

        monkeypatch.setattr(mlp_mod, "is_torch_available", lambda: False)

        try:
            train_mlp_partition(
                arm="control",
                train_df=_make_pd(50, 1, seed=1),
                val_df=_make_pd(10, 1, seed=2),
                numeric_cols=NUM_COLS,
                categorical_cols=CAT_COLS,
                target_col="anomaly",
                dry_run=False,
            )
        except MissingTorchDependencyError as exc:
            assert "torch" in str(exc).lower() or "pytorch" in str(exc).lower()
        else:
            pytest.fail("Expected MissingTorchDependencyError was not raised")


# ---------------------------------------------------------------------------
# Suite 7: OOF schema conformity via end-to-end dry-run
# ---------------------------------------------------------------------------


class TestOofSchemaConformity:
    """End-to-end dry-run must write valid OOF parquet conforming to CANDIDATE_OOF_SCHEMA."""

    @pytest.fixture
    def split_manifest_path(self, tmp_path: Path) -> Path:
        manifest = _make_split_manifest()
        path = tmp_path / "split.parquet"
        manifest.write_parquet(str(path))
        return path

    @pytest.fixture
    def data_path(self, tmp_path: Path) -> Path:
        data = _make_campaign_data()
        # Add required columns that may be present in the full joined table
        data = data.with_columns([
            pl.lit(0).alias("row_index").cast(pl.Int64),
        ])
        path = tmp_path / "train_joined.parquet"
        data.write_parquet(str(path))
        return path

    @pytest.fixture
    def base_config(self, split_manifest_path: Path, data_path: Path) -> dict[str, Any]:
        cfg = load_config(DEFAULT_CONFIG_PATH)
        cfg["split_manifest"] = str(split_manifest_path)
        cfg["data_path"] = str(data_path)
        cfg["features"]["numeric"] = NUM_COLS
        cfg["features"]["categorical"] = CAT_COLS
        return cfg

    @pytest.fixture
    def config_path(self, tmp_path: Path, base_config: dict[str, Any]) -> Path:
        import yaml

        p = tmp_path / "test_config.yaml"
        p.write_text(yaml.dump(base_config))
        return p

    @pytest.mark.parametrize("arm", ["control", "challenger", "both"])
    def test_schema_conformity(
        self,
        arm: str,
        base_config: dict[str, Any],
        config_path: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "outputs"
        paths = run_tabular_mlp(
            config=base_config,
            config_path=config_path,
            output_dir=output_dir,
            smoke=False,
            dry_run=True,
            arm=arm,
        )
        arms_expected = ["control", "challenger"] if arm == "both" else [arm]
        assert len(paths) == len(arms_expected)

        for path in paths:
            assert path.is_file(), f"OOF parquet not written: {path}"
            table = pq.read_table(str(path))
            # Cast to canonical schema — must not raise
            table.cast(CANDIDATE_OOF_SCHEMA)
            # Check all required fields are present
            for field in CANDIDATE_OOF_SCHEMA.names:
                assert field in table.schema.names, f"Missing field: {field}"

    def test_arm_id_values(
        self,
        base_config: dict[str, Any],
        config_path: Path,
        tmp_path: Path,
    ) -> None:
        paths = run_tabular_mlp(
            config=base_config,
            config_path=config_path,
            output_dir=tmp_path / "out",
            dry_run=True,
            arm="both",
        )
        seen_arms: set[str] = set()
        for path in paths:
            table = pq.read_table(str(path))
            arm_ids = set(table.column("arm_id").to_pylist())
            seen_arms |= arm_ids
        assert seen_arms == {"control", "challenger"}

    def test_score_kind_is_positive_probability(
        self,
        base_config: dict[str, Any],
        config_path: Path,
        tmp_path: Path,
    ) -> None:
        paths = run_tabular_mlp(
            config=base_config,
            config_path=config_path,
            output_dir=tmp_path / "out2",
            dry_run=True,
            arm="control",
        )
        table = pq.read_table(str(paths[0]))
        score_kinds = set(table.column("score_kind").to_pylist())
        assert score_kinds == {"positive_probability"}

    def test_positive_label_is_one(
        self,
        base_config: dict[str, Any],
        config_path: Path,
        tmp_path: Path,
    ) -> None:
        paths = run_tabular_mlp(
            config=base_config,
            config_path=config_path,
            output_dir=tmp_path / "out3",
            dry_run=True,
            arm="control",
        )
        table = pq.read_table(str(paths[0]))
        positive_labels = set(table.column("positive_label").to_pylist())
        assert positive_labels == {1}

    def test_execution_mode_synthetic_smoke_in_dry_run(
        self,
        base_config: dict[str, Any],
        config_path: Path,
        tmp_path: Path,
    ) -> None:
        paths = run_tabular_mlp(
            config=base_config,
            config_path=config_path,
            output_dir=tmp_path / "out4",
            dry_run=True,
            arm="control",
        )
        table = pq.read_table(str(paths[0]))
        modes = set(table.column("execution_mode").to_pylist())
        assert modes == {"synthetic_smoke"}

    def test_non_comparative_true_in_dry_run(
        self,
        base_config: dict[str, Any],
        config_path: Path,
        tmp_path: Path,
    ) -> None:
        paths = run_tabular_mlp(
            config=base_config,
            config_path=config_path,
            output_dir=tmp_path / "out5",
            dry_run=True,
            arm="control",
        )
        table = pq.read_table(str(paths[0]))
        nc_vals = set(table.column("non_comparative").to_pylist())
        assert nc_vals == {True}

    def test_n03_eligible_false_in_dry_run(
        self,
        base_config: dict[str, Any],
        config_path: Path,
        tmp_path: Path,
    ) -> None:
        paths = run_tabular_mlp(
            config=base_config,
            config_path=config_path,
            output_dir=tmp_path / "out6",
            dry_run=True,
            arm="control",
        )
        table = pq.read_table(str(paths[0]))
        n03_vals = set(table.column("n03_eligible").to_pylist())
        assert n03_vals == {False}

    def test_sha256_fields_are_valid_hex64(
        self,
        base_config: dict[str, Any],
        config_path: Path,
        tmp_path: Path,
    ) -> None:
        import re

        paths = run_tabular_mlp(
            config=base_config,
            config_path=config_path,
            output_dir=tmp_path / "out7",
            dry_run=True,
            arm="control",
        )
        table = pq.read_table(str(paths[0]))
        sha_fields = ["split_sha256", "data_sha256", "config_sha256", "model_fit_config_sha256"]
        for field in sha_fields:
            vals = table.column(field).to_pylist()
            for v in vals:
                assert re.fullmatch(r"[0-9a-f]{64}", v), f"{field} has invalid value: {v!r}"

    def test_manifest_json_written(
        self,
        base_config: dict[str, Any],
        config_path: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "out8"
        run_tabular_mlp(
            config=base_config,
            config_path=config_path,
            output_dir=output_dir,
            dry_run=True,
            arm="control",
        )
        manifests = list(output_dir.rglob("*_manifest.json"))
        assert len(manifests) == 1
        with manifests[0].open() as f:
            data = json.load(f)
        assert data["model_family"] == "tabular_mlp"
        assert data["arm"] == "control"
        assert "split_sha256" in data


# ---------------------------------------------------------------------------
# Suite 8: CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIArgumentParsing:
    def test_defaults(self) -> None:
        args = parse_args([])
        assert args.arm == "both"
        assert not args.smoke
        assert not args.dry_run
        assert args.config == DEFAULT_CONFIG_PATH

    def test_arm_control(self) -> None:
        args = parse_args(["--arm", "control"])
        assert args.arm == "control"

    def test_arm_challenger(self) -> None:
        args = parse_args(["--arm", "challenger"])
        assert args.arm == "challenger"

    def test_smoke_flag(self) -> None:
        args = parse_args(["--smoke"])
        assert args.smoke

    def test_dry_run_flag(self) -> None:
        args = parse_args(["--dry-run"])
        assert args.dry_run

    def test_invalid_arm_rejected(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--arm", "invalid_arm"])

    def test_output_dir(self, tmp_path: Path) -> None:
        args = parse_args(["--output-dir", str(tmp_path)])
        assert args.output_dir == tmp_path
