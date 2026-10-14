"""Contract tests for the 3-arm synthetic sampling architecture.

Covers fold-local isolation, Tabular Mixup mathematics, SMOTE/Mixup positive
ratios, config/CLI handling, dry-run execution for every family x arm and
OOF schema conformity. No real model training is launched at scale.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import pytest
from scripts.train_synthetic_sampling import (
    DEFAULT_CONFIG_PATH,
    load_config,
    parse_args,
    run_synthetic_sampling,
)

from project.models import synthetic_sampling as ss
from project.models.candidate_artifacts import CANDIDATE_OOF_SCHEMA
from project.models.gbdt_standalone import MissingGbdtDependencyError, is_gbdt_available
from project.models.synthetic_sampling import (
    SUPPORTED_ARMS,
    SUPPORTED_FAMILIES,
    apply_fold_local_smote,
    apply_fold_local_tabular_mixup,
    normalize_arm,
    normalize_family,
    train_synthetic_sampling_partition,
    validate_predictor_cols,
)

NUMERIC_COLS = ["meter_reading", "square_feet", "air_temperature", "hour"]
CAT_COLS = ["primary_use"]
FEATURE_COLS = [*NUMERIC_COLS, *CAT_COLS]
USES = ["Education", "Office", "Lodging"]


def _make_frame(n: int, n_pos: int, seed: int, building: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    anomaly = np.zeros(n, dtype=np.int64)
    anomaly[rng.choice(n, size=n_pos, replace=False)] = 1
    return pl.DataFrame(
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


@pytest.fixture
def train_val() -> tuple[pl.DataFrame, pl.DataFrame]:
    return _make_frame(400, 8, seed=1, building=1), _make_frame(120, 6, seed=2, building=3)


@pytest.fixture
def xy(train_val: tuple[pl.DataFrame, pl.DataFrame]) -> tuple[pd.DataFrame, np.ndarray]:
    train = train_val[0]
    return train.select(FEATURE_COLS).to_pandas(), train["anomaly"].to_numpy()


@pytest.fixture
def split_manifest() -> pl.DataFrame:
    rows = [
        {"building_id": b, "repeat": 0, "fold": f, "split": "dev"}
        for b, f in [(1, 0), (2, 0), (3, 1), (4, 1)]
    ]
    return pl.DataFrame(rows)


@pytest.fixture
def campaign_data() -> pl.DataFrame:
    return pl.concat(
        [_make_frame(150, 12, seed=10 + b, building=b) for b in (1, 2, 3, 4)]
    )


@pytest.fixture
def config() -> dict[str, Any]:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg["features"]["predictors"] = FEATURE_COLS
    cfg["features"]["categorical"] = CAT_COLS
    cfg["split_manifest"] = "outputs/splits/nonexistent_split.parquet"
    cfg["data_path"] = "outputs/prepared/nonexistent_data.parquet"
    return cfg


def _ratio(y: np.ndarray) -> float:
    return float(np.mean(y))


# ==============================================================================
# Identifiers and predictor allowlist
# ==============================================================================


class TestNormalization:
    def test_aliases(self) -> None:
        assert normalize_family("LGBM") == "lightgbm"
        assert normalize_family("extra_trees") == "extratrees"
        assert normalize_arm("SMOTENC") == "smote"
        assert normalize_arm("tabular_mixup") == "mixup"

    def test_unsupported(self) -> None:
        with pytest.raises(ValueError, match="Unsupported model family"):
            normalize_family("catboost")
        with pytest.raises(ValueError, match="Unsupported sampling arm"):
            normalize_arm("adasyn")

    def test_forbidden_predictors_rejected(self) -> None:
        for bad in ("gte_hour", "site_id", "building_id", "timestamp", "anomaly"):
            with pytest.raises(ValueError, match="Forbidden predictor"):
                validate_predictor_cols(["meter_reading", bad])
        with pytest.raises(ValueError, match="Duplicate"):
            validate_predictor_cols(["a", "a"])
        with pytest.raises(ValueError, match="empty"):
            validate_predictor_cols([])


# ==============================================================================
# Samplers
# ==============================================================================


class TestResampledRatios:
    @pytest.mark.parametrize("sampler", ["smote", "mixup"])
    def test_reaches_target_ratio(
        self, xy: tuple[pd.DataFrame, np.ndarray], sampler: str
    ) -> None:
        X, y = xy
        assert _ratio(y) == pytest.approx(0.02)
        fn = apply_fold_local_smote if sampler == "smote" else apply_fold_local_tabular_mixup
        X_res, y_res = fn(X, y, CAT_COLS, target_ratio=0.08, seed=7)
        assert 0.08 <= _ratio(y_res) < 0.085
        assert len(X_res) == len(y_res)
        assert list(X_res.columns) == list(X.columns)

    @pytest.mark.parametrize("sampler", ["smote", "mixup"])
    def test_real_rows_preserved_and_synthetic_appended(
        self, xy: tuple[pd.DataFrame, np.ndarray], sampler: str
    ) -> None:
        X, y = xy
        fn = apply_fold_local_smote if sampler == "smote" else apply_fold_local_tabular_mixup
        X_res, y_res = fn(X, y, CAT_COLS, seed=3)
        n = len(X)
        np.testing.assert_array_equal(y_res[:n], y)
        assert set(y_res[n:]) == {1}
        np.testing.assert_allclose(X_res[NUMERIC_COLS].iloc[:n].to_numpy(), X[NUMERIC_COLS].to_numpy())
        assert (X_res[CAT_COLS[0]].iloc[:n].to_numpy() == X[CAT_COLS[0]].to_numpy()).all()

    @pytest.mark.parametrize("sampler", ["smote", "mixup"])
    def test_deterministic_and_seed_sensitive(
        self, xy: tuple[pd.DataFrame, np.ndarray], sampler: str
    ) -> None:
        X, y = xy
        fn = apply_fold_local_smote if sampler == "smote" else apply_fold_local_tabular_mixup
        a, _ = fn(X, y, CAT_COLS, seed=5)
        b, _ = fn(X, y, CAT_COLS, seed=5)
        c, _ = fn(X, y, CAT_COLS, seed=6)
        pd.testing.assert_frame_equal(a, b)
        assert not a.equals(c)

    @pytest.mark.parametrize("sampler", ["smote", "mixup"])
    def test_noop_when_ratio_met_or_too_few_positives(
        self, xy: tuple[pd.DataFrame, np.ndarray], sampler: str
    ) -> None:
        X, y = xy
        fn = apply_fold_local_smote if sampler == "smote" else apply_fold_local_tabular_mixup
        X_res, y_res = fn(X, y, CAT_COLS, target_ratio=0.01)
        assert len(X_res) == len(X)
        y_one = np.zeros_like(y)
        y_one[0] = 1
        X_res, y_res = fn(X, y_one, CAT_COLS)
        assert len(X_res) == len(X) and y_res.sum() == 1

    def test_input_validation(self, xy: tuple[pd.DataFrame, np.ndarray]) -> None:
        X, y = xy
        with pytest.raises(ValueError, match="target_ratio"):
            apply_fold_local_smote(X, y, CAT_COLS, target_ratio=1.5)
        with pytest.raises(ValueError, match="binary"):
            apply_fold_local_tabular_mixup(X, y + 2, CAT_COLS)
        with pytest.raises(ValueError, match="categorical_cols"):
            apply_fold_local_smote(X, y, ["nope"])
        with pytest.raises(ValueError, match="alpha"):
            apply_fold_local_tabular_mixup(X, y, CAT_COLS, alpha=0.0)


class TestTabularMixupMath:
    def test_convex_combination_and_category_preservation(self) -> None:
        # Two positives that differ in every column: every synthetic value must
        # lie in the closed interval spanned by the parents.
        X = pd.DataFrame(
            {
                "a": [0.0, 10.0] + [5.0] * 98,
                "b": [-4.0, 4.0] + [0.0] * 98,
                "primary_use": ["X", "Y"] + ["Z"] * 98,
            }
        )
        y = np.zeros(100, dtype=np.int64)
        y[:2] = 1
        X_res, y_res = apply_fold_local_tabular_mixup(
            X, y, ["primary_use"], target_ratio=0.5, alpha=0.4, seed=11
        )
        synth = X_res.iloc[100:]
        assert len(synth) > 0 and set(y_res[100:]) == {1}
        assert synth["a"].between(0.0, 10.0).all()
        assert synth["b"].between(-4.0, 4.0).all()
        # Both parents share the same lambda: a = 10 * (1 - lam), b = -4 + 8 * (1 - lam).
        np.testing.assert_allclose(synth["b"].to_numpy(), -4.0 + 0.8 * synth["a"].to_numpy(), atol=1e-9)
        assert set(synth["primary_use"]) <= {"X", "Y"}
        assert set(synth["primary_use"]) == {"X", "Y"}  # Bernoulli picks both parents

    def test_categorical_follows_lambda(self) -> None:
        # Parent order is random, so P(cat == parent with larger weight) ~ E[max(lam, 1-lam)]
        # is not the check; instead verify the category agrees with the dominant
        # parent far above chance when lambda is pushed to the extremes (alpha small).
        X = pd.DataFrame(
            {"a": [0.0, 1.0] + [0.5] * 98, "primary_use": ["X", "Y"] + ["Z"] * 98}
        )
        y = np.zeros(100, dtype=np.int64)
        y[:2] = 1
        X_res, _ = apply_fold_local_tabular_mixup(
            X, y, ["primary_use"], target_ratio=0.6, alpha=0.05, seed=2
        )
        synth = X_res.iloc[100:]
        # a == weight on parent "Y"; category "Y" is chosen with that same probability.
        agree = np.where(synth["a"] > 0.5, synth["primary_use"] == "Y", synth["primary_use"] == "X")
        assert agree.mean() > 0.8

    def test_missing_numeric_takes_other_parent(self) -> None:
        X = pd.DataFrame(
            {"a": [np.nan, 3.0] + [1.0] * 8, "primary_use": ["X", "Y"] + ["Z"] * 8}
        )
        y = np.zeros(10, dtype=np.int64)
        y[:2] = 1
        X_res, _ = apply_fold_local_tabular_mixup(X, y, ["primary_use"], target_ratio=0.5, seed=1)
        assert X_res["a"].iloc[10:].eq(3.0).all()


class TestSmoteNative:
    def test_synthetic_numeric_within_positive_hull_and_categories_valid(
        self, xy: tuple[pd.DataFrame, np.ndarray], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ss, "_load_smotenc", lambda: None)
        X, y = xy
        X_res, y_res = apply_fold_local_smote(X, y, CAT_COLS, seed=4)
        pos = X[y == 1]
        synth = X_res.iloc[len(X):]
        for col in NUMERIC_COLS:
            assert synth[col].between(pos[col].min(), pos[col].max()).all()
        assert set(synth["primary_use"]) <= set(USES)
        assert 0.08 <= _ratio(y_res) < 0.085

    def test_no_categorical_columns(self, xy: tuple[pd.DataFrame, np.ndarray]) -> None:
        X, y = xy
        X_num = X[NUMERIC_COLS]
        X_res, y_res = apply_fold_local_smote(X_num, y, [], seed=1)
        assert list(X_res.columns) == NUMERIC_COLS
        assert 0.08 <= _ratio(y_res) < 0.085

    def test_imblearn_path_used_when_available(
        self, xy: tuple[pd.DataFrame, np.ndarray], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, Any] = {}

        class FakeSMOTENC:
            def __init__(self, **kwargs: Any) -> None:
                calls.update(kwargs)

            def fit_resample(self, X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
                n_new = calls["sampling_strategy"][1] - int(y.sum())
                extra = X[y == 1].iloc[:1].sample(n=n_new, replace=True, random_state=0)
                return (
                    pd.concat([X, extra], ignore_index=True),
                    np.concatenate([y, np.ones(n_new, dtype=y.dtype)]),
                )

        monkeypatch.setattr(ss, "_load_smotenc", lambda: FakeSMOTENC)
        X, y = xy
        X_res, y_res = apply_fold_local_smote(X, y, CAT_COLS, seed=9, k_neighbors=5)
        assert calls["categorical_features"] == [FEATURE_COLS.index("primary_use")]
        assert calls["random_state"] == 9
        assert 0.08 <= _ratio(y_res) < 0.085
        assert set(X_res["primary_use"]) <= set(USES)


# ==============================================================================
# Fold-local isolation
# ==============================================================================


class TestFoldLocalIsolation:
    @pytest.mark.parametrize("arm", SUPPORTED_ARMS)
    def test_validation_fold_never_modified(
        self, train_val: tuple[pl.DataFrame, pl.DataFrame], arm: str
    ) -> None:
        train_df, val_df = train_val
        val_before = val_df.clone()
        train_before = train_df.clone()
        preds = train_synthetic_sampling_partition(
            "extratrees", arm, train_df, val_df, FEATURE_COLS, CAT_COLS, "anomaly",
            params={"n_estimators": 5, "seed": 1},
        )
        assert preds.shape == (val_df.height,)
        assert val_df.equals(val_before)
        assert train_df.equals(train_before)

    @pytest.mark.parametrize("arm", ["smote", "mixup"])
    def test_samplers_never_see_validation_rows(
        self,
        train_val: tuple[pl.DataFrame, pl.DataFrame],
        arm: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        train_df, val_df = train_val
        seen: list[int] = []
        real = ss.apply_arm_sampling

        def spy(arm_id: str, X: pd.DataFrame, y: np.ndarray, *a: Any, **kw: Any) -> Any:
            seen.append(len(X))
            return real(arm_id, X, y, *a, **kw)

        monkeypatch.setattr(ss, "apply_arm_sampling", spy)
        train_synthetic_sampling_partition(
            "lightgbm", arm, train_df, val_df, FEATURE_COLS, CAT_COLS, "anomaly",
            dry_run=True,
        )
        assert seen == [train_df.height]

    def test_dry_run_arms_differ_from_natural(
        self, train_val: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        train_df, val_df = train_val
        out = {
            arm: train_synthetic_sampling_partition(
                "lightgbm", arm, train_df, val_df, FEATURE_COLS, CAT_COLS, "anomaly",
                dry_run=True,
            )
            for arm in SUPPORTED_ARMS
        }
        assert out["natural"].mean() == pytest.approx(0.02, abs=0.01)
        assert out["smote"].mean() == pytest.approx(0.08, abs=0.01)
        assert out["mixup"].mean() == pytest.approx(0.08, abs=0.01)

    def test_categorical_must_be_in_features(
        self, train_val: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        train_df, val_df = train_val
        with pytest.raises(ValueError, match="not in feature_cols"):
            train_synthetic_sampling_partition(
                "lightgbm", "natural", train_df, val_df, NUMERIC_COLS, CAT_COLS,
                "anomaly", dry_run=True,
            )

    def test_forbidden_feature_rejected(
        self, train_val: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        train_df, val_df = train_val
        with pytest.raises(ValueError, match="Forbidden predictor"):
            train_synthetic_sampling_partition(
                "lightgbm", "natural", train_df, val_df, [*FEATURE_COLS, "building_id"],
                CAT_COLS, "anomaly", dry_run=True,
            )


# ==============================================================================
# Model execution
# ==============================================================================


class TestModelExecution:
    @pytest.mark.parametrize("family", SUPPORTED_FAMILIES)
    @pytest.mark.parametrize("arm", SUPPORTED_ARMS)
    def test_dry_run_all_combinations(
        self, train_val: tuple[pl.DataFrame, pl.DataFrame], family: str, arm: str
    ) -> None:
        train_df, val_df = train_val
        kwargs: dict[str, Any] = {"params": {"seed": 3}, "dry_run": True}
        a = train_synthetic_sampling_partition(
            family, arm, train_df, val_df, FEATURE_COLS, CAT_COLS, "anomaly", **kwargs
        )
        b = train_synthetic_sampling_partition(
            family, arm, train_df, val_df, FEATURE_COLS, CAT_COLS, "anomaly", **kwargs
        )
        assert a.dtype == np.float64 and a.shape == (val_df.height,)
        assert np.all((a >= 0.0) & (a <= 1.0))
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("arm", SUPPORTED_ARMS)
    def test_extratrees_real_fit_small(
        self, train_val: tuple[pl.DataFrame, pl.DataFrame], arm: str
    ) -> None:
        train_df, val_df = train_val
        preds = train_synthetic_sampling_partition(
            "extratrees", arm, train_df, val_df, FEATURE_COLS, CAT_COLS, "anomaly",
            params={"n_estimators": 10, "seed": 0},
        )
        assert preds.shape == (val_df.height,)
        assert np.all((preds >= 0.0) & (preds <= 1.0))

    def test_extratrees_handles_missing_and_unseen_categories(
        self, train_val: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        train_df, val_df = train_val
        train_df = train_df.with_columns(
            pl.when(pl.int_range(pl.len()) % 7 == 0)
            .then(None)
            .otherwise(pl.col("meter_reading"))
            .alias("meter_reading")
        )
        val_df = val_df.with_columns(pl.lit("Unseen").alias("primary_use"))
        preds = train_synthetic_sampling_partition(
            "extratrees", "mixup", train_df, val_df, FEATURE_COLS, CAT_COLS, "anomaly",
            params={"n_estimators": 5},
        )
        assert np.isfinite(preds).all()

    def test_lightgbm_real_fit_requires_dependency(
        self, train_val: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        train_df, val_df = train_val
        args = ("lightgbm", "smote", train_df, val_df, FEATURE_COLS, CAT_COLS, "anomaly")
        if is_gbdt_available("lightgbm"):
            preds = train_synthetic_sampling_partition(
                *args, params={"n_estimators": 5, "num_leaves": 4, "min_child_samples": 2}
            )
            assert preds.shape == (val_df.height,)
        else:
            with pytest.raises(MissingGbdtDependencyError):
                train_synthetic_sampling_partition(*args)

    def test_empty_validation_returns_empty(
        self, train_val: tuple[pl.DataFrame, pl.DataFrame]
    ) -> None:
        train_df, val_df = train_val
        preds = train_synthetic_sampling_partition(
            "lightgbm", "mixup", train_df, val_df.head(0), FEATURE_COLS, CAT_COLS,
            "anomaly", dry_run=True,
        )
        assert preds.shape == (0,)


# ==============================================================================
# Config and CLI
# ==============================================================================


class TestConfigAndCli:
    def test_config_declares_campaign(self) -> None:
        cfg = load_config(DEFAULT_CONFIG_PATH)
        assert set(cfg["models"]) == set(SUPPORTED_FAMILIES)
        assert cfg["sampling_arms"] == list(SUPPORTED_ARMS)
        assert cfg["sampling"]["target_ratio"] == 0.08
        assert cfg["sampling"]["mixup_alpha"] == 0.4
        predictors = cfg["features"]["predictors"]
        assert len(predictors) == 31 and len(set(predictors)) == 31
        assert cfg["features"]["categorical"] == ["primary_use"]
        validate_predictor_cols(predictors)
        assert not any(c.startswith("gte_") for c in predictors)
        assert cfg["split_manifest"] == "outputs/splits/building_folds_repeated.parquet"
        assert (cfg["folds"], cfg["repeats"]) == (4, 3)

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_cli_defaults(self) -> None:
        args = parse_args([])
        assert (args.family, args.arm) == ("all", "all")
        assert not args.smoke and not args.dry_run
        assert args.config == DEFAULT_CONFIG_PATH
        assert args.output_dir is None

    def test_cli_flags(self, tmp_path: Path) -> None:
        args = parse_args(
            ["--family", "extratrees", "--arm", "mixup", "--smoke", "--dry-run",
             "--config", "x.yaml", "--output-dir", str(tmp_path)]
        )
        assert (args.family, args.arm) == ("extratrees", "mixup")
        assert args.smoke and args.dry_run
        assert args.config == Path("x.yaml") and args.output_dir == tmp_path

    def test_cli_rejects_unknown_choices(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--family", "xgboost"])
        with pytest.raises(SystemExit):
            parse_args(["--arm", "adasyn"])

    def test_undeclared_allowlist_rejected(
        self, config: dict[str, Any], split_manifest: pl.DataFrame,
        campaign_data: pl.DataFrame, tmp_path: Path,
    ) -> None:
        config["features"]["predictors"] = []
        with pytest.raises(ValueError, match="allowlist"):
            run_synthetic_sampling(
                "lightgbm", "natural", config, DEFAULT_CONFIG_PATH, tmp_path,
                dry_run=True, split_df=split_manifest, train_df=campaign_data,
            )

    def test_main_dry_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        config: dict[str, Any], split_manifest: pl.DataFrame, campaign_data: pl.DataFrame,
    ) -> None:
        import scripts.train_synthetic_sampling as script

        real = script.run_synthetic_sampling
        monkeypatch.setattr(script, "load_config", lambda _p: config)
        monkeypatch.setattr(
            script,
            "run_synthetic_sampling",
            lambda **kw: real(**kw, split_df=split_manifest, train_df=campaign_data),
        )
        monkeypatch.setattr(
            sys, "argv",
            ["train_synthetic_sampling.py", "--family", "lightgbm", "--arm", "smote",
             "--dry-run", "--output-dir", str(tmp_path)],
        )
        assert script.main() == 0
        assert len(list(tmp_path.rglob("*_candidate_oof.parquet"))) == 1


# ==============================================================================
# End-to-end schema conformity
# ==============================================================================


class TestOofSchemaConformity:
    def test_all_families_and_arms_conform(
        self, tmp_path: Path, config: dict[str, Any],
        split_manifest: pl.DataFrame, campaign_data: pl.DataFrame,
    ) -> None:
        paths = run_synthetic_sampling(
            "all", "all", config, DEFAULT_CONFIG_PATH, tmp_path,
            smoke=True, dry_run=True, split_df=split_manifest, train_df=campaign_data,
        )
        assert len(paths) == len(SUPPORTED_FAMILIES) * len(SUPPORTED_ARMS)
        assert len(CANDIDATE_OOF_SCHEMA.names) == 24
        seen: set[tuple[str, str]] = set()
        for oof_path in paths:
            table = pq.read_table(oof_path)
            assert table.schema.names == CANDIDATE_OOF_SCHEMA.names
            for field in CANDIDATE_OOF_SCHEMA:
                assert table.schema.field(field.name).type == field.type
                assert table.column(field.name).null_count == 0
            df = pl.read_parquet(oof_path)
            fam, arm = df["model_family"][0], df["arm_id"][0]
            seen.add((fam, arm))
            assert df["candidate_id"][0] == f"{fam}_{arm}"
            assert set(df["score_kind"]) == {"positive_probability"}
            assert set(df["positive_label"]) == {1}
            assert set(df["execution_mode"]) == {"synthetic_smoke"}
            assert set(df["non_comparative"]) == {True}
            assert set(df["n03_eligible"]) == {False}
            assert set(df["experiment_id"]) == {"regime_synthetic_sampling"}
            preds = df["prediction"].to_numpy()
            assert np.all((preds >= 0.0) & (preds <= 1.0))

            # Validation rows are genuine: exactly the held-out source rows, no
            # synthetic row ever appears (folds 0+1 cover all 4 buildings once).
            assert df.height == campaign_data.height
            assert df.select(["repeat", "fold", "row_index"]).is_unique().all()
            assert df["target"].sum() == campaign_data["anomaly"].sum()

            (transport,) = list(oof_path.parent.glob("*_transport_validation.json"))
            manifest = json.loads(transport.read_text())
            assert manifest["schema_version"] == 1
            assert manifest["non_comparative"] is True
            assert manifest["n03_eligible"] is False
            assert manifest["records_count"] == table.num_rows
            assert manifest["oof_parquet"] == str(oof_path)
            assert manifest["validation_synthetic_rows"] == 0
            assert manifest["fold_local_sampling"] is (arm != "natural")
            if arm != "natural":
                assert manifest["sampling_params"]["target_ratio"] == 0.08
        assert seen == {(f, a) for f in SUPPORTED_FAMILIES for a in SUPPORTED_ARMS}


# ==============================================================================
# Dtype-safety regression tests
# ==============================================================================


class TestUndeclaredStringColumnRejected:
    """Undeclared non-numeric columns must raise a clear ValueError, not crash
    cryptically when SMOTE/Mixup/ExtraTrees try to cast them to float64."""

    def _make_frame_with_string(self, n: int = 200, n_pos: int = 4) -> pd.DataFrame:
        """n_pos must be low enough (<<8%) so the sampler actually runs sampling."""
        rng = np.random.default_rng(99)
        labels = np.zeros(n, dtype=np.int8)
        labels[:n_pos] = 1
        return pd.DataFrame(
            {
                "meter_reading": rng.random(n).astype("float32"),
                "air_temperature": rng.random(n).astype("float32"),
                # undeclared string column mimicking building_hour / building_meter
                "building_hour": np.array([f"{b}-0-0" for b in rng.integers(1, 10, n)]),
                "primary_use": np.array(USES)[rng.integers(0, 3, n)],
                "anomaly": labels,
            }
        )

    def test_smote_rejects_undeclared_string_col(self) -> None:
        df = self._make_frame_with_string()
        X = df.drop(columns=["anomaly"])
        y = df["anomaly"].to_numpy()
        with pytest.raises(ValueError, match="non-numeric"):
            apply_fold_local_smote(X, y, categorical_cols=["primary_use"])

    def test_mixup_rejects_undeclared_string_col(self) -> None:
        df = self._make_frame_with_string()
        X = df.drop(columns=["anomaly"])
        y = df["anomaly"].to_numpy()
        with pytest.raises(ValueError, match="non-numeric"):
            apply_fold_local_tabular_mixup(X, y, categorical_cols=["primary_use"])

    def test_extratrees_encode_rejects_undeclared_string_col(self) -> None:
        df = self._make_frame_with_string()
        X = df.drop(columns=["anomaly"])
        with pytest.raises(ValueError, match="non-numeric"):
            ss._encode_for_extratrees(X, X, categorical_cols=["primary_use"])


class TestAllowlistNoProhibitedColumns:
    """The config predictor allowlist must contain no forbidden identifiers."""

    def test_no_gte_columns_in_allowlist(self) -> None:
        cfg = load_config(DEFAULT_CONFIG_PATH)
        predictors = cfg["features"]["predictors"]
        bad = [c for c in predictors if c.startswith("gte_")]
        assert bad == [], f"gte_* columns in allowlist: {bad}"

    def test_no_id_timestamp_target_in_allowlist(self) -> None:
        cfg = load_config(DEFAULT_CONFIG_PATH)
        predictors = set(cfg["features"]["predictors"])
        forbidden = {"building_id", "site_id", "meter_id", "timestamp", "anomaly"}
        found = predictors & forbidden
        assert found == set(), f"Forbidden identifiers in allowlist: {found}"

    def test_no_building_proxy_cols_in_allowlist(self) -> None:
        """building_hour and building_meter are large_string proxies; must be gone."""
        cfg = load_config(DEFAULT_CONFIG_PATH)
        predictors = set(cfg["features"]["predictors"])
        assert "building_hour" not in predictors, "building_hour must not be in allowlist"
        assert "building_meter" not in predictors, "building_meter must not be in allowlist"


class TestRealDataDryRunSmoke:
    """Smoke test against actual train_joined.parquet (skipped if absent).

    Uses the config's predictor allowlist; only a small subset of buildings so
    it completes fast. Verifies no crash with the 31-feature list.
    """

    REAL_DATA_PATH = Path("outputs/prepared/train_joined.parquet")
    SUBSET_BUILDINGS = 3

    @pytest.fixture
    def real_data_subset(self) -> pl.DataFrame | None:
        if not self.REAL_DATA_PATH.exists():
            return None
        df = pl.read_parquet(self.REAL_DATA_PATH)
        bids = df["building_id"].unique().sort()[:self.SUBSET_BUILDINGS]
        return df.filter(pl.col("building_id").is_in(bids))

    @pytest.mark.parametrize("arm", ["smote", "mixup"])
    def test_dry_run_no_crash_with_config_allowlist(
        self,
        arm: str,
        real_data_subset: pl.DataFrame | None,
        tmp_path: Path,
    ) -> None:
        if real_data_subset is None:
            pytest.skip("outputs/prepared/train_joined.parquet not present")
        cfg = load_config(DEFAULT_CONFIG_PATH)
        # Build a minimal split manifest so folds cover the subset buildings.
        buildings = real_data_subset["building_id"].unique().to_list()
        split_rows: list[dict] = []
        for i, b in enumerate(buildings):
            # alternate folds so both train and val folds exist
            split_rows.append(
                {"building_id": b, "fold": i % 2, "repeat": 0, "split_seed": 42, "split": "dev"}
            )
        split_df = pl.DataFrame(split_rows)
        paths = run_synthetic_sampling(
            "lightgbm",
            arm,
            cfg,
            DEFAULT_CONFIG_PATH,
            tmp_path,
            dry_run=True,
            smoke=True,
            split_df=split_df,
            train_df=real_data_subset,
        )
        assert len(paths) == 1
