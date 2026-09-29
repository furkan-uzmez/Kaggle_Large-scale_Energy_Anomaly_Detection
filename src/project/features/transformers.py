"""Train-fold-fitted tabular and causal grouped-time transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder


class TrainFoldFeatureMask(BaseEstimator, TransformerMixin):
    """Drop true constants fitted on training data and flag quasi-constants."""

    def __init__(self, quasi_constant_threshold: float = 0.95) -> None:
        self.quasi_constant_threshold = quasi_constant_threshold

    def fit(self, X: pd.DataFrame, y: Any = None) -> TrainFoldFeatureMask:
        frame = _validated_frame(X)
        if frame.empty:
            raise ValueError("fit data must contain at least one row")
        if not 0 < self.quasi_constant_threshold <= 1:
            raise ValueError("quasi_constant_threshold must be in (0, 1]")

        self.feature_names_in_ = tuple(frame.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        self.constant_features_ = tuple(
            column
            for column in self.feature_names_in_
            if frame[column].nunique(dropna=False) <= 1
        )

        constant_set = set(self.constant_features_)
        quasi_constant_features: list[str] = []
        for column in self.feature_names_in_:
            if column in constant_set:
                continue
            frequencies = frame[column].value_counts(dropna=False, normalize=True)
            if not frequencies.empty and frequencies.iloc[0] >= self.quasi_constant_threshold:
                quasi_constant_features.append(column)
        self.quasi_constant_features_ = tuple(quasi_constant_features)
        self.output_features_ = tuple(
            column for column in self.feature_names_in_ if column not in constant_set
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _validated_frame(X)
        _validate_schema(frame, self.feature_names_in_)
        return frame.loc[:, self.output_features_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        _ensure_fitted(self)
        if input_features is not None and tuple(input_features) != self.feature_names_in_:
            raise ValueError("input_features must match the schema seen during fit")
        return np.asarray(self.output_features_, dtype=object)


class _TrainFitImputationPreprocessor(BaseEstimator, TransformerMixin):
    """Fit numeric/categorical imputation and encoding on one training fold."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> _TrainFitImputationPreprocessor:
        frame = _validated_frame(X)
        if frame.empty:
            raise ValueError("fit data must contain at least one row")
        if frame.shape[1] == 0:
            raise ValueError("fit data must contain at least one feature")

        self.feature_names_in_ = tuple(frame.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        numeric_columns = tuple(
            column for column in self.feature_names_in_ if is_numeric_dtype(frame[column].dtype)
        )
        categorical_columns = tuple(
            column for column in self.feature_names_in_ if column not in numeric_columns
        )

        transformers: list[tuple[str, Any, tuple[str, ...]]] = []
        if numeric_columns:
            numeric_pipeline = FeatureUnion(
                [
                    (
                        "imputed",
                        SimpleImputer(strategy="median", keep_empty_features=True),
                    ),
                    (
                        "missing_indicator",
                        MissingIndicator(features="all", error_on_new=False),
                    ),
                ]
            )
            transformers.append(("numeric", numeric_pipeline, numeric_columns))
        if categorical_columns:
            categorical_pipeline = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                    ),
                    (
                        "one_hot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                    ),
                ]
            )
            transformers.append(("categorical", categorical_pipeline, categorical_columns))

        self.transformer_ = ColumnTransformer(
            transformers,
            remainder="drop",
            verbose_feature_names_out=True,
        )
        self.transformer_.fit(frame)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        _ensure_fitted(self)
        frame = _validated_frame(X)
        _validate_schema(frame, self.feature_names_in_)
        return self.transformer_.transform(frame)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        _ensure_fitted(self)
        if input_features is not None and tuple(input_features) != self.feature_names_in_:
            raise ValueError("input_features must match the schema seen during fit")
        return self.transformer_.get_feature_names_out()


@dataclass(frozen=True)
class GroupedTimeFeatureSpec:
    """Configuration for grouped lag and trailing rolling-mean features."""

    group_columns: tuple[str, ...]
    timestamp_column: str
    value_column: str
    lags: tuple[int, ...] = (1,)
    rolling_windows: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.group_columns or any(not column.strip() for column in self.group_columns):
            raise ValueError("group_columns must contain non-empty column names")
        if not self.timestamp_column.strip() or not self.value_column.strip():
            raise ValueError("timestamp_column and value_column must be non-empty")
        if not self.lags and not self.rolling_windows:
            raise ValueError("at least one lag or rolling window is required")
        if any(not isinstance(lag, int) or isinstance(lag, bool) or lag < 1 for lag in self.lags):
            raise ValueError("lags must be positive integers")
        if any(
            not isinstance(window, int) or isinstance(window, bool) or window < 1
            for window in self.rolling_windows
        ):
            raise ValueError("rolling_windows must be positive integers")
        if len(set(self.group_columns)) != len(self.group_columns):
            raise ValueError("group_columns must not contain duplicates")
        if len(set(self.lags)) != len(self.lags) or len(set(self.rolling_windows)) != len(
            self.rolling_windows
        ):
            raise ValueError("lags and rolling_windows must not contain duplicates")


def make_native_nan_preprocessor(
    quasi_constant_threshold: float = 0.95,
) -> TrainFoldFeatureMask:
    """Create a fold-fitted constant mask that leaves native missing values intact."""

    return TrainFoldFeatureMask(quasi_constant_threshold=quasi_constant_threshold)


def make_imputation_preprocessor() -> _TrainFitImputationPreprocessor:
    """Create train-fit numeric/categorical imputers and a train-fit one-hot encoder."""

    return _TrainFitImputationPreprocessor()


def add_grouped_shift_rolling_features(
    frame: pd.DataFrame,
    specification: GroupedTimeFeatureSpec,
) -> pd.DataFrame:
    """Add per-group lag and trailing-mean features without reordering source rows."""

    source = _validated_frame(frame)
    required_columns = (
        *specification.group_columns,
        specification.timestamp_column,
        specification.value_column,
    )
    missing_columns = tuple(column for column in required_columns if column not in source.columns)
    if missing_columns:
        raise ValueError(f"required time-feature columns are missing: {missing_columns}")
    if source[specification.timestamp_column].isna().any():
        raise ValueError("timestamp_column must not contain missing values")
    key_columns = [*specification.group_columns, specification.timestamp_column]
    if source.duplicated(subset=key_columns).any():
        raise ValueError("duplicate group/timestamp keys are not allowed")
    if not is_numeric_dtype(source[specification.value_column].dtype):
        raise ValueError("value_column must be numeric")

    output_names = [
        *(f"{specification.value_column}_lag_{lag}" for lag in specification.lags),
        *(
            f"{specification.value_column}_rolling_mean_{window}"
            for window in specification.rolling_windows
        ),
    ]
    collisions = tuple(name for name in output_names if name in source.columns)
    if collisions:
        raise ValueError(f"generated feature columns already exist: {collisions}")

    position_column = "__project_original_row_order__"
    while position_column in source.columns:
        position_column = f"_{position_column}"

    ordered = source.copy()
    ordered[position_column] = np.arange(len(ordered))
    ordered = ordered.sort_values(key_columns, kind="mergesort", na_position="last")
    grouped_values = ordered.groupby(
        list(specification.group_columns), sort=False, dropna=False
    )[specification.value_column]

    for lag in specification.lags:
        feature_name = f"{specification.value_column}_lag_{lag}"
        ordered[feature_name] = grouped_values.shift(lag).to_numpy()

    for window in specification.rolling_windows:
        feature_name = f"{specification.value_column}_rolling_mean_{window}"
        rolling_mean = grouped_values.transform(
            lambda values, window=window: values.shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )
        ordered[feature_name] = rolling_mean.to_numpy()

    ordered = ordered.sort_values(position_column, kind="mergesort")
    return ordered.drop(columns=position_column)


def _validated_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("input must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError("input DataFrame columns must be unique")
    return frame


def _validate_schema(frame: pd.DataFrame, fitted_columns: tuple[str, ...]) -> None:
    if tuple(frame.columns) != fitted_columns:
        raise ValueError("inference schema columns must match the schema seen during fit")


def _ensure_fitted(transformer: Any) -> None:
    if not hasattr(transformer, "feature_names_in_"):
        raise ValueError("transformer must be fitted before use")
