"""Fail-closed feature joins preserving row count and test submission order lazily."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from project.data.validate import (
    validate_building_disjointness,
    validate_duplicate_column_equality,
    validate_keys_uniqueness,
    validate_sample_submission_alignment,
    validate_target_presence,
)

DEFAULT_TRAIN_DUPLICATE_COPIES: Sequence[str] = ("meter_reading", "anomaly")
DEFAULT_TEST_DUPLICATE_COPIES: Sequence[str] = ("building_id", "timestamp", "meter_reading")


def build_feature_joins(
    tables: Mapping[str, pl.DataFrame | pl.LazyFrame],
    train_duplicate_copies: Sequence[str] = DEFAULT_TRAIN_DUPLICATE_COPIES,
    test_duplicate_copies: Sequence[str] = DEFAULT_TEST_DUPLICATE_COPIES,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Build one-to-one feature joins for train and test tables preserving Polars LazyFrames.

    Contract:
      - Train joins on ('building_id', 'timestamp').
      - Test joins on 'row_id' with maintain_order='left'.
      - Asserts exact key uniqueness and exact key-set equality before joining.
      - Asserts target presence in train and absence in test.
      - Asserts train and test buildings are disjoint.
      - Fails closed if any declared duplicate feature copy is missing or differs.
      - Drops redundant duplicate copies from feature tables.
      - Preserves test and sample-submission row order.
      - Returns (train_joined_lf, test_joined_lf) as Polars LazyFrames for streaming sinks.
    """
    required_tables = ["train", "test", "sample_submission", "train_features", "test_features"]
    for req in required_tables:
        if req not in tables:
            raise KeyError(f"Missing required table '{req}' in tables mapping")

    # Ensure lazy representations
    train_lf = tables["train"].lazy() if isinstance(tables["train"], pl.DataFrame) else tables["train"]
    test_lf = tables["test"].lazy() if isinstance(tables["test"], pl.DataFrame) else tables["test"]
    sample_sub_lf = (
        tables["sample_submission"].lazy()
        if isinstance(tables["sample_submission"], pl.DataFrame)
        else tables["sample_submission"]
    )
    train_feat_lf = (
        tables["train_features"].lazy()
        if isinstance(tables["train_features"], pl.DataFrame)
        else tables["train_features"]
    )
    test_feat_lf = (
        tables["test_features"].lazy()
        if isinstance(tables["test_features"], pl.DataFrame)
        else tables["test_features"]
    )

    # 1. Target presence checks
    validate_target_presence(train_lf, target_col="anomaly", should_be_present=True, table_name="train")
    validate_target_presence(test_lf, target_col="anomaly", should_be_present=False, table_name="test")
    validate_target_presence(test_feat_lf, target_col="anomaly", should_be_present=False, table_name="test_features")

    # 2. Key uniqueness checks
    train_keys = ["building_id", "timestamp"]
    validate_keys_uniqueness(train_lf, train_keys, "train")
    validate_keys_uniqueness(train_feat_lf, train_keys, "train_features")

    test_keys = ["row_id"]
    validate_keys_uniqueness(test_lf, test_keys, "test")
    validate_keys_uniqueness(test_feat_lf, test_keys, "test_features")
    validate_keys_uniqueness(sample_sub_lf, test_keys, "sample_submission")

    # 3. Key set equality checks
    train_count = train_lf.select(pl.len()).collect(engine="streaming").item()
    train_feat_count = train_feat_lf.select(pl.len()).collect(engine="streaming").item()
    if train_count != train_feat_count:
        raise ValueError(
            f"Key set mismatch between train and train_features (row counts differ: {train_count} vs {train_feat_count})"
        )

    train_missing = (
        train_lf.select(train_keys)
        .join(train_feat_lf.select(train_keys), on=train_keys, how="anti")
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if train_missing > 0:
        raise ValueError(
            f"Key set mismatch between train and train_features ({train_missing} missing keys)"
        )

    test_count = test_lf.select(pl.len()).collect(engine="streaming").item()
    test_feat_count = test_feat_lf.select(pl.len()).collect(engine="streaming").item()
    if test_count != test_feat_count:
        raise ValueError(
            f"Key set mismatch between test and test_features (row counts differ: {test_count} vs {test_feat_count})"
        )

    test_missing = (
        test_lf.select(test_keys)
        .join(test_feat_lf.select(test_keys), on=test_keys, how="anti")
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if test_missing > 0:
        raise ValueError(
            f"Key set mismatch between test and test_features ({test_missing} missing keys)"
        )

    # 4. Building disjointness
    validate_building_disjointness(train_lf, test_lf)

    # 5. Fail closed if declared duplicate feature copies are missing or differ
    train_feat_schema = train_feat_lf.collect_schema()
    for col in train_duplicate_copies:
        if col not in train_feat_schema:
            raise ValueError(
                f"train_features missing declared duplicate copy column '{col}'"
            )
        validate_duplicate_column_equality(
            train_lf,
            train_feat_lf,
            col_name=col,
            join_keys=train_keys,
            name1="train",
            name2="train_features",
        )

    test_feat_schema = test_feat_lf.collect_schema()
    for col in test_duplicate_copies:
        if col not in test_feat_schema:
            raise ValueError(
                f"test_features missing declared duplicate copy column '{col}'"
            )
        validate_duplicate_column_equality(
            test_lf,
            test_feat_lf,
            col_name=col,
            join_keys=test_keys,
            name1="test",
            name2="test_features",
        )

    # Drop redundant duplicate columns from feature tables before joining
    train_feat_clean = train_feat_lf.drop([c for c in train_duplicate_copies if c in train_feat_schema])
    test_feat_clean = test_feat_lf.drop([c for c in test_duplicate_copies if c in test_feat_schema])

    # 6. Build lazy joins
    train_joined = train_lf.join(train_feat_clean, on=train_keys, how="inner")
    test_joined = test_lf.join(test_feat_clean, on=test_keys, how="inner", maintain_order="left")

    # 7. Sample submission order alignment
    validate_sample_submission_alignment(test_joined, sample_sub_lf, row_id_col="row_id")

    return train_joined, test_joined
