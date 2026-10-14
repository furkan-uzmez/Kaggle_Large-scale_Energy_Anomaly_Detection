"""Task 5 inner-fold guards on generated fixtures; no model fitting or data IO.

Folds are positional inside one supplied training partition. Tests assert the
whole-building invariant, exact once-only validation coverage, determinism,
audit digests and every explicit insufficient-support failure, using real
StratifiedGroupKFold rather than a reimplemented oracle.
"""

import importlib

import numpy as np
import pandas as pd
import pytest

from candidate_fixtures import binary_target, core_frame, synthetic_partitions


def api():
    return importlib.import_module("project.experiments.inner_cv")


def blocked():
    return importlib.import_module("project.experiments.candidate_contracts").BlockedExperimentError


def series(values, index=None, name="building_id"):
    return pd.Series(values, index=index, name=name)


def crafted():
    """Real sklearn leaves fold 0 validation one-class here despite group support."""
    groups = series([f"g{i}" for i in (0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4)])
    target = pd.Series(np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0], dtype=np.int64), name="anomaly")
    return target, groups


def test_inner_folds_exclude_whole_buildings():
    frame = core_frame()
    groups = frame.attrs["groups"]
    folds = api().build_inner_folds(binary_target(frame), groups, n_splits=2)
    for train, valid in folds:
        assert not set(groups.iloc[train]) & set(groups.iloc[valid])
    assert sorted(i for _, valid in folds for i in valid) == list(range(len(frame)))


@pytest.mark.parametrize("n_splits", [2, 3, 4])
def test_every_row_is_validated_exactly_once_without_group_overlap(n_splits):
    frame = core_frame()
    groups = frame.attrs["groups"]
    folds = api().build_inner_folds(binary_target(frame), groups, n_splits=n_splits)
    assert len(folds) == n_splits
    covered = [index for _, valid in folds for index in valid.tolist()]
    assert sorted(covered) == list(range(len(frame)))
    assert len(covered) == len(set(covered))
    for train, valid in folds:
        assert set(train.tolist()) | set(valid.tolist()) == set(range(len(frame)))
        assert not set(train.tolist()) & set(valid.tolist())
        assert not set(groups.iloc[train]) & set(groups.iloc[valid])


def test_folds_are_positional_inside_a_restricted_training_partition():
    partition = synthetic_partitions()[0]
    target, groups = partition.y_train, partition.groups_train
    assert list(groups.index) == [2, 3, 6, 7, 10, 11, 14, 15]
    folds = api().build_inner_folds(target, groups, n_splits=2)
    flat = sorted(index for _, valid in folds for index in valid.tolist())
    assert flat == list(range(len(groups)))
    for train, valid in folds:
        assert train.dtype.kind in "iu" and valid.dtype.kind in "iu"
        assert not set(groups.iloc[train]) & set(groups.iloc[valid])
        assert set(groups.iloc[train]) | set(groups.iloc[valid]) == set(groups)


def test_returned_arrays_are_immutable_and_preserve_input_row_order():
    frame = core_frame()
    groups = frame.attrs["groups"]
    before = groups.tolist()
    folds = api().build_inner_folds(binary_target(frame), groups, n_splits=2)
    assert isinstance(folds, tuple)
    for fold in folds:
        assert isinstance(fold, tuple) and len(fold) == 2
        for side in fold:
            assert isinstance(side, np.ndarray)
            assert side.tolist() == sorted(side.tolist())
            with pytest.raises(ValueError):
                side[0] = 999
    assert groups.tolist() == before


def test_both_classes_are_present_in_every_fold_side():
    frame = core_frame()
    target, groups = binary_target(frame), frame.attrs["groups"]
    for n_splits in (2, 3):
        for train, valid in api().build_inner_folds(target, groups, n_splits=n_splits):
            assert set(target.iloc[train]) == {0, 1}
            assert set(target.iloc[valid]) == {0, 1}


def test_repeated_calls_and_equal_seeds_are_identical():
    frame = core_frame()
    target, groups = binary_target(frame), frame.attrs["groups"]
    first = api().build_inner_folds(target, groups, n_splits=2, seed=42)
    second = api().build_inner_folds(target, groups, n_splits=2, seed=42)
    assert [(t.tolist(), v.tolist()) for t, v in first] == [(t.tolist(), v.tolist()) for t, v in second]


def test_distinct_seeds_change_group_membership():
    frame = core_frame()
    target, groups = binary_target(frame), frame.attrs["groups"]
    memberships = {
        seed: tuple(tuple(sorted(set(groups.iloc[valid]))) for _, valid in
                    api().build_inner_folds(target, groups, n_splits=2, seed=seed))
        for seed in (42, 7)
    }
    assert memberships[42] != memberships[7]


def test_digest_is_opaque_stable_and_sensitive():
    frame = core_frame()
    target, groups = binary_target(frame), frame.attrs["groups"]
    folds = api().build_inner_folds(target, groups, n_splits=2, seed=42)
    other = api().build_inner_folds(target, groups, n_splits=2, seed=7)
    digest = api().inner_membership_digest(folds, groups)
    assert isinstance(digest, str) and len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
    assert digest == api().inner_membership_digest(
        api().build_inner_folds(target, groups, n_splits=2, seed=42), groups)
    assert digest != api().inner_membership_digest(other, groups)
    assert digest != api().inner_membership_digest(
        api().build_inner_folds(target, groups, n_splits=3, seed=42), groups)
    for group in set(groups):
        assert group not in digest


def test_digest_tracks_group_identity_not_only_positions():
    frame = core_frame()
    target, groups = binary_target(frame), frame.attrs["groups"]
    folds = api().build_inner_folds(target, groups, n_splits=2, seed=42)
    renamed = series([f"site_{value}" for value in groups], index=groups.index)
    assert api().inner_membership_digest(folds, groups) != api().inner_membership_digest(folds, renamed)


def test_digest_rejects_folds_that_do_not_match_supplied_groups():
    frame = core_frame()
    target, groups = binary_target(frame), frame.attrs["groups"]
    folds = api().build_inner_folds(target, groups, n_splits=2, seed=42)
    with pytest.raises(ValueError, match="group"):
        api().inner_membership_digest(folds, groups.iloc[:8])
    with pytest.raises(ValueError, match="fold"):
        api().inner_membership_digest((), groups)


def test_post_split_one_class_fold_blocks_without_seed_retry():
    target, groups = crafted()
    with pytest.raises(blocked()) as error:
        api().build_inner_folds(target, groups, n_splits=3)
    assert error.value.code == "infeasible_inner_folds"
    assert api().build_inner_folds(target, groups, n_splits=2)


class DriftingSplitter:
    """Stand-in for a future/changed splitter that violates the fold contract.

    The verification guards must fail closed rather than trust the collaborator,
    so this substitutes the splitter boundary only; group/class support, the
    real target and the real group metadata stay untouched.
    """

    def __init__(self, folds, **kwargs):
        self._folds = folds
        self.kwargs = kwargs

    def split(self, X, y, groups):
        for train, valid in self._folds:
            yield np.array(train, dtype=np.int64), np.array(valid, dtype=np.int64)


def test_group_leaking_splitter_output_is_rejected(monkeypatch):
    frame = core_frame()
    target, groups = binary_target(frame), frame.attrs["groups"]
    leaky = ((list(range(9)), list(range(8, 16))), (list(range(8, 16)), list(range(8))))
    monkeypatch.setattr(api(), "StratifiedGroupKFold",
                        lambda **kwargs: DriftingSplitter(leaky, **kwargs))
    with pytest.raises(blocked()) as error:
        api().build_inner_folds(target, groups, n_splits=2)
    assert error.value.code == "infeasible_inner_folds"


def test_incomplete_validation_coverage_from_splitter_is_rejected(monkeypatch):
    frame = core_frame()
    target, groups = binary_target(frame), frame.attrs["groups"]
    short = ((list(range(8, 16)), list(range(6))), (list(range(8)), list(range(8, 16))))
    monkeypatch.setattr(api(), "StratifiedGroupKFold",
                        lambda **kwargs: DriftingSplitter(short, **kwargs))
    with pytest.raises(blocked()) as error:
        api().build_inner_folds(target, groups, n_splits=2)
    assert error.value.code == "infeasible_inner_folds"


def test_duplicated_validation_coverage_from_splitter_is_rejected(monkeypatch):
    frame = core_frame()
    target, groups = binary_target(frame), frame.attrs["groups"]
    repeated = ((list(range(8, 16)), list(range(8))), (list(range(8, 16)), list(range(8))))
    monkeypatch.setattr(api(), "StratifiedGroupKFold",
                        lambda **kwargs: DriftingSplitter(repeated, **kwargs))
    with pytest.raises(blocked(), match="exactly once"):
        api().build_inner_folds(target, groups, n_splits=2)


def test_fold_that_drops_training_rows_is_rejected(monkeypatch):
    frame = core_frame()
    target, groups = binary_target(frame), frame.attrs["groups"]
    partial = ((list(range(8, 16)), list(range(8))), (list(range(6)), list(range(8, 16))))
    monkeypatch.setattr(api(), "StratifiedGroupKFold",
                        lambda **kwargs: DriftingSplitter(partial, **kwargs))
    with pytest.raises(blocked(), match="every training row"):
        api().build_inner_folds(target, groups, n_splits=2)


def test_splitter_returning_too_few_folds_is_rejected(monkeypatch):
    frame = core_frame()
    target, groups = binary_target(frame), frame.attrs["groups"]
    two = ((list(range(8, 16)), list(range(8))), (list(range(8)), list(range(8, 16))))
    monkeypatch.setattr(api(), "StratifiedGroupKFold",
                        lambda **kwargs: DriftingSplitter(two, **kwargs))
    with pytest.raises(blocked(), match="of 3 inner folds"):
        api().build_inner_folds(target, groups, n_splits=3)


def test_unsorted_group_order_is_not_reordered_to_fit_a_split():
    frame = core_frame()
    order = [14, 15, 0, 1, 8, 9, 4, 5, 12, 13, 2, 3, 10, 11, 6, 7]
    shuffled = frame.loc[order]
    target, groups = binary_target(shuffled), shuffled.attrs["groups"].loc[order]
    assert groups.tolist() != sorted(groups.tolist())
    folds = api().build_inner_folds(target, groups, n_splits=2)
    for train, valid in folds:
        assert not set(groups.iloc[train]) & set(groups.iloc[valid])
        for indices in (train, valid):
            for position in indices.tolist():
                assert groups.iloc[position] == f"building_{order[position] // 2:02d}"
    assert sorted(i for _, valid in folds for i in valid.tolist()) == list(range(16))


def test_interleaved_groups_are_isolated_without_reordering_rows():
    """Non-contiguous group rows leak every group if the splitter sees sorted IDs."""
    groups = series(["a", "b", "c", "d"] * 3)
    target = pd.Series(np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1], dtype=np.int64))
    folds = api().build_inner_folds(target, groups, n_splits=2)
    for train, valid in folds:
        assert not set(groups.iloc[train]) & set(groups.iloc[valid])
    assert sorted(i for _, valid in folds for i in valid.tolist()) == list(range(12))


def test_fewer_groups_than_folds_blocks():
    groups = series(["a", "a", "b", "b"])
    target = pd.Series(np.array([0, 1, 0, 1], dtype=np.int64))
    with pytest.raises(blocked()) as error:
        api().build_inner_folds(target, groups, n_splits=3)
    assert error.value.code == "insufficient_inner_groups"


def test_single_class_group_support_below_folds_blocks():
    groups = series(["a", "a", "b", "b", "c", "c", "d", "d"])
    target = pd.Series(np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64))
    with pytest.raises(blocked()) as error:
        api().build_inner_folds(target, groups, n_splits=3)
    assert error.value.code == "insufficient_inner_groups"


def test_one_class_partition_blocks():
    groups = series(["a", "a", "b", "b", "c", "c"])
    target = pd.Series(np.zeros(6, dtype=np.int64))
    with pytest.raises(blocked()) as error:
        api().build_inner_folds(target, groups, n_splits=2)
    assert error.value.code == "one_class_inner_partition"


def test_misaligned_group_index_fails():
    frame = core_frame()
    target = binary_target(frame)
    groups = series(frame.attrs["groups"].tolist(), index=np.arange(100, 116))
    with pytest.raises(ValueError, match="index"):
        api().build_inner_folds(target, groups, n_splits=2)


def test_group_length_mismatch_fails():
    frame = core_frame()
    with pytest.raises(ValueError, match="index"):
        api().build_inner_folds(binary_target(frame), frame.attrs["groups"].iloc[:8], n_splits=2)


@pytest.mark.parametrize("bad", [None, np.nan, pd.NA])
def test_null_groups_fail(bad):
    frame = core_frame()
    groups = frame.attrs["groups"].astype(object)
    groups.iloc[3] = bad
    with pytest.raises(ValueError, match="group"):
        api().build_inner_folds(binary_target(frame), groups, n_splits=2)


def test_mixed_type_groups_fail():
    frame = core_frame()
    groups = frame.attrs["groups"].astype(object)
    groups.iloc[0] = 7
    with pytest.raises(ValueError, match="group"):
        api().build_inner_folds(binary_target(frame), groups, n_splits=2)


def test_integer_group_identities_are_supported():
    frame = core_frame()
    groups = series(np.repeat(np.arange(8), 2), index=frame.index)
    folds = api().build_inner_folds(binary_target(frame), groups, n_splits=2)
    assert sorted(i for _, valid in folds for i in valid.tolist()) == list(range(16))
    assert len(api().inner_membership_digest(folds, groups)) == 64


@pytest.mark.parametrize("n_splits", [1, 0, -1, 6, True, False, 2.0, "3", None])
def test_invalid_n_splits_fails(n_splits):
    frame = core_frame()
    with pytest.raises((ValueError, TypeError), match="n_splits"):
        api().build_inner_folds(binary_target(frame), frame.attrs["groups"], n_splits=n_splits)


@pytest.mark.parametrize("seed", [True, -1, 2.5, "42", None])
def test_invalid_seed_fails(seed):
    frame = core_frame()
    with pytest.raises((ValueError, TypeError), match="seed"):
        api().build_inner_folds(binary_target(frame), frame.attrs["groups"], n_splits=2, seed=seed)


@pytest.mark.parametrize("values", [
    np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64),
    np.array([0.0, 1.0] * 6),
    np.array([False, True] * 6),
])
def test_non_binary_integer_target_fails(values):
    groups = series([f"g{i // 2}" for i in range(12)])
    with pytest.raises(ValueError, match="target"):
        api().build_inner_folds(pd.Series(values), groups, n_splits=2)


def test_target_must_be_a_series_aligned_to_groups():
    frame = core_frame()
    with pytest.raises((ValueError, TypeError), match="target"):
        api().build_inner_folds(binary_target(frame).to_numpy(), frame.attrs["groups"], n_splits=2)


def test_null_target_fails():
    groups = series([f"g{i // 2}" for i in range(8)])
    target = pd.Series([0, 1, 0, 1, 0, 1, 0, None], dtype="Int64")
    with pytest.raises(ValueError, match="target"):
        api().build_inner_folds(target, groups, n_splits=2)


@pytest.mark.parametrize("change", ["group_type", "index_type", "index_order", "binding", "positions", "fold_order"])
def test_digest_binds_typed_ordered_rows_groups_and_fold_positions(change):
    groups = series([7, 7, 8, 8, 9, 9, 10, 10])
    folds = ((np.arange(4, 8), np.arange(4)), (np.arange(4), np.arange(4, 8)))
    before = api().inner_membership_digest(folds, groups)
    if change == "group_type":
        groups = groups.astype(str)
    elif change == "index_type":
        groups.index = groups.index.astype(str)
    elif change == "index_order":
        groups.index = groups.index[::-1]
    elif change == "binding":
        groups = groups.iloc[[2, 3, 0, 1, 4, 5, 6, 7]].set_axis(groups.index)
    elif change == "positions":
        folds = tuple((train[::-1], valid[::-1]) for train, valid in folds)
    else:
        folds = folds[::-1]
    assert before != api().inner_membership_digest(folds, groups)


def test_digest_string_serialization_cannot_collide_at_delimiters():
    folds = ((np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3])))
    first = series(["a", "b\x1fc", "d", "e"])
    second = series(["a\x1fb", "c", "d", "e"])
    assert api().inner_membership_digest(folds, first) != api().inner_membership_digest(folds, second)


@pytest.mark.parametrize("index", [[0, 0, 2, 3], [0, 1, None, 3]])
def test_standalone_fold_and_digest_apis_reject_invalid_row_identity(index):
    groups = series(["a", "a", "b", "b"], index=index)
    target = pd.Series([0, 1, 0, 1], index=index, dtype="int64")
    folds = ((np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3])))
    with pytest.raises(ValueError, match="index"):
        api().build_inner_folds(target, groups, n_splits=2)
    with pytest.raises(ValueError, match="index"):
        api().inner_membership_digest(folds, groups)


@pytest.mark.parametrize("values", [[True] * 4, [1.0] * 4, ["a", "a", 7, 7], ["", "", "b", "b"]])
def test_direct_digest_rejects_invalid_group_domain(values):
    folds = ((np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3])))
    with pytest.raises(ValueError, match="group"):
        api().inner_membership_digest(folds, series(values))


@pytest.mark.parametrize("bad", [
    (([2, 2, 3], [0, 1]), ([0, 1], [2, 3])),
    (([1, 2, 3], [0, 1]), ([0, 1], [2, 3])),
    (([2, 3], [0]), ([0, 1], [2, 3])),
    (([2, 3], [0, 1]), ([2, 3], [0, 1])),
    (([1, 3], [0, 2]), ([0, 2], [1, 3])),
])
def test_direct_digest_rejects_invalid_fold_partitions(bad):
    folds = tuple((np.array(train), np.array(valid)) for train, valid in bad)
    with pytest.raises(ValueError, match="fold"):
        api().inner_membership_digest(folds, series(["a", "a", "b", "b"]))


@pytest.mark.parametrize("operation", ["validate", "split", "digest"])
def test_repeated_level_names_do_not_invalidate_compound_row_identity(operation):
    index = pd.MultiIndex.from_tuples(
        [("a", 0), ("a", 1), ("b", 0), ("b", 1)], names=["identity", "identity"]
    )
    target = pd.Series([0, 1, 0, 1], index=index, dtype="int64")
    groups = series([0, 0, 1, 1], index=index)
    if operation == "validate":
        api().validate_inner_inputs(target, groups)
    elif operation == "split":
        folds = api().build_inner_folds(target, groups, n_splits=2)
        assert sorted(position for _, valid in folds for position in valid) == list(range(4))
    else:
        folds = ((np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3])))
        assert len(api().inner_membership_digest(folds, groups)) == 64


@pytest.mark.parametrize("tuples", [
    [("a", 0), ("a", 0), ("b", 0), ("b", 1)],
    [("a", 0), ("a", None), ("b", 0), ("b", 1)],
])
def test_repeated_level_names_still_reject_duplicate_rows_and_null_components(tuples):
    index = pd.MultiIndex.from_tuples(tuples, names=["identity", "identity"])
    target = pd.Series([0, 1, 0, 1], index=index, dtype="int64")
    groups = series([0, 0, 1, 1], index=index)
    folds = ((np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3])))
    for operation in (
        lambda: api().validate_inner_inputs(target, groups),
        lambda: api().build_inner_folds(target, groups, n_splits=2),
        lambda: api().inner_membership_digest(folds, groups),
    ):
        with pytest.raises(ValueError, match="index"):
            operation()
