"""Deterministic group-isolated inner folds inside one supplied training partition.

Folds are built only from training rows/labels/groups that a caller already
restricted; this module never reads data, generates outer splits, or fits a
model. Returned indices are positional within the supplied partition and
preserve the caller's original row order, so group identities stay out of the
feature space and no row is reordered to make a split feasible.

Insufficient support fails closed with an explicit blocker. Seeds are used
exactly as supplied: no retry loop, no score-dependent seed selection and no
row-wise fallback that would leak a building across a fold boundary.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype
from sklearn.model_selection import StratifiedGroupKFold

from project.experiments.candidate_contracts import BlockedExperimentError

MIN_SPLITS = 2
MAX_SPLITS = 5
MAX_SEED = 2**32 - 1
InnerFolds = tuple[tuple[np.ndarray, np.ndarray], ...]


def _n_splits(n_splits: object) -> int:
    if type(n_splits) is not int:
        raise TypeError(f"n_splits must be an integer in [{MIN_SPLITS}, {MAX_SPLITS}]; booleans are invalid")
    if not MIN_SPLITS <= n_splits <= MAX_SPLITS:
        raise ValueError(f"n_splits must be an integer in [{MIN_SPLITS}, {MAX_SPLITS}]")
    return n_splits


def _seed(seed: object) -> int:
    if type(seed) is not int:
        raise TypeError(f"seed must be an integer in [0, {MAX_SEED}]; booleans are invalid")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be an integer in [0, {MAX_SEED}]")
    return seed


def _target(y: object) -> pd.Series:
    if not isinstance(y, pd.Series):
        raise TypeError("inner target must be a pandas Series aligned to the supplied groups")
    if len(y) == 0:
        raise ValueError("inner target requires nonempty training rows")
    _row_identity(y.index)
    if not is_integer_dtype(y.dtype) or is_bool_dtype(y.dtype):
        raise ValueError("inner target must use a nonboolean integer dtype")
    if y.isna().any():
        raise ValueError("inner target must not contain null values")
    observed = set(y.tolist())
    if not observed <= {0, 1}:
        raise ValueError("inner target must contain binary class labels 0 and 1 only")
    return y


def _groups(groups: object, y: pd.Series) -> pd.Series:
    if not isinstance(groups, pd.Series):
        raise TypeError("group metadata must be a pandas Series aligned to the inner target")
    if not groups.index.equals(y.index):
        raise ValueError("group metadata index must align exactly with the inner target index")
    _row_identity(groups.index)
    values = groups.tolist()
    if groups.isna().any():
        raise ValueError("group metadata must be nonnull")
    strings = all(isinstance(v, str) and bool(v.strip()) for v in values)
    integers = all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
                   for v in values)
    if not (strings or integers):
        raise ValueError("group metadata must be consistently typed opaque strings or integer IDs")
    return groups


def _row_identity(index: pd.Index) -> None:
    if not index.is_unique or any(
        index.get_level_values(level).isna().any() for level in range(index.nlevels)
    ):
        raise ValueError("inner row index must be unique and nonnull")


def validate_inner_inputs(y: pd.Series, groups: pd.Series) -> None:
    """Validate aligned nonempty binary labels and group identities without splitting."""
    _groups(groups, _target(y))


def _support(y: pd.Series, groups: pd.Series, n_splits: int) -> None:
    classes = set(y.tolist())
    if classes != {0, 1}:
        raise BlockedExperimentError(
            "one_class_inner_partition",
            f"inner partition observed only class {sorted(classes)}; both classes are required",
        )
    unique_groups = len(set(groups.tolist()))
    if unique_groups < n_splits:
        raise BlockedExperimentError(
            "insufficient_inner_groups",
            f"{unique_groups} distinct groups cannot fill {n_splits} group-disjoint inner folds",
        )
    for label in (0, 1):
        supporting = len(set(groups[y == label].tolist()))
        if supporting < n_splits:
            raise BlockedExperimentError(
                "insufficient_inner_groups",
                f"class {label} appears in only {supporting} groups; {n_splits} inner folds "
                "cannot each hold both classes",
            )


def _verify(folds: InnerFolds, y: pd.Series, groups: pd.Series, n_splits: int) -> None:
    rows = len(y)
    if len(folds) != n_splits:
        raise BlockedExperimentError(
            "infeasible_inner_folds", f"splitter produced {len(folds)} of {n_splits} inner folds")
    covered: list[int] = []
    for position, (train, valid) in enumerate(folds):
        if train.size == 0 or valid.size == 0:
            raise BlockedExperimentError(
                "infeasible_inner_folds", f"inner fold {position} has an empty side")
        if set(train.tolist()) | set(valid.tolist()) != set(range(rows)):
            raise BlockedExperimentError(
                "infeasible_inner_folds", f"inner fold {position} does not partition every training row")
        if set(groups.iloc[train]) & set(groups.iloc[valid]):
            raise BlockedExperimentError(
                "infeasible_inner_folds", f"inner fold {position} shares a group across its boundary")
        for side, indices in (("train", train), ("validation", valid)):
            if set(y.iloc[indices]) != {0, 1}:
                raise BlockedExperimentError(
                    "infeasible_inner_folds",
                    f"inner fold {position} {side} side lacks both classes; seeds are not retried",
                )
        covered.extend(valid.tolist())
    if sorted(covered) != list(range(rows)):
        raise BlockedExperimentError(
            "infeasible_inner_folds", "inner validation folds must cover every training row exactly once")


def _frozen(indices: np.ndarray) -> np.ndarray:
    array = np.sort(np.asarray(indices, dtype=np.int64))
    array.setflags(write=False)
    return array


def build_inner_folds(
    y: pd.Series, groups: pd.Series, n_splits: int = 3, seed: int = 42,
) -> InnerFolds:
    """Return group-disjoint positional (train, validation) folds for one partition.

    `y` and `groups` must describe exactly the rows a caller already restricted
    to a training partition; indices are positional inside that partition, not
    labels of the caller's frame. Validation folds cover every supplied row
    exactly once, no group spans a fold boundary, and both classes are present
    on both sides of every fold. Unsatisfiable support raises
    BlockedExperimentError instead of degrading the isolation guarantee.
    """
    n_splits = _n_splits(n_splits)
    seed = _seed(seed)
    y = _target(y)
    groups = _groups(groups, y)
    _support(y, groups, n_splits)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    try:
        raw = splitter.split(np.zeros((len(y), 1)), y.to_numpy(dtype=np.int64),
                             groups.to_numpy(copy=True))
        folds = tuple((_frozen(train), _frozen(valid)) for train, valid in raw)
    except ValueError as error:
        raise BlockedExperimentError(
            "infeasible_inner_folds", f"stratified group splitting failed: {error}") from error
    _verify(folds, y, groups, n_splits)
    return folds


def inner_membership_digest(folds: InnerFolds, groups: pd.Series) -> str:
    """Hash v2 typed ordered row/group bindings and every ordered fold position.

    Validate complete group-disjoint partitions even when called directly.
    Only the SHA-256 digest is returned; raw identities remain in memory.
    """
    if not isinstance(folds, tuple) or not folds:
        raise ValueError("inner folds must be a nonempty tuple of positional index pairs")
    if not all(isinstance(f, tuple) and len(f) == 2 for f in folds):
        raise ValueError("each inner fold must be a (train, validation) index pair")
    if not isinstance(groups, pd.Series) or groups.isna().any() or len(groups) == 0:
        raise ValueError("group metadata must be a nonempty nonnull pandas Series")
    _groups(groups, groups)
    rows = len(groups)
    partitions = []
    covered = []
    for train, valid in folds:
        sides = []
        for indices in (train, valid):
            array = np.asarray(indices)
            if array.ndim != 1 or array.dtype.kind not in "iu" or array.size == 0:
                raise ValueError("inner fold indices must be nonempty one-dimensional integers")
            if array.min() < 0 or array.max() >= rows:
                raise ValueError(
                    "inner fold positions must address the supplied group partition; "
                    "group count disagrees with the fold indices")
            positions = array.tolist()
            if len(set(positions)) != len(positions):
                raise ValueError("inner fold positions must be unique on each side")
            sides.append(positions)
        train_positions, valid_positions = sides
        if (set(train_positions) & set(valid_positions)
                or set(train_positions) | set(valid_positions) != set(range(rows))):
            raise ValueError("inner fold must partition every training row without overlap")
        if set(groups.iloc[train_positions]) & set(groups.iloc[valid_positions]):
            raise ValueError("inner fold shares a group across its boundary")
        covered.extend(valid_positions)
        partitions.append(sides)
    if sorted(covered) != list(range(rows)):
        raise ValueError("inner validation folds must cover every training row exactly once")
    payload = {
        "version": "inner_membership_v2",
        "rows": [_typed_identity(value) for value in groups.index.tolist()],
        "groups": [_typed_identity(value) for value in groups.tolist()],
        "folds": partitions,
    }
    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _typed_identity(value: object) -> list:
    """Encode identity scalars without delimiter or int/string collisions."""
    if isinstance(value, tuple):
        return ["tuple", [_typed_identity(part) for part in value]]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, (bool, np.bool_)):
        return ["bool", bool(value)]
    if isinstance(value, (int, np.integer)):
        return ["int", str(int(value))]
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return ["float", float(value).hex()]
    if isinstance(value, pd.Timestamp):
        return ["timestamp", value.isoformat()]
    if isinstance(value, pd.Timedelta):
        return ["timedelta", value.isoformat()]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    raise ValueError("inner row index identity type is unsupported for typed audit serialization")
