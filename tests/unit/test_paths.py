from __future__ import annotations

from pathlib import Path

import pytest

from project.config import DEFAULT_SEED
from project.paths import (
    RAW_CSV_FILENAMES,
    SAMPLE_SUBMISSION_CSV,
    TEST_CSV,
    TEST_FEATURES_CSV,
    TRAIN_CSV,
    TRAIN_FEATURES_CSV,
    ProjectPaths,
    find_project_root,
)


def test_constants_match_expected_competition_filenames():
    assert TRAIN_CSV == "train.csv"
    assert TEST_CSV == "test.csv"
    assert SAMPLE_SUBMISSION_CSV == "sample_submission.csv"
    assert TRAIN_FEATURES_CSV == "train_features.csv"
    assert TEST_FEATURES_CSV == "test_features.csv"
    assert RAW_CSV_FILENAMES == (
        "train.csv",
        "test.csv",
        "sample_submission.csv",
        "train_features.csv",
        "test_features.csv",
    )
    assert DEFAULT_SEED == 42


def test_project_paths_from_root_resolves_expected_structure(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    paths = ProjectPaths.from_root(root)

    assert paths.root == root.resolve()
    assert paths.data_dir == root.resolve() / "data"
    assert paths.raw_data_dir == root.resolve() / "data" / "energy-anomaly-detection"
    assert paths.outputs_dir == root.resolve() / "outputs"

    assert paths.train_csv == paths.raw_data_dir / "train.csv"
    assert paths.test_csv == paths.raw_data_dir / "test.csv"
    assert paths.sample_submission_csv == paths.raw_data_dir / "sample_submission.csv"
    assert paths.train_features_csv == paths.raw_data_dir / "train_features.csv"
    assert paths.test_features_csv == paths.raw_data_dir / "test_features.csv"

    # Explicit output directories
    assert paths.lineage_dir == paths.outputs_dir / "lineage"
    assert paths.prepared_dir == paths.outputs_dir / "prepared"
    assert paths.splits_dir == paths.outputs_dir / "splits"
    assert paths.runs_dir == paths.outputs_dir / "runs"
    assert paths.model_cards_dir == paths.outputs_dir / "model_cards"
    assert paths.feature_selection_dir == paths.outputs_dir / "feature_selection"
    assert paths.hpo_dir == paths.outputs_dir / "hpo"
    assert paths.robustness_dir == paths.outputs_dir / "robustness"
    assert paths.ensemble_dir == paths.outputs_dir / "ensemble"
    assert paths.reports_dir == paths.outputs_dir / "reports"
    assert paths.shift_dir == paths.outputs_dir / "shift"
    assert paths.submission_path == paths.outputs_dir / "submission.csv"


def test_project_root_resolver_from_repo_root_and_notebooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = tmp_path / "project_workspace"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("[project]\nname='lead-anomaly'\n")
    notebooks_dir = repo_root / "notebooks"
    notebooks_dir.mkdir()

    # Discover from repo root
    discovered_from_root = find_project_root(repo_root)
    assert discovered_from_root == repo_root.resolve()

    # Discover from notebooks subdirectory
    discovered_from_notebooks = find_project_root(notebooks_dir)
    assert discovered_from_notebooks == repo_root.resolve()

    # Discover via cwd when inside notebooks
    monkeypatch.chdir(notebooks_dir)
    paths_from_cwd = ProjectPaths.resolve()
    assert paths_from_cwd.root == repo_root.resolve()
    assert paths_from_cwd.raw_data_dir == repo_root.resolve() / "data" / "energy-anomaly-detection"
    assert paths_from_cwd.outputs_dir == repo_root.resolve() / "outputs"


def test_find_project_root_raises_when_no_pyproject_found(tmp_path: Path):
    non_project_dir = tmp_path / "isolated_dir"
    non_project_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="pyproject.toml"):
        find_project_root(non_project_dir)



def test_paths_contain_no_hardcoded_user_home():
    synthetic_root = Path("/opt/isolated_build/workspace")
    paths = ProjectPaths.from_root(synthetic_root)

    for attr, val in vars(paths).items():
        if isinstance(val, Path):
            path_str = str(val)
            assert not path_str.startswith("/home/"), f"Path {attr}={path_str} contained /home/"
            assert str(synthetic_root) in path_str, f"Path {attr}={path_str} did not root under {synthetic_root}"


def test_ensure_directories_creates_output_dirs(tmp_path: Path):
    root = tmp_path / "sandbox"
    root.mkdir()
    paths = ProjectPaths.from_root(root)
    assert not paths.outputs_dir.exists()

    expected_output_dirs = [
        paths.outputs_dir,
        paths.lineage_dir,
        paths.prepared_dir,
        paths.splits_dir,
        paths.runs_dir,
        paths.model_cards_dir,
        paths.feature_selection_dir,
        paths.hpo_dir,
        paths.robustness_dir,
        paths.ensemble_dir,
        paths.reports_dir,
        paths.shift_dir,
    ]

    assert set(paths.output_dirs) == set(expected_output_dirs)

    paths.ensure_directories()

    for out_dir in expected_output_dirs:
        assert out_dir.is_dir(), f"Expected directory {out_dir} to exist and be a directory"
