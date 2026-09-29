"""Repository and output path resolution utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Source raw CSV filename constants
TRAIN_CSV: str = "train.csv"
TEST_CSV: str = "test.csv"
SAMPLE_SUBMISSION_CSV: str = "sample_submission.csv"
TRAIN_FEATURES_CSV: str = "train_features.csv"
TEST_FEATURES_CSV: str = "test_features.csv"

RAW_CSV_FILENAMES: Sequence[str] = (
    TRAIN_CSV,
    TEST_CSV,
    SAMPLE_SUBMISSION_CSV,
    TRAIN_FEATURES_CSV,
    TEST_FEATURES_CSV,
)


def find_project_root(start: Path | None = None) -> Path:
    """Find repository root by looking for pyproject.toml upwards from start directory.

    Raises:
        FileNotFoundError: If no pyproject.toml is found in start or any of its parent directories.
    """
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(
        f"Could not locate project root containing 'pyproject.toml' upwards from '{current}'"
    )


@dataclass(frozen=True)
class ProjectPaths:
    """Strongly-typed project paths resolved relative to project root."""

    root: Path
    data_dir: Path
    raw_data_dir: Path
    outputs_dir: Path

    # Raw files
    train_csv: Path
    test_csv: Path
    sample_submission_csv: Path
    train_features_csv: Path
    test_features_csv: Path

    # Derived output directories
    lineage_dir: Path
    prepared_dir: Path
    splits_dir: Path
    runs_dir: Path
    model_cards_dir: Path
    feature_selection_dir: Path
    hpo_dir: Path
    robustness_dir: Path
    ensemble_dir: Path
    reports_dir: Path
    shift_dir: Path

    # Final submission artifact
    submission_path: Path

    @classmethod
    def from_root(cls, root: Path | str) -> ProjectPaths:
        resolved_root = Path(root).resolve()
        data_dir = resolved_root / "data"
        raw_data_dir = data_dir / "energy-anomaly-detection"
        outputs_dir = resolved_root / "outputs"

        return cls(
            root=resolved_root,
            data_dir=data_dir,
            raw_data_dir=raw_data_dir,
            outputs_dir=outputs_dir,
            train_csv=raw_data_dir / TRAIN_CSV,
            test_csv=raw_data_dir / TEST_CSV,
            sample_submission_csv=raw_data_dir / SAMPLE_SUBMISSION_CSV,
            train_features_csv=raw_data_dir / TRAIN_FEATURES_CSV,
            test_features_csv=raw_data_dir / TEST_FEATURES_CSV,
            lineage_dir=outputs_dir / "lineage",
            prepared_dir=outputs_dir / "prepared",
            splits_dir=outputs_dir / "splits",
            runs_dir=outputs_dir / "runs",
            model_cards_dir=outputs_dir / "model_cards",
            feature_selection_dir=outputs_dir / "feature_selection",
            hpo_dir=outputs_dir / "hpo",
            robustness_dir=outputs_dir / "robustness",
            ensemble_dir=outputs_dir / "ensemble",
            reports_dir=outputs_dir / "reports",
            shift_dir=outputs_dir / "shift",
            submission_path=outputs_dir / "submission.csv",
        )

    @classmethod
    def resolve(cls, start: Path | None = None) -> ProjectPaths:
        """Resolve ProjectPaths automatically from start path or current directory."""
        return cls.from_root(find_project_root(start))

    @property
    def output_dirs(self) -> tuple[Path, ...]:
        """All derived output directories."""
        return (
            self.outputs_dir,
            self.lineage_dir,
            self.prepared_dir,
            self.splits_dir,
            self.runs_dir,
            self.model_cards_dir,
            self.feature_selection_dir,
            self.hpo_dir,
            self.robustness_dir,
            self.ensemble_dir,
            self.reports_dir,
            self.shift_dir,
        )

    def ensure_directories(self) -> None:
        """Create derived output directories if they do not exist."""
        for dir_path in self.output_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
