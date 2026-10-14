"""Unit and contract tests for bounded T6 full nested HGB tuning lane."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from candidate_fixtures import synthetic_partitions
from project.experiments.candidate_contracts import (
    STUDY_SPACE,
    BlockedExperimentError,
    CandidateSpec,
    FoldPartition,
    StudySpec,
    TrainingPartition,
    training_view,
)
from project.experiments.full_nested_tuning import (
    FULL_HPO_DEFAULT_K,
    FULL_HPO_DEFAULT_SEED,
    FULL_HPO_DEFAULT_T,
    FullTuningResult,
    compute_fit_config_sha256,
    full_hpo_study_spec,
    partition_planned_units,
    required_full_fits,
    required_full_trials,
    tune_full_outer_partition,
)
from project.experiments.full_run_contracts import (
    FullCandidateSnapshot,
    FullFitJournal,
    FullResourcePolicy,
    FullRunSpec,
    SourceBinding,
)
from project.features.registry import build_lead_baseline_registry

PREDICTOR_COLUMNS = build_lead_baseline_registry().names


def test_recording_direct_lifecycle_compatibility_without_real_fit(tmp_path, outer_training_partition, test_study_spec):
    """Recording pipeline methods only: direct API still debits and never refits reuse."""
    from types import MethodType

    import numpy as np

    from project.experiments.nested_tuning import build_hgb_tuning_arm

    journal, _ = _make_hpo_journal_fixture(tmp_path, test_study_spec)
    fits = []
    def recording_factory(params):
        model = build_hgb_tuning_arm(params)
        def record_fit(self, X, y):
            assert journal.used_model_fits == len(fits) + 1
            fits.append(len(X))
            return self
        model.fit = MethodType(record_fit, model)
        model.predict_proba = MethodType(lambda self, X: np.full((len(X), 2), 0.5), model)
        return model
    try:
        result = tune_full_outer_partition(outer_training_partition, test_study_spec, journal,
                                           journal.root / "study", arm_factory=recording_factory)
        assert len(fits) == test_study_spec.n_trials * test_study_spec.inner_folds + 2
        assert result.selected_trial_id == 0
        resumed = tune_full_outer_partition(outer_training_partition, test_study_spec, journal,
            journal.root / "study", arm_factory=lambda *_: pytest.fail("unaccounted completed refit"))
        assert resumed.default_model is None and resumed.selected_model is None
        assert journal.used_model_fits == len(fits)
    finally:
        journal.close()


def test_recording_direct_rejects_ignored_fixed_configuration_before_debit(tmp_path, outer_training_partition, test_study_spec):
    from dataclasses import replace

    original, spec = _make_hpo_journal_fixture(tmp_path, test_study_spec)
    source_root = original.source_root
    original.close()
    snapshot = spec.candidates[0]
    changed = replace(snapshot, spec=replace(snapshot.spec, model_params=(("random_state", 41),)))
    with FullFitJournal(tmp_path / "changed-journal", replace(spec, candidates=(changed,)), source_root=source_root) as journal:
        with pytest.raises(ValueError, match="configuration|parameters"):
            tune_full_outer_partition(outer_training_partition, test_study_spec, journal, journal.root / "study",
                arm_factory=lambda *_: pytest.fail("invalid configuration reached construction"))
        assert journal.used_model_fits == 0


def test_recording_prediction_disk_auc_matches_independent_pairwise_ties(tmp_path, outer_training_partition, monkeypatch):
    """Known nonconstant recording predictions; neither estimator construction nor fit."""
    import numpy as np
    import pyarrow as pa

    from project.experiments import full_nested_tuning as tuning

    training = outer_training_partition
    positions = np.arange(len(training.y_train))
    labels = training.y_train.to_numpy()
    probabilities = np.asarray([0.1, 0.5, 0.5, 0.9] * ((len(labels) + 3) // 4))[:len(labels)]
    path = tmp_path / "recording_predictions.parquet"
    with pq.ParquetWriter(path, tuning.INNER_SCHEMA) as writer:
        for start in range(0, len(labels), 3):
            batch = positions[start:start + 3]
            writer.write_table(pa.Table.from_arrays([
                pa.array(batch, type=pa.int64()), pa.array(probabilities[batch], type=pa.float64()),
                pa.array(labels[batch], type=pa.int64()),
                pa.array([json.dumps(int(value)) for value in training.X_train.index[batch]], type=pa.string()),
                pa.array([json.dumps(value) for value in training.groups_train.iloc[batch]], type=pa.string()),
            ], schema=tuning.INNER_SCHEMA))
    positive, negative = probabilities[labels == 1], probabilities[labels == 0]
    expected = ((positive[:, None] > negative).sum() + 0.5 * (positive[:, None] == negative).sum()) / (len(positive) * len(negative))
    monkeypatch.setattr(tuning, "MAX_BATCH_ROWS", 3)
    monkeypatch.setattr(pq, "read_table", lambda *_args, **_kwargs: pytest.fail("disk AUC must stream"))
    assert tuning._prediction_auc(path, positions, labels, training) == expected


def _make_hpo_journal_fixture(
    root: Path,
    study_spec: StudySpec,
    experiment_id: str = "regime_same_family_hpo_vs_default",
) -> tuple[FullFitJournal, FullRunSpec]:
    """Helper creating a valid FullFitJournal binding for HPO testing."""
    root = Path(root)
    source_dir = root / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)

    # Minimal dummy sources for verify_bindings
    for name in ("matrix.yaml", "data.parquet", "split.parquet", "producer.py"):
        p = source_dir / name
        p.write_text(f"# dummy {name}")

    nb_path = source_dir / "nb.ipynb"
    nb_path.write_text(json.dumps({"cells": [{"id": "code-cell-1", "cell_type": "code"}]}))

    sources = tuple(
        SourceBinding(
            role=role,
            path=str((source_dir / name).relative_to(source_dir)),
            sha256=hashlib.sha256((source_dir / name).read_bytes()).hexdigest(),
        )
        for role, name in (
            ("matrix", "matrix.yaml"),
            ("data", "data.parquet"),
            ("split", "split.parquet"),
            ("producer", "producer.py"),
            ("notebook", "nb.ipynb"),
        )
    )

    candidate_spec = CandidateSpec(
        experiment_id=experiment_id,
        feature_pool=PREDICTOR_COLUMNS,
        model_family="hist_gradient_boosting",
        model_params=(),
    )
    snapshot = FullCandidateSnapshot(
        spec=candidate_spec,
        arms=("control", "challenger"),
        source_notebook="nb.ipynb",
        source_cell_id="code-cell-1",
        study=study_spec,
        study_method="random_search",
    )
    spec = FullRunSpec(
        run_id="test-hpo-run",
        candidates=(snapshot,),
        sources=sources,
        environment=(("python", "test"),),
        resource_policy=FullResourcePolicy(informed_opt_out=True),
    )
    journal = FullFitJournal(root / "journal", spec, source_root=source_dir)
    return journal, spec


@pytest.fixture
def test_study_spec() -> StudySpec:
    """Explicit minimal 2-trial, 2-fold study spec for fast tests."""
    return StudySpec(
        n_trials=2,
        inner_folds=2,
        seed=42,
        parameter_lists=(
            ("learning_rate", (0.03, 0.1)),
            ("max_leaf_nodes", (15, 31)),
        ),
    )


@pytest.fixture
def outer_training_partition() -> TrainingPartition:
    """Outer training partition fixture (16 rows, 4 groups, repeat 0, fold 0)."""
    partitions = synthetic_partitions()
    return training_view(partitions[0])


# ---------------------------------------------------------------------------
# Test 1: Accounting formula: 12*(T*K + 2) fits and 12*T trials
# ---------------------------------------------------------------------------


def test_required_fits_and_trials_accounting():
    """Verify exact model fit and trial formulas for full campaign HPO."""
    assert required_full_fits(n_trials=8, inner_folds=3) == 12 * (8 * 3 + 2)  # 312
    assert required_full_trials(n_trials=8) == 12 * 8  # 96
    assert required_full_fits(n_trials=2, inner_folds=2) == 12 * (2 * 2 + 2)  # 72
    assert required_full_trials(n_trials=2) == 12 * 2  # 24
    assert required_full_fits(n_trials=2, inner_folds=2, outer_partitions=1) == 6
    assert required_full_trials(n_trials=2, outer_partitions=1) == 2

    with pytest.raises(ValueError, match="n_trials"):
        required_full_fits(n_trials=0, inner_folds=3)
    with pytest.raises(ValueError, match="n_trials"):
        required_full_fits(n_trials=True, inner_folds=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="inner_folds"):
        required_full_fits(n_trials=8, inner_folds=1)
    with pytest.raises(ValueError, match="inner_folds"):
        required_full_fits(n_trials=8, inner_folds=6)


def test_approved_full_hpo_study_spec():
    """Approved full campaign spec: finite ParameterSampler, seed 42, T=8, K=3."""
    spec = full_hpo_study_spec()
    assert spec.n_trials == FULL_HPO_DEFAULT_T == 8
    assert spec.inner_folds == FULL_HPO_DEFAULT_K == 3
    assert spec.seed == FULL_HPO_DEFAULT_SEED == 42
    assert len(spec.parameter_lists) == 4
    param_names = {name for name, _ in spec.parameter_lists}
    assert param_names == set(STUDY_SPACE.keys())


def test_partition_planned_units_matches_journal_contracts(test_study_spec):
    """partition_planned_units lists exact matched control/challenger units."""
    units = partition_planned_units(
        experiment_id="regime_same_family_hpo_vs_default",
        repeat=0,
        fold=0,
        study_spec=test_study_spec,
    )
    assert len(units) == 6
    assert units[0] == "regime_same_family_hpo_vs_default/control/r0/f0/default_refit"
    assert units[1] == "regime_same_family_hpo_vs_default/challenger/r0/f0/inner/trial0/fold0"
    assert units[2] == "regime_same_family_hpo_vs_default/challenger/r0/f0/inner/trial0/fold1"
    assert units[3] == "regime_same_family_hpo_vs_default/challenger/r0/f0/inner/trial1/fold0"
    assert units[4] == "regime_same_family_hpo_vs_default/challenger/r0/f0/inner/trial1/fold1"
    assert units[5] == "regime_same_family_hpo_vs_default/challenger/r0/f0/selected_refit"


# ---------------------------------------------------------------------------
# Test 2: Input guards & prohibition of outer labels
# ---------------------------------------------------------------------------


def test_tune_outer_training_rejects_fold_partition_or_validation_fields(
    tmp_path, test_study_spec
):
    """Tuning must strictly reject FoldPartition or held-out validation fields."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    partitions = synthetic_partitions()
    fold_part = partitions[0]
    assert isinstance(fold_part, FoldPartition)

    with pytest.raises(TypeError, match="FoldPartition is prohibited"):
        tune_full_outer_partition(
            training=fold_part,  # type: ignore[arg-type]
            study_spec=test_study_spec,
            journal=journal,
            study_dir=journal.root / "study",
        )


def test_tune_outer_training_rejects_unregistered_columns_or_wrong_pool(
    tmp_path, test_study_spec, outer_training_partition
):
    """Predictors must strictly match the CORE_FEATURE_POOL."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    bad_df = outer_training_partition.X_train.drop(columns=["hour"])
    bad_partition = TrainingPartition(
        repeat=0,
        fold=0,
        X_train=bad_df,
        y_train=outer_training_partition.y_train,
        groups_train=outer_training_partition.groups_train,
    )
    with pytest.raises(ValueError, match="CORE feature pool"):
        tune_full_outer_partition(
            training=bad_partition,
            study_spec=test_study_spec,
            journal=journal,
            study_dir=journal.root / "study",
        )


# ---------------------------------------------------------------------------
# Defect (5): Spec verification, directory location, and symlink prohibition
# ---------------------------------------------------------------------------


def test_tune_outer_training_rejects_study_spec_mismatch_with_journal_snapshot(
    tmp_path, test_study_spec, outer_training_partition
):
    """Provided StudySpec must equal journal snapshot before any write."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    different_spec = StudySpec(
        n_trials=3,  # Differs from snapshot (2)
        inner_folds=2,
        seed=42,
        parameter_lists=test_study_spec.parameter_lists,
    )
    study_dir = journal.root / "study"
    with pytest.raises(BlockedExperimentError, match="study_spec_mismatch"):
        tune_full_outer_partition(
            training=outer_training_partition,
            study_spec=different_spec,
            journal=journal,
            study_dir=study_dir,
        )
    # Assert nothing was created in study_dir before failing
    assert not study_dir.exists()


def test_tune_outer_training_rejects_study_dir_outside_journal_root(
    tmp_path, test_study_spec, outer_training_partition
):
    """Study directory must be strictly located beneath the journal root."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    outside_dir = tmp_path / "outside_study"
    with pytest.raises(ValueError, match="under journal root"):
        tune_full_outer_partition(
            training=outer_training_partition,
            study_spec=test_study_spec,
            journal=journal,
            study_dir=outside_dir,
        )


def test_tune_outer_training_rejects_symlink_db_files(
    tmp_path, test_study_spec, outer_training_partition
):
    """Database and WAL/SHM files must not be symlinks."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    study_dir = journal.root / "study"
    study_dir.mkdir(parents=True, exist_ok=True)
    target_file = tmp_path / "target.db"
    target_file.touch()
    db_symlink = study_dir / f"full_study_r{outer_training_partition.repeat}_f{outer_training_partition.fold}.sqlite3"
    db_symlink.symlink_to(target_file)

    with pytest.raises(ValueError, match="symlink"):
        tune_full_outer_partition(
            training=outer_training_partition,
            study_spec=test_study_spec,
            journal=journal,
            study_dir=study_dir,
        )


# ---------------------------------------------------------------------------
# Defect (2): Journal occupancy check & supervisor-owned interruption
# ---------------------------------------------------------------------------


def test_tune_rejects_occupied_journal_without_interrupting_active(
    tmp_path, test_study_spec, outer_training_partition
):
    """Tuner must reject occupied journal with active attempt without interrupting it."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    # Simulate an active attempt left by another worker or process
    units = partition_planned_units(
        "regime_same_family_hpo_vs_default", 0, 0, test_study_spec
    )
    active = journal.begin(units[0])
    assert active.status == "running"

    study_dir = journal.root / "study"
    with pytest.raises(BlockedExperimentError, match="journal_occupied"):
        tune_full_outer_partition(
            training=outer_training_partition,
            study_spec=test_study_spec,
            journal=journal,
            study_dir=study_dir,
        )

    # Verify tuner DID NOT touch or reconcile the active attempt
    updated_records = [a for a in journal.attempts() if a.attempt_id == active.attempt_id]
    assert len(updated_records) == 1
    assert updated_records[0].status == "running"


# ---------------------------------------------------------------------------
# Test 3 & Defect (1): Normal run fits models; Resume returns verified artifacts with NO model refit
# ---------------------------------------------------------------------------


def test_tune_full_outer_partition_completes_and_returns_models(
    tmp_path, test_study_spec, outer_training_partition
):
    """Execute complete outer partition tuning (T=2, K=2) and verify returned artifacts."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    study_dir = journal.root / "study"

    constructor_calls = 0
    fit_calls = 0

    from project.experiments.nested_tuning import build_hgb_tuning_arm

    def tracking_factory(params: dict[str, Any]) -> Pipeline:
        nonlocal constructor_calls
        constructor_calls += 1
        arm = build_hgb_tuning_arm(params)
        orig_fit = arm.fit

        def wrapped_fit(*args, **kwargs):
            nonlocal fit_calls
            fit_calls += 1
            return orig_fit(*args, **kwargs)

        arm.fit = wrapped_fit  # type: ignore[method-assign]
        return arm

    result = tune_full_outer_partition(
        training=outer_training_partition,
        study_spec=test_study_spec,
        journal=journal,
        study_dir=study_dir,
        arm_factory=tracking_factory,
    )

    assert isinstance(result, FullTuningResult)
    assert result.repeat == 0
    assert result.fold == 0
    assert result.selected_trial_id in (0, 1)
    assert isinstance(result.selected_params, tuple)
    assert len(result.selected_params) > 0

    # For T=2, K=2: 1 default refit + 4 inner fits + 1 selected refit = 6 constructors & 6 fits
    assert constructor_calls == 6
    assert fit_calls == 6

    # Verify returned models are fitted
    assert isinstance(result.default_model, Pipeline)
    assert isinstance(result.selected_model, Pipeline)
    default_probs = result.default_model.predict_proba(outer_training_partition.X_train)
    selected_probs = result.selected_model.predict_proba(outer_training_partition.X_train)
    assert default_probs.shape == (len(outer_training_partition.X_train), 2)
    assert selected_probs.shape == (len(outer_training_partition.X_train), 2)

    # Verify journal accounting: exactly 6 units completed
    completed_units = journal.verify_completed()
    expected_units = partition_planned_units(
        "regime_same_family_hpo_vs_default", 0, 0, test_study_spec
    )
    for u in expected_units:
        assert u in completed_units

    fit_sha = compute_fit_config_sha256(
        "regime_same_family_hpo_vs_default", "challenger", result.selected_params
    )
    assert result.selected_params_hash == fit_sha


def test_resume_returns_verified_artifacts_without_calling_constructor_or_fit(
    tmp_path, test_study_spec, outer_training_partition
):
    """Resume must NOT silently refit; it must return verified artifacts with None models."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    study_dir = journal.root / "study"

    # First run completes normally
    result1 = tune_full_outer_partition(
        training=outer_training_partition,
        study_spec=test_study_spec,
        journal=journal,
        study_dir=study_dir,
    )
    assert result1.default_model is not None
    assert result1.selected_model is not None

    # Resume run: install tracking factory that fails if constructor or fit is called
    resume_constructors = 0
    resume_fits = 0

    def forbidden_factory(params: dict[str, Any]) -> Pipeline:
        nonlocal resume_constructors
        resume_constructors += 1
        raise AssertionError("constructor called during resume without explicit retry")

    result2 = tune_full_outer_partition(
        training=outer_training_partition,
        study_spec=test_study_spec,
        journal=journal,
        study_dir=study_dir,
        arm_factory=forbidden_factory,
    )

    # Must not call constructor or fit
    assert resume_constructors == 0
    assert resume_fits == 0

    # Models must be None rather than fabricated
    assert result2.default_model is None
    assert result2.selected_model is None

    # Verified evidence and metadata must match
    assert result2.default_evidence == result1.default_evidence
    assert result2.selected_refit_evidence == result1.selected_refit_evidence
    assert result2.selected_trial_id == result1.selected_trial_id
    assert result2.selected_params == result1.selected_params
    assert result2.selected_params_hash == result1.selected_params_hash


# ---------------------------------------------------------------------------
# Defect (6): Compact full manifest & streamed Parquet evidence
# ---------------------------------------------------------------------------


def test_compact_full_manifest_in_sqlite(
    tmp_path, test_study_spec, outer_training_partition
):
    """Study manifest must be compact: binds source root and digests, no row/group lists or fake caps."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    study_dir = journal.root / "study"

    result = tune_full_outer_partition(
        training=outer_training_partition,
        study_spec=test_study_spec,
        journal=journal,
        study_dir=study_dir,
    )

    conn = sqlite3.connect(result.study_db_path)
    try:
        row = conn.execute("SELECT manifest_payload_json FROM study").fetchone()
        assert row is not None
        manifest = json.loads(row[0])

        # Must bind actual source root and digests
        assert "source_root" in manifest
        assert "candidate_spec" in manifest
        assert "environment" in manifest
        assert "resource_policy" in manifest
        assert "feature_encoding_digest" in manifest
        assert "training_digests" in manifest
        assert "study_spec" in manifest

        # Prohibit fake caps and million-row identity lists
        assert "budget" not in manifest or manifest["budget"].get("cpu_threads") != 1
        assert "row_identities" not in manifest
        assert "group_identities" not in manifest
    finally:
        conn.close()


def test_inner_validation_evidence_is_stored_as_parquet(
    tmp_path, test_study_spec, outer_training_partition
):
    """Inner validation predictions must be in chunked/columnar Parquet with compact evidence."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    study_dir = journal.root / "study"

    tune_full_outer_partition(
        training=outer_training_partition,
        study_spec=test_study_spec,
        journal=journal,
        study_dir=study_dir,
    )

    unit_0_0 = "regime_same_family_hpo_vs_default/challenger/r0/f0/inner/trial0/fold0"
    attempt_0_0 = next(a for a in journal.attempts() if a.unit == unit_0_0)
    attempt_dir = journal.root / attempt_0_0.artifact_directory

    parquet_file = attempt_dir / "inner_predictions.parquet"
    evidence_json = attempt_dir / "evidence.json"

    assert parquet_file.is_file()
    assert evidence_json.is_file()

    table = pq.read_table(parquet_file)
    assert "validation_position" in table.column_names
    assert "probability" in table.column_names
    assert "target" in table.column_names

    meta = json.loads(evidence_json.read_text())
    assert "validation_positions" not in meta
    assert "probabilities" not in meta
    assert "score" in meta
    assert "positions_sha256" in meta
    assert "parquet_sha256" in meta
    assert meta["trial_id"] == 0
    assert meta["fold_index"] == 0


# ---------------------------------------------------------------------------
# Test 5: Durable debit before estimator construction
# ---------------------------------------------------------------------------


def test_durable_debit_before_construction_charges_on_failure(
    tmp_path, test_study_spec, outer_training_partition, monkeypatch
):
    """If arm construction fails, the journal attempt must already be debited."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    study_dir = journal.root / "study"

    from project.experiments import full_nested_tuning

    original_builder = full_nested_tuning.build_hgb_tuning_arm
    call_count = 0

    def failing_builder(params):
        nonlocal call_count
        call_count += 1
        if call_count == 2:  # Fail on second fit (first inner trial)
            raise RuntimeError("exploding estimator constructor")
        return original_builder(params)

    with pytest.raises(RuntimeError, match="exploding estimator constructor"):
        tune_full_outer_partition(
            training=outer_training_partition,
            study_spec=test_study_spec,
            journal=journal,
            study_dir=study_dir,
            arm_factory=failing_builder,
        )

    attempts = journal.attempts()
    assert len(attempts) >= 2
    failed_attempt = attempts[-1]
    assert failed_attempt.status == "failed"
    assert "exploding estimator constructor" in (failed_attempt.reason or "")


# ---------------------------------------------------------------------------
# Defect (7): Deterministic tie break with GENUINE predictions & state separation
# ---------------------------------------------------------------------------


def test_deterministic_tie_break_prefers_lower_trial_id_with_genuine_predictions(
    tmp_path, test_study_spec, outer_training_partition, monkeypatch
):
    """When candidate trials have genuine equal mean ROC-AUC scores, lower trial_id wins."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    study_dir = journal.root / "study"

    result = tune_full_outer_partition(
        training=outer_training_partition,
        study_spec=test_study_spec,
        journal=journal,
        study_dir=study_dir,
    )

    # The tiny HGB fixture has fewer than min_samples_leaf rows: its genuinely
    # fitted predictors are constant, yielding an independent tied AUC of .5.
    scores = []
    for attempt in journal.attempts():
        if "/inner/" in attempt.unit:
            predictions = pq.read_table(journal.root / attempt.artifact_directory / "inner_predictions.parquet")
            score = roc_auc_score(predictions["target"].to_numpy(), predictions["probability"].to_numpy())
            assert score == 0.5
            scores.append(score)
    assert len(scores) == test_study_spec.n_trials * test_study_spec.inner_folds
    assert result.selected_trial_id == 0

    # Ensure evidence verification succeeds on resume with these genuine predictions
    result_resumed = tune_full_outer_partition(
        training=outer_training_partition,
        study_spec=test_study_spec,
        journal=journal,
        study_dir=study_dir,
    )
    assert result_resumed.selected_trial_id == 0


def test_selection_finished_distinct_from_refit_completed(
    tmp_path, test_study_spec, outer_training_partition, monkeypatch
):
    """Database status must reflect 'selection_finished' before 'completed', and avoid double finish."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    study_dir = journal.root / "study"

    from project.experiments.nested_tuning import build_hgb_tuning_arm

    fit_count = 0

    def exploding_refit_factory(params: dict[str, Any]) -> Pipeline:
        nonlocal fit_count
        fit_count += 1
        arm = build_hgb_tuning_arm(params)
        orig_fit = arm.fit

        def wrapped_fit(*args, **kwargs):
            if fit_count == 6:  # Explode on selected refit fit call
                raise RuntimeError("selected refit hardware failure")
            return orig_fit(*args, **kwargs)

        arm.fit = wrapped_fit  # type: ignore[method-assign]
        return arm

    with pytest.raises(RuntimeError, match="selected refit hardware failure"):
        tune_full_outer_partition(
            training=outer_training_partition,
            study_spec=test_study_spec,
            journal=journal,
            study_dir=study_dir,
            arm_factory=exploding_refit_factory,
        )

    # In database, study status must be 'selection_finished', NOT 'completed'
    db_path = study_dir / f"full_study_r{outer_training_partition.repeat}_f{outer_training_partition.fold}.sqlite3"
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT status, selected_trial_id FROM study").fetchone()
        assert row is not None
        assert row[0] == "selection_finished"
        assert row[1] in (0, 1)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Defect (3) & (4): Tamper tests for trial params and joint evidence digests
# ---------------------------------------------------------------------------


def test_resume_rejects_tampered_trial_params_in_database(
    tmp_path, test_study_spec, outer_training_partition
):
    """Tampering with trial.params_json in SQLite must fail closed and refuse unevaluated params."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    study_dir = journal.root / "study"

    tune_full_outer_partition(
        training=outer_training_partition,
        study_spec=test_study_spec,
        journal=journal,
        study_dir=study_dir,
    )

    db_path = study_dir / f"full_study_r{outer_training_partition.repeat}_f{outer_training_partition.fold}.sqlite3"
    conn = sqlite3.connect(db_path)
    try:
        # Alter params_json in SQLite to an unevaluated parameter value
        tampered_params = {"learning_rate": 0.999, "max_leaf_nodes": 999}
        conn.execute(
            "UPDATE trial SET params_json = ? WHERE trial_id = 0",
            (json.dumps(tampered_params),),
        )
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(BlockedExperimentError, match="study_tampered"):
        tune_full_outer_partition(
            training=outer_training_partition,
            study_spec=test_study_spec,
            journal=journal,
            study_dir=study_dir,
        )


def test_resume_rejects_joint_evidence_tampering(
    tmp_path, test_study_spec, outer_training_partition
):
    """Joint evidence tamper (mutating both Parquet and evidence.json) must fail closed against journal seal."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    study_dir = journal.root / "study"

    tune_full_outer_partition(
        training=outer_training_partition,
        study_spec=test_study_spec,
        journal=journal,
        study_dir=study_dir,
    )

    unit_0_0 = "regime_same_family_hpo_vs_default/challenger/r0/f0/inner/trial0/fold0"
    attempt_0_0 = next(a for a in journal.attempts() if a.unit == unit_0_0)
    attempt_dir = journal.root / attempt_0_0.artifact_directory

    parquet_file = attempt_dir / "inner_predictions.parquet"
    evidence_json = attempt_dir / "evidence.json"

    # Joint tamper: rewrite parquet and rewrite evidence.json so they agree with each other
    tampered_table = pq.read_table(parquet_file)
    import pyarrow as pa

    probabilities = tampered_table["probability"].to_numpy().copy()
    probabilities[0] = 0.99 if probabilities[0] != 0.99 else 0.01
    tampered_table = tampered_table.set_column(1, "probability", pa.array(probabilities))
    pq.write_table(tampered_table, parquet_file)
    new_parquet_sha = hashlib.sha256(parquet_file.read_bytes()).hexdigest()

    meta = json.loads(evidence_json.read_text())
    meta["parquet_sha256"] = new_parquet_sha
    meta_bytes = json.dumps(meta, indent=2).encode("utf-8")
    evidence_json.write_bytes(meta_bytes)

    # Resume must check attempt.evidence digests sealed in the journal and fail
    with pytest.raises(BlockedExperimentError, match="study_tampered"):
        tune_full_outer_partition(
            training=outer_training_partition,
            study_spec=test_study_spec,
            journal=journal,
            study_dir=study_dir,
        )


def test_resume_rejects_tampered_refit_evidence(
    tmp_path, test_study_spec, outer_training_partition
):
    """Tampering with outer refit evidence JSON must fail closed on resume."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    study_dir = journal.root / "study"

    tune_full_outer_partition(
        training=outer_training_partition,
        study_spec=test_study_spec,
        journal=journal,
        study_dir=study_dir,
    )

    default_unit = "regime_same_family_hpo_vs_default/control/r0/f0/default_refit"
    default_attempt = next(a for a in journal.attempts() if a.unit == default_unit)
    refit_json = journal.root / default_attempt.artifact_directory / "fit_default_refit.json"
    refit_json.write_text('{"tampered": true}')

    with pytest.raises(BlockedExperimentError, match="study_tampered"):
        tune_full_outer_partition(
            training=outer_training_partition,
            study_spec=test_study_spec,
            journal=journal,
            study_dir=study_dir,
        )


def test_inner_predictions_are_batched_and_resume_never_collects_parquet(
    tmp_path, test_study_spec, outer_training_partition, monkeypatch
):
    from project.experiments import full_nested_tuning as tuning

    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    monkeypatch.setattr(tuning, "MAX_BATCH_ROWS", 3, raising=False)
    original = tuning._extract_pos_proba
    sizes = []

    def bounded_prediction(arm, frame):
        sizes.append(len(frame))
        assert len(frame) <= 3, "inner validation predictions must be streamed"
        return original(arm, frame)

    monkeypatch.setattr(tuning, "_extract_pos_proba", bounded_prediction)
    result = tune_full_outer_partition(outer_training_partition, test_study_spec, journal, journal.root / "study")
    assert sizes

    def forbidden_population_read(*args, **kwargs):
        raise AssertionError("resume must stream evidence, not collect parquet")

    monkeypatch.setattr(tuning.pq, "read_table", forbidden_population_read)
    resumed = tune_full_outer_partition(outer_training_partition, test_study_spec, journal, journal.root / "study")
    assert resumed.selected_params == result.selected_params


def test_failed_refit_requires_explicit_retry_authorization(
    tmp_path, test_study_spec, outer_training_partition
):
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)

    def failed_constructor(params):
        raise RuntimeError("hardware failure")

    with pytest.raises(RuntimeError, match="hardware failure"):
        tune_full_outer_partition(outer_training_partition, test_study_spec, journal, journal.root / "study",
                                  arm_factory=failed_constructor)
    before = journal.attempts()
    with pytest.raises(BlockedExperimentError, match="explicit_retry_required"):
        tune_full_outer_partition(outer_training_partition, test_study_spec, journal, journal.root / "study")
    assert journal.attempts() == before


def test_cancellation_terminalizes_only_owned_debit(tmp_path, test_study_spec, outer_training_partition):
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)

    def cancelled_constructor(params):
        raise KeyboardInterrupt("cancel fixture")

    with pytest.raises(KeyboardInterrupt):
        tune_full_outer_partition(outer_training_partition, test_study_spec, journal, journal.root / "study",
                                  arm_factory=cancelled_constructor)
    attempts = journal.attempts()
    assert len(attempts) == 1
    assert attempts[0].status == "interrupted"


def test_selected_retry_charges_one_new_fit_without_refitting_completed_default(
    tmp_path, test_study_spec, outer_training_partition
):
    from project.experiments.nested_tuning import build_hgb_tuning_arm

    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    calls = 0

    def interrupted_factory(params):
        nonlocal calls
        calls += 1
        if calls == 6:
            raise RuntimeError("selected construction failure")
        return build_hgb_tuning_arm(params)

    with pytest.raises(RuntimeError):
        tune_full_outer_partition(outer_training_partition, test_study_spec, journal, journal.root / "study",
                                  arm_factory=interrupted_factory)
    fits = 0

    def retry_factory(params):
        arm = build_hgb_tuning_arm(params)
        original = arm.fit

        def count_fit(*args, **kwargs):
            nonlocal fits
            fits += 1
            return original(*args, **kwargs)

        arm.fit = count_fit
        return arm

    result = tune_full_outer_partition(outer_training_partition, test_study_spec, journal, journal.root / "study",
                                      arm_factory=retry_factory, retry_reason="parent: retry selected fixture failure")
    assert fits == 1
    assert result.default_model is None
    assert result.selected_model is not None
    assert len(journal.attempts()) == 7
    assert len(journal.verify_completed()) == 6


@pytest.mark.parametrize("mutation", ["extra_trial", "manifest", "score"])
def test_resume_authenticates_entire_study_against_canonical_trials_and_journal(
    tmp_path, test_study_spec, outer_training_partition, mutation
):
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    result = tune_full_outer_partition(outer_training_partition, test_study_spec, journal, journal.root / "study")
    with sqlite3.connect(result.study_db_path) as connection:
        if mutation == "extra_trial":
            row = connection.execute("SELECT * FROM trial WHERE trial_id=0").fetchone()
            connection.execute("INSERT INTO trial VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (row[0], 999, row[2], row[3], "completed", 1.0, None, row[7], row[8]))
        elif mutation == "manifest":
            connection.execute("UPDATE study SET manifest_payload_json='{}'")
        else:
            connection.execute("UPDATE fold_score SET score=0.123 WHERE trial_id=0 AND fold_index=0")
    before = journal.attempts()
    with pytest.raises(BlockedExperimentError, match="study_tampered"):
        tune_full_outer_partition(outer_training_partition, test_study_spec, journal, journal.root / "study")
    assert journal.attempts() == before


def test_resume_rejects_changed_membership_or_spec(
    tmp_path, test_study_spec, outer_training_partition
):
    """If study spec changes (e.g. seed changed), resume must fail closed."""
    journal, _ = _make_hpo_journal_fixture(tmp_path / "run", test_study_spec)
    study_dir = journal.root / "study"

    tune_full_outer_partition(
        training=outer_training_partition,
        study_spec=test_study_spec,
        journal=journal,
        study_dir=study_dir,
    )

    different_spec = StudySpec(
        n_trials=2,
        inner_folds=2,
        seed=999,  # Changed seed
        parameter_lists=test_study_spec.parameter_lists,
    )

    with pytest.raises((BlockedExperimentError, ValueError)):
        tune_full_outer_partition(
            training=outer_training_partition,
            study_spec=different_spec,
            journal=journal,
            study_dir=study_dir,
        )


def test_fit_config_sha256_is_deterministic_and_parameter_sensitive():
    """Different parameters produce distinct SHA-256 hashes."""
    params_a = (("learning_rate", 0.03), ("max_leaf_nodes", 15))
    params_b = (("learning_rate", 0.1), ("max_leaf_nodes", 31))

    sha_a = compute_fit_config_sha256("regime_same_family_hpo_vs_default", "challenger", params_a)
    sha_b = compute_fit_config_sha256("regime_same_family_hpo_vs_default", "challenger", params_b)

    assert len(sha_a) == 64
    assert len(sha_b) == 64
    assert sha_a != sha_b
    assert sha_a == compute_fit_config_sha256("regime_same_family_hpo_vs_default", "challenger", params_a)
