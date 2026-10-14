"""Real Parquet development-only audits, with no estimator fits."""
import importlib.util
from dataclasses import replace

import polars as pl
import pytest

from full_campaign_fixtures import write_admission_fixture
from project.experiments.full_run_contracts import _file_hash


def audit_module():
    assert importlib.util.find_spec("project.experiments.full_development_audits"), "development audit is not implemented"
    from project.experiments import full_development_audits
    return full_development_audits


def rebound(fixture):
    return replace(fixture.spec, sources=tuple(
        replace(s, sha256=_file_hash(fixture.root / s.path)) for s in fixture.spec.sources))


def test_dev_predicate_excludes_vault_poison_and_preserves_null_nan(tmp_path):
    audit = audit_module()
    fixture = write_admission_fixture(tmp_path / "source")
    frame = fixture.full_frame.with_row_index("i").with_columns(
        pl.when(pl.col("i") == 0).then(-2.5)
        .when(pl.col("i") == 1).then(None)
        .when(pl.col("i") == 2).then(float("nan"))
        .otherwise(pl.col("meter_reading")).alias("meter_reading"),
        pl.when(pl.col("building_id").is_in(["bldg_ivy", "bldg_jade"]))
        .then(999).otherwise(pl.col("anomaly")).alias("anomaly"),
    ).drop("i")
    frame.write_parquet(fixture.data_path)
    report = audit.audit_full_development(fixture.root, rebound(fixture), audit.AuditPlan((2,), 42))
    assert report["development_rows"] == 52
    assert report["sentinel"]["invalid_values"] == [-2.5]
    assert report["sentinel"]["null_count"] == 1
    assert report["sentinel"]["nan_count"] == 1
    assert report["sentinel"]["replacement_value"] is None
    assert len(report["target_encoding"]) == 12
    assert all(p["plans"][0]["status"] == "supported" for p in report["target_encoding"])
    assert all(len(p["plans"][0]["membership_sha256"]) == 64 for p in report["target_encoding"])
    assert len(report["development_identity_sha256"]) == 64
    assert audit.write_audit_evidence(tmp_path / "real-fixture-report.json", report)


def test_wrong_split_rejected_before_data_scan(tmp_path, monkeypatch):
    audit = audit_module()
    fixture = write_admission_fixture(tmp_path / "source")
    manifest = pl.read_parquet(fixture.split_path).with_columns(pl.lit(99).alias("seed"))
    manifest.write_parquet(fixture.split_path)
    monkeypatch.setattr(audit, "_scan_development", lambda *a: pytest.fail("data scanned before split validation"))
    with pytest.raises(ValueError, match="seed"):
        audit.audit_full_development(fixture.root, rebound(fixture), audit.AuditPlan((2,), 42))


def test_post_scan_source_mutation_rejected(tmp_path, monkeypatch):
    audit = audit_module()
    fixture = write_admission_fixture(tmp_path / "source")
    original = audit._scan_development
    def mutate(*args):
        result = original(*args)
        (fixture.root / "src/project/features/registry.py").write_text("# changed\n")
        return result
    monkeypatch.setattr(audit, "_scan_development", mutate)
    with pytest.raises(ValueError):
        audit.audit_full_development(fixture.root, fixture.spec, audit.AuditPlan((2,), 42))


def test_source_hash_mismatch_rejected(tmp_path):
    audit = audit_module()
    fixture = write_admission_fixture(tmp_path / "source")
    fixture.data_path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        audit.audit_full_development(fixture.root, fixture.spec, audit.AuditPlan((2,), 42))


def test_training_only_groups_and_class_failure_are_reported(tmp_path):
    audit = audit_module()
    fixture = write_admission_fixture(tmp_path / "source")
    report = audit.audit_full_development(fixture.root, fixture.spec, audit.AuditPlan((2, 3, 4, 5), 42))
    for part in report["target_encoding"]:
        assert part["training_group_count"] == 6
        assert part["outer_validation_groups_used"] is False
        assert len(part["plans"]) == 4
        for plan in part["plans"]:
            if plan["status"] == "supported":
                assert all(min(side["class_counts"]) > 0 for fold in plan["folds"] for side in fold.values())


def test_evidence_is_strict_and_exclusive(tmp_path):
    audit = audit_module()
    path = tmp_path / "evidence.json"
    digest = audit.write_audit_evidence(path, {"actual": 3})
    assert digest == _file_hash(path)
    with pytest.raises(FileExistsError):
        audit.write_audit_evidence(path, {"actual": 4})
    with pytest.raises(ValueError):
        audit.write_audit_evidence(tmp_path / "bad.json", {"bad": float("nan")})


def test_enumeration_overrun_never_returns_partial_invalid_rule(tmp_path):
    audit = audit_module()
    fixture = write_admission_fixture(tmp_path / "source")
    fixture.full_frame.with_row_index("i").with_columns(
        (-pl.col("i").cast(pl.Float64) - 1).alias("meter_reading")
    ).drop("i").write_parquet(fixture.data_path)
    report = audit.audit_full_development(fixture.root, rebound(fixture),
                                         audit.AuditPlan((2,), 42, 52, 2))
    sentinel = report["sentinel"]
    assert sentinel["status"] == "enumeration_overrun"
    assert sentinel["invalid_values"] is None
    assert sentinel["unique_negative_count"] is None
    assert sentinel["negative_count"] == 52


def test_inner_fingerprint_matches_real_full_io_training_identity(tmp_path):
    audit = audit_module()
    from project.experiments.full_candidate_io import (
        build_development_index,
        load_full_training_partition,
    )
    from project.experiments.inner_cv import build_inner_folds, inner_membership_digest
    fixture = write_admission_fixture(tmp_path / "source")
    report = audit.audit_full_development(fixture.root, fixture.spec, audit.AuditPlan((2,), 42))
    index = build_development_index(fixture.root, fixture.spec, fixture.root / "fixture-index.sqlite")
    for part in report["target_encoding"]:
        training = load_full_training_partition(index, fixture.spec, part["repeat"], part["fold"])
        folds = build_inner_folds(training.y_train, training.groups_train, 2, 42)
        assert part["plans"][0]["membership_sha256"] == inner_membership_digest(folds, training.groups_train)


def test_mutation_produces_no_published_evidence(tmp_path, monkeypatch):
    audit = audit_module()
    fixture = write_admission_fixture(tmp_path / "source")
    destination = tmp_path / "must-not-exist.json"
    original = audit._audit_inner
    def mutate(*args):
        result = original(*args)
        (fixture.root / "scripts/campaign.py").write_text("# mutated during inner audit\n")
        return result
    monkeypatch.setattr(audit, "_audit_inner", mutate)
    with pytest.raises(ValueError, match="changed"):
        report = audit.audit_full_development(fixture.root, fixture.spec, audit.AuditPlan((2,), 42))
        audit.write_audit_evidence(destination, report)
    assert not destination.exists()


def test_leaking_scanner_is_rejected(tmp_path, monkeypatch):
    audit = audit_module()
    import pyarrow as pa
    fixture = write_admission_fixture(tmp_path / "source")
    class LeakingDataset:
        def scanner(self, **kwargs):
            assert kwargs["filter"] is not None
            assert kwargs["batch_size"] <= 64_000
            return self
        def to_batches(self):
            return [pa.RecordBatch.from_pandas(fixture.full_frame.to_pandas())]
    monkeypatch.setattr(audit.ds, "dataset", lambda *a, **kw: LeakingDataset())
    with pytest.raises(ValueError, match="leaked"):
        audit.audit_full_development(fixture.root, fixture.spec, audit.AuditPlan((2,), 42))


def test_source_symlink_escape_rejected(tmp_path):
    audit = audit_module()
    fixture = write_admission_fixture(tmp_path / "source")
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(fixture.data_path.read_bytes())
    fixture.data_path.unlink()
    fixture.data_path.symlink_to(outside)
    with pytest.raises(ValueError):
        audit.audit_full_development(fixture.root, fixture.spec, audit.AuditPlan((2,), 42))


def test_outer_one_class_and_environment_mismatch_fail_closed(tmp_path):
    audit = audit_module()
    fixture = write_admission_fixture(tmp_path / "source")
    with pytest.raises(ValueError, match="environment"):
        audit.audit_full_development(fixture.root, replace(fixture.spec, environment=(("python", "wrong"),)))
    fixture.full_frame.with_columns(pl.lit(0).alias("anomaly")).write_parquet(fixture.data_path)
    with pytest.raises(ValueError, match="outer partition"):
        audit.audit_full_development(fixture.root, rebound(fixture), audit.AuditPlan((2,), 42))
