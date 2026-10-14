"""Read-only fixture preflight and authentic, stale-sensitive launch grants."""

import hashlib
import importlib
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from tests.full_campaign_fixtures import write_fixture

from project.experiments.full_campaign_admission import CORE, FullCampaignIntent
from project.experiments.full_run_contracts import SourceBinding

ROOT = Path(__file__).resolve().parents[2]


def api():
    from project import experiments
    assert importlib.util.find_spec(experiments.__name__ + ".full_campaign_preflight"), (
        "read-only authenticated preflight is missing"
    )
    return importlib.import_module(experiments.__name__ + ".full_campaign_preflight")


@pytest.fixture
def full_request(tmp_path):
    fixture = write_fixture(tmp_path)
    for relative in ("configs/experiments/feature_campaign.yaml", "configs/experiments/baselines.yaml",
                     "notebooks/02_baseline_and_candidate_design.ipynb"):
        (tmp_path / relative).write_bytes((ROOT / relative).read_bytes())
    # The entire scientific implementation inventory is real fixture source,
    # not a caller-supplied allowlist or implementation-ready boolean.
    for path in (ROOT / "src/project").rglob("*.py"):
        destination = tmp_path / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/train.py").write_bytes((ROOT / "scripts/train.py").read_bytes())
    (tmp_path / "scripts/campaign.py").write_bytes((ROOT / "scripts/campaign.py").read_bytes())
    sources = tuple(SourceBinding(s.role, s.path, hashlib.sha256((tmp_path / s.path).read_bytes()).hexdigest())
                    for s in fixture.spec.sources)
    baseline = "configs/experiments/baselines.yaml"
    sources += (SourceBinding("baseline", baseline, hashlib.sha256((tmp_path / baseline).read_bytes()).hexdigest()),)
    candidate = replace(fixture.spec.candidates[0], source_cell_id="cand-matrix-render",
                        spec=replace(fixture.spec.candidates[0].spec, feature_pool=CORE))
    spec = replace(fixture.spec, candidates=(candidate,), sources=sources,
                   development_rows=52, vault_rows=4)
    return tmp_path, spec


def intent(module, spec):
    return FullCampaignIntent(replace(spec, environment=module.current_environment()), True,
                              resource_opt_out_record="user: all development rows sequential; no RAM/CPU/wall caps")


def data_preflight(module, root, request):
    function = getattr(module, "preflight_full_campaign_data", None)
    assert callable(function), "narrow authenticated data-preflight API is missing"
    return function(root, request)


def test_data_preflight_is_not_launch_authorization_and_inventory_is_complete(full_request):
    module = api()
    root, spec = full_request
    before = sorted(str(p) for p in root.rglob("*"))
    admitted = data_preflight(module, root, intent(module, spec))
    assert admitted.spec.development_rows == 52
    assert admitted.train_rows == 56
    assert len(admitted.rows) == 21
    assert {r.record.status for r in admitted.rows} == {"blocked", "prohibited"}
    assert admitted.data_ready_experiment_ids == ("feat_missingness_indicators",)
    assert admitted.execution_authorized is False
    assert all("full_dispatch_unavailable" in row.record.blocker_codes for row in admitted.rows)
    assert admitted.resource_opt_out_record.startswith("user:")
    assert all(not r.record.n03_eligible for r in admitted.rows)
    assert sorted(str(p) for p in root.rglob("*")) == before
    module.verify_full_data_preflight(root, admitted, spec=admitted.spec)
    for phase in ("launch", "resume", "completion"):
        with pytest.raises((ValueError, TypeError)):
            module.verify_admitted_full_run(root, admitted, spec=admitted.spec, phase=phase)


@pytest.mark.parametrize("change", ["denied", "environment", "hash", "counts", "class", "repeat", "duplicate", "timestamp", "optout"])
def test_preflight_denies_invalid_inputs_without_writes_or_fit(full_request, change, monkeypatch):
    module = api()
    root, spec = full_request
    request = intent(module, spec)
    if change == "denied":
        request = replace(request, explicit_full_request=False)
    elif change == "environment":
        request = replace(request, spec=replace(request.spec, environment=(("python", "forged"),)))
    elif change == "hash":
        request = replace(request, spec=replace(request.spec, sources=tuple(replace(s, sha256="0" * 64) if s.role == "data" else s for s in spec.sources)))
    elif change == "counts":
        request = replace(request, spec=replace(request.spec, development_rows=53))
    elif change == "optout":
        request = replace(request, resource_opt_out_record=None)
    else:
        role = "split" if change == "repeat" else "data"
        source = next(s for s in spec.sources if s.role == role)
        path = root / source.path
        frame = pl.read_parquet(path)
        if change == "class":
            frame = frame.with_columns(pl.lit(0, dtype=pl.Int64).alias("anomaly"))
        elif change == "repeat":
            frame = frame.filter(~((pl.col("repeat") == 2) & (pl.col("building_id") == "bldg_amber")))
        elif change == "duplicate":
            frame = pl.concat([frame, frame.head(1)])
        else:
            frame = frame.with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))
        frame.write_parquet(path)
        request = replace(request, spec=replace(request.spec, sources=tuple(
            replace(s, sha256=hashlib.sha256(path.read_bytes()).hexdigest()) if s.role == role else s for s in spec.sources)))
    from sklearn.ensemble import HistGradientBoostingClassifier
    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", lambda *a, **kw: pytest.fail("preflight fitted"))
    before = sorted(str(p) for p in root.rglob("*"))
    with pytest.raises((ValueError, TypeError)):
        data_preflight(module, root, request)
    assert sorted(str(p) for p in root.rglob("*")) == before


@pytest.mark.parametrize("change", ["spec", "grant", "source", "implementation", "accounting", "root", "boolean"])
def test_grant_rejects_forgery_staleness_and_mutation_at_every_boundary(full_request, change):
    module = api()
    root, spec = full_request
    grant = data_preflight(module, root, intent(module, spec))
    requested = grant.spec
    if change == "spec":
        requested = replace(requested, run_id="other")
    elif change == "grant":
        grant = replace(grant, train_rows=999)
    elif change == "source":
        (root / "configs/experiments/baselines.yaml").write_text("forged")
    elif change == "implementation":
        (root / "src/project/experiments/candidate_dispatch.py").write_text("forged")
    elif change == "accounting":
        (root / "src/project/experiments/fit_accounting.py").write_text("forged")
    elif change == "root":
        root = root.parent
    else:
        grant = True
    with pytest.raises((ValueError, TypeError)):
        module.verify_full_data_preflight(root, grant, spec=requested)


def test_microseconds_are_losslessly_canonicalized_without_rewriting_input(full_request):
    module = api()
    root, spec = full_request
    source = next(s for s in spec.sources if s.role == "data")
    path = root / source.path
    pl.read_parquet(path).with_columns(
        (pl.lit(1_700_000_000_000_000) + pl.int_range(pl.len())).cast(pl.Datetime("us")).alias("timestamp")
    ).write_parquet(path)
    spec = replace(spec, sources=tuple(replace(s, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                      if s.role == "data" else s for s in spec.sources))
    grant = data_preflight(module, root, intent(module, spec))
    assert grant.timestamp_unit == "us"
    assert grant.canonical_timestamp_unit == "ns"
    assert pl.read_parquet_schema(path)["timestamp"] == pl.Datetime("us")


@pytest.mark.parametrize("change", ["baseline_model", "linkage", "scientific_blocker"])
def test_preflight_cannot_promote_rehashed_scientific_changes(full_request, change):
    import yaml
    module = api()
    root, spec = full_request
    if change == "baseline_model":
        relative = "configs/experiments/baselines.yaml"
        baseline = yaml.safe_load((root / relative).read_bytes())
        baseline["models"]["hist_gradient_boosting"]["params"]["early_stopping"] = True
        (root / relative).write_text(yaml.safe_dump(baseline))
        spec = replace(spec, sources=tuple(replace(s, sha256=hashlib.sha256((root / relative).read_bytes()).hexdigest())
                                          if s.path == relative else s for s in spec.sources))
    else:
        candidate = spec.candidates[0]
        if change == "linkage":
            candidate = replace(candidate, source_cell_id="cand-title")
        else:
            candidate = replace(candidate, spec=replace(candidate.spec, experiment_id="feat_core_vs_provided"))
        spec = replace(spec, candidates=(candidate,))
    with pytest.raises(ValueError):
        data_preflight(module, root, intent(module, spec))


def test_preflight_checks_every_outer_fold_before_first_fit(full_request):
    module = api()
    root, spec = full_request
    source = next(s for s in spec.sources if s.role == "data")
    path = root / source.path
    # Retain both global classes but remove class 1 from repeat0/fold0 only.
    pl.read_parquet(path).with_columns(
        pl.when(pl.col("building_id").is_in(["bldg_amber", "bldg_birch"]))
        .then(0).otherwise(pl.col("anomaly")).alias("anomaly")
    ).write_parquet(path)
    spec = replace(spec, sources=tuple(replace(s, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                      if s.role == "data" else s for s in spec.sources))
    with pytest.raises(ValueError, match="outer training/validation class support"):
        data_preflight(module, root, intent(module, spec))


def test_copied_partitions_are_not_independent_repeats(full_request):
    module = api()
    root, spec = full_request
    source = next(s for s in spec.sources if s.role == "split")
    path = root / source.path
    manifest = pl.read_parquet(path)
    reference = manifest.filter(pl.col("repeat") == 0).select("building_id", pl.col("fold").alias("copied"))
    manifest.join(reference, on="building_id").with_columns(
        pl.when(pl.col("repeat") == 2).then(pl.col("copied")).otherwise(pl.col("fold")).alias("fold")
    ).drop("copied").write_parquet(path)
    spec = replace(spec, sources=tuple(replace(s, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                      if s.role == "split" else s for s in spec.sources))
    with pytest.raises(ValueError, match="distinct"):
        data_preflight(module, root, intent(module, spec))


def test_full_launch_grant_stays_unavailable_without_integrated_worker(full_request):
    module = api()
    root, spec = full_request
    (root / "src/project/experiments/full_campaign_worker.py").unlink()
    with pytest.raises(ValueError, match="full_dispatch_unavailable"):
        module.preflight_full_campaign(root, intent(module, spec))


@pytest.mark.parametrize("relative", [
    "src/project/experiments/full_candidate_runner.py",
    "src/project/experiments/full_campaign_runtime.py",
    "src/project/experiments/full_candidate_io.py",
    "src/project/experiments/full_campaign_cli.py",
    "scripts/campaign.py",
    "src/project/experiments/full_candidate_worker.py",
    "src/project/experiments/full_campaign_evidence.py",
])
def test_data_evidence_binds_execution_closure_and_future_modules(full_request, relative):
    module = api()
    root, spec = full_request
    path = root / relative
    if not path.exists():
        path.write_text("# future route, not approval\n")
    evidence = data_preflight(module, root, intent(module, spec))
    assert relative in {source.path for source in evidence.implementation_sources}
    path.write_text("# changed execution closure\n")
    with pytest.raises(ValueError, match="implementation"):
        module.verify_full_data_preflight(root, evidence, spec=evidence.spec)


def test_canonical_data_evidence_builds_actual_fixture_index_and_packets(full_request):
    from project.experiments import full_candidate_io as io
    module = api()
    root, spec = full_request
    evidence = data_preflight(module, root, intent(module, spec))
    before = evidence.spec.binding_sha256
    index = io.build_development_index(root, evidence.spec, root / "canonical.sqlite3")
    packet = io.load_full_training_partition(index, evidence.spec, 0, 0)
    assert tuple(packet.X_train.columns) == CORE
    assert evidence.spec.binding_sha256 == before
    assert set(packet.y_train) == {0, 1}
    assert tuple(next(io.iter_full_validation_batches(index, evidence.spec, 0, 0))["X"].columns) == CORE


@pytest.mark.parametrize("column", ["meter_reading", "year_built", "primary_use", "building_id"])
def test_unsupported_input_dtypes_fail_before_data_evidence(full_request, column):
    module = api()
    root, spec = full_request
    source = next(s for s in spec.sources if s.role == "data")
    path = root / source.path
    frame = pl.read_parquet(path)
    if column == "building_id":
        frame = frame.with_columns(pl.col(column).str.len_chars().cast(pl.Int32).alias(column))
        split = next(s for s in spec.sources if s.role == "split")
        split_path = root / split.path
        pl.read_parquet(split_path).with_columns(pl.col(column).str.len_chars().cast(pl.Int32).alias(column)).write_parquet(split_path)
    else:
        frame = frame.with_columns(pl.col(column).cast(pl.String).str.split("").alias(column))
    frame.write_parquet(path)
    spec = replace(spec, sources=tuple(replace(s, sha256=hashlib.sha256((root / s.path).read_bytes()).hexdigest())
                                      if s.role in ("data", "split") else s for s in spec.sources))
    with pytest.raises(ValueError, match="dtype"):
        data_preflight(module, root, intent(module, spec))


@pytest.mark.parametrize("role", ["data", "split"])
def test_mutation_during_data_checks_invalidates_evidence(full_request, role, monkeypatch):
    module = api()
    root, spec = full_request
    source = next(s for s in spec.sources if s.role == role)
    original = module.pq.ParquetFile
    changed = False

    class MutatingParquetFile:
        def __init__(self, path):
            self.reader = original(path)

        def __getattr__(self, name):
            return getattr(self.reader, name)

        def iter_batches(self, **kwargs):
            nonlocal changed
            for batch in self.reader.iter_batches(**kwargs):
                yield batch
                if not changed:
                    # Mutate real fixture bytes after a projected read, while
                    # allowing the complete real preflight checks to continue.
                    path = root / source.path
                    frame = pl.read_parquet(path)
                    if role == "data":
                        frame = frame.with_columns(
                            pl.when(pl.col("building_id") == "bldg_amber")
                            .then(1 - pl.col("anomaly")).otherwise(pl.col("anomaly")).alias("anomaly")
                        )
                    else:
                        frame = frame.with_columns((pl.col("seed") + 1).alias("seed"))
                    frame.write_parquet(path)
                    changed = True

    monkeypatch.setattr(module.pq, "ParquetFile", MutatingParquetFile)
    with pytest.raises(ValueError, match="hash"):
        data_preflight(module, root, intent(module, spec))
    assert changed


def test_poisoned_vault_labels_do_not_enter_data_support_or_scoring_packets(full_request):
    from project.experiments import full_candidate_io as io
    module = api()
    root, spec = full_request
    source = next(s for s in spec.sources if s.role == "data")
    path = root / source.path
    pl.read_parquet(path).with_columns(
        pl.when(pl.col("building_id").is_in(["bldg_ivy", "bldg_jade"]))
        .then(999).otherwise(pl.col("anomaly")).alias("anomaly")
    ).write_parquet(path)
    spec = replace(spec, sources=tuple(replace(s, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                                      if s.role == "data" else s for s in spec.sources))
    evidence = data_preflight(module, root, intent(module, spec))
    index = io.build_development_index(root, evidence.spec, root / "poisoned.sqlite3")
    assert set(io.load_full_training_partition(index, evidence.spec, 0, 0).y_train) == {0, 1}
    for batch in io.iter_full_validation_batches(index, evidence.spec, 0, 0):
        assert set(batch["target"]) <= {0, 1}
        assert not {"bldg_ivy", "bldg_jade"} & set(batch["building_id"])


def test_versions_include_scientific_transitive_dependencies(full_request):
    module = api()
    assert {"python", "scipy", "joblib", "threadpoolctl", "scikit-learn", "pyarrow"} <= dict(module.current_environment()).keys()
