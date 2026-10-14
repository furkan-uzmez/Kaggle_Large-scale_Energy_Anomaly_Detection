"""Full-campaign CLI-mode tests: config required, read-only validate, no defaults."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import scripts.campaign as campaign_cli
import yaml

from full_campaign_fixtures import write_admission_fixture


def fixture_cli_config(root):
    fixture = write_admission_fixture(root)
    snapshot = fixture.spec.candidates[0]
    candidate = asdict(snapshot.spec)
    candidate["model_params"] = dict(snapshot.spec.model_params)
    candidate["method_params"] = dict(snapshot.spec.method_params)
    candidate.update(arms=list(snapshot.arms), source_notebook=snapshot.source_notebook,
                     source_cell_id=snapshot.source_cell_id)
    document = {
        "run_id": fixture.spec.run_id, "candidates": [candidate],
        "sources": {binding.role: binding.path for binding in fixture.spec.sources},
        "environment": dict(fixture.spec.environment),
        "resource_opt_out_record": "user: fixture only acceptance",
        "development_rows": fixture.spec.development_rows, "vault_rows": fixture.spec.vault_rows,
        "run_directory": "outputs/fixture-campaign",
    }
    config = fixture.root / "fixture-intent.yaml"
    config.write_text(yaml.safe_dump(document))
    return fixture, config


def test_full_run_uses_genuine_admission_and_actual_workers(tmp_path, capsys):
    fixture, config = fixture_cli_config(tmp_path / "source")
    result = campaign_cli.main(["--full-run", "--project-root", str(fixture.root), "--config", str(config)])
    report = json.loads(capsys.readouterr().out)
    assert (fixture.root / "outputs/fixture-campaign/spec.json").is_file(), report
    assert result == 2  # Real fixture execution is not comparative completion.
    assert report["status"] == "finished_with_blockers"
    assert report["experiments"][0]["status"] == "completed"
    assert len(report["inventory"]) == 21
    assert report["n03_eligible"] is False
    run = fixture.root / "outputs/fixture-campaign"
    before = {str(path): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    resumed = campaign_cli.main(["--full-resume", "--project-root", str(fixture.root), "--config", str(config)])
    assert resumed == 2  # All21 omissions remain blockers, not fake campaign completion.
    assert json.loads(capsys.readouterr().out) == report
    assert {str(path): path.read_bytes() for path in run.rglob("*") if path.is_file()} == before


def test_resume_cli_relays_explicit_retry_reason(monkeypatch, tmp_path, capsys):
    calls = []

    def launch(config, root, **kwargs):
        calls.append(kwargs)
        return {"status": "finished_with_blockers"}

    monkeypatch.setattr(campaign_cli.full_campaign_cli, "launch", launch)
    result = campaign_cli.main(["--full-resume", "--config", "fixture.yaml", "--project-root", str(tmp_path),
                                "--full-retry-reason", "user: explicit owned-attempt retry"])
    assert result == 2
    assert calls == [{"resume": True, "retry_reason": "user: explicit owned-attempt retry"}]


def test_full_validate_genuine_fixture_is_readonly(tmp_path, capsys):
    fixture, config = fixture_cli_config(tmp_path / "source")
    before = sorted(str(path) for path in fixture.root.rglob("*"))
    result = campaign_cli.main(["--full-validate-only", "--project-root", str(fixture.root), "--config", str(config)])
    report = json.loads(capsys.readouterr().out)
    assert result == 0, report
    assert report["status"] == "ready_for_ordinary_transport"
    assert report["execution_authorized"] is False
    assert sorted(str(path) for path in fixture.root.rglob("*")) == before


def test_full_validate_requires_config(capsys):
    rc = campaign_cli.main(["--full-validate-only"])
    captured = capsys.readouterr()
    assert rc == 2
    report = json.loads(captured.out)
    assert report["status"] == "blocked_missing_config"


def test_full_run_requires_config(capsys):
    rc = campaign_cli.main(["--full-run"])
    assert rc == 2
    assert "blocked_missing_config" in capsys.readouterr().out


def test_full_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        campaign_cli.main(["--full-run", "--full-validate-only"])
    with pytest.raises(SystemExit):
        campaign_cli.main(["--full-run", "--candidate-validate-only"])


def test_existing_modes_still_work(capsys):
    rc = campaign_cli.main(["--candidate-validate-only", "--experiment", "feat_meter_deviation_ratio"])
    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out)["design_valid"] is True
