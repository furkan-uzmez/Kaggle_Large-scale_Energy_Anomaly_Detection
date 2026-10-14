"""CLI admission checks use isolated roots and fail-closed execution boundaries."""

import json
import runpy
from pathlib import Path

import pytest

from project.experiments import campaign_smoke

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cli():
    return runpy.run_path(str(ROOT / "scripts/campaign.py"))["main"]


def test_validate_only_design_is_blocked_without_writes(cli, tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("read-only validation reached the smoke launcher")

    monkeypatch.setitem(cli.__globals__, "launch_smoke", forbidden)
    monkeypatch.setitem(cli.__globals__, "referenced_file", lambda *args: config)
    config = tmp_path / "design.yaml"
    config.write_text("resource_budget: budget\nbaseline_config: baseline\nprediction_contract: {}\n")
    monkeypatch.setitem(cli.__globals__, "validate_admission", lambda *args: {
        "design_valid": True, "executable": False,
        "status": "blocked_unapproved_compute", "errors": [],
    })
    monkeypatch.setitem(cli.__globals__, "validate_split_contract", lambda *args: {"valid": True})
    assert cli(["--validate-only", "--project-root", str(tmp_path)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked_unapproved_compute"
    assert report["design_valid"] is True
    assert report["split_contract"]["valid"] is True
    assert list(tmp_path.iterdir()) == [config]
    with pytest.raises(SystemExit) as error:
        cli(["--validate-only", "--smoke"])
    assert error.value.code == 2
    assert "not allowed with argument --validate-only" in capsys.readouterr().err


def test_smoke_requires_explicit_versioned_config_and_experiment(cli, tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("missing explicit arguments reached the smoke launcher")

    monkeypatch.setitem(cli.__globals__, "launch_smoke", forbidden)
    assert cli(["--smoke", "--project-root", str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked_unapproved_compute"
    assert list(tmp_path.iterdir()) == []


def test_smoke_pending_review_does_not_launch_or_write(cli, tmp_path, monkeypatch, capsys):
    # Only small admission YAML metadata is read; no datasets or split files.
    read_yaml = campaign_smoke._read_yaml_hash
    settings, digest = read_yaml(ROOT, campaign_smoke.SMOKE_CONFIG, "smoke config")
    settings = dict(settings, launch_review_status="pending")

    def synthetic_loader(root, reference, label):
        assert root == tmp_path
        if reference == campaign_smoke.SMOKE_CONFIG:
            return settings, digest
        return read_yaml(ROOT, reference, label)

    def forbidden(*args, **kwargs):
        pytest.fail("pending review reached preflight, execution, or an output write")

    monkeypatch.setattr(campaign_smoke, "_read_yaml_hash", synthetic_loader)
    monkeypatch.setattr(campaign_smoke, "host_preflight", forbidden)
    monkeypatch.setattr(campaign_smoke, "run_smoke_guarded", forbidden)
    monkeypatch.setattr(campaign_smoke, "_write_json", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    assert cli([
        "--smoke", "--project-root", str(tmp_path),
        "--config", campaign_smoke.SMOKE_CONFIG,
        "--experiment", campaign_smoke.EXPERIMENT,
    ]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["config_valid"] is True
    assert report["executable"] is False
    assert report["status"] == "blocked_parent_review"
    assert report["errors"] == []
    assert list(tmp_path.iterdir()) == []
