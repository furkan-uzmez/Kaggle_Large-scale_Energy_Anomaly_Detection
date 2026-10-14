"""Unit tests for campaign CLI candidate extensions (Task 12)."""

import json
from pathlib import Path

import pytest
import scripts.campaign as campaign_cli


def test_cli_candidate_validate_only_success(capsys):
    rc = campaign_cli.main(["--candidate-validate-only", "--experiment", "feat_meter_deviation_ratio"])
    captured = capsys.readouterr()
    assert rc == 0
    report = json.loads(captured.out)
    assert report["design_valid"] is True
    assert report["diagnostic_ready"] is True
    assert report["executed"] is False


def test_cli_candidate_validate_only_blocked(capsys):
    rc = campaign_cli.main(["--candidate-validate-only", "--experiment", "regime_same_family_hpo_vs_default"])
    captured = capsys.readouterr()
    assert rc == 2
    report = json.loads(captured.out)
    assert report["diagnostic_ready"] is False
    blocker_codes = [b["code"] for b in report.get("blockers", [])]
    assert "hpo_not_ready" in blocker_codes


def test_cli_candidate_synthetic_smoke_success(tmp_path: Path, capsys, monkeypatch):
    # Verify synthetic smoke executes, succeeds, and NEVER writes to project outputs/
    project_root = Path(__file__).resolve().parents[2]
    outputs_dir = project_root / "outputs"
    before_files = set(outputs_dir.glob("**/*")) if outputs_dir.exists() else set()

    rc = campaign_cli.main(["--candidate-synthetic-smoke", "--experiment", "feat_meter_deviation_ratio"])
    captured = capsys.readouterr()
    assert rc == 0
    result = json.loads(captured.out)
    assert result["status"] == "completed"
    assert result["non_comparative"] is True
    assert result["n03_eligible"] is False

    after_files = set(outputs_dir.glob("**/*")) if outputs_dir.exists() else set()
    # No new files created in outputs_dir
    assert after_files == before_files


def test_cli_mutual_exclusivity():
    with pytest.raises(SystemExit):
        campaign_cli.main(["--candidate-validate-only", "--candidate-synthetic-smoke"])

    with pytest.raises(SystemExit):
        campaign_cli.main(["--candidate-validate-only", "--smoke"])

    with pytest.raises(SystemExit):
        campaign_cli.main(["--candidate-synthetic-smoke", "--validate-only"])

    with pytest.raises(SystemExit):
        campaign_cli.main(["--smoke", "--validate-only"])


def test_cli_existing_flags_unbroken(capsys):
    # Old smoke requires explicit config and experiment
    rc = campaign_cli.main(["--smoke"])
    captured = capsys.readouterr()
    assert rc == 2
    report = json.loads(captured.out)
    assert report["status"] == "blocked_unapproved_compute"
