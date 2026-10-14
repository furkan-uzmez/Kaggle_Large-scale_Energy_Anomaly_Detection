"""Launch denial must precede all filesystem and fitting side effects."""

import pytest

from tests.full_campaign_fixtures import write_fixture
from project.experiments import full_candidate_runner as runner
from tests.unit.test_full_campaign_preflight import full_request  # noqa: F401


def test_raw_spec_cannot_create_run_directory(tmp_path, monkeypatch):
    fixture = write_fixture(tmp_path / "source")
    output = tmp_path / "must-not-exist"

    def forbidden(*args, **kwargs):
        pytest.fail("index or fit reached before authorization")

    monkeypatch.setattr(runner.io, "build_development_index", forbidden)
    with pytest.raises((TypeError, ValueError), match="[Pp]reflight|evidence|capability"):
        runner.run_full_campaign(fixture.spec, fixture.root, output)
    assert not output.exists()


def test_capability_resolver_never_promotes_in_process_runner(tmp_path):
    from project.experiments.full_campaign_capabilities import resolve_full_campaign_capability

    result = resolve_full_campaign_capability(tmp_path)
    assert result.available is False
    assert "full_dispatch_unavailable" in result.blocker_codes
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("change", ["none", "forged", "stale", "capability"])
def test_data_evidence_alone_never_launches(full_request, tmp_path, monkeypatch, change):  # noqa: F811
    from dataclasses import replace
    from project.experiments.full_campaign_admission import FullCampaignIntent
    from project.experiments.full_campaign_capabilities import FullCampaignCapability, resolve_full_campaign_capability
    from project.experiments.full_campaign_preflight import current_environment, preflight_full_campaign_data

    root, spec = full_request
    spec = replace(spec, environment=current_environment())
    evidence = preflight_full_campaign_data(root, FullCampaignIntent(
        spec, True, resource_opt_out_record="existing reviewed test opt-out"))
    capability = resolve_full_campaign_capability(root)
    if change == "forged":
        evidence = replace(evidence, seal="0" * 64)
    elif change == "stale":
        source = next(s for s in spec.sources if s.role == "data")
        (root / source.path).write_bytes(b"changed")
    elif change == "capability":
        capability = FullCampaignCapability(str(root), available=True, blocker_codes=())
    output = tmp_path / "denied-output"
    before = set(root.rglob("*"))

    def forbidden(*args, **kwargs):
        pytest.fail("denied input reached index or fit")

    monkeypatch.setattr(runner.io, "build_development_index", forbidden)
    with pytest.raises((TypeError, ValueError)):
        runner.run_full_campaign(spec, root, output, data_preflight=evidence, capability=capability)
    assert not output.exists()
    assert set(root.rglob("*")) == before


def test_cli_does_not_manufacture_resource_consent(tmp_path):
    import yaml
    from project.experiments.full_campaign_cli import load_full_spec

    path = tmp_path / "intent.yaml"
    path.write_text(yaml.safe_dump({"run_id": "test", "candidates": [], "sources": {}, "environment": {}}))
    with pytest.raises(ValueError, match="resource_opt_out_record"):
        load_full_spec(path, tmp_path)


@pytest.mark.parametrize("mode", ["--full-run", "--full-resume", "--full-validate-only"])
def test_cli_unavailable_does_not_lock_or_run(tmp_path, monkeypatch, capsys, mode):
    import json
    import scripts.campaign as cli

    def forbidden(*args, **kwargs):
        pytest.fail("unavailable CLI reached launch")

    monkeypatch.setattr(cli.full_campaign_cli, "inspect", lambda *args: {
        "status": "data_ready_execution_unavailable", "execution_authorized": False,
        "blocker_codes": ["full_dispatch_unavailable"],
    })
    monkeypatch.setattr(cli, "run_full_campaign", forbidden, raising=False)
    assert cli.main([mode, "--config", "intent.yaml", "--project-root", str(tmp_path)]) == 2
    assert json.loads(capsys.readouterr().out)["execution_authorized"] is False
    assert not list(tmp_path.iterdir())
