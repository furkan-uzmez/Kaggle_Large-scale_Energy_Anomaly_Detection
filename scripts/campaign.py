"""Read-only campaign validation or explicitly reviewed bounded CPU SMOKE."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from project.experiments import full_campaign_cli
from project.experiments.campaign_budget import (
    validate_admission,
    validate_split_contract,
)
from project.experiments.campaign_ledger import referenced_file
from project.experiments.campaign_smoke import launch_smoke, validate_smoke_config
from project.experiments.candidate_admission import validate_candidate_admission
from project.experiments.candidate_contracts import CandidateSpec, FitBudget, RunSpec
from project.experiments.candidate_runner import run_synthetic_campaign


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--candidate-validate-only", action="store_true")
    modes.add_argument("--candidate-synthetic-smoke", action="store_true")
    modes.add_argument("--full-validate-only", action="store_true")
    modes.add_argument("--full-run", action="store_true")
    modes.add_argument("--full-resume", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config")
    parser.add_argument("--experiment")
    parser.add_argument("--full-retry-reason", help="Explicit reason for retrying proven-exited ordinary attempts")
    args = parser.parse_args(argv)
    if args.full_retry_reason is not None and not args.full_resume:
        parser.error("--full-retry-reason is only valid with --full-resume")
    root = args.project_root.resolve()
    try:
        if args.candidate_validate_only:
            exp_id = args.experiment or "feat_meter_deviation_ratio"
            family = "hist_gradient_boosting"
            if exp_id == "model_extratrees_core_features":
                family = "extra_trees"
            elif exp_id == "model_autogluon_core_features":
                family = "autogluon"
            if exp_id == "feat_preserve_raw_vs_invalid_sentinel":
                methods = (
                    ("rule", ("meter_reading", (-999.0,), -1.0, "docs/evidence/sentinel_synthetic.json", "0" * 64)),
                )
            elif exp_id == "regime_same_family_hpo_vs_default":
                methods = (("study_spec", (("n_trials", 2), ("inner_folds", 2))),)
            else:
                methods = ()
            fits = 16 if exp_id == "regime_same_family_hpo_vs_default" else 8
            trials = 4 if exp_id == "regime_same_family_hpo_vs_default" else 0
            budget = FitBudget(fits, trials, 30.0, 256 * 1024**2, 1)
            core_cols = (
                "meter_reading", "square_feet", "year_built", "floor_count", "hour",
                "weekday", "month", "year", "primary_use",
            )
            run_spec = RunSpec(
                1,
                (CandidateSpec(exp_id, core_cols, family, (), methods),),
                budget,
                "synthetic_smoke",
                "notebooks/02_baseline_and_candidate_design.ipynb",
                "cell-cli",
            )
            campaign_path = referenced_file(root, args.config or "configs/experiments/feature_campaign.yaml", "campaign")
            campaign = yaml.safe_load(campaign_path.read_text())
            campaign_sha256 = hashlib.sha256(
                json.dumps(campaign, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            source_design = {
                "campaign": campaign,
                "campaign_sha256": campaign_sha256,
                "candidate_links": {
                    exp_id: {
                        "source_notebook": run_spec.source_notebook,
                        "source_cell_id": run_spec.source_cell_id,
                        "campaign_sha256": campaign_sha256,
                    }
                },
                "diagnostic": {
                    "generator": "candidate_fixtures_v1",
                    "max_rows": 16,
                    "groups": 4,
                    "folds": 2,
                    "repeats": 1,
                    "cpu_threads": 1,
                },
            }
            report = validate_candidate_admission(run_spec, source_design, budget)
            print(json.dumps(report.to_dict(), allow_nan=False, sort_keys=True))
            return 0 if report.diagnostic_ready else 2

        if args.candidate_synthetic_smoke:
            exp_id = args.experiment or "feat_meter_deviation_ratio"
            family = "hist_gradient_boosting"
            if exp_id == "model_extratrees_core_features":
                family = "extra_trees"
            elif exp_id == "model_autogluon_core_features":
                family = "autogluon"
            budget = FitBudget(8, 0, 30.0, 256 * 1024**2, 1)
            core_cols = (
                "meter_reading", "square_feet", "year_built", "floor_count", "hour",
                "weekday", "month", "year", "primary_use",
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                methods = ()
                if exp_id == "feat_preserve_raw_vs_invalid_sentinel":
                    evidence_rel = "docs/evidence/sentinel_synthetic.json"
                    evidence_file = temp_path / evidence_rel
                    evidence_file.parent.mkdir(parents=True, exist_ok=True)
                    payload = b'{"invalid_values": [-999.0], "column": "meter_reading"}'
                    evidence_file.write_bytes(payload)
                    digest = hashlib.sha256(payload).hexdigest()
                    methods = (
                        ("rule", ("meter_reading", (-999.0,), -1.0, evidence_rel, digest)),
                        ("evidence_root", str(temp_path)),
                    )
                run_spec = RunSpec(
                    1,
                    (CandidateSpec(exp_id, core_cols, family, (), methods),),
                    budget,
                    "synthetic_smoke",
                    "notebooks/02_baseline_and_candidate_design.ipynb",
                    "cell-cli",
                )
                result = run_synthetic_campaign(run_spec, temp_path)
                print(json.dumps(asdict(result), allow_nan=False, sort_keys=True))
                return 0 if result.status == "completed" else 2

        if args.full_validate_only:
            if not args.config:
                report = {"status": "blocked_missing_config",
                          "errors": ["--full-validate-only requires a concrete --config full-intent snapshot"]}
                print(json.dumps(report, allow_nan=False, sort_keys=True))
                return 2
            report = full_campaign_cli.inspect(args.config, root)
            print(json.dumps(report, allow_nan=False, sort_keys=True))
            return 0 if report["status"] == "ready_for_ordinary_transport" else 2

        if args.full_run or args.full_resume:
            if not args.config:
                report = {"status": "blocked_missing_config",
                          "errors": ["--full-run/--full-resume require a concrete --config full-intent snapshot"]}
                print(json.dumps(report, allow_nan=False, sort_keys=True))
                return 2
            report = full_campaign_cli.launch(args.config, root, resume=args.full_resume,
                                              retry_reason=args.full_retry_reason)
            print(json.dumps(report, allow_nan=False, sort_keys=True))
            return 2

        if args.smoke:
            if not args.config or not args.experiment:
                report = {"returncode": 2, "status": "blocked_unapproved_compute",
                          "errors": ["--smoke requires explicit --config and --experiment"]}
            else:
                report = launch_smoke(root, args.config, args.experiment)
            print(json.dumps(report, allow_nan=False, sort_keys=True))
            return report["returncode"]
        campaign = yaml.safe_load(referenced_file(root, args.config or "configs/experiments/feature_campaign.yaml", "campaign").read_text())
        if isinstance(campaign, dict) and campaign.get("scope") == "campaign_smoke":
            report = validate_smoke_config(root, campaign, args.experiment or "")
            print(json.dumps(report, allow_nan=False, sort_keys=True))
            return 0 if report["executable"] else 2
        budget = yaml.safe_load(referenced_file(root, campaign["resource_budget"], "budget").read_text())
        baseline = yaml.safe_load(referenced_file(root, campaign["baseline_config"], "baseline config").read_text())
        report = validate_admission(campaign, budget, baseline)
        report["split_contract"] = validate_split_contract(root, campaign["prediction_contract"])
        if not report["split_contract"]["valid"]:
            report["executable"] = False
            report["errors"].extend(report["split_contract"]["errors"])
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        report = {"design_valid": False, "executable": False,
                  "status": "rejected_by_gate", "errors": [str(error)],
                  "execution_authorized": False, "n03_eligible": False}
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0 if report["executable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
