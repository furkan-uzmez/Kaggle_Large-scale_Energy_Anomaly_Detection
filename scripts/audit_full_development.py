"""Read-only development audit CLI; no launch/admission or replacement selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from project.experiments.full_development_audits import (
    AuditPlan,
    audit_full_development,
    write_audit_evidence,
)
from project.experiments.full_snapshot_codec import decode_snapshot

# Single pinned authority consumed by the readiness gate. A fresh immutable
# version is published by passing an explicit ``--destination``; the default
# never silently replaces an already-pinned version.
DEFAULT_EVIDENCE_PATH = "docs/evidence/full_development_audit_v3.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--snapshot", type=Path, required=True, help="Authenticated full snapshot codec JSON")
    parser.add_argument("--destination", default=DEFAULT_EVIDENCE_PATH,
                        help="Evidence path relative to --root (or absolute); never overwritten")
    parser.add_argument("--max-development-rows", type=int, default=1_399_631)
    parser.add_argument("--max-invalid-values", type=int, default=256)
    parser.add_argument("--inner-splits", type=int, nargs="+", default=[2, 3, 4, 5])
    args = parser.parse_args()
    spec = decode_snapshot(json.loads(args.snapshot.read_text()))
    plan = AuditPlan(tuple(args.inner_splits), 42, args.max_development_rows, args.max_invalid_values)
    report = audit_full_development(args.root, spec, plan)
    destination = Path(args.destination)
    if not destination.is_absolute():
        destination = args.root / destination
    digest = write_audit_evidence(destination, report)
    print(json.dumps({"evidence": str(destination), "sha256": digest,
                      "execution_authorized": False}, allow_nan=False))


if __name__ == "__main__":
    main()
