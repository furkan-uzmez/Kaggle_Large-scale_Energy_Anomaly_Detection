"""Confined, locked, append-only campaign proposal and review evidence."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from project.features.availability import (
    select_predictor_columns,
    validate_feature_availability,
)
from project.features.registry import FeatureMetadata, FeatureRegistry

DISPOSITIONS = {"planned", "blocked", "excluded"}
OUTCOMES = {"completed", "failed", "rejected_by_gate", "blocked"}
LEDGER_REFERENCE = "outputs/runs/campaign_ledger.jsonl"
CAMPAIGN_REFERENCE = "configs/experiments/feature_campaign.yaml"
NOTEBOOK_REFERENCES = {"notebooks/01_data_understanding.ipynb",
                       "notebooks/02_baseline_and_candidate_design.ipynb"}
MAX_BATCH_BYTES = 4 * 1024 * 1024
MAX_HISTORY_BYTES = 16 * 1024 * 1024
NON_PREDICTORS = {"anomaly", "target", "label", "y", "id", "building_id", "fold",
                  "repeat", "seed", "split", "row_index", "support_stratum"}


class CampaignValidationError(ValueError):
    """Invalid campaign evidence, including schema and provenance violations."""


def referenced_file(root: Path, reference: Any, label: str) -> Path:
    """Resolve existing relative project evidence; reject symlink components."""
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"{label} reference is required")
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} reference must be project-relative")
    path = root.resolve()
    for component in relative.parts:
        path /= component
        if path.is_symlink():
            raise ValueError(f"{label} reference must not follow symlink")
    if not path.is_file():
        raise ValueError(f"{label} reference must be an existing project file")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reserved(name: str) -> bool:
    name = name.casefold()
    return name in NON_PREDICTORS or name.endswith("_id") or name.startswith(("source_", "fold_"))


def validate_predictors(predictors: Any, registry: FeatureRegistry,
                        prediction_time: str | None = None) -> None:
    """Use the project's typed lineage and scoring-time availability contract."""
    if not isinstance(registry, FeatureRegistry):
        raise CampaignValidationError("predictor registry must be authoritative typed FeatureRegistry")
    if not isinstance(predictors, list) or any(not isinstance(p, str) for p in predictors):
        raise ValueError("predictors must be a list of registered names")
    if len(predictors) != len(set(predictors)):
        raise ValueError("duplicate predictor")
    if not predictors:
        return
    try:
        scoring_time = datetime.fromisoformat(prediction_time)
    except (ValueError, TypeError) as error:
        raise ValueError("predictor prediction_time must be an ISO datetime") from error
    if scoring_time.tzinfo is None:
        raise ValueError("predictor prediction_time must be timezone-aware")
    for predictor in predictors:
        meta = registry.get(predictor)
        if not isinstance(meta, FeatureMetadata):
            raise CampaignValidationError(f"unregistered predictor: {predictor}")
        if (_reserved(predictor) or any(_reserved(c) for c in meta.source_columns)
                or meta.family.casefold().replace("-", "_") in
                {"identifier", "target", "source_metadata", "grouping", "split"}):
            raise ValueError(f"reserved predictor or source lineage: {predictor}")
        validate_feature_availability(meta, scoring_time)
    if select_predictor_columns(predictors, registry, scoring_time) != tuple(predictors):
        raise ValueError("predictor selection excludes submitted columns")


@contextmanager
def _locked_ledger(root: Path):
    """Anchor opens to no-follow directory FDs and lock the ledger inode."""
    descriptors = []
    try:
        parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(parent)
        for name in ("outputs", "runs"):
            try:
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            except FileNotFoundError:
                os.mkdir(name, dir_fd=parent)
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            descriptors.append(child)
            parent = child
        fd = os.open("campaign_ledger.jsonl", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                     0o600, dir_fd=parent)
        descriptors.append(fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_nlink != 1:
            raise ValueError("ledger must be a regular, non-hardlinked file")
        fcntl.flock(fd, fcntl.LOCK_EX)
        def verify_identity():
            for parent_fd, child_fd, name in zip(descriptors, descriptors[1:],
                                                 ("outputs", "runs", "campaign_ledger.jsonl"), strict=False):
                opened = os.fstat(child_fd)
                current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
                    raise ValueError("ledger confinement path changed while locked")

        verify_identity()
        yield fd, verify_identity
    except OSError as error:
        if error.errno in {40, 20}:
            raise ValueError("ledger confinement forbids symlink/follow directory opens") from error
        raise
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


class CampaignLedger:
    """Separate proposal submissions and subsequent review events.

    Each proposal_id is unique. Review event_id is unique across all events and
    links an existing completed proposal_id/run_id. No history row is rewritten.
    Validation is atomic under an exclusive advisory lock shared by API writers.
    Handled write/fsync errors truncate to the previous offset under that lock;
    success fsyncs the file. Power-loss/process-kill transactional durability is
    not promised: malformed/truncated history blocks further appends for recovery.
    """

    def __init__(self, path: Path, root: Path, campaign: Mapping[str, Any]) -> None:
        self.root = Path(root).resolve()
        self.path = Path(os.path.abspath(path))
        if self.path != self.root / LEDGER_REFERENCE:
            raise ValueError(f"ledger confinement requires {LEDGER_REFERENCE}")
        for component in (self.root / "outputs", self.path.parent, self.path):
            if component.is_symlink():
                raise ValueError("ledger confinement forbids symlink")
        matrix = campaign.get("experiment_matrix") if isinstance(campaign, Mapping) else None
        if not isinstance(matrix, list) or any(not isinstance(r, Mapping) for r in matrix):
            raise ValueError("experiment matrix must contain mappings")
        self.experiments = {}
        for row in matrix:
            identity = row.get("id")
            if not isinstance(identity, str) or not identity.strip() or identity in self.experiments:
                raise ValueError("invalid or duplicate experiment IDs in matrix")
            self.experiments[identity] = dict(row)
        self.campaign_id = campaign.get("campaign_id")

    def append(self, records: Iterable[Mapping[str, Any]], registry: FeatureRegistry) -> None:
        """Validate current history and complete batch under lock, then append."""
        records = [dict(record) if isinstance(record, Mapping) else record for record in records]
        if not records:
            return
        for record in records:
            self.validate(record, registry)
        payload = ("\n".join(json.dumps(row, allow_nan=False, sort_keys=True)
                             for row in records) + "\n").encode("utf-8")
        if len(payload) > MAX_BATCH_BYTES:
            raise ValueError("campaign ledger batch exceeds bounded append size")
        with _locked_ledger(self.root) as (fd, verify_identity):
            size = os.fstat(fd).st_size
            if size + len(payload) > MAX_HISTORY_BYTES:
                raise ValueError("campaign ledger exceeds bounded history size")
            os.lseek(fd, 0, os.SEEK_SET)
            raw = bytearray()
            while chunk := os.read(fd, min(65536, size - len(raw))):
                raw.extend(chunk)
            if raw and not raw.endswith(b"\n"):
                raise ValueError("history is truncated; explicit recovery required")
            try:
                history = [json.loads(line) for line in raw.splitlines()]
                self._validate_sequence(history, registry)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"invalid campaign ledger history: {error}") from error
            self._validate_sequence(history + records, registry)
            verify_identity()
            os.lseek(fd, size, os.SEEK_SET)
            try:
                remaining = memoryview(payload)
                while remaining:
                    written = os.write(fd, remaining)
                    if written <= 0:
                        raise OSError("ledger append made no progress")
                    remaining = remaining[written:]
                os.fsync(fd)
            except OSError:
                os.ftruncate(fd, size)
                os.fsync(fd)
                raise

    def _validate_sequence(self, records: list, registry: FeatureRegistry) -> None:
        seen, proposals, reviewed = set(), {}, set()
        for record in records:
            self.validate(record, registry)
            event_type = record.get("event_type", "proposal")
            identity = record["event_id"] if event_type == "review" else record["proposal_id"]
            if identity in seen:
                raise ValueError(f"duplicate submitted event/proposal ID: {identity}")
            seen.add(identity)
            if event_type == "proposal":
                proposals[record["proposal_id"]] = record
                continue
            original = proposals.get(record["proposal_id"])
            if (original is None or original.get("outcome") != "completed"
                    or original["experiment_id"] != record["experiment_id"]
                    or original["run_id"] != record["run_id"]):
                raise ValueError("review lifecycle requires linked completed proposal/run")
            if record["proposal_id"] in reviewed:
                raise ValueError("review lifecycle already finalized")
            reviewed.add(record["proposal_id"])

    def validate(self, record: Mapping[str, Any], registry: FeatureRegistry) -> None:
        """Validate source/config lineage, typed predictors, and run evidence."""
        if not isinstance(record, Mapping):
            raise CampaignValidationError("ledger record must be a mapping")
        for key in ("proposal_id", "experiment_id", "reason"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"{key} must be nonempty")
        if record["experiment_id"] not in self.experiments:
            raise ValueError("unknown experiment ID")
        event_type = record.get("event_type", "proposal")
        if event_type == "review":
            self._validate_review(record)
            return
        if event_type != "proposal":
            raise ValueError("invalid event_type")
        if not isinstance(record.get("criteria"), str) or not record["criteria"].strip():
            raise ValueError("criteria must be nonempty")
        if record.get("disposition") not in DISPOSITIONS:
            raise ValueError("invalid disposition; use excluded with reason for N/A")
        experiment = self.experiments[record["experiment_id"]]
        if experiment.get("status") == "N/A" and record["disposition"] != "excluded":
            raise ValueError("N/A experiment must remain excluded")
        if experiment.get("execution_status") == "blocked" and record["disposition"] == "planned":
            raise ValueError("blocked experiment requires blocked/excluded disposition")
        self._validate_config(record)
        self._validate_source(record.get("source_reference"))
        if record.get("disposition") != "excluded" and not record.get("predictors"):
            raise ValueError("applicable proposal requires predictors")
        validate_predictors(record.get("predictors"), registry, record.get("prediction_time"))
        outcome = record.get("outcome")
        if outcome is not None and outcome not in OUTCOMES:
            raise ValueError("invalid run outcome")
        if outcome is not None and (not isinstance(record.get("run_id"), str) or not record["run_id"].strip()):
            raise ValueError("completion/run outcome requires run ID")
        if outcome == "completed":
            if record["disposition"] != "planned":
                raise ValueError("completion cannot belong to blocked/excluded proposal")
            self._validate_completion(record)
        elif record.get("artifacts") or record.get("run_reference"):
            raise ValueError("completion artifacts require completed run evidence")
        if outcome is None and record.get("run_id") is not None:
            raise ValueError("run ID requires a run outcome")

    def _validate_source(self, source: Any) -> None:
        if not isinstance(source, Mapping) or source.get("path") not in NOTEBOOK_REFERENCES:
            raise ValueError("source must reference Notebook01/02")
        path = referenced_file(self.root, source["path"], "source")
        cell_id = source.get("cell_id")
        notebook = json.loads(path.read_text())
        if (not isinstance(cell_id, str) or not cell_id.strip() or
                not isinstance(notebook, Mapping) or not isinstance(notebook.get("cells"), list) or
                cell_id not in {cell.get("id") for cell in notebook["cells"] if isinstance(cell, Mapping)}):
            raise ValueError("source notebook cell does not exist")

    def _validate_config(self, record: Mapping[str, Any]) -> None:
        path = referenced_file(self.root, record.get("config_reference"), "config")
        if path.parent != self.root / "configs/experiments" or path.suffix not in {".yaml", ".yml"}:
            raise ValueError("config must be campaign YAML or linked versioned experiment config")
        config = yaml.safe_load(path.read_text())
        if not isinstance(config, Mapping):
            raise CampaignValidationError("config must be a mapping")
        if record["config_reference"] == CAMPAIGN_REFERENCE:
            matrix = config.get("experiment_matrix")
            if (config.get("campaign_id") != self.campaign_id or not isinstance(matrix, list)
                    or self.experiments[record["experiment_id"]] not in matrix):
                raise ValueError("campaign config does not link actual experiment")
        elif (config.get("experiment_id") != record["experiment_id"]
              or type(config.get("version")) is not int or config["version"] <= 0):
            raise ValueError("versioned config must link actual matrix experiment")

    def _validate_completion(self, record: Mapping[str, Any]) -> None:
        expected = f"outputs/runs/{record['run_id']}/run_status.json"
        if record.get("run_reference") != expected:
            raise ValueError("completion requires canonical run status linkage")
        path = referenced_file(self.root, expected, "completion")
        if record.get("run_status_sha256") != file_sha256(path):
            raise ValueError("completion run status hash mismatch")
        status = json.loads(path.read_text())
        if not isinstance(status, Mapping) or any(status.get(key) != value for key, value in {
            "run_id": record["run_id"], "experiment_id": record["experiment_id"], "status": "completed",
        }.items()):
            raise ValueError("completion run linkage mismatch")
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("completion requires hashed artifacts")
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise CampaignValidationError("artifact must be a mapping")
            artifact_path = referenced_file(self.root, artifact.get("path"), "artifact")
            if not artifact_path.is_relative_to(path.parent) or artifact_path == path:
                raise ValueError("artifact must belong to linked run directory")
            if artifact.get("sha256") != file_sha256(artifact_path):
                raise ValueError("artifact hash mismatch")
        review = record.get("review")
        if review is not None and (not isinstance(review, Mapping) or review.get("status") != "pending"):
            raise ValueError("completion review may be pending; reviewed requires linked review event")

    def _validate_review(self, record: Mapping[str, Any]) -> None:
        for key in ("event_id", "run_id"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"review {key} must be nonempty")
        review = record.get("review")
        if not isinstance(review, Mapping) or review.get("status") != "reviewed":
            raise ValueError("review event requires reviewed evidence")
        path = referenced_file(self.root, review.get("reference"), "review")
        if not path.is_relative_to(self.root / "outputs/runs" / record["run_id"]):
            raise ValueError("review must belong to linked run")
        evidence = json.loads(path.read_text())
        if not isinstance(evidence, Mapping) or any(evidence.get(key) != record[key] for key in ("run_id", "experiment_id")):
            raise ValueError("review evidence linkage mismatch")
        if evidence.get("status") != "reviewed":
            raise ValueError("review evidence status mismatch")
