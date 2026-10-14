"""Immutable full-run contracts and durable accounting (no fitting or launch)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Literal, Self

if TYPE_CHECKING:
    from project.experiments.full_campaign_admission import FullAdmissionRow

from project.experiments.candidate_contracts import (
    CandidateSpec,
    StudySpec,
    _integer,
    _is_sha256,
    _pairs,
    _text,
    _validate_managed_path,
)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _freeze_json(value: object) -> object:
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _relative(value: str) -> None:
    _text(value, "relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ValueError("path must be canonical and confined")


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fsync_directory_chain(directory: Path, ancestor: Path) -> None:
    """Persist child entries bottom-up through the containing ancestor."""
    directory = _validate_managed_path(directory, "durable directory", root=ancestor, directory=True)
    while True:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if directory == ancestor:
            return
        directory = directory.parent


@dataclass(frozen=True)
class FullResourcePolicy:
    informed_opt_out: bool
    rss_limit_enforced: bool = False
    max_rss_bytes: None = None
    cpu_limit_enforced: bool = False
    cpu_threads: None = None
    wall_limit_enforced: bool = False
    max_wall_seconds: None = None

    def __post_init__(self) -> None:
        for name in ("rss_limit_enforced", "cpu_limit_enforced", "wall_limit_enforced"):
            if getattr(self, name) is not False:
                raise ValueError("full resource enforcement flags must be exact false")
        for name in ("max_rss_bytes", "cpu_threads", "max_wall_seconds"):
            if getattr(self, name) is not None:
                raise ValueError("opted-out resource values must be null")
        if self.informed_opt_out is not True:
            raise ValueError("explicit informed opt-out is required")


@dataclass(frozen=True)
class SourceBinding:
    role: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _text(self.role, "source role")
        _relative(self.path)
        if not _is_sha256(self.sha256):
            raise ValueError("source hash must be lowercase SHA-256")

    def verify(self, root: Path) -> Path:
        path = _validate_managed_path(root / self.path, "source", root=root)
        if _file_hash(path) != self.sha256:
            raise ValueError(f"source hash mismatch: {self.path}")
        return path


@dataclass(frozen=True)
class FullCandidateSnapshot:
    spec: CandidateSpec
    arms: tuple[str, ...]
    source_notebook: str
    source_cell_id: str
    study: StudySpec | None = None
    study_method: str | None = None
    selector_arms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.spec, CandidateSpec):
            raise TypeError("snapshot requires CandidateSpec")
        for names in (self.arms, self.selector_arms):
            if not isinstance(names, tuple) or len(set(names)) != len(names):
                raise ValueError("arms must be unique immutable tuples")
            for name in names:
                if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                    raise ValueError("invalid arm identity")
        if not self.arms or not set(self.selector_arms) <= set(self.arms):
            raise ValueError("selector arms must belong to resolved arms")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.spec.experiment_id):
            raise ValueError("invalid experiment identity")
        _relative(self.source_notebook)
        _text(self.source_cell_id, "source cell")
        if self.source_cell_id.startswith("cell-cli"):
            raise ValueError("placeholder source cell is not evidence")
        hpo = self.spec.experiment_id == "regime_same_family_hpo_vs_default"
        if hpo != isinstance(self.study, StudySpec):
            raise ValueError("HPO requires explicit finite StudySpec; other recipes cannot have studies")
        if hpo and self.study_method != "random_search":
            raise ValueError("explicit supported study method required; admission owns method approval")
        if not hpo and self.study_method is not None:
            raise ValueError("study method without study")

    @property
    def planned_units(self) -> tuple[str, ...]:
        units = []
        for repeat in range(3):
            for fold in range(4):
                prefix = f"{self.spec.experiment_id}"
                if self.study is not None:
                    if self.arms != ("control", "challenger"):
                        raise ValueError("HPO requires exact matched control/challenger arms")
                    units.append(f"{prefix}/control/r{repeat}/f{fold}/default_refit")
                    for trial in range(self.study.n_trials):
                        for inner in range(self.study.inner_folds):
                            units.append(f"{prefix}/challenger/r{repeat}/f{fold}/inner/trial{trial}/fold{inner}")
                    units.append(f"{prefix}/challenger/r{repeat}/f{fold}/selected_refit")
                else:
                    for arm in self.arms:
                        base = f"{prefix}/{arm}/r{repeat}/f{fold}"
                        if arm in self.selector_arms:
                            units.append(f"{base}/selector")
                        units.append(f"{base}/model")
        return tuple(units)


# These are reviewed variants of ONE authentic matrix row, not new matrix rows.
SEMANTIC_VARIANTS = tuple("dim_semantic_block_drop_" + block
                          for block in ("calendar", "measurement", "building_metadata"))


def source_matrix_identity(identity: str) -> str:
    return "dim_semantic_block_drop" if identity in SEMANTIC_VARIANTS else identity


@dataclass(frozen=True)
class FullRunSpec:
    """Declarative snapshot; verify_bindings checks bytes, not scientific admission."""

    run_id: str
    candidates: tuple[FullCandidateSnapshot, ...]
    sources: tuple[SourceBinding, ...]
    environment: tuple[tuple[str, str], ...]
    resource_policy: FullResourcePolicy
    execution_mode: Literal["full_run"] = "full_run"
    version: int = 1
    folds: int = 4
    repeats: int = 3
    split_seeds: tuple[int, ...] = (42, 43, 44)
    estimator_seed: int = 42
    positive_label: int = 1
    development_rows: int = 1_399_631
    vault_rows: int = 349_863
    vault_excluded: bool = True

    def __post_init__(self) -> None:
        _text(self.run_id, "run_id")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.run_id):
            raise ValueError("run_id must be a canonical single-component identifier")
        for name, expected in (("version", 1), ("folds", 4), ("repeats", 3), ("estimator_seed", 42), ("positive_label", 1)):
            _integer(getattr(self, name), name, (expected, expected))
        for name in ("development_rows", "vault_rows"):
            _integer(getattr(self, name), name)
        if (self.execution_mode != "full_run" or self.vault_excluded is not True
                or self.split_seeds != (42, 43, 44)
                or not isinstance(self.split_seeds, tuple)
                or any(type(seed) is not int for seed in self.split_seeds)):
            raise ValueError("full frozen protocol and vault exclusion are required")
        if not isinstance(self.resource_policy, FullResourcePolicy):
            raise TypeError("typed FullResourcePolicy required")
        if (not isinstance(self.candidates, tuple) or not self.candidates
                or any(not isinstance(c, FullCandidateSnapshot) for c in self.candidates)
                or len({c.spec.experiment_id for c in self.candidates}) != len(self.candidates)):
            raise ValueError("unique immutable candidate snapshots required")
        if (not isinstance(self.sources, tuple) or any(not isinstance(s, SourceBinding) for s in self.sources)
                or len({s.path for s in self.sources}) != len(self.sources)
                or not {"matrix", "data", "split", "producer", "notebook"} <= {s.role for s in self.sources}):
            raise ValueError("complete unique source evidence required")
        _pairs(self.environment, "environment")
        if not self.environment or any(not isinstance(v, str) or not v.strip() for _, v in self.environment):
            raise ValueError("explicit environment versions required")
        if len(set(self.planned_units)) != len(self.planned_units):
            raise ValueError("duplicate planned unit")

    @property
    def planned_units(self) -> tuple[str, ...]:
        return tuple(unit for candidate in self.candidates for unit in candidate.planned_units)

    @property
    def planned_trials(self) -> int:
        return sum(12 * c.study.n_trials for c in self.candidates if c.study is not None)

    @property
    def binding_sha256(self) -> str:
        return _digest(_json(asdict(self)))

    def verify_bindings(self, root: str | Path) -> None:
        root = _validate_managed_path(root, "source root", directory=True)
        notebooks = {}
        for source in self.sources:
            path = source.verify(root)
            if source.role == "notebook":
                notebooks[source.path] = json.loads(path.read_text())
        for candidate in self.candidates:
            notebook = notebooks.get(candidate.source_notebook)
            if notebook is None or sum(cell.get("id") == candidate.source_cell_id for cell in notebook.get("cells", [])) != 1:
                raise ValueError("source notebook cell evidence missing or ambiguous")


FullStatus = Literal["pending_implementation", "ready", "blocked", "prohibited", "running", "awaiting_validation", "completed", "failed", "cancelled", "interrupted"]


@dataclass(frozen=True)
class AdmittedFullRun:
    """Process-local preflight evidence, not a serializable caller permission.

    Only preflight issues authentic seals. Fresh supervisors/resume processes must
    repeat read-only preflight; JSON/deserialization is deliberately not admission.
    Journal APIs remain unchanged; runners verify this grant before any writes.
    """

    spec: FullRunSpec
    source_root: str
    rows: tuple[FullAdmissionRow, ...]
    implementation_sources: tuple[SourceBinding, ...]
    actual_environment: tuple[tuple[str, str], ...]
    resource_opt_out_record: str
    train_rows: int
    timestamp_unit: str
    timestamp_time_zone: str | None
    ready_experiment_ids: tuple[str, ...]
    seal: str
    canonical_timestamp_unit: str = "ns"

    def verify_bindings(self, root: str | Path, *, phase: str = "launch", completion_admission: OrdinaryCompletionAdmission | None = None) -> None:
        from project.experiments.full_campaign_preflight import verify_admitted_full_run
        verify_admitted_full_run(root, self, spec=self.spec, phase=phase, completion_admission=completion_admission)


@dataclass(frozen=True)
class OrdinaryCompletionAdmission:
    """Durable parent authentication, not a public boolean or validator checksum."""

    source_root: str
    run_root: str
    destination: str
    spec_sha256: str
    proof_nonce: str
    authority_id: str
    receipt_json: str
    publication_json: str
    signature: str


@dataclass(frozen=True)
class FullStatusRecord:
    """Status claims carry seals; consumers must independently verify those seals."""

    experiment_id: str
    status: FullStatus
    blocker_codes: tuple[str, ...] = ()
    n03_eligible: bool = False
    validation_sha256: str | None = None
    supervisor_sha256: str | None = None

    def __post_init__(self) -> None:
        _text(self.experiment_id, "experiment_id")
        if self.status not in ("pending_implementation", "ready", "blocked", "prohibited", "running", "awaiting_validation", "completed", "failed", "cancelled", "interrupted"):
            raise ValueError("invalid full status")
        if not isinstance(self.blocker_codes, tuple) or len(set(self.blocker_codes)) != len(self.blocker_codes):
            raise ValueError("blocker codes must be unique immutable tuples")
        for code in self.blocker_codes:
            _text(code, "blocker code")
        for digest in (self.validation_sha256, self.supervisor_sha256):
            if digest is not None and not _is_sha256(digest):
                raise ValueError("completion proof must be SHA-256")
        if type(self.n03_eligible) is not bool:
            raise ValueError("eligibility must be exact boolean")
        if self.n03_eligible and (self.status != "completed" or self.blocker_codes
                                 or self.validation_sha256 is None or self.supervisor_sha256 is None):
            raise ValueError("eligibility requires completion, validation and supervisor proofs")


@dataclass(frozen=True)
class ArtifactEvidence:
    path: str
    sha256: str

    def __post_init__(self) -> None:
        _relative(self.path)
        if not _is_sha256(self.sha256):
            raise ValueError("evidence hash must be SHA-256")


@dataclass(frozen=True)
class FullAttemptRecord:
    attempt_id: str
    unit: str
    status: str
    artifact_directory: str
    selected_parameters: tuple[tuple[str, object], ...] = ()
    evidence: tuple[ArtifactEvidence, ...] = ()
    reason: str | None = None


class FullFitJournal:
    """Single-owner append-only attempt journal; no budget, lease or eligibility grant.

    Keep ownership until close. Runtime must explicitly reconcile any running
    attempt after interruption. begin/retry durably charge before construction.
    seal verifies and fsyncs files; parent runtime still owns OOF/exit validation.
    """

    def __init__(self, root: str | Path, spec: FullRunSpec, *, source_root: str | Path) -> None:
        if not isinstance(spec, FullRunSpec):
            raise TypeError("FullRunSpec required")
        self._root = _validate_managed_path(root, "run root", directory=True)
        self._spec = spec
        self._source_root = _validate_managed_path(source_root, "source root", directory=True)
        self._binding_snapshot = self._operational_snapshot()
        self._mutex = RLock()
        self._read_only = False
        self._fd = None
        self._conn = None
        spec.verify_bindings(self.source_root)
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            _fsync_directory_chain(self.root, Path(self.root.anchor))
            lock = _validate_managed_path(self.root / ".full_fit.lock", "journal lock", root=self.root)
            self._fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("full journal busy") from error
            db = self.root / "full_fit.sqlite3"
            for suffix in ("", "-journal", "-wal", "-shm"):
                _validate_managed_path(Path(str(db) + suffix), "journal", root=self.root)
            self._conn = sqlite3.connect(db, check_same_thread=False)
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS binding (snapshot TEXT NOT NULL, head TEXT NOT NULL, count INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS events (sequence INTEGER PRIMARY KEY, payload TEXT NOT NULL, digest TEXT NOT NULL);
            """)
            _fsync_directory_chain(self.root, self.root)
            rows = self._conn.execute("SELECT snapshot FROM binding").fetchall()
            snapshot = self._binding_snapshot
            if not rows:
                if self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]:
                    raise ValueError("missing journal binding")
                self._conn.execute("INSERT INTO binding VALUES (?, ?, 0)", (snapshot, _digest(snapshot)))
                self._conn.commit()
            elif rows != [(snapshot,)]:
                raise ValueError("immutable run binding mismatch")
            self._restore()
            self.verify_completed()
        except BaseException:
            self.close()
            raise

    @classmethod
    def open_readonly(cls, root: str | Path, spec: FullRunSpec, *, source_root: str | Path) -> Self:
        """Reopen existing parent evidence without creating or reconciling units."""
        if type(spec) is not FullRunSpec:
            raise TypeError("exact FullRunSpec required")
        journal = cls.__new__(cls)
        journal._root = _validate_managed_path(root, "journal evidence root", directory=True)
        journal._spec = spec
        journal._source_root = _validate_managed_path(source_root, "source root", directory=True)
        journal._mutex = RLock()
        journal._read_only = True
        journal._fd = None
        journal._conn = None
        journal._binding_snapshot = journal._operational_snapshot()
        spec.verify_bindings(journal.source_root)
        try:
            lock = _validate_managed_path(journal.root / ".full_fit.lock", "existing journal lock", root=journal.root)
            journal._fd = os.open(lock, os.O_RDONLY | os.O_NOFOLLOW)
            fcntl.flock(journal._fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            database = journal.root / "full_fit.sqlite3"
            for suffix in ("", "-journal", "-wal", "-shm"):
                _validate_managed_path(Path(str(database) + suffix), "existing journal evidence", root=journal.root)
            journal._conn = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
            journal._restore()
            journal.verify_completed()
            return journal
        except BaseException:
            journal.close()
            raise

    @property
    def root(self) -> Path:
        return self._root

    @property
    def spec(self) -> FullRunSpec:
        return self._spec

    @property
    def source_root(self) -> Path:
        return self._source_root

    def _operational_snapshot(self) -> str:
        return _json({"root": str(self.root), "source_root": str(self.source_root),
                      "spec": asdict(self.spec)})

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._mutex:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None

    def _restore(self) -> None:
        if self._conn is None:
            raise ValueError("journal closed")
        bindings = self._conn.execute("SELECT snapshot, head, count FROM binding").fetchall()
        if (len(bindings) != 1 or bindings[0][0] != self._binding_snapshot
                or self._operational_snapshot() != self._binding_snapshot):
            raise ValueError("immutable operational run binding mismatch")
        snapshot, head, count = bindings[0]
        previous = _digest(snapshot)
        records = {}
        sequence = 0
        for expected, (sequence, payload, digest) in enumerate(self._conn.execute("SELECT * FROM events ORDER BY sequence"), 1):
            if sequence != expected or digest != _digest(previous + payload):
                raise ValueError("journal event hash mismatch")
            event = json.loads(payload)
            attempt_id = event["attempt_id"]
            if event["kind"] == "begin":
                if attempt_id in records or event["unit"] not in self.spec.planned_units:
                    raise ValueError("invalid journal planned unit")
                parameters = _freeze_json(event["parameters"])
                _pairs(parameters, "journal selected parameters")
                records[attempt_id] = FullAttemptRecord(attempt_id, event["unit"], "running", event["directory"], parameters)
            elif event["kind"] == "seal":
                current = records[attempt_id]
                if current.status != "running" or current.evidence:
                    raise ValueError("invalid journal seal")
                records[attempt_id] = replace(current, evidence=tuple(ArtifactEvidence(**e) for e in event["evidence"]), status="awaiting_validation")
            elif event["kind"] == "finish":
                current = records[attempt_id]
                if current.status not in ("running", "awaiting_validation") or (event["status"] == "completed" and not current.evidence):
                    raise ValueError("invalid journal finish")
                records[attempt_id] = replace(current, status=event["status"], reason=event["reason"])
            else:
                raise ValueError("unknown journal event")
            previous = digest
        if (previous, sequence) != (head, count):
            raise ValueError("journal head hash mismatch")
        self._records = records

    def _append(self, event: dict) -> None:
        self._restore()
        payload = _json(event)
        head, count = self._conn.execute("SELECT head, count FROM binding").fetchone()
        digest = _digest(head + payload)
        with self._conn:
            self._conn.execute("INSERT INTO events VALUES (?, ?, ?)", (count + 1, payload, digest))
            self._conn.execute("UPDATE binding SET head=?, count=?", (digest, count + 1))
        self._restore()

    def attempts(self) -> tuple[FullAttemptRecord, ...]:
        with self._mutex:
            self._restore()
            return tuple(self._records.values())

    @property
    def used_model_fits(self) -> int:
        return sum(not a.unit.endswith("/selector") for a in self.attempts())

    @property
    def used_selector_fits(self) -> int:
        return sum(a.unit.endswith("/selector") for a in self.attempts())

    def begin(self, unit: str, *, selected_parameters: tuple[tuple[str, object], ...] = (),
              prepare_attempt: Callable[[FullAttemptRecord], None] | None = None) -> FullAttemptRecord:
        return self._begin(unit, selected_parameters, None, prepare_attempt)

    def retry(self, unit: str, *, reason: str, selected_parameters: tuple[tuple[str, object], ...] = (),
              prepare_attempt: Callable[[FullAttemptRecord], None] | None = None) -> FullAttemptRecord:
        _text(reason, "explicit retry reason")
        return self._begin(unit, selected_parameters, reason, prepare_attempt)

    def _begin(self, unit: str, parameters: tuple, retry_reason: str | None,
               prepare_attempt: Callable[[FullAttemptRecord], None] | None = None) -> FullAttemptRecord:
        if self._read_only:
            raise ValueError("read-only journal cannot debit or retry")
        _pairs(parameters, "selected parameters")
        with self._mutex:
            self._restore()
            self.verify_completed()
            if unit not in self.spec.planned_units:
                raise ValueError("unknown planned unit")
            history = [a for a in self._records.values() if a.unit == unit]
            if any(a.status == "completed" for a in history):
                raise ValueError("completed unit cannot be overwritten")
            if any(a.status in ("running", "awaiting_validation") for a in self._records.values()):
                raise ValueError("campaign already has an active attempt")
            if bool(history) != (retry_reason is not None):
                raise ValueError("unfinished attempted work requires explicit retry; first work uses begin")
            attempt_id = uuid.uuid4().hex
            directory = f"attempts/{attempt_id}"
            destination = _validate_managed_path(self.root / directory, "attempt directory", root=self.root, directory=True)
            destination.parent.mkdir(exist_ok=True)
            destination.mkdir()
            _fsync_directory_chain(destination, self.root)
            if prepare_attempt is not None:
                prepare_attempt(FullAttemptRecord(attempt_id, unit, "running", directory, parameters))
            self._append({"kind": "begin", "attempt_id": attempt_id, "unit": unit, "directory": directory, "parameters": parameters, "retry_reason": retry_reason})
            return self._records[attempt_id]

    def _verify_evidence(self, record: FullAttemptRecord) -> None:
        for evidence in record.evidence:
            path = _validate_managed_path(self.root / evidence.path, "evidence", root=self.root)
            if not path.is_relative_to(self.root / record.artifact_directory):
                raise ValueError("evidence must belong to its exclusive attempt directory")
            if _file_hash(path) != evidence.sha256:
                raise ValueError("evidence hash mismatch")
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            _fsync_directory_chain(path.parent, self.root)

    def seal(self, attempt_id: str, evidence: tuple[ArtifactEvidence, ...]) -> None:
        if self._read_only:
            raise ValueError("read-only journal cannot seal")
        if (not isinstance(evidence, tuple) or not evidence
                or any(not isinstance(e, ArtifactEvidence) for e in evidence)
                or len({e.path for e in evidence}) != len(evidence)):
            raise ValueError("unique typed immutable evidence required")
        with self._mutex:
            self._restore()
            record = self._records[attempt_id]
            if record.status != "running":
                raise ValueError("attempt can only be sealed once while running")
            self._verify_evidence(replace(record, evidence=evidence))
            self._append({"kind": "seal", "attempt_id": attempt_id, "evidence": [asdict(e) for e in evidence]})

    def finish(self, attempt_id: str, status: str, *, reason: str | None = None) -> None:
        if self._read_only:
            raise ValueError("read-only journal cannot finish")
        if status not in ("completed", "failed", "interrupted", "cancelled"):
            raise ValueError("invalid terminal attempt status")
        if status != "completed":
            _text(reason, "failure/interruption reason")
        with self._mutex:
            self._restore()
            record = self._records[attempt_id]
            if record.status not in ("running", "awaiting_validation"):
                raise ValueError("attempt already finished")
            if status == "completed":
                if not record.evidence:
                    raise ValueError("completion requires sealed evidence")
                self.spec.verify_bindings(self.source_root)
                self._verify_evidence(record)
            self._append({"kind": "finish", "attempt_id": attempt_id, "status": status, "reason": reason})

    def verify_completed(self) -> tuple[str, ...]:
        with self._mutex:
            self._restore()
            self.spec.verify_bindings(self.source_root)
            completed = []
            for record in self._records.values():
                if record.status == "completed":
                    self._verify_evidence(record)
                    completed.append(record.unit)
            return tuple(completed)
