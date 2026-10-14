"""Transport-only HPO lifecycle; no journal opening, estimators or subprocesses.

Integration must authenticate a dedicated HPO worker request binding this plan,
then relay envelopes through exclusive event/ack files (the ordinary worker's
request codec and ParentLifecycleService intentionally reject this lane).
Only the supervisor instantiates ParentHpoLifecycle, borrowing its owned journal.
Its required verifier independently checks training-only fold membership,
predictions/AUC and refit evidence before completion or completed-proof reuse.
Signatures authenticate transport, not scientific validity or worker honesty.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from project.experiments.fit_accounting import FitAccounting
from project.experiments.full_nested_tuning import (
    compute_fit_config_sha256,
    full_hpo_study_spec,
    partition_planned_units,
    validate_full_hpo_factory,
)
from project.experiments.full_run_contracts import (
    ArtifactEvidence,
    FullAttemptRecord,
    FullFitJournal,
    FullRunSpec,
)
from project.experiments.full_snapshot_codec import (
    canonical_json,
    exact_fields,
    parse_json,
)
from project.experiments.nested_tuning import (
    DEFAULT_HGB_PARAMS,
    _generate_trial_params,
)

EXPERIMENT = "regime_same_family_hpo_vs_default"


def sign_message(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Use the existing strict canonical codec and HMAC-SHA256 wire envelope."""
    if type(key) is not str or not key:
        raise ValueError("nonempty transport key required")
    body = canonical_json(payload)
    return {"payload": parse_json(body), "signature": hmac.new(
        key.encode(), body.encode(), hashlib.sha256).hexdigest()}


def _authenticate(envelope: dict[str, Any], key: str) -> dict[str, Any]:
    # Detach transport data from mutable collaborator objects through strict JSON.
    data = exact_fields(parse_json(canonical_json(envelope)), {"payload", "signature"})
    if type(data["payload"]) is not dict:
        raise ValueError("object payload required")
    expected = sign_message(data["payload"], key)["signature"]
    if type(data["signature"]) is not str or not hmac.compare_digest(data["signature"], expected):
        raise ValueError("invalid HPO lifecycle signature")
    return data["payload"]


@dataclass(frozen=True)
class HpoFitIdentity:
    spec_sha256: str
    study_sha256: str
    work_id: str
    nonce: str
    experiment_id: str
    repeat: int
    fold: int
    phase: str
    trial_id: int | None
    inner_fold: int | None
    unit: str
    parameters: tuple[tuple[str, object], ...]
    config_sha256: str
    environment_sha256: str
    encoding_sha256: str
    request_sha256: str

    def payload(self) -> dict[str, Any]:
        return parse_json(canonical_json(asdict(self)))


@dataclass(frozen=True)
class HpoLifecyclePlan:
    """Approved immutable partition plan, established by the parent request."""

    spec: FullRunSpec
    repeat: int
    fold: int
    work_id: str
    nonce: str
    request_sha256: str = ""

    def __post_init__(self) -> None:
        if type(self.spec) is not FullRunSpec:
            raise TypeError("exact FullRunSpec required")
        if (type(self.repeat) is not int or not 0 <= self.repeat < 3
                or type(self.fold) is not int or not 0 <= self.fold < 4):
            raise ValueError("invalid outer partition identity")
        if any(type(value) is not str or not value.strip() for value in (self.work_id, self.nonce)):
            raise ValueError("fresh work/nonce identity required")
        snapshots = [item for item in self.spec.candidates if item.spec.experiment_id == EXPERIMENT]
        if (len(snapshots) != 1 or snapshots[0].study is None
                or canonical_json(asdict(snapshots[0].study)) != canonical_json(asdict(full_hpo_study_spec()))
                or snapshots[0].study_method != "random_search"):
            raise ValueError("approved exact T8 K3 seed42 finite random StudySpec required")
        validate_full_hpo_factory(snapshots[0])

    def _identity(self, phase: str, trial: int | None = None, inner: int | None = None) -> HpoFitIdentity:
        study = full_hpo_study_spec()
        if phase == "default_refit":
            params, arm = DEFAULT_HGB_PARAMS, "control"
            suffix = phase
        else:
            if type(trial) is not int or not 0 <= trial < study.n_trials:
                raise ValueError("invalid trial identity")
            params, arm = _generate_trial_params(study)[trial], "challenger"
            suffix = phase
            if phase == "inner":
                if type(inner) is not int or not 0 <= inner < study.inner_folds:
                    raise ValueError("invalid inner fold identity")
                suffix = f"inner/trial{trial}/fold{inner}"
        unit = f"{EXPERIMENT}/{arm}/r{self.repeat}/f{self.fold}/{suffix}"
        return HpoFitIdentity(
            self.spec.binding_sha256,
            hashlib.sha256(canonical_json(asdict(study)).encode()).hexdigest(),
            self.work_id, self.nonce, EXPERIMENT, self.repeat, self.fold,
            phase, trial, inner, unit, tuple(sorted(params.items())),
            compute_fit_config_sha256(EXPERIMENT, arm, params),
            hashlib.sha256(canonical_json(self.spec.environment).encode()).hexdigest(),
            hashlib.sha256(canonical_json([asdict(source) for source in self.spec.sources
                                           if source.role == "encoding"]).encode()).hexdigest(),
            self.request_sha256,
        )

    def default(self) -> HpoFitIdentity:
        return self._identity("default_refit")

    def inner(self, trial_id: int, inner_fold: int) -> HpoFitIdentity:
        return self._identity("inner", trial_id, inner_fold)

    def selected(self, trial_id: int) -> HpoFitIdentity:
        return self._identity("selected_refit", trial_id)


CompletionVerifier = Callable[[HpoFitIdentity, FullAttemptRecord, tuple[ArtifactEvidence, ...]], float | None]


class ParentHpoLifecycle:
    """Dedicated parent handler, not a second journal or ordinary worker gate.

    handle(envelope) returns an authenticated acknowledgement only after durable
    journal mutation. Lost acknowledgement requires explicit parent reconciliation
    and a fresh request/nonce, never replaying this service. Retry reasons are
    parent-only input; workers cannot request extra fits implicitly.
    """

    def __init__(self, journal: FullFitJournal, plan: HpoLifecyclePlan, key: str, *,
                 verify_completion: CompletionVerifier,
                 retry_reasons: Mapping[str, str] | None = None) -> None:
        if type(journal) is not FullFitJournal or journal.spec.binding_sha256 != plan.spec.binding_sha256:
            raise ValueError("parent-owned matching journal required")
        if not callable(verify_completion):
            raise TypeError("independent parent completion verifier required")
        sign_message({}, key)
        self.journal, self.plan, self.key = journal, plan, key
        self.verify_completion = verify_completion
        self.retry_reasons = dict(retry_reasons or {})
        planned = partition_planned_units(EXPERIMENT, plan.repeat, plan.fold, full_hpo_study_spec())
        if any(unit not in planned or type(reason) is not str or not reason.strip()
               for unit, reason in self.retry_reasons.items()):
            raise ValueError("invalid explicit parent retry authorization")
        self._sequence = 0
        self._position = 0
        self._active: FullAttemptRecord | None = None
        self._scores: dict[tuple[int, int], float] = {}
        self._failed = False

    def terminalize_owned(self, status: str, *, reason: str) -> None:
        """Supervisor calls only after confirming its worker/process tree exited.

        No journal-wide sweep: unowned attempts belong to their own supervisors.
        This transport module neither signals processes nor asserts process exit.
        """
        if status not in ("failed", "interrupted", "cancelled") or type(reason) is not str or not reason.strip():
            raise ValueError("explicit parent terminal status/reason required")
        if self._active is not None:
            actual = next(item for item in self.journal.attempts() if item.attempt_id == self._active.attempt_id)
            if actual.unit != self._expected().unit:
                raise ValueError("owned HPO unit identity changed")
            if actual.status in ("running", "awaiting_validation"):
                self.journal.finish(actual.attempt_id, status, reason=reason)
            self._active = None
        self._failed = True

    def _expected(self) -> HpoFitIdentity:
        if self._position == 0:
            return self.plan.default()
        if 1 <= self._position <= 24:
            trial, inner = divmod(self._position - 1, 3)
            return self.plan.inner(trial, inner)
        if self._position == 25:
            winner = min(range(8), key=lambda trial: (-math.fsum(
                self._scores[(trial, fold)] for fold in range(3)) / 3, trial))
            return self.plan.selected(winner)
        raise ValueError("HPO lifecycle already completed")

    def _verify(self, identity: HpoFitIdentity, record: FullAttemptRecord,
                evidence: tuple[ArtifactEvidence, ...]) -> None:
        if record.unit != identity.unit or canonical_json(record.selected_parameters) != canonical_json(identity.parameters):
            raise ValueError("completed proof configuration identity mismatch")
        score = self.verify_completion(identity, record, evidence)
        if identity.phase == "inner":
            if type(score) is not float or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("parent verified finite inner AUC required")
            self._scores[(identity.trial_id, identity.inner_fold)] = score
        elif score is not None:
            raise ValueError("outer refit must not use held-out selection scores")

    def handle(self, envelope: dict[str, Any]) -> dict[str, Any]:
        payload = exact_fields(_authenticate(envelope, self.key),
                               {"version", "kind", "event", "seq", "identity", "evidence", "reason", "attempt_id"})
        if type(payload["version"]) is not int or payload["version"] != 1 or payload["kind"] != "hpo_event":
            raise ValueError("unsupported HPO event version/kind")
        if type(payload["seq"]) is not int or payload["seq"] != self._sequence + 1:
            raise ValueError("HPO lifecycle replay/out-of-order sequence")
        if self._failed:
            raise ValueError("failed session requires fresh parent retry authorization")
        identity = self._expected()
        if canonical_json(payload["identity"]) != canonical_json(identity.payload()):
            raise ValueError("HPO exact study/spec/trial/fold/config identity mismatch")
        event = payload["event"]
        expected_attempt = self._active.attempt_id if self._active is not None else None
        if payload["attempt_id"] != expected_attempt:
            raise ValueError("HPO event attempt identity mismatch")
        if event == "acquire" and self._active is None:
            if payload["evidence"] != [] or payload["reason"] is not None:
                raise ValueError("acquire cannot supply proof or retry authorization")
            self.journal.verify_completed()
            history = [record for record in self.journal.attempts() if record.unit == identity.unit]
            completed = [record for record in history if record.status == "completed"]
            if completed:
                record = completed[0]
                self._verify(identity, record, record.evidence)
                status = "reused"
                self._position += 1
            else:
                if history:
                    reason = self.retry_reasons.get(identity.unit)
                    if reason is None:
                        raise ValueError("explicit parent retry required")
                    record = self.journal.retry(identity.unit, reason=reason, selected_parameters=identity.parameters)
                else:
                    record = self.journal.begin(identity.unit, selected_parameters=identity.parameters)
                self._active = record
                status = "authorized"
        elif event in ("completed", "failed", "interrupted") and self._active is not None:
            record = self._active
            if event == "completed":
                if payload["reason"] is not None or type(payload["evidence"]) is not list:
                    raise ValueError("invalid completion evidence")
                evidence = tuple(ArtifactEvidence(**exact_fields(item, {"path", "sha256"}))
                                 for item in payload["evidence"])
                self.journal.seal(record.attempt_id, evidence)
                record = next(item for item in self.journal.attempts() if item.attempt_id == record.attempt_id)
                self._active = record
                self._verify(identity, record, record.evidence)
                self.journal.finish(record.attempt_id, "completed")
                self._position += 1
                status = "completed"
            else:
                if payload["evidence"] != [] or type(payload["reason"]) is not str or not payload["reason"].strip():
                    raise ValueError("failure reason required")
                self.journal.finish(record.attempt_id, event, reason=payload["reason"])
                self._failed = True
                status = event
            record = next(item for item in self.journal.attempts() if item.attempt_id == record.attempt_id)
            self._active = None
        else:
            raise ValueError("invalid HPO lifecycle transition")
        self._sequence = payload["seq"]
        return sign_message({**payload, "status": status, "record": asdict(record)}, self.key)


class WorkerHpoLifecycle(FitAccounting):
    """Worker relay: never receives a journal, never mutates parent storage.

    run_fit is the guarded constructor boundary. Transport exceptions poison the
    session; uncertain debit/completion must be reconciled by the parent. Reuse
    returns None with no model constructor (prediction shards need separate proof).
    """

    def __init__(self, plan: HpoLifecyclePlan, key: str,
                 exchange: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        sign_message({}, key)
        self.plan, self.key, self.exchange = plan, key, exchange
        self._sequence = 0
        self._used = 0
        self._identity: HpoFitIdentity | None = None
        self._record: FullAttemptRecord | None = None
        self._completed_proof: FullAttemptRecord | None = None
        self._poisoned = False

    @property
    def completed_proof(self) -> FullAttemptRecord:
        """Last authenticated completion/reuse record, never a worker availability flag."""
        if self._completed_proof is None:
            raise ValueError("authenticated completed proof required")
        return self._completed_proof

    @property
    def reserved_model_fits(self) -> int:
        return 26

    @property
    def used_model_fits(self) -> int:
        return self._used

    def _send(self, event: str, *, evidence: tuple[ArtifactEvidence, ...] = (),
              reason: str | None = None) -> tuple[str, FullAttemptRecord]:
        if self._poisoned or self._identity is None:
            raise ValueError("fresh authenticated worker session required")
        self._sequence += 1
        payload = {"version": 1, "kind": "hpo_event", "event": event, "seq": self._sequence, "identity": self._identity.payload(),
                   "evidence": [asdict(item) for item in evidence], "reason": reason,
                   "attempt_id": self._record.attempt_id if self._record is not None else None}
        try:
            ack = exact_fields(_authenticate(self.exchange(sign_message(payload, self.key)), self.key),
                               set(payload) | {"status", "record"})
            if canonical_json({name: ack[name] for name in payload}) != canonical_json(payload):
                raise ValueError("HPO acknowledgement identity/sequence mismatch")
            statuses = {"acquire": ("authorized", "reused"), "completed": ("completed",), "failed": ("failed",), "interrupted": ("interrupted",)}
            if ack["status"] not in statuses[event]:
                raise ValueError("invalid HPO acknowledgement status")
            raw = exact_fields(ack["record"], set(FullAttemptRecord.__dataclass_fields__))
            proof = tuple(ArtifactEvidence(**exact_fields(item, {"path", "sha256"})) for item in raw["evidence"])
            record = FullAttemptRecord(**{**raw, "evidence": proof,
                                         "selected_parameters": tuple(tuple(item) for item in raw["selected_parameters"])})
            required_status = {"authorized": "running", "reused": "completed", "completed": "completed", "failed": "failed", "interrupted": "interrupted"}[ack["status"]]
            if (record.unit != self._identity.unit or canonical_json(record.selected_parameters) != canonical_json(self._identity.parameters)
                    or record.status != required_status or type(record.attempt_id) is not str
                    or not record.attempt_id or record.artifact_directory != f"attempts/{record.attempt_id}"
                    or (required_status == "completed" and not proof)
                    or (event != "acquire" and (self._record is None or self._record.attempt_id != record.attempt_id))):
                raise ValueError("invalid parent attempt proof identity")
            return ack["status"], record
        except BaseException:
            self._poisoned = True
            raise

    def before_fit(self) -> None:
        if self._record is not None:
            raise ValueError("fit already authorized; no unaccounted additional fit")
        status, self._record = self._send("acquire")
        if status == "authorized":
            self._used += 1

    def run_fit(self, identity: HpoFitIdentity, constructor: Callable[[], Any],
                perform: Callable[[Any, FullAttemptRecord], tuple[ArtifactEvidence, ...]]) -> Any | None:
        if self._identity is not None:
            raise ValueError("worker fit already active")
        self._identity = identity
        self._completed_proof = None
        try:
            self.before_fit()
            if self._record.status == "completed":
                self._completed_proof = self._record
                return None
            try:
                model = constructor()
                proof = perform(model, self._record)
            except BaseException as error:
                event = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
                try:
                    self._send(event, reason=f"{type(error).__name__}: {error}")
                except BaseException as transport_error:  # noqa: BLE001 - preserve original fit/interruption
                    error.add_note(f"Parent terminal acknowledgement failed: {type(transport_error).__name__}")
                self._poisoned = True
                raise
            _, self._completed_proof = self._send("completed", evidence=proof)
            return model
        finally:
            self._identity = None
            self._record = None
