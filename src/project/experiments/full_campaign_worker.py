"""Isolated subprocess worker for full candidate campaign execution.

Executes a single work unit (experiment / arm / repeat / fold) in a fresh
subprocess. Debits phases through parent-authorized HMAC handshakes, loads
only one partition into memory, computes OOF predictions, and closes an
exclusive OOF shard without creating supervisor completion evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa

from project.experiments import full_candidate_io as io
from project.experiments.candidate_contracts import (
    CandidateSpec,
    _validate_managed_path,
)
from project.experiments.candidate_dispatch import (
    CandidateArm,
    _configuration,
    fit_arm,
)
from project.experiments.fit_accounting import FitAccounting
from project.experiments.full_run_contracts import FullRunSpec, _file_hash
from project.experiments.full_snapshot_codec import (
    canonical_json,
    configuration_digest,
    exact_fields,
    parse_json,
    validate_request,
)
from project.experiments.full_snapshot_codec import semantic_digest as _semantic_digest
from project.models.candidate_artifacts import (
    candidate_oof_schema,
    write_candidate_oof_batches,
)
from project.models.candidate_scores import extract_candidate_score


def semantic_digest(year_built_encoding: str, provenance_sha256: str) -> str:
    """Compatibility export of the shared canonical semantic binding."""
    return _semantic_digest(year_built_encoding, provenance_sha256)


def _oof_schema(meta: io._IndexMeta) -> pa.Schema:
    """Derive the canonical OOF schema from the authoritative index metadata."""
    building_type = pa.int64() if meta.building_dtype == "int64" else pa.string()
    return candidate_oof_schema(building_type, pa.timestamp(meta.timestamp_unit, tz=meta.timestamp_time_zone))


SELECTOR_SUCCESS_ACK_TIMEOUT_SECONDS = 300.0


def _ack_budget_seconds(event_name: str, request_budget: float) -> float:
    """Give the selector_success phase a generous budget for the heavy parent verify."""
    if event_name == "selector_success":
        return max(request_budget, SELECTOR_SUCCESS_ACK_TIMEOUT_SECONDS)
    return request_budget


def _verify_fitted_target_encoding(arm: CandidateArm, audited_digest: str) -> None:
    """Reject challenger fitted state inconsistent with its pre-debit audit."""
    if arm.arm_id == "control":
        return
    encoder = arm.named_steps.get("features")
    if getattr(encoder, "cross_fitted_", None) is not True:
        raise ValueError("target encoder was not cross-fitted")
    if getattr(encoder, "inner_membership_digest_", None) != audited_digest:
        raise ValueError("target encoder inner membership differs from audited partition")


def _write_oof_shard(
    output_path: Path, batches: Iterable[pa.Table], schema: pa.Schema
) -> tuple[str, int]:
    """Consume exactly one bounded batch at a time and hash the closed shard."""
    total_rows = 0

    def counted_batches():
        nonlocal total_rows
        for table in batches:
            total_rows += table.num_rows
            yield table

    write_candidate_oof_batches(output_path, counted_batches(), schema)
    return _file_hash(output_path), total_rows


def _publish_message(path: Path, envelope: dict) -> None:
    """Publish complete JSON exclusively; polling never sees partial JSON."""
    temporary = path.parent / (".message-" + secrets.token_hex(16))
    try:
        with temporary.open("x") as stream:
            stream.write(canonical_json(envelope))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_worker_request(directory: Path, payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Write HMAC-signed worker-request.json for subprocess authorization."""
    validate_request(payload)
    if directory.name != payload["attempt_id"]:
        raise ValueError("worker attempt directory identity mismatch")
    payload = dict(payload, nonce=secrets.token_hex(32))
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("fresh empty worker attempt directory required")
    directory.mkdir(parents=True, exist_ok=True)
    body = canonical_json(payload)
    signature = hmac.new(key.encode(), body.encode(), hashlib.sha256).hexdigest()
    request_file = directory / "worker-request.json"
    _publish_message(request_file, {"payload": payload, "signature": signature})
    return payload


def _authenticate_request(directory: Path, key: str) -> tuple[dict, FullRunSpec]:
    path = _validate_managed_path(directory / "worker-request.json", "worker request", root=directory)
    envelope = exact_fields(parse_json(path.read_text()), {"payload", "signature"})
    payload = envelope["payload"]
    expected = hmac.new(key.encode(), canonical_json(payload).encode(), hashlib.sha256).hexdigest()
    if type(envelope["signature"]) is not str or not hmac.compare_digest(envelope["signature"], expected):
        raise ValueError("invalid worker request signature")
    spec = validate_request(payload)
    if directory.name != payload["attempt_id"]:
        raise ValueError("worker attempt directory identity mismatch")
    return payload, spec


class ParentLifecycleService:
    """Parent supervisor handler for debit and completion handshakes."""

    def __init__(
        self,
        directory: str | Path,
        key: str,
        *,
        on_debit: Callable[[str, dict[str, Any]], None] | None = None,
        on_selector_success: Callable[[dict[str, Any]], None] | None = None,
        on_shard_closed: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.key = key
        self.on_debit = on_debit
        self.on_selector_success = on_selector_success
        self.on_shard_closed = on_shard_closed
        self.shard_evidence: dict[str, Any] | None = None
        self._serviced_events: dict[str, str] = {}
        self._sequence = 0
        self._request: dict | None = None
        self._states: list[tuple[str, str]] = []

    def _initialize(self) -> None:
        try:
            request, spec = _authenticate_request(self.directory, self.key)
        except OSError as exc:
            raise ValueError("authenticated lifecycle request required") from exc
        self._request = request
        snapshot = next(item for item in spec.candidates if item.spec.experiment_id == request["experiment_id"])
        if request["arm_id"] in snapshot.selector_arms:
            self._states.extend([("debit", "selector"), ("selector_success", "selector")])
        self._states.extend([("debit", "model"), ("shard_closed", "model")])

    def service_once(self) -> None:
        """Poll and respond to any pending worker events with signed acks."""
        event_files = sorted(self.directory.glob("event-*.json"))
        if event_files and self._request is None:
            self._initialize()
        for event_path in event_files:
            if event_path.name in self._serviced_events:
                if _file_hash(event_path) != self._serviced_events[event_path.name]:
                    raise ValueError("previously serviced event changed")
                continue
            try:
                raw = event_path.read_text()
                data = exact_fields(parse_json(raw), {"payload", "signature"})
                payload = data["payload"]
                expected_sig = hmac.new(
                    self.key.encode(),
                    canonical_json(payload).encode(),
                    hashlib.sha256,
                ).hexdigest()
                if type(data["signature"]) is not str or not hmac.compare_digest(data["signature"], expected_sig):
                    raise ValueError("invalid lifecycle event signature")
            except OSError as exc:
                raise ValueError("unreadable lifecycle event") from exc

            event_name = payload.get("event")
            seq = payload.get("seq")
            phase = payload.get("phase")
            names = {"event", "seq", "phase", "work_id", "attempt_id", "nonce", "unit"}
            if event_name == "shard_closed":
                names |= {"sha256", "num_rows", "path"}
            elif event_name == "selector_success":
                names |= {"sha256", "path"}
            exact_fields(payload, names)
            request = self._request
            if any(payload[name] != request[name] or type(payload[name]) is not str
                   for name in ("work_id", "attempt_id", "nonce")):
                raise ValueError("lifecycle unit/nonce identity mismatch")
            unit = request["work_id"].rsplit("/", 1)[0] + "/" + str(phase)
            if payload["unit"] != unit or type(seq) is not int or seq != self._sequence + 1:
                raise ValueError("lifecycle replay or out-of-order sequence")
            if event_path.name != f"event-{seq:04d}.json" or not self._states or (event_name, phase) != self._states[0]:
                raise ValueError("invalid lifecycle phase transition")
            ack_path = self.directory / f"ack-{seq:04d}.json"
            if ack_path.exists():
                raise ValueError("lifecycle replay: acknowledgement already exists; use a fresh attempt")

            if event_name == "debit":
                if self.on_debit is None:
                    raise ValueError("durable parent debit callback required")
                self.on_debit(phase, payload)
            elif event_name == "selector_success":
                if payload["path"] != str(self.directory / "selector-outcome.json"):
                    raise ValueError("selector outcome path mismatch")
                outcome = _validate_managed_path(payload["path"], "selector outcome", root=self.directory)
                if _file_hash(outcome) != payload["sha256"]:
                    raise ValueError("selector outcome hash mismatch")
                if self.on_selector_success is None:
                    raise ValueError("durable selector completion callback required")
                self.on_selector_success(payload)
            elif event_name == "shard_closed":
                if (type(payload["num_rows"]) is not int or payload["num_rows"] <= 0
                        or type(payload["path"]) is not str
                        or payload["path"] != str(self.directory / "shard.parquet")
                        or type(payload["sha256"]) is not str):
                    raise ValueError("invalid shard identity or row count")
                shard = _validate_managed_path(payload["path"], "shard", root=self.directory)
                if _file_hash(shard) != payload["sha256"]:
                    raise ValueError("shard hash mismatch")
                self.shard_evidence = payload
                if self.on_shard_closed is not None:
                    self.on_shard_closed(payload)

            ack_payload = {name: payload[name] for name in names - {"sha256", "num_rows", "path"}}
            ack_payload["status"] = "authorized"
            ack_sig = hmac.new(
                self.key.encode(),
                canonical_json(ack_payload).encode(),
                hashlib.sha256,
            ).hexdigest()
            _publish_message(ack_path, {"payload": ack_payload, "signature": ack_sig})
            self._sequence = seq
            self._states.pop(0)
            self._serviced_events[event_path.name] = hashlib.sha256(raw.encode()).hexdigest()


class _WorkerFitAccountant(FitAccounting):
    """Worker-side fit accountant relaying phase debits through HMAC events."""

    def __init__(self, has_selector: bool, send_event: Callable[..., dict[str, Any]]) -> None:
        self.has_selector = has_selector
        self.send_event = send_event
        self._fit_count = 0
        self.selector_evidence = None

    @property
    def reserved_model_fits(self) -> int:
        return 2 if self.has_selector else 1

    @property
    def used_model_fits(self) -> int:
        return self._fit_count

    def before_fit(self) -> None:
        if self.has_selector:
            if self._fit_count == 0:
                self.send_event("debit", phase="selector")
                self._fit_count += 1
            else:
                if self.selector_evidence is None:
                    raise ValueError("actual fitted selector outcome required before model debit")
                self.send_event("selector_success", phase="selector", **self.selector_evidence)
                self.send_event("debit", phase="model")
                self._fit_count += 1
        else:
            self.send_event("debit", phase="model")
            self._fit_count += 1


def _run_hash(spec: FullRunSpec, role: str) -> str:
    for source in spec.sources:
        if source.role == role:
            return source.sha256
    raise ValueError(f"source role not bound: {role}")


def _build_batch_table(
    batch: dict[str, Any],
    predictions: np.ndarray,
    spec: CandidateSpec,
    arm: CandidateArm,
    admitted_spec: FullRunSpec,
    repeat: int,
    fold: int,
    fit_sha: str,
    schema: pa.Schema,
    arm_id: str | None = None,
) -> pa.Table:
    effective_arm_id = arm_id if arm_id is not None else arm.arm_id
    size = len(batch["row_index"])
    columns = {
        "candidate_id": [f"{spec.experiment_id}_{effective_arm_id}"] * size,
        "recipe_id": [spec.experiment_id] * size,
        "model_family": [spec.model_family] * size,
        "model_config_id": [f"{spec.experiment_id}_{effective_arm_id}_config"] * size,
        "row_index": [int(value) for value in batch["row_index"]],
        "repeat": [repeat] * size,
        "fold": [fold] * size,
        "building_id": list(batch["building_id"]),
        "timestamp": pl.Series(batch["timestamp"])
        .cast(pl.Datetime(schema.field("timestamp").type.unit, schema.field("timestamp").type.tz))
        .to_arrow()
        .cast(schema.field("timestamp").type),
        "prediction": [float(value) for value in predictions],
        "target": [int(value) for value in batch["target"]],
        "split_sha256": [_run_hash(admitted_spec, "split")] * size,
        "data_sha256": [_run_hash(admitted_spec, "data")] * size,
        "config_sha256": [admitted_spec.binding_sha256] * size,
        "run_id": [admitted_spec.run_id] * size,
        "split_seed": [int(batch["split_seed"])] * size,
        "experiment_id": [spec.experiment_id] * size,
        "arm_id": [effective_arm_id] * size,
        "score_kind": [arm.score_kind] * size,
        "positive_label": [1] * size,
        "execution_mode": ["full"] * size,
        "non_comparative": [False] * size,
        "n03_eligible": [False] * size,
        "model_fit_config_sha256": [fit_sha] * size,
    }
    return pa.Table.from_pydict(columns, schema=schema)


def run(work_id: str, directory: str | Path) -> None:
    """Subprocess entrypoint: authenticate request, debit, fit, write shard."""
    dir_path = Path(directory)
    key = os.environ.get("FULL_CAMPAIGN_WORKER_KEY")
    if not key:
        sys.stderr.write("missing FULL_CAMPAIGN_WORKER_KEY\n")
        sys.exit(1)

    try:
        payload, full_spec = _authenticate_request(dir_path, key)
    except Exception as error:  # noqa: BLE001
        sys.stderr.write(f"signature verification error: {error}\n")
        sys.exit(1)

    if payload.get("work_id") != work_id:
        sys.stderr.write(f"work_id mismatch: {payload.get('work_id')} != {work_id}\n")
        sys.exit(1)

    project_root = _validate_managed_path(payload["project_root"], "project root", directory=True)
    full_spec.verify_bindings(project_root)
    provenance = next(source for source in full_spec.sources if source.role == "encoding")
    provenance_path = provenance.verify(project_root)
    provenance_data = parse_json(provenance_path.read_text())
    if type(provenance_data) is not dict or provenance_data.get("encoding_id") != payload["year_built_encoding"]:
        raise ValueError("encoding provenance contract mismatch")
    # Run-owned index only; reject traversal/symlinks before opening it.
    index_path = _validate_managed_path(payload["index_path"], "development index", root=dir_path.parent.parent)
    if not index_path.is_file():
        sys.stderr.write("index file missing\n")
        sys.exit(1)

    actual_index_sha = _file_hash(index_path)
    if actual_index_sha != payload["index_sha256"]:
        sys.stderr.write("index sha mismatch\n")
        sys.exit(1)

    index_connection, index_meta = io._open_bound_index(index_path, full_spec)
    index_connection.close()
    schema = _oof_schema(index_meta)

    seq = 0

    def send_event_and_wait_ack(event_name: str, **kwargs: Any) -> dict[str, Any]:
        nonlocal seq
        seq += 1
        phase = kwargs["phase"]
        event_payload = {"event": event_name, "seq": seq, "work_id": work_id,
                         "attempt_id": payload["attempt_id"], "nonce": payload["nonce"],
                         "unit": work_id.rsplit("/", 1)[0] + "/" + phase, **kwargs}
        event_sig = hmac.new(
            key.encode(),
            canonical_json(event_payload).encode(),
            hashlib.sha256,
        ).hexdigest()
        event_path = dir_path / f"event-{seq:04d}.json"
        _publish_message(event_path, {"payload": event_payload, "signature": event_sig})

        ack_path = dir_path / f"ack-{seq:04d}.json"
        deadline = time.monotonic() + _ack_budget_seconds(event_name, payload["ack_timeout_seconds"])
        while not ack_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not ack_path.is_file():
            sys.stderr.write(f"parent acknowledgement timed out for seq {seq}\n")
            sys.exit(1)

        try:
            ack_data = exact_fields(parse_json(ack_path.read_text()), {"payload", "signature"})
            ack_p = ack_data["payload"]
            ack_s = ack_data["signature"]
            exp_ack_s = hmac.new(
                key.encode(),
                canonical_json(ack_p).encode(),
                hashlib.sha256,
            ).hexdigest()
            expected_ack = {name: event_payload[name] for name in
                            ("event", "seq", "phase", "work_id", "attempt_id", "nonce", "unit")}
            expected_ack["status"] = "authorized"
            if type(ack_s) is not str or not hmac.compare_digest(ack_s, exp_ack_s) or ack_p != expected_ack or type(ack_p.get("seq")) is not int:
                sys.stderr.write(f"invalid ack signature or status for seq {seq}\n")
                sys.exit(1)
            return ack_p
        except Exception as error:  # noqa: BLE001
            sys.stderr.write(f"ack verification failed: {error}\n")
            sys.exit(1)

    experiment_id = payload["experiment_id"]
    arm_id = payload["arm_id"]
    repeat = payload["repeat"]
    fold = payload["fold"]
    encoding = payload["year_built_encoding"]

    target_snapshot = next(s for s in full_spec.candidates if s.spec.experiment_id == experiment_id)
    cand_spec = target_snapshot.spec

    # Build arm
    from project.experiments.full_campaign_capabilities import (
        build_reviewed_ordinary_arms,
    )
    arms = build_reviewed_ordinary_arms(target_snapshot, year_built_encoding=encoding)
    if arm_id in arms:
        arm = arms[arm_id]
    elif arm_id == "control" and "passthrough" in arms:
        arm = arms["passthrough"]
    elif arm_id == "challenger" and "passthrough" in arms and len(arms) == 2:
        other_key = next(k for k in arms if k != "passthrough")
        arm = arms[other_key]
    else:
        raise KeyError(f"arm_id {arm_id!r} not found in arms: {list(arms.keys())}")

    underlying = arm.transformer_or_pipeline
    has_selector = (
        hasattr(underlying, "named_steps")
        and "selector" in underlying.named_steps
        and underlying.named_steps["selector"] != "passthrough"
    )

    accountant = _WorkerFitAccountant(has_selector, send_event_and_wait_ack)

    # Load partition and fit
    partition = io.load_full_training_partition(index_path, full_spec, repeat, fold)
    if list(cand_spec.feature_pool) != list(partition.X_train.columns):
        partition = replace(partition, X_train=partition.X_train[list(cand_spec.feature_pool)])
    audited_te_digest = None
    if cand_spec.experiment_id == "feat_conditional_cross_fitted_target_encoding":
        from project.experiments.full_development_readiness import (
            verify_target_encoding_partition,
        )

        audited_te_digest = verify_target_encoding_partition(project_root, full_spec, partition)
    if has_selector:
        from sklearn.base import clone

        from project.experiments.full_campaign_evidence import selector_outcome

        pipeline = clone(underlying)
        X = pipeline.named_steps["preprocessor"].fit_transform(partition.X_train, partition.y_train)
        accountant.before_fit()
        selected = pipeline.named_steps["selector"].fit_transform(X, partition.y_train)
        outcome = selector_outcome(pipeline, partition, payload)
        outcome_path = dir_path / "selector-outcome.json"
        _publish_message(outcome_path, outcome)
        accountant.selector_evidence = {"path": str(outcome_path), "sha256": _file_hash(outcome_path)}
        accountant.before_fit()
        pipeline.named_steps["model"].fit(selected, partition.y_train)
        fitted = replace(arm, transformer_or_pipeline=pipeline)
    else:
        fitted = fit_arm(arm, partition, accountant)

    if audited_te_digest is not None:
        _verify_fitted_target_encoding(fitted, audited_te_digest)

    fit_sha = configuration_digest(_configuration(fitted.transformer_or_pipeline))

    def oof_batches():
        for batch in io.iter_full_validation_batches(index_path, full_spec, repeat, fold):
            X_val = batch["X"]
            if list(cand_spec.feature_pool) != list(X_val.columns):
                X_val = X_val[list(cand_spec.feature_pool)]
            score = extract_candidate_score(fitted, X_val, arm.score_kind)
            yield _build_batch_table(batch, score.values, cand_spec, arm, full_spec, repeat, fold, fit_sha, schema, arm_id=arm_id)

    output_path = dir_path / payload["output_path"]
    shard_sha, total_rows = _write_oof_shard(output_path, oof_batches(), schema)
    send_event_and_wait_ack("shard_closed", phase="model", sha256=shard_sha, num_rows=total_rows, path=str(output_path))
