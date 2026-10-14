# Exact RBF SVC contract readiness v1

Status: **isolated executable backend implemented; full-route promotion and launch not authorized here**.
The contract, dedicated worker, their unit tests and this document are owned by this lane.
No prepared/test/vault data was opened; no real estimator fits, installations,
shared-source edits or readiness-gate changes occurred.

## Scientific contract

`build_full_rbf_svc_arms(snapshot, year_built_encoding=...)` accepts the exact
`dim_pca_vs_passthrough` / `svm` snapshot with canonical `CORE`, identities
`control` and `challenger`, no selector fits and no study. It reuses existing
`candidate_dispatch` and `features.selection` graphs rather than replacing them.
The original factory identities `passthrough` / `pca` are mapped once to full
identities. Requested parameters must match typed frozen defaults; the resolved
graphs are independently compared using `configuration_digest(_configuration(...))`.

- Both arms: exact libsvm RBF SVC, C=1.0, gamma=scale, estimator seed 42,
  natural class weights, max_iter=-1, frozen cache_size=200.0. No sample reduction,
  alternative kernel, LinearSVC, approximation, calibration or HPO.
- Numeric block: training-fitted median imputation and standard scaling; only
  challenger applies numeric-only PCA at variance fraction 0.95, full solver,
  no whitening. Categorical imputation/OHE remains outside PCA, train-fitted,
  with unknown categories ignored. The existing output registry traces PCA
  coordinates to numeric inputs and OHE coordinates to categorical inputs.
- Four building-disjoint folds × three repeats, split seeds 42/43/44:
  24 planned model units, zero supervised selector units. PCA is learned
  preprocessing, not an additional classifier/selector debit.
- Scores are explicit `decision_margin`, positive label 1. ROC-AUC/AP ranking
  only; LogLoss/Brier are null with `requires_calibrated_probability`.
  No probability-pool eligibility is asserted. sklearn 1.9's frozen disabled
  probability default is the `"deprecated"` sentinel, not boolean False;
  the tests also require absence of `predict_proba`.
- Existing strict FullRunSpec snapshot transport round-trips unchanged. New
  estimate/cardinality objects are NOT added to its codec or treated as admission.

## Deterministic risk estimate and lineage

`RBFDevelopmentCardinality` requires full spec digest, exact data/split bindings,
development cardinality and all twelve actual validation cardinalities. Every
repeat must sum exactly to the full development population; boolean counts,
empty partitions, reduced folds and source mismatches are rejected. Training
counts are derived by subtraction, never by assuming row-balanced building folds.
`estimate_full_rbf_svc` reports per-arm configuration digests, source bindings,
24 fits, 3*n_dev OOF rows per arm, maximum train count, and n_train²*8 bytes.

**This is source-bound declared metadata, not authenticated data evidence.**
Parent preflight must derive and independently authenticate the counts from
development-only membership/index evidence before persisting the estimate.
The pure API neither verifies file bytes nor supplies a launch seal.

Previously approved design evidence reports development=1,399,631 and maximum
outer training=1,051,408. At that maximum, the conceptual dense float64 Gram size
is 8,843,670,259,712 bytes (8.84 decimal TB). This is NOT a claim that sklearn
libsvm allocates a dense Gram matrix. Its kernel cache is distinct from total
model/preprocessing/process memory; cache_size is an estimator setting, not a
campaign RSS cap. PCA reduces numeric width, not row count or kernel pair count.

Severity: **severe compute risk; empirical completion not guaranteed**.
No wall-time estimate, deadline or ETA is supplied. The approved resource policy
retains all campaign enforcement flags false and limits null. Opt-out does not
prove computational feasibility. Official sklearn SVM documentation describes
dataset/cache-dependent solver scaling between quadratic and cubic in samples:
<https://github.com/scikit-learn/scikit-learn/blob/1.9.1/doc/modules/svm.rst>.
The scale figures above come from the approved design, not a new data scan.

## Cancellation contract

Explicit parent cancellation remains possible without a guessed training time
limit. Parent runtime must stop the process group, verify descendant cleanup,
append cancelled status while preserving fit debit, quarantine partial outputs,
and stop the queue for reconciliation. Cancelled/incomplete arms have no N03
completion or probability eligibility. Retry requires explicit request, fresh
attempt paths and unchanged source/config/environment bindings; no substitution
or silent reduced work is allowed. This module declares these semantics only;
it does not implement process termination or journal mutation.

## Terminal evidence

Command (installed environment, no package synchronization):

```sh
rtk proxy .venv/bin/python -B -m pytest -p no:cacheprovider tests/unit/test_full_rbf_svc_contract.py -q
```

- Red: 19 expected assertion failures against an importable docstring-only
  scaffold, each reporting missing RBF contract behavior; no collection errors.
- Initial green run exposed one incorrect test assumption about sklearn's
  probability default; existing source explicitly documents its 1.9 sentinel.
  The assertion was corrected without changing the frozen factory.
- Green: 20 passed. Tests cover exact configs, identity/digest stability,
  strict snapshot transport, planned counts, scientific drift rejection,
  source/count mismatch, recording-only train routing, numeric-only PCA/OHE
  construction and transform-without-refit, lineage and margin guardrails.
- Final rerun: 20 passed in 1.47s. `rtk proxy .venv/bin/ruff check
  src/project/experiments/full_rbf_svc_contract.py
  tests/unit/test_full_rbf_svc_contract.py` reported `All checks passed!`.
- Recording boundaries replace all invoked learned estimator fits. These tests
  do not prove actual libsvm execution, empirical PCA behavior or full throughput.
  Parent's tiny actual SVC proof remains deferred until ordinary fitting is idle.

## Minimal parent integration and promotion tests (not performed)

1. Add a narrowly authenticated RBF branch that invokes
   `build_full_rbf_svc_arms` at every construction/admission/resume/completion
   boundary. Keep ordinary probability-only checks unchanged; do not relax all
   recipe score types. Bind this module in the route's implementation closure.
2. Snapshot canonical CORE, svm family and control/challenger arms. Bind actual
   notebook/cell, data/split/encoding/environment sources; do not use the unit
   tests' fixture references as real evidence.
3. Derive twelve development validation counts in authenticated preflight/index
   code, construct the declared cardinality and persist the deterministic report.
   Preserve cancellation, uncapped supervision and existing durable fit accounting.
4. After current ordinary fixture fitting stops, run a tiny real SVC/PCA fixture
   proof sequentially: both arms, frozen 4×3, 24 durable debits, train-only
   preprocessing, margin shards, complete coverage, null probability guardrails,
   exact digests, tamper rejection and parent completion authentication.
5. Add cancellation/partial-artifact/no-next-fit tests, changed-count/source
   rejection at admission/resume, and margin exclusion from probability pools.
   Only parent may promote reviewed capability membership/readiness after these
   checks; pure contract success alone is insufficient. No full launch promised.

Approved references: `docs/superpowers/specs/2026-10-04-full-campaign-decisions.md`
§6 and `docs/superpowers/specs/2026-10-04-sequential-full-campaign-design.md`.

## Dedicated executable backend (continuation)

Entrypoint: `project.experiments.full_rbf_svc_worker:run` (the value of
`BACKEND_TARGET`). It also accepts `python -m project.experiments.full_rbf_svc_worker
--work-id ... --directory ...` under the authenticated parent supervisor, with
per-attempt `FULL_CAMPAIGN_WORKER_KEY`. This is an actual backend, not a launch
command that bypasses admission. No shared worker/runner/admission/capability/config
file was modified, and no capability membership was promoted.

Parent-only preparation `prepare_rbf_request(directory, payload, key)` reuses the
existing exact snapshot codec and signed `worker-request.json`. Existing request
fields bind source/spec/encoding/provenance/semantic digests, environment in spec,
run-owned index byte hash, experiment/arm/seed/fold/work identity, parent PID,
fresh nonce and exclusive attempt/shard paths. Unsupported configs are rejected
by `validate_full_rbf_svc_snapshot` **without estimator construction**. The exact
installed `cache_size` parameter is integer 200; the numeric estimate remains
200 MiB. Typed literal validation is also tested against the actual frozen SVC
parameter graph so an explicitly declared valid default is not incorrectly blocked.

The worker authenticates the request, verifies confined source bytes and encoding
metadata, checks actual environment and parent process, verifies the immutable
index hash/no live SQLite sidecars, and reuses its bound integrity/population/
4×3 class-support checks. It reads only the development index for fitting and
scoring; it does not parse prepared/test/vault rows. Source authentication does
stream source bytes for SHA-256, which is not row loading or scoring.

Only one full training packet is loaded. Its complete row-index/group/label
membership is independently compared to the selected training query before
debit. A signed, identity-checked parent acknowledgement of **one model debit**
must arrive before pipelines are constructed or any preprocessing/model fit
begins. There is no journal inside this child, no fabricated bounded budget,
no probability fitting, no selector debit and no implicit retry.

Validation is streamed in <=64,000-row batches. Exact row identity, opaque
building identity, physical timestamps, integer labels, frozen fold/repeat/seed
and feature schema are independently compared to the bound validation query.
Missing/extra/corrupted coverage fails closed. Canonical schema v1 preserves
string/int64 buildings and canonical nanosecond timestamps/timezone. Raw ranking
margins, source/spec and actual pipeline config hashes are written exclusively
with initial eligibility false. After source/index rechecks and file/directory
fsync, the worker reports a hash-bound `shard_closed` event. Parent exit/descendant
cleanup, journal seal and scientific validation are still required: this event
does **not** mean completed. Interrupted/failed attempts retain their debit and
partial files; no completion event is fabricated.

### Minimal integration hooks (parent only)

1. Select `BACKEND_TARGET` only through an authenticated dedicated RBF capability
   branch. Bind the contract/worker and their reused transport/I/O/scoring closure.
   Keep ordinary probability gates and other owners' routes unchanged.
2. Call `prepare_rbf_request` with the parent-bound FullRunSpec and immutable
   development index. Then create `make_parent_rbf_service(..., on_debit=...,
   on_shard_closed=...)`. The model debit callback must authenticate the requested
   unit against the journal and **durably append/commit** its begin/retry before
   returning. The callback protocol alone cannot prove journal durability.
3. Pass `service.service_once` as `run_work_unit(..., poll_callback=...)`, propagate
   the per-attempt key through its environment binding, retain the real attempt/
   nonce/spec bindings, expected `shard.parquet`, existing campaign lock,
   cancellation, process identity and descendant-cleanup gates. Acknowledgement
   timeout is a communication timeout, not an imposed training wall-time cap.
4. Only after exit zero, cleanup and signed shard evidence, independently validate
   all 24 units / complete two-arm OOF, authentic raw labels and fitted lineage,
   then seal via parent authority. No worker-written completed status is accepted.

### Continuation test evidence and deferred proof

The new worker suite first produced **10 expected missing-behavior assertion
failures** against an importable scaffold (no collection/setup errors). Recording
boundary tests cover HMAC/nonce/attempt/parent/index/environment rejection,
debit-before-construction, incomplete training, validation identity/target
corruption, acknowledgement denial, fitting failure, prior-attempt replay,
path escape, canonical margin shards and all **24 planned units** with one
recorded model fit each and complete 3*n_dev OOF per arm.

A further fail-first regression caught float validation labels silently accepted
through Arrow coercion; the worker now requires integer target/row-index arrays.
Another fail-first contract regression caught integer cache_size=200 being
incorrectly rejected as a typed frozen default; it now matches installed sklearn.
An additional fail-first raw-identity regression caught submicrosecond timestamp
corruption being truncated in a microsecond-source identity comparison. The worker
now requires a lossless timestamp round-trip before canonical nanosecond output;
integer-building/microsecond-source preservation is also verified. An interrupted
recording fit retains its debit event and never sends a shard acknowledgement.
All executed estimator fit boundaries are recording stubs. Source fixtures contain
metadata-only bytes; the development SQLite tables are generated directly from
synthetic values. These tests are transport/coverage evidence, **not empirical
SVC/PCA performance or real supervisor/journal completion proof**.

`manual_real_two_arm_smoke(tmp_path)` in `tests/unit/test_full_rbf_svc_worker.py`
is an explicitly named parent-only callable, not a collected pytest test. It is
prepared but **not executed**. Parent may invoke it only after other fitting stops
and sequential smoke is permitted; fuller actual subprocess/durable-journal/
all-24-unit acceptance still belongs to parent integration.

The full-scale severity remains **severe compute risk** with no ETA or promise of
feasibility. This isolated executable backend does not mean the campaign route,
real tiny-fit evidence or full experiment has been admitted/completed.

Final terminal verification (continuation):

```sh
rtk proxy .venv/bin/python -B -m pytest -p no:cacheprovider tests/unit/test_full_rbf_svc_worker.py tests/unit/test_full_rbf_svc_contract.py -q --tb=short
# 39 passed in 5.28s
rtk proxy .venv/bin/ruff check src/project/experiments/full_rbf_svc_worker.py src/project/experiments/full_rbf_svc_contract.py tests/unit/test_full_rbf_svc_worker.py tests/unit/test_full_rbf_svc_contract.py
# All checks passed!
rtk proxy .venv/bin/python -B -m project.experiments.full_rbf_svc_worker --help
# Exit zero; dedicated --work-id / --directory CLI available. No fits launched.
```
