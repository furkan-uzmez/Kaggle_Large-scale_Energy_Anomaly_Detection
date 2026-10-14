# Remaining feature experiments: provenance audit v1

Audit date: 2026-10-05. Scope: source code, notebook **source cells only**, existing schema/lineage metadata, downloaded competition description, and its linked upstream code. No raw/prepared/test/vault row data was opened, no population audit or fit was run, and no producer, registry, configuration, or admission was changed. Parent owns validation and promotion.

## Actionable disposition

| Exact experiment | Disposition | Evidence and required dependency |
| --- | --- | --- |
| `feat_core_vs_provided` | **Blocked: no authenticated additional independent raw producer** | Seven supplied weather fields are plausible additional measurements, but their exact LEAD export producer and observed-hour release/revision policy are missing. Supply the actual generation/export notebook or script, immutable source version, source input bindings and availability evidence, then obtain parent approval of the ordered additional pool. Do not substitute core copies, holiday/calendar derivatives, GTE or history. |
| `feat_group_aggregates_blocked` | **Blocked: point-in-time and unseen-building history policy absent** | Linked upstream weather windows are centered, not past-only; full-year GTE aggregates have no cutoff. Supply a causal producer with declared event/availability cutoffs, fold membership, history access policy, cold-start fallback and parent-approved semantics. Existing downloaded aggregate columns cannot be promoted by name. |
| `blocked_unverified_gte_family` | **Blocked: authentic upstream unsafe construction found; exact LEAD export remains unbound** | Upstream GTE encodes full-training `log1p(meter_reading)` in-sample with no grouped cross-fit or time cutoff. This is not proof of direct LEAD `anomaly` dependence, but it does not satisfy campaign fold/PIT boundaries. Supply LEAD-specific generation/export lineage and actual fit population, bad-reading mask producer, priors and cutoff evidence; otherwise exclude. A newly recomputed causal feature is a separately reviewed recipe, not authentication of the supplied values. |

All three remain non-runnable. No `full_provenance_contracts.py` or speculative producer tests were created because no approved leakage-safe supplied producer was established.

## Prediction contract and authority

The project classifies the current observed hour, keyed by `(building_id, timestamp)`, with positive `anomaly=1`, no forecast horizon, building-disjoint development folds and a fixed vault. This is recorded in `notebooks/01_data_understanding.ipynb`, cells `lead-features`, `lead-decisions`, and Notebook02 cell `cand-inherited`. The current reading is allowed; this does **not** allow future readings, held-out labels, or fitted statistics over held-out rows. Source arrival/revision times and history release delay remain unspecified. Do not replace the frozen grouped evaluation with a chronological protocol in this evidence lane.

`src/project/features/registry.py:142-263` defines only nine verified core predictors. `src/project/features/provided.py:101-103` returns an empty repository producer map. Its resolver requires actual bounded notebook/cell evidence and independently rejects duplicate-only pools, GTE, history, identifiers and derived raw substitutes (`:79-96,155-188`). `availability.py:19-44` requires verified provenance, independent target dependence and availability by prediction time. Metadata declarations alone do not establish these facts.

The 2026-10-04 decision log, `docs/superpowers/specs/2026-10-04-full-campaign-decisions.md:49-55`, explicitly preserves the group/GTE blocks; the full design's eligibility table (`:72-96`) preserves the provided-pool block. This report does not supersede those approvals.

## Authentic accessible upstream evidence

The downloaded competition description `kaggle_competation_pages/data.md:40` links the ASHRAE winning solution preprocessing script as the feature reference. This is an authentic source link, not an exact LEAD CSV build manifest.

GitHub's commits API for that script returned commit **`ccfb4d627645e0379d811e127bcd6ae346999aa9`**, author/committer date **2020-05-14T11:38:16Z**, message `rank1 revised files`. The API reports an unsigned commit; immutable addressing is not a cryptographic author signature. Both source files below were fetched successfully at that commit and SHA-256 was computed directly on fetched bytes:

| Source | SHA-256 |
| --- | --- |
| [02_preprocess_data.py](https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/blob/ccfb4d627645e0379d811e127bcd6ae346999aa9/solutions/rank-1/scripts/02_preprocess_data.py) | `e44a99c219f673ffd310f64cab223d3c0b4cc690c7cba1243c28d6d07426d784` |
| [ashrae/encoders.py](https://github.com/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/blob/ccfb4d627645e0379d811e127bcd6ae346999aa9/solutions/rank-1/ashrae/encoders.py) | `f78ad95b0b8a6b9e7b9a5577061441bb5d25fdd950547c452e9222b84abe3472` |

Commit metadata endpoint: `https://api.github.com/repos/buds-lab/ashrae-great-energy-predictor-3-solution-analysis/commits?path=solutions/rank-1/scripts/02_preprocess_data.py&per_page=1` (queried on audit date).

### Source-level leakage and lineage table

| Check | Concrete source evidence | Verdict / implication |
| --- | --- | --- |
| GTE input target | Preprocessing `:179-190` sets training `target=np.log1p(train.meter_reading)`, selects `good_train` by `is_bad_meter_reading`, fits on that table, transforms training and test. | Confirmed measurement-derived upstream aggregation, **not** demonstrated LEAD anomaly-label encoding. Current-row measurement availability does not justify full-population aggregate state. |
| GTE fitting boundary | Encoder `:38-45` groups all passed rows; `:79-81` implements `fit_transform` as fit then transform on the same frame. No timestamp cutoff or fold exclusion occurs. | Confirmed in-sample/full-population construction in this upstream implementation; future observations can enter historical group statistics and no campaign validation/vault boundary is enforced. Exact LEAD fit population unknown. |
| GTE priors and missing groups | Encoder `:31-36,64-75` uses frame target mean or existing prior columns, and substitutes that prior for missing/nonfinite mapped values. Preprocessing `:181` sets test target to upstream training mean. | Fallback also depends on upstream fitted/global population. Requires provenance for each prior, not merely top-level column names. |
| Weather aggregate timing | Preprocessing `:75-86` uses `rolling(..., center=True)`; `:144-146` calls windows 7 and 73. The fields called max/min are 0.95/0.05 quantiles, not extrema. | Confirmed non-causal centered producer logic; a `lag` suffix is misleading evidence of past-only access. Not admissible under an observed-hour cutoff without a different authenticated producer. |
| Weather cleaning | `process_weather` `:43-72` includes timezone offset shifts, bidirectional spline interpolation and whole-frame median fallback. | The function is not inherently PIT-safe. **Important:** main `:141-142` discards its returned frame, so this script alone does not prove its returned interpolation values were saved. Offset mutation happens in place; exact LEAD export and applied cleaning remain unknown. |
| Static/calendar additions | Preprocessing `:89-123` creates cyclic/calendar interactions, subtracts 1900 from construction year, concatenates building identifiers and creates US federal holiday flags. | These are derivatives or identifier interactions, not independent additional raw measurements. A valid deterministic holiday recipe still cannot be relabeled a raw provided producer. |
| Exact export bridge | Upstream `:205-209` writes `train_clean.pkl`/`test_clean.pkl`; local contract names LEAD CSVs. | Missing LEAD extraction, building selection, label attachment, meter/unit conversion, dtype reduction and CSV export commitment. Matching names and a competition hyperlink do not establish byte-level generation lineage. |

The unsafe upstream logic is demonstrated; whether every supplied LEAD value was produced with that exact logic is **insufficient evidence**, not a confirmed CSV-specific leakage finding.

## Additional raw pool investigation

`configs/data_contract.yaml:37-116,117-196` and existing `outputs/lineage/row_lineage.json` `source_schemas` list seven weather measurements: `air_temperature`, `cloud_coverage`, `dew_temperature`, `precip_depth_1_hr`, `sea_level_pressure`, `wind_direction`, `wind_speed`. They are additional to core and plausible target-independent physical sources. None has a LEAD-specific upstream export binding, source arrival timestamp, revision policy or verified repository producer. Even observed-hour weather can arrive late or be retrospectively corrected/imputed. Therefore none is approved here.

Other metadata fields do not close the raw-pool gap: `site_id` is an identifier; building static attributes are already core; `meter_reading` and `anomaly` in feature files are documented duplicate copies; calendar/cyclic/holiday/interactions are derived; weather windows and GTE are aggregated. The observed source schema does not list an independent supplied `meter` field even though upstream code uses it.

Lineage metadata dated `2026-09-29T08:52:08.400419+00:00` records CSV byte hashes, schemas, row counts and one-to-one join validation. Those attest supplied inputs and local joining, **not** original feature generation or release timing. Its `hashes_validated=true` is a historical metadata claim; raw data hashes were not recomputed in this audit.

Notebook01 cell `lead-feature-audit` only inventories names and assigns pending/blocked status. Cell `lead-setup` loads existing lineage/schema; neither produces supplied features. Notebook02 cells `cand-inherited`, `cand-baselines`, `cand-matrix` preserve safety blocks; they are not production provenance. No bounded real producer cell was found in the two top-level notebooks' source cells.

### Probe/vendor/smoothing names

`tests/unit/test_training_contract.py:220-244` explicitly parameterizes `gte_probe`, `gte_vendor_raw`, `gte_reading_smooth` as **unapproved predictors**, creates synthetic values `[1.0,2.0,3.0]`, and asserts rejection. `tests/unit/test_domain_shift.py:44` creates another synthetic `gte_probe` for a domain diagnostic fixture. None of these names appears in the authoritative supplied schema metadata or constitutes authentic production lineage. There is no evidence to conclude these fixture fields use anomaly labels, future readings or holdout data; equally there is no basis to promote them. Bare `vendor_raw`/`reading_smooth` cannot inherit provenance from test aliases.

## Precise artifacts needed before any promotion

1. **Export authentication:** actual LEAD producing script/notebook with immutable commit or exact byte hash, real notebook cell IDs where applicable, source input hashes and build parameters; mapping from source records to the existing LEAD feature CSV hashes. Explain ASHRAE-to-LEAD building/time/unit mapping and every upstream cleaning step. A downloaded-file hash without this bridge is insufficient.
2. **Availability matrix:** per requested field, event time, release/arrival time or authoritative release policy, timezone, latency, revisions/backfill and missing-data handling; prove all inputs available by the observed-hour cutoff. No fabricated timestamps.
3. **Group/history semantics:** approved past-only window boundary, eligible source cohort and exclusion of future/vault/test rows; distinguish train-fitted population state from held-out-building inference-time history. Building-disjoint validation does not by itself forbid all unlabeled past observations, but access must be explicitly approved and replayable. Require deterministic cold-start/unseen-group fallback and same-time peer availability evidence.
4. **GTE-specific binding:** establish actual encoded response, bad-reading-mask origin, fit population and each hierarchical prior. Existing supplied full-training values cannot be made fold-local by filtering columns after the fact. If recomputation is authorized, state whether it encodes readings or anomaly labels and register it as the correct derived family.
5. **Parent review:** only after evidence establishes a safe producer propose the minimal typed contract with exact source hashes/cell bindings and ordered raw additions. Parent then owns registry/config/admission changes and full verification. This audit deliberately proposes no producer instance.

Future narrow contract tests should reject missing/tampered source hashes and fictitious cell IDs; reject duplicate core/derived/GTE/history substitutions in raw pools; prove future-row and held-out-label changes cannot change allowed historical features; prove only outer-train rows influence fitted state; check timestamp/availability inequality, unsorted/timezone inputs, cold starts and missing groups. These are proposed acceptance tests, not tests executed or claimed passing here. No estimator fit is necessary to test these invariants.

## Local evidence snapshot hashes

These hashes were computed from source/metadata bytes during this audit (not from dataset rows). They permit the parent to detect subsequent changes; they do not certify upstream authenticity.

| File | SHA-256 |
| --- | --- |
| `configs/data_contract.yaml` | `ae8943be35751d344ca35538fe7657f503e50dfc8d56aad6b3bb0660335c8c17` |
| `outputs/lineage/row_lineage.json` | `24daaf6bf12893ac361914fdb0440bb395be614ca78a223f157ced0c20a14fa0` |
| `src/project/features/registry.py` | `4ddb08e2c24b86f9285cb27905b25395e20a2daa2074775e3f4b7ac1a65c2a58` |
| `src/project/features/provided.py` | `f6beed15969dbdb9118b40ff9ae5e32cd0b12ca3b08bb1c89de4f9cebc8acb19` |
| `src/project/features/availability.py` | `bfb0b6683f8a3cd46c936c1681bbf821ee7ced217cc64adbab1ddcfcbf43c30b` |
| `notebooks/01_data_understanding.ipynb` | `354555702f0567d1be2b63cf82c0c24443a79cbd0bcbfb557f505d52e669f87c` |
| `notebooks/02_baseline_and_candidate_design.ipynb` | `218ae1440df3bb33aaadad137eba955426108f3d8d5c2e692311574b0231924b` |
| `kaggle_competation_pages/data.md` | `d915143d0d018d1034a6e563ca035936b9536d0bdfef51d4594b145f8cf49db5` |
| `tests/unit/test_training_contract.py` | `7bd3fb415b2e4ea1b4c50d755ed6086834a0c5c686b460723a97c3051ed4c0fe` |

Verification performed: static source/cell inspection, successful pinned upstream retrieval and direct source/metadata SHA-256 computation. No production code added, so no test suite or estimator execution was required. Residual uncertainty is the missing LEAD generation bridge and point-in-time release policy, not a lack of feature names.
