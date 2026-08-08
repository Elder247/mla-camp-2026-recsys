# Implementation plan

Source of truth:
`/Users/astrofimuk/REPS/mla_camp_recsys/docs/next_iterations_plan.md`.
This file tracks implementation status only; it does not redefine priorities.

| block | status | gate |
|---|---|---|
| Audit VM/git/data/artifacts/GPU/baseline | completed | clean `main`, H100 visible through torch, schemas and legacy metrics captured |
| Docs/config/run contract | completed | commit `786f69e`; 10 pytest + 10 legacy unittest pass |
| Iteration 0 baseline pipeline | completed | temporal/full runs, natural pool, cache parity, strict 10k submission |
| Iteration 1 candidate generators | completed, rejected by gate | temporal merged SC@500 `0.686766` is below I0 `0.704875` |
| Iteration 1 feature v2 | completed | full temporal schema, memory and timing verified |
| Iteration 1 SC-aware CatBoost | completed, rejected by gate | temporal CatBoost SC@50 `0.616045` is below I0 `0.622388` |
| Iteration 1.1 fast value | completed, rejected by honest gate | temporal completed in 40 minutes; RRF/CatBoost SC@50 `0.609868/0.594359` |
| Iteration 1.2 fast quality | completed, not promoted | honest RRF/CatBoost SC@50 `0.613506/0.613103` |
| TwoTower v2 full | completed | 21.8M pairs; probe SC@500 `0.707873`, SC@50 `0.518348` |
| 10M weekly OOF cycle | running | 8 weeks/2.18M clicks; smoke → temporal → gated full detached chain armed |
| Iteration 2/3 | blocked by design | do not start before Iterations 0/1 reproduce |

Implementation proceeds in small commits matching these blocks. Architecture
changes update `docs/architecture.md`; command changes update
`docs/commands.md`; scope/priorities change only in the source-of-truth plan.

Iteration 0 smoke evidence:

- `20260807_2300_i0_smoke` and `20260807_2330_i0_smoke_repeat` completed;
- both produced 36,459/36,271 merged train/holdout rows and 10,000 feature rows per split;
- natural positives: train 16/20, holdout 11/20; misses stayed out of fit;
- candidate and ranker metrics were identical across runs;
- direct and cached top-50 matched for all 40 requests in the repeat run.

Iteration 0 temporal evidence (`20260807_1845_i0_temporal`):

- fixed split: 6,499 fit and 3,500 holdout request groups;
- natural top-500: 4,305/6,499 fit groups and 2,198/3,500 holdout
  groups contain a retrieved positive; 2,194/1,302 misses remain uninjected;
- holdout RRF Recall@50 / SC Recall@50: `0.517566 / 0.610910`;
- holdout CatBoost Recall@50 / SC Recall@50: `0.523279 / 0.622388`;
- candidate SC ceiling at 500: `0.704875`; source complementarity and feature
  importance are persisted in the run;
- 17/17 post-run tests passed; sampled direct/cache parity checked 40 requests
  with zero mismatches; 323 parquet manifests have two candidate schemas and
  one feature schema.

Iteration 0 full evidence (`20260808_0040_i0_full`):

- full-train/test merged rows: 18,193,769 / 18,201,657; feature rows:
  4,999,500 / 5,000,000;
- full sampled direct/cache parity: 40 requests, zero mismatches;
- batch prediction: 10,000 HitLogIDs, exactly 50 unique indexed BannerIDs per
  row; strict validation `ok=true`, zero short/unknown/error rows;
- a first validation correctly rejected 10,661 unknown values originating only
  from stale history rankings; commit `12f0c33` now rejects history candidates
  absent from frozen index metadata, and the resumed history cache has zero
  unknown IDs;
- final run contract contains 327 output manifests, one feature schema and two
  candidate schemas; the run occupies 1.3 GiB and records a peak pipeline RSS
  upper bound of 8.27 GiB;
- the full run records its initial SHA plus append-only clean resume SHAs and
  finishes with `status=completed` and no stale error.

Iteration 1 candidate smoke evidence (`20260808_0815_i1_cg_smoke`):

- eight enabled sources materialize into 576 source/merged parquet partitions;
- query-click, query-SC and exact query-region use frozen pre-validation history;
  user/region/global use walk-forward train state and frozen holdout state;
- direct and cached 8-source top-50 match on 40/40 requests with zero
  mismatches; all enabled sources have provenance columns in the merged schema;
- 28/28 post-smoke tests pass; 644 manifests have two candidate schemas and
  one feature schema;
- the 20-request sample is contract evidence only. User/region/global coverage
  is zero on the first 20 fit requests and must be judged on the full temporal
  holdout; no source is accepted from smoke metrics.

Iteration 1 feature-v2 implementation evidence:

- `feature_v1` order and baseline behavior remain unchanged;
- v2 includes raw/static value, group/domain, context, retrieval agreement,
  URL/text and configured counter/cross families;
- offline/full counter artifacts have separate paths and manifests;
- strict ASOF unit tests exclude same-timestamp targets and verify frozen
  inference cutoff semantics;
- counters are documented as click/event counts because no impression stream
  is available; no unsupported CTR feature is synthesized;
- resolved I0/I1 schemas contain 37/276 unique ordered features;
- 35/35 tests pass before the feature-v2 VM smoke.

Iteration 1 CatBoost implementation evidence:

- `ranker_raw_sc_label` consumes raw SourceCost divided only by the configured
  global scale; a unit test distinguishes it from the log-SC control;
- metadata records the exact label column and scale;
- the isolated train stage writes native CatBoost `PredictionValuesChange`
  importance. SHAP and permutation importance were removed before the full
  temporal run to avoid a redundant groupwise analysis pass.

Execution reliability follow-up:

- Linux stage RSS now comes from per-process `/proc` sampling and records its
  measurement method, rather than inheriting an earlier child maximum;
- a disconnected SSH/stdout consumer no longer raises `BrokenPipeError` in the
  orchestrator; stage output continues into the durable run log.

Acceleration contract prepared for the selected I1 full run:

- configured three-worker scheduler overlaps only independent stages, with a
  cross-process run metadata lock and at most one GPU generator per group;
- train/test merge and feature stages run as split pairs;
- Two-Tower batch inference preserves exact single-request top-50 IDs and
  scores on 4/4 real smoke requests;
- only native CatBoost importance is produced in either scope;
- 41/41 tests pass in an isolated VM copy without changing the active temporal
  run.

Iteration 1 feature/ranker smoke evidence (`20260808_1230_i1_v2_smoke`):

- completed end-to-end at `2026-08-08T08:37:34+03:00`, after a cache-safe
  resume from the deliberate no-`PYTHONPATH` failure gate;
- counter artifact has 20 click events and the offline frozen cutoff;
- train/holdout feature outputs contain 20,000 rows each, one identical schema,
  276 unique ordered features and zero NaN/Inf values;
- cache parity covers 40/40 requests with zero mismatches; 36/36 post-run tests
  pass;
- raw-SC CatBoost metadata records `label_raw_sc` and scale `1,000,000`; the
  smoke run produced the earlier diagnostic reports, while production runs now
  keep only native CatBoost importance;
- feature stages take 15.5/15.4 seconds with 2.66 GiB stage RSS; training and
  reports take 7.3 seconds with 661 MiB RSS;
- tiny-sample RRF/CatBoost SC@50 is `0.510672/0.504502`. This is contract-only
  evidence and is not used to accept or reject I1.

Iteration 1 temporal evidence (`20260808_1245_i1_temporal`):

- all eight generators, both merges, cache parity and both 276-feature splits
  completed; train/holdout feature construction took `2293.1/1225.4` seconds;
- CatBoost fit used 4,367,000 natural-pool rows and 2,206,000 validation rows;
  best iteration is `845`, and only native `PredictionValuesChange` importance
  is retained;
- merged candidate SC Recall@500 is `0.686766`, below the I0 gate
  `0.704875`; CatBoost SC Recall@50 is `0.616045`, below `0.622388`;
- promotion status is `gate_rejected`; the I1 full refit was intentionally not
  launched, so the verified I0 full submission remains the accepted model.

Iteration 1.1 fast-value decision:

- user history has zero holdout recall; region history contributes no unique
  clicked banners; query-region and global sources add only about `0.04%` of
  unique SourceCost, while duplicated query-click/query-SC sources dilute RRF;
- four counter families (region, domain, group, query) retain `6.775/7.353`
  total counter importance; six low-value families and cross features are
  removed;
- an initial RRF grid over cached I0 candidates selected TF-IDF weight/quota
  `0.25/500`, Two-Tower `1.25/750`, history `2.0/200`. That estimate was not
  accepted because the cached I0 history contained banners later rejected by
  the frozen one-million-banner submission index;
- the run uses 151 features, a 500-row pool, safe three-worker scheduling and a
  three-hour configured wall budget. Temporal selection chooses the better of
  RRF and CatBoost; an RRF-selected full run skips features and model fit.

Iteration 1.1 temporal evidence (`20260808_1230_i1_fast_temporal`):

- the complete temporal pipeline finished in `39m57s`; merge train/holdout took
  `711/379s`, features `795/440s`, and CatBoost only `142s`;
- the valid history source has 129,344 holdout rows versus 151,025 in pre-filter
  I0. The removed 21,681 rows are stale banners outside the submission index,
  so restoring them would violate the run/submission contract;
- RRF/CatBoost SC@50 is `0.609868/0.594359`; candidate SC@500 is `0.691417`;
  the honest gate rejected full refit as designed.

Iteration 1.2 fast-quality decision:

- restore TF-IDF/Two-Tower depth to `1000/1000` while keeping the ranker pool at
  500; the extra retrieval depth affects fusion but not feature-table size;
- return query-click, query-SourceCost and query-region sources because native
  I1 importance assigns them about `6.74`, `18.45` and non-zero importance,
  while user/region/global generators remain excluded;
- keep only region/domain/group/query counter families and omit cross features;
- use the I1 `ranker_raw_sc_label` objective with 900 iterations, covering its
  observed best iteration 845 without the original 1500-iteration ceiling;
- a corrected RRF grid using only index-valid history selects `K=40`, weights
  TF-IDF `0.25`, Two-Tower `1.0`, legacy history `3.0`, query-SC `2.0`; query
  click/region remain feature-only at zero RRF weight. Expected honest RRF
  SC@50 is `0.613506`; CatBoost is the primary improvement hypothesis.

TwoTower v2 implementation evidence:

- the existing 100m-derived prepared table has 21,813,184 clicked pairs and is
  retained in full; the earlier 10m subsampling idea was rejected because the
  baseline full epoch takes only minutes and data loss is not justified;
- six independent field embeddings use configurable
  `ceil_to_8(6 * n ** 0.25)` dimensions capped at 96;
- both towers use four full-matrix cross layers, three deep
  Linear/LayerNorm/GELU layers and a normalized 64-dimensional output;
- the real-YT smoke trained two BF16 steps, used 0.85 GB peak GPU memory,
  exported a `[1000, 64]` index and produced 4/4 exact batch-vs-single top-50
  matches with 50 unique banners each;
- 51/51 project tests pass before full-data training. The old Step3 artifact is
  untouched and the new trainer refuses to overwrite non-empty artifacts.

10M weekly OOF implementation evidence:

- raw `train_10m` is exactly 10,000,000 rows and yields 2,180,453 clicks across
  eight contiguous weeks; its read-only week-stat query completed in 7.6s;
- the lifecycle contract is `predict -> freeze_pool -> attach_labels -> update`;
  model plus optimizer checkpoints make it resume-safe by week;
- 750 deterministic OOF request groups per week and a compact full click-event
  stream are produced in one pass;
- snapshot-aware batch inference, external strict-ASOF query/user/global
  history, OOF/validation disjointness and configured-family counters have unit
  coverage; the full project suite passes 64/64 before supervisors;
- static history and the old full-data tower are excluded from OOF pools;
  temporal/full each train CatBoost once, and full remains gate-controlled;
- pipeline and model tracking use UnderDeep `camp-2026/modern-plumber` with a
  masked local JSONL fallback; the required client is installed and only token
  presence was checked;
- weekly preparation, model training, smoke, temporal promotion and full are
  detached sequential processes, so loss of the local internet connection does
  not stop them.
