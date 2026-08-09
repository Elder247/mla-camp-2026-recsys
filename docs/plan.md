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
| 10M weekly OOF cycle | completed, rejected on private | temporal blend `0.617587`; private `0.5607`, below prior `0.5818` |
| Iteration-0 full leaderboard check | completed, current best | private SC Recall@50 `0.5836`, Recall@50 `0.4818` |
| 100M-history sampled walk-forward | in progress | must beat I0 temporal `0.704875/0.622388` before full promotion |
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
- weekly training completed all eight snapshots in 487 seconds; the artifact
  contains 6,000 OOF requests and 2,180,453 strict-ASOF history events;
- snapshot-aware batch inference, external strict-ASOF query/user/global
  history, OOF/validation disjointness and configured-family counters have unit
  coverage; the full project suite passes 64/64 before supervisors;
- static history and the old full-data tower are excluded from OOF pools;
  temporal/full each train CatBoost once, and full remains gate-controlled;
- val/test reuse the verified full-100M v2 retriever of the same architecture,
  while OOF rows remain bound to their 10M pre-update weekly snapshots;
- pipeline and model tracking use UnderDeep `camp-2026/modern-plumber` with a
  masked local JSONL fallback; a live contract-smoke successfully created a
  run and sent metrics/summary, while only token presence was checked;
- weekly preparation, model training, smoke, temporal promotion and full are
  detached sequential processes, so loss of the local internet connection does
  not stop them.

10M weekly OOF temporal evidence (`20260808_1830_i2_wf10m_temporal`):

- the natural candidate pool contains 500 rows per request with no positive
  injection; train/holdout features contain `6,249,500/1,750,000` rows;
- candidate SC Recall@500 is `0.694238`, above the configured `0.690669` gate;
- one raw-SourceCost CatBoost fit stopped at iteration `178`; fit time was
  `4m39s`, and native importance for all 139 features is persisted;
- CatBoost alone reaches Recall/SC Recall@50 `0.501857/0.612802`;
- a preconfigured scalar rank blend was tested without regenerating candidates
  or refitting the model. Weight `0.75 CatBoost / 0.25 RRF` reaches
  `0.510711/0.617587`, passing the `0.616045` ranker gate;
- the blend probe takes 13 seconds and records wall time, peak RSS and the full
  grid in `metrics/rank_blend_fine.json`; the supervisor selected 179 trees for
  the single full-data CatBoost fit.

10M weekly OOF full/private evidence (`20260808_2030_i2_wf10m_full`):

- the complete full run took `33m13s`; TF-IDF remained the bottleneck at
  `25m56s` for 15,999 full-train requests and `16m36s` for 10,000 test
  requests, overlapped by the scheduler;
- eight-worker merge took `30/18s` for `7,999,500/5,000,000` rows; feature
  construction took `286/191s`; the single 179-tree CatBoost fit took `55s`;
- strict validation passed 10,000 rows with exactly 50 indexed, unique banners
  per row. The 2.76 MB file hash is
  `aca536c7c3c6bf8d0431087ae5da54e279d2e52b64f6efd4aa6cbd094fd529b2`;
- leaderboard submission at `2026-08-09 00:28 MSK` scored SC Recall@50
  `0.5607`, Recall@50 `0.4306`, Recall@10 `0.2519`. It is below the previous
  private best `0.5818` and is rejected despite passing temporal gates;
- the private/temporal gap and lower 10M candidate ceiling show that the
  history reduction is not quality-neutral. The next retrieval hypothesis must
  retain chronological leakage safety while restoring the 100M history scope.

Iteration-0 leaderboard verification (`20260808_0040_i0_full`):

- the strict 10,000-row Iteration-0 artifact was uploaded unchanged and scored
  private SC Recall@50 `0.5836`, Recall@50 `0.4818`, Recall@10 `0.3566`;
- it is the current accepted personal best and confirms that replacing the
  complete train history with 10M is not quality-neutral.

Current fast 100M walk-forward decision:

- all 100M impressions are used to materialize the full chronological click
  stream; OOF labels, query history and ASOF counters therefore retain the
  complete history scope;
- only TwoTower gradient updates are deterministically sampled at request level
  (`10%`, capped at 400k examples per week). Multi-click targets stay together,
  and the sample spans the complete week instead of taking its first rows;
- user history remains disabled because its honest holdout recall was
  negligible. Query, query-region and the cheap global-SC provenance are
  retained; global finds few but disproportionately high-SourceCost clicks;
- the natural ranker pool is 750 rather than 500: in the 10M run the 500-row
  fused pool had SC ceiling `0.694238`, below TwoTower alone at 500
  (`0.701323`). The extra 250 rows affect only the short merge/feature/ranker
  stages, not TF-IDF or TwoTower inference;
- CatBoost is still fitted once per temporal/full scope on the same natural
  candidate pool. A cheap cached CatBoost/RRF alpha probe is now run
  automatically before promotion;
- promotion is stricter than the current best: candidate SC@500 must exceed
  `0.704875` and the selected ranker SC@50 must exceed `0.622388`.

100M walk-forward temporal evidence (`20260809_0050_i2_wf100m_temporal`):

- the writable YT materialization contains `21,813,184` chronological clicks;
  weekly TwoTower updates sample `2,180,888` examples while preserving all
  history events and exactly `6,000` pre-update OOF requests;
- the merged natural pool reaches Recall/SC Recall@500
  `0.633819/0.710200`; standalone CatBoost reaches SC Recall@50 `0.636936`;
- the cached rank-linear probe improves SC Recall@50 to `0.643664`, above the
  previous honest gate `0.622388`, and selects 482 trees for the full fit;
- the first promoted full run exposed a full-scope contract bug before upload:
  untimed test requests were treated as timestamp zero, so every frozen
  query/query-region/global history source emitted zero test rows;
- the leakage-safe fix warms one state from all full-history events and freezes
  it for every unlabeled test request. Candidate reuse now also compares code,
  artifact and data fingerprints, preventing reuse of the old zero-row files;
- the next short ranker hypothesis weights each request by its summed
  SourceCost, clipped at temporal p99 (`236M`) and normalized to mean one.
  Candidate and feature parquet are reused only after strict contract checks;
- TF-IDF request scoring now has an optional fork-based CPU path. A real-model
  64-request smoke produced identical 64,000 rows and a `1.82x` speedup with
  four workers; the full test suite passes `83/83` in an isolated VM checkout.

100M control full/private evidence (`20260809_0230_i2_wf100m_full`):

- the run completed with 11,999,250 full-train and 7,500,000 test feature rows;
  its strict 10,000 x 50 submission contract passed with SHA-256
  `c1149a6f9b74f564eed56fc83b77408cc58c13f2c6a57a8682387906cafc680c`;
- all three test history sources emitted zero rows because untimed test
  requests were evaluated before the frozen state. The control submission at
  `2026-08-09 02:40:52 MSK` scored SC Recall@50 `0.5113`, Recall@50 `0.4337`
  and Recall@10 `0.2544`;
- the result is rejected. It confirms that the temporal/full history mismatch
  is material and that promotion must include a non-empty frozen-history test
  contract check, not only temporal gates.

SourceCost group-weight decision (`20260809_0345_i2_scgw_temporal`):

- the cached temporal run retained the exact natural candidate ceiling at
  SC Recall@500 `0.710200`; train/holdout features contain
  `9,374,250/2,625,000` rows;
- p99-clipped, mean-normalized request SourceCost weights reduced standalone
  CatBoost SC Recall@50 from `0.636936` to `0.632985`;
- a complete cached rank-linear blend grid took 18 seconds and peaked at
  `0.643123` for alpha `0.6`, below the unweighted `0.643664`; the weighting
  hypothesis is rejected and is not promoted to full;
- feature construction took `539/334s` and CatBoost only `135s`. New ideas use
  a progressive `1M smoke -> 10M temporal gate -> 100M confirmation` policy,
  while final quality runs keep the full chronological history;
- UnderDeep is active at `camp-2026/modern-plumber`: candidate/ranker metrics,
  primary SC, timings, peak memory, submission checks and native CatBoost
  importance are backed up locally and sent fail-open without reading tokens.

Fixed-history 100M full/private evidence (`20260809_0410_i2_scgw_fixed_full`):

- the full run used the accepted unweighted temporal model, 482 CatBoost trees
  and the selected `0.6 CatBoost / 0.4 RRF` blend; despite the legacy run id,
  the resolved config records experiment `i2_walk_forward_100m_s10_fast_quality`;
- frozen test history is now present for all 10,000 requests: query,
  query-region and global generators emitted `424,118`, `72,797` and `750,000`
  rows. Temporal cache parity passed `40/40` checks with zero mismatches;
- the end-to-end cached full run took `22m35s`. Parallel parity took `207s`,
  feature construction took `613/472s` for `11,999,250/7,500,000` train/test
  rows, the single full CatBoost fit took `136s`, and submission inference plus
  validation took `29s`;
- strict validation passed 10,000 rows with exactly 50 candidates per request.
  The 2.6 MB artifact SHA-256 is
  `0d6a9e0d5d09c09182a7c800912feba2e861006fff804b53e80ee85e1350e05b`;
- the leaderboard submission at `2026-08-09 03:38:39 MSK` scored SC Recall@50
  `0.6171`, Recall@50 `0.5096` and Recall@10 `0.3673`. This is a new personal
  best, `+3.35 pp` SC over Iteration 0, but remains below the `0.65` target;
- the next cheap hypothesis applies a bounded SourceCost geometry only inside
  the accepted rank order's prefix. It is tuned on the existing temporal
  holdout and does not regenerate candidates/features or refit CatBoost.

Bounded SourceCost geometry temporal decision:

- a 37-combination cached grid completed in `30.2s` on all 3,500 temporal
  requests. The best `0.6` blend with exponent `0.15` inside only the top 75
  improves SC Recall@50 from `0.643664` to `0.646060` (`+0.24 pp`), with
  Recall@50 `0.538703`;
- the hypothesis passes the honest temporal gate and is materialized as a new
  immutable prediction variant. The accepted blend submission is retained
  unchanged for direct private comparison.
- the strict 10,000 x 50 variant (SHA-256
  `33585c7f2f16ab58a6aa1f0e4e2908142bf6478ed3b71508903c2ae21bf57477`)
  scored private SC Recall@50 `0.6184`, Recall@50 `0.5053` and Recall@10
  `0.3650` at `2026-08-09 03:44:08 MSK`. The `+0.13 pp` private SC gain is
  accepted, but its small size rules out geometry-only tuning as the path to
  `0.65`; the next experiment targets strict chronology/history retrieval.

History aggregate fix and cached loss decision (`20260809_0500_i2_history_features_temporal`):

- temporal history aliases now preserve click count and summed SourceCost in
  the common merged contract. On sampled train/holdout feature partitions the
  two numeric history fields are non-zero for about 15% of rows; native
  CatBoost importance also confirms that the signal is used;
- the 100M control completed in about 24 minutes. The corrected YetiRank model
  reaches CatBoost/blend/geometry SC Recall@50
  `0.637776/0.643585/0.646101`; the `+0.004 pp` geometry change is treated as
  noise and is not promoted by itself;
- the safe feature patch path reuses an exact donor only after row count,
  request/banner identity, schema and manifest validation. It replaces only
  four history columns, preserves Arrow nullability, and passed a real
  292,500-row parquet smoke plus the complete test suite;
- experiment sizing is now progressive: `1M` is a code smoke, `10M` is the
  default gate for new candidate/feature hypotheses, and `100M` is reserved
  for confirmation. When exact 100M features already exist, ranker-only probes
  reuse them because this is both faster and less noisy than rebuilding 10M.

Ranker-only loss probes and runtime contract:

- `materialize_ranker_probe.py` validates a completed donor, successful cache
  parity and equality of every upstream semantic config field before
  hardlinking immutable data/candidates/features. It records all reused stages
  with zero wall time and retains donor provenance; changed feature/candidate
  semantics fail closed;
- two full 100M temporal losses then completed sequentially, including both
  blend probes, in roughly five minutes. QuerySoftMax trained in `72s` and
  reached geometry SC Recall@50 `0.649355`;
- QueryRMSE trained in `44s`, selected 185 trees and reached CatBoost/blend/
  geometry SC Recall@50 `0.623318/0.647522/0.649637`. Its geometry result is
  `+0.35 pp` above the previous honest best `0.646101`, so QueryRMSE is the
  accepted loss for promotion. No broad loss/Optuna search was run.

Fast QueryRMSE full/private evidence (`20260809_0450_i2_qrmse_full`):

- the `history_features` reuse profile hardlinked validated full data,
  counters and ten source-candidate stages from the fixed-history donor. It
  recomputed merge and parity, then patched all `32/32` full-train and `32/32`
  test feature partitions instead of rebuilding 19.5M feature rows;
- the full run completed from `04:47:24` to `04:55:37 MSK` (`8m13s`). Merge
  took `149/95s`, parity `214s`, feature refresh `35/21s`, the single 185-tree
  QueryRMSE fit `42s`, and submission inference/validation `32/1s`;
- parity passed `40/40` with zero mismatches. Feature schemas contain
  `11,999,250/7,500,000` rows; `history_click_count_log1p` is the ninth most
  important native CatBoost feature (`2.94` importance);
- strict submission validation passed 10,000 rows with exactly 50 indexed,
  unique banners per request. The immutable 2.76 MB artifact SHA-256 is
  `f8380a30c475f06291e0b5c277ff6962f59474606a58ae574fa2b22130a8debe`;
- leaderboard upload `wf100m qrmse history geometry` at `04:57:58 MSK` scored
  SC Recall@50 `0.6206`, Recall@50 `0.5079` and Recall@10 `0.3647`. It is the
  new accepted personal best (`+0.22 pp` SC over `0.6184`).
