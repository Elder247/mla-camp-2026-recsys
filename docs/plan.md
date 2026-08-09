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

Recency-history 10M gate (`20260809_0515_i2_recent10m_temporal`):

- 1M remains a code/schema smoke; the first metric-bearing experiment uses
  the existing 10M walk-forward OOF artifact and reuses unchanged TF-IDF and
  TwoTower candidates from `20260808_1830_i2_wf10m_temporal`;
- two config-gated generators rank strictly prior query and exact query-region
  clicks by last event time. Existing SourceCost generators keep their old
  ordering, and all temporal generators now emit canonical history provenance
  so the corresponding CatBoost flags are populated;
- the temporal experiment uses a bounded 600-row pool and the accepted
  QueryRMSE loss. Promotion requires an improvement over the matched 10M
  candidate ceiling `0.694238` and best blend SC Recall@50 `0.617587`, plus
  positive source complementarity; only then is the hypothesis repeated on
  100M/full data.
- the next already bounded fallback is mean historical SourceCost per click.
  It removes the frequency multiplier present in summed SourceCost while
  retaining the same past-only state; it will run on 10M only if recency alone
  is not sufficient for direct promotion.
- the complete run took `15m07s`; cache parity passed `40/40` with zero
  mismatches and feature row counts are `7,499,400/2,100,000` for
  train/holdout. TF-IDF reused in about 14 seconds, TwoTower inference took
  `258/75s`, feature construction `262/104s`, and the single QueryRMSE fit
  `35s`;
- recency query/query-region standalone SC Recall@50 is
  `0.416359/0.195767`, below summed SourceCost `0.441894/0.197284`. The merged
  candidate ceiling is `0.694285` at 500, effectively unchanged from the
  matched `0.694238` baseline;
- the cached blend/geometry probes peak at SC Recall@50
  `0.611230/0.613980`, below the old 10M blend `0.617587`. Recency is rejected
  and is not promoted to 100M; the next 10M run tests mean SourceCost per click.

Mean-SourceCost 10M gate (`20260809_0530_i2_mean10m_temporal`):

- the run completed in `11m09s`; TF-IDF reused in `13/14s` and TwoTower in
  `1.6/1.7s`. Parity passed `40/40` with zero mismatches, features contain
  `7,499,400/2,100,000` train/holdout rows, and the single QueryRMSE fit took
  `41s`;
- mean query/query-region SC Recall@50 is `0.418495/0.196648`, again below the
  summed variants `0.441894/0.197284`. The merged SC Recall@500 is exactly the
  matched baseline `0.694238`, so the candidate ceiling did not improve;
- cached blend/geometry peak at `0.616202/0.617111`, slightly below the old
  10M `0.617587`. Mean history is rejected and no redundant 100M candidate
  build is launched;
- both added history variants changed ordering but not membership. The next
  fastest non-redundant test moves to the already cached 100M features and
  compares QueryRMSE on raw SourceCost against the accepted log-SourceCost
  model. This is a ranker-only probe: no candidates or features are rebuilt.

100M ranker-only target/depth/ensemble decision:

- `20260809_0545_i2_rawqrmse_rankonly` reused 919 immutable files (2.08 GB
  logical), trained in `41s` and selected 57 trees. Raw-SourceCost QueryRMSE
  peaks at only `0.634536` after score blending, so the raw target is rejected;
- log-QueryRMSE depth 10 and 6 peak after geometry at `0.645358` and
  `0.647200`; both are below depth 8 `0.649637`, confirming depth 8;
- a bounded 40-combination rank ensemble of the accepted depth-8 QueryRMSE and
  YetiRank models completed in `36.6s`. Equal model ranks blended equally with
  RRF, followed by exponent `0.2` inside top 75, reaches a new honest best
  SC Recall@50 `0.650073` (Recall@50 `0.541560`);
- a narrower refinement reproduced the same exact optimum. The gain over the
  single model is small (`+0.044 pp`) but stable; because a full cached run is
  only minutes, it is promoted as a low-cost private check rather than claimed
  as a structural improvement.

Cached QueryRMSE/YetiRank ensemble full/private evidence
(`20260809_0601_i2_ensemble_full`):

- the corrected ranker-reuse contract deliberately recomputed
  `train_ranker`, `make_submission` and `validate_submission`; 923 upstream
  files (2.94 GB logical) were hardlinked from the completed full donor;
- the full YetiRank fit took `123.0s`, two-model inference `36.7s` and strict
  validation `1.3s`. Both CatBoost artifacts have the same ordered 133-feature
  contract; UnderDeep initialization, stage metrics, summary and finish were
  recorded successfully;
- the 2.62 MB submission (SHA-256
  `2d4dea778e9e7ef5de58fcbc14cd029dd67171513b793cee790e90f9aeacd603`)
  contains exactly 10,000 unique requests and 50 unique indexed banners per
  request, with no null or invalid IDs;
- leaderboard upload `wf100m qrmse+yeti ensemble e02 n75` at
  `2026-08-09 06:19:48 MSK` scored SC Recall@50 `0.6216`, Recall@50 `0.5066`
  and Recall@10 `0.3639`. The private gain over the accepted QueryRMSE model is
  `+0.10 pp`, consistent in direction with the small temporal gain, so the
  ensemble becomes the new personal best;
- subsequent feature/candidate ideas use `1M` only for code/schema smoke and
  the existing `10M` walk-forward artifact for the metric gate. A 100M build is
  allowed only after a non-noisy improvement in candidate complementarity,
  SC Recall@500 and final SC Recall@50.

Natural-pool binary ranker probe (`20260809_0625_i2_binary_rankonly`):

- the binary target is defined only from the natural candidate pool; no clicked
  candidate is injected. The ranker-only run reused 919 validated immutable
  files and trained a 340-tree depth-8 QueryRMSE model in `49.8s`;
- binary CatBoost/blend SC Recall@50 is `0.617312/0.637823`. Its best bounded
  ensemble with the accepted log-SourceCost model reaches `0.649234`, below
  the accepted honest `0.650073`; the hypothesis is rejected without a full
  run.

Controlled TwoTower ordering gate (10M):

- strict chronological training (`20260809_0635_i2_tt_chrono10m_probe`) took
  `71.5s`, candidate export `35.9s`, and reached SC Recall@50/500
  `0.455262/0.654963`;
- the otherwise identical shuffled control
  (`20260809_0640_i2_tt_shuffled10m_probe`) took `81.6s`, export `34.9s`, and
  reached `0.434049/0.662663`. Chronology improves the important top-50 by
  `+2.12 pp`, while shuffle retains `+0.77 pp` at 500;
- the pair is complementary: its oracle-union SC Recall@500 is `0.683531`,
  roughly `+2.09 pp` over the stronger individual source. Therefore neither
  model is discarded. One strict chronological 100M model is promoted as the
  next bounded experiment; the existing shuffled 100M artifact remains the
  control and possible second retrieval source.

Active bounded confirmation:

- detached training of `two_tower_v2_dcn4_mlp3_chrono_100m` started at
  `2026-08-09 06:44 MSK`; its detached supervisor will automatically run the
  retrieval probe against `20260809_0500_i2_history_features_temporal` after
  the immutable artifact is complete. The expected train/export/probe cycle is
  about `15 minutes`; no temporal/full candidate rebuild is started before
  this retrieval gate.

100M chronological retrieval decision (`20260809_0645_i2_tt_chrono100m_probe`):

- strict chronological training processed `21,813,184` rows in `672.8s`
  (`32.4k rows/s`), exported one million candidate embeddings in `35.6s`, and
  finished with validation in-batch accuracy `0.848`;
- against the existing shuffled 100M final model, chronological SC Recall@50
  is `0.538797` versus `0.499927` (`+3.89 pp`) and Recall@50 is `0.476150`
  versus `0.451585`. Its SC Recall@500 is `0.697190` versus `0.704623`, so the
  old model remains useful at the tail;
- complementarity is strong enough to promote: oracle-union SC Recall@500 is
  `0.716937`, chronological-only hits contribute `1.136%` of total SourceCost,
  and mean top-50 Jaccard is only `0.307`;
- the first final evaluation invocation used the old source alias and failed
  before metric computation. Re-running only evaluation with the resolved
  `two_tower_v2_walk_forward` alias completed; neither training nor inference
  was repeated for the accepted report.

Chronological walk-forward promotion:

- the original completed OOF artifact remains untouched. A versioned variant
  manifest points its eight predict-before-update weekly snapshots at the same
  leakage-safe models and switches only the post-training validation/test
  fallback to the chronological 100M artifact;
- OOF requests and 739 MB history events are hardlinked into the variant, so
  config interpolation and provenance remain self-contained without copying
  data. The incomplete first variant and failed `20260809_0700` preflight are
  retained for audit; corrected `...chrono_final_v2` passed the artifact
  contract and all `119` unit tests;
- `20260809_0705_i2_chrono_temporal` is active. A detached supervisor will tune
  only bounded QueryRMSE geometry and QueryRMSE/YetiRank rank ensembles, then
  launch `20260809_0720_i2_chrono_full` only if honest SC Recall@50 exceeds
  `0.650073` and merged candidate SC Recall@500 remains at least `0.700`.

Chronological temporal and cross-pool decision:

- `20260809_0705_i2_chrono_temporal` completed in `23m28s`; parity passed
  `40/40` with zero mismatches. Merge took `116/33s`, features `533/323s`, the
  181-tree QueryRMSE fit `42s`, and evaluation `25s`;
- chronological-only merged SC Recall@500 improved from `0.710200` to
  `0.712886`, but its best QueryRMSE/YetiRank geometry reached only `0.647209`,
  below the accepted `0.650073`. The single-pool supervisor therefore rejected
  full as intended;
- the old and chronological ranked pools were then fused without rebuilding
  candidates, features or models. A bounded temporal grid selected old/new
  weights `0.6/0.4`, RRF constant `10`, and SourceCost exponent `0.2` inside
  top `75`. Honest SC Recall@50/500 is `0.653148/0.719973`, with Recall@50
  `0.549272`; this is `+0.31 pp` over the accepted temporal best and passes the
  cross-pool promotion gate;
- the initial reusable tuner kept every full ordering and took `880s` with
  `20.7 GB` peak RSS. This is safe on the 2 TB VM but is retained only as the
  auditable selection run; production materialization evaluates exactly the
  selected configuration once;
- `20260809_0720_i2_chrono_full` is now running detached. A second detached
  watcher will combine its test ranking with completed old full
  `20260809_0601_i2_ensemble_full`, emit a separate parquet/manifest, and run
  the strict 10,000-request, 50-unique-indexed-banner validation before upload.

Chronological cross-pool full/private result:

- `20260809_0720_i2_chrono_full` completed in about `26.5m`; the expensive
  full feature build took about `10m`, CatBoost fit `53s`, and submission
  inference `37s`;
- the selected old/chronological cross-pool materialization produced exactly
  10,000 unique requests and 50 unique indexed banners per request. Its
  parquet SHA-256 is
  `2c60f676b2e18f6992f97dc2b53e6482f8cbd9d2107618479e22d930fe8d8a0f`;
- leaderboard upload `chrono100m crosspool qrmse+yeti e02 n75` scored private
  SC Recall@50 `0.6262`, Recall@50 `0.5153` and Recall@10 `0.3750`. This is a
  new personal best (`+0.46 pp` SC Recall@50 over `0.6216`), but it is still
  below the `0.65` target;
- a subsequent 42-second cached four-source top-50 ensemble selected on the
  earlier half of holdout reached only `0.63168` SC Recall@50 on the later
  half and was rejected without a full run.

Dataset-size gate for subsequent TwoTower hypotheses:

- the already trained strict-chronological 10M model required `71.5s` for
  training and about `36s` for the one-million-banner export. Its fresh
  candidate probe generated holdout candidates in `73.2s`;
- against the full-data TwoTower baseline, 10M SC Recall@50/500 is
  `0.455262/0.654963` versus `0.518348/0.707873`. The 10M model is therefore
  rejected for promotion even though it is useful for fast hypothesis
  screening;
- 1M remains schema/code smoke only. New model ideas use 10M as a cheap gate;
  only a clearly positive holdout result is retrained on 100M. When complete
  history coverage is needed cheaply, prefer deterministic 10% gradient
  sampling over truncating the history to its first 10M impressions.

TwoTower v3 BPE/multi-positive gate and validation fit (2026-08-09):

- v3 restores a shared 16,384-token BPE vocabulary for query/title/text, adds
  a query-region interaction embedding, widens the towers to 384 hidden and 96
  output dimensions, and uses four DCNv2 cross layers plus three residual MLP
  layers. Batch size is 1,024, temperature is 0.05, and the symmetric
  contrastive loss masks repeated query-region and banner IDs as positives;
- the tokenizer was fitted on the existing chronological 10M corpus in
  `126.1s` (2.56 GB peak RSS). The 10M v3 model trained on 2,180,453 clicked
  pairs in `165.8s` and exported all one million banners in `61.6s`;
- honest 10M retrieval improved over the matching chronological v2 control:
  SC Recall@50 `0.466180` versus `0.455262`, and SC Recall@500 `0.675005`
  versus `0.654963`. Mean top-50 Jaccard against the old full-data source is
  `0.213`, so the source is complementary and passed the 100M promotion gate;
- a strictly temporal low-LR fine-tune of the chronological 100M v2 model used
  only the first 6,499 validation rows and evaluated on the untouched last
  3,500. It took `1.8s` plus `42.1s` export and improved SC Recall@50 from
  `0.538797` to `0.549648`, Recall@50 from `0.476150` to `0.483005`, and
  SC Recall@500 from `0.697190` to `0.700997`;
- SourceCost geometry selected exponent `0.3` inside the first 100 candidates,
  raising honest direct-model SC Recall@50 to `0.575618`. The corresponding
  all-validation full-fit direct submission passed the strict 10,000-by-50
  contract but scored only `0.5525` private SC Recall@50 (Recall@50 `0.4482`,
  Recall@10 `0.2805`). It is rejected as a standalone solution: validation fit
  remains useful only as an additional complementary pool;
- the 100M v3 YT table contains 21,813,184 chronological clicked pairs and was
  prepared in `990.9s` with 548 MB peak RSS. Full v3 training processed all
  pairs in `1621.5s` at 13.45k rows/s, used 1.08 GB peak GPU memory and 2.39 GB
  peak RSS; the one-million-banner 96D export took `62.6s`;
- the honest 100M probe generated holdout candidates in `76.6s`. v3 improved
  chronological v2 SC Recall@50 from `0.538797` to `0.556319`, Recall@50 from
  `0.476150` to `0.493288`, and SC Recall@500 from `0.697190` to `0.716480`.
  Their oracle-union SC Recall@500 is `0.719602`, mean top-50 Jaccard is
  `0.319`, and v3-only hits contribute `0.999%` of total SourceCost;
- all retrieval gates passed. A versioned v3 predict-before-update OOF cycle is
  running on eight weeks. It reuses the accepted deterministic OOF request
  sample and 739 MB past-only history by hardlink, while training fresh BPE v3
  weekly snapshots with 10% gradient sampling. The failed pre-`r1` orchestration
  artifact is retained for audit; no YT merge or user artifact was overwritten.

The cached SC-selected CatBoost variant was also checked privately. Despite
honest temporal SC Recall@50 `0.655090`, its cross-pool submission scored
`0.6257`, below the accepted `0.6262`; it is not retained. The current accepted
private solution remains `chrono100m crosspool qrmse+yeti e02 n75` at SC
Recall@50 `0.6262`.

Feature-rich TwoTower screen (2026-08-09):

- v4 added SourceCost-weighted multi-positive contrastive training. On 10M it
  improved SC Recall@50 from `0.466180` to `0.489119`; on 100M it reached
  `0.554266`, slightly below v3 `0.556319`, but retained complementary hits;
- v5 added Client/Order/Caesar/SKU/price/domain banner embeddings and a second
  independent BannerID hash. Its 10M SC Recall@50/500 was
  `0.475818/0.686159`, with low top-50 Jaccard `0.229` against v4;
- v6 added device/age/gender query-context embeddings and fixed null-valued
  categorical lists. It improved 10M SC Recall@50 to `0.516974`, Recall@50 to
  `0.423593`, and SC Recall@500 to `0.689685`; its new-only hits contributed
  `1.774%` of total SourceCost against v5;
- v7 kept v6 features but used the larger 4,096-example contrastive batch. It
  reached the strongest 10M SC Recall@50 `0.546966` and Recall@50 `0.443588`,
  with SC Recall@500 `0.670947`. Its oracle union with v6 is `0.582055` at 50
  and `0.700336` at 500, so v7 was selected by the bounded complementarity
  gate for exactly one 100M fit;
- the detached sequence is immutable and sequential: v7 100M fit and honest
  retrieval probe, a direct holdout-tuned TwoTower-only submission, then
  leakage-safe predict-before-update OOF, cached three-tower temporal CatBoost,
  promotion gate, and full only on an honest improvement. No two GPU training
  stages overlap.

TwoTower v7 100M, OOF and private ensemble evidence (2026-08-09):

- v7 processed `21,813,184` chronological examples in `2391.7s` at about
  `9.12k rows/s`; one-million-banner export took another `86.5s`. The final
  validation in-batch accuracy was `0.8075`;
- the honest 100M retrieval probe reached SC Recall@50/500
  `0.627165/0.729225` and Recall@50 `0.533847`, versus v3
  `0.556319/0.716480` and `0.493288`. The oracle-union SC Recall@50 was
  `0.646987`, confirming that v7 is both stronger and still complementary;
- standalone v7 full-fit submissions scored only `0.58851-0.59999` privately,
  so direct retrieval is not the final ranking. A cached three-pool RRF of the
  old full ranking, chronological full ranking and v7 candidates improved the
  private best to SC Recall@50 `0.649876`, Recall@50 `0.5340`. The accepted
  robust parameters are weights `0.10/0.30/0.60`, RRF constant `10`, and
  SourceCost exponent `0.1` inside top `75`;
- leakage-safe weekly predict-before-update generation completed all `8/8`
  v7 snapshots in `1652.7s`. Each target week is predicted before its labels
  are attached and before the next update, so CatBoost never sees
  target-dependent positive injection.

Selected-tower CatBoost temporal and cached ensemble gate (2026-08-09):

- `20260809_1455_i4_feature_temporal` completed in `2151.9s`; parity passed
  `40/40` with zero mismatches. It produced 172-column leakage-safe features,
  trained one 400-tree QueryRMSE CatBoost in `54.1s`, and saved default
  PredictionValuesChange importance;
- selected v7 OOF candidates reached SC Recall@50/500
  `0.594602/0.725935`. The merged natural pool reached SC Recall@500
  `0.724841`; CatBoost, fixed blend and RRF reached SC Recall@50
  `0.648592/0.651454/0.640953`. The standalone promotion threshold
  `0.6531475` was not passed, so the original supervisor correctly did not
  launch full;
- a cached four-pool probe then showed that the new CatBoost ranking is useful
  as a complementary source. The refined stable three-useful-pool mixture
  (chronological ranking / v7 candidates / new CatBoost blend) uses weights
  `0.15/0.55/0.30`, RRF constant `5`, and no value-geometry reorder. Its early,
  late and full temporal SC Recall@50 are `0.680696/0.664263/0.673145`, with
  full Recall@50 `0.575835`;
- this cross-pool evidence justifies one bounded full materialization,
  `20260809_1610_i4_feature_full`, launched detached with the temporal-selected
  `366` CatBoost iterations. The full output will be accepted only after the
  strict 10,000-request submission contract and an autonomous private check.
- the full run completed in about `54m`. Its first CatBoost process fitted the
  model but then exposed a full-scope contract bug: SourceCost tree selection
  was requested despite the intentional absence of a full validation pool.
  Commit `c3bf25e` makes full scope preserve the temporal-selected fixed tree
  count; targeted tests passed `18/18`, and resume reused every completed
  candidate/merge/feature stage. The corrected 366-tree fit took `56.2s`;
- default CatBoost importance is led by query-history SC reciprocal rank
  (`11.59`), neural/history reciprocal-rank cross (`3.94`), query-history
  normalized score (`3.57`), query-SC/neural cross (`3.54`) and
  query-region-history reciprocal rank (`3.13`);
- the refined output independently passed 10,000 unique HitLogIDs, exactly 50
  unique indexed non-null banners per row and full-file SHA-256
  `d4bc49bdc311f4f912b9d4425b587ee252c621cb026914b4cff95645f80c9f2f`.
  Private SC Recall@50 was `0.645828` with Recall@50 `0.5391`, below the
  accepted `0.649876`; this submission is rejected rather than blended again.
- a 68.7-second cached two-pool stability grid then combined the accepted
  private-best proxy with the refined pool. The 90/10 variant improved both
  early and late temporal SC Recall@50 by `0.00074/0.00050` and moved private
  only marginally to `0.649887`. The more useful 80/20 variant (RRF `5`,
  SourceCost exponent `0.05`, top `75`) also improved both temporal halves and
  reached private SC Recall@50 `0.650064`, Recall@50 `0.5341`, Recall@10
  `0.3781`. It is the first accepted result above the 65% target; weights with
  no two-half improvement were not uploaded.

LogQ TwoTower gate (2026-08-09):

- v8 keeps the accepted v7 architecture and adds configurable sampled-softmax
  correction. Query-to-banner logits subtract the batch-frequency
  `log Q(banner)` term; the symmetric banner-to-query direction subtracts
  `log Q(query)`. Positive identity uses the raw BannerID where available, so
  hash collisions cannot create false positives. `logq_power` and the
  correction mode live in YAML rather than code;
- the implementation is shared by ordinary, validation-fine-tune and weekly
  walk-forward training. The complete unit suite passed `164/164`;
- the 10M screen trained on `2,180,453` examples in `277.2s` at `7.87k rows/s`
  and exported one million 96D banner embeddings in `89.5s`. Its standalone
  SC Recall@50 decreased from v7 `0.546966` to `0.540453`, while Recall@50
  improved from `0.443588` to `0.449586` and SC Recall@500 improved from
  `0.670947` to `0.690273`;
- complementarity passed the bounded promotion gate: a cached v7/v8 ensemble
  improved honest full-holdout SC Recall@50 by about `1.1 pp` over the matching
  v7-only geometry control, with mean top-50 Jaccard only `0.323`. Exactly one
  100M logQ fit is queued after the active full-run GPU CatBoost stage; its
  honest retrieval probe and cached ensemble grid are already chained as
  detached processes, without overlapping two GPU jobs.

LogQ 100M and accepted private result (2026-08-09):

- v8 processed all `21,813,184` chronological examples in `2402.2s` at
  `9.08k rows/s`; the one-million-banner 96D export took `87.4s`. Final
  validation loss/accuracy were `0.5906/0.8085`, with `2.27 GB` peak GPU
  allocation and `2.57 GB` peak RSS;
- the first probe invocation stopped before computation because its dedicated
  experiment YAML was missing. The failed log is retained; commit `02fcf34`
  adds the candidate-only config, `14/14` targeted tests passed, and the same
  run resumed without retraining. Prepare/generate/evaluate then took
  `2.18/76.26/9.40s` and produced 3.5M candidate rows in 32 partitions;
- honest v8 SC Recall@50 improved from v7 `0.627165` to `0.634545`, and
  Recall@50 from `0.533847` to `0.548129`. SC Recall@500 decreased slightly
  from `0.729225` to `0.724169`; mean top-50 Jaccard is `0.448`, so v8 is used
  as a small complementary source rather than as a replacement;
- a two-half temporal gate selected 90% of the accepted private-best proxy and
  10% v8, RRF constant `0`, SourceCost exponent `0.15` inside top `100`.
  SC Recall@50 gains over the control were `+0.006125/+0.003784/+0.005049`
  on early/late/full holdout respectively;
- the bounded validation full-fit and test-only inference took `93.84s` and
  `32.09s`; final materialization took `4.41s`. Strict validation passed
  10,000 unique HitLogIDs with exactly 50 unique indexed banners per row. The
  full parquet SHA-256 is
  `d21795410bd15523304fbf77f181eb70d8ede72df9adb220a08a722292b59ae4`;
- leaderboard entry `18a772a74dee` (`best90 logq10 rrf0 e015 n100`) scored
  private SC Recall@50 `0.6545213914`, Recall@50 `0.5348` and Recall@10
  `0.3750`. This is the new accepted personal best, `+0.004458` absolute SC
  Recall@50 over the previous `0.6500635747` result.

Half-strength logQ screen (2026-08-09):

- the only near-equivalent cached neighbour changed the accepted blend's RRF
  constant from `0` to `5`. It scored private SC Recall@50 `0.6520615664`
  (entry `283984d38d92`) versus accepted `0.6545213914` and is rejected; no
  weaker cached neighbours are uploaded;
- `logq_power=0.5` on 10M processed `2,180,453` examples in `238.6s` at
  `9.14k rows/s` and exported the one-million-banner index in `84.3s`.
  Its validation loss/accuracy `1.2269/0.6865` slightly improved over the
  matching power-1 run `1.2405/0.6850`;
- against power 1, honest SC Recall@50 improved from `0.540453` to `0.546591`
  while Recall@50 changed from `0.449586` to `0.448729`; SC Recall@500 changed
  from `0.690273` to `0.686787`. Oracle-union SC Recall@50 is `0.595069`, mean
  top-50 Jaccard is `0.415`, and new-only hits carry `0.55%` of total
  SourceCost. The complementary top-50 gain passes the gate for exactly one
  100M power-0.5 fit; no other logQ power is promoted concurrently.

Half-strength logQ 100M/private result (2026-08-09):

- the promoted model processed all `21,813,184` chronological examples in
  `2396.5s` at `9.10k rows/s`; one-million-banner export took `84.8s`.
  Data wait was `31.2%`, peak GPU allocation was `2.27 GB`, and peak RSS was
  `2.56 GB`;
- its honest SC Recall@50/500 were `0.629096/0.729742`, versus
  `0.634545/0.724169` for power 1. Recall@50 was `0.539846` versus `0.548129`.
  The top-50 oracle union reached `0.655003` SC Recall with mean Jaccard
  `0.497`; new-only hits contributed `0.150%` of total SourceCost;
- a two-half gate selected `65%` of the accepted-best proxy and `35%` of the
  half-strength source, RRF constant `10`, SourceCost exponent `0.15` and
  rerank top `75`. Early/late/full temporal gains were
  `+0.000762/+0.000607/+0.000690`;
- validation full-fit and test inference took `91.83s/34.19s`; materialization
  took `4.04s`. The output passed 10,000 unique HitLogIDs, exactly 50 unique
  indexed non-null banners per row, and has full SHA-256
  `596b03087f5e86e82be54b1ba6ca58f49dd4e54e14156755e7f9295cee85e3fb`;
- leaderboard entry `188edcb5bfa2` scored private SC Recall@50
  `0.6532185375`, Recall@50 `0.5325` and Recall@10 `0.3704`. It is rejected
  below the accepted power-1 blend at `0.6545213914`; the 100M artifact remains
  useful only as a complementary candidate source.

Next fast TwoTower screens (2026-08-09):

- commit `2e977dc` adds config-isolated 10M screens for batch size `8192`,
  SourceCost weight power `0.75`, a one-million-bucket second BannerID hash,
  128D output, two hashed CryptaID user embeddings, and the previously unused
  six-category Income context. The full test suite passed `180/180` and the
  Income YQL validated without creating or replacing a table;
- the existing loader spent `31.2%` of wall time waiting for YT. All new
  screens use an order-preserving two-batch prefetch, already used by the OOF
  trainer, so data reads overlap GPU work without changing model semantics;
- the six trials run sequentially and detached. Only an honest SC/Recall and
  complementarity winner can enter one combined 10M check and at most one
  subsequent 100M fit.
