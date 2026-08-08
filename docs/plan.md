# Implementation plan

Source of truth:
`/Users/astrofimuk/REPS/mla_camp_recsys/docs/next_iterations_plan.md`.
This file tracks implementation status only; it does not redefine priorities.

| block | status | gate |
|---|---|---|
| Audit VM/git/data/artifacts/GPU/baseline | completed | clean `main`, H100 visible through torch, schemas and legacy metrics captured |
| Docs/config/run contract | completed | commit `786f69e`; 10 pytest + 10 legacy unittest pass |
| Iteration 0 baseline pipeline | completed | temporal/full runs, natural pool, cache parity, strict 10k submission |
| Iteration 1 candidate generators | in progress | implementation/smoke complete; full temporal complementarity and SC ceiling pending |
| Iteration 1 feature v2 | pending | chunk parity, leakage-safe schema, memory/timing report |
| Iteration 1 SC-aware CatBoost | pending | honest SC@50 win, importance, validated batch prediction |
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
