# Requirements

`/Users/astrofimuk/REPS/mla_camp_recsys/docs/next_iterations_plan.md` is the
source of truth. This file extracts the acceptance criteria that the VM
implementation must enforce.

## Objective and decision metric

The system remains a two-stage recommender: independent candidate generators,
deterministic union/deduplication, CatBoost, optional post-processing, top 50.
Experiments are selected on the fixed temporal holdout by SourceCost Recall@50.
Recall@50, SC@10, SourceCost quantiles, candidate complementarity, latency and
memory are diagnostics/tie-breakers. AUC, train loss and in-batch accuracy are
never sufficient acceptance criteria.

## Fixed baseline

The frozen baseline uses TF-IDF (quota 1000, RRF weight 0.75), sampled-softmax
Two-Tower step3 (quota 1000, weight 1.25), history (quota 200, weight 2.0), a
2200-candidate union and a 500-candidate CatBoost pool with 37 features.

The legacy 4,999-request pseudo-holdout metrics are retained only for parity:

| model | Recall@50 | SourceCost Recall@50 | SourceCost Recall@500 |
|---|---:|---:|---:|
| RRF | 0.5363 | 0.6658 | 0.7519 |
| CatBoost | 0.5377 | 0.6680 | 0.7519 |

They are not valid model-selection metrics after the temporal contract lands.

## Iteration 0 acceptance

- Every command is config-driven and writes to `runs/<run_id>/`.
- Resolved config, environment/input manifest, logs, timings, peak memory,
  candidate/ranker metrics, output schemas and row counts are persisted.
- Every heavy stage is a subprocess. Resume is allowed only after config/input
  fingerprint and schema validation.
- Train and inference use the same natural candidate pool and cutoff. Targets
  never affect membership. Groups without a positive stay in candidate metrics
  but are excluded from CatBoost fit as `missed_positive_groups`.
- The fixed chronological split is used for all model-selection experiments.
  All rows of a request stay together and train/holdout request IDs are disjoint.
- Offline counters use only data before the row/snapshot cutoff. `offline` and
  `full` scopes are separate artifacts and cannot be interchanged.
- Source and merged candidates are cached as deterministic parquet partitions.
- Recall and SourceCost Recall are reported at 1, 5, 10, 20, 50, 100, 500 and
  1000, together with source complementarity and pool ceiling.
- Cached and direct paths return identical smoke top-50; identical seed/config
  runs have identical schemas/row counts and numerically stable metrics.
- Submission validation checks 10,000 unique HitLogIDs, unique BannerIDs, list
  length, dtypes, nulls and membership in the one-million-banner index.

## Iteration 1 acceptance

- Query click, query SourceCost, query-region, user, regional SourceCost and
  global SourceCost generators are measured independently. A source stays only
  if it improves union SC ceiling, adds valuable unique hits, helps a weak
  segment, or has verified ranker value without hurting honest SC@50.
- Feature v2 is built in bounded chunks from the natural pool. Retrieval,
  context, text/static, past-only counters, missingness and cross features are
  versioned in the artifact schema.
- Raw SourceCost is present. Direct-SC ranker labels/weights and CTR*SourceCost
  are controlled experiments against the old log-SC baseline.
- Feature importance and complementarity reports exist before full refit.
- Batch retrieval and CatBoost inference pass parity tests against the
  single-request path.

## Operational and security requirements

- Paths, quotas, pool sizes, features, seeds, losses and model parameters live
  in YAML. Absolute VM paths live only in `configs/paths.yaml`.
- `common/` is a stable external contract and is not refactored.
- Existing models, parquet files and predictions are immutable inputs unless a
  new versioned artifact is explicitly created.
- Token values are never read, printed, logged or committed. Code may check only
  whether a required file/environment variable exists.
- Every production smoke/temporal/full run is linked to UnderDeep project
  `camp-2026`, experiment `modern-plumber`. Tracking is fail-open and always
  keeps a masked local JSONL backup, so a tracking/network outage cannot stop
  model computation.
- Data, model binaries, large parquet files and run directories are gitignored.
