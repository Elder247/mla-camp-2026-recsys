# Architecture

## Data flow

```text
immutable train/val/test/index + frozen retrieval models
  -> prepare_data / temporal request split
  -> prepare_counters (offline or full scope)
  -> one generate_candidates subprocess per source
  -> deterministic parquet union and natural ranker_pool cutoff
  -> chunked feature parquet
  -> CatBoost fit or batch predict
  -> temporal evaluation / strict submission validation
```

Legacy TF-IDF and step3 Two-Tower are loaded through the current solution
protocol. Their mathematics and artifacts remain frozen in Iteration 0. The
new pipeline owns orchestration, caching, validation and reporting.

## Configuration composition

`src/mla_recsys/config.py` merges, in order:

1. `configs/base.yaml` — stage graph and behavior defaults;
2. `configs/paths.yaml` — all absolute VM paths;
3. `configs/splits/temporal.yaml` — immutable time/counter boundaries;
4. `configs/experiments/<name>.yaml` — experiment-only differences;
5. command-line dotlist overrides.

The resolved document is written before computation. Its SHA-256 fingerprint
is part of every stage and output manifest.

## Fixed temporal split

The 10,000 validation click rows contain 9,999 request groups. The group-level
65th percentile fixes the boundary at Unix `1782056217`
(`2026-06-21T15:36:57Z`):

- fit: `ShowTime < 1782056217` — 6,499 requests;
- honest holdout: `ShowTime >= 1782056217` — 3,500 requests.

One request has two rows; it has a single timestamp and remains atomic. Split
validation fails if future data contain a group spanning the boundary.

## Run and artifact contract

```text
runs/<run_id>/
  config.yaml
  manifest.json
  result.json
  logs/<stage>.log
  stages/<stage>.json
  metrics/{candidates,ranker,holdout}.json
  candidates/<split>/<source>/*.parquet
  features/<split>/*.parquet
  models/
  reports/{timing,feature_importance,source_complementarity}.csv
  predictions/{holdout,test}_top50.parquet
```

`manifest.json` records git state, host, GPU, package versions and input
fingerprints without reading secret values. `result.json` is updated atomically.
Each parquet/model has an adjacent output manifest containing config/input
fingerprints, schema, row count, size and content fingerprint. Resume requires
all expected fields and an existing output to match.

## Scope separation

`offline` artifacts are the only ones allowed in hypothesis selection. Their
counters/history are derived from train sources with cutoffs before validation.
`full` artifacts are created under a different path/fingerprint after choosing
the experiment and may use train plus all available validation data for hidden
test inference. Metrics from the two scopes are never merged.

## Iteration 1 candidate histories

The frozen pre-validation `train_100m` history artifact is exposed as three
independent sources: query ordered by clicks, query ordered by direct
SourceCost, and exact query-region ordered by direct SourceCost. Query-region
has no implicit query-only fallback, so its complementarity is measurable.

User, region and global sources are native run-scoped generators. For `train`
and `full_train`, they rank from state observed at strictly earlier timestamps;
all requests sharing a timestamp are ranked before that timestamp is observed.
For `holdout`, state is frozen after `train`; for `test`, it is frozen after
`full_train`. Targets from the evaluated split are therefore never used to
construct its candidate membership. Region/global value scores use configured
minimum support and Bayesian shrinkage; user history uses exact prior clicks.

## Memory and execution

`scripts/run_pipeline.py` is a standard-library orchestrator. It launches each
configured stage as a subprocess, streams combined stdout/stderr into a stage
log and records wall time/peak RSS. Candidate sources run sequentially by
default (`max_parallel_cg: 1`); GPU sources must not overlap.
