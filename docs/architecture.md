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
Text normalization used by feature/counter infrastructure is a project-owned
copy of the frozen `common.text` contract, so detached non-login runs do not
depend on an ambient `PYTHONPATH`; `common/` itself remains unchanged.

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
  underdeep_run.json
  underdeep_metrics.jsonl
```

`manifest.json` records git state, host, GPU, package versions and input
fingerprints without reading secret values. `result.json` is updated atomically.
Each parquet/model has an adjacent output manifest containing config/input
fingerprints, schema, row count, size and content fingerprint. Resume requires
all expected fields and an existing output to match.

Ranker-only experiments may use `materialize_ranker_probe.py`. Reuse is not a
blind directory copy: the donor must be completed, cache parity must be green,
and the normalized upstream config must be identical. Files are hardlinked
with a copy fallback and every synthetic completed stage records
`reused_from`. The `history_features` profile stops at source candidates so a
changed merge/history contract is recomputed and validated before training.

The orchestrator creates one UnderDeep run in project `camp-2026`, experiment
`modern-plumber`. Stage wall/RSS values, temporal candidate/ranker metrics and
the configured top native CatBoost importances are uploaded.
`underdeep_metrics.jsonl` is written before every remote call. UnderDeep uses a
buffered `StopSendingData` policy and a bounded finish timeout; missing client,
token or network therefore degrades to local-only tracking.

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

Temporal query and exact query-region history also support independent
recency-ranked and mean-SourceCost variants. They use the same strictly-past
state and candidate universe as the summed-SourceCost variants, but order
banners by the last event time or SourceCost per historical click. The score
policy is an explicit `score_mode` in experiment config; adding a variant
therefore cannot silently change an existing generator. Variants are disabled
by default and must pass the 10M complementarity/SC gate before a 100M
confirmation.

User, region and global sources are native run-scoped generators. For `train`
and `full_train`, they rank from state observed at strictly earlier timestamps;
all requests sharing a timestamp are ranked before that timestamp is observed.
For `holdout`, state is frozen after `train`; for `test`, it is frozen after
`full_train`. Targets from the evaluated split are therefore never used to
construct its candidate membership. Region/global value scores use configured
minimum support and Bayesian shrinkage; user history uses exact prior clicks.

## Iteration 1 features and counters

`feature_v1` remains byte-for-byte ordered for the Iteration 0 baseline.
`feature_v2` adds configured groups for retrieval agreement, request context,
candidate static metadata, text/URL matching, past-only counters and crosses.
Raw `SourceCost` is present alongside its log transform; `GroupExportID`,
ClientID and URL domain are derived from the immutable one-million-banner
index. Group/domain population and value statistics depend only on that static
index and do not use validation labels.

The available validation history is click-only: it has no trustworthy
non-click impression stream. Counter artifacts therefore explicitly store
click events and never label event ratios as CTR. Offline scope persists only
`train` events and freezes them at `1782056217`; full scope persists
`full_train` and freezes at `1782248326`. Training lookups use an exact
`event.show_time < row.show_time` boundary, including exclusion of all events
at the same timestamp. Holdout/test consume only the frozen fit artifact.
Configured 7-day, 28-day and all-history windows cover banner, group, domain,
query, region, user and selected pair families. Every feature parquet manifest
fingerprints both the counter parquet and its scope manifest.

The Iteration 1 ranker consumes `label_raw_sc / raw_sc_scale`; the scale is one
global configured constant and does not change relative target values. It does
not log, clip or apply a second heuristic weight. The log-SourceCost label
remains a separately configured control. Training writes the configured native
CatBoost `PredictionValuesChange` importance report. SHAP and permutation
importance are deliberately disabled to keep the production run simple and
avoid a second groupwise analysis pass.

A promoted two-model ensemble remains a rank-level postprocessor, not a new
training architecture. Both CatBoost models must expose the identical ordered
feature contract. Their within-request ranks are averaged with a configured
weight, the result is blended with the frozen RRF rank, and the already bounded
SourceCost geometry may rerank only the configured prefix. Model paths and all
three weights are part of the resolved full-run config and output manifest.

## Memory and execution

`scripts/run_pipeline.py` is a standard-library orchestrator. It launches each
configured stage as a subprocess, streams combined stdout/stderr into a stage
log and records wall time/peak RSS. On Linux, RSS is sampled from the stage
process itself instead of reusing the process-wide historical maximum. A
dropped SSH/stdout stream disables echo but does not terminate the child or
its durable log. Candidate sources run sequentially by default
(`max_parallel_cg: 1`); GPU sources must not overlap.

Iteration 1 raises the configured worker limit to three. The scheduler groups
only dependency-independent work, holds GPU generators to one per group, runs
the two split merges together, and runs the two split feature builds together.
Run/result/timing writes are protected by a cross-process file lock, while
every child still owns its log, stage JSON, metrics and manifests.

The I1 Two-Tower wrapper encodes query batches and performs the same exact
matrix scan/top-k against the frozen candidate embeddings. Batch size is a
source config value. Direct batch-vs-single top-50 parity is mandatory before
use; the first four real smoke requests matched IDs and scores exactly.

## Fast value iteration

`i1_fast_value` keeps only the three productive I0 sources. Its temporal run
showed that truncating TF-IDF/Two-Tower before fusion loses useful overlap, so
it is retained as a measured speed control rather than a promotion candidate.
The natural ranker pool stays at 500. Feature v2 keeps retrieval, context,
static/text groups and only the four counter families retaining most counter
importance: region, domain, group and query. This produces 151 ordered features
instead of 276 and halves the CatBoost rows.

`i1_fast_quality` restores TF-IDF/Two-Tower source depth to 1,000 without
increasing the 500-row ranker pool. Query-click, query-SourceCost and
query-region histories are materialized as ranking features; only query-SC has
a positive tuned RRF weight. The other two remain zero-weight feature-only
sources. The SC-aware ranker uses the raw SourceCost label and a 900-iteration
ceiling selected from the prior temporal best iteration 845.

Temporal evaluation selects the better of RRF and CatBoost by SourceCost
Recall@50. The selected ranking method is passed to full inference. If RRF
wins, full feature construction and CatBoost fit are omitted entirely; the
submission is read directly from deterministic merged `pre_rank`. Each fast
pipeline launch has a configured three-hour wall-time budget checked between
isolated stage groups.

## TwoTower v2

The v2 retriever is implemented entirely inside `mla_two_stage/two_tower_v2`;
`common` and the frozen Step3 artifact remain unchanged. It streams the full
prepared 100m-derived click table (21,813,184 pairs) and keeps separate
embedding tables for query words, region, banner ID, ad group, title words and
text words. Per-field dimensions follow
`ceil_to_8(6 * cardinality ** 0.25)`, capped to `[8, 96]` by config.

Each tower projects the concatenated field embeddings to 256 dimensions. A
four-layer full-matrix DCNv2 branch and a three-layer
Linear/LayerNorm/GELU branch run in parallel; their outputs are concatenated,
projected to 64 dimensions and L2-normalized. Training preserves the proven
sampled-softmax objective with batch size 512 (one positive and 511 in-batch
negatives), BF16 autocast and a full epoch. No training-row subsampling is used.

Training writes a new immutable artifact directory with resolved YAML, model,
one-million-banner embeddings/metadata, terminal log, metrics and manifest. It
refuses to overwrite a non-empty artifact. The v2 batch generator uses the
same exact matrix scan/top-k contract as the baseline retriever.

## 10M walk-forward OOF cycle

`two_tower_v2_walk_forward_10m` is the fast honest development cycle. The raw
10M table contains 2,180,453 clicked pairs across eight contiguous Monday UTC
weeks. A versioned YQL stage materializes only fields required by the
TwoTower, sorted by `(week_start, show_time)`. The 1M option is reserved for
contract smoke because it is too small for stable weekly ranker data. The 100M
variant remains a later final-data option after the 10M gate.

For week `w`, execution order is fixed in `schedule.json`:

1. export the current model and one-million-banner embeddings;
2. freeze candidates for sampled requests from week `w`;
3. attach that week's click/SourceCost targets only after membership freezes;
4. update the same model and optimizer on all clicks from week `w`;
5. checkpoint model plus optimizer, then continue to `w + 1`.

The first week is predicted by the seeded random model. Each later week uses a
checkpoint trained only through earlier weeks. Later fixed validation/test
dates reuse the already verified full-100M v2 artifact of the same architecture
(trained only on pre-validation raw data). This avoids both a weaker 10M final
retriever and a redundant retrain. The generator chooses a snapshot from
request `show_time`; it never uses the full model for an OOF week.

Sampling keeps 750 deterministic request hashes per week (up to 6,000 OOF
groups) and preserves multi-click targets. The same streaming pass writes a
compact 2.18M-row history-event parquet. Query, query-region, user and global
candidate sources consume it with a strict `event_time < request_time`
boundary. Static full-history generators and the old full-data TwoTower are
disabled for OOF because their own-week labels would leak.

The temporal CatBoost train split is `weekly OOF + the original 6,499 fit
requests`; the unchanged 3,500-request holdout remains the promotion gate.
Full ranker data is `weekly OOF + all 9,999 validation requests`. Each scope
fits CatBoost exactly once from its natural 500-candidate pools. Full remains a
distinct post-selection fit, not a second fit inside temporal validation.

The configurable ranker target may be binary click, log-SourceCost or scaled
raw SourceCost. All three targets are attached only after the identical natural
candidate pool has been frozen; `ranker_binary` therefore supports cheap
click-probability/value-model complementarity tests without reintroducing
positive injection.

The 10M config retains TF-IDF, weekly TwoTower, past-only query/query-region,
user and global generators. Optional query/query-region recency variants are
the current fast candidate-development gate. Existing honest feature
importance keeps only query and region counter families; native
static/retrieval/text features remain. Counter lookup streams parquet and
materializes only configured families. Merge uses compact Arrow dictionaries
and partition workers; neither
optimization introduces target-dependent candidate injection.
