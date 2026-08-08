# Architecture rules

These invariants are review blockers.

1. `common/` is external. New infrastructure lives in this repository and
   existing legacy generators are integrated through thin adapters.
2. Candidate membership is target-independent. One function applies quotas,
   deduplication, deterministic ordering and `ranker_pool` for train, holdout and
   inference. The target is joined only after the pool is frozen.
3. Split before learning. The temporal boundary is immutable; a request group
   cannot cross splits. Hidden test is never used for model selection.
4. Counters are past-only. Weekly row snapshots use weeks strictly before the
   row week. Offline validation uses sources before validation; full test scope
   is a distinct version with its own cutoff and source list.
5. Cache identity is semantic: config fingerprint + input fingerprints + code
   version + scope + artifact version. A filename alone is never proof of reuse.
6. Every heavy stage is an independent subprocess with its own log, timing,
   result and output manifest. The orchestrator imports no ML libraries.
7. Large tables are streamed/chunked and partitioned. Do not accumulate tens of
   millions of rows in Python lists or one pandas DataFrame.
8. Schemas are contracts. IDs have explicit integer types, features are ordered
   and versioned, ties end with ascending BannerID, and every enabled generator
   must appear in the merged provenance columns.
9. Experiment claims require temporal-holdout SourceCost Recall@50 plus candidate
   ceiling/complementarity. AUC, loss and leaderboard alone do not justify a
   change.
10. Config controls behavior. No absolute path, quota, feature flag, pool size,
    seed or model hyperparameter is hardcoded outside YAML defaults.
11. Runs are append-only. Existing artifacts are not overwritten; a completed
    run can only be inspected or reused after fingerprint/schema validation.
12. DeepRanker and broad Optuna remain disabled until Iterations 0 and 1 pass
    their documented gates.
13. Learned OOF retrieval predicts before updating on the target week. The
    CatBoost pool for that row must use the corresponding pre-update snapshot;
    a final model trained through all weeks is valid only for later val/test.
14. UnderDeep is observability, not a dependency. Every event is first written
    to a masked local backup; missing tokens/client/network must not fail a run.
