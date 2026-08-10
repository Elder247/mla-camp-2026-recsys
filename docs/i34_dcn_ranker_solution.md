# i34: leakage-safe DCNv2 ranker

## Problem and expected result

The current private-best `v22 protective ANN conservative` reaches private
SourceCost Recall@50 `0.6988`.  Its retrieval universe is strong, but the last
natural-pool CatBoost screen selected only 25 trees and did not improve the
base order.  The immediate goal is to test whether a compact interaction model
can reorder the already cached natural candidate pool without another
retrieval fit or any target-dependent candidate injection.

Expected artifacts are a reproducible temporal DCNv2 screen, request-level
paired bootstrap against the accepted control, and (only after the temporal
gate passes) one full-fit/test ranking and strict `10,000 x 50` submission.

## Approach

1. Reuse the immutable i26 natural-pool feature cache: 159 as-of-time numeric,
   retrieval, text, context, and counter features.
2. Train a small DCNv2 ranker only on the temporal train split.  The objective
   is sampled listwise softmax with request-level, capped SourceCost weights.
   Every group includes its observed positives, the strongest pre-ranked hard
   negatives, and deterministic tail negatives.
3. Add zero-initialized hashed embeddings for query, user, banner, and their
   query-banner/user-banner crosses.  Query and user values come from the
   request table; no labels are used to construct them.  Unseen hashes remain
   zero unless a train collision exists.
4. Select the checkpoint on the earlier temporal half and report the later
   half independently.  Export a deterministic top-500 ranking for downstream
   blending with the accepted protective ANN ranking.
5. Run paired bootstrap by request for SC/Recall at the required cutoffs.  Only
   a positive early and late delta, non-negative Recall@50 and SC@500, and a
   stable bootstrap sign permit a full run.

## Data sources and contracts

| Source | Path | Use | Constraint |
|---|---|---|---|
| i26 temporal run | `runs/20260810_0220_i26_reranker_temporal` | cached train/holdout natural-pool features and requests | train/holdout request IDs must be disjoint and time ordered |
| accepted protective temporal predictions | i33 run artifacts | control and final blend input | immutable ranking; no leaderboard tuning |
| canonical banner index | immutable run artifacts | SourceCost geometry and ID validation | train/canonical metadata only |

## Definition gate

| Term | Concrete definition | Source |
|---|---|---|
| SourceCost Recall@K | ratio of summed SourceCost for clicked targets found in top K to summed SourceCost of all clicked targets | repository evaluator and saved leaderboard metric contract |
| temporal train/holdout | configured `temporal_val_v1` request partitions in the i26 manifest | i26 resolved config and manifest |
| natural candidate pool | candidates emitted by configured retrieval sources before labels are attached; absent clicked targets stay absent | feature/cache manifests and ranker-label tests |
| paired bootstrap | resample complete requests and recompute ratio-of-sums SC delta per sample | task requirement |

## Assumptions, risks, and locked decisions

- The i26 cache is reused because its manifests and cache-parity report are
  already validated; rebuilding it would spend time without changing the
  hypothesis.
- The DCNv2 score is not fed back into CatBoost in-fold.  It is evaluated as a
  separate ranking source, avoiding score-feature leakage.
- The main risk is overfitting only 6,499 temporal training requests.  The
  safeguards are zero-initialized residual ranking, capped group weights,
  early-half checkpoint selection, and a later-half promotion gate.
- If this branch fails quickly, the sole reserve is a cached/small BM25F plus
  character-ngram complementarity screen.  No wide architecture or geometry
  search is allowed.

Locked: one listwise DCNv2 screen first; no full fit before the temporal and
bootstrap gates; no leaderboard upload for a merely offline-positive neighbor
without strict validation.

## Completion criteria

- [ ] train/holdout leakage checks and tests pass;
- [ ] DCNv2 temporal early/late/full metrics are recorded;
- [ ] paired request bootstrap and cost-tail sensitivity are recorded;
- [ ] promoted full output passes exactly 10,000 rows and 50 unique valid IDs;
- [ ] private SourceCost Recall@50 is strictly above `0.6988`;
- [ ] documentation, manifests, hashes, timings, and small commits are saved.
