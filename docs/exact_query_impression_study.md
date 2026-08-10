# Exact-query history study

## Question

The leaderboard jump from `h23_w6_val` to `exact_query*_h23w6_fallback_v1`
suggested that exact-query history was a separate candidate source.  The study
tested two leakage-safe explanations:

1. candidates come from every historical impression for the literal query,
   rather than only clicked banners;
2. a broad exact-query pool is reordered by a semantic model or by a learned
   mixed-pool ranker before the fallback is appended.

The fixed source is the official 100M training table.  Its maximum `ShowTime`
is `1781643598`, before the temporal holdout.  Earlier validation requests are
added only when scoring the later holdout or the test set.

## Candidate artifact

`yql/exact_query_impression_candidates.yql` aggregates literal
`SearchQuery × BannerID` impression statistics and retains the union of the
top 100 candidates by nine ranking policies.  The resulting immutable artifact
contains 557,307 rows, 11,709 queries and 369,789 banners.  Its SHA-256 is
`235b43eba7c734f9dcbb2286adea3e90cab074acab733f5da8e56a5b2c6c1dd9`.

Normalization is not material here: raw exact matching covers about 70% of
test requests, while the project normalizer adds only three requests.  The
older clicked-history source is therefore not the explanation: its best
temporal gain is only about 0.44 percentage points and its best private result
is 70.23% SC Recall@50.

## Temporal results

The comparison uses the protected v27 prediction as fallback.  Its temporal
SC Recall@50 is 71.72% (73.37% early, 69.78% late).

| Ranking | Prefix | Temporal | Early gain | Late gain |
|---|---:|---:|---:|---:|
| impression recency + 25% two-tower rank | 20 | **73.23%** | +0.67 pp | +2.49 pp |
| impression recency | 30 | 73.14% | -0.01 pp | +3.10 pp |
| 7-day clicks + 25% two-tower rank | 20 | 73.10% | +0.64 pp | +2.24 pp |
| CatBoost inside the exact pool + 25% two-tower rank | 20 | 73.10% | +0.64 pp | +2.25 pp |
| CatBoost over `h500 + exact` | 10 | 71.04% | -0.13 pp | -1.32 pp |

The full membership ceiling of `fallback ∪ exact-impressions` is 75.07%, so
the selected simple hybrid captures 1.50 of the available 3.35 percentage
points.  A 400-tree exact-only CatBoost was trained on 322,008 rows from 3,484
positive early groups.  A separate 250-tree `h500 + exact` CatBoost was trained
on 4,172,934 rows from 5,414 positive groups and evaluated on 2,587,644 rows.
Neither learned ranker beats the simple temporal hybrid.

The selected hybrid was also checked with 10,000 paired bootstrap iterations.
The request bootstrap gives a +1.50 pp point estimate with 95% interval
`[+0.33, +3.02]` pp and `P(gain > 0) = 99.64%`.  Clustering by user gives
`[+0.33, +3.10]` pp and `P(gain > 0) = 99.69%` (3,499 user clusters).

## Materialized private candidates

All files contain exactly 10,000 unique `HitLogID` rows and 50 unique banner
IDs per row.

| File | SHA-256 |
|---|---|
| `test_top50_imp_recency_m25_e20_v27.parquet` | `89ef2ec81f3f8cb2c99c9d3c580bf47f91e00e5b982fdaa00a78031e12f3a810` |
| `test_top50_imp_recency_e30_v27.parquet` | `317e59f8d7ffaa863735199a68f3837759643b022bfe3d0f8203aaea2e7a957a` |
| `test_top50_imp_clicks7_m25_e20_v27.parquet` | `b2a7ef9e42ae199f342efd160b7618e19f29656fd152413f20d81f16ff04437e` |

## Private results

| Run | Ranking | SC Recall@50 | Recall@50 | Recall@10 |
|---|---|---:|---:|---:|
| `9fe0c4899caf` | recency + 25% model, exact-20 | **71.24%** | **60.98%** | 44.15% |
| `ddb6a675618a` | recency, exact-30 | 70.90% | 60.65% | 41.44% |
| `8fe64b549cbd` | 7-day clicks + 25% model, exact-20 | 71.01% | 60.80% | **45.31%** |

The selected run improves the protected v27 private SC Recall@50 from 70.05%
to 71.24%, an absolute gain of 1.19 percentage points.  The private result
therefore confirms both the all-impressions candidate source and the small
benefit from semantic reordering.  The exact-only and `h500 + exact` learned
rankers fail to improve temporal metrics, and the remaining private gap to the
78% mixed-pool solutions cannot be explained by either tested hypothesis.
