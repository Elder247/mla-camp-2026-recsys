# ML Camp two-stage recommender

This repository is the clean implementation of the competition system.  The
legacy teacher solutions remain read-only baselines in sibling directories.

The first milestone provides:

- a common adapter for existing retrieval solutions;
- weighted reciprocal-rank fusion with deduplication and provenance;
- an inference module compatible with `common/evaluate.py` and
  `common/predict.py`;
- candidate-level diagnostics on validation clicks;
- unit and integration smoke tests.

The strong candidate layer combines three complementary sources:

- TF-IDF for exact lexical matching;
- a sampled-softmax Two-Tower trained on 21.8M clicked pairs;
- leakage-safe query/query-region click history aggregated only from train.

On the first 100 validation clicks, the current three-source RRF reaches
Recall@50 `0.48` and SourceCost-recall@50 `0.658`; the sampled-softmax
Two-Tower alone reaches `0.34` and `0.451` respectively. These figures are a
fast diagnostic sample, not the final holdout estimate.

On 4,999 untouched requests excluded from ranker training, CatBoost improves
the tuned RRF as follows:

| Metric | RRF | CatBoost |
|---|---:|---:|
| Recall@1 | 0.1194 | 0.1474 |
| SourceCost-recall@1 | 0.1563 | 0.2237 |
| Recall@5 | 0.2937 | 0.3247 |
| SourceCost-recall@5 | 0.3955 | 0.4478 |
| Recall@50 | 0.5363 | 0.5377 |
| SourceCost-recall@50 | 0.6658 | 0.6680 |

Recall@500 is `0.6497` for both methods because the second stage reranks the
same candidate pool rather than generating new objects.

## VM layout

```text
~/workspace/
├── tfidf_step1/       # frozen lexical baseline + index/validation data
├── step2_ce/          # frozen pointwise Two-Tower baseline
└── mla_two_stage/     # this repository
```

The repository does not copy large model files. `configs/baselines.json`
references the existing immutable artifacts by absolute path.

YQL uses `/home/astrofimuk/.yql/token` through the client's `token_path` API;
YT uses `/home/astrofimuk/.yt/token`. Token values are never copied into the
repository or shell commands. Install Yandex-only packages from the internal
PyPI (`https://pypi.yandex-team.ru/simple/`).

## Commands

```bash
cd ~/workspace/mla_two_stage

# Pure unit tests (do not load model artifacts)
python3 -m unittest discover -s tests -v

# Load both million-candidate models and rank one synthetic request
./run.sh smoke

# Train the strong sampled-softmax Two-Tower on 21.8M clicked pairs
./run.sh train-step3

# Build train-only query and query-region historical candidates via YQL
./run.sh prepare-history

# Train a simple CatBoost second stage on a deterministic val split
./run.sh train-ranker

# Smoke/diagnostics with TF-IDF + sampled-softmax Two-Tower
./run.sh smoke-strong
./run.sh diagnose-strong-quick

# Smoke/diagnostics for all three candidate generators
./run.sh smoke-history
./run.sh diagnose-history-quick

# Candidate recall report on the first 100 validation clicks
./run.sh diagnose-quick

# Full 10k validation candidate report
./run.sh diagnose-full

# Evaluate the current fused RRF ordering at top-50
./run.sh evaluate

# Produce the hidden-test parquet using the current fused ordering
./run.sh predict

# Validate or predict with TF-IDF + Two-Tower + history
./run.sh evaluate-history
./run.sh predict-history

# Full-val check (contains ranker-training requests; not a holdout metric)
./run.sh evaluate-ranker
./run.sh predict-ranker

# Honest comparison on requests excluded from the first 5k training clicks
./run.sh evaluate-holdout
```

## Current status

RRF is deliberately only the first-stage fusion baseline. Candidate source
ranks, scores and provenance are retained in every returned candidate. The
next milestone uses this exact inference-time candidate distribution to build
simple cross-source features and train the CatBoost ranker.
