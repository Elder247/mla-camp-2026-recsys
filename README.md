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

## VM layout

```text
~/workspace/
├── tfidf_step1/       # frozen lexical baseline + index/validation data
├── step2_ce/          # frozen pointwise Two-Tower baseline
└── mla_two_stage/     # this repository
```

The repository does not copy large model files. `configs/baselines.json`
references the existing immutable artifacts by absolute path.

## Commands

```bash
cd ~/workspace/mla_two_stage

# Pure unit tests (do not load model artifacts)
python3 -m unittest discover -s tests -v

# Load both million-candidate models and rank one synthetic request
./run.sh smoke

# Candidate recall report on the first 100 validation clicks
./run.sh diagnose-quick

# Full 10k validation candidate report
./run.sh diagnose-full

# Evaluate the current fused RRF ordering at top-50
./run.sh evaluate

# Produce the hidden-test parquet using the current fused ordering
./run.sh predict
```

## Current status

RRF is deliberately only the first-stage fusion baseline.  Candidate source
ranks, scores and provenance are retained in every returned candidate so the
next milestone can train CatBoost on the exact inference-time candidate
distribution.

