# Commands

Run commands from `/home/astrofimuk/workspace/mla_two_stage` with the existing
environment `/home/astrofimuk/workspace/step2_ce/.venv/bin/python`.

## Contract and tests

```bash
/home/astrofimuk/workspace/step2_ce/.venv/bin/python -m pytest tests -q
/home/astrofimuk/workspace/step2_ce/.venv/bin/python scripts/run_pipeline.py --help
/home/astrofimuk/workspace/step2_ce/.venv/bin/python scripts/run_pipeline.py \
  experiment=i0_reproduce run_id=20260807_2200_i0 mode=smoke --dry-run
```

## Iteration 0 stage commands

```bash
python scripts/prepare_data.py experiment=i0_reproduce run_id=<id> scope=offline
python scripts/prepare_counters.py experiment=i0_reproduce run_id=<id> scope=offline
python scripts/generate_candidates.py experiment=i0_reproduce run_id=<id> split=train
python scripts/generate_candidates.py experiment=i0_reproduce run_id=<id> split=holdout
python scripts/merge_candidates.py experiment=i0_reproduce run_id=<id> split=train
python scripts/merge_candidates.py experiment=i0_reproduce run_id=<id> split=holdout
python scripts/validate_cache_parity.py experiment=i0_reproduce run_id=<id> mode=offline
python scripts/build_features.py experiment=i0_reproduce run_id=<id> split=train
python scripts/build_features.py experiment=i0_reproduce run_id=<id> split=holdout
python scripts/train_ranker.py experiment=i0_reproduce run_id=<id>
python scripts/evaluate_run.py experiment=i0_reproduce run_id=<id>
python scripts/inspect_run.py runs/<id>
```

## Full refit and submission

```bash
python scripts/run_pipeline.py experiment=i0_reproduce run_id=<id> mode=full
python scripts/make_submission.py experiment=i0_reproduce run_id=<id> mode=full scope=full
python scripts/validate_submission.py experiment=i0_reproduce run_id=<id> mode=full scope=full
```

The full orchestrator defaults to resume only when output manifests validate:

```bash
python scripts/run_pipeline.py experiment=i0_reproduce run_id=<id> mode=offline
python scripts/run_pipeline.py experiment=i0_reproduce run_id=<id> mode=full
python scripts/run_pipeline.py experiment=i1_more_cg_features_sc run_id=<id> mode=offline
python scripts/run_pipeline.py experiment=i1_more_cg_features_sc run_id=<id> mode=full
```

Legacy `./run.sh` commands remain available for frozen-baseline parity but are
not the experiment contract.
