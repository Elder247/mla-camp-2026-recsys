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

Generate one Iteration 1 source without running the complete graph:

```bash
python scripts/generate_candidates.py \
  experiment=i1_more_cg_features_sc run_id=<id> mode=offline scope=offline \
  split=train cg=history_user_v1
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

I1 uses the configured safe parallel scheduler automatically. It can be left
detached across client/network disconnects:

```bash
nohup env -u PYTHONPATH \
  /home/astrofimuk/workspace/step2_ce/.venv/bin/python scripts/run_pipeline.py \
  experiment=i1_more_cg_features_sc run_id=<id> mode=full scope=full \
  > /tmp/<id>.log 2>&1 < /dev/null &
```

Only independent CPU/split stages overlap; two GPU generators never overlap.

An Iteration 1 ranker run writes the native CatBoost
`PredictionValuesChange` report to
`runs/<id>/reports/feature_importance.csv`. SHAP and permutation importance are
disabled. Selection still uses `metrics/holdout.json` SourceCost Recall, not
CatBoost's internal objective.

Legacy `./run.sh` commands remain available for frozen-baseline parity but are
not the experiment contract.

## Fast value run

```bash
nohup env -u PYTHONPATH \
  /home/astrofimuk/workspace/step2_ce/.venv/bin/python scripts/run_pipeline.py \
  experiment=i1_fast_value run_id=<temporal-id> mode=offline scope=offline \
  paths.root=/home/astrofimuk/workspace/mla_two_stage_accel \
  paths.runs=/home/astrofimuk/workspace/mla_two_stage/runs \
  paths.cache=/home/astrofimuk/workspace/mla_two_stage/cache \
  paths.immutable_artifacts=/home/astrofimuk/workspace/mla_two_stage/artifacts \
  > /tmp/<temporal-id>.log 2>&1 < /dev/null &
```

After starting temporal, the detached selector can promote the better of RRF
and CatBoost and pass that choice to full inference:

```bash
python scripts/continue_to_full.py \
  --experiment i1_fast_value \
  --temporal-run <temporal-id> \
  --full-run <full-id> \
  --source-runs /home/astrofimuk/workspace/mla_two_stage/runs \
  --output-runs /home/astrofimuk/workspace/mla_two_stage/runs \
  --immutable-artifacts /home/astrofimuk/workspace/mla_two_stage/artifacts
```

Use `i1_fast_quality` for the follow-up that restores full TF-IDF/Two-Tower
depth, retains only high-importance query history signals, and switches back to
the raw-SourceCost label. The launch and selector commands are identical except
for `--experiment i1_fast_quality` / `experiment=i1_fast_quality`.

## TwoTower v2

Contract smoke (two train steps and a 1,000-banner export):

```bash
python scripts/train_two_tower_v2.py \
  --config configs/two_tower/v2_dcn4_mlp3_smoke.yaml
```

Full 100m-derived click training and canonical one-million-banner export:

```bash
nohup env PYTHONUNBUFFERED=1 \
  /home/astrofimuk/workspace/step2_ce/.venv/bin/python \
  scripts/train_two_tower_v2.py \
  --config configs/two_tower/v2_dcn4_mlp3_full.yaml \
  > /tmp/two_tower_v2_dcn4_mlp3_full.terminal.log 2>&1 < /dev/null &
```

The artifact-owned log is
`artifacts/two_tower_v2_dcn4_mlp3_full/train.log`; resolved config, metrics and
manifest are stored in the same directory.
