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

## 10M weekly OOF and automatic end-to-end chain

Validate the modifying YQL before launch, then create the versioned weekly
table. The token value is never passed on the command line:

```bash
python scripts/prepare_two_tower_weekly_dataset.py \
  --config configs/two_tower/v2_walk_forward_10m.yaml --validate-only
python scripts/prepare_two_tower_weekly_dataset.py \
  --config configs/two_tower/v2_walk_forward_10m.yaml
python scripts/query_two_tower_week_stats.py \
  --config configs/two_tower/v2_walk_forward_10m.yaml
```

Weekly training can be run directly or by its detached supervisor:

```bash
python scripts/train_two_tower_walk_forward.py \
  --config configs/two_tower/v2_walk_forward_10m.yaml
python scripts/continue_walk_forward_training.py \
  --config configs/two_tower/v2_walk_forward_10m.yaml
```

The current complete detached chain is:

```bash
nohup python scripts/continue_walk_forward_pipeline.py \
  --training-state /home/astrofimuk/workspace/mla_two_stage/artifacts/two_tower_v2_walk_forward_10m/training_supervisor.json \
  --walk-forward-artifact /home/astrofimuk/workspace/mla_two_stage/artifacts/two_tower_v2_walk_forward_10m \
  --final-artifact-override /home/astrofimuk/workspace/mla_two_stage/artifacts/two_tower_v2_dcn4_mlp3_full \
  --experiment i2_walk_forward_10m_fast_quality \
  --smoke-run 20260808_1845_i2_wf10m_smoke2 \
  --temporal-run 20260808_1830_i2_wf10m_temporal \
  --full-run 20260808_2030_i2_wf10m_full \
  --python /home/astrofimuk/workspace/step2_ce/.venv/bin/python \
  --runs /home/astrofimuk/workspace/mla_two_stage/runs \
  --cache /home/astrofimuk/workspace/mla_two_stage/cache \
  --immutable-artifacts /home/astrofimuk/workspace/mla_two_stage/artifacts \
  > /home/astrofimuk/workspace/mla_two_stage/artifacts/two_tower_v2_walk_forward_10m/pipeline_sequence.terminal.log \
  2>&1 < /dev/null &
```

It waits for weekly training, runs smoke, then fixed temporal validation,
selects RRF or CatBoost by SC Recall@50, and launches full only when candidate
SC@500 and ranker SC@50 exceed their configured gates. All pipeline runs log to
UnderDeep `camp-2026/modern-plumber` and keep a local JSONL fallback.

## Leaderboard submission

The local scorer is `http://a100-1.vla.yp-c.yandex.net:8083/`. Before upload,
run the strict submission stage and copy only its validated output from the VM:

```bash
scp astrofimuk@astrofimuk.ml-camp.ws.deep.yandex.net:\
/home/astrofimuk/workspace/mla_two_stage/runs/<full-id>/predictions/test_top50.parquet \
./<full-id>-test_top50.parquet
```

The form accepts a Parquet file up to 64 MiB with `HitLogID` and `BannerID`.
Use the immutable full run ID as the submission name and record the returned
SC Recall@50/Recall@50 in `docs/plan.md` before starting the next hypothesis.
