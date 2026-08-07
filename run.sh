#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/astrofimuk/workspace/step2_ce/.venv/bin/python"
COMMON="/home/astrofimuk/workspace/step2_ce/common"
VAL_FILE="/home/astrofimuk/workspace/tfidf_step1/data/val_clicks.parquet"
TEST_FILE="/home/astrofimuk/workspace/tfidf_step1/data/test_clicks.parquet"
ARTIFACT_DIR="$ROOT/artifacts/rrf"
STRONG_ARTIFACT_DIR="$ROOT/artifacts/rrf_strong"
STRONG_HISTORY_ARTIFACT_DIR="$ROOT/artifacts/rrf_strong_history"
RANKER_ARTIFACT_DIR="$ROOT/artifacts/catboost_history"
INDEX_FILE="/home/astrofimuk/workspace/tfidf_step1/data/index_1m.parquet"
export PYTHONPATH="$ROOT:/home/astrofimuk/workspace/step2_ce${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$ARTIFACT_DIR" "$STRONG_ARTIFACT_DIR" "$STRONG_HISTORY_ARTIFACT_DIR" "$RANKER_ARTIFACT_DIR" "$ROOT/run/metrics" "$ROOT/run/predictions"
cp "$ROOT/configs/baselines.json" "$ARTIFACT_DIR/config.json"
cp "$ROOT/configs/strong.json" "$STRONG_ARTIFACT_DIR/config.json"
cp "$ROOT/configs/strong_history.json" "$STRONG_HISTORY_ARTIFACT_DIR/config.json"

case "${1:-help}" in
  test)
    "$PYTHON" -m unittest discover -s "$ROOT/tests" -v
    ;;
  smoke)
    "$PYTHON" "$ROOT/scripts/smoke.py"
    ;;
  smoke-strong)
    "$PYTHON" "$ROOT/scripts/smoke.py" --config "$ROOT/configs/strong.json"
    ;;
  smoke-history)
    "$PYTHON" "$ROOT/scripts/smoke.py" --config "$ROOT/configs/strong_history.json"
    ;;
  prepare-history)
    "$PYTHON" "$ROOT/scripts/run_yql.py" \
      "$ROOT/yql/history_candidates.yql" \
      --output "$ROOT/artifacts/history/history_candidates.parquet"
    "$PYTHON" "$ROOT/scripts/build_history_artifact.py" \
      --history "$ROOT/artifacts/history/history_candidates.parquet" \
      --index "$INDEX_FILE" \
      --output-dir "$ROOT/artifacts/history" --top-k 200
    ;;
  build-history)
    "$PYTHON" "$ROOT/scripts/build_history_artifact.py" \
      --history "$ROOT/artifacts/history/history_candidates.parquet" \
      --index "$INDEX_FILE" \
      --output-dir "$ROOT/artifacts/history" --top-k 200
    ;;
  train-ranker)
    "$PYTHON" "$ROOT/scripts/train_ranker.py" \
      --config "$ROOT/configs/strong_history.json" \
      --val-file "$VAL_FILE" --output-dir "$RANKER_ARTIFACT_DIR" \
      --max-clicks 5000 --candidate-pool 500 \
      --iterations 600 --depth 8 --learning-rate 0.07 --task-type GPU
    ;;
  tune-fusion)
    "$PYTHON" "$ROOT/scripts/tune_fusion.py" \
      --config "$ROOT/configs/strong_history.json" --val-file "$VAL_FILE" \
      --max-clicks 100 --output "$ROOT/run/metrics/fusion_tuning_quick.json"
    ;;
  diagnose-quick)
    "$PYTHON" "$ROOT/scripts/evaluate_candidates.py" \
      --max-clicks 100 \
      --output "$ROOT/run/metrics/candidates_quick.json"
    ;;
  diagnose-full)
    "$PYTHON" "$ROOT/scripts/evaluate_candidates.py" \
      --max-clicks 10000 \
      --output "$ROOT/run/metrics/candidates_full.json"
    ;;
  diagnose-strong-quick)
    "$PYTHON" "$ROOT/scripts/evaluate_candidates.py" \
      --config "$ROOT/configs/strong.json" --max-clicks 100 \
      --output "$ROOT/run/metrics/candidates_strong_quick.json"
    ;;
  diagnose-strong-full)
    "$PYTHON" "$ROOT/scripts/evaluate_candidates.py" \
      --config "$ROOT/configs/strong.json" --max-clicks 10000 \
      --output "$ROOT/run/metrics/candidates_strong_full.json"
    ;;
  diagnose-history-quick)
    "$PYTHON" "$ROOT/scripts/evaluate_candidates.py" \
      --config "$ROOT/configs/strong_history.json" --max-clicks 100 \
      --output "$ROOT/run/metrics/candidates_history_quick.json"
    ;;
  diagnose-history-full)
    "$PYTHON" "$ROOT/scripts/evaluate_candidates.py" \
      --config "$ROOT/configs/strong_history.json" --max-clicks 10000 \
      --output "$ROOT/run/metrics/candidates_history_full.json"
    ;;
  train-step3)
    "$PYTHON" "$ROOT/code_maxim/step3_fps/train_yt.py" \
      --artifact-dir "$ROOT/artifacts/step3_fps" \
      --index-file "$INDEX_FILE" --batch-size 512 \
      --shuffle-buffer 20000 --validate-every-rows 200000 \
      --log-every 50 --no-tensorboard --device cuda
    ;;
  evaluate)
    "$PYTHON" "$COMMON/evaluate.py" \
      --code "$ROOT/solution/inference.py" \
      --artifact-dir "$ARTIFACT_DIR" \
      --val-file "$VAL_FILE" \
      --max-clicks 10000 --workers 1 --no-underdeep \
      --output "$ROOT/run/metrics/rrf_validation.json"
    ;;
  evaluate-strong)
    "$PYTHON" "$COMMON/evaluate.py" \
      --code "$ROOT/solution/inference.py" \
      --artifact-dir "$STRONG_ARTIFACT_DIR" \
      --val-file "$VAL_FILE" \
      --max-clicks 10000 --workers 1 --no-underdeep \
      --output "$ROOT/run/metrics/rrf_strong_validation.json"
    ;;
  evaluate-history)
    "$PYTHON" "$COMMON/evaluate.py" \
      --code "$ROOT/solution/inference.py" \
      --artifact-dir "$STRONG_HISTORY_ARTIFACT_DIR" \
      --val-file "$VAL_FILE" \
      --max-clicks 10000 --workers 1 --no-underdeep \
      --output "$ROOT/run/metrics/rrf_history_validation.json"
    ;;
  evaluate-ranker)
    "$PYTHON" "$COMMON/evaluate.py" \
      --code "$ROOT/solution/inference.py" \
      --artifact-dir "$RANKER_ARTIFACT_DIR" \
      --val-file "$VAL_FILE" \
      --max-clicks 10000 --workers 1 --no-underdeep \
      --output "$ROOT/run/metrics/catboost_validation.json"
    ;;
  evaluate-holdout)
    "$PYTHON" "$ROOT/scripts/evaluate_holdout.py" \
      --artifact-dir "$RANKER_ARTIFACT_DIR" --val-file "$VAL_FILE" \
      --train-clicks 5000 --all-clicks 10000 \
      --output "$ROOT/run/metrics/rrf_vs_catboost_holdout.json"
    ;;
  predict)
    "$PYTHON" "$COMMON/predict.py" \
      --code "$ROOT/solution/inference.py" \
      --artifact-dir "$ARTIFACT_DIR" \
      --test-file "$TEST_FILE" --top-k 50 --workers 1 \
      --output "$ROOT/run/predictions/rrf_test_top50.parquet" --force
    ;;
  predict-history)
    "$PYTHON" "$COMMON/predict.py" \
      --code "$ROOT/solution/inference.py" \
      --artifact-dir "$STRONG_HISTORY_ARTIFACT_DIR" \
      --test-file "$TEST_FILE" --top-k 50 --workers 1 \
      --output "$ROOT/run/predictions/rrf_history_test_top50.parquet" --force
    ;;
  predict-ranker)
    "$PYTHON" "$COMMON/predict.py" \
      --code "$ROOT/solution/inference.py" \
      --artifact-dir "$RANKER_ARTIFACT_DIR" \
      --test-file "$TEST_FILE" --top-k 50 --workers 1 \
      --output "$ROOT/run/predictions/catboost_test_top50.parquet" --force
    ;;
  *)
    echo "Usage: $0 {test|smoke|smoke-strong|smoke-history|prepare-history|build-history|train-step3|train-ranker|tune-fusion|diagnose-quick|diagnose-full|diagnose-strong-quick|diagnose-strong-full|diagnose-history-quick|diagnose-history-full|evaluate|evaluate-strong|evaluate-history|evaluate-ranker|evaluate-holdout|predict|predict-history|predict-ranker}" >&2
    exit 2
    ;;
esac
