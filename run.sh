#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/astrofimuk/workspace/step2_ce/.venv/bin/python"
COMMON="/home/astrofimuk/workspace/step2_ce/common"
VAL_FILE="/home/astrofimuk/workspace/tfidf_step1/data/val_clicks.parquet"
TEST_FILE="/home/astrofimuk/workspace/tfidf_step1/data/test_clicks.parquet"
ARTIFACT_DIR="$ROOT/artifacts/rrf"

mkdir -p "$ARTIFACT_DIR" "$ROOT/run/metrics" "$ROOT/run/predictions"
cp "$ROOT/configs/baselines.json" "$ARTIFACT_DIR/config.json"

case "${1:-help}" in
  test)
    python3 -m unittest discover -s "$ROOT/tests" -v
    ;;
  smoke)
    "$PYTHON" "$ROOT/scripts/smoke.py"
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
  evaluate)
    "$PYTHON" "$COMMON/evaluate.py" \
      --code "$ROOT/solution/inference.py" \
      --artifact-dir "$ARTIFACT_DIR" \
      --val-file "$VAL_FILE" \
      --max-clicks 10000 --workers 1 --no-underdeep \
      --output "$ROOT/run/metrics/rrf_validation.json"
    ;;
  predict)
    "$PYTHON" "$COMMON/predict.py" \
      --code "$ROOT/solution/inference.py" \
      --artifact-dir "$ARTIFACT_DIR" \
      --test-file "$TEST_FILE" --top-k 50 --workers 1 \
      --output "$ROOT/run/predictions/rrf_test_top50.parquet" --force
    ;;
  *)
    echo "Usage: $0 {test|smoke|diagnose-quick|diagnose-full|evaluate|predict}" >&2
    exit 2
    ;;
esac

