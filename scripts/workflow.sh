#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-.venv/bin/python}"
UVICORN="${UVICORN:-.venv/bin/uvicorn}"

LMSYS_PARQUET_GLOB="${LMSYS_PARQUET_GLOB:-/Users/surendrashukla/projects/tiny_llm/data/lmsys/lmsys-chat-1m/*.parquet}"
MAX_SAMPLES="${MAX_SAMPLES:-300000}"
NUM_PROMPTS="${NUM_PROMPTS:-25}"
MIN_TURNS="${MIN_TURNS:-3}"
MIN_TURN_CHARS="${MIN_TURN_CHARS:-20}"
MIN_ASCII_RATIO="${MIN_ASCII_RATIO:-0.99}"
EXTRA_DATASET="${EXTRA_DATASET:-}"
EXTRA_SPLIT="${EXTRA_SPLIT:-train}"
EXTRA_LOCAL_PARQUET_GLOB="${EXTRA_LOCAL_PARQUET_GLOB:-}"
EXTRA_MAX_SAMPLES="${EXTRA_MAX_SAMPLES:-50000}"

usage() {
  cat <<'EOF'
Usage:
  scripts/workflow.sh data        # build dataset from LMSYS parquet
  scripts/workflow.sh data-synth  # build synthetic dataset
  scripts/workflow.sh train       # train model
  scripts/workflow.sh infer       # run inference on test prompts
  scripts/workflow.sh serve       # start FastAPI server
  scripts/workflow.sh pipeline    # data + train + infer
EOF
}

run_data() {
  cmd=(
    "$PY" prepare_dataset_lmsys.py
    --local-parquet-glob "$LMSYS_PARQUET_GLOB" \
    --max-samples "$MAX_SAMPLES" \
    --num-prompts "$NUM_PROMPTS" \
    --min-turns "$MIN_TURNS" \
    --min-turn-chars "$MIN_TURN_CHARS" \
    --min-ascii-ratio "$MIN_ASCII_RATIO" \
    --extra-max-samples "$EXTRA_MAX_SAMPLES"
  )

  if [[ -n "$EXTRA_DATASET" ]]; then
    cmd+=(--extra-dataset "$EXTRA_DATASET" --extra-split "$EXTRA_SPLIT")
  fi
  if [[ -n "$EXTRA_LOCAL_PARQUET_GLOB" ]]; then
    cmd+=(--extra-local-parquet-glob "$EXTRA_LOCAL_PARQUET_GLOB")
  fi

  "${cmd[@]}"
}

run_data_synth() {
  "$PY" prepare_dataset.py
}

run_train() {
  "$PY" tiny_llm.py
}

run_infer() {
  "$PY" inference.py
}

run_serve() {
  "$UVICORN" api_server:app --host 127.0.0.1 --port 8000 --reload
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

case "$1" in
  data) run_data ;;
  data-synth) run_data_synth ;;
  train) run_train ;;
  infer) run_infer ;;
  serve) run_serve ;;
  pipeline)
    run_data
    run_train
    run_infer
    ;;
  *)
    usage
    exit 1
    ;;
esac
