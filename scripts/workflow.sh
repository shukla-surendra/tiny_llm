#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PY:-}" ]]; then
  PY_BIN="$PY"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="python"
else
  PY_BIN="python3"
fi

LMSYS_PARQUET_GLOB="${LMSYS_PARQUET_GLOB:-data/raw/lmsys_lmsys-chat-1m/**/*.parquet}"
MAX_SAMPLES="${MAX_SAMPLES:-300000}"
NUM_PROMPTS="${NUM_PROMPTS:-25}"
MIN_TURNS="${MIN_TURNS:-3}"
MIN_TURN_CHARS="${MIN_TURN_CHARS:-24}"
MIN_ASCII_RATIO="${MIN_ASCII_RATIO:-0.995}"
EXTRA_DATASET="${EXTRA_DATASET:-}"
EXTRA_SPLIT="${EXTRA_SPLIT:-train}"
EXTRA_LOCAL_PARQUET_GLOB="${EXTRA_LOCAL_PARQUET_GLOB:-}"
EXTRA_MAX_SAMPLES="${EXTRA_MAX_SAMPLES:-50000}"

usage() {
  cat <<'EOF'
Usage:
  scripts/workflow.sh data        # build dataset from LMSYS parquet
  scripts/workflow.sh data-synth  # build synthetic dataset
  scripts/workflow.sh audit       # audit train/test dataset quality
  scripts/workflow.sh train       # train model
  scripts/workflow.sh infer       # run inference on test prompts
  scripts/workflow.sh eval        # run heuristic quality evaluation
  scripts/workflow.sh serve       # start FastAPI server
  scripts/workflow.sh upload <id> # upload model artifacts to HF Hub
  scripts/workflow.sh pipeline    # data + train + infer + eval
EOF
}

run_data() {
  cmd=(
    "$PY_BIN" prepare_dataset_lmsys.py
    --local-parquet-glob "$LMSYS_PARQUET_GLOB"
    --max-samples "$MAX_SAMPLES"
    --num-prompts "$NUM_PROMPTS"
    --min-turns "$MIN_TURNS"
    --min-turn-chars "$MIN_TURN_CHARS"
    --min-ascii-ratio "$MIN_ASCII_RATIO"
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
  "$PY_BIN" prepare_dataset.py
}

run_audit() {
  "$PY_BIN" audit_dataset.py
}

run_train() {
  "$PY_BIN" tiny_llm.py
}

run_infer() {
  "$PY_BIN" inference.py
}

run_eval() {
  "$PY_BIN" eval_quality.py --compare-last --out-jsonl logs/quality_history.jsonl
}

run_serve() {
  "$PY_BIN" -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
}

run_upload() {
  if [[ -z "${1:-}" ]]; then
    echo "Error: Repo ID required. Usage: ./scripts/workflow.sh upload username/repo-name"
    exit 1
  fi
  "$PY_BIN" scripts/upload_to_hf.py --repo-id "$1"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

case "$1" in
  data) run_data ;;
  data-synth) run_data_synth ;;
  audit) run_audit ;;
  train) run_train ;;
  infer) run_infer ;;
  eval) run_eval ;;
  serve) run_serve ;;
  upload)
    shift
    run_upload "${1:-}"
    ;;
  pipeline)
    run_data
    run_train
    run_infer
    run_eval
    ;;
  *)
    usage
    exit 1
    ;;
esac
