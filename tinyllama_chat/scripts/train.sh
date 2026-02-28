#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${PY:-}" ]]; then
  PY_BIN="$PY"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="python"
else
  PY_BIN="python3"
fi

MODEL_ID="${MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
DATASET="${DATASET:-HuggingFaceH4/ultrachat_200k}"
DATASET_SPLIT="${DATASET_SPLIT:-train_sft}"
MAX_SAMPLES="${MAX_SAMPLES:-20000}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/tinyllama_lora}"
SEQ_LEN="${SEQ_LEN:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-2e-4}"
EPOCHS="${EPOCHS:-1}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
RESUME="${RESUME:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

cmd=(
  "$PY_BIN" "${PROJECT_DIR}/train_tinyllama_lora.py"
  --model-id "$MODEL_ID"
  --dataset "$DATASET"
  --dataset-split "$DATASET_SPLIT"
  --max-samples "$MAX_SAMPLES"
  --output-dir "$OUTPUT_DIR"
  --seq-len "$SEQ_LEN"
  --batch-size "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --lr "$LR"
  --epochs "$EPOCHS"
  --save-strategy "$SAVE_STRATEGY"
  --save-steps "$SAVE_STEPS"
  --save-total-limit "$SAVE_TOTAL_LIMIT"
)

if [[ "$RESUME" == "1" ]]; then
  cmd+=(--resume)
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  cmd+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

"${cmd[@]}"
