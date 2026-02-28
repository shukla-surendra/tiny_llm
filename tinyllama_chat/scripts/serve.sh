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

BASE_MODEL_ID="${BASE_MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
ADAPTER_PATH="${ADAPTER_PATH:-${PROJECT_DIR}/outputs/tinyllama_lora/final}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"

"$PY_BIN" "${PROJECT_DIR}/serve_tinyllama_lora.py" \
  --base-model-id "$BASE_MODEL_ID" \
  --adapter-path "$ADAPTER_PATH" \
  --host "$HOST" \
  --port "$PORT"
