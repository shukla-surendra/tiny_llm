#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PY:-}" ]]; then
  PY_BIN="$PY"
elif command -v python >/dev/null 2>&1; then
  PY_BIN="python"
else
  PY_BIN="python3"
fi

# Toggle gated LMSYS dataset download.
INCLUDE_GATED_LMSYS="${INCLUDE_GATED_LMSYS:-1}"

# Token for gated datasets. Optional if you've already run `huggingface-cli login`.
HF_TOKEN="${HF_TOKEN:-}"

# Final preprocessing knobs.
MAX_SAMPLES="${MAX_SAMPLES:-400000}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
MIN_TURNS="${MIN_TURNS:-3}"
MIN_TURN_CHARS="${MIN_TURN_CHARS:-24}"
MIN_ASCII_RATIO="${MIN_ASCII_RATIO:-0.995}"

RAW_ROOT="data/raw"

PUBLIC_DATASETS=(
  "HuggingFaceH4/ultrachat_200k"
  "OpenAssistant/oasst1"
  "zidankhan/databricks-dolly-15k"
  "HuggingFaceTB/smoltalk"
)

GATED_DATASETS=(
  "lmsys/lmsys-chat-1m"
)

slugify() {
  echo "$1" | tr '/:' '__'
}

download_dataset() {
  local dataset_id="$1"
  local out_dir="$RAW_ROOT/$(slugify "$dataset_id")"
  echo "[download] $dataset_id -> $out_dir"

  if [[ -n "$HF_TOKEN" ]]; then
    "$PY_BIN" prepare_dataset_lmsys.py \
      --dataset "$dataset_id" \
      --token "$HF_TOKEN" \
      --download-parquet-dir "$out_dir" \
      --download-only
  else
    "$PY_BIN" prepare_dataset_lmsys.py \
      --dataset "$dataset_id" \
      --use-auth \
      --download-parquet-dir "$out_dir" \
      --download-only
  fi
}

echo "[stage] downloading public datasets"
mkdir -p "$RAW_ROOT"
for ds in "${PUBLIC_DATASETS[@]}"; do
  download_dataset "$ds"
done

if [[ "$INCLUDE_GATED_LMSYS" == "1" ]]; then
  echo "[stage] downloading gated datasets"
  for ds in "${GATED_DATASETS[@]}"; do
    download_dataset "$ds"
  done
else
  echo "[stage] skipping gated datasets (INCLUDE_GATED_LMSYS=$INCLUDE_GATED_LMSYS)"
fi

echo "[stage] collecting parquet files"
PARQUET_GLOBS=()
while IFS= read -r d; do
  if find "$d" -type f -name '*.parquet' | grep -q .; then
    PARQUET_GLOBS+=("$d/**/*.parquet")
  fi
done < <(find "$RAW_ROOT" -mindepth 1 -maxdepth 1 -type d -print | sort)
if [[ ${#PARQUET_GLOBS[@]} -eq 0 ]]; then
  echo "[error] no parquet files found under $RAW_ROOT"
  exit 1
fi

echo "[stage] building train/test prompts"
cmd=(
  "$PY_BIN" prepare_dataset_lmsys.py
  --local-parquet-glob "${PARQUET_GLOBS[0]}"
  --max-samples "$MAX_SAMPLES"
  --num-prompts "$NUM_PROMPTS"
  --min-turns "$MIN_TURNS"
  --min-turn-chars "$MIN_TURN_CHARS"
  --min-ascii-ratio "$MIN_ASCII_RATIO"
)

for ((i=1; i<${#PARQUET_GLOBS[@]}; i++)); do
  cmd+=(--extra-local-parquet-glob "${PARQUET_GLOBS[$i]}")
done

"${cmd[@]}"
echo "[done] wrote:"
echo "  data/train.txt"
echo "  data/test.txt"
echo "  data/test_prompts.txt"
