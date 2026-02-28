---
language:
- en
license: mit
library_name: pytorch
tags:
- text-generation
- causal-lm
datasets:
- lmsys/lmsys-chat-1m
pipeline_tag: text-generation
---

# Tiny LLM (Local Training + Inference + API)

This project trains and serves a small GPT-style language model on conversation data.

## What this project includes

- Dataset preparation:
  - `prepare_dataset.py` (synthetic conversations)
  - `prepare_dataset_lmsys.py` (LMSYS parquet -> train/test/prompts)
  - `audit_dataset.py` (train/test quality audit)
- Training:
  - `tiny_llm.py`
- Inference test:
  - `inference.py`
- Quality evaluation:
  - `eval_quality.py`
- FastAPI server:
  - `api_server.py`
- Workflow script:
  - `scripts/workflow.sh`
- Storage:
  - `scripts/upload_to_hf.py` (Upload artifacts to Hugging Face Hub)

## Current model details

The current default model in `tiny_llm.py` is:

- Architecture: GPT-style decoder (causal attention, pre-norm residual blocks, tied embeddings)
- `context_length`: `1024`
- `embed_size`: `768`
- `num_heads`: `12`
- `num_layers`: `12`
- `dropout`: `0.1`
- `batch_size`: `1`
- `grad_accum_steps`: `32` (effective batch update every 32 micro-steps)
- Loss: assistant-targeted masked next-token loss (focuses training on `Assistant:` turns)
- Optimization: warmup + cosine LR decay, gradient clipping (`max_norm=1.0`)
- Tokenizer: `gpt2` (`tiktoken`)
- Checkpoints:
  - `tiny_llm_checkpoint_latest.pt` (periodic resume checkpoint)
  - `tiny_llm_checkpoint_best.pt` (best by test loss)
  - `tiny_llm_checkpoint.pt` (serving checkpoint, updated from best)
  - `tiny_llm_checkpoint_final.pt` (checkpoint at end of run)

### Parameter count (current config)

- **152,791,296 trainable parameters** (about **152.8M**)

What is a parameter:
- A parameter is a learned numeric value (weight or bias) updated by backpropagation.
- During training, these values are adjusted so next-token predictions get lower loss.

Impact of parameter count:
- More parameters: higher capacity and usually better fit on complex data.
- More parameters: higher memory use and slower training/inference.
- Larger models generally require more data and training compute to avoid overfitting.

How this was calculated from `tiny_llm.py`:

- Variables used:
  - `vocab_size = 50257` (from GPT-2 tokenizer)
  - `context_length = 1024`
  - `embed_size = 768`
  - `num_layers = 16`
  - `num_heads = 12` (affects attention shape, not total formula independently once `embed_size` is fixed)
- Weight tying:
  - `lm_head.weight = token_emb.weight`, so output head does not add a second vocab projection matrix.

Breakdown:

- Token embedding: `vocab_size * embed_size`
  - `50257 * 768 = 38,597,376`
- Positional embedding: `context_length * embed_size`
  - `1024 * 768 = 786,432`
- Per Transformer block (`16` blocks):
  - Attention params:
    - `in_proj_weight`: `3E*E`
    - `in_proj_bias`: `3E`
    - `out_proj_weight`: `E*E`
    - `out_proj_bias`: `E`
  - MLP params:
    - `E*(4E) + (4E)` and `(4E)*E + E`
  - LayerNorms:
    - two layer norms, each has `2E` params (weight + bias)
  - Per block total with `E=768`: `7,087,872`
  - All blocks: `7,087,872 * 16 = 113,405,952`
- Final LayerNorm: `2E = 1,536`

Final total:

- `38,597,376 + 786,432 + 113,405,952 + 1,536 = 152,791,296`

This is a small model for local experimentation, not a production-scale LLM.

## About “1B parameter model”

This project is **not** a 1B model right now.

- Current: ~152.8M params
- 1B means ~1,000,000,000 params (about 8x larger than current)

To approach 1B, you generally need much larger settings (example direction):

- much higher `embed_size` (e.g., 1536+)
- many more layers (e.g., 24+)
- larger context window
- significantly more training data and steps
- much more compute and memory than typical local laptop training

## Quickstart

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Build dataset (LMSYS local parquet path default in script)

```bash
./scripts/workflow.sh data
```

Optional dataset audit before training:

```bash
./scripts/workflow.sh audit
```

Single script to download + merge + parse all conversational datasets:

```bash
HF_TOKEN=hf_xxx ./scripts/prepare_all_datasets.sh
```

This script:
- downloads public datasets (`UltraChat`, `OASST1`, `Dolly`, `SmolTalk`)
- optionally downloads gated `lmsys/lmsys-chat-1m` (`INCLUDE_GATED_LMSYS=1`)
- merges parquet inputs
- writes `data/train.txt`, `data/test.txt`, `data/test_prompts.txt`

Skip gated LMSYS if needed:

```bash
INCLUDE_GATED_LMSYS=0 ./scripts/prepare_all_datasets.sh
```

Add another dataset (optional) while generating data:

```bash
EXTRA_DATASET='HuggingFaceH4/ultrachat_200k' EXTRA_SPLIT='train_sft' ./scripts/workflow.sh data
```

Or add another local parquet dataset:

```bash
EXTRA_LOCAL_PARQUET_GLOB='/path/to/another_dataset/*.parquet' ./scripts/workflow.sh data
```

3) Train

```bash
./scripts/workflow.sh train
```

Training auto-resumes from `tiny_llm_checkpoint_latest.pt` if present.
Use `Ctrl+C` to stop safely; the script now writes a resumable latest checkpoint before exit.

4) Inference test

```bash
./scripts/workflow.sh infer
```

5) Run quality evaluation

```bash
./scripts/workflow.sh eval
```

6) Start API server

```bash
./scripts/workflow.sh serve
```

## Google Colab Compatibility

The project is compatible with Colab as-is after these steps:

1. Clone and install deps:
```bash
!git clone <your-repo-url>
%cd tiny_llm
!pip install -r requirements.txt
```

2. Prepare data:
```bash
!bash scripts/prepare_all_datasets.sh
```
If using gated LMSYS in Colab:
```bash
import os
os.environ["HF_TOKEN"] = "hf_xxx"
```

3. Train:
```bash
!bash scripts/workflow.sh train
```

4. Evaluate:
```bash
!bash scripts/workflow.sh eval
```

All paths in scripts are relative (for example `data/raw/...`, `data/train.txt`, `logs/...`), so no machine-specific absolute path is required.

## Resume Across Machines (GPU <-> Mac)

You can train on a GPU machine, copy checkpoints, and resume on Mac (or switch back later).

See `docs/MIGRATION.md` for:
- exact files to copy
- `rsync`/`scp` commands
- resume commands
- pre-resume validation checklist

## Quality Tracking (Long Training Runs)

To verify that quality is actually improving over days/weeks:

1. Use training eval history from `tiny_llm.py`:
- File: `logs/train_eval_history.csv`
- Logged every `eval_interval` steps with:
  - `train_loss`
  - `test_loss`
  - `test_perplexity`
  - `best_test_loss`
  - `step`, `est_epoch`, `processed_tokens`, `total_training_hours`

2. Use checkpoint quality trend:
- Command:
```bash
./scripts/workflow.sh eval
```
- File: `logs/quality_history.jsonl`
- Script compares against previous run and prints delta.

3. Use dataset audit gate before long runs:
- Command:
```bash
./scripts/workflow.sh audit
```
- Verify:
  - `noise_line_rate < 0.01`
  - `ascii_ratio > 0.98`
  - `assistant_exact_overlap_rate_vs_test` near `0.0`

Recommended acceptance signals:
- `best_test_loss` trends down over time.
- `heuristic_quality_score_0_to_100` trends up (or at least stable while loss drops).
- `role_leak_rate` and `placeholder_noise_rate` trend down.

If loss improves but quality score degrades, it usually means data quality/format noise is hurting generation quality.

## Conversational Datasets (Access)

- `HuggingFaceH4/ultrachat_200k`: public
- `OpenAssistant/oasst1`: public
- `zidankhan/databricks-dolly-15k`: public
- `HuggingFaceTB/smoltalk`: public (compact multi-domain chats)
- `allenai/tulu-v2-sft-mixture`: public (instruction/chat mixture)
- `lmsys/lmsys-chat-1m`: gated (public card, file access requires accepting terms + login)

Public dataset mix example:

```bash
EXTRA_DATASET='HuggingFaceH4/ultrachat_200k' EXTRA_SPLIT='train_sft' ./scripts/workflow.sh data
```

Another public dataset example:

```bash
EXTRA_DATASET='OpenAssistant/oasst1' EXTRA_SPLIT='train' ./scripts/workflow.sh data
```

## Hugging Face Token (for gated datasets)

How to get token:

1. Go to https://huggingface.co/settings/tokens
2. Click **New token**
3. Select at least **Read** permission
4. Create and copy token (starts with `hf_...`)

Use it one of these ways:

```bash
export HF_TOKEN=hf_xxx
```

or interactive login:

```bash
hf auth login
hf auth whoami
```

After login, simplest full download + parse flow:

```bash
./scripts/prepare_all_datasets.sh
```

Download gated dataset parquet files by code (no manual web download):

```bash
python prepare_dataset_lmsys.py \
  --dataset lmsys/lmsys-chat-1m \
  --token "$HF_TOKEN" \
  --download-parquet-dir data/lmsys/lmsys-chat-1m \
  --download-only
```

Then build train/test from downloaded parquet:

```bash
python prepare_dataset_lmsys.py \
  --local-parquet-glob 'data/raw/lmsys_lmsys-chat-1m/**/*.parquet' \
  --max-samples 300000
```

## API example

`POST /generate`

```json
{
  "prompt": "System: You are a helpful coding assistant for docker workflows.\nUser: How can you assist me with container startup failures?\nAssistant:",
  "max_new_tokens": 80,
  "do_sample": true,
  "temperature": 0.7,
  "top_k": 30,
  "top_p": 0.9,
  "repetition_penalty": 1.15
}
```

## Resume and Scheduled Training

`tiny_llm.py` now supports:

- periodic checkpoint save every `save_every_steps`
- resume after interruption (`resume_training = True`)
- best-checkpoint tracking by `test_loss`
- cumulative wall-clock training time tracking across resumes (`total_training_seconds`)

Stop and resume flow:

1. Start training:
```bash
./scripts/workflow.sh train
```
2. Stop training safely:
- Press `Ctrl+C`
- Script saves `tiny_llm_checkpoint_latest.pt`
3. Resume training later:
```bash
./scripts/workflow.sh train
```
4. Start fresh without resuming:
```bash
RESUME_TRAINING=0 ./scripts/workflow.sh train
```

To run daily at a fixed time on macOS/Linux with `cron` (example: 2:00 AM):

```bash
crontab -e
```

Add:

```cron
0 2 * * * cd /path/to/tiny_llm && /bin/zsh -lc './scripts/workflow.sh train >> logs/train.log 2>&1'
```

Create logs directory once:

```bash
mkdir -p logs
```
