# Tiny LLM (Local Training + Inference + API)

This project trains and serves a small GPT-style language model on conversation data.

## What this project includes

- Dataset preparation:
  - `prepare_dataset.py` (synthetic conversations)
  - `prepare_dataset_lmsys.py` (LMSYS parquet -> train/test/prompts)
- Training:
  - `tiny_llm.py`
- Inference test:
  - `inference.py`
- FastAPI server:
  - `api_server.py`
- Workflow script:
  - `scripts/workflow.sh`

## Current model details

The current default model in `tiny_llm.py` is:

- Architecture: GPT-style decoder (causal attention, pre-norm residual blocks, tied embeddings)
- `context_length`: `128`
- `embed_size`: `768`
- `num_heads`: `12`
- `num_layers`: `12`
- `dropout`: `0.1`
- `batch_size`: `1`
- `grad_accum_steps`: `8` (effective batch update every 8 micro-steps)
- Tokenizer: `gpt2` (`tiktoken`)
- Checkpoints:
  - `tiny_llm_checkpoint_latest.pt` (periodic resume checkpoint)
  - `tiny_llm_checkpoint_best.pt` (best by test loss)
  - `tiny_llm_checkpoint.pt` (serving checkpoint, updated from best)
  - `tiny_llm_checkpoint_final.pt` (checkpoint at end of run)

### Parameter count (current config)

- **123,751,680 trainable parameters** (about **123.8M**)

How this was calculated from `tiny_llm.py`:

- Variables used:
  - `vocab_size = 50257` (from GPT-2 tokenizer)
  - `context_length = 128`
  - `embed_size = 768`
  - `num_layers = 12`
  - `num_heads = 12` (affects attention shape, not total formula independently once `embed_size` is fixed)
- Weight tying:
  - `lm_head.weight = token_emb.weight`, so output head does not add a second vocab projection matrix.

Breakdown:

- Token embedding: `vocab_size * embed_size`
  - `50257 * 768 = 38,597,376`
- Positional embedding: `context_length * embed_size`
  - `128 * 768 = 98,304`
- Per Transformer block (`12` blocks):
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
  - All blocks: `7,087,872 * 12 = 85,054,464`
- Final LayerNorm: `2E = 1,536`

Final total:

- `38,597,376 + 98,304 + 85,054,464 + 1,536 = 123,751,680`

This is a small model for local experimentation, not a production-scale LLM.

## About “1B parameter model”

This project is **not** a 1B model right now.

- Current: ~30.0M params
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
.venv/bin/pip install -r requirements.txt
```

2) Build dataset (LMSYS local parquet path default in script)

```bash
./scripts/workflow.sh data
```

3) Train

```bash
./scripts/workflow.sh train
```

Training auto-resumes from `tiny_llm_checkpoint_latest.pt` if present.

4) Inference test

```bash
./scripts/workflow.sh infer
```

5) Start API server

```bash
./scripts/workflow.sh serve
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

To run daily at a fixed time on macOS/Linux with `cron` (example: 2:00 AM):

```bash
crontab -e
```

Add:

```cron
0 2 * * * cd /Users/surendrashukla/projects/tiny_llm && /bin/zsh -lc './scripts/workflow.sh train >> logs/train.log 2>&1'
```

Create logs directory once:

```bash
mkdir -p logs
```
