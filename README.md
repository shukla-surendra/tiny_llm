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
- `embed_size`: `256`
- `num_heads`: `8`
- `num_layers`: `4`
- `dropout`: `0.1`
- Tokenizer: `gpt2` (`tiktoken`)
- Checkpoint file: `tiny_llm_checkpoint.pt`

### Parameter count (current config)

- **16,058,112 trainable parameters** (about **16.1M**)

This is a small model for local experimentation, not a production-scale LLM.

## About “1B parameter model”

This project is **not** a 1B model right now.

- Current: ~16.1M params
- 1B means ~1,000,000,000 params (about 62x larger than current)

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
