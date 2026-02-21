# Training Migration Guide (GPU <-> Mac)

Use this when you train on a cloud GPU for some time, then continue on Mac (MPS/CPU), or switch back.

## What is supported

- Sequential resume across machines: supported.
- Simultaneous combined CPU+GPU training in this project: not supported.

`tiny_llm.py` runs on one device per process (`cuda`, `mps`, or `cpu`) and resumes from checkpoint.

## Files to move

Minimum required:
- `tiny_llm_checkpoint_latest.pt`

Recommended to keep full history:
- `tiny_llm_checkpoint_best.pt`
- `tiny_llm_checkpoint.pt`
- `tiny_llm_checkpoint_final.pt`
- `logs/train_eval_history.csv`

Data and code consistency:
- `data/train.txt`
- `data/test.txt`
- same project code version (`tiny_llm.py`, related scripts)

## Cloud GPU -> Mac (resume)

On Mac, from repo root:

```bash
rsync -avz user@gpu-host:/path/to/tiny_llm/tiny_llm_checkpoint_latest.pt .
rsync -avz user@gpu-host:/path/to/tiny_llm/tiny_llm_checkpoint_best.pt . || true
rsync -avz user@gpu-host:/path/to/tiny_llm/logs/train_eval_history.csv logs/ || true
```

If you prefer `scp`:

```bash
scp user@gpu-host:/path/to/tiny_llm/tiny_llm_checkpoint_latest.pt .
scp user@gpu-host:/path/to/tiny_llm/tiny_llm_checkpoint_best.pt . 2>/dev/null || true
```

Resume:

```bash
RESUME_TRAINING=1 python3 tiny_llm.py
```

## Mac -> Cloud GPU (resume)

From Mac repo root:

```bash
rsync -avz tiny_llm_checkpoint_latest.pt user@gpu-host:/path/to/tiny_llm/
rsync -avz tiny_llm_checkpoint_best.pt user@gpu-host:/path/to/tiny_llm/ || true
rsync -avz logs/train_eval_history.csv user@gpu-host:/path/to/tiny_llm/logs/ || true
```

Then on GPU host:

```bash
cd /path/to/tiny_llm
RESUME_TRAINING=1 python3 tiny_llm.py
```

## Verification checklist before resuming

- Python deps installed (`pip install -r requirements.txt`)
- Checkpoint exists: `tiny_llm_checkpoint_latest.pt`
- `data/train.txt` and `data/test.txt` exist
- Model architecture config in code unchanged for the run:
  - `embed_size`
  - `num_heads`
  - `num_layers`
  - `dropout`
- `context_length` can be larger than dataset; code will derive `effective_context_length`.

Quick checkpoint metadata check:

```bash
python3 - <<'PY'
import torch
ckpt = torch.load("tiny_llm_checkpoint_latest.pt", map_location="cpu")
print("step:", ckpt.get("step"))
print("context_length:", ckpt.get("context_length"))
print("embed_size:", ckpt.get("embed_size"))
print("num_layers:", ckpt.get("num_layers"))
print("num_heads:", ckpt.get("num_heads"))
print("grad_accum_steps:", ckpt.get("grad_accum_steps"))
PY
```

## Important notes

- Optimizer state is portable across CPU/CUDA/MPS via `map_location`.
- If you change architecture fields and then resume, `load_state_dict` can fail on shape mismatch.
- If interrupted, `tiny_llm.py` writes `tiny_llm_checkpoint_latest.pt`; always sync that file first.
