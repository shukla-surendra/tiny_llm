# TinyLlama Separate Setup (Train + Serve)

This folder is isolated from your custom `tiny_llm.py` flow and is meant for quick chat fine-tuning on a pretrained tiny model.

Default model:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

Default dataset:
- `HuggingFaceH4/ultrachat_200k` split `train_sft`

## 1) Install

```bash
cd tinyllama_chat
python -m pip install -r requirements.txt
```

## 2) Train LoRA adapter

```bash
cd tinyllama_chat
bash scripts/train.sh
```

Useful knobs:

```bash
MAX_SAMPLES=5000 EPOCHS=1 SEQ_LEN=1024 GRAD_ACCUM=16 bash scripts/train.sh
```

Checkpoint/resume knobs:

```bash
SAVE_STEPS=100 SAVE_TOTAL_LIMIT=3 RESUME=1 bash scripts/train.sh
```

Resume from a specific checkpoint:

```bash
RESUME_FROM_CHECKPOINT=outputs/tinyllama_lora/checkpoint-500 bash scripts/train.sh
```

Output adapter path:
- `tinyllama_chat/outputs/tinyllama_lora/final`
- checkpoints are written under `tinyllama_chat/outputs/tinyllama_lora/checkpoint-*`

## 3) Serve API

```bash
cd tinyllama_chat
bash scripts/serve.sh
```

Server:
- Health: `GET http://127.0.0.1:8001/health`
- Generate: `POST http://127.0.0.1:8001/generate`

Example request:

```bash
curl -X POST http://127.0.0.1:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain LoRA in simple terms.",
    "max_new_tokens": 180,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.1
  }'
```

## Notes for MacBook Pro 24GB

- Defaults are tuned for Apple Silicon memory constraints.
- If you hit numerical issues on MPS, run:

```bash
python train_tinyllama_lora.py --force-float32
```

- First run downloads model + dataset from Hugging Face and can take time.
