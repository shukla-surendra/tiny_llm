import argparse
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, description="User prompt")
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, gt=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1, le=200)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class GenerateResponse(BaseModel):
    completion: str
    full_text: str
    device: str


def build_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    req: GenerateRequest,
) -> tuple[str, str]:
    formatted = build_prompt(tokenizer=tokenizer, prompt=req.prompt)
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        do_sample=True,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repetition_penalty=req.repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return completion, full_text


def create_app(base_model_id: str, adapter_path: str) -> FastAPI:
    device = detect_device()
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    adapter_dir = Path(adapter_path).expanduser().resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter path not found: {adapter_dir}. "
            "Train first, or point --adapter-path to an existing checkpoint directory."
        )
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"{adapter_dir} does not look like a PEFT adapter directory "
            "(missing adapter_config.json)."
        )

    # Use adapter tokenizer when present (final export), otherwise fallback to base tokenizer.
    tokenizer_source = str(adapter_dir) if (adapter_dir / "tokenizer_config.json").exists() else base_model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model = model.to(device)
    model.eval()

    app = FastAPI(title="TinyLlama LoRA API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "device": device,
            "base_model": base_model_id,
            "adapter_path": str(adapter_dir),
        }

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        completion, full_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            req=req,
        )
        return GenerateResponse(completion=completion, full_text=full_text, device=device)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve TinyLlama + LoRA adapter")
    parser.add_argument(
        "--base-model-id",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model used during LoRA fine-tuning",
    )
    parser.add_argument(
        "--adapter-path",
        default="outputs/tinyllama_lora/final",
        help="Path to adapter output from training",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    app = create_app(base_model_id=args.base_model_id, adapter_path=args.adapter_path)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
