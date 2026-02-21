from pathlib import Path
from typing import Optional

import tiktoken
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

checkpoint_path = Path("tiny_llm_checkpoint.pt")
device = "mps" if torch.backends.mps.is_available() else "cpu"


class TinyLLM(nn.Module):
    def __init__(self, vocab_size, context_length, embed_size, num_heads, num_layers):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(context_length, embed_size)

        block = nn.TransformerEncoderLayer(
            embed_size,
            num_heads,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers)
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        _, seq_len = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(seq_len, device=x.device))
        h = tok + pos
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1,
        )
        h = self.transformer(h, mask=causal_mask)
        return self.lm_head(h)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, description="Conversation prompt text")
    max_new_tokens: int = Field(default=80, ge=1, le=512)
    do_sample: bool = False
    temperature: float = Field(default=1.0, gt=0.0, le=5.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=50000)


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    full_text: str
    model_context_length: int
    device: str


def sample_next_token(logits, do_sample, temperature, top_k):
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        probs = torch.softmax(vals, dim=-1)
        chosen = torch.multinomial(probs, num_samples=1)
        return idx.gather(-1, chosen)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_text(model, tokenizer, context_length, req: GenerateRequest):
    ids = torch.tensor(tokenizer.encode(req.prompt), device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(req.max_new_tokens):
            window = ids[:, -context_length:]
            logits = model(window)
            next_logits = logits[:, -1, :]
            next_token = sample_next_token(
                logits=next_logits,
                do_sample=req.do_sample,
                temperature=req.temperature,
                top_k=req.top_k,
            )
            ids = torch.cat([ids, next_token], dim=1)

    full_text = tokenizer.decode(ids[0].tolist())
    completion = full_text[len(req.prompt):] if full_text.startswith(req.prompt) else full_text
    return completion, full_text


def load_runtime():
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found. Run `.venv/bin/python tiny_llm.py` first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = tiktoken.get_encoding(checkpoint.get("tokenizer", "gpt2"))

    model = TinyLLM(
        vocab_size=checkpoint["vocab_size"],
        context_length=checkpoint["context_length"],
        embed_size=checkpoint["embed_size"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, tokenizer, model


app = FastAPI(title="Tiny LLM API", version="1.0.0")
checkpoint, tokenizer, model = load_runtime()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "context_length": checkpoint["context_length"],
        "dataset_type": checkpoint.get("dataset_type", "unknown"),
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        completion, full_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            context_length=checkpoint["context_length"],
            req=req,
        )
        return GenerateResponse(
            prompt=req.prompt,
            completion=completion,
            full_text=full_text,
            model_context_length=checkpoint["context_length"],
            device=device,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
