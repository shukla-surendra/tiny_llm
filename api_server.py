from pathlib import Path
from typing import Optional

import tiktoken
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

checkpoint_path = Path("tiny_llm_checkpoint.pt")
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, seq_len, _ = x.shape
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1,
        )
        out, _ = self.attn(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask,
            need_weights=False,
        )
        return self.dropout(out)


class MLP(nn.Module):
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GPTBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_size)
        self.attn = CausalSelfAttention(embed_size, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_size)
        self.mlp = MLP(embed_size, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, context_length, embed_size, num_heads, num_layers, dropout):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(context_length, embed_size)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [GPTBlock(embed_size, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, x):
        _, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.lm_head(h)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, description="Conversation prompt text")
    max_new_tokens: int = Field(default=100, ge=1, le=512)
    do_sample: bool = False
    temperature: float = Field(default=1.0, gt=0.0, le=5.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=50000)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    full_text: str
    model_context_length: int
    device: str
    architecture: str


def sample_next_token(logits, do_sample, temperature, top_k, top_p):
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
    if top_p is not None:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum <= top_p
        keep[..., 0] = True
        filtered = torch.zeros_like(probs)
        filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_probs * keep)
        norm = filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = filtered / norm
    return torch.multinomial(probs, num_samples=1)


def encode_text(tokenizer, text):
    return tokenizer.encode(text, disallowed_special=())


def apply_repetition_penalty(next_logits, ids, penalty):
    if penalty <= 1.0:
        return next_logits
    adjusted = next_logits.clone()
    recent_ids = ids[0, -128:].unique()
    adjusted[:, recent_ids] = adjusted[:, recent_ids] / penalty
    return adjusted


def postprocess_completion(text):
    cleaned = text.lstrip()
    if cleaned.startswith("Assistant:"):
        cleaned = cleaned[len("Assistant:"):].lstrip()
    for marker in ("\nUser:", "\nSystem:"):
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx]
    return cleaned.strip()


@torch.no_grad()
def generate_text(model, tokenizer, context_length, req: GenerateRequest):
    model.eval()
    ids = torch.tensor(encode_text(tokenizer, req.prompt), device=device).unsqueeze(0)
    for _ in range(req.max_new_tokens):
        window = ids[:, -context_length:]
        logits = model(window)
        next_logits = apply_repetition_penalty(
            next_logits=logits[:, -1, :],
            ids=ids,
            penalty=req.repetition_penalty,
        )
        next_token = sample_next_token(
            logits=next_logits,
            do_sample=req.do_sample,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        ids = torch.cat([ids, next_token], dim=1)

    full_text = tokenizer.decode(ids[0].tolist())
    raw_completion = full_text[len(req.prompt):] if full_text.startswith(req.prompt) else full_text
    completion = postprocess_completion(raw_completion)
    return completion, full_text


def load_runtime():
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found. Run `.venv/bin/python tiny_llm.py` first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = tiktoken.get_encoding(checkpoint.get("tokenizer", "gpt2"))

    model = TinyGPT(
        vocab_size=checkpoint["vocab_size"],
        context_length=checkpoint["context_length"],
        embed_size=checkpoint["embed_size"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint.get("dropout", 0.0),
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
        "architecture": checkpoint.get("architecture", "unknown"),
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
            architecture=checkpoint.get("architecture", "unknown"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
