import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import tiktoken

PROMPT_SEPARATOR = "<END_PROMPT>"


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


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


def parse_args():
    p = argparse.ArgumentParser(description="Run quality evaluation on a checkpoint.")
    p.add_argument("--checkpoint", default="tiny_llm_checkpoint.pt")
    p.add_argument("--prompts", default="data/test_prompts.txt")
    p.add_argument("--max-prompts", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--repetition-penalty", type=float, default=1.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-jsonl", default="logs/quality_history.jsonl")
    p.add_argument("--compare-last", action="store_true")
    return p.parse_args()


def load_prompts(path):
    default_prompts = [
        "System: You are a helpful coding assistant for docker workflows.\nUser: How do I debug container startup failures?\nAssistant:",
        "System: You are a helpful coding assistant for python debugging.\nUser: Why do I get KeyError in pandas and how can I fix it?\nAssistant:",
        "System: You are a helpful coding assistant for SQL optimization.\nUser: My query is slow on a large table. What should I check first?\nAssistant:",
        "System: You are a helpful coding assistant for unit testing.\nUser: How should I structure tests for a new API endpoint?\nAssistant:",
        "System: You are a helpful coding assistant for incident response.\nUser: Production latency doubled after deploy. What immediate steps should I take?\nAssistant:",
    ]
    p = Path(path)
    if not p.exists():
        return default_prompts
    text = p.read_text(encoding="utf-8")
    if PROMPT_SEPARATOR in text:
        prompts = [x.strip() for x in text.split(PROMPT_SEPARATOR) if x.strip()]
        return prompts if prompts else default_prompts
    prompts = [line.strip() for line in text.splitlines() if line.strip()]
    return prompts if prompts else default_prompts


def encode_text(tokenizer, text):
    return tokenizer.encode(text, disallowed_special=())


def apply_repetition_penalty(next_logits, ids, penalty=1.25):
    if penalty <= 1.0:
        return next_logits
    adjusted = next_logits.clone()
    recent_ids = ids[0, -128:].unique()
    adjusted[:, recent_ids] = adjusted[:, recent_ids] / penalty
    return adjusted


def sample_next_token(logits, temperature=0.5, top_k=15, top_p=0.8):
    logits = logits / max(temperature, 1e-5)
    k = min(top_k, logits.size(-1))
    kth_vals = torch.topk(logits, k, dim=-1).values[..., -1].unsqueeze(-1)
    logits = torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    keep = cumsum <= top_p
    keep[..., 0] = True
    filtered = torch.zeros_like(probs)
    filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_probs * keep)
    probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(model, tokenizer, prompt, context_length, args):
    ids = torch.tensor(encode_text(tokenizer, prompt), device=DEVICE).unsqueeze(0)
    model.eval()
    for _ in range(args.max_new_tokens):
        window = ids[:, -context_length:]
        logits = model(window)
        next_logits = apply_repetition_penalty(
            logits[:, -1, :], ids, penalty=args.repetition_penalty
        )
        if args.do_sample:
            next_token = sample_next_token(
                next_logits,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, next_token], dim=1)
    full_text = tokenizer.decode(ids[0].tolist())
    completion = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
    return completion.strip()


def word_repetition_ratio(text):
    words = [w for w in text.lower().split() if w]
    if not words:
        return 1.0
    uniq = len(set(words))
    return 1.0 - (uniq / len(words))


def ascii_ratio(text):
    if not text:
        return 1.0
    return sum(ord(c) < 128 for c in text) / len(text)


def placeholder_noise_ratio(text):
    if not text:
        return 0.0
    markers = ("NAME_", "��", "�")
    if any(m in text for m in markers):
        return 1.0
    return 0.0


def load_last_row(path):
    p = Path(path)
    if not p.exists():
        return None
    last = None
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if not last:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None


def append_report(path, report):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{args.checkpoint} not found. Run `python tiny_llm.py` first.")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    tokenizer = tiktoken.get_encoding(checkpoint.get("tokenizer", "gpt2"))
    model = TinyGPT(
        vocab_size=checkpoint["vocab_size"],
        context_length=checkpoint["context_length"],
        embed_size=checkpoint["embed_size"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint.get("dropout", 0.0),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    prompts = load_prompts(args.prompts)[: args.max_prompts]
    non_empty = 0
    ascii_scores = []
    repeat_scores = []
    role_leak_count = 0
    lengths = []
    placeholder_noise = []
    examples = []

    for p in prompts:
        completion = generate(model, tokenizer, p, checkpoint["context_length"], args)
        if completion.strip():
            non_empty += 1
        if "\nUser:" in completion or "\nSystem:" in completion:
            role_leak_count += 1
        ascii_scores.append(ascii_ratio(completion))
        repeat_scores.append(word_repetition_ratio(completion))
        placeholder_noise.append(placeholder_noise_ratio(completion))
        lengths.append(len(completion))
        if len(examples) < 3:
            examples.append((p, completion))

    score = 100.0
    score -= (1.0 - (non_empty / max(1, len(prompts)))) * 35.0
    score -= max(0.0, statistics.mean(repeat_scores) - 0.35) * 35.0
    score -= max(0.0, 0.96 - statistics.mean(ascii_scores)) * 100.0
    score -= (role_leak_count / max(1, len(prompts))) * 15.0
    score -= statistics.mean(placeholder_noise) * 15.0
    score = max(0.0, min(100.0, score))

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(ckpt_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "checkpoint_best_test_loss": float(checkpoint.get("best_test_loss", -1.0)),
        "device": DEVICE,
        "do_sample": bool(args.do_sample),
        "seed": int(args.seed),
        "prompts_count": len(prompts),
        "non_empty_rate": non_empty / max(1, len(prompts)),
        "avg_completion_chars": statistics.mean(lengths) if lengths else 0.0,
        "avg_ascii_ratio": statistics.mean(ascii_scores) if ascii_scores else 1.0,
        "avg_word_repetition_ratio": statistics.mean(repeat_scores) if repeat_scores else 1.0,
        "role_leak_rate": role_leak_count / max(1, len(prompts)),
        "placeholder_noise_rate": statistics.mean(placeholder_noise) if placeholder_noise else 0.0,
        "heuristic_quality_score_0_to_100": score,
    }

    previous = load_last_row(args.out_jsonl) if args.compare_last else None
    append_report(args.out_jsonl, report)

    print("=== Quality Report ===")
    print(f"checkpoint: {report['checkpoint']}")
    print(f"checkpoint_step: {report['checkpoint_step']}")
    print(f"checkpoint_best_test_loss: {report['checkpoint_best_test_loss']:.4f}")
    print(f"prompts: {report['prompts_count']}")
    print(f"non_empty_rate: {report['non_empty_rate']:.3f}")
    print(f"avg_completion_chars: {report['avg_completion_chars']:.1f}")
    print(f"avg_ascii_ratio: {report['avg_ascii_ratio']:.3f}")
    print(f"avg_word_repetition_ratio: {report['avg_word_repetition_ratio']:.3f}")
    print(f"role_leak_rate: {report['role_leak_rate']:.3f}")
    print(f"placeholder_noise_rate: {report['placeholder_noise_rate']:.3f}")
    print(f"heuristic_quality_score_0_to_100: {report['heuristic_quality_score_0_to_100']:.1f}")
    print(f"saved_report: {args.out_jsonl}")

    if previous:
        delta = report["heuristic_quality_score_0_to_100"] - previous.get(
            "heuristic_quality_score_0_to_100", 0.0
        )
        delta_loss = report["checkpoint_best_test_loss"] - previous.get(
            "checkpoint_best_test_loss", report["checkpoint_best_test_loss"]
        )
        print("\n=== Delta vs Previous Eval ===")
        print(f"quality_score_delta: {delta:+.2f}")
        print(f"best_test_loss_delta: {delta_loss:+.4f}")

    print("\n=== Sample Outputs ===")
    for i, (p, c) in enumerate(examples, start=1):
        print(f"\n[{i}] PROMPT:\n{p}\n")
        print("COMPLETION:")
        print(c if c else "[empty]")


if __name__ == "__main__":
    main()
