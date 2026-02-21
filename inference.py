from pathlib import Path

import torch
import torch.nn as nn
import tiktoken

checkpoint_path = "tiny_llm_checkpoint.pt"
sample_prompts_path = "data/test_prompts.txt"
max_new_tokens = 60
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
PROMPT_SEPARATOR = "<END_PROMPT>"


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


def load_prompts(path):
    text = Path(path).read_text(encoding="utf-8")
    if PROMPT_SEPARATOR in text:
        prompts = [p.strip() for p in text.split(PROMPT_SEPARATOR) if p.strip()]
        if prompts:
            return prompts
    return [line.strip() for line in text.splitlines() if line.strip()]


def encode_text(tokenizer, text):
    return tokenizer.encode(text, disallowed_special=())


def sample_next_token(logits, temperature=0.8, top_k=40, top_p=0.9):
    logits = logits / temperature
    # Top-k
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
    
    probs = torch.softmax(logits, dim=-1)
    
    # Top-p (Nucleus)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    probs[indices_to_remove] = 0
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    return torch.multinomial(probs, 1)


@torch.no_grad()
def generate(model, tokenizer, prompt, context_length, max_new_tokens):
    model.eval()
    ids = torch.tensor(encode_text(tokenizer, prompt), device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        window = ids[:, -context_length:]
        logits = model(window)
        next_token = sample_next_token(logits[:, -1, :])
        ids = torch.cat([ids, next_token], dim=1)
    return tokenizer.decode(ids[0].tolist())


if not Path(checkpoint_path).exists():
    raise FileNotFoundError(
        f"{checkpoint_path} not found. Run `python tiny_llm.py` first."
    )
if not Path(sample_prompts_path).exists():
    raise FileNotFoundError(
        f"{sample_prompts_path} not found. Run `python prepare_dataset.py` first."
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

prompts = load_prompts(sample_prompts_path)
print(f"Loaded checkpoint: {checkpoint_path}")
print(f"Testing {len(prompts)} prompt(s) from: {sample_prompts_path}\n")

for i, prompt in enumerate(prompts, start=1):
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        context_length=checkpoint["context_length"],
        max_new_tokens=max_new_tokens,
    )
    print(f"[{i}]")
    print("PROMPT:")
    print(prompt)
    print()
    print("RESPONSE:")
    print(output)
    print("=" * 80)
