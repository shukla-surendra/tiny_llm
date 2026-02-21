from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from tqdm import trange

# -------- CONFIG --------
train_data_path = Path("data/train.txt")
test_data_path = Path("data/test.txt")

# ~120M-parameter configuration with conservative micro-batching for MPS laptops.
context_length = 128
embed_size = 768
num_heads = 12
num_layers = 12
dropout = 0.1

batch_size = 1
grad_accum_steps = 8
lr = 3e-4
steps = 6000
eval_interval = 50
eval_batches = 20

checkpoint_path = "tiny_llm_checkpoint.pt"
latest_checkpoint_path = "tiny_llm_checkpoint_latest.pt"
best_checkpoint_path = "tiny_llm_checkpoint_best.pt"
final_checkpoint_path = "tiny_llm_checkpoint_final.pt"
max_new_tokens = 80
save_every_steps = 200
resume_training = True

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def load_text(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `.venv/bin/python prepare_dataset.py` first."
        )
    return path.read_text(encoding="utf-8")


def get_batch(tokens, ctx_len):
    max_start = len(tokens) - ctx_len
    ix = torch.randint(0, max_start, (batch_size,), device=device)
    x = torch.stack([tokens[i:i + ctx_len] for i in ix])
    y = torch.stack([tokens[i + 1:i + ctx_len + 1] for i in ix])
    return x, y


def encode_text(tokenizer, text):
    return tokenizer.encode(text, disallowed_special=())


def sample_next_token(logits, temperature=0.9, top_k=40, top_p=0.95):
    logits = logits / temperature
    k = min(top_k, logits.size(-1))
    vals, idx = torch.topk(logits, k, dim=-1)
    probs = torch.softmax(vals, dim=-1)
    if top_p is not None:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = cumsum <= top_p
        keep[..., 0] = True
        filtered = torch.zeros_like(probs)
        filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_probs * keep)
        norm = filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = filtered / norm
    chosen = torch.multinomial(probs, num_samples=1)
    return idx.gather(-1, chosen)


def apply_repetition_penalty(next_logits, ids, penalty=1.1):
    if penalty <= 1.0:
        return next_logits
    adjusted = next_logits.clone()
    recent_ids = ids[0, -128:].unique()
    adjusted[:, recent_ids] = adjusted[:, recent_ids] / penalty
    return adjusted


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads")
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
        self.context_length = context_length
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(context_length, embed_size)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [GPTBlock(embed_size, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)

        # Common in GPT models: tie input embedding and output projection weights.
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


@torch.no_grad()
def estimate_loss(model, train_tokens, test_tokens, ctx_len, vocab_size):
    model.eval()
    out = {}
    for split_name, split_tokens in (("train", train_tokens), ("test", test_tokens)):
        losses = []
        for _ in range(eval_batches):
            xb, yb = get_batch(split_tokens, ctx_len)
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))
            losses.append(loss.item())
        out[split_name] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def generate(model, tokenizer, prompt, ctx_len, max_new_tokens, do_sample=True):
    model.eval()
    ids = torch.tensor(encode_text(tokenizer, prompt), device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        window = ids[:, -ctx_len:]
        logits = model(window)
        next_logits = logits[:, -1, :]
        next_logits = apply_repetition_penalty(next_logits, ids, penalty=1.1)
        if do_sample:
            next_token = sample_next_token(next_logits)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, next_token], dim=1)
    return tokenizer.decode(ids[0].tolist())


def make_checkpoint_payload(model, optimizer, step, best_test_loss, vocab_size, ctx_len):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_test_loss": best_test_loss,
        "vocab_size": vocab_size,
        "context_length": ctx_len,
        "embed_size": embed_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "grad_accum_steps": grad_accum_steps,
        "tokenizer": "gpt2",
        "architecture": "gpt_decoder_pre_norm_weight_tied",
        "train_data_path": str(train_data_path),
        "test_data_path": str(test_data_path),
    }


train_text = load_text(train_data_path)
test_text = load_text(test_data_path)

enc = tiktoken.get_encoding("gpt2")
train_tokens = torch.tensor(encode_text(enc, train_text), device=device)
test_tokens = torch.tensor(encode_text(enc, test_text), device=device)
vocab_size = enc.n_vocab

if len(train_tokens) < 2 or len(test_tokens) < 2:
    raise ValueError("Train/test datasets must each contain at least 2 tokens.")

effective_context_length = min(context_length, len(train_tokens) - 1, len(test_tokens) - 1)
if effective_context_length < context_length:
    print(
        f"Info: reducing context_length from {context_length} "
        f"to {effective_context_length} for available dataset size."
    )

model = TinyGPT(
    vocab_size=vocab_size,
    context_length=effective_context_length,
    embed_size=embed_size,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
best_test_loss = float("inf")
start_step = 0

if resume_training and Path(latest_checkpoint_path).exists():
    resume_ckpt = torch.load(latest_checkpoint_path, map_location=device)
    model.load_state_dict(resume_ckpt["model_state_dict"])
    if "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    start_step = int(resume_ckpt.get("step", -1)) + 1
    best_test_loss = float(resume_ckpt.get("best_test_loss", best_test_loss))
    print(f"Resumed from {latest_checkpoint_path} at step {start_step}")

progress = trange(steps, desc="training", unit="step", initial=start_step)
optimizer.zero_grad(set_to_none=True)
for step in range(start_step, steps):
    if step % eval_interval == 0 or step == steps - 1:
        losses = estimate_loss(
            model=model,
            train_tokens=train_tokens,
            test_tokens=test_tokens,
            ctx_len=effective_context_length,
            vocab_size=vocab_size,
        )
        progress.set_postfix(
            train_loss=f"{losses['train']:.4f}",
            test_loss=f"{losses['test']:.4f}",
        )
        if losses["test"] < best_test_loss:
            best_test_loss = losses["test"]
            best_payload = make_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                step=step,
                best_test_loss=best_test_loss,
                vocab_size=vocab_size,
                ctx_len=effective_context_length,
            )
            torch.save(best_payload, best_checkpoint_path)
            torch.save(best_payload, checkpoint_path)

    xb, yb = get_batch(train_tokens, effective_context_length)
    logits = model(xb)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))

    loss_to_backprop = loss / grad_accum_steps
    loss_to_backprop.backward()
    if ((step - start_step + 1) % grad_accum_steps == 0) or (step == steps - 1):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    progress.set_postfix(
        batch_loss=f"{loss.item():.4f}",
    )
    progress.update(1)

    if (step + 1) % save_every_steps == 0:
        latest_payload = make_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            step=step,
            best_test_loss=best_test_loss,
            vocab_size=vocab_size,
            ctx_len=effective_context_length,
        )
        torch.save(latest_payload, latest_checkpoint_path)

final_payload = make_checkpoint_payload(
    model=model,
    optimizer=optimizer,
    step=steps - 1,
    best_test_loss=best_test_loss,
    vocab_size=vocab_size,
    ctx_len=effective_context_length,
)
torch.save(final_payload, final_checkpoint_path)
torch.save(final_payload, latest_checkpoint_path)
if not Path(checkpoint_path).exists():
    torch.save(final_payload, checkpoint_path)
print(f"Saved latest checkpoint: {latest_checkpoint_path}")
print(f"Saved final checkpoint: {final_checkpoint_path}")
print(f"Best-serving checkpoint: {checkpoint_path}")

prompt = (
    "System: You are a helpful coding assistant for unit testing.\n"
    "User: My tests are flaky in CI. Give me a quick plan.\n"
    "Assistant:"
)
generated = generate(
    model=model,
    tokenizer=enc,
    prompt=prompt,
    ctx_len=effective_context_length,
    max_new_tokens=max_new_tokens,
    do_sample=True,
)
completion = generated[len(prompt):] if generated.startswith(prompt) else generated
for marker in ("\nUser:", "\nSystem:"):
    idx = completion.find(marker)
    if idx != -1:
        completion = completion[:idx]
completion = completion.lstrip()
if completion.startswith("Assistant:"):
    completion = completion[len("Assistant:"):].lstrip()
print("\nGenerated Completion:")
print(completion if completion.strip() else "[empty completion]")
