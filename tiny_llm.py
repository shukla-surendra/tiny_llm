import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from tqdm import trange

# -------- CONFIG --------
train_data_path = Path("data/train.txt")
test_data_path = Path("data/test.txt")

# ~120M-parameter configuration with conservative micro-batching for MPS laptops.
context_length = 1024
embed_size = 768
num_heads = 12
num_layers = 12
dropout = 0.1

batch_size = 1
grad_accum_steps = 32
lr = 3e-4
min_lr = 3e-5
steps = 1000000 # 1M Steps
eval_interval = 50
eval_batches = 20
eval_history_path = Path("logs/train_eval_history.csv")

checkpoint_path = "tiny_llm_checkpoint.pt"
latest_checkpoint_path = "tiny_llm_checkpoint_latest.pt"
best_checkpoint_path = "tiny_llm_checkpoint_best.pt"
final_checkpoint_path = "tiny_llm_checkpoint_final.pt"
max_new_tokens = 80
save_every_steps = 200
resume_training = True
resume_training = os.getenv("RESUME_TRAINING", "1") == "1"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def load_text(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python prepare_dataset.py` first."
        )
    return path.read_text(encoding="utf-8")


def get_batch(tokens, target_mask, ctx_len):
    max_start = len(tokens) - ctx_len
    ix = torch.randint(0, max_start, (batch_size,), device=device)
    x = torch.stack([tokens[i:i + ctx_len] for i in ix])
    y = torch.stack([tokens[i + 1:i + ctx_len + 1] for i in ix])
    y_mask = torch.stack([target_mask[i + 1:i + ctx_len + 1] for i in ix]).float()
    return x, y, y_mask


def encode_text(tokenizer, text):
    return tokenizer.encode(text, disallowed_special=())


def encode_with_assistant_mask(tokenizer, text):
    ids = []
    mask = []
    for line in text.splitlines(keepends=True):
        line_ids = encode_text(tokenizer, line)
        is_assistant = line.lstrip().startswith("Assistant:")
        ids.extend(line_ids)
        mask.extend([1 if is_assistant else 0] * len(line_ids))
    return (
        torch.tensor(ids, device=device),
        torch.tensor(mask, device=device, dtype=torch.float32),
    )


def masked_next_token_loss(logits, targets, target_mask, vocab_size):
    token_losses = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction="none",
    ).view(targets.shape)
    mask_sum = target_mask.sum()
    if mask_sum > 0:
        return (token_losses * target_mask).sum() / mask_sum
    return token_losses.mean()


def lr_for_step(step_idx):
    warmup_steps = max(200, int(steps * 0.02))
    if step_idx < warmup_steps:
        return lr * float(step_idx + 1) / float(warmup_steps)
    decay_steps = max(1, steps - warmup_steps)
    progress = min(1.0, max(0.0, (step_idx - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)).item())
    return min_lr + (lr - min_lr) * cosine


def append_eval_history(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "step",
                "est_epoch",
                "lr",
                "train_loss",
                "test_loss",
                "test_perplexity",
                "best_test_loss",
                "improved",
                "processed_tokens",
                "total_training_hours",
            ],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def safe_perplexity(loss_value):
    # Keep exponent bounded to avoid inf in logs when loss is high early in training.
    clipped = min(float(loss_value), 20.0)
    return float(torch.exp(torch.tensor(clipped)).item())


def format_duration(total_seconds):
    total_seconds = int(max(0, total_seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


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


def apply_repetition_penalty(next_logits, ids, penalty=1.1, window_size=None):
    if penalty <= 1.0:
        return next_logits
    adjusted = next_logits.clone()
    effective_window = ids.size(1) if window_size is None else max(1, int(window_size))
    recent_ids = ids[0, -effective_window:].unique()
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
def estimate_loss(model, train_tokens, train_mask, test_tokens, test_mask, ctx_len, vocab_size):
    model.eval()
    out = {}
    for split_name, split_tokens, split_mask in (
        ("train", train_tokens, train_mask),
        ("test", test_tokens, test_mask),
    ):
        losses = []
        for _ in range(eval_batches):
            xb, yb, mb = get_batch(split_tokens, split_mask, ctx_len)
            logits = model(xb)
            loss = masked_next_token_loss(logits, yb, mb, vocab_size)
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
        next_logits = apply_repetition_penalty(
            next_logits,
            ids,
            penalty=1.1,
            window_size=ctx_len,
        )
        if do_sample:
            next_token = sample_next_token(next_logits)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, next_token], dim=1)
    return tokenizer.decode(ids[0].tolist())


def make_checkpoint_payload(
    model,
    optimizer,
    step,
    best_test_loss,
    vocab_size,
    ctx_len,
    processed_tokens,
    total_training_seconds,
):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_test_loss": best_test_loss,
        "processed_tokens": processed_tokens,
        "total_training_seconds": float(total_training_seconds),
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
train_tokens, train_target_mask = encode_with_assistant_mask(enc, train_text)
test_tokens, test_target_mask = encode_with_assistant_mask(enc, test_text)
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
processed_tokens = 0
total_training_seconds = 0.0

if resume_training and Path(latest_checkpoint_path).exists():
    resume_ckpt = torch.load(latest_checkpoint_path, map_location=device)
    model.load_state_dict(resume_ckpt["model_state_dict"])
    if "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
    start_step = int(resume_ckpt.get("step", -1)) + 1
    best_test_loss = float(resume_ckpt.get("best_test_loss", best_test_loss))
    processed_tokens = int(
        resume_ckpt.get(
            "processed_tokens",
            start_step * batch_size * effective_context_length,
        )
    )
    total_training_seconds = float(resume_ckpt.get("total_training_seconds", 0.0))
    print(f"Resumed from {latest_checkpoint_path} at step {start_step}")
    print(f"Cumulative training time: {format_duration(total_training_seconds)}")

run_start_time = time.time()


def current_total_training_seconds():
    return total_training_seconds + (time.time() - run_start_time)

progress = trange(steps, desc="training", unit="step", initial=start_step)
optimizer.zero_grad(set_to_none=True)
last_completed_step = start_step - 1
interrupted = False
try:
    for step in range(start_step, steps):
        if step % eval_interval == 0 or step == steps - 1:
            losses = estimate_loss(
                model=model,
                train_tokens=train_tokens,
                train_mask=train_target_mask,
                test_tokens=test_tokens,
                test_mask=test_target_mask,
                ctx_len=effective_context_length,
                vocab_size=vocab_size,
            )
            est_epoch = processed_tokens / len(train_tokens)
            improved = losses["test"] < best_test_loss
            if improved:
                best_test_loss = losses["test"]
                best_payload = make_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_test_loss=best_test_loss,
                    vocab_size=vocab_size,
                    ctx_len=effective_context_length,
                    processed_tokens=processed_tokens,
                    total_training_seconds=current_total_training_seconds(),
                )
                torch.save(best_payload, best_checkpoint_path)
                torch.save(best_payload, checkpoint_path)

            append_eval_history(
                eval_history_path,
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "step": step,
                    "est_epoch": f"{est_epoch:.6f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.8e}",
                    "train_loss": f"{losses['train']:.6f}",
                    "test_loss": f"{losses['test']:.6f}",
                    "test_perplexity": f"{safe_perplexity(losses['test']):.6f}",
                    "best_test_loss": f"{best_test_loss:.6f}",
                    "improved": int(improved),
                    "processed_tokens": processed_tokens,
                    "total_training_hours": f"{current_total_training_seconds() / 3600.0:.4f}",
                },
            )
            progress.set_postfix(
                train_loss=f"{losses['train']:.4f}",
                test_loss=f"{losses['test']:.4f}",
                test_ppl=f"{safe_perplexity(losses['test']):.1f}",
                est_epoch=f"{est_epoch:.3f}",
                total_h=f"{current_total_training_seconds() / 3600.0:.2f}",
            )

        xb, yb, mb = get_batch(train_tokens, train_target_mask, effective_context_length)
        logits = model(xb)
        loss = masked_next_token_loss(logits, yb, mb, vocab_size)

        loss_to_backprop = loss / grad_accum_steps
        loss_to_backprop.backward()
        if ((step - start_step + 1) % grad_accum_steps == 0) or (step == steps - 1):
            current_lr = lr_for_step(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        progress.set_postfix(
            batch_loss=f"{loss.item():.4f}",
            est_epoch=f"{processed_tokens / len(train_tokens):.3f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
        progress.update(1)
        processed_tokens += batch_size * effective_context_length
        last_completed_step = step

        if (step + 1) % save_every_steps == 0:
            latest_payload = make_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                step=step,
                best_test_loss=best_test_loss,
                vocab_size=vocab_size,
                ctx_len=effective_context_length,
                processed_tokens=processed_tokens,
                total_training_seconds=current_total_training_seconds(),
            )
            torch.save(latest_payload, latest_checkpoint_path)
except KeyboardInterrupt:
    interrupted = True
    print("\nTraining interrupted. Saving resumable checkpoint...")
finally:
    progress.close()
    total_training_seconds = current_total_training_seconds()

current_step = max(last_completed_step, start_step - 1)
latest_payload = make_checkpoint_payload(
    model=model,
    optimizer=optimizer,
    step=current_step,
    best_test_loss=best_test_loss,
    vocab_size=vocab_size,
    ctx_len=effective_context_length,
    processed_tokens=processed_tokens,
    total_training_seconds=total_training_seconds,
)
torch.save(latest_payload, latest_checkpoint_path)

if interrupted:
    print(f"Saved latest checkpoint: {latest_checkpoint_path}")
    print(f"Total cumulative training time: {format_duration(total_training_seconds)}")
    print("Resume training by running: python tiny_llm.py")
    raise SystemExit(0)

final_payload = make_checkpoint_payload(
    model=model,
    optimizer=optimizer,
    step=max(current_step, steps - 1),
    best_test_loss=best_test_loss,
    vocab_size=vocab_size,
    ctx_len=effective_context_length,
    processed_tokens=processed_tokens,
    total_training_seconds=total_training_seconds,
)
torch.save(final_payload, final_checkpoint_path)
torch.save(final_payload, latest_checkpoint_path)
if not Path(checkpoint_path).exists():
    torch.save(final_payload, checkpoint_path)
print(f"Saved latest checkpoint: {latest_checkpoint_path}")
print(f"Saved final checkpoint: {final_checkpoint_path}")
print(f"Best-serving checkpoint: {checkpoint_path}")
print(f"Eval history CSV: {eval_history_path}")
print(f"Total cumulative training time: {format_duration(total_training_seconds)}")

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
