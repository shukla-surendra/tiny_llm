from pathlib import Path

import torch
import torch.nn as nn
import tiktoken

# -------- CONFIG --------
train_data_path = Path("data/train.txt")
test_data_path = Path("data/test.txt")

context_length = 128
embed_size = 128
num_heads = 4
num_layers = 2
batch_size = 16
lr = 1e-3
steps = 500
eval_interval = 50
eval_batches = 20

checkpoint_path = "tiny_llm_checkpoint.pt"
max_new_tokens = 60

device = "mps" if torch.backends.mps.is_available() else "cpu"


def load_text(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `.venv/bin/python prepare_dataset.py` first."
        )
    return path.read_text(encoding="utf-8")


train_text = load_text(train_data_path)
test_text = load_text(test_data_path)

enc = tiktoken.get_encoding("gpt2")
train_tokens = torch.tensor(enc.encode(train_text), device=device)
test_tokens = torch.tensor(enc.encode(test_text), device=device)
vocab_size = enc.n_vocab

if len(train_tokens) < 2 or len(test_tokens) < 2:
    raise ValueError("Train/test datasets must each contain at least 2 tokens.")

effective_context_length = min(context_length, len(train_tokens) - 1, len(test_tokens) - 1)
if effective_context_length < context_length:
    print(
        f"Info: reducing context_length from {context_length} "
        f"to {effective_context_length} for available dataset size."
    )


def get_batch(split):
    tokens = train_tokens if split == "train" else test_tokens
    max_start = len(tokens) - effective_context_length
    ix = torch.randint(0, max_start, (batch_size,), device=device)
    x = torch.stack([tokens[i:i + effective_context_length] for i in ix])
    y = torch.stack([tokens[i + 1:i + effective_context_length + 1] for i in ix])
    return x, y


class TinyLLM(nn.Module):
    def __init__(self, vocab_size, context_length, embed_size, num_heads, num_layers):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(context_length, embed_size)

        block = nn.TransformerEncoderLayer(
            embed_size,
            num_heads,
            batch_first=True
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


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "test"):
        losses = []
        for _ in range(eval_batches):
            xb, yb = get_batch(split)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def generate(model, prompt):
    ids = torch.tensor(enc.encode(prompt), device=device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            ids = ids[:, -effective_context_length:]
            logits = model(ids)
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
    return enc.decode(ids[0].tolist())


model = TinyLLM(
    vocab_size=vocab_size,
    context_length=effective_context_length,
    embed_size=embed_size,
    num_heads=num_heads,
    num_layers=num_layers,
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)

for step in range(steps):
    if step % eval_interval == 0 or step == steps - 1:
        losses = estimate_loss(model)
        print(
            f"step {step:4d} "
            f"train_loss {losses['train']:.4f} "
            f"test_loss {losses['test']:.4f}"
        )

    xb, yb = get_batch("train")
    logits = model(xb)
    loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "context_length": effective_context_length,
        "embed_size": embed_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "tokenizer": "gpt2",
        "train_data_path": str(train_data_path),
        "test_data_path": str(test_data_path),
        "dataset_type": "conversation",
    },
    checkpoint_path,
)
print(f"Saved checkpoint: {checkpoint_path}")

print("\nGenerated:")
print(
    generate(
        model,
        "System: You are a helpful coding assistant for python debugging.\n"
        "User: I need help diagnosing a flaky unit test.\n"
        "Assistant:",
    )
)
