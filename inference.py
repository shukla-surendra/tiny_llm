import torch
import torch.nn as nn
import tiktoken
from pathlib import Path

checkpoint_path = "mini_gpt_checkpoint.pt"
sample_prompts_path = "data/test_prompts.txt"
max_new_tokens = 40
device = "mps" if torch.backends.mps.is_available() else "cpu"
PROMPT_SEPARATOR = "<END_PROMPT>"


class MiniGPT(nn.Module):
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
        _, T = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        h = tok + pos
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device),
            diagonal=1,
        )
        h = self.transformer(h, mask=causal_mask)
        return self.lm_head(h)


def load_prompts(path):
    text = Path(path).read_text(encoding="utf-8")
    if PROMPT_SEPARATOR in text:
        prompts = [p.strip() for p in text.split(PROMPT_SEPARATOR) if p.strip()]
        if prompts:
            return prompts
    return [line.strip() for line in text.splitlines() if line.strip()]


def generate(model, tokenizer, prompt, context_length, max_new_tokens):
    ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ids = ids[:, -context_length:]
            logits = model(ids)
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)

    return tokenizer.decode(ids[0].tolist())


if not Path(checkpoint_path).exists():
    raise FileNotFoundError(
        f"{checkpoint_path} not found. Run `.venv/bin/python tiny_llm.py` first."
    )
if not Path(sample_prompts_path).exists():
    raise FileNotFoundError(
        f"{sample_prompts_path} not found. Run `.venv/bin/python prepare_dataset.py` first."
    )

checkpoint = torch.load(checkpoint_path, map_location=device)
tokenizer = tiktoken.get_encoding(checkpoint.get("tokenizer", "gpt2"))

model = MiniGPT(
    vocab_size=checkpoint["vocab_size"],
    context_length=checkpoint["context_length"],
    embed_size=checkpoint["embed_size"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
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
