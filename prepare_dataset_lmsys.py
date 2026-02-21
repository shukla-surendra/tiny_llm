import argparse
import re
import random
from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path("data")
TRAIN_PATH = OUTPUT_DIR / "train.txt"
TEST_PATH = OUTPUT_DIR / "test.txt"
TEST_PROMPTS_PATH = OUTPUT_DIR / "test_prompts.txt"
PROMPT_SEPARATOR = "\n\n<END_PROMPT>\n\n"
HF_CACHE_DIR = OUTPUT_DIR / "hf_cache"

ROLE_MAP = {
    "user": "User",
    "assistant": "Assistant",
    "system": "System",
    "human": "User",
    "gpt": "Assistant",
}
ALLOWED_ROLES = {"System", "User", "Assistant"}


def normalize_role(role):
    if role is None:
        return None
    key = str(role).strip().lower()
    return ROLE_MAP.get(key)


def clean_text(text):
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\uFFFD", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_quality_text(text, min_chars, min_ascii_ratio):
    if len(text) < min_chars:
        return False
    printable_ratio = sum(ch.isprintable() or ch in "\n\t" for ch in text) / len(text)
    if printable_ratio < 0.98:
        return False
    ascii_ratio = sum(ord(ch) < 128 for ch in text) / len(text)
    if ascii_ratio < min_ascii_ratio:
        return False
    alpha_count = sum(ch.isalpha() for ch in text)
    if alpha_count < max(5, len(text) // 25):
        return False
    return True


def extract_turns(row, min_chars, min_ascii_ratio):
    # Common LMSYS format uses a `conversation` field containing role/content dicts.
    # Fallbacks below keep this script robust to minor schema variants.
    conv = row.get("conversation") or row.get("conversations") or row.get("messages")
    if not isinstance(conv, list):
        return []

    turns = []
    for item in conv:
        if not isinstance(item, dict):
            continue
        role = normalize_role(item.get("role") or item.get("from"))
        if role not in ALLOWED_ROLES:
            continue
        text = item.get("content") or item.get("value")
        if not role or not text:
            continue
        text = clean_text(text)
        if not text or not is_quality_text(text, min_chars=min_chars, min_ascii_ratio=min_ascii_ratio):
            continue
        turns.append((role, text))
    return turns


def turns_to_text(turns):
    return "\n".join(f"{role}: {text}" for role, text in turns)


def prompt_from_turns(turns):
    if len(turns) < 2:
        return None

    # Prefer prompt ending right before an assistant answer.
    last_assistant_idx = -1
    for i in range(len(turns) - 1, -1, -1):
        if turns[i][0] == "Assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx > 0:
        prefix = turns[:last_assistant_idx]
    else:
        prefix = turns

    if not prefix:
        return None
    return turns_to_text(prefix) + "\nAssistant:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lmsys/lmsys-chat-1m")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--local-parquet-glob",
        default=None,
        help="Local parquet glob, e.g. 'data/lmsys/lmsys-chat-1m/*.parquet'",
    )
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument(
        "--cache-dir",
        default=str(HF_CACHE_DIR),
        help="Cache directory for datasets/arrow files",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-prompts", type=int, default=25)
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--min-turn-chars", type=int, default=12)
    parser.add_argument("--min-ascii-ratio", type=float, default=0.95)
    args = parser.parse_args()

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    if args.local_parquet_glob:
        ds = load_dataset(
            "parquet",
            data_files=args.local_parquet_glob,
            split="train",
            cache_dir=args.cache_dir,
        )
        dataset_label = f"local parquet: {args.local_parquet_glob}"
    else:
        ds = load_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
        dataset_label = f"{args.dataset} ({args.split})"

    rows = []
    max_samples = min(args.max_samples, len(ds))
    for i in range(max_samples):
        turns = extract_turns(
            ds[i],
            min_chars=args.min_turn_chars,
            min_ascii_ratio=args.min_ascii_ratio,
        )
        if len(turns) < args.min_turns:
            continue
        if "Assistant" not in [r for r, _ in turns] or "User" not in [r for r, _ in turns]:
            continue
        rows.append(turns)

    if len(rows) < 10:
        raise ValueError(
            "Too few valid conversations parsed. Check dataset access/schema and try again."
        )

    random.shuffle(rows)
    split_idx = int(len(rows) * args.train_ratio)
    train_rows = rows[:split_idx]
    test_rows = rows[split_idx:]

    train_text = "\n\n".join(turns_to_text(t) for t in train_rows) + "\n"
    test_text = "\n\n".join(turns_to_text(t) for t in test_rows) + "\n"
    TRAIN_PATH.write_text(train_text, encoding="utf-8")
    TEST_PATH.write_text(test_text, encoding="utf-8")

    prompts = []
    for turns in test_rows[: args.num_prompts]:
        p = prompt_from_turns(turns)
        if p:
            prompts.append(p)
    TEST_PROMPTS_PATH.write_text(PROMPT_SEPARATOR.join(prompts), encoding="utf-8")

    print(f"Dataset: {dataset_label}")
    print(f"Parsed conversations: {len(rows)}")
    print(f"Wrote train: {TRAIN_PATH}")
    print(f"Wrote test: {TEST_PATH}")
    print(f"Wrote prompts: {TEST_PROMPTS_PATH} ({len(prompts)} prompts)")


if __name__ == "__main__":
    main()
