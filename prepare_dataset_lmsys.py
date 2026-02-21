import argparse
import os
import re
import random
import glob
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

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
PLACEHOLDER_PATTERNS = (
    "NAME_",
    "PERSON_",
    "EMAIL_",
    "URL_",
)


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
    if any(p in text for p in PLACEHOLDER_PATTERNS):
        return False
    if text.count("```") > 2:
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


def extract_turns_instruction(row, min_chars, min_ascii_ratio):
    # Common instruction/qa schemas (alpaca/flan-like).
    instruction = row.get("instruction") or row.get("prompt") or row.get("question")
    input_text = row.get("input")
    output = row.get("output") or row.get("response") or row.get("answer") or row.get("completion")
    if not instruction or not output:
        return []

    user_text = clean_text(instruction)
    if input_text:
        user_text = f"{user_text}\n{clean_text(input_text)}"
    assistant_text = clean_text(output)
    if not is_quality_text(user_text, min_chars=min_chars, min_ascii_ratio=min_ascii_ratio):
        return []
    if not is_quality_text(assistant_text, min_chars=min_chars, min_ascii_ratio=min_ascii_ratio):
        return []
    return [("User", user_text), ("Assistant", assistant_text)]


def extract_turns_auto(row, min_chars, min_ascii_ratio):
    turns = extract_turns(row, min_chars=min_chars, min_ascii_ratio=min_ascii_ratio)
    if turns:
        return turns
    return extract_turns_instruction(row, min_chars=min_chars, min_ascii_ratio=min_ascii_ratio)


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
    parser.add_argument("--use-auth", action="store_true", help="Use logged-in HF auth token")
    parser.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env var)")
    parser.add_argument(
        "--local-parquet-glob",
        default=None,
        help="Local parquet glob, e.g. 'data/raw/lmsys_lmsys-chat-1m/**/*.parquet'",
    )
    parser.add_argument(
        "--extra-local-parquet-glob",
        action="append",
        default=[],
        help="Repeatable additional local parquet globs",
    )
    parser.add_argument(
        "--download-parquet-dir",
        default=None,
        help="Download parquet files from HF dataset repo into this local directory",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download parquet files then exit",
    )
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument(
        "--extra-dataset",
        action="append",
        default=[],
        help="Repeatable additional HF dataset ids",
    )
    parser.add_argument("--extra-split", default="train")
    parser.add_argument("--extra-max-samples", type=int, default=50000)
    parser.add_argument(
        "--cache-dir",
        default=str(HF_CACHE_DIR),
        help="Cache directory for datasets/arrow files",
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-prompts", type=int, default=25)
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--min-turn-chars", type=int, default=24)
    parser.add_argument("--min-ascii-ratio", type=float, default=0.995)
    args = parser.parse_args()

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    token = args.token or os.getenv("HF_TOKEN")
    token_arg = token if token else (True if args.use_auth else None)

    def expand_local_parquet_files(pattern):
        # Support recursive globs such as data/raw/**/.parquet and plain file paths.
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return [m for m in matches if Path(m).is_file()]
        p = Path(pattern)
        return [str(p)] if p.is_file() else []

    def load_rows_from_local_files(file_list, total_cap):
        loaded = []
        for fpath in file_list:
            remaining = total_cap - len(loaded)
            if remaining <= 0:
                break
            try:
                ds_part = load_dataset(
                    "parquet",
                    data_files=fpath,
                    split="train",
                    cache_dir=args.cache_dir,
                    token=token_arg,
                )
            except Exception as e:
                print(f"[warn] skipping unreadable parquet file: {fpath} ({e})")
                continue
            loaded.extend(load_rows(ds_part, remaining))
        return loaded

    if args.download_parquet_dir:
        Path(args.download_parquet_dir).mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.dataset,
            repo_type="dataset",
            local_dir=args.download_parquet_dir,
            allow_patterns=["*.parquet", "*.json", "*.md", "*.txt"],
            token=token_arg,
        )
        args.local_parquet_glob = str(Path(args.download_parquet_dir) / "*.parquet")
        print(f"Downloaded dataset parquet files to: {args.download_parquet_dir}")
        if args.download_only:
            print("Download-only mode enabled. Exiting.")
            return

    def load_rows(dataset_obj, sample_cap):
        loaded_rows = []
        max_samples = min(sample_cap, len(dataset_obj))
        for i in range(max_samples):
            turns = extract_turns_auto(
                dataset_obj[i],
                min_chars=args.min_turn_chars,
                min_ascii_ratio=args.min_ascii_ratio,
            )
            if len(turns) < args.min_turns:
                continue
            roles = [r for r, _ in turns]
            if "Assistant" not in roles or "User" not in roles:
                continue
            loaded_rows.append(turns)
        return loaded_rows

    if args.local_parquet_glob:
        main_files = expand_local_parquet_files(args.local_parquet_glob)
        if not main_files:
            raise ValueError(
                f"No parquet files matched --local-parquet-glob: {args.local_parquet_glob}"
            )
        rows = load_rows_from_local_files(main_files, args.max_samples)
        dataset_label = f"local parquet files: {len(main_files)} from {args.local_parquet_glob}"
    else:
        ds_main = load_dataset(
            args.dataset,
            split=args.split,
            cache_dir=args.cache_dir,
            token=token_arg,
        )
        dataset_label = f"{args.dataset} ({args.split})"
        rows = load_rows(ds_main, args.max_samples)

    for extra_glob in args.extra_local_parquet_glob:
        extra_files = expand_local_parquet_files(extra_glob)
        if not extra_files:
            print(f"[warn] no parquet files matched extra glob: {extra_glob}")
            continue
        rows.extend(load_rows_from_local_files(extra_files, args.extra_max_samples))
        dataset_label += f" + local parquet files: {len(extra_files)} from {extra_glob}"

    for extra_ds in args.extra_dataset:
        ds_extra = load_dataset(
            extra_ds,
            split=args.extra_split,
            cache_dir=args.cache_dir,
            token=token_arg,
        )
        rows.extend(load_rows(ds_extra, args.extra_max_samples))
        dataset_label += f" + {extra_ds} ({args.extra_split})"

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
