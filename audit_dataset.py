from pathlib import Path
import statistics


TRAIN_PATH = Path("data/train.txt")
TEST_PATH = Path("data/test.txt")


def load_text(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return path.read_text(encoding="utf-8")


def line_stats(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    role_counts = {"System": 0, "User": 0, "Assistant": 0, "Other": 0}
    assistant_lengths = []
    user_lengths = []
    system_lengths = []
    noisy_lines = 0

    for ln in lines:
        role = "Other"
        content = ln
        for r in ("System:", "User:", "Assistant:"):
            if ln.startswith(r):
                role = r[:-1]
                content = ln[len(r):].strip()
                break
        role_counts[role] += 1
        if role == "Assistant":
            assistant_lengths.append(len(content))
        elif role == "User":
            user_lengths.append(len(content))
        elif role == "System":
            system_lengths.append(len(content))
        if "NAME_" in ln or "�" in ln:
            noisy_lines += 1

    ascii_ratio = (
        sum(1 for ch in text if ord(ch) < 128) / max(1, len(text))
    )
    words = [w for w in text.split() if w]
    unique_word_ratio = len(set(words)) / max(1, len(words))

    return {
        "lines": len(lines),
        "role_counts": role_counts,
        "assistant_avg_chars": statistics.mean(assistant_lengths) if assistant_lengths else 0.0,
        "user_avg_chars": statistics.mean(user_lengths) if user_lengths else 0.0,
        "system_avg_chars": statistics.mean(system_lengths) if system_lengths else 0.0,
        "noise_line_rate": noisy_lines / max(1, len(lines)),
        "ascii_ratio": ascii_ratio,
        "unique_word_ratio": unique_word_ratio,
        "assistant_line_set": set(
            ln for ln in lines if ln.startswith("Assistant:")
        ),
    }


def print_split(name, stats):
    total = max(1, stats["lines"])
    rc = stats["role_counts"]
    print(f"== {name} ==")
    print(f"lines: {stats['lines']}")
    print(
        "role_distribution: "
        f"System={rc['System']/total:.3f}, "
        f"User={rc['User']/total:.3f}, "
        f"Assistant={rc['Assistant']/total:.3f}, "
        f"Other={rc['Other']/total:.3f}"
    )
    print(f"assistant_avg_chars: {stats['assistant_avg_chars']:.1f}")
    print(f"user_avg_chars: {stats['user_avg_chars']:.1f}")
    print(f"system_avg_chars: {stats['system_avg_chars']:.1f}")
    print(f"noise_line_rate: {stats['noise_line_rate']:.4f}")
    print(f"ascii_ratio: {stats['ascii_ratio']:.4f}")
    print(f"unique_word_ratio: {stats['unique_word_ratio']:.4f}")


def main():
    train_text = load_text(TRAIN_PATH)
    test_text = load_text(TEST_PATH)
    train = line_stats(train_text)
    test = line_stats(test_text)

    overlap = len(train["assistant_line_set"] & test["assistant_line_set"])
    overlap_denom = max(1, len(test["assistant_line_set"]))
    overlap_rate = overlap / overlap_denom

    print_split("TRAIN", train)
    print()
    print_split("TEST", test)
    print()
    print("== CROSS-SPLIT ==")
    print(f"assistant_exact_overlap_lines: {overlap}")
    print(f"assistant_exact_overlap_rate_vs_test: {overlap_rate:.4f}")
    print()
    print("Recommended targets:")
    print("- noise_line_rate < 0.01")
    print("- ascii_ratio > 0.98")
    print("- assistant_exact_overlap_rate_vs_test close to 0.0")


if __name__ == "__main__":
    main()
