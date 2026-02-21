import random
from pathlib import Path

OUTPUT_DIR = Path("data")
TRAIN_PATH = OUTPUT_DIR / "train.txt"
TEST_PATH = OUTPUT_DIR / "test.txt"
TEST_PROMPTS_PATH = OUTPUT_DIR / "test_prompts.txt"

SEED = 42
NUM_CONVERSATIONS = 2200
TRAIN_RATIO = 0.9
PROMPT_SEPARATOR = "\n\n<END_PROMPT>\n\n"

domains = [
    "python debugging",
    "sql optimization",
    "ml experiment tracking",
    "api authentication",
    "frontend performance",
    "incident triage",
    "docker workflows",
    "unit testing",
    "feature rollout",
    "security hardening",
]

tones = ["concise", "supportive", "direct", "pragmatic", "step-by-step"]
constraints = [
    "no external dependencies",
    "must keep backward compatibility",
    "limited to 2 hours",
    "safe for production",
    "works on low-resource laptops",
]

user_intents = [
    "I need help diagnosing a failure in {domain}.",
    "Can you show a checklist for {domain}?",
    "How should I structure a quick plan for {domain}?",
    "I am stuck and need an actionable path for {domain}.",
    "Give me a practical way to improve {domain}.",
]

assistant_openers = [
    "Let us keep this {tone} and reproducible.",
    "Here is a {tone} plan you can execute immediately.",
    "I will keep this {tone} and focused on measurable outcomes.",
]

follow_ups = [
    "Can you tailor this to the constraint: {constraint}?",
    "What should I test first before rollout?",
    "Can you provide a minimal example I can run now?",
]


def build_conversation(i):
    domain = random.choice(domains)
    tone = random.choice(tones)
    constraint = random.choice(constraints)
    issue_id = random.randint(1000, 9999)

    user_1 = random.choice(user_intents).format(domain=domain)
    assistant_1 = (
        f"{random.choice(assistant_openers).format(tone=tone)} "
        f"Step 1: capture baseline metrics. "
        f"Step 2: isolate one variable at a time. "
        f"Step 3: validate with a repeatable test."
    )
    user_2 = random.choice(follow_ups).format(constraint=constraint)
    assistant_2 = (
        f"Use issue-{issue_id}. "
        f"Apply constraint: {constraint}. "
        f"Start with a small change, add assertions, then run regression checks."
    )

    turns = [
        f"System: You are a helpful coding assistant for {domain}.",
        f"User: {user_1}",
        f"Assistant: {assistant_1}",
        f"User: {user_2}",
        f"Assistant: {assistant_2}",
    ]
    return "\n".join(turns)


def prompt_from_conversation(conversation):
    turns = conversation.splitlines()
    if len(turns) < 4:
        return None
    return "\n".join(turns[:4] + ["Assistant:"])


def main():
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conversations = [build_conversation(i) for i in range(NUM_CONVERSATIONS)]
    random.shuffle(conversations)

    split_idx = int(len(conversations) * TRAIN_RATIO)
    train_data = conversations[:split_idx]
    test_data = conversations[split_idx:]

    TRAIN_PATH.write_text("\n\n".join(train_data) + "\n", encoding="utf-8")
    TEST_PATH.write_text("\n\n".join(test_data) + "\n", encoding="utf-8")

    prompts = []
    for item in test_data[:25]:
        prompt = prompt_from_conversation(item)
        if prompt:
            prompts.append(prompt)
    TEST_PROMPTS_PATH.write_text(PROMPT_SEPARATOR.join(prompts), encoding="utf-8")

    print(f"Wrote {len(train_data)} conversations to {TRAIN_PATH}")
    print(f"Wrote {len(test_data)} conversations to {TEST_PATH}")
    print(f"Wrote {len(prompts)} prompts to {TEST_PROMPTS_PATH}")


if __name__ == "__main__":
    main()
