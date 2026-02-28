import argparse
import math
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def find_last_checkpoint(output_dir: Path) -> str | None:
    if not output_dir.exists():
        return None
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None

    def checkpoint_step(p: Path) -> int:
        suffix = p.name.replace("checkpoint-", "")
        return int(suffix) if suffix.isdigit() else -1

    checkpoints.sort(key=checkpoint_step)
    last = checkpoints[-1]
    if checkpoint_step(last) < 0:
        return None
    return str(last)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_dtype(device: str, force_float32: bool) -> torch.dtype:
    if force_float32:
        return torch.float32
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def format_example(example: dict[str, Any], tokenizer: AutoTokenizer) -> str:
    if "messages" in example and isinstance(example["messages"], list):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    if "prompt" in example and "response" in example:
        messages = [
            {"role": "user", "content": str(example["prompt"])},
            {"role": "assistant", "content": str(example["response"])},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    if "instruction" in example and "output" in example:
        user_text = str(example["instruction"])
        if "input" in example and example["input"]:
            user_text = f"{user_text}\n\nInput:\n{example['input']}"
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": str(example["output"])},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    if "text" in example:
        return str(example["text"])

    raise ValueError(
        "Could not infer text format from dataset row. Supported keys: "
        "messages, prompt/response, instruction/output, or text."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for TinyLlama chat model")
    parser.add_argument(
        "--model-id",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceH4/ultrachat_200k",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--dataset-split",
        default="train_sft",
        help="Dataset split for training",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="Limit number of rows for quick iterations",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/tinyllama_lora",
        help="Where to write LoRA adapter",
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Absolute warmup steps. If 0, computed from warmup-ratio.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--save-strategy",
        default="steps",
        choices=["steps", "epoch"],
        help="Checkpoint save strategy",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save a checkpoint every N optimizer steps when using save-strategy=steps.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep on disk.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Automatically resume from latest checkpoint in output-dir if available.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="Explicit checkpoint path to resume from. Overrides --resume auto-detect.",
    )
    parser.add_argument(
        "--force-float32",
        action="store_true",
        help="Use float32 if float16 causes MPS instability",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = detect_device()
    dtype = model_dtype(device=device, force_float32=args.force_float32)
    print(f"[runtime] device={device} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    raw_ds: Dataset = load_dataset(args.dataset, split=args.dataset_split)
    if args.max_samples > 0:
        raw_ds = raw_ds.select(range(min(args.max_samples, len(raw_ds))))

    def to_text(example: dict[str, Any]) -> dict[str, str]:
        return {"text": format_example(example=example, tokenizer=tokenizer)}

    text_ds = raw_ds.map(
        to_text,
        remove_columns=raw_ds.column_names,
        desc="Formatting chat text",
    )

    def tokenize_rows(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.seq_len,
            padding=False,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    tokenized = text_ds.map(
        tokenize_rows,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    report_to = []
    if os.getenv("WANDB_API_KEY"):
        report_to = ["wandb"]

    samples_per_update = max(1, args.batch_size * args.grad_accum)
    updates_per_epoch = max(1, math.ceil(len(tokenized) / samples_per_update))
    total_train_steps = max(1, math.ceil(updates_per_epoch * args.epochs))
    warmup_steps = args.warmup_steps
    if warmup_steps <= 0:
        warmup_steps = int(total_train_steps * args.warmup_ratio)

    print(
        f"[train] updates_per_epoch={updates_per_epoch} "
        f"total_train_steps~{total_train_steps} warmup_steps={warmup_steps}"
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim="adamw_torch",
        report_to=report_to,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized,
        "data_collator": data_collator,
    }
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        try:
            trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)
        except TypeError:
            trainer = Trainer(**trainer_kwargs)

    resume_checkpoint = None
    if args.resume_from_checkpoint:
        resume_checkpoint = args.resume_from_checkpoint
    elif args.resume:
        resume_checkpoint = find_last_checkpoint(output_dir)

    if resume_checkpoint:
        print(f"[resume] continuing from {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        print("[resume] no checkpoint found; starting fresh")
        trainer.train()

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[done] adapter saved at {final_dir}")


if __name__ == "__main__":
    main()
