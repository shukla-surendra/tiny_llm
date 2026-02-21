# LLM Development Guide (Beginner-Friendly)

This guide explains your current project end-to-end and gives the core concepts needed to understand LLM development.

## 1) What you built

You built a tiny **autoregressive language model**:
- Input: text conversations
- Task: predict the **next token** repeatedly
- Model: Transformer-based network (mini GPT-style)
- Output: generated assistant-like responses

In this repo:
- Dataset creation: `prepare_dataset.py`
- Training: `tiny_llm.py`
- Inference/testing: `inference.py`
- Quality evaluation: `eval_quality.py`

---

## 2) Big picture: how modern LLMs work

At a high level, LLM development has these stages:
1. Collect text data
2. Tokenize text into integers
3. Train model to predict next token
4. Save checkpoint
5. Run inference (generation)
6. Evaluate quality and iterate

Your project follows exactly this shape, but at tiny scale.

---

## 3) Data pipeline in your project

### 3.1 Why data matters most

Model quality is heavily determined by:
- Data quality
- Data diversity
- Data scale

Architecture matters, but data is often the biggest lever.

### 3.2 What `prepare_dataset.py` does

It creates synthetic conversation data with roles:
- `System:`
- `User:`
- `Assistant:`

Why this is useful:
- Teaches the model conversation format
- Teaches role-conditioned behavior
- Gives predictable structure for learning

It then splits data:
- `data/train.txt` (for learning)
- `data/test.txt` (for held-out evaluation)
- `data/test_prompts.txt` (for inference prompts)

Why split train/test:
- Train loss alone can mislead (overfitting)
- Test loss checks if model generalizes to unseen examples

---

## 4) Tokenization

### 4.1 What is a token?

A token is a chunk of text (word piece / subword / symbol) mapped to an integer id.
Example idea:
- "Assistant" -> 1234
- ":" -> 25
- " test" -> 902

### 4.2 Why tokenization exists

Neural networks process numbers, not raw strings. Tokenization converts text to numeric form.

### 4.3 In this project

You use GPT-2 tokenizer via `tiktoken`.
- Pros: fast, stable, standard subword behavior
- Result: text -> list of token ids -> tensor

---

## 5) Model architecture: why this is Transformer-based

Your model has:
- Token embedding layer
- Positional embedding layer
- Stacked Transformer encoder blocks
- Linear output head to vocabulary logits

This is Transformer-style, not a plain feed-forward NN.

### 5.1 Components and why

1. Token embeddings
- Converts token IDs to dense vectors
- Why: ids are arbitrary integers; embeddings learn semantic geometry

2. Positional embeddings
- Add position information (token order)
- Why: self-attention is order-agnostic by default

3. Transformer layers
- Self-attention + feed-forward sublayers
- Why: attention lets every token condition on earlier context effectively

4. LM head (`Linear`)
- Maps hidden state -> logits over full vocabulary
- Why: model must produce probability for every possible next token

### 5.2 Causal mask (critical)

You added a **causal attention mask**.

Why needed:
- During training, position `t` must not see future tokens `> t`
- Without mask, model can "cheat" by reading answers ahead
- With mask, training matches real generation behavior

This is what makes it GPT-like autoregressive modeling.

---

## 6) Training objective (next-token prediction)

Given sequence tokens:
- Input `x`: `[t0, t1, t2, ... t(n-1)]`
- Target `y`: `[t1, t2, t3, ... t(n)]`

The model predicts each next token.

Loss function: **cross-entropy** over vocabulary.

Why cross-entropy:
- Standard for classification-like token prediction
- Penalizes wrong probability distributions

---

## 7) Batch sampling and context length

### 7.1 Context length

`context_length` is how many previous tokens model can attend to.

Why important:
- Larger context = potentially better reasoning over long text
- But memory/compute cost grows significantly

### 7.2 Random windows from token stream

Training samples random chunks from train/test token tensors.

Why random windows:
- Efficient stochastic training
- Better coverage of dataset across steps

---

## 8) Optimization

Optimizer used: `AdamW`

Why AdamW:
- Robust default for Transformer training
- Handles noisy gradients well
- Decoupled weight decay works better than classic Adam regularization in many setups

Learning rate (`lr`) controls update size:
- Too high: unstable/diverges
- Too low: slow learning

---

## 9) Evaluation during training

You periodically compute:
- `train_loss`
- `test_loss`

Why both:
- Train loss: how well model fits seen data
- Test loss: how well model generalizes

What to watch:
- If train keeps improving but test worsens -> overfitting
- If both flat high -> underfitting or poor setup

---

## 10) Checkpoint saving/loading

After training, checkpoint stores:
- Model weights (`state_dict`)
- Hyperparameters (heads/layers/embed/context)
- Tokenizer name and dataset metadata
- Optimizer state + step for resume
- Cumulative wall-clock training time (`total_training_seconds`) across resumed runs

Why save metadata too:
- Inference must recreate exact architecture
- Shape mismatch happens if config differs

`inference.py` loads checkpoint and rebuilds model before generation.

Current checkpoint flow:
- `tiny_llm_checkpoint_latest.pt` for periodic resume saves
- `tiny_llm_checkpoint_best.pt` for best validation checkpoint
- `tiny_llm_checkpoint.pt` used for serving/inference
- `tiny_llm_checkpoint_final.pt` saved at end of training run

Safe stop and resume:
- Press `Ctrl+C` during training to stop.
- Training catches the interrupt and saves `tiny_llm_checkpoint_latest.pt`.
- Run `python tiny_llm.py` again to resume from the saved step.
- Run with `RESUME_TRAINING=0 python tiny_llm.py` to start a fresh run.

---

## 11) Inference/generation

Generation loop:
1. Encode prompt -> tokens
2. Run model
3. Take next token (currently greedy `argmax`)
4. Append token
5. Repeat

Why it works:
- Model learned `P(next_token | previous_tokens)`
- Repeating this builds full text

Current decoding: greedy
- Deterministic
- Can be repetitive

Common improvements:
- Temperature
- Top-k/top-p sampling
- Repetition penalties

---

## 12) Why outputs still look repetitive

This is expected for tiny projects because:
- Dataset is synthetic and patterned
- Model is very small
- Training steps are limited
- Greedy decoding reduces diversity

This does **not** mean the pipeline is wrong.
It means scale + decoding strategy + data richness are limited.

---

## 13) Mapping this tiny project to real LLM stacks

Your project vs production LLM systems:

1. Data
- Yours: generated synthetic conversations
- Real: huge web/code/books/chat corpora + filtering + dedup

2. Tokenization
- Yours: GPT-2 tokenizer
- Real: custom tokenizer choices and large-scale preprocessing

3. Model
- Yours: tiny Transformer
- Real: billions of parameters, many layers/heads, optimized kernels

4. Training
- Yours: single-machine basic loop
- Real: distributed training (data/model/pipeline parallel)

5. Post-training
- Yours: none
- Real: instruction tuning, preference optimization (e.g., RLHF/DPO), safety layers

6. Inference
- Yours: local greedy decode
- Real: optimized serving engines, batching, KV cache, quantization

---

## 14) Practical learning roadmap from here

If your goal is "understand LLM dev", do this order:

1. Understand current code deeply
- Read `prepare_dataset.py`, `tiny_llm.py`, `inference.py` line by line

2. Add sampling
- Implement temperature + top-k in `inference.py`
- Compare quality changes

3. Improve data realism
- Replace synthetic generator with real text/chat files
- Keep train/test split

4. Add validation metrics
- Perplexity from cross-entropy

5. Add logging
- Save losses per step to CSV
- Plot train vs test curve

6. Scale model gradually
- Increase context, layers, embed size slowly
- Track speed/memory changes

7. Learn advanced topics
- Mixed precision
- Gradient clipping
- Learning-rate schedules
- Weight tying
- Rotary embeddings
- KV caching

---

## 15) Commands cheat-sheet

Create dataset:
```bash
python prepare_dataset.py
```

Train model:
```bash
python tiny_llm.py
```

Run inference tests:
```bash
python inference.py
```

Run heuristic quality evaluation:
```bash
python eval_quality.py
```

Track quality trend over time (append + compare):
```bash
python eval_quality.py --compare-last --out-jsonl logs/quality_history.jsonl
```

Train on LMSYS Chat 1M:
```bash
python prepare_dataset_lmsys.py --dataset lmsys/lmsys-chat-1m --split train --max-samples 200000
python tiny_llm.py
```

One-command workflow (bash script):
```bash
./scripts/workflow.sh data
./scripts/workflow.sh audit
./scripts/workflow.sh train
./scripts/workflow.sh infer
./scripts/workflow.sh eval
./scripts/workflow.sh serve
```

Training also writes `logs/train_eval_history.csv` so you can track `test_loss` and `test_perplexity` over long runs.

One script to download and prepare all conversational datasets:
```bash
HF_TOKEN=hf_xxx ./scripts/prepare_all_datasets.sh
```

Mix an additional dataset during data generation:
```bash
EXTRA_DATASET='HuggingFaceH4/ultrachat_200k' EXTRA_SPLIT='train_sft' ./scripts/workflow.sh data
```

Or run all non-server steps:
```bash
./scripts/workflow.sh pipeline
```

---

## 16) Important conceptual glossary

- LLM: Large Language Model
- Token: integer id for text piece
- Vocabulary: all possible tokens
- Embedding: learned vector representation
- Context window: max tokens considered at once
- Logits: raw model scores before softmax
- Softmax: converts logits to probabilities
- Cross-entropy: prediction loss
- Autoregressive: predicts next token from previous ones
- Causal mask: blocks access to future tokens
- Overfitting: memorizing train data, weak generalization
- Checkpoint: saved model state + config

---

## 17) Short answer to your earlier question

"Is this Transformer-based or just a normal NN?"

It is Transformer-based.
Specifically: a small GPT-style autoregressive language model with causal masking.
