# Autoresearch-Safety: Building Safe LLMs from the Ground Up

**Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch)**

## The Problem

Current LLMs are built wrong from the start. The standard pipeline — pretrain on
next-token prediction, then bolt on alignment via RLHF/DPO — treats safety as a
patch rather than a foundation. This leads to:

- **Mimicry over truth**: The model learns to sound right, not be right
- **Opaque representations**: Dense superposition makes internal states uninterpretable
- **Objective mismatch**: Cross-entropy loss has zero connection to helpfulness or honesty
- **Structural overconfidence**: Softmax outputs are poorly calibrated by construction

## The Thesis

Many alignment problems are **architectural** — they emerge from how we build and
train models, not from insufficient post-hoc tuning. This project explores whether
we can build safety-relevant properties directly into the model architecture and
training objective during pretraining.

## How It Works

This project uses the [autoresearch](https://github.com/karpathy/autoresearch)
framework: an AI agent autonomously modifies `train.py`, runs 5-minute training
experiments on a single GPU, evaluates results, and iterates. The key difference
is that our agent optimizes for **both language modeling performance (val_bpb) and
safety-relevant metrics**.

### Research Directions

1. **Representation Structure** — Regularizers and architectures that encourage
   disentangled, interpretable internal representations
2. **Uncertainty Awareness** — Native uncertainty estimation built into the
   architecture (not bolted on after)
3. **Controllability** — Explicit steering dimensions that make behavior predictable
4. **Objective Shaping** — Auxiliary pretraining losses that encourage consistency,
   calibration, and structured representations
5. **Attention Interpretability** — Architectural choices that produce more
   human-readable attention patterns

### Project Structure

```
prepare.py      # Data pipeline, tokenizer, eval (DO NOT MODIFY)
train.py        # Model, optimizer, training loop (AGENT MODIFIES THIS)
program.md      # Research instructions for the AI agent
safety_eval.py  # Safety-oriented evaluation metrics
experiments.tsv # Experiment tracking log
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download data and train tokenizer
python prepare.py

# Run baseline training (5 minutes)
uv run train.py

# For autonomous research: point an AI agent at program.md
```

## Safety Metrics

Beyond standard val_bpb, we track:

- **Representation diversity**: Cosine similarity and effective rank of hidden states
- **Prediction calibration**: Expected calibration error (ECE), entropy, confidence
- **Attention interpretability**: Head specialization and pattern entropy

## Credits

- Training infrastructure: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- Optimizer: Muon + AdamW (from autoresearch)
- Thesis and safety research direction: this project
