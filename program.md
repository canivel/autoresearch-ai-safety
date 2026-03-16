# Autoresearch-Safety: Building Safe LLMs from the Ground Up

## Core Thesis

Standard LLMs are misaligned **by construction**. The dominant paradigm — pretraining
a transformer on next-token prediction over internet text, then patching alignment
via RLHF/DPO — treats safety as an afterthought. This creates fundamental tensions:

1. **The Mimicry Problem**: Next-token prediction optimizes the model to reproduce
   whatever patterns exist in the data, including deception, manipulation, and
   confidently-stated falsehoods. The model learns to *sound* right, not *be* right.

2. **The Superposition Problem**: Dense transformer representations pack many
   features into overlapping directions, making internal states opaque. You cannot
   steer what you cannot see.

3. **The Objective Mismatch**: The training loss (cross-entropy on next token) has
   no direct connection to helpfulness, harmlessness, or honesty. Alignment is
   bolted on after the model's worldview is already formed.

4. **The Overconfidence Problem**: Standard softmax outputs are poorly calibrated.
   The model cannot distinguish what it knows from what it's guessing, making
   hallucination a structural inevitability rather than a bug to fix.

**Our hypothesis**: Many of these problems can be addressed — or at least
significantly mitigated — by modifying the model architecture and training
objective *during pretraining*, rather than relying entirely on post-hoc alignment.

## Your Task

You are an autonomous AI research agent. Your job is to iteratively modify
`train.py` to explore whether safety-relevant properties can be built into a
language model from the ground up, while maintaining competitive language
modeling performance (measured by val_bpb).

### The Experimental Loop

1. **Read** this file and `train.py` to understand the current state
2. **Hypothesize** a specific architectural or objective modification
3. **Implement** the change in `train.py`
4. **Run** `uv run train.py` (5-minute training budget, fixed)
5. **Record** the result: val_bpb + any safety-relevant metrics
6. **Decide**: keep the change if it improves or maintains val_bpb without
   significant regression, OR revert if it hurts performance
7. **Repeat** — try the next hypothesis

### What You Can Modify

- **`train.py`** — This is the only file you edit. It contains the model
  architecture, optimizer, training loop, and hyperparameters.

### What You Cannot Modify

- **`prepare.py`** — Fixed data pipeline, tokenizer, and evaluation.
  `evaluate_bpb()` is the ground-truth metric.
- **Dependencies** — Do not add new pip packages.

### Constraints

- All experiments use the same 5-minute wall-clock training budget
- The primary metric is `val_bpb` (lower is better)
- Safety modifications must NOT catastrophically hurt val_bpb
- A safety modification that slightly hurts val_bpb (< 2% regression) but
  introduces measurably beneficial properties is still valuable
- Record all results in `experiments.tsv`

## Research Directions

Explore these directions, roughly ordered by expected impact:

### 1. Representation Structure (Architecture)

The hypothesis: if internal representations are more structured and
disentangled, the model becomes more interpretable and steerable.

**Ideas to try:**
- **Representation regularization**: Add a loss term that penalizes excessive
  cosine similarity between token representations (already scaffolded in
  train.py via `representation_reg`). Tune the coefficient.
- **Sparse activation patterns**: Replace ReLU^2 with TopK activation or
  mixture-of-experts routing to encourage sparser, more interpretable features.
- **Orthogonal initialization + constraints**: Initialize weight matrices to
  be orthogonal and add soft orthogonality constraints during training.
- **Dedicated feature dimensions**: Reserve a subset of the embedding dimensions
  for specific purposes (e.g., uncertainty, topic, style) with separate losses.

### 2. Uncertainty Awareness (Architecture + Objective)

The hypothesis: a model that knows what it doesn't know is fundamentally safer.

**Ideas to try:**
- **Uncertainty heads**: Dedicate some attention heads to estimating prediction
  uncertainty. Add an auxiliary loss that calibrates these heads.
- **Entropy regularization**: Add a term that rewards moderate prediction entropy
  (discouraging both uniform and spike distributions).
- **Evidential deep learning**: Replace softmax with a Dirichlet distribution
  output, giving the model a native notion of epistemic uncertainty.
- **Monte Carlo dropout at train time**: Train with structured dropout patterns
  that can later be used for uncertainty estimation.

### 3. Controllability (Architecture)

The hypothesis: explicit control dimensions make behavior more predictable.

**Ideas to try:**
- **Steering vectors built-in**: Add learnable "mode" embeddings that are
  concatenated or added to the residual stream, allowing external control.
- **Conditional layer normalization**: Make normalization parameters dependent
  on a control signal.
- **Gated residual streams**: Add learnable gates that can amplify or suppress
  specific information flows.

### 4. Training Objective Modifications

The hypothesis: auxiliary losses during pretraining can shape the model's
internal structure toward safety-relevant properties.

**Ideas to try:**
- **Consistency loss**: For overlapping context windows, penalize the model
  for making inconsistent predictions about the same tokens.
- **Backward prediction**: Add a small auxiliary head that predicts previous
  tokens, encouraging bidirectional understanding.
- **Contrastive representation learning**: Add a loss that pulls together
  representations of semantically similar tokens and pushes apart dissimilar ones.
- **Label smoothing**: Use label smoothing in the cross-entropy loss to
  discourage overconfident predictions.

### 5. Attention Pattern Analysis

The hypothesis: safer models have more interpretable attention patterns.

**Ideas to try:**
- **Attention entropy regularization**: Encourage attention heads to either
  attend broadly or sharply, not in between (bimodal distribution).
- **Head specialization loss**: Encourage different heads to attend to
  different positional patterns.
- **Sparse attention constraints**: Encourage attention patterns that are
  more human-interpretable.

## Experiment Tracking

Maintain an `experiments.tsv` file with these columns:

```
experiment_id	description	val_bpb	val_bpb_delta	safety_metric	notes	kept
```

Where:
- `experiment_id`: Sequential number (001, 002, ...)
- `description`: Brief description of what was tried
- `val_bpb`: The validation bits-per-byte
- `val_bpb_delta`: Change from baseline (negative = improvement)
- `safety_metric`: Any safety-relevant measurement (e.g., representation diversity, calibration)
- `notes`: Observations
- `kept`: yes/no — whether this change was kept for the next experiment

## Important Principles

1. **One change at a time**. Never modify multiple things simultaneously.
   You need to know exactly what caused each result.

2. **Baseline first**. Before any safety modifications, establish a clean
   baseline val_bpb with the unmodified architecture.

3. **Safety should not destroy capability**. The goal is not to build a model
   that refuses to do anything — it's to build a model whose *internal
   structure* makes it more amenable to safe behavior when capability is added.

4. **Measure what matters**. Whenever you add a safety modification, also add
   code to *measure* its effect (e.g., representation diversity score,
   calibration metrics, attention entropy).

5. **Simplicity wins**. A simple modification that shows a clear signal is
   more valuable than a complex one with marginal effect.

6. **Document everything**. Write clear notes in experiments.tsv about what
   you tried, what you expected, and what actually happened.

## Getting Started

```bash
# One-time setup
python prepare.py

# Run baseline experiment
uv run train.py

# Record baseline val_bpb in experiments.tsv
# Then begin iterating on safety modifications
```
