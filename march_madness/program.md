# Autoresearch: March Madness Prediction Optimization

## Core Thesis

Predicting NCAA tournament outcomes is a well-studied probabilistic forecasting
problem. The evaluation metric (Brier score) rewards well-calibrated predictions.
Our goal: systematically iterate on the prediction model to minimize Brier score
on historical tournament data, then submit the best model for the 2026 competition.

## Your Task

You are an autonomous AI research agent. Your job is to iteratively modify
`predict.py` to improve Brier score on the backtest (2022-2025 tournaments).

### The Experimental Loop

1. **Read** `predict.py` and `experiments.tsv` to understand the current state
2. **Hypothesize** a specific model improvement
3. **Implement** the change in `predict.py`
4. **Run** `python predict.py --data-dir $DATA_DIR --backtest` to evaluate
5. **Record** the result in `experiments.tsv`
6. **Decide**: keep the change if Brier score improves, OR revert if it regresses
7. **Repeat** — try the next hypothesis

### What You Can Modify

- **`predict.py`** — This is the only file you edit. It contains the Elo system,
  feature engineering, team stats, and the prediction ensemble.

### What You Cannot Modify

- **The data files** — Fixed Kaggle competition dataset
- **The evaluation metric** — Brier score on tournament games
- **The submission format** — ID,Pred CSV matching SampleSubmissionStage2.csv

### Constraints

- The primary metric is **Brier score** (lower is better, baseline ~0.25)
- Current baseline: Brier=0.1742, Accuracy=75.4% on 2022-2025 tournaments
- All backtest evaluations must use the same seasons (2022-2025)
- Changes must not break the submission file generation
- Record all results in `experiments.tsv`

## Research Directions

Explore these directions, roughly ordered by expected impact:

### 1. Elo System Tuning

**Ideas to try:**
- **K-factor optimization**: Tune K for regular season vs tournament games
- **Home advantage calibration**: Optimize the home court Elo bonus
- **Season reversion rate**: Tune how much ratings revert toward the mean each season
- **Margin of victory scaling**: Adjust the MOV multiplier function
- **Recency weighting**: Weight recent games more heavily within a season
- **Separate tournament Elo**: Different K-factor for tournament games (high stakes = more informative)

### 2. Feature Engineering

**Ideas to try:**
- **Win/loss record features**: Win percentage, conference win %, last N games
- **Strength of schedule**: Average opponent Elo rating
- **Momentum features**: Recent form (last 10 games performance)
- **Conference strength**: Average Elo/stats of conference peers
- **Tournament experience**: Historical tournament appearances and performance
- **Coaching features**: Coach tournament win rate from MTeamCoaches.csv

### 3. Advanced Statistics

**Ideas to try:**
- **Four Factors**: eFG%, TO%, OR%, FT rate (Dean Oliver's four factors)
- **Tempo-adjusted stats**: Points per possession rather than raw points
- **Defensive metrics**: Opponent shooting percentages, blocks per possession
- **Three-point dependency**: Reliance on 3-point shooting (variance indicator)
- **Free throw rate**: FTA/FGA as a proxy for aggressiveness/discipline

### 4. Ensemble Weights & Calibration

**Ideas to try:**
- **Optimize ensemble weights**: Grid search or gradient descent on log-odds weights
- **Platt scaling**: Fit a logistic regression on backtest predictions
- **Isotonic regression**: Non-parametric calibration of raw predictions
- **Seed-Elo interaction**: Adjust predictions based on seed differential × Elo differential
- **Bayesian model averaging**: Weight component models by their individual Brier scores

### 5. Model Architecture

**Ideas to try:**
- **Logistic regression on features**: Replace ensemble with trained logistic regression
- **Gradient boosted trees**: Use historical tournament games as training data
- **Bradley-Terry model**: Pairwise comparison model fit on season results
- **Glicko-2 ratings**: Extension of Elo with rating deviation (uncertainty)
- **Historical seed matchup priors**: P(seed X beats seed Y) from all historical data

### 6. Ordinal Rankings Integration

**Ideas to try:**
- **System selection**: Use only the most predictive ranking systems (e.g., POM, SAG, MOR)
- **Weighted average by system quality**: Weight systems by their historical accuracy
- **Rank differential features**: Use rank gap as a feature rather than converting to probability
- **Consensus ranking**: Use median rank instead of mean for robustness

## Experiment Tracking

Maintain an `experiments.tsv` file with these columns:

```
experiment_id	description	brier_score	brier_delta	accuracy	notes	kept
```

Where:
- `experiment_id`: Sequential number (001, 002, ...)
- `description`: Brief description of what was tried
- `brier_score`: Average Brier score on 2022-2025 backtest
- `brier_delta`: Change from previous best (negative = improvement)
- `accuracy`: Prediction accuracy percentage
- `notes`: Observations, per-season breakdown
- `kept`: yes/no — whether this change was kept

## Important Principles

1. **One change at a time**. Never modify multiple things simultaneously.
   You need to know exactly what caused each result.

2. **Baseline first**. The current model (Elo + stats + seeds + ordinals) is
   the baseline at Brier=0.1742.

3. **Calibration matters more than accuracy**. A well-calibrated 0.65 prediction
   is better than an overconfident 0.85 that's wrong 25% of the time.

4. **Respect the data**. With only ~134 tournament games per year, overfitting
   is a real risk. Prefer simple, robust improvements.

5. **Simplicity wins**. A simple modification that shows a clear signal is
   more valuable than a complex one with marginal effect.

6. **Document everything**. Write clear notes about what you tried, what you
   expected, and what actually happened.

## Getting Started

```bash
DATA_DIR=/root/.cache/kagglehub/competitions/march-machine-learning-mania-2026

# Run baseline backtest
python predict.py --data-dir $DATA_DIR --backtest

# Generate submission
python predict.py --data-dir $DATA_DIR --output submission.csv
```
