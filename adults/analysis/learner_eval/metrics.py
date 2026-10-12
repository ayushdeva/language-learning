"""Scoring. Input: one row per predicted token with columns ex (task index), history (tasks seen before), y, p."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

EPS = 1e-6
HISTORY_BINS = [0, 10, 50, 200, np.inf]
HISTORY_LABELS = ["0-9", "10-49", "50-199", "200+"]


def log_loss(y, p) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def skill(y, p, base: float) -> float:
    """1 - LogLoss(model) / LogLoss(always predict the base rate). 0 = no better than the base rate, 1 = perfect."""
    return 1 - log_loss(y, p) / log_loss(y, np.full(len(y), base))


def auroc_within(df: pd.DataFrame) -> float:
    """AUROC inside each exercise, pooled over all (wrong, right) token pairs from the same exercise.
    Same learner, same moment, so learner ability cancels out: does the model know WHICH words they will miss?
    (Within-learner AUROC is not used: a learner's past error rate is mechanically anti-correlated with their
    remaining tokens, which unfairly penalises any feature built from their history.)"""
    df = df.assign(y=df.y.astype(np.int64))   # int8 labels overflow in n1 * (n1 + 1)
    r = df.groupby("ex").p.rank()
    g = df.assign(r=r * df.y).groupby("ex").agg(n=("y", "size"), n1=("y", "sum"), r1=("r", "sum"))
    g = g[(g.n1 > 0) & (g.n1 < g.n)]
    pairs = g.n1 * (g.n - g.n1)
    return float((g.r1 - g.n1 * (g.n1 + 1) / 2).sum() / pairs.sum())


def score(df: pd.DataFrame, base: float) -> dict:
    y, p = df.y.to_numpy(), df.p.to_numpy()
    return {
        "skill": skill(y, p, base),
        "log_loss": log_loss(y, p),
        "auroc": float(roc_auc_score(y, p)),
        "auroc_within_ex": auroc_within(df),
        "f1@0.5": float(f1_score(y, p >= 0.5, zero_division=0)),
    }


def skill_by_history(df: pd.DataFrame, base: float) -> dict:
    bins = pd.cut(df.history, HISTORY_BINS, right=False, labels=HISTORY_LABELS)
    return {str(b): skill(g.y.to_numpy(), g.p.to_numpy(), base) for b, g in df.groupby(bins, observed=True)}


def calibration(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    bins = pd.qcut(df.p, n_bins, duplicates="drop")
    return df.groupby(bins, observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"), tokens=("y", "size"))
