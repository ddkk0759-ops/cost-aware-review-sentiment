"""
Cost-matrix utilities (Task 1).

Cost convention:
    cost(true=1, pred=0) = COST_FN  (miss negative review)
    cost(true=0, pred=1) = COST_FP  (mis-flag positive review)
    cost(correct)        = 0
Reference: Elkan (2001), "The Foundations of Cost-Sensitive Learning".
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer

from . import config


def total_cost(
    y_true,
    y_pred,
    cost_fn: float = config.COST_FN,
    cost_fp: float = config.COST_FP,
) -> float:
    """Return business cost = cost_fn * #FN + cost_fp * #FP."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fp = cm[0, 1]
    fn = cm[1, 0]
    return float(cost_fn * fn + cost_fp * fp)


def cost_breakdown(y_true, y_pred,
                   cost_fn: float = config.COST_FN,
                   cost_fp: float = config.COST_FP) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "FN_cost": float(cost_fn * fn),
        "FP_cost": float(cost_fp * fp),
        "total_cost": float(cost_fn * fn + cost_fp * fp),
        "cost_per_sample": float((cost_fn * fn + cost_fp * fp) / max(len(y_true), 1)),
    }


def cost_scorer(cost_fn: float = config.COST_FN, cost_fp: float = config.COST_FP):
    """sklearn-compatible scorer that *maximises* the negative total cost
    (so Optuna / GridSearch using `greater_is_better=True` will minimise cost)."""

    def _score(y_true, y_pred):
        return -total_cost(y_true, y_pred, cost_fn=cost_fn, cost_fp=cost_fp)

    return make_scorer(_score, greater_is_better=True)


def find_optimal_threshold(
    y_true,
    proba_pos,
    cost_fn: float = config.COST_FN,
    cost_fp: float = config.COST_FP,
    grid: np.ndarray | None = None,
) -> Tuple[float, float]:
    """
    Search the threshold τ that minimises total_cost on (y_true, proba_pos).

    proba_pos: array (N,) of P(class=1) — i.e. probability of negative review.
    Returns (best_threshold, best_cost).
    """
    y_true = np.asarray(y_true)
    proba_pos = np.asarray(proba_pos)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_thr = 0.5
    best_cost = float("inf")
    for thr in grid:
        y_pred = (proba_pos >= thr).astype(int)
        c = total_cost(y_true, y_pred, cost_fn=cost_fn, cost_fp=cost_fp)
        if c < best_cost:
            best_cost = c
            best_thr = float(thr)
    return best_thr, best_cost


def class_weight_dict(cost_fn: float = config.COST_FN,
                      cost_fp: float = config.COST_FP) -> dict:
    """sklearn ``class_weight`` baked from the cost ratio."""
    return {0: cost_fp, 1: cost_fn}


def torch_class_weight_tensor(cost_fn: float = config.COST_FN,
                              cost_fp: float = config.COST_FP):
    """Tensor for ``nn.CrossEntropyLoss(weight=...)`` — index 0 = class 0."""
    import torch
    return torch.tensor([cost_fp, cost_fn], dtype=torch.float32)
