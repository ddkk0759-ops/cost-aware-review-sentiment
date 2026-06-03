"""Unified evaluation reporter — accuracy / P/R/F1 / total cost / threshold."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from . import config
from .cost import cost_breakdown, find_optimal_threshold, total_cost


def evaluate_predictions(
    y_true,
    y_pred,
    name: str = "model",
    proba_pos: Optional[np.ndarray] = None,
    verbose: bool = True,
    decision_threshold: Optional[float] = None,
) -> dict:
    """Compute metrics for a single (y_true, y_pred) pair.

    If ``decision_threshold`` is set (with ``proba_pos``), the reported
    ``best_threshold`` / ``total_cost@best_thr`` / ``*_@best_thr`` columns
    reflect that fixed threshold (e.g. Stacking τ from validation hold-out)
    instead of re-searching an oracle threshold on ``y_true``.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_neg = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    p_neg = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    r_neg = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    cb = cost_breakdown(y_true, y_pred)

    row = {
        "model": name,
        "accuracy": acc,
        "precision_neg": p_neg,
        "recall_neg": r_neg,
        "f1_neg": f1_neg,
        **cb,
    }

    if proba_pos is not None:
        proba_pos = np.asarray(proba_pos)
        if decision_threshold is not None:
            thr = float(decision_threshold)
            y_pred_thr = (proba_pos >= thr).astype(int)
            row["best_threshold"] = thr
            row["total_cost@best_thr"] = float(total_cost(y_true, y_pred_thr))
            row["f1_neg@best_thr"] = f1_score(y_true, y_pred_thr, pos_label=1, zero_division=0)
            row["recall_neg@best_thr"] = recall_score(y_true, y_pred_thr, pos_label=1, zero_division=0)
        else:
            thr, cost_thr = find_optimal_threshold(y_true, proba_pos)
            y_pred_thr = (proba_pos >= thr).astype(int)
            row["best_threshold"] = thr
            row["total_cost@best_thr"] = cost_thr
            row["f1_neg@best_thr"] = f1_score(y_true, y_pred_thr, pos_label=1, zero_division=0)
            row["recall_neg@best_thr"] = recall_score(y_true, y_pred_thr, pos_label=1, zero_division=0)

    if verbose:
        print(f"\n=== {name} ===")
        print(f"Accuracy:          {acc:.4f}")
        print(f"Precision (neg):   {p_neg:.4f}")
        print(f"Recall    (neg):   {r_neg:.4f}")
        print(f"F1        (neg):   {f1_neg:.4f}")
        print(f"FN={cb['FN']}  FP={cb['FP']}  total_cost={cb['total_cost']:.0f}")
        if proba_pos is not None:
            print(f"best_threshold={row['best_threshold']:.2f}  "
                  f"total_cost@best={row['total_cost@best_thr']:.0f}  "
                  f"recall_neg@best={row['recall_neg@best_thr']:.4f}")
        print(classification_report(
            y_true, y_pred,
            target_names=[config.LABEL_NAMES[0], config.LABEL_NAMES[1]],
            zero_division=0,
        ))
    return row


def compare_models(rows, sort_by: str = "total_cost") -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if sort_by in df.columns:
        ascending = sort_by in ("total_cost", "total_cost@best_thr")
        df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    return df
