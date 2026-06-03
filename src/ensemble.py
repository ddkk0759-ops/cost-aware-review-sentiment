"""
Stacking ensemble (Task 2).

Strategy
--------
1. Generate **out-of-fold (OOF)** predictions for every base model on the
   training set with K_FOLD splits — this is the meta-learner's training
   signal and is leak-free by construction.
2. Each base model is also re-trained on the *full* training set and used
   to predict on the test set.  These test predictions are the meta-learner's
   test features.
3. Meta features = concat(P(class=1)) across all M base models, so the meta
   matrix has shape (N, M).  We use a single column per model rather than
   2 columns because the two columns are perfectly anti-correlated.
4. Meta learner: ``LogisticRegression(class_weight={0:1, 1:5})`` so the
   ensemble inherits the cost-sensitive objective from Task 1.
5. Decision threshold τ on ``P(差评)``: ``fit_stacking_tune_threshold`` holds
   out a stratified slice of ``meta_train``, fits a provisional meta-LR,
   runs ``find_optimal_threshold`` on that slice, refits on full
   ``meta_train``, and applies ``(proba >= τ)`` at test time (see
   ``config.STACK_THRESHOLD_VAL_FRACTION``).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config
from .cost import class_weight_dict, find_optimal_threshold


def kfold_oof_classic(
    name: str,
    refit_fn: Callable,            # (params, X_tr, y_tr) -> fitted clf
    proba_fn: Callable,            # (clf, X) -> proba_pos (N,)
    params: dict,
    X_train,                       # sparse / dense feature matrix
    y_train,
    X_test,
    n_splits: int = config.K_FOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """OOF training-set probas + test probas (test averaged over folds)."""
    y_train = np.asarray(y_train)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    oof = np.zeros(len(y_train), dtype=np.float32)
    test_acc = np.zeros(X_test.shape[0], dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_train)), y_train), 1):
        print(f"  [{name}] fold {fold}/{n_splits} (train={len(tr_idx)} val={len(va_idx)})")
        clf = refit_fn(params, X_train[tr_idx], y_train[tr_idx])
        oof[va_idx] = proba_fn(clf, X_train[va_idx])
        test_acc += proba_fn(clf, X_test) / n_splits
    return oof, test_acc


def kfold_oof_seq(
    name: str,
    train_one_fn: Callable,        # creates model, trains it, returns dict
    indices_train: np.ndarray,
    y_train: List[int],
    indices_test: np.ndarray,
    y_test: List[int],
    n_splits: int = config.K_FOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """OOF probas for sequence models.  ``train_one_fn`` must accept
    (tr_idx_arr, val_idx_arr, full_test_idx_arr) and return val/test
    probabilities."""
    y_train = np.asarray(y_train)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    oof = np.zeros(len(y_train), dtype=np.float32)
    test_acc = np.zeros(len(indices_test), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_train)), y_train), 1):
        print(f"  [{name}] fold {fold}/{n_splits} (train={len(tr_idx)} val={len(va_idx)})")
        val_proba, test_proba = train_one_fn(tr_idx, va_idx)
        oof[va_idx] = val_proba
        test_acc += test_proba / n_splits
    return oof, test_acc


def soft_voting(prob_matrix: np.ndarray) -> np.ndarray:
    """``prob_matrix`` shape (M, N) — averaged probability of class 1."""
    return prob_matrix.mean(axis=0)


def build_meta(probs_train: Dict[str, np.ndarray], names: List[str]) -> np.ndarray:
    """Stack a name-ordered list of OOF probas into the meta-matrix."""
    return np.stack([probs_train[n] for n in names], axis=1)


def fit_stacking(
    meta_train: np.ndarray,
    y_train,
    cost_aware: bool = True,
    C: float = 1.0,
) -> LogisticRegression:
    cw = class_weight_dict() if cost_aware else None
    clf = LogisticRegression(C=C, max_iter=2000, class_weight=cw)
    clf.fit(meta_train, y_train)
    return clf


def fit_stacking_tune_threshold(
    meta_train: np.ndarray,
    y_train,
    meta_test: np.ndarray,
    cost_aware: bool = True,
    C: float = 1.0,
    val_fraction: float | None = None,
    random_state: int | None = None,
) -> Tuple[LogisticRegression, float, float, np.ndarray]:
    """Fit stacking meta-LR, calibrate decision threshold τ on a stratified
    hold-out slice of *training* meta rows, refit on full meta_train, return
    ``(clf, tau, total_cost_on_calib_val, test_proba_pos)``.

    Predictions should use ``(test_proba_pos >= tau).astype(int)`` instead of
    a fixed 0.5 cut-off.  ``total_cost_on_calib_val`` is the cost achieved by
    τ on the held-out slice (under ``find_optimal_threshold``).
    """
    y = np.asarray(y_train)
    meta_train = np.asarray(meta_train, dtype=np.float32)
    vf = float(config.STACK_THRESHOLD_VAL_FRACTION if val_fraction is None else val_fraction)
    rs = int(config.SEED if random_state is None else random_state)
    n = len(y)
    if n < 20 or vf <= 0.0 or vf >= 0.45:
        clf = fit_stacking(meta_train, y, cost_aware=cost_aware, C=C)
        proba_te = clf.predict_proba(meta_test)[:, 1].astype(np.float32)
        return clf, 0.5, float("nan"), proba_te
    idx = np.arange(n)
    tr_idx, va_idx = train_test_split(
        idx, test_size=vf, stratify=y, random_state=rs
    )
    clf_part = fit_stacking(meta_train[tr_idx], y[tr_idx], cost_aware=cost_aware, C=C)
    va_proba = clf_part.predict_proba(meta_train[va_idx])[:, 1]
    tau, val_cost = find_optimal_threshold(y[va_idx], va_proba)
    clf_final = fit_stacking(meta_train, y, cost_aware=cost_aware, C=C)
    proba_te = clf_final.predict_proba(meta_test)[:, 1].astype(np.float32)
    return clf_final, float(tau), float(val_cost), proba_te
