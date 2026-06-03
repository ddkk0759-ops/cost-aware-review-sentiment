"""Error-analysis utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config


_NEG_WORDS = re.compile(r"\b(not|never|no|none|nothing|nobody|nowhere|cannot|n't)\b", re.I)
_ADVERS = re.compile(r"\b(but|however|although|though|yet|whereas)\b", re.I)
_EXCLAM = re.compile(r"!")
_QUESTION = re.compile(r"\?")
_ALL_CAPS = re.compile(r"\b[A-Z]{3,}\b")


def linguistic_features(text: str) -> dict:
    return {
        "negation": bool(_NEG_WORDS.search(text)),
        "adversative": bool(_ADVERS.search(text)),
        "exclam": bool(_EXCLAM.search(text)),
        "question": bool(_QUESTION.search(text)),
        "all_caps": bool(_ALL_CAPS.search(text)),
        "len": len(text.split()),
    }


def high_confidence_errors(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba_pos: np.ndarray,
    threshold: float = config.HIGH_CONF_THRESHOLD,
    top_n: int | None = 20,
) -> pd.DataFrame:
    """Return high-confidence misclassifications, sorted by descending
    confidence.  Confidence here = max(P(0), P(1)).

    ``top_n``: max rows to return; ``None`` or ``0`` keeps **all** rows with
    confidence ≥ ``threshold`` (useful for full exports).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    proba_pos = np.asarray(proba_pos)
    confidence = np.maximum(proba_pos, 1 - proba_pos)
    err_mask = y_pred != y_true
    df = pd.DataFrame({
        "idx": np.where(err_mask)[0],
        "text": [texts[i] for i in np.where(err_mask)[0]],
        "true": y_true[err_mask],
        "pred": y_pred[err_mask],
        "proba_pos": proba_pos[err_mask],
        "confidence": confidence[err_mask],
    })
    df["true_name"] = df["true"].map(config.LABEL_NAMES)
    df["pred_name"] = df["pred"].map(config.LABEL_NAMES)
    hc = df[df["confidence"] >= threshold].sort_values("confidence", ascending=False)
    # top_n is None or 0 → keep all rows above threshold
    return hc.head(top_n) if top_n else hc


def misclassified_dataframe(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba_pos: np.ndarray,
) -> pd.DataFrame:
    """Every index where ``y_pred != y_true`` with text and probabilities."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    proba_pos = np.asarray(proba_pos)
    err = y_pred != y_true
    idx = np.where(err)[0]
    if len(idx) == 0:
        return pd.DataFrame(
            columns=["idx", "text", "true", "pred", "proba_pos", "confidence",
                     "true_name", "pred_name"])
    conf = np.maximum(proba_pos[idx], 1 - proba_pos[idx])
    out = pd.DataFrame({
        "idx": idx.astype(int),
        "text": [texts[int(i)] for i in idx],
        "true": y_true[idx].astype(int),
        "pred": y_pred[idx].astype(int),
        "proba_pos": proba_pos[idx].astype(float),
        "confidence": conf.astype(float),
    })
    out["true_name"] = out["true"].map(config.LABEL_NAMES)
    out["pred_name"] = out["pred"].map(config.LABEL_NAMES)
    return out


def all_wrong_dataframe(
    texts: List[str],
    y_true: np.ndarray,
    preds_dict: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Rows where every model in ``preds_dict`` disagrees with ``y_true``."""
    diss = model_disagreement(preds_dict, y_true)
    mask = diss["all_wrong_mask"]
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=["idx", "text", "true", "true_name", *[
            f"pred_{n}" for n in preds_dict]])
    rows = []
    for i in idx:
        i = int(i)
        row = {
            "idx": i,
            "text": texts[i],
            "true": int(y_true[i]),
            "true_name": config.LABEL_NAMES[int(y_true[i])],
        }
        for n, p in preds_dict.items():
            row[f"pred_{n}"] = int(np.asarray(p)[i])
        rows.append(row)
    return pd.DataFrame(rows)


def majority_wrong_dataframe(
    texts: List[str],
    y_true: np.ndarray,
    preds_dict: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Rows where at most half of the models are correct vs ``y_true``."""
    diss = model_disagreement(preds_dict, y_true)
    mask = diss["majority_wrong_mask"]
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return pd.DataFrame(columns=["idx", "text", "true", "true_name",
                                      "n_models_correct"])
    cc = diss["correct_count"]
    rows = []
    for i in idx:
        i = int(i)
        rows.append({
            "idx": i,
            "text": texts[i],
            "true": int(y_true[i]),
            "true_name": config.LABEL_NAMES[int(y_true[i])],
            "n_models_correct": int(cc[i]),
        })
    return pd.DataFrame(rows)


def export_error_sample_corpus(
    texts: List[str],
    y_true: np.ndarray,
    preds_dict: Dict[str, np.ndarray],
    stack_pred: np.ndarray,
    stack_proba: np.ndarray,
    log_dir: Optional[Path] = None,
    high_conf_top_n: Optional[int] = None,
) -> Dict[str, Path]:
    """
    Persist **full** error-related tables under ``log_dir`` (default
    ``config.LOG_DIR``), in addition to any small demo slices elsewhere.

    Writes
    ------
    - ``stacking_errors_all.csv`` — every Stacking misclassification.
    - ``stacking_high_conf_errors_all.csv`` — every wrong prediction with
      confidence ≥ ``HIGH_CONF_THRESHOLD`` (not truncated).
    - ``stacking_high_conf_errors_top200.csv`` — optional top-200 by confidence
      for quick browsing (if ``high_conf_top_n`` is not None).
    - ``dispute_all_models_wrong.csv`` — all_wrong bucket (full text).
    - ``dispute_majority_wrong.csv`` — majority-wrong bucket.
    """
    log_dir = Path(log_dir or config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    out: Dict[str, Path] = {}

    df_stack = misclassified_dataframe(texts, y_true, stack_pred, stack_proba)
    p = log_dir / "stacking_errors_all.csv"
    df_stack.to_csv(p, index=False, encoding="utf-8")
    out["stacking_errors_all"] = p

    df_hc_all = high_confidence_errors(
        texts, y_true, stack_pred, stack_proba,
        threshold=config.HIGH_CONF_THRESHOLD,
        top_n=None,
    )
    p = log_dir / "stacking_high_conf_errors_all.csv"
    df_hc_all.to_csv(p, index=False, encoding="utf-8")
    out["stacking_high_conf_errors_all"] = p

    if high_conf_top_n is not None and high_conf_top_n > 0:
        p = log_dir / f"stacking_high_conf_errors_top{high_conf_top_n}.csv"
        df_hc_all.head(high_conf_top_n).to_csv(p, index=False, encoding="utf-8")
        out["stacking_high_conf_errors_top"] = p

    df_aw = all_wrong_dataframe(texts, y_true, preds_dict)
    p = log_dir / "dispute_all_models_wrong.csv"
    df_aw.to_csv(p, index=False, encoding="utf-8")
    out["dispute_all_models_wrong"] = p

    df_mw = majority_wrong_dataframe(texts, y_true, preds_dict)
    p = log_dir / "dispute_majority_wrong.csv"
    df_mw.to_csv(p, index=False, encoding="utf-8")
    out["dispute_majority_wrong"] = p

    print("[error-export] wrote:")
    print(f"  stacking_errors_all.csv: {len(df_stack)} rows -> {out['stacking_errors_all']}")
    print(f"  stacking_high_conf_errors_all.csv: {len(df_hc_all)} rows -> {out['stacking_high_conf_errors_all']}")
    if "stacking_high_conf_errors_top" in out:
        print(f"  {out['stacking_high_conf_errors_top'].name}: {min(high_conf_top_n or 0, len(df_hc_all))} rows -> {out['stacking_high_conf_errors_top']}")
    print(f"  dispute_all_models_wrong.csv: {len(df_aw)} rows -> {out['dispute_all_models_wrong']}")
    print(f"  dispute_majority_wrong.csv: {len(df_mw)} rows -> {out['dispute_majority_wrong']}")
    return out


def model_disagreement(
    preds_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
) -> dict:
    """Per-sample agreement statistics across base models."""
    y_true = np.asarray(y_true)
    correct_mask = {n: p == y_true for n, p in preds_dict.items()}
    correct_count = sum(correct_mask[n].astype(int) for n in correct_mask)
    return {
        "correct_mask": correct_mask,
        "correct_count": correct_count,
        "all_correct_mask": correct_count == len(preds_dict),
        "all_wrong_mask": correct_count == 0,
        "majority_wrong_mask": correct_count <= len(preds_dict) // 2,
    }


def error_linguistic_stats(
    texts: List[str],
    err_mask: np.ndarray,
) -> pd.DataFrame:
    err_idx = np.where(err_mask)[0]
    correct_idx = np.where(~err_mask)[0]
    err_feats = pd.DataFrame([linguistic_features(texts[i]) for i in err_idx])
    correct_feats = pd.DataFrame([linguistic_features(texts[i]) for i in correct_idx])

    bool_cols = ["negation", "adversative", "exclam", "question", "all_caps"]
    stats = pd.DataFrame({
        "错判样本(%)": (err_feats[bool_cols].mean() * 100).round(2),
        "正确样本(%)": (correct_feats[bool_cols].mean() * 100).round(2),
    })
    stats["差值(错-对)"] = (stats["错判样本(%)"] - stats["正确样本(%)"]).round(2)
    return stats, err_feats, correct_feats
