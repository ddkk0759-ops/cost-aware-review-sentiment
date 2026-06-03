"""
LLM-based label-error analysis (Task 3).

Workflow
--------
1. Pick disputed samples in two buckets:
     (A) "all wrong"  — every base model AND the stacking ensemble disagree
                        with the ground-truth label.
     (B) "high-conf wrong" — stacking ensemble is wrong with confidence ≥ τ
                              (suspected mislabel candidates).
2. Call the configured LLM provider on each sample (sequential by default;
   tunable batch_size for rate-limited APIs).
3. Persist:
     - raw_responses.jsonl   (one record per sample, full request/response)
     - summary.csv           (per-sample agreement / suspected mislabel flag)
     - agreement_stats.csv   (aggregate rates per bucket)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config
from .llm_providers import LLMProvider, get_provider


# ────────────────────────── sample selection ──────────────────────────

def select_all_wrong(
    y_true: np.ndarray,
    base_preds: dict,                      # {model_name: pred_array}
    ensemble_pred: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Indices where every base model AND (optionally) ensemble are wrong."""
    y_true = np.asarray(y_true)
    mask = np.ones(len(y_true), dtype=bool)
    for p in base_preds.values():
        mask &= (np.asarray(p) != y_true)
    if ensemble_pred is not None:
        mask &= (np.asarray(ensemble_pred) != y_true)
    return np.where(mask)[0]


def select_high_conf_wrong(
    y_true: np.ndarray,
    pred: np.ndarray,
    proba_pos: np.ndarray,
    threshold: float = config.HIGH_CONF_THRESHOLD,
    top_n: Optional[int] = None,
) -> np.ndarray:
    """Indices where prediction is wrong with confidence ≥ threshold,
    sorted by descending confidence."""
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    proba_pos = np.asarray(proba_pos)
    confidence = np.maximum(proba_pos, 1 - proba_pos)
    err = pred != y_true
    cand = np.where(err & (confidence >= threshold))[0]
    cand = cand[np.argsort(-confidence[cand])]
    if top_n is not None:
        cand = cand[:top_n]
    return cand


# ──────────────────────────── batch judging ─────────────────────────────

def judge_samples(
    indices: Sequence[int],
    texts: List[str],
    y_true,
    model_pred: np.ndarray,
    proba_pos: np.ndarray,
    bucket: str,
    provider: Optional[LLMProvider] = None,
    sleep_s: float = 0.0,
    log_every: int = 5,
) -> List[dict]:
    """Run the LLM provider on the given indices and return rich records."""
    provider = provider or get_provider()
    print(f"[llm-judge] backend={provider.backend} model={provider.model} "
          f"bucket={bucket} N={len(indices)}")

    records: List[dict] = []
    y_true = np.asarray(y_true)
    proba_pos = np.asarray(proba_pos)
    for i, idx in enumerate(indices, 1):
        text = texts[int(idx)]
        true_label = int(y_true[idx])
        m_pred = int(model_pred[idx])
        proba = float(proba_pos[idx])
        confidence = max(proba, 1 - proba)

        result = provider.predict(text)

        record = {
            "bucket": bucket,
            "idx": int(idx),
            "text": text,
            "true_label": true_label,
            "true_label_name": config.LABEL_NAMES[true_label],
            "model_pred": m_pred,
            "model_pred_name": config.LABEL_NAMES[m_pred],
            "model_proba_pos": proba,
            "model_confidence": confidence,
            "llm_label": result.label,
            "llm_label_name": (config.LABEL_NAMES.get(result.label, "n/a")
                               if result.label in (0, 1) else "n/a"),
            "llm_confidence": result.confidence,
            "llm_rationale": result.rationale,
            "llm_backend": result.backend,
            "llm_model": result.model,
            "llm_error": result.error,
            "raw_response": result.raw_response,
            "agree_with_model":
                (result.label == m_pred) if result.label in (0, 1) else None,
            "agree_with_truth":
                (result.label == true_label) if result.label in (0, 1) else None,
            "suspected_label_error":
                (result.label == m_pred and result.label != true_label
                 if result.label in (0, 1) else None),
        }
        records.append(record)
        if log_every and i % log_every == 0:
            print(f"  [{bucket}] {i}/{len(indices)} done")
        if sleep_s > 0:
            time.sleep(sleep_s)
    return records


# ───────────────────── persistence + aggregate stats ────────────────────

def save_records(records: List[dict],
                 jsonl_path: Path,
                 csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "raw_response"}
                       for r in records])
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  saved -> {jsonl_path}  +  {csv_path}  ({len(records)} rows)")


def aggregate_stats(records: Iterable[dict]) -> pd.DataFrame:
    """Per-bucket aggregate: agreement rates and mislabel rate."""
    df = pd.DataFrame(records)
    if df.empty:
        return df
    out_rows = []
    for bucket, sub in df.groupby("bucket"):
        valid = sub[sub["llm_label"].isin([0, 1])]
        n = len(sub)
        n_valid = len(valid)
        out_rows.append({
            "bucket": bucket,
            "n_total": n,
            "n_valid_response": n_valid,
            "valid_rate": round(n_valid / max(n, 1), 4),
            "agree_with_model_rate": round(valid["agree_with_model"].mean(), 4)
                                     if n_valid else None,
            "agree_with_truth_rate": round(valid["agree_with_truth"].mean(), 4)
                                     if n_valid else None,
            "suspected_label_error_rate": round(
                valid["suspected_label_error"].mean(), 4)
                                          if n_valid else None,
            "n_suspected_label_error": int(valid["suspected_label_error"].sum())
                                       if n_valid else 0,
        })
    return pd.DataFrame(out_rows)
