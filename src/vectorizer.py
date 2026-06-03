"""TF-IDF vectoriser + Optuna search (cached)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import resample

from . import config


def build_tfidf(params: dict) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=params.get("max_features", config.TFIDF_MAX_FEATURES),
        ngram_range=(1, params.get("ngram_max", config.TFIDF_NGRAM_RANGE[1])),
        min_df=params.get("min_df", config.TFIDF_MIN_DF),
        sublinear_tf=params.get("sublinear_tf", config.TFIDF_SUBLINEAR_TF),
    )


def tune_tfidf(
    train_texts: List[str],
    train_labels: List[int],
    n_trials: int = config.OPTUNA_TRIALS,
    sample_size: int = config.OPTUNA_TFIDF_SAMPLE_SIZE,
    cache_path: Path | None = None,
) -> dict:
    """Optuna-search the TF-IDF hyperparams (LR + 5-fold f1_neg)."""
    if cache_path is not None and cache_path.exists():
        return joblib.load(cache_path)

    import optuna
    from optuna.samplers import TPESampler

    if len(train_texts) > sample_size:
        sub_texts, sub_labels = resample(
            train_texts, train_labels,
            replace=False, n_samples=sample_size,
            stratify=train_labels, random_state=config.SEED,
        )
    else:
        sub_texts, sub_labels = train_texts, train_labels

    sub_labels = np.asarray(sub_labels)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_features": trial.suggest_int("max_features", 3000, 20000, step=1000),
            "ngram_max": trial.suggest_int("ngram_max", 1, 3),
            "min_df": trial.suggest_int("min_df", 1, 5),
            "sublinear_tf": trial.suggest_categorical("sublinear_tf", [True, False]),
        }
        vec = build_tfidf(params)
        X = vec.fit_transform(sub_texts)
        clf = LogisticRegression(C=1.0, max_iter=500, n_jobs=1)
        skf = StratifiedKFold(n_splits=config.OPTUNA_CV_FOLDS,
                              shuffle=True, random_state=config.SEED)
        return cross_val_score(clf, X, sub_labels, scoring="f1",
                               cv=skf, n_jobs=1).mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=config.SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best["best_value"] = study.best_value
    if cache_path is not None:
        joblib.dump(best, cache_path)
    print(f"[tfidf-optuna] best={best}")
    return best
