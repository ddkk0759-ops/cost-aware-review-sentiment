"""
Train classical models with Optuna (dual objective: f1_neg and total_cost).

The two objectives are searched **independently** so we can show whether
optimising directly on cost actually beats f1-tuning + class_weight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import config
from .cost import class_weight_dict, cost_scorer
from .models_classic import (
    lightgbm_training_params,
    make_lightgbm,
    make_linear_svm,
    make_logreg,
    make_naive_bayes,
    make_xgboost,
    xgboost_training_params,
)


def _scoring_for(objective: str):
    if objective == "f1":
        return "f1"
    if objective == "cost":
        return cost_scorer()
    raise ValueError(objective)


def _direction_for(objective: str) -> str:
    return "maximize"  # f1 maximised; cost_scorer returns -cost so also maximise


def _suggest_logreg(trial):
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "solver": "liblinear",
        "max_iter": 1000,
    }


def _suggest_svm(trial):
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "max_iter": 2000,
        "dual": "auto",
    }


def _suggest_lgbm(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5, log=True),
        "random_state": config.SEED,
    }
    p.update(lightgbm_training_params())
    return p


def _suggest_xgb(trial):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5, log=True),
        "eval_metric": "logloss",
        "random_state": config.SEED,
    }
    p.update(xgboost_training_params())
    return p


_SUGGESTERS = {
    "LogisticRegression": _suggest_logreg,
    "LinearSVM": _suggest_svm,
    "LightGBM": _suggest_lgbm,
    "XGBoost": _suggest_xgb,
}


def _make_for_trial(name: str, params: dict, cost_aware: bool):
    if name == "LogisticRegression":
        return make_logreg(params, cost_aware=cost_aware)
    if name == "LinearSVM":
        # No calibration during the inner CV — saves time.
        return make_linear_svm(params, cost_aware=cost_aware, calibrate=False)
    if name == "LightGBM":
        return make_lightgbm(params, cost_aware=cost_aware)
    if name == "XGBoost":
        return make_xgboost(params, cost_aware=cost_aware)
    raise ValueError(name)


def _make_final(name: str, params: dict):
    if name == "LogisticRegression":
        return make_logreg(params, cost_aware=True)
    if name == "LinearSVM":
        return make_linear_svm(params, cost_aware=True, calibrate=True)
    if name == "LightGBM":
        return make_lightgbm(params, cost_aware=True)
    if name == "XGBoost":
        return make_xgboost(params, cost_aware=True)
    if name == "NaiveBayes":
        return make_naive_bayes()
    raise ValueError(name)


def tune_one(
    name: str,
    X,
    y,
    objective: str = "f1",
    n_trials: int = config.OPTUNA_TRIALS,
    cache_path: Optional[Path] = None,
) -> dict:
    """Run Optuna for one model on one objective."""
    if cache_path is not None and cache_path.exists():
        return joblib.load(cache_path)

    import optuna
    from optuna.samplers import TPESampler

    suggest = _SUGGESTERS[name]
    scoring = _scoring_for(objective)
    skf = StratifiedKFold(n_splits=config.OPTUNA_CV_FOLDS,
                          shuffle=True, random_state=config.SEED)

    def obj(trial):
        params = suggest(trial)
        clf = _make_for_trial(name, params, cost_aware=True)
        return cross_val_score(clf, X, y, scoring=scoring, cv=skf, n_jobs=1).mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=_direction_for(objective),
                                sampler=TPESampler(seed=config.SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best["best_value"] = study.best_value
    best["objective"] = objective
    if cache_path is not None:
        joblib.dump(best, cache_path)
    return best


def fit_classic_model(name: str, params: dict, X_train, y_train):
    """Refit on the full training set, NB uses sample_weight for cost-awareness."""
    if name == "NaiveBayes":
        # MultinomialNB has no class_weight — emulate via sample_weight.
        cw = class_weight_dict()
        sw = np.where(np.asarray(y_train) == 1, cw[1], cw[0])
        clf = make_naive_bayes(params)
        clf.fit(X_train, y_train, sample_weight=sw)
        return clf
    clf = _make_final(name, params)
    clf.fit(X_train, y_train)
    return clf


def predict_proba_pos(clf, X) -> np.ndarray:
    """Robust probability extraction.  Fallbacks for SVC without calibration."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        from scipy.special import expit
        return expit(clf.decision_function(X))
    return clf.predict(X).astype(float)


def train_all_classic(
    X_train,
    y_train,
    X_test,
    y_test,
    objective: str = "f1",
    n_trials: int = config.OPTUNA_TRIALS,
    cache_dir: Optional[Path] = None,
    skip_optuna_for: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """
    Train all 5 classic models (NB has no Optuna).

    Returns dict[name] -> {model, best_params, train_proba, test_proba,
                            test_pred}.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    skip_optuna_for = set(skip_optuna_for or ["NaiveBayes"])
    out: Dict[str, dict] = {}
    for name in config.CLASSIC_MODEL_NAMES:
        print(f"\n--- training {name} (objective={objective}) ---")
        if name in skip_optuna_for:
            params = {}
        else:
            cache = (cache_dir / f"{name}__{objective}.joblib") if cache_dir else None
            params = tune_one(name, X_train, y_train, objective=objective,
                              n_trials=n_trials, cache_path=cache)
            params.pop("best_value", None)
            params.pop("objective", None)
            print(f"  best params: {params}")
        clf = fit_classic_model(name, params, X_train, y_train)
        proba_train = predict_proba_pos(clf, X_train)
        proba_test = predict_proba_pos(clf, X_test)
        pred_test = (proba_test >= 0.5).astype(int)
        out[name] = {
            "model": clf,
            "best_params": params,
            "train_proba": proba_train,
            "test_proba": proba_test,
            "test_pred": pred_test,
        }
        if cache_dir is not None:
            joblib.dump(clf, cache_dir / f"{name}__{objective}.model.joblib")
    return out
