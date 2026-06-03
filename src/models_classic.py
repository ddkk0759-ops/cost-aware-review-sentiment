"""
Classical (non-sequence) model factories with cost-aware ``class_weight``.

All builders return an *unfitted* estimator already wired with the cost ratio
(class_weight or scale_pos_weight) so that training loss reflects the
business cost matrix.
"""

from __future__ import annotations

from typing import Optional

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from . import config
from .cost import class_weight_dict


def xgboost_training_params() -> dict:
    """
    XGBoost device / tree backend. GPU when config allows and package supports it.
    XGBoost 2.x: tree_method=hist + device=cuda; 1.x: gpu_hist + gpu_predictor.
    """
    if not config.classic_tree_use_cuda():
        return {"tree_method": "hist", "device": "cpu", "n_jobs": -1}
    try:
        import xgboost as xgb
        parts = xgb.__version__.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        major, minor = 1, 7
    if (major, minor) >= (2, 0):
        return {"tree_method": "hist", "device": "cuda", "n_jobs": 1}
    return {
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "gpu_id": 0,
        "n_jobs": 1,
    }


def lightgbm_training_params() -> dict:
    """
    LightGBM device.

    默认 **CPU**：pip 安装的 LightGBM 通常未带 ``-DUSE_CUDA=1``，设 ``device=cuda`` 会报
    ``CUDA Tree Learner was not enabled in this build``。

    若你使用自行编译的 GPU 版 LightGBM，设环境变量 ``LIGHTGBM_USE_CUDA=1``。
    """
    if not config.lightgbm_use_cuda():
        return {"device": "cpu", "n_jobs": -1, "verbose": -1}
    try:
        import lightgbm as lgb
        ver = tuple(int(x) for x in lgb.__version__.split(".")[:2])
    except Exception:
        ver = (3, 3)
    if ver >= (4, 0):
        return {"device": "cuda", "n_jobs": 1, "verbose": -1}
    return {
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "n_jobs": 1,
        "verbose": -1,
    }


def make_naive_bayes(params: Optional[dict] = None):
    """MultinomialNB does not support class_weight; we emulate it via
    ``sample_weight`` in the fit-call inside train_classic."""
    return MultinomialNB(**(params or {}))


def make_logreg(params: Optional[dict] = None, cost_aware: bool = True):
    p = dict(params or {})
    p.setdefault("max_iter", 1000)
    p.setdefault("solver", "liblinear")
    if cost_aware:
        p["class_weight"] = class_weight_dict()
    return LogisticRegression(**p)


def make_linear_svm(params: Optional[dict] = None, cost_aware: bool = True,
                    calibrate: bool = True):
    p = dict(params or {})
    p.setdefault("max_iter", 2000)
    p.setdefault("dual", "auto")
    if cost_aware:
        p["class_weight"] = class_weight_dict()
    base = LinearSVC(**p)
    if calibrate:
        return CalibratedClassifierCV(base, cv=3)
    return base


def make_lightgbm(params: Optional[dict] = None, cost_aware: bool = True):
    from lightgbm import LGBMClassifier
    p = dict(params or {})
    p.setdefault("n_estimators", 300)
    p.setdefault("learning_rate", 0.1)
    p.setdefault("num_leaves", 63)
    p.setdefault("random_state", config.SEED)
    if cost_aware:
        p["class_weight"] = class_weight_dict()
    # 覆盖 trial / 缓存里的 device、n_jobs，保证与当前 CLASSIC_TREE_DEVICE 一致
    p.update(lightgbm_training_params())
    return LGBMClassifier(**p)


def make_xgboost(params: Optional[dict] = None, cost_aware: bool = True):
    from xgboost import XGBClassifier
    p = dict(params or {})
    p.setdefault("n_estimators", 300)
    p.setdefault("learning_rate", 0.1)
    p.setdefault("max_depth", 6)
    p.setdefault("eval_metric", "logloss")
    p.setdefault("random_state", config.SEED)
    if cost_aware:
        # XGB uses scale_pos_weight = (#neg / #pos) * desired_cost_ratio.
        # For balanced training labels (50/50) the ratio is just COST_FN / COST_FP.
        p.setdefault("scale_pos_weight", config.COST_FN / config.COST_FP)
    p.update(xgboost_training_params())
    return XGBClassifier(**p)


CLASSIC_BUILDERS = {
    "NaiveBayes": make_naive_bayes,
    "LogisticRegression": make_logreg,
    "LinearSVM": make_linear_svm,
    "LightGBM": make_lightgbm,
    "XGBoost": make_xgboost,
}
