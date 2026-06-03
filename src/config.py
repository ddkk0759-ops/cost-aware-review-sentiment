"""
Global configuration.

Cost matrix design (task 1)
---------------------------
Business priority: missing a negative review (FN, true=1, pred=0) is far more
costly than mis-flagging a positive review (FP, true=0, pred=1) — the former
loses early-warning ability for brand crises while the latter only adds a small
review-overhead.

We adopt the canonical 5 : 1 cost ratio used in the cost-sensitive learning
literature (Elkan, 2001, "The Foundations of Cost-Sensitive Learning",
IJCAI-01).  This ratio is widely re-used in fraud / churn / sentiment-risk
papers as a well-balanced default.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------- paths ----------
ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "train.ft.txt"
TEST_PATH = ROOT / "test.ft.txt"

OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
LOG_DIR = OUTPUTS / "logs"
MODEL_DIR = OUTPUTS / "models"
LLM_DIR = OUTPUTS / "llm_judgments"
for _p in (FIG_DIR, LOG_DIR, MODEL_DIR, LLM_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------- reproducibility ----------
SEED = 42

# ---------- data scale toggle ----------
# 设为 True 即使用全量 (3.6M / 400k)；False 时按 DATA_FRACTION 分层抽样
USE_FULL_DATA = False
DATA_FRACTION = 0.02

# ---------- label semantics ----------
# 1 = negative review (positive class for the metric, what we worry about missing)
# 0 = positive review
LABEL_NEG = 1
LABEL_POS = 0
LABEL_NAMES = {0: "好评(0)", 1: "差评(1)"}

# ---------- cost matrix (Elkan 2001) ----------
COST_FN = 5.0   # 漏判差评（true=1, pred=0）的代价
COST_FP = 1.0   # 误伤好评（true=0, pred=1）的代价
COST_TP = 0.0
COST_TN = 0.0
COST_RATIO_DESC = "C_FN : C_FP = 5 : 1  (Elkan 2001, IJCAI-01)"

# ---------- train / val split ----------
VAL_FRACTION = 0.10
# Stacking meta-LR：分层留出一份训练行，仅用其 calibrate 决策阈值 τ（find_optimal_threshold）
STACK_THRESHOLD_VAL_FRACTION = float(
    os.environ.get("STACK_THRESHOLD_VAL_FRACTION", str(VAL_FRACTION))
)

# ---------- sequence-model hyper-params ----------
MAX_LEN = 150
EMBED_DIM = 300
# 大显存可调大（环境变量 BATCH_SIZE 覆盖）；32GB 级可试 512～1024
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "512"))
EPOCHS = 5
# 序列模型防过拟合（小样本 / 强容量模型时尤其有用）
SEQ_WEIGHT_DECAY = float(os.environ.get("SEQ_WEIGHT_DECAY", "1e-4"))
SEQ_LABEL_SMOOTHING = float(os.environ.get("SEQ_LABEL_SMOOTHING", "0.05"))
# 验证 loss 连续若干 epoch 不降则停训；0 = 不提前结束，但仍会加载「验证 loss 最低」那一轮的权重
SEQ_EARLY_STOPPING_PATIENCE = int(os.environ.get("SEQ_EARLY_STOPPING_PATIENCE", "2"))
# MeanPooling / LSTM / TextCNN 中的 Dropout；Transformer 见下条
SEQ_DROPOUT = float(os.environ.get("SEQ_DROPOUT", "0.4"))
SEQ_TRANSFORMER_DROPOUT = float(os.environ.get("SEQ_TRANSFORMER_DROPOUT", "0.15"))
# DataLoader：多进程 + pin 内存，减轻 GPU 饥饿
TORCH_NUM_WORKERS = int(os.environ.get("TORCH_NUM_WORKERS", "4"))
TORCH_PIN_MEMORY = os.environ.get("TORCH_PIN_MEMORY", "1") not in ("0", "false", "False")
TORCH_CUDNN_BENCHMARK = os.environ.get("TORCH_CUDNN_BENCHMARK", "1") not in ("0", "false", "False")
# CUDA 上混合精度（序列模型）
SEQ_USE_AMP = os.environ.get("SEQ_USE_AMP", "1") not in ("0", "false", "False")
W2V_NAME = "word2vec-google-news-300"
# gensim 官方 CDN 不稳定时，把已下载的 GoogleNews-vectors-negative300.bin.gz（或 .kv）路径设到环境变量
W2V_LOCAL_PATH = os.environ.get("W2V_LOCAL_PATH", "").strip()

# ---------- TF-IDF defaults ----------
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 1
TFIDF_SUBLINEAR_TF = True

# ---------- Optuna ----------
OPTUNA_TRIALS = 15           # 默认 15；快速调试可在 notebook 改 5
OPTUNA_CV_FOLDS = 5
OPTUNA_TFIDF_SAMPLE_SIZE = 50_000

# ---------- classic tree models (LightGBM / XGBoost) ----------
# auto: 若 PyTorch 能检测到 CUDA 则 XGBoost 走 GPU；LightGBM 见下条
CLASSIC_TREE_DEVICE = os.environ.get("CLASSIC_TREE_DEVICE", "auto").strip().lower()
# PyPI 上 LightGBM 多为 CPU  wheel（无 CUDA Tree Learner）。若你本地是自行编译的 GPU 版，设：
#   export LIGHTGBM_USE_CUDA=1
LIGHTGBM_USE_CUDA = os.environ.get("LIGHTGBM_USE_CUDA", "0").strip().lower() in (
    "1", "true", "yes",
)

# ---------- stopwords ----------
RATIO_LOW = 0.7
RATIO_HIGH = 1.3
RATIO_MIN_DF = 10
RATIO_TOP_N = 1000

DOMAIN_MIN_DF = 20
DOMAIN_MI_BOTTOM_N = 500
DOMAIN_CHI2_BOTTOM_N = 500
DOMAIN_VADER_NEUTRAL_THRESHOLD = 0.05

CUSTOM_STOPWORDS = {
    "one", "get", "would", "could", "really", "much", "even", "first",
    "thing", "make", "made", "way", "still", "also", "another", "many",
    "lot", "want", "go", "going", "back", "come", "say", "said", "think",
    "know", "see", "look", "find", "work", "buy", "use", "used",
    "book", "movie", "read", "cd", "album", "game", "dvd", "bought",
    "purchase", "version", "piece", "unit", "series", "page", "copy",
    "minute", "hour", "day", "week", "month", "year",
    "try", "tried", "started", "went", "came", "took",
}

# ---------- ensemble ----------
K_FOLD = 3 # 5-fold OOF for stacking; 序列模型耗时大，2% 数据下若太慢可在 notebook 改 3

# ---------- error analysis ----------
HIGH_CONF_THRESHOLD = 0.90

# ---------- LLM ----------
# 后端选择：stub / openai / anthropic / ollama
LLM_BACKEND = os.environ.get("LLM_BACKEND", "stub")
LLM_DEMO_N_HIGH_CONF = 20      # 高置信度错判 demo 取 N 条
LLM_DEMO_N_ALL_WRONG = 20      # 集成全错 demo 取 N 条
LLM_OPENAI_MODEL = "gpt-4o-mini"
LLM_ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"
LLM_OLLAMA_MODEL = "llama3.1:8b"

# ---------- model display names ----------
SEQ_MODEL_NAMES = ["MeanPooling", "LSTM", "Transformer", "TextCNN"]
CLASSIC_MODEL_NAMES = ["NaiveBayes", "LogisticRegression", "LinearSVM",
                      "LightGBM", "XGBoost"]
ALL_BASE_MODEL_NAMES = SEQ_MODEL_NAMES + CLASSIC_MODEL_NAMES


def classic_tree_use_cuda() -> bool:
    """Whether XGBoost (and optionally LightGBM) may use GPU."""
    if CLASSIC_TREE_DEVICE == "cpu":
        return False
    if CLASSIC_TREE_DEVICE == "cuda":
        return True
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def lightgbm_use_cuda() -> bool:
    """LightGBM GPU only if CUDA build + explicit opt-in (see LIGHTGBM_USE_CUDA)."""
    return classic_tree_use_cuda() and LIGHTGBM_USE_CUDA
