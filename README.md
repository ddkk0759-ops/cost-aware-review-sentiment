# Cost-Sensitive Sentiment Analysis of Amazon Reviews

**Multi-model stacking ensemble for negative-review early warning, with end-to-end cost sensitivity and LLM-assisted error attribution.**

> 面向业务代价的亚马逊评论情感分析：多模型集成与错误归因。
> 将差评定义为业务正类，在「类别权重 → 加权损失 → 超参目标 → 集成元学习 → 推理阈值」全链路统一注入 5:1 代价信号。

---

## Overview

In e-commerce review monitoring, classification errors are **asymmetric**: missing a negative
review (a false negative, FN) delays crisis intervention, whereas mis-flagging a positive
review (a false positive, FP) only adds review-overhead. This project treats the **negative
review as the positive class** and minimises a business cost `C_total = 5·FN + 1·FP` instead
of plain accuracy.

The pipeline trains **9 heterogeneous base learners** across two feature channels and fuses
them with a **cost-aware stacking ensemble**, then runs an **LLM-assisted error analysis** to
separate genuine model errors from label noise.

### Key features

- **Dual-channel features** — a sparse, interpretable TF-IDF channel and a dense Word2Vec
  sequence channel.
- **9 base learners** — Naive Bayes, Logistic Regression, Linear SVM, LightGBM, XGBoost
  (classic) + MeanPooling, BiLSTM, Transformer, TextCNN (sequence).
- **End-to-end cost consistency** — the 5:1 cost signal is injected at class weights, loss
  functions, the Optuna search objective, the stacking meta-learner, and the decision threshold.
- **Stratified 3-fold OOF stacking** — leak-free meta-features fused by a cost-weighted
  logistic-regression meta-learner.
- **LLM-assisted error attribution** — GPT-4o-mini re-judges disputed misclassifications to
  estimate the label-noise rate, validated against a human-labelled subset.

---

## Headline results

Full Amazon split (3.6M train / 400k test), negative review = positive class,
`C_FN : C_FP = 5 : 1`.

| Model                         | Accuracy | P(neg) | R(neg) | F1(neg) |  C_total |
| ----------------------------- | :------: | :----: | :----: | :-----: | -------: |
| **Stacking (9 models, cost-aware)** | **0.9012** | 0.8476 | **0.9783** | **0.9083** | **56,880** |
| Stacking (sequence only)      |  0.9004  | 0.8471 | 0.9771 | 0.9075  |   58,160 |
| Soft Voting (9 models)        |  0.8736  | 0.8086 | 0.9789 | 0.8856  |   67,440 |
| LSTM                          |  0.8869  | 0.8316 | 0.9702 | 0.8956  |   69,080 |
| Linear SVM                    |  0.8903  | 0.8856 | 0.8964 | 0.8910  |  126,760 |
| Naive Bayes                   |  0.6978  | 0.6246 | 0.9918 | 0.7665  |  127,440 |

The cost-aware stacking ensemble achieves the lowest business cost, while the high-accuracy
Linear SVM still incurs ~2.2× the cost due to its low negative-recall — illustrating that
**high accuracy does not imply low business cost**.

LLM error attribution on 812 disputed misclassifications: **71.4%** (95% CI 68.2%–74.5%)
agree with the original star-derived label, suggesting systematic label noise.

---

## Repository structure

```
.
├── src/                          # core library
│   ├── config.py                 # global config (5:1 cost ratio, seeds, hyper-params)
│   ├── data.py                   # data loading / stratified sampling / split
│   ├── stopwords.py              # 3-layer stopword system (NLTK + ratio + domain)
│   ├── preprocess_classic.py     # classic-channel preprocessing
│   ├── preprocess_seq.py         # sequence-channel preprocessing (W2V token map)
│   ├── vectorizer.py             # TF-IDF + Optuna
│   ├── w2v.py                    # Word2Vec embedding loading
│   ├── cost.py                   # cost-matrix utilities & threshold search
│   ├── models_classic.py         # NB / LR / SVM / LightGBM / XGBoost factory
│   ├── train_classic.py          # dual-objective Optuna (F1_neg + total cost)
│   ├── models_seq.py             # MeanPooling / LSTM / Transformer / TextCNN
│   ├── train_seq.py              # weighted cross-entropy training
│   ├── ensemble.py               # 3-fold OOF stacking + soft voting
│   ├── evaluate.py               # unified P/R/F1 + cost reporting
│   ├── error_analysis.py         # high-confidence / disputed / linguistic analysis
│   ├── llm_judge.py              # LLM-assisted error attribution
│   ├── llm_providers/            # stub / openai / anthropic / ollama backends
│   └── viz.py                    # plotting
├── outputs/
│   ├── figures/                  # paper figures (.png)
│   ├── logs/                     # summary metric tables (.csv)
│   └── llm_judgments/            # LLM judgment summaries
├── sentiment_analysis.ipynb      # end-to-end demo notebook (results inline)
├── requirements.txt
├── LICENSE
└── README.md
```

> Note: trained model binaries (`outputs/models/`), large per-sample error dumps, and the raw
> dataset are intentionally **git-ignored** — they are regenerable by running the pipeline.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt

# one-off NLTK resources
python -m nltk.downloader stopwords vader_lexicon punkt punkt_tab \
    averaged_perceptron_tagger_eng wordnet omw-1.4
```

## Data

This project uses the FastText Amazon review polarity corpus (3.6M train / 400k test).
Download `train.ft.txt` and `test.ft.txt` and place them in the repository root
(they are git-ignored). See the FastText / Kaggle "Amazon Reviews for Sentiment Analysis"
dataset.

## Quick start

```bash
jupyter lab sentiment_analysis.ipynb
```

Run the notebook top-to-bottom; it walks through preprocessing, the 9 base learners, the
cost-aware stacking ensemble, threshold optimisation, and the LLM error analysis.

### Data-scale toggle

By default the pipeline runs on a 2% stratified sample for fast iteration. Edit `src/config.py`:

```python
USE_FULL_DATA = False   # set True for the full 3.6M / 400k run
DATA_FRACTION = 0.02    # sample fraction when USE_FULL_DATA is False
```

### LLM backend

The error-analysis step defaults to a VADER-based `stub` backend (no network / key needed).
To use a real API:

```bash
export LLM_BACKEND=openai          # or anthropic / ollama
export OPENAI_API_KEY=sk-...
```

No API keys are stored in the repository; all credentials are read from environment variables.

---

## Method at a glance

1. **Preprocessing** — lowercase, regex tokenisation, and a 3-layer stopword system
   (NLTK + e-commerce words → data-driven ratio words → domain words via MI / chi-square / VADER).
2. **Dual-channel features** — TF-IDF (1–2 gram, 5000-dim) and Word2Vec (300-dim, max_len 150).
3. **9 base learners** — 5 classic on TF-IDF, 4 neural on Word2Vec; all carry a 5:1 class weight.
4. **Cost-sensitive optimisation** — Optuna minimises `C_total`; decision threshold τ is tuned
   on a held-out validation split.
5. **Stacking ensemble** — stratified 3-fold OOF meta-features fused by a cost-weighted LR meta-learner.
6. **Error attribution** — an LLM re-judges disputed errors to estimate the label-noise rate.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{amazon_cost_sensitive_sentiment,
  title  = {Cost-Sensitive Sentiment Analysis of Amazon Reviews:
            Multi-Model Stacking Ensemble and Error Attribution},
  author = {Your Name},
  year   = {2025},
  note   = {https://github.com/your-username/your-repo}
}
```

## License

Released under the [MIT License](LICENSE).
