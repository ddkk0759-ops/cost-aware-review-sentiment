"""Plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from . import config
from .utils import setup_matplotlib_chinese

setup_matplotlib_chinese()


def _save(fig, name: str):
    out = config.FIG_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  fig -> {out}")


def plot_label_distribution(labels, name: str = "label_distribution.png"):
    counts = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar([config.LABEL_NAMES[i] for i in counts.index], counts.values,
           color=["#4C9BE8", "#E84C4C"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title("标签分布")
    _save(fig, name)
    plt.show()


def plot_length_hist(lengths, max_len: int, name: str = "length_distribution.png"):
    median_len = int(np.median(lengths))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(lengths, bins=50, color="#4C9BE8", edgecolor="white")
    ax.axvline(median_len, color="red", linestyle="--", label=f"median={median_len}")
    ax.axvline(max_len, color="orange", linestyle="--", label=f"MAX_LEN={max_len}")
    ax.set_xlabel("Token count")
    ax.set_ylabel("Frequency")
    ax.set_title("训练集句长分布")
    ax.legend()
    _save(fig, name)
    plt.show()


def plot_training_curves(histories: Dict[str, dict],
                         name: str = "training_curves.png"):
    colors = ["#4C9BE8", "#E84C4C", "#4CE86E", "#E8A64C", "#9B59B6", "#34495E"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, (k, h) in enumerate(histories.items()):
        c = colors[i % len(colors)]
        axes[0].plot(h["train_loss"], color=c, linestyle="--", alpha=0.6,
                     label=f"{k} (train)")
        axes[0].plot(h["val_loss"], color=c, label=f"{k} (val)")
        axes[1].plot(h["val_acc"], color=c, label=k)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    _save(fig, name)
    plt.show()


def plot_confusion_matrices(items, name: str = "confusion_matrices.png"):
    """items: iterable of (title, y_true, y_pred)."""
    items = list(items)
    n = len(items)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (title, y_true, y_pred) in zip(axes, items):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[config.LABEL_NAMES[0], config.LABEL_NAMES[1]],
                    yticklabels=[config.LABEL_NAMES[0], config.LABEL_NAMES[1]],
                    ax=ax, cbar=False)
        ax.set_title(title)
        ax.set_xlabel("预测")
        ax.set_ylabel("真实")
    for ax in axes[len(items):]:
        ax.axis("off")
    _save(fig, name)
    plt.show()


def plot_cost_comparison(df: pd.DataFrame, value_col: str = "total_cost",
                         name: str = "cost_comparison.png",
                         title: str = "各模型总代价对比 (越低越好)"):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    df_sorted = df.sort_values(value_col, ascending=True)
    bars = ax.bar(df_sorted["model"], df_sorted[value_col],
                  color="#9B59B6", edgecolor="white")
    for b, v in zip(bars, df_sorted[value_col]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(value_col)
    ax.set_title(title)
    plt.xticks(rotation=20)
    _save(fig, name)
    plt.show()


def plot_metric_comparison(df: pd.DataFrame, value_col: str = "accuracy",
                           name: str = "metric_comparison.png",
                           title: str | None = None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    df_sorted = df.sort_values(value_col, ascending=False)
    bars = ax.bar(df_sorted["model"], df_sorted[value_col],
                  color="#4C9BE8", edgecolor="white")
    for b, v in zip(bars, df_sorted[value_col]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(value_col)
    ax.set_title(title or f"{value_col} comparison")
    ax.set_ylim(max(0, df_sorted[value_col].min() - 0.02),
                min(1.0, df_sorted[value_col].max() + 0.02))
    plt.xticks(rotation=20)
    _save(fig, name)
    plt.show()


def plot_linguistic_stats(stats_df: pd.DataFrame,
                          err_lens: Sequence[int],
                          correct_lens: Sequence[int],
                          name: str = "error_linguistic_stats.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    stats_df[["错判样本(%)", "正确样本(%)"]].plot.bar(
        ax=axes[0], color=["#E84C4C", "#4C9BE8"]
    )
    axes[0].set_title("Linguistic features: errors vs correct")
    axes[0].set_ylabel("Occurrence (%)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].hist([err_lens, correct_lens], bins=40, label=["Errors", "Correct"],
                 color=["#E84C4C", "#4C9BE8"], alpha=0.7, density=True)
    axes[1].set_title("Length distribution: errors vs correct")
    axes[1].set_xlabel("# words")
    axes[1].set_ylabel("density")
    axes[1].legend()
    _save(fig, name)
    plt.show()
