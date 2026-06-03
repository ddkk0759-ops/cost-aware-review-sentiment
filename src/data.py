"""FastText-format data loader and stratified subsampling."""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from . import config


_LABEL_RE = re.compile(r"(__label__\d+)\s+(.*)")


def parse_file(path) -> Tuple[List[str], List[int]]:
    """
    Parse a FastText-format file:
        __label__1  ->  1 (negative review, business positive class)
        __label__2  ->  0 (positive review)
    """
    texts: List[str] = []
    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _LABEL_RE.match(line)
            if not m:
                continue
            label = 1 if m.group(1) == "__label__1" else 0
            texts.append(m.group(2))
            labels.append(label)
    return texts, labels


def stratified_subsample(
    texts: List[str],
    labels: List[int],
    fraction: float,
    seed: int = config.SEED,
) -> Tuple[List[str], List[int]]:
    if not (0 < fraction < 1):
        raise ValueError("fraction must be in (0, 1)")
    idx = np.arange(len(texts))
    keep_idx, _ = train_test_split(
        idx, train_size=fraction, random_state=seed, stratify=labels
    )
    keep_idx = np.sort(keep_idx)
    return [texts[i] for i in keep_idx], [labels[i] for i in keep_idx]


def load_train_test(
    use_full: bool | None = None,
    fraction: float | None = None,
    seed: int | None = None,
):
    """Load FastText files; honour the data scale toggle.

    Defaults are read from ``config`` on each call (not at import time), so
    changes after ``%autoreload`` or editing ``config.py`` take effect without
    restarting the kernel.
    """
    if use_full is None:
        use_full = config.USE_FULL_DATA
    if fraction is None:
        fraction = config.DATA_FRACTION
    if seed is None:
        seed = config.SEED
    train_texts_full, train_labels_full = parse_file(config.TRAIN_PATH)
    test_texts_full, test_labels_full = parse_file(config.TEST_PATH)

    if use_full:
        train_texts, train_labels = train_texts_full, train_labels_full
        test_texts, test_labels = test_texts_full, test_labels_full
    else:
        train_texts, train_labels = stratified_subsample(
            train_texts_full, train_labels_full, fraction=fraction, seed=seed
        )
        test_texts, test_labels = stratified_subsample(
            test_texts_full, test_labels_full, fraction=fraction, seed=seed
        )

    print(f"训练集大小: {len(train_texts):,} / 全量 {len(train_texts_full):,}")
    print(f"测试集大小: {len(test_texts):,} / 全量 {len(test_texts_full):,}")
    print(f"训练集标签分布: {Counter(train_labels)}")
    print(f"测试集标签分布: {Counter(test_labels)}")
    return train_texts, train_labels, test_texts, test_labels


def make_train_val_split(
    texts: List[str],
    labels: List[int],
    val_fraction: float = config.VAL_FRACTION,
    seed: int = config.SEED,
):
    """Single hold-out split for stacking meta training when not using OOF."""
    n = len(texts)
    train_idx, val_idx = train_test_split(
        np.arange(n),
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )
    sub_train = ([texts[i] for i in train_idx], [labels[i] for i in train_idx])
    val = ([texts[i] for i in val_idx], [labels[i] for i in val_idx])
    return sub_train, val, train_idx, val_idx
