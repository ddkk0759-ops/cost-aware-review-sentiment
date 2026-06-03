"""
Three-layer stopword system.

1. Base layer:   NLTK English stopwords + CUSTOM_STOPWORDS (domain-aware
                 manual extension, see config.CUSTOM_STOPWORDS).
2. Ratio layer:  data-driven — words that are very frequent and have a near-1
                 positive/negative ratio are treated as neutral noise.
3. Domain layer: words that score in the bottom on both Mutual Information
                 and chi-squared against the label, *and* have |VADER
                 compound| < threshold.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Set, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif

from . import config


def _ensure_nltk():
    import nltk
    for resource, path in [
        ("stopwords", "corpora/stopwords"),
        ("vader_lexicon", "sentiment/vader_lexicon"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def base_stopwords() -> Set[str]:
    """NLTK English + manual domain-specific extension."""
    _ensure_nltk()
    from nltk.corpus import stopwords
    return set(stopwords.words("english")) | set(config.CUSTOM_STOPWORDS)


def compute_ratio_stopwords(
    docs: Iterable[str],
    labels: Iterable[int],
    min_df: int = config.RATIO_MIN_DF,
    ratio_low: float = config.RATIO_LOW,
    ratio_high: float = config.RATIO_HIGH,
    top_n: int = config.RATIO_TOP_N,
    cache_path=None,
) -> Tuple[Set[str], dict, dict]:
    """
    Return (ratio_stopwords, ratios, total_freqs).

    ratio = (df_pos + 1) / (df_neg + 1)
    Words with ratio in [ratio_low, ratio_high] AND in the top_n most frequent
    among such candidates are considered "neutral high-frequency".
    """
    if cache_path is not None and cache_path.exists():
        return joblib.load(cache_path)

    docs = list(docs)
    labels = np.asarray(list(labels))
    vec = CountVectorizer(min_df=min_df, binary=True)
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()

    pos_mask = labels == 0
    neg_mask = labels == 1
    df_pos = np.asarray(X[pos_mask].sum(axis=0)).ravel()
    df_neg = np.asarray(X[neg_mask].sum(axis=0)).ravel()
    ratios = (df_pos + 1) / (df_neg + 1)
    total_freq = df_pos + df_neg

    candidates = [
        (vocab[i], total_freq[i], ratios[i])
        for i in range(len(vocab))
        if ratio_low <= ratios[i] <= ratio_high
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = candidates[:top_n]

    ratio_stop = {w for w, _, _ in selected}
    ratios_dict = {w: float(r) for w, _, r in selected}
    total_freq_dict = {w: int(f) for w, f, _ in selected}

    if cache_path is not None:
        joblib.dump((ratio_stop, ratios_dict, total_freq_dict), cache_path)
    return ratio_stop, ratios_dict, total_freq_dict


def compute_domain_stopwords(
    docs: Iterable[str],
    labels: Iterable[int],
    min_df: int = config.DOMAIN_MIN_DF,
    mi_bottom_n: int = config.DOMAIN_MI_BOTTOM_N,
    chi2_bottom_n: int = config.DOMAIN_CHI2_BOTTOM_N,
    vader_thr: float = config.DOMAIN_VADER_NEUTRAL_THRESHOLD,
    cache_path=None,
) -> Set[str]:
    """
    Three-way intersection: bottom MI ∩ bottom chi² ∩ |VADER compound| < thr.
    """
    if cache_path is not None and cache_path.exists():
        return joblib.load(cache_path)

    _ensure_nltk()
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    docs = list(docs)
    labels = np.asarray(list(labels))
    vec = CountVectorizer(min_df=min_df, binary=True)
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()

    mi = mutual_info_classif(X, labels, discrete_features=True, random_state=config.SEED)
    chi, _ = chi2(X, labels)

    bottom_mi = set(np.argsort(mi)[:mi_bottom_n].tolist())
    bottom_chi = set(np.argsort(chi)[:chi2_bottom_n].tolist())
    intersect = bottom_mi & bottom_chi
    candidates = [vocab[i] for i in intersect]

    sia = SentimentIntensityAnalyzer()
    domain_stop = {
        w for w in candidates if abs(sia.polarity_scores(w)["compound"]) < vader_thr
    }

    if cache_path is not None:
        joblib.dump(domain_stop, cache_path)
    return domain_stop


def union_stopwords(*sets: Set[str]) -> Set[str]:
    out: Set[str] = set()
    for s in sets:
        out |= set(s)
    return out
