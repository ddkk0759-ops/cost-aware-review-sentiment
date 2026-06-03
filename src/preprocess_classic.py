"""
Classical-pipeline text preprocessing (used by TF-IDF + sklearn models).

Steps:
    lower → strip URL/HTML/digits/punct → tokenize (NLTK) → stopword removal
    → WordNet lemmatize.
"""

from __future__ import annotations

import os
import re
import string
import zipfile
from functools import lru_cache
from typing import Iterable, List, Optional, Set

from . import config


_URL = re.compile(r"http\S+|www\S+|https\S+", re.I)
_HTML = re.compile(r"<.*?>")
_DIGIT = re.compile(r"\d+")
_PUNCT_TBL = str.maketrans("", "", string.punctuation)


def _ensure_nltk():
    """Download NLTK data if missing; drop corrupt zips and retry (fixes BadZipFile)."""
    import nltk

    def _remove_corrupt_zip(relative: str) -> None:
        for root in nltk.data.path:
            zpath = os.path.join(root, relative + ".zip")
            if not os.path.isfile(zpath):
                continue
            try:
                with zipfile.ZipFile(zpath, "r") as zf:
                    if zf.testzip() is not None:
                        raise zipfile.BadZipFile("corrupt member")
            except zipfile.BadZipFile:
                try:
                    os.remove(zpath)
                except OSError:
                    pass

    for resource, path in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        for _ in range(3):
            try:
                nltk.data.find(path)
                break
            except LookupError:
                nltk.download(resource, quiet=True)
            except zipfile.BadZipFile:
                _remove_corrupt_zip(path)
                nltk.download(resource, quiet=True)


_ensure_nltk()
from nltk.stem import WordNetLemmatizer  # noqa: E402
from nltk.tokenize import word_tokenize  # noqa: E402

_LEMMA = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = text.lower()
    text = _URL.sub(" ", text)
    text = _HTML.sub(" ", text)
    text = _DIGIT.sub(" ", text)
    text = text.translate(_PUNCT_TBL)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@lru_cache(maxsize=200_000)
def _lemmatize_token(tok: str) -> str:
    return _LEMMA.lemmatize(tok)


def preprocess(
    text: str,
    stopwords: Optional[Set[str]] = None,
    do_lemma: bool = True,
) -> str:
    """Return a single space-joined string ready for TF-IDF."""
    text = clean_text(text)
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    if do_lemma:
        tokens = [_lemmatize_token(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]
    return " ".join(tokens)


def preprocess_corpus(
    texts: Iterable[str],
    stopwords: Optional[Set[str]] = None,
    do_lemma: bool = True,
    desc: str = "preprocess",
) -> List[str]:
    out = []
    texts = list(texts)
    n = len(texts)
    step = max(n // 10, 1)
    for i, t in enumerate(texts):
        out.append(preprocess(t, stopwords=stopwords, do_lemma=do_lemma))
        if (i + 1) % step == 0:
            print(f"  [{desc}] {i+1:,}/{n:,}")
    return out
