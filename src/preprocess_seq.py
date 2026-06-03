"""
Sequence-pipeline preprocessing.

Build a token map (raw_token -> standardized_token) once on the full train+test
corpus, then advanced_tokenize is just a dict lookup.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

from . import config


def _ensure_nltk():
    import nltk
    for resource, path in [
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass


def _wn_pos(tag: str) -> str:
    from nltk.corpus import wordnet
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


_FALLBACK_BRE_AME = {
    "favourite": "favorite", "colour": "color", "behaviour": "behavior",
    "honour": "honor", "flavour": "flavor", "neighbour": "neighbor",
    "theatre": "theater", "centre": "center", "travelled": "traveled",
    "travelling": "traveling", "cancelled": "canceled", "labelled": "labeled",
    "organisation": "organization", "recognise": "recognize",
    "realise": "realize", "analyse": "analyze", "defence": "defense",
    "offence": "offense", "licence": "license", "practise": "practice",
}


def _to_american(word: str, breame_fn) -> str:
    if breame_fn is not None:
        try:
            ok, get = breame_fn
            if ok(word):
                return get(word)
        except Exception:
            pass
    return _FALLBACK_BRE_AME.get(word, word)


def _expand_and_split(text: str, contractions_mod) -> List[str]:
    text = text.lower()
    text = contractions_mod.fix(text)
    return re.findall(r"[a-z]+", text)


def build_token_map(
    train_texts: List[str],
    test_texts: List[str],
    w2v_model,
) -> Tuple[Dict[str, str], Counter]:
    """Build the standardisation token map shared across train + test."""
    _ensure_nltk()
    import contractions
    import nltk
    from nltk.stem import WordNetLemmatizer
    from spellchecker import SpellChecker

    try:
        from breame.spelling import (
            american_spelling_exists,
            get_american_spelling,
        )
        breame_fn = (american_spelling_exists, get_american_spelling)
    except ImportError:
        breame_fn = None

    print("[seq-pipeline] step 1/4: contractions + token collection")
    counter: Counter = Counter()
    for t in train_texts:
        counter.update(_expand_and_split(t, contractions))
    for t in test_texts:
        counter.update(_expand_and_split(t, contractions))
    unique_tokens = list(counter.keys())
    print(f"  unique raw tokens: {len(unique_tokens):,}")

    print("[seq-pipeline] step 2/4: POS tagging")
    pos_tags = nltk.pos_tag(unique_tokens)

    print("[seq-pipeline] step 3/4: spell-correct on words missing from W2V")
    spell = SpellChecker(language="en", distance=2)
    to_check = [tok for tok in unique_tokens if tok not in w2v_model]
    print(f"  W2V miss count (sent to spellcheck): {len(to_check):,}")
    unknown = spell.unknown(to_check)
    correction: Dict[str, str] = {}
    for w in unknown:
        if not w.isalpha() or len(w) < 3:
            continue
        cand = spell.correction(w)
        if cand and cand != w and cand in w2v_model:
            correction[w] = cand
    print(f"  spell-correct hits (also in W2V): {len(correction):,}")

    print("[seq-pipeline] step 4/4: build token_map (americanise + correct + lemma)")
    lemma = WordNetLemmatizer()
    token_map: Dict[str, str] = {}
    for tok, tag in pos_tags:
        out = _to_american(tok, breame_fn)
        if out not in w2v_model and out in correction:
            out = correction[out]
        out = lemma.lemmatize(out, _wn_pos(tag))
        token_map[tok] = out
    changed = sum(1 for k, v in token_map.items() if k != v)
    print(f"  token_map size={len(token_map):,}  effective changes={changed:,}")
    return token_map, counter


def make_tokenize_fn(token_map: Dict[str, str]):
    """Return a fast tokenizer using the prebuilt map."""
    import contractions

    def tokenize(text: str) -> List[str]:
        text = text.lower()
        text = contractions.fix(text)
        raw = re.findall(r"[a-z]+", text)
        return [token_map.get(t, t) for t in raw]

    return tokenize


def build_vocab(train_texts: List[str], tokenize) -> Tuple[List[str], Dict[str, int]]:
    PAD, UNK = "<PAD>", "<UNK>"
    counter: Counter = Counter()
    for t in train_texts:
        counter.update(tokenize(t))
    vocab = [PAD, UNK] + [w for w, _ in counter.most_common()]
    word2idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word2idx


def build_embedding_matrix(vocab: List[str], word2idx: Dict[str, int], w2v_model):
    import numpy as np
    PAD, UNK = "<PAD>", "<UNK>"
    embed_dim = w2v_model.vector_size
    mat = np.zeros((len(vocab), embed_dim), dtype=np.float32)
    hit, miss = 0, 0
    rng = np.random.RandomState(config.SEED)
    for word, idx in word2idx.items():
        if word in (PAD, UNK):
            continue
        if word in w2v_model:
            mat[idx] = w2v_model[word]
            hit += 1
        else:
            mat[idx] = rng.normal(0, 0.1, embed_dim).astype(np.float32)
            miss += 1
    hit_idx = [idx for w, idx in word2idx.items()
               if w not in (PAD, UNK) and w in w2v_model]
    if hit_idx:
        mat[word2idx[UNK]] = mat[hit_idx].mean(axis=0)
    print(f"[w2v] hit={hit}  miss={miss}  ({miss / max(hit + miss, 1) * 100:.1f}%)")
    return mat


def text_to_indices(text: str, tokenize, word2idx, max_len: int) -> List[int]:
    PAD = word2idx["<PAD>"]
    UNK = word2idx["<UNK>"]
    tokens = tokenize(text)[:max_len]
    idxs = [word2idx.get(t, UNK) for t in tokens]
    idxs += [PAD] * (max_len - len(idxs))
    return idxs
