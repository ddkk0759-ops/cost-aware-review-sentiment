"""Load pretrained Word2Vec (KeyedVectors) for the sequence pipeline."""

from __future__ import annotations

import os
import time
import urllib.error
from pathlib import Path

from . import config


def _load_from_local(path: Path):
    from gensim.models import KeyedVectors

    suf = "".join(path.suffixes).lower()
    name = path.name.lower()
    if path.suffix == ".kv" or name.endswith(".kv"):
        return KeyedVectors.load(str(path))
    if ".bin" in suf or "googlenews" in name or path.suffix.lower() == ".bin":
        return KeyedVectors.load_word2vec_format(str(path), binary=True)
    return KeyedVectors.load_word2vec_format(str(path), binary=False)


def load_keyed_vectors():
    """
    Return gensim KeyedVectors (``__contains__`` / ``__getitem__`` / ``vector_size``).

    1. If ``W2V_LOCAL_PATH`` **environment variable** (checked at call time) or
       ``config.W2V_LOCAL_PATH`` points to an existing file, load from disk
       (``.kv``, ``.bin`` / ``.bin.gz``, or text ``.vec``).
       (Call-time env first, so notebook can ``os.environ[...]=...`` before load
       without re-importing ``config``.)
    2. Otherwise ``gensim.downloader.api.load(config.W2V_NAME)``, retrying on
       transient ``RemoteDisconnected`` / timeout-style errors.
    """
    local = os.environ.get("W2V_LOCAL_PATH", "").strip() or (config.W2V_LOCAL_PATH or "").strip()
    if local:
        p = Path(local).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"W2V_LOCAL_PATH is not a file: {p.resolve()}")
        return _load_from_local(p)

    import gensim.downloader as api

    attempts = max(1, int(os.environ.get("W2V_DOWNLOAD_ATTEMPTS", "8")))
    base_delay = float(os.environ.get("W2V_DOWNLOAD_RETRY_SEC", "3"))
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return api.load(config.W2V_NAME)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last = e
            if attempt + 1 >= attempts:
                break
            delay = min(base_delay * (2**attempt), 120.0)
            print(
                f"[w2v] download failed ({type(e).__name__}: {e}); "
                f"retry in {delay:.0f}s ({attempt + 1}/{attempts})..."
            )
            time.sleep(delay)
    raise RuntimeError(
        "Could not download/load Word2Vec via gensim. Options: "
        "(1) export W2V_LOCAL_PATH=/path/to/GoogleNews-vectors-negative300.bin.gz "
        "(2) use HTTP(S)_PROXY if required "
        "(3) increase W2V_DOWNLOAD_ATTEMPTS / W2V_DOWNLOAD_RETRY_SEC."
    ) from last
