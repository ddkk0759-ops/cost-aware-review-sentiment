"""
Stub LLM provider — used when no API key is configured.

We need *something* deterministic so the rest of the analysis pipeline
(agreement statistics, suspected-mislabel rates, etc.) can be exercised
end-to-end without a real key.

Heuristic: a tiny VADER-based scorer.  This is intentionally unsophisticated;
it exists only to keep the demo alive.  Real evaluation should switch
``LLM_BACKEND`` to ``openai`` / ``anthropic`` / ``ollama``.
"""

from __future__ import annotations

import json

from .base import LLMProvider, LLMResult


def _ensure_vader():
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


class StubProvider(LLMProvider):
    backend = "stub"
    model = "vader-heuristic"

    def __init__(self):
        _ensure_vader()
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self._sia = SentimentIntensityAnalyzer()

    def predict(self, text: str) -> LLMResult:
        score = self._sia.polarity_scores(text or "")
        compound = score["compound"]
        # compound >= 0.05 -> positive (label 0); <= -0.05 -> negative (label 1)
        if compound <= -0.05:
            label = 1
            rationale = "VADER: 整体负向评分占优"
        elif compound >= 0.05:
            label = 0
            rationale = "VADER: 整体正向评分占优"
        else:
            label = 0 if compound >= 0 else 1
            rationale = "VADER: 中性，按符号兜底"
        confidence = min(1.0, abs(compound) + 0.5)
        payload = {"label": label, "confidence": confidence, "rationale": rationale}
        raw = json.dumps(payload, ensure_ascii=False)
        return self._result_from_payload(payload, raw)
