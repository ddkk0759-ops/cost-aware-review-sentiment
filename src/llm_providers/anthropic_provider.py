"""Anthropic (Claude) provider."""

from __future__ import annotations

import os

from .. import config
from .base import LLMProvider, SYSTEM_PROMPT, parse_llm_json


class AnthropicProvider(LLMProvider):
    backend = "anthropic"
    model = config.LLM_ANTHROPIC_MODEL

    def __init__(self, model: str | None = None):
        import anthropic
        self.model = model or config.LLM_ANTHROPIC_MODEL
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def predict(self, text: str):
        prompt = self._build_prompt(text)
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = ""
            for block in resp.content:
                if hasattr(block, "text"):
                    raw += block.text
            payload = parse_llm_json(raw) or {}
            return self._result_from_payload(payload, raw)
        except Exception as e:
            return self._result_from_payload({}, "", error=str(e))
