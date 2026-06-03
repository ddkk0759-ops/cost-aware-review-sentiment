"""OpenAI / OpenAI-compatible Chat Completions provider."""

from __future__ import annotations

import os

from .. import config
from .base import LLMProvider, SYSTEM_PROMPT, parse_llm_json


class OpenAIProvider(LLMProvider):
    backend = "openai"
    model = config.LLM_OPENAI_MODEL

    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.model = model or config.LLM_OPENAI_MODEL
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def predict(self, text: str):
        prompt = self._build_prompt(text)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            payload = parse_llm_json(raw) or {}
            return self._result_from_payload(payload, raw)
        except Exception as e:
            return self._result_from_payload({}, "", error=str(e))
