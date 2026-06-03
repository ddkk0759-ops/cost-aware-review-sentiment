"""Local Ollama provider."""

from __future__ import annotations

import json
import os

from .. import config
from .base import LLMProvider, SYSTEM_PROMPT, parse_llm_json


class OllamaProvider(LLMProvider):
    backend = "ollama"
    model = config.LLM_OLLAMA_MODEL

    def __init__(self, model: str | None = None,
                 host: str | None = None):
        import requests
        self._requests = requests
        self.model = model or config.LLM_OLLAMA_MODEL
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def predict(self, text: str):
        prompt = self._build_prompt(text)
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0},
            }
            r = self._requests.post(
                f"{self.host}/api/chat", json=payload, timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            raw = data.get("message", {}).get("content", "") or json.dumps(data)
            parsed = parse_llm_json(raw) or {}
            return self._result_from_payload(parsed, raw)
        except Exception as e:
            return self._result_from_payload({}, "", error=str(e))
