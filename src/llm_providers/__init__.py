"""LLM provider plug-in registry."""

from __future__ import annotations

import os
from typing import Optional

from .. import config
from .base import LLMProvider, LLMResult


def get_provider(backend: Optional[str] = None) -> LLMProvider:
    """Resolve provider by name; default to config.LLM_BACKEND."""
    name = (backend or config.LLM_BACKEND).lower()
    if name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("[llm] OPENAI_API_KEY not found; falling back to stub")
            from .stub_provider import StubProvider
            return StubProvider()
        from .openai_provider import OpenAIProvider
        return OpenAIProvider()
    if name == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[llm] ANTHROPIC_API_KEY not found; falling back to stub")
            from .stub_provider import StubProvider
            return StubProvider()
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    if name == "ollama":
        from .ollama_provider import OllamaProvider
        return OllamaProvider()
    from .stub_provider import StubProvider
    return StubProvider()


__all__ = ["LLMProvider", "LLMResult", "get_provider"]
