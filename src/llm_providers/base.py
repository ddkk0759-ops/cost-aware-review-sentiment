"""Common LLM provider interface."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Optional


SYSTEM_PROMPT = (
    "你是一个严谨的电商评论情感判定助手。"
    "用户会给你一条 Amazon 风格的英文评论原文，你需要判定它属于差评还是好评。"
    "差评的常见信号：失望、退货、烂、骗人、不值、退款、生气、糟糕、never buy again 等。"
    "好评的常见信号：喜欢、推荐、值、满意、love it、worth it、amazing 等。"
    "若评论本身存在语义矛盾（前褒后贬 / 反讽 / 转折），请判定整体倾向。"
    "你必须只用 JSON 输出，键固定为：label (0 表示好评，1 表示差评), "
    "confidence (0~1), rationale (一句话中文解释，<=40 字)。"
)

USER_TEMPLATE = (
    '请判定下列评论的情感。\n'
    '只返回 JSON：{{"label": 0 或 1, "confidence": 浮点, "rationale": "..."}}\n'
    '评论原文：\n"""\n{text}\n"""'
)


@dataclass
class LLMResult:
    label: int                          # 0 = positive, 1 = negative
    confidence: float
    rationale: str
    raw_response: str
    backend: str
    model: str
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_llm_json(raw: str) -> Optional[dict]:
    """Best-effort extraction of the JSON object from raw LLM text."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = _JSON_BLOCK.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class LLMProvider:
    """Abstract LLM client.  Subclasses override ``predict``."""

    backend: str = "base"
    model: str = "n/a"

    def predict(self, text: str) -> LLMResult:  # pragma: no cover
        raise NotImplementedError

    def _build_prompt(self, text: str) -> str:
        return USER_TEMPLATE.format(text=text.replace('"""', "'''"))

    def _result_from_payload(self, payload: dict, raw: str,
                             error: Optional[str] = None) -> LLMResult:
        label = int(payload.get("label", -1)) if payload else -1
        if label not in (0, 1):
            label = -1
        try:
            conf = float(payload.get("confidence", 0.0)) if payload else 0.0
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        rationale = str(payload.get("rationale", "")) if payload else ""
        return LLMResult(
            label=label,
            confidence=conf,
            rationale=rationale,
            raw_response=raw,
            backend=self.backend,
            model=self.model,
            error=error,
        )
