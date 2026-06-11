"""Minimal OpenAI-compatible chat caller, GPT-5.x-safe.

Newer models reject ``max_tokens``/``temperature`` and burn a big reasoning
budget, so on a 400 we retry with ``max_completion_tokens`` floored high (the
same gotcha the app's llm_client handles). Returns the assistant text or "".
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import requests

from . import config


def chat(messages: List[Dict[str, str]], max_tokens: int = 1400, temperature: float = 0.7) -> str:
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = f"{config.OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": config.AUTOPOST_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    if resp.status_code == 400:
        # Reasoning models: swap to max_completion_tokens (floored) and drop temp.
        payload.pop("max_tokens", None)
        payload.pop("temperature", None)
        payload["max_completion_tokens"] = max(max_tokens, 2000)
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()


def chat_json(messages: List[Dict[str, str]], max_tokens: int = 1400) -> Any:
    """Chat call that must return JSON. Strips ```json fences and parses
    tolerantly (bare array or object)."""
    raw = chat(messages, max_tokens=max_tokens, temperature=0.6)
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        # Last resort: grab the outermost {...} or [...] block.
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise ValueError(f"LLM did not return JSON: {raw[:300]}")
