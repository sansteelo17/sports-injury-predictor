"""Optional small-LLM helper for grounded narrative generation.

Supports:
- `NARRATIVE_LLM_PROVIDER=ollama`
- `NARRATIVE_LLM_PROVIDER=openai_compatible`

If provider/env is not configured or request fails, callers should use fallback text.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _provider() -> str:
    return (os.getenv("NARRATIVE_LLM_PROVIDER", "none") or "none").strip().lower()


def llm_enabled() -> bool:
    return _provider() in {"ollama", "openai_compatible"}


def _clean_output(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) > 2:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _build_grounded_prompt(
    task: str,
    player_name: str,
    context_chunks: List[Dict[str, Any]],
    fallback_text: str,
    require_open_question: bool = True,
) -> str:
    lines: List[str] = []
    for chunk in context_chunks[:8]:
        text = (chunk.get("text") or "").strip()
        if text:
            lines.append(f"- {text}")
    context_block = "\n".join(lines) if lines else "- No additional context chunks."

    question_rule = (
        "- include one open-ended football question at the end (no label/prefix)\n"
        if require_open_question
        else "- do not end with a question\n"
    )

    return (
        f"Task: {task}\n"
        f"Player: {player_name}\n\n"
        "Grounded facts (use only these + football common sense):\n"
        f"{context_block}\n\n"
        "Writing requirements:\n"
        "- 2 to 4 sentences\n"
        "- conversational football language with confidence and heat\n"
        "- no markdown, no lists\n"
        "- do not invent injuries, fixtures, or odds\n"
        "- avoid repetitive sentence openings\n"
        f"{question_rule}"
        "- keep it decisive and useful for an FPL/odds audience\n\n"
        f"If uncertain, stay close to this fallback:\n{fallback_text}"
    )


def _call_ollama(system_prompt: str, user_prompt: str) -> Optional[str]:
    base_url = (os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434") or "").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    temperature = float(os.getenv("NARRATIVE_LLM_TEMPERATURE", "0.55"))
    timeout = float(os.getenv("NARRATIVE_LLM_TIMEOUT_SECONDS", "10"))

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.warning(f"Ollama returned HTTP {resp.status_code}")
            return None
        data = resp.json()
        return _clean_output(data.get("response", ""))
    except Exception as e:
        logger.warning(f"Ollama narrative generation failed: {e}")
        return None


def _call_openai_compatible(system_prompt: str, user_prompt: str) -> Optional[str]:
    base_url = (os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") or "").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("NARRATIVE_LLM_TEMPERATURE", "0.55"))
    timeout = float(os.getenv("NARRATIVE_LLM_TIMEOUT_SECONDS", "10"))

    if not api_key:
        logger.warning("OPENAI_API_KEY not set; skipping openai_compatible narrative generation")
        return None

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": temperature,
                "max_tokens": 220,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.warning(f"openai_compatible returned HTTP {resp.status_code}")
            return None
        data = resp.json()
        content = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content", "")
        return _clean_output(content)
    except Exception as e:
        logger.warning(f"openai_compatible narrative generation failed: {e}")
        return None


def generate_grounded_narrative(
    task: str,
    player_name: str,
    context_chunks: List[Dict[str, Any]],
    fallback_text: str,
    require_open_question: bool = True,
) -> str:
    """Generate a grounded narrative with optional LLM; fallback deterministically."""
    if not llm_enabled():
        return fallback_text

    system_prompt = (
        "You are Yara, a football analytics assistant. Be vivid but factual. "
        "Never invent stats. Use only provided facts."
    )
    user_prompt = _build_grounded_prompt(
        task=task,
        player_name=player_name,
        context_chunks=context_chunks,
        fallback_text=fallback_text,
        require_open_question=require_open_question,
    )

    provider = _provider()
    output: Optional[str] = None
    if provider == "ollama":
        output = _call_ollama(system_prompt, user_prompt)
    elif provider == "openai_compatible":
        output = _call_openai_compatible(system_prompt, user_prompt)

    if not output:
        return fallback_text
    return output
