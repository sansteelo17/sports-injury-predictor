"""Optional small-LLM helper for grounded narrative generation.

Supports:
- `NARRATIVE_LLM_PROVIDER=ollama`
- `NARRATIVE_LLM_PROVIDER=openai_compatible`

If provider/env is not configured or request fails, callers should use fallback text.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Yara voice bible ────────────────────────────────────────────────────────

YARA_SYSTEM_PROMPT = """You are Yara, a sharp football injury analyst. You sound like a senior writer at The Athletic who also builds prediction models. You watch every game. You speak with authority.

VOICE:
- Write like you are talking to a friend who manages an FPL team. Direct, punchy, zero fluff.
- Active voice only. Never passive. Never "it should be noted" or "it is worth considering".
- Every sentence must contain a real number, a player name, or a specific fact. No filler sentences.
- Use football language naturally: "clean sheet", "set-piece threat", "running on fumes", "nailed on", "bench cover".
- You ARE the model. Say "I have him at 62%" not "the model predicts". Say "I see" not "analysis suggests".
- Short sentences. Vary rhythm. One long, one short. Punch at the end.
- Lead with the most distinctive player-specific or fixture-specific fact available. If a line could fit 20 players, rewrite it.
- Avoid stock phrases like "worth watching", "fixture-dependent", "the body is a concern", or "good week to start" unless the facts make that exact point unavoidable.
- For non-risk sections, do not re-tell the full injury story unless the injury context directly changes the football decision.
- 2 to 4 sentences. Never more. Every word earns its place.
- No markdown. No bullets. No lists. No emojis. No quotation marks around your output.
- Never invent stats, injuries, fixtures, or odds. Use ONLY the provided facts.

EXAMPLES:

High risk — fixture congestion:
"Salah has played 90 minutes in six of his last seven. Two midweek, one extra-time cup game. I have him at 74% risk heading into the weekend and that number climbs every minute he stays on the pitch."

Elevated risk — post-injury return:
"42 days since the hamstring and Saka is back in full training. The body is willing but I still have him at 38% — recurrence rates spike in the first three weeks back. Ease him in."

Low risk — strong profile:
"Three injuries in eight seasons and none since 2023. I have Rice at 11% risk, bottom five in the league. He is as close to bulletproof as the Premier League gets."

FPL insight — start case:
"Palmer has scored in three of his last five and faces a Wolves side leaking 1.9 goals per game at home. Even at 34% risk I would start him and not think twice."

FPL insight — bench case:
"I have Isak at 61% risk on a short turnaround after Thursday night. The body of work says sit him this week and bring him back fresh for the double."
"""


# ── Prompt building ─────────────────────────────────────────────────────────

def _build_grounded_prompt(
    task: str,
    player_name: str,
    context_chunks: List[Dict[str, Any]],
    fallback_text: str,
    require_open_question: bool = True,
) -> str:
    lines: List[str] = []
    for chunk in context_chunks[:10]:
        text = (chunk.get("text") or "").strip()
        if text:
            lines.append(f"- {text}")
    context_block = "\n".join(lines) if lines else "- No additional context chunks."

    question_rule = (
        "End with one open-ended football question — no label, no prefix, just the question.\n"
        if require_open_question
        else "Do not end with a question.\n"
    )

    return (
        f"TASK: {task}\n"
        f"PLAYER: {player_name}\n\n"
        "FACTS (use ONLY these — never invent):\n"
        f"{context_block}\n\n"
        "RULES:\n"
        f"- 2 to 4 sentences maximum. Hard limit.\n"
        f"- {question_rule}"
        f"- Sound like the examples in your system prompt. Match that voice exactly.\n"
    )


def _build_bundle_prompt(
    player_name: str,
    sections: List[Dict[str, Any]],
) -> str:
    blocks: List[str] = []
    tag_names = ", ".join(section["tag"] for section in sections)
    for section in sections:
        facts = section.get("context_chunks") or []
        fact_lines: List[str] = []
        for chunk in facts[:10]:
            text = (chunk.get("text") or "").strip()
            if text:
                fact_lines.append(f"- {text}")
        fact_block = "\n".join(fact_lines) if fact_lines else "- No additional context chunks."
        fallback_text = (section.get("fallback_text") or "").strip()
        blocks.append(
            f"SECTION: {section['tag']}\n"
            f"TASK: {section['task']}\n"
            "FACTS (use ONLY these — never invent):\n"
            f"{fact_block}\n"
            "FALLBACK DRAFT:\n"
            f"{fallback_text or 'No fallback draft provided.'}\n"
        )

    return (
        f"PLAYER: {player_name}\n"
        "You are writing multiple independent narrative sections for the same player.\n"
        "Each section must sound distinct and answer only its own question.\n"
        "Do not repeat the same opening, the same injury-history line, or the same closing across sections.\n"
        "Keep every section to 2 to 4 sentences. Hard limit.\n"
        "Return ONLY tagged sections. No markdown. No explanations outside the tags.\n"
        f"Use exactly these tags once each: {tag_names}.\n\n"
        + "\n".join(blocks)
    )


# ── LLM calls ───────────────────────────────────────────────────────────────

def _provider() -> str:
    return (os.getenv("NARRATIVE_LLM_PROVIDER", "none") or "none").strip().lower()


def llm_enabled() -> bool:
    return _provider() in {"ollama", "openai_compatible"}


def _clean_output(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) > 2:
        cleaned = cleaned[1:-1].strip()
    # Strip any markdown formatting the LLM might have added
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)
    # Strip bullet points or list markers
    cleaned = re.sub(r"^[\-\*•]\s+", "", cleaned, flags=re.MULTILINE)
    return cleaned


def _enforce_sentence_limit(text: str, max_sentences: int = 4) -> str:
    """Hard-cap output at max_sentences. Keeps complete sentences only."""
    if not text:
        return text
    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(parts) <= max_sentences:
        return text.strip()
    return " ".join(parts[:max_sentences]).strip()


def _call_ollama(system_prompt: str, user_prompt: str, max_output_tokens: int = 300) -> Optional[str]:
    base_url = (os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434") or "").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    temperature = float(os.getenv("NARRATIVE_LLM_TEMPERATURE", "0.6"))
    timeout = float(os.getenv("NARRATIVE_LLM_TIMEOUT_SECONDS", "10"))

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_output_tokens},
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


def _post_openai_chat(
    base_url: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: float,
):
    """POST a chat completion, adapting to per-model parameter quirks.

    Newer OpenAI models (GPT-5.x, o-series) reject ``max_tokens`` (want
    ``max_completion_tokens``) and only allow the default ``temperature``. Rather
    than hardcode a model-name allowlist, we send the broadly-compatible payload
    first and, on a 400 that names an unsupported parameter, mutate that one
    field and retry. Returns the final ``requests.Response`` (caller inspects it).
    """
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = dict(payload)
    for _ in range(3):
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 400:
            return resp
        low = (resp.text or "").lower()
        mutated = False
        if "max_tokens" in low and "max_completion_tokens" in low and "max_tokens" in payload:
            # Models that require max_completion_tokens are reasoning models
            # (GPT-5.x, o-series): they spend part of the budget on hidden
            # reasoning, so a tight cap leaves the visible content empty. Floor
            # it high enough that reasoning + answer both fit, and cap reasoning
            # effort — a 3-sentence scouting note needs no deep reasoning, and
            # the default effort pushes latency past the request timeout (the
            # "only one player works" symptom: slow calls abort to template).
            requested = int(payload.pop("max_tokens") or 0)
            payload["max_completion_tokens"] = max(requested, 1200)
            payload.setdefault("reasoning_effort", os.getenv("OPENAI_REASONING_EFFORT", "low"))
            mutated = True
        if "temperature" in low and "temperature" in payload and (
            "unsupported" in low or "does not support" in low or "only the default" in low
        ):
            payload.pop("temperature", None)
            mutated = True
        if "reasoning_effort" in low and "reasoning_effort" in payload and (
            "unsupported" in low or "does not support" in low or "invalid" in low
        ):
            payload.pop("reasoning_effort", None)
            mutated = True
        if not mutated:
            return resp
    return resp


def _call_openai_compatible(system_prompt: str, user_prompt: str, max_output_tokens: int = 300) -> Optional[str]:
    base_url = (os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") or "").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("NARRATIVE_LLM_TEMPERATURE", "0.6"))
    timeout = float(os.getenv("NARRATIVE_LLM_TIMEOUT_SECONDS", "10"))

    if not api_key:
        logger.warning("OPENAI_API_KEY not set; skipping openai_compatible narrative generation")
        return None

    try:
        resp = _post_openai_chat(
            base_url,
            api_key,
            {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_output_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout,
        )
        if resp.status_code != 200:
            body_preview = (resp.text or "").strip().replace("\n", " ")[:400]
            logger.warning(
                "openai_compatible returned HTTP %s for model=%s base_url=%s body=%s",
                resp.status_code,
                model,
                base_url,
                body_preview,
            )
            return None
        data = resp.json()
        content = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content", "")
        return _clean_output(content)
    except Exception as e:
        logger.warning(
            "openai_compatible narrative generation failed for model=%s base_url=%s: %s",
            model,
            base_url,
            e,
        )
        return None


# ── Startup health check ──────────────────────────────────────────────────────

def probe_openai_models() -> None:
    """Probe the configured OpenAI model(s) at startup and log loudly on failure.

    A wrong/unavailable ``OPENAI_MODEL`` or ``INTL_OPENAI_MODEL`` causes every
    narrative call to 400 and silently fall back to deterministic template text.
    This sends one tiny (max_tokens=1) completion per distinct configured model
    so a bad id is surfaced at boot instead of hiding behind plausible-looking
    fallback copy. Never raises — startup must not depend on the LLM.
    """
    provider = _provider()
    intl_provider = (os.getenv("INTL_NARRATIVE_LLM_PROVIDER", "") or "").strip().lower()
    if "openai_compatible" not in {provider, intl_provider}:
        return

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; narratives will use deterministic template text.")
        return

    base_url = (os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") or "").rstrip("/")
    club_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    intl_model = (os.getenv("INTL_OPENAI_MODEL", "") or "").strip()

    # {model: which-narratives-it-serves} so the log says what breaks if it fails.
    to_check: Dict[str, str] = {}
    if provider == "openai_compatible":
        to_check[club_model] = "club-league"
    if intl_provider == "openai_compatible" and intl_model:
        to_check[intl_model] = "World Cup"

    for model, scope in to_check.items():
        try:
            resp = _post_openai_chat(
                base_url,
                api_key,
                {
                    "model": model,
                    "temperature": float(os.getenv("NARRATIVE_LLM_TEMPERATURE", "0.6")),
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "ping"}],
                },
                timeout=15,
            )
            if resp.status_code == 200:
                logger.info("LLM model OK: %s (%s narratives)", model, scope)
            else:
                body_preview = (resp.text or "").strip().replace("\n", " ")[:300]
                logger.error(
                    "LLM model FAILED: %s (%s narratives) returned HTTP %s. "
                    "These narratives will SILENTLY fall back to template text. body=%s",
                    model, scope, resp.status_code, body_preview,
                )
        except Exception as e:
            logger.error(
                "LLM model probe errored for %s (%s narratives): %s. "
                "These narratives may fall back to template text.",
                model, scope, e,
            )


# ── Public API ───────────────────────────────────────────────────────────────

def generate_grounded_narrative(
    task: str,
    player_name: str,
    context_chunks: List[Dict[str, Any]],
    fallback_text: str,
    require_open_question: bool = True,
) -> str:
    """Generate a grounded narrative with optional LLM; fallback deterministically."""
    if not llm_enabled():
        return _enforce_sentence_limit(fallback_text)

    system_prompt = YARA_SYSTEM_PROMPT
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
        return _enforce_sentence_limit(fallback_text)
    return _enforce_sentence_limit(output)


def _extract_tagged_section(text: str, tag: str) -> Optional[str]:
    pattern = rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>"
    match = re.search(pattern, text or "", flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return _clean_output(match.group(1))


def generate_grounded_section_bundle(
    player_name: str,
    sections: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Generate multiple grounded narrative sections in one model call.

    Each input section must include:
    - tag: stable output tag name
    - task: section-specific instruction
    - context_chunks: grounded fact chunks
    - fallback_text: deterministic fallback for that section
    """
    bundle: Dict[str, str] = {}
    if not sections:
        return bundle

    for section in sections:
        tag = str(section.get("tag") or "").strip()
        if tag:
            bundle[tag] = _enforce_sentence_limit(str(section.get("fallback_text") or "").strip())

    if not llm_enabled():
        return bundle

    system_prompt = (
        YARA_SYSTEM_PROMPT
        + "\n\nSTRUCTURE:\n"
        + "- Follow the requested XML-like tags exactly.\n"
        + "- Keep each section self-contained and distinct from the others.\n"
        + "- If two sections share a fact, phrase it differently and use it for different football decisions.\n"
    )
    user_prompt = _build_bundle_prompt(player_name=player_name, sections=sections)

    provider = _provider()
    output: Optional[str] = None
    max_output_tokens = max(700, 180 * len(sections))
    if provider == "ollama":
        output = _call_ollama(system_prompt, user_prompt, max_output_tokens=max_output_tokens)
    elif provider == "openai_compatible":
        output = _call_openai_compatible(system_prompt, user_prompt, max_output_tokens=max_output_tokens)

    if not output:
        logger.warning(
            "Narrative bundle generation fell back to deterministic copy for %s (provider=%s, sections=%s)",
            player_name,
            provider,
            [str(section.get("tag") or "") for section in sections],
        )
        return bundle

    for section in sections:
        tag = str(section.get("tag") or "").strip()
        if not tag:
            continue
        extracted = _extract_tagged_section(output, tag)
        if extracted:
            bundle[tag] = _enforce_sentence_limit(extracted)

    return bundle
