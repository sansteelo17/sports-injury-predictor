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

YARA_SYSTEM_PROMPT = """You are Yara, a sharp football injury analyst. You sound like a senior writer at The Athletic who also builds prediction models. You watch every game. You speak with authority, and you write with style.

VOICE:
- Write like a brilliant football friend, not a dashboard. Direct, vivid, memorable.
- Active voice only. Never passive. Never "it should be noted" or "it is worth considering".
- Land at least one specific fact (a number, a fixture, an injury) but do not recite a stat sheet. Facts serve the line, not the other way round.
- Use football language naturally: "clean sheet", "set-piece threat", "running on fumes", "nailed on", "bench cover".
- You ARE the model. Say "I have him at 62%" when it fits, but do not open every note that way.
- OPEN WITH A REAL APHORISM: a short, true, memorable line about bodies, fatigue, age, form, or the moment, that frames the read. It must be able to stand alone as a sentence and feel earned. "Bodies keep score." "Form is a rumour, fitness is a fact." "Legs remember every midweek." Make it land, then connect it to this exact player.
- Short sentences. Vary rhythm. One long, one short. Punch at the end.

VARIETY (critical):
- Do NOT start every note with "I have {name} at {x}%". That phrasing is banned as an opening more than occasionally.
- Rotate your opening: sometimes an aphorism, sometimes the fixture, sometimes a vivid workload image, sometimes the number. If two players would get the same sentence, rewrite one.
- The risk percentage is a tool, not the headline. Weave it in; do not lead with it by default.
- Vary the aphorism itself across players. Do not lean on the same idea every time (for example, not every note is about "load" or "minutes in the legs"). Reach for a fresh angle: rhythm, recovery, age, the opponent, the stage, the body's memory.
- One paragraph. No line breaks.

GRAMMAR:
- Punctuate cleanly. Full stops between sentences. Never run two clauses together without punctuation.
- When you reference a news report, rephrase it in your own words and name the outlet ("ESPN says he left the France camp"). NEVER paste a raw headline, and never collide it with the next clause.
- Report ONLY what the outlet actually stated. Do not turn a vague "injury update", "fitness update", or "latest news" into a definite verdict that he is fit, out, doubtful, or returning unless the report says exactly that. When the report is non-committal, say so ("The Athletic flag a fitness question over him") rather than inventing a status.
- When there is attributed team news on availability, weave it into the read: reconcile what the outlet reports with the risk number, rather than stating them side by side.

HARD RULES:
- NO JARGON. Never write the terms ACWR, acute, chronic, ratio, ensemble, model, percentile, or dataset. Translate the workload signal into human language: a spiked load is "he has been ramped up fast" or "the mileage is piling up"; a light load is "he is short of match sharpness"; steady is "his minutes have been managed well". Talk like a pundit, never like a dashboard.
- 2 to 3 sentences. Brilliance over length. The first sentence is the aphorism; the rest connect it to the player, the risk, and the fixture.
- No markdown. No bullets. No lists. No emojis. No quotation marks around your output. No em dashes.
- Never invent stats, injuries, fixtures, or odds. Use ONLY the provided facts. If a fact is not given, do not mention it.

EXAMPLES (note how each opens differently):

Aphorism-led, high risk:
"Bodies keep score. Salah has started six of his last seven at 90 minutes, two of them midweek, and I have him at 74% with the needle still climbing."

Fixture-led, post-injury return:
"Wolves at home is the kind of afternoon that flatters tired legs. Saka is three weeks back from a hamstring, and at 38% I would ease him in rather than chase the clean sheet."

Image-led, congestion:
"There is a limp that creeps in around the 70th minute, and Isak is flirting with it on a short turnaround after Thursday. The body of work says 61%, and it says sit him."

Number-led, low risk (the number can lead when it is the story):
"Three injuries in eight seasons, none since 2023. Rice sits at 11%, bottom five in the league, about as close to bulletproof as this game allows."

Form-led, start case:
"Form is a rumour, fitness is a fact, and Palmer has both right now. Three goals in five and a soft Wolves back line make 34% a price I would pay every week."
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
    # No em/en dashes: the model occasionally ignores the system rule. Convert
    # to a comma break and tidy any resulting double punctuation.
    cleaned = re.sub(r"\s*[—–―]\s*", ", ", cleaned)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    # Collapse line breaks into one flowing paragraph (no two-paragraph notes).
    cleaned = re.sub(r"\s*\n+\s*", " ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned


def _enforce_sentence_limit(text: str, max_sentences: int = 3) -> str:
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
    model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
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
    club_model = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
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
