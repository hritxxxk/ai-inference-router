"""Real (or simulated fallback) model calls for the inference router."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator, Tuple

try:  # pragma: no cover - import guard for optional deps
    from google import genai
except ImportError:  # pragma: no cover - dependency missing only in CI
    genai = None  # type: ignore

try:  # pragma: no cover - import guard for optional deps
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

from src.config import settings


logger = logging.getLogger(__name__)

GOOGLE_CLIENT = genai.Client(api_key=settings.google_api_key) if (genai and settings.google_api_key) else None
OPENAI_CLIENT = AsyncOpenAI(api_key=settings.openai_api_key) if (AsyncOpenAI and settings.openai_api_key) else None

async def _simulate_response(label: str, prompt: str, delay: float) -> Tuple[str, float]:
    await asyncio.sleep(delay)
    return f"{label}: Simulated response for '{prompt[:60]}...'", delay


def _extract_google_text(response: object) -> str:
    text = getattr(response, "text", None)
    if text:
        return text
    candidates = getattr(response, "candidates", []) or []
    parts = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)
    return "\n".join(parts).strip()


async def _call_google_model(model_name: str, label: str, prompt: str, fallback_delay: float) -> Tuple[str, float]:
    if GOOGLE_CLIENT is None:
        logger.debug("Google API key missing; using simulated response for %s", model_name)
        return await _simulate_response(label, prompt, fallback_delay)

    start = time.perf_counter()
    try:
        response = await GOOGLE_CLIENT.aio.models.generate_content(model=model_name, contents=prompt)
        text = _extract_google_text(response) or f"{label}: (empty response)"
    except Exception as exc:  # pragma: no cover - safety net for production
        logger.exception("Google model %s failed: %s", model_name, exc)
        if settings.google_api_key:
            raise
        return await _simulate_response(label, prompt, fallback_delay)
    latency = time.perf_counter() - start
    return text, latency


async def _call_openai_model(model_name: str, prompt: str, fallback_delay: float) -> Tuple[str, float]:
    label = model_name
    if OPENAI_CLIENT is None:
        logger.debug("OpenAI API key missing; using simulated response for %s", model_name)
        return await _simulate_response(label, prompt, fallback_delay)

    start = time.perf_counter()
    try:
        response = await OPENAI_CLIENT.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip() if response.choices else ""
        if not text:
            text = f"{label}: (empty response)"
    except Exception as exc:  # pragma: no cover - safety net for production
        logger.exception("OpenAI model %s failed: %s", model_name, exc)
        if settings.openai_api_key:
            raise
        return await _simulate_response(label, prompt, fallback_delay)
    latency = time.perf_counter() - start
    return text, latency


async def call_gemini_pro_stream(prompt: str) -> AsyncGenerator[str, None]:
    """Stream chunks from the premium Gemini model (falls back to simulated chunks)."""
    response_text, _ = await call_gemini_pro(prompt)
    chunk_size = 200
    for idx in range(0, len(response_text), chunk_size):
        yield response_text[idx: idx + chunk_size]


async def call_gemini_pro(prompt: str) -> Tuple[str, float]:
    """Invoke Gemini 3.1 Pro via the google-genai SDK (or simulate when offline)."""
    return await _call_google_model(
        settings.gemini_model_name,
        "Gemini 3.1 Pro",
        prompt,
        fallback_delay=1.2,
    )


async def call_fine_tuned_gemma3(prompt: str) -> Tuple[str, float]:
    """Invoke Gemma 3 27B via google-genai or fall back to a fast simulation."""
    return await _call_google_model(
        settings.gemma_model_name,
        "Gemma 3 27B",
        prompt,
        fallback_delay=0.5,
    )


async def call_code_specialist(prompt: str) -> Tuple[str, float]:
    """Invoke GPT-5.3 Codex via the OpenAI SDK (code specialist)."""
    return await _call_openai_model(settings.code_model_name, prompt, fallback_delay=0.7)


async def call_math_reasoner(prompt: str) -> Tuple[str, float]:
    """Invoke Gemini 3 Deep Think for math-heavy prompts."""
    return await _call_google_model(
        settings.math_model_name,
        "Gemini 3 Deep Think",
        prompt,
        fallback_delay=1.0,
    )
