"""Routing executor with smart fallback logic."""

import logging
from typing import Awaitable, Callable, Dict, Optional, Tuple

from src.config import settings
from src.services.model_calls import (
    call_code_specialist,
    call_fine_tuned_gemma3,
    call_gemini_pro,
    call_math_reasoner,
)
from src.services.routing_engine import RoutingDecision


logger = logging.getLogger(__name__)

ModelCall = Callable[[str], Awaitable[Tuple[str, float]]]


MODEL_CALLS: Dict[str, ModelCall] = {
    "Gemma3": call_fine_tuned_gemma3,
    "Gemini-2.5-Pro": call_gemini_pro,
    "CodeLlama-Sim": call_code_specialist,
    "MathHammer": call_math_reasoner,
}

MODEL_COSTS: Dict[str, float] = {
    "Gemma3": settings.gemma_cost,
    "Gemini-2.5-Pro": settings.gemini_cost,
    "CodeLlama-Sim": settings.code_model_cost,
    "MathHammer": settings.math_model_cost,
}


async def _invoke_model(model_name: str, prompt: str) -> Tuple[str, float]:
    call = MODEL_CALLS.get(model_name)
    if call is None:
        raise ValueError(f"Unknown model {model_name}")
    return await call(prompt)


async def get_ai_response(prompt: str, decision: RoutingDecision) -> Tuple[str, str, float, float, bool]:
    """Execute the routing decision with optional fallback."""
    models = [decision.target_model]
    if decision.fallback_model:
        models.append(decision.fallback_model)

    last_error: Optional[Exception] = None
    for idx, model_name in enumerate(models):
        try:
            response, latency = await _invoke_model(model_name, prompt)
            cost = MODEL_COSTS.get(model_name, settings.gemini_cost)
            return response, model_name, latency, cost, idx > 0
        except Exception as exc:  # pragma: no cover - safety net
            logger.warning("Model %s failed with %s", model_name, exc)
            last_error = exc
            continue

    raise RuntimeError(f"All models failed for prompt: {prompt[:30]}...") from last_error
