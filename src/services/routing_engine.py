"""Advanced routing engine selecting the best model per prompt/task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.services.task_analyzer import PromptAnalysis, get_task_analyzer
from src.services.weight_provider import WeightProvider, get_weight_provider
from src.config import settings


@dataclass
class RoutingDecision:
    target_model: str
    fallback_model: Optional[str]
    confidence: float
    task_type: str
    reasons: List[str]
    features: Dict[str, float]

    @property
    def is_high_complexity(self) -> bool:
        return self.features.get("complexity_score", 0.0) >= 0.55


class TaskRouter:
    """Score prompts against multiple routing heads and pick the best."""

    def __init__(self, weight_provider: WeightProvider | None = None) -> None:
        self._task_analyzer = get_task_analyzer()
        self._weight_provider = weight_provider or get_weight_provider()

    def _features_from_analysis(self, analysis: PromptAnalysis) -> Dict[str, float]:
        features = {
            "normalized_tokens": analysis.normalized_tokens,
            "complexity_score": analysis.complexity_score,
            "code_signal": analysis.keyword_signals.get("code", 0.0) * 0.5 + analysis.semantic_signals.get("code", 0.0),
            "math_signal": analysis.keyword_signals.get("math", 0.0) * 0.5 + analysis.semantic_signals.get("math", 0.0),
            "reasoning_signal": analysis.keyword_signals.get("reasoning", 0.0) * 0.5 + analysis.semantic_signals.get("reasoning", 0.0),
            "translation_signal": analysis.keyword_signals.get("translation", 0.0) * 0.5 + analysis.semantic_signals.get("translation", 0.0),
            "summarization_signal": analysis.keyword_signals.get("summarization", 0.0) * 0.5 + analysis.semantic_signals.get("summarization", 0.0),
        }
        return features

    def _score_head(self, head: Dict[str, float], features: Dict[str, float]) -> float:
        z = head.get("bias", 0.0)
        for name, weight in head.items():
            if name == "bias":
                continue
            z += features.get(name, 0.0) * weight
        return 1.0 / (1.0 + math.exp(-z))

    def _reason_for_head(self, model_name: str, analysis: PromptAnalysis, score: float) -> List[str]:
        reasons = [f"semantic task={analysis.dominant_task}", f"confidence={score:.2f}"]
        if analysis.token_count > 256:
            reasons.append("long_prompt")
        if model_name == "Gemma3" and analysis.complexity_score > 0.4:
            reasons.append("still within low-complexity threshold")
        if model_name == "Gemini-2.5-Pro" and analysis.complexity_score > 0.7:
            reasons.append("high reasoning demand")
        if model_name == "CodeLlama-Sim":
            reasons.append("code keywords detected")
        if model_name == "MathHammer":
            reasons.append("math signals detected")
        return reasons

    async def route(
        self,
        prompt: str,
        client_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> RoutingDecision:
        analysis = await self._task_analyzer.analyze(prompt, embedding)
        features = self._features_from_analysis(analysis)
        heads = self._weight_provider.get_heads() or settings.router_heads
        scores = {name: self._score_head(head, features) for name, head in heads.items()}
        target_model = max(scores, key=scores.get)
        confidence = scores[target_model]

        if target_model == "Gemma3":
            fallback = "Gemini-2.5-Pro"
        elif target_model == "Gemini-2.5-Pro":
            fallback = None
        elif target_model == "CodeLlama-Sim":
            fallback = "Gemma3"
        else:
            fallback = "Gemini-2.5-Pro"

        reasons = self._reason_for_head(target_model, analysis, confidence)

        return RoutingDecision(
            target_model=target_model,
            fallback_model=fallback,
            confidence=confidence,
            task_type=analysis.dominant_task,
            reasons=reasons,
            features=features,
        )


_router: TaskRouter | None = None


def get_task_router() -> TaskRouter:
    global _router
    if _router is None:
        _router = TaskRouter()
    return _router
