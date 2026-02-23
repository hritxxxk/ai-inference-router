"""Prompt task-type and complexity analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import numpy as np
import asyncio
import re

from src.utils.token_counter import count_tokens
from src.services.embedding_provider import get_embedding_model
from src.config import settings


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)


@dataclass
class PromptAnalysis:
    token_count: int
    keyword_signals: Dict[str, float]
    semantic_signals: Dict[str, float]
    dominant_task: str
    complexity_score: float

    @property
    def normalized_tokens(self) -> float:
        return min(self.token_count / 512.0, 1.5)


class TaskAnalyzer:
    """Analyze prompts for downstream routing decisions."""

    def __init__(self) -> None:
        self._model = get_embedding_model()
        self._prototype_embeddings: Dict[str, np.ndarray] = {}
        self._task_keywords = settings.task_keywords
        self._task_prototypes = settings.task_prototypes
        self._keyword_patterns: Dict[str, List[re.Pattern[str]]] = {
            task: [self._compile_pattern(term) for term in terms]
            for task, terms in self._task_keywords.items()
        }
        self._prepare_prototypes()

    def _compile_pattern(self, term: str) -> re.Pattern[str]:
        escaped = re.escape(term.strip())
        if all(ch.isalnum() or ch == "_" for ch in term.replace(" ", "")):
            pattern = rf"\b{escaped}\b"
        else:
            pattern = escaped
        return re.compile(pattern, re.IGNORECASE)

    def _prepare_prototypes(self) -> None:
        for task, prompts in self._task_prototypes.items():
            vectors = self._model.encode(prompts)
            self._prototype_embeddings[task] = np.array(vectors)

    def _keyword_signal(self, prompt: str, token_count: int) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        length_factor = min(1.0, max(token_count, 1) / 128.0)
        for task, patterns in self._keyword_patterns.items():
            hits = sum(1 for pattern in patterns if pattern.search(prompt))
            scores[task] = (hits / max(len(patterns), 1)) * length_factor
        return scores

    def _semantic_signal(self, prompt: str, embedding: Optional[Sequence[float]]) -> Dict[str, float]:
        if embedding is None:
            vector = self._model.encode([prompt])[0]
        else:
            vector = np.asarray(embedding, dtype=np.float32)
        signals: Dict[str, float] = {}
        for task, vectors in self._prototype_embeddings.items():
            sims = [_cosine_similarity(vector, row) for row in vectors]
            signals[task] = max(sims) if sims else 0.0
        return signals

    def analyze_sync(self, prompt: str, embedding: Optional[Sequence[float]] = None) -> PromptAnalysis:
        token_count = count_tokens(prompt)
        keyword_scores = self._keyword_signal(prompt, token_count)
        semantic_scores = self._semantic_signal(prompt, embedding)

        combined_scores: Dict[str, float] = {}
        for task in self._task_prototypes:
            combined_scores[task] = 0.55 * semantic_scores.get(task, 0.0) + 0.45 * keyword_scores.get(task, 0.0)

        dominant_task = max(combined_scores, key=combined_scores.get, default="reasoning")
        complexity_score = min(1.0, 0.5 * (token_count / 256.0) + max(combined_scores.values() or [0.0]))

        return PromptAnalysis(
            token_count=token_count,
            keyword_signals=keyword_scores,
            semantic_signals=semantic_scores,
            dominant_task=dominant_task,
            complexity_score=complexity_score,
        )

    async def analyze(self, prompt: str, embedding: Optional[Sequence[float]] = None) -> PromptAnalysis:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.analyze_sync, prompt, embedding)


_analyzer: TaskAnalyzer | None = None


def get_task_analyzer() -> TaskAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = TaskAnalyzer()
    return _analyzer
