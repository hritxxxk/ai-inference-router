"""Shared embedding model provider.

Loading sentence-transformer models is expensive and should only happen once per
process. This module exposes a cached factory so every service (classifier,
router, cache warmers) can request the same instance without triggering a new
download or duplicating GPU/CPU memory.
"""

from functools import lru_cache
from typing import List, Optional
import logging
import asyncio
from sentence_transformers import SentenceTransformer
from src.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Return a lazily-initialized, cached embedding model instance."""
    try:
        return SentenceTransformer(settings.embedding_model_name)
    except Exception as exc:
        logger.error("Failed to load embedding model %s: %s", settings.embedding_model_name, exc)
        raise


class EmbeddingHandle:
    """Lightweight wrapper for injecting the cached model."""

    def __init__(self) -> None:
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = get_embedding_model()
        return self._model


def get_embedding_handle() -> EmbeddingHandle:
    """Convenience helper to inject into routers or analyzers."""
    return EmbeddingHandle()


def embed_text_sync(text: str) -> List[float]:
    """Synchronous helper returning a single embedding vector."""
    model = get_embedding_model()
    vector = model.encode([text])[0]
    return vector.tolist()


async def embed_text(text: str) -> List[float]:
    """Async wrapper that moves embedding work off the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, embed_text_sync, text)
