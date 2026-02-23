"""
Module for semantic caching in the AI inference router using ChromaDB and Sentence Transformers.

This module implements a persistent semantic cache that uses vector embeddings
to store and retrieve responses based on semantic similarity.
"""

import asyncio
import chromadb
from chromadb.api.types import EmbeddingFunction
from typing import Optional, List, Dict, Any
from src.config import settings
import logging
import time

from src.services.embedding_provider import get_embedding_model

logger = logging.getLogger(__name__)


class _SharedEmbeddingFunction(EmbeddingFunction[List[str]]):
    """Chroma embedding function backed by the shared SentenceTransformer."""

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or settings.embedding_model_name
        self._model = get_embedding_model()

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A003 - signature required by Chroma
        embeddings = self._model.encode(input)
        return embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)

    def embed_query(self, input: str) -> List[float]:
        embedding = self._model.encode([input])[0]
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self.__call__(input)

    @staticmethod
    def name() -> str:
        return "shared-sentence-transformer"

    def get_config(self) -> Dict[str, Any]:
        return {"model_name": self._model_name}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "_SharedEmbeddingFunction":
        return _SharedEmbeddingFunction(config.get("model_name"))

    def default_space(self) -> str:
        return settings.semantic_distance_metric.lower()

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]

class SemanticCache:
    """
    Semantic cache using ChromaDB for vector storage and Sentence Transformers for embeddings.
    """
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_db_path)
        self.embedding_function = _SharedEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="semantic_cache",
            embedding_function=self.embedding_function
        )
        self.similarity_threshold = settings.similarity_threshold
        self.ttl_seconds = settings.cache_ttl_seconds
        self.distance_metric = settings.semantic_distance_metric.lower()
        self._fallback_store: Dict[str, str] = {}
    
    def lookup(self, prompt: str, embedding: Optional[List[float]] = None) -> Optional[str]:
        """
        Look up a similar prompt in ChromaDB.
        """
        try:
            query_kwargs = {
                "n_results": 1,
                "include": ["distances", "metadatas", "documents"],
            }
            if embedding is not None:
                query_kwargs["query_embeddings"] = [list(embedding)]
            else:
                query_kwargs["query_texts"] = [prompt]

            results = self.collection.query(**query_kwargs)
            
            ids = results.get('ids')
            if (
                not isinstance(ids, list)
                or not ids
                or not isinstance(ids[0], list)
                or not ids[0]
            ):
                return None

            result_id = ids[0][0]

            metadatas = results.get('metadatas')
            metadata = {}
            if (
                isinstance(metadatas, list)
                and metadatas
                and isinstance(metadatas[0], list)
                and metadatas[0]
            ):
                metadata = metadatas[0][0] or {}
            
            if self.ttl_seconds > 0:
                ts = metadata.get("timestamp")
                if ts and (time.time() - ts) > self.ttl_seconds:
                    self.collection.delete(ids=[result_id])
                    return None

            # Convert distance to similarity depending on metric
            distances = results.get('distances')
            if (
                not isinstance(distances, list)
                or not distances
                or not isinstance(distances[0], list)
                or not distances[0]
            ):
                return None

            distance = distances[0][0]
            if self.distance_metric == "cosine":
                similarity = 1 - distance
            else:  # l2 or others
                similarity = 1.0 / (1.0 + distance)
            
            if similarity >= self.similarity_threshold:
                documents = results.get('documents')
                if (
                    isinstance(documents, list)
                    and documents
                    and isinstance(documents[0], list)
                    and documents[0]
                ):
                    return documents[0][0]
        except Exception as e:
            logger.error(f"Semantic cache lookup failed: {e}")
        
        return self._fallback_store.get(prompt)
    
    def store(self, prompt: str, response: str, embedding: Optional[List[float]] = None) -> None:
        """
        Store a new prompt-response pair in ChromaDB.
        """
        try:
            # Use prompt hash or uuid as ID
            import hashlib
            prompt_id = hashlib.md5(prompt.encode()).hexdigest()
            
            metadata = {"prompt": prompt, "timestamp": time.time()}
            if embedding is None:
                # self.embedding_function returns List[List[float]]
                embedding_vector = self.embedding_function([prompt])[0]
            else:
                embedding_vector = list(embedding)
            self.collection.add(
                documents=[response],
                metadatas=[metadata],
                ids=[prompt_id],
                embeddings=[embedding_vector]
            )
            self._enforce_limits()
        except Exception as e:
            logger.error(f"Semantic cache store failed: {e}")
        finally:
            self._fallback_store[prompt] = response

    def _enforce_limits(self) -> None:
        if settings.max_cache_size <= 0:
            return
        try:
            total = self.collection.count()
            if total <= settings.max_cache_size:
                return
            overflow = total - settings.max_cache_size
            peek_size = min(max(overflow, 10), total)
            snapshots = self.collection.peek(limit=peek_size)
            zipped = list(zip(snapshots.get("ids", []), snapshots.get("metadatas", [])))
            if not zipped:
                return
            zipped.sort(key=lambda item: (item[1] or {}).get("timestamp", 0))
            ids_to_delete = [item[0] for item in zipped[:overflow]]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
        except Exception as exc:
            logger.warning("Failed to enforce cache limit: %s", exc)


# Global instance
semantic_cache = SemanticCache()

async def semantic_cache_lookup(prompt: str, embedding: Optional[List[float]] = None) -> Optional[str]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, semantic_cache.lookup, prompt, embedding)


def semantic_cache_store(prompt: str, response: str, embedding: Optional[List[float]] = None) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        semantic_cache.store(prompt, response, embedding)
    else:
        loop.create_task(asyncio.to_thread(semantic_cache.store, prompt, response, embedding))
