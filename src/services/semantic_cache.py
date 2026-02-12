"""
Module for semantic caching in the AI inference router using ChromaDB and Sentence Transformers.

This module implements a persistent semantic cache that uses vector embeddings
to store and retrieve responses based on semantic similarity.
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import Optional
from src.config import settings
import logging

logger = logging.getLogger(__name__)

class SemanticCache:
    """
    Semantic cache using ChromaDB for vector storage and Sentence Transformers for embeddings.
    """
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_db_path)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model_name
        )
        self.collection = self.client.get_or_create_collection(
            name="semantic_cache",
            embedding_function=self.embedding_function
        )
        self.similarity_threshold = settings.similarity_threshold
    
    def lookup(self, prompt: str) -> Optional[str]:
        """
        Look up a similar prompt in ChromaDB.
        """
        try:
            results = self.collection.query(
                query_texts=[prompt],
                n_results=1
            )
            
            if not results['ids'][0]:
                return None
            
            # ChromaDB distance: lower is more similar (0.0 is exact match)
            # For cosine similarity, threshold might need adjustment
            # distance = 1 - cosine_similarity
            distance = results['distances'][0][0]
            similarity = 1 - distance
            
            if similarity >= self.similarity_threshold:
                return results['documents'][0][0]
        except Exception as e:
            logger.error(f"Semantic cache lookup failed: {e}")
        
        return None
    
    def store(self, prompt: str, response: str) -> None:
        """
        Store a new prompt-response pair in ChromaDB.
        """
        try:
            # Use prompt hash or uuid as ID
            import hashlib
            prompt_id = hashlib.md5(prompt.encode()).hexdigest()
            
            self.collection.add(
                documents=[response],
                metadatas=[{"prompt": prompt}],
                ids=[prompt_id]
            )
        except Exception as e:
            logger.error(f"Semantic cache store failed: {e}")


# Global instance
semantic_cache = SemanticCache()

async def semantic_cache_lookup(prompt: str) -> Optional[str]:
    return semantic_cache.lookup(prompt)

def semantic_cache_store(prompt: str, response: str) -> None:
    semantic_cache.store(prompt, response)
