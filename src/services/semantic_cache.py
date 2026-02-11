"""
Module for semantic caching in the AI inference router.

This module implements a semantic cache that stores and retrieves responses
based on semantic similarity rather than exact matches. This allows for
efficient reuse of responses when users submit slightly different prompts
that have similar meaning, reducing both computational cost and response time.
"""

import random
from typing import Optional
from src.utils.token_counter import count_tokens
from src.config import settings


class SemanticCache:
    """
    Semantic cache that simulates vector database lookup using mocked cosine similarity.
    
    This implementation stores previous prompts and their responses to avoid
    recomputation of similar requests. In a production environment, this would
    interface with a vector database like Pinecone, Weaviate, or FAISS for
    efficient similarity search.
    """
    
    def __init__(self):
        """
        Initialize the semantic cache with an empty dictionary.
        """
        self.cache = {}
        self.similarity_threshold = settings.similarity_threshold
    
    def _calculate_mock_similarity(self, prompt1: str, prompt2: str) -> float:
        """
        Mock function to simulate cosine similarity calculation between two prompts.
        
        This is a simplified implementation that uses Jaccard similarity as a proxy
        for cosine similarity. In a real implementation, this would use vector
        embeddings and proper cosine similarity calculation.
        
        Args:
            prompt1: First prompt to compare
            prompt2: Second prompt to compare
            
        Returns:
            Mock similarity score between 0.0 and 1.0
        """
        # Calculate a mock similarity based on token overlap and length
        tokens1 = set(prompt1.lower().split())
        tokens2 = set(prompt2.lower().split())
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        # Jaccard similarity as a simple proxy for cosine similarity
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        # Add some randomness to simulate real-world similarity matching
        noise = random.uniform(-0.1, 0.1)
        similarity = max(0.0, min(1.0, jaccard_similarity + noise))
        
        return similarity
    
    def lookup(self, prompt: str) -> Optional[str]:
        """
        Look up a similar prompt in the cache.
        
        Iterates through cached prompts and calculates similarity to the input.
        Returns the cached response if a sufficiently similar prompt is found.
        
        Args:
            prompt: Input prompt to search for
            
        Returns:
            Cached response if similar prompt exists, None otherwise
        """
        for cached_prompt, cached_response in self.cache.items():
            similarity = self._calculate_mock_similarity(prompt, cached_prompt)
            
            if similarity >= self.similarity_threshold:
                return cached_response
        
        return None
    
    def store(self, prompt: str, response: str) -> None:
        """
        Store a new prompt-response pair in the cache.
        
        This method adds a new entry to the cache. In a production system,
        you might want to implement cache eviction policies to prevent
        unbounded growth.
        
        Args:
            prompt: Input prompt
            response: Generated response
        """
        # Implement basic size limiting to prevent unbounded growth
        if len(self.cache) >= settings.max_cache_size:
            # Simple eviction: remove first item (FIFO)
            if settings.eviction_policy == "FIFO":
                first_key = next(iter(self.cache))
                del self.cache[first_key]
            # TODO: Implement other eviction policies (LRU, TTL) as needed
        
        self.cache[prompt] = response


# Global instance for the POC
# In a production system, this would be managed by a dependency injection framework
semantic_cache = SemanticCache()


async def semantic_cache_lookup(prompt: str) -> Optional[str]:
    """
    Async wrapper for semantic cache lookup.
    
    This function provides an async interface to the semantic cache lookup
    functionality, maintaining compatibility with async request handlers.
    
    Args:
        prompt: Input prompt to search for
        
    Returns:
        Cached response if similar prompt exists, None otherwise
    """
    return semantic_cache.lookup(prompt)


def semantic_cache_store(prompt: str, response: str) -> None:
    """
    Store a new prompt-response pair in the cache.
    
    This function provides a synchronous interface to store responses in the
    semantic cache.
    
    Args:
        prompt: Input prompt
        response: Generated response
    """
    semantic_cache.store(prompt, response)