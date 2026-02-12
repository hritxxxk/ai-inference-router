"""
Module for classifying the complexity of AI prompts using embeddings and heuristics.
"""

from typing import Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.token_counter import count_tokens
from src.config import settings

# Initialize model for complexity embedding
# In production, this would be a shared service or managed via dependency injection
model = SentenceTransformer(settings.embedding_model_name)

# Reference embeddings for complex reasoning tasks
COMPLEX_SAMPLES = [
    "Analyze the financial report and identify key risks",
    "Explain the step-by-step process of quantum entanglement",
    "Compare and contrast the industrial revolution with the digital age",
    "Summarize this 50-page legal document highlighting liabilities",
    "Reason through the ethical implications of artificial intelligence"
]
complex_embeddings = model.encode(COMPLEX_SAMPLES)

def classify_complexity(prompt: str) -> Tuple[str, dict]:
    """
    Classify prompt complexity using heuristics and embedding similarity.
    """
    # 1. Heuristic: Token count
    token_count = count_tokens(prompt)
    is_high_token_count = token_count > settings.token_threshold
    
    # 2. Heuristic: Keyword matching
    complex_keywords = [
        "analyze", "extract", "compare", "reason", "why", "how", "explain", 
        "summarize", "evaluate", "assess", "determine", "investigate", 
        "examine", "interpret", "outline", "describe", "discuss"
    ]
    has_complex_keywords = any(keyword in prompt.lower() for keyword in complex_keywords)
    
    # 3. Embedding-based complexity detection
    prompt_embedding = model.encode([prompt])
    
    # Calculate cosine similarity with complex samples
    similarities = np.dot(prompt_embedding, complex_embeddings.T) / (
        np.linalg.norm(prompt_embedding) * np.linalg.norm(complex_embeddings, axis=1)
    )
    max_similarity = float(np.max(similarities))
    
    # A prompt is HIGH complexity if it's long, has keywords, OR is semantically similar to complex tasks
    # We use a lower threshold for embedding similarity to catch "reasoning-like" prompts
    is_semantically_complex = max_similarity > 0.6 
    
    complexity = "HIGH" if (is_high_token_count or has_complex_keywords or is_semantically_complex) else "LOW"
    
    metadata = {
        "token_count": token_count,
        "has_complex_keywords": has_complex_keywords,
        "is_high_token_count": is_high_token_count,
        "semantic_complexity_score": round(max_similarity, 3),
        "is_semantically_complex": is_semantically_complex
    }
    
    return complexity, metadata
