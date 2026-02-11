"""
Module for classifying the complexity of AI prompts.

This module contains logic to determine whether an incoming prompt requires
a computationally expensive model or can be handled by a lighter-weight model.
The classification is based on both token count and semantic indicators of
complexity, optimizing for both cost and performance.
"""

from typing import Tuple
from src.utils.token_counter import count_tokens
from src.config import settings


def classify_complexity(prompt: str) -> Tuple[str, dict]:
    """
    Classify the complexity of a prompt based on token count and reasoning keywords.
    
    This function determines whether a prompt should be routed to an expensive
    model (for complex reasoning) or a cheaper model (for simple queries).
    The decision is based on two factors:
    1. Token count: Prompts with more than the configured threshold tokens are considered complex
    2. Semantic indicators: Prompts containing reasoning keywords are complex
    
    Args:
        prompt: Input prompt to classify
        
    Returns:
        Tuple of (complexity level, metadata about classification)
        - complexity level: Either "HIGH" or "LOW"
        - metadata: Dictionary with details about the classification
    """
    # Define complex reasoning keywords that indicate need for sophisticated processing
    complex_keywords = [
        "analyze", "extract", "compare", "reason", "why", "how", "explain", 
        "summarize", "evaluate", "assess", "determine", "investigate", 
        "examine", "interpret", "outline", "describe", "discuss"
    ]
    
    # Count tokens using tiktoken for accurate complexity assessment
    # This is more reliable than character or word count for LLM pricing
    token_count = count_tokens(prompt)
    
    # Check for complex intent based on keywords
    has_complex_intent = any(keyword in prompt.lower() for keyword in complex_keywords)
    
    # Determine complexity based on token count and intent using configurable threshold
    is_high_token_count = token_count > settings.token_threshold
    
    # Classify as HIGH complexity if either condition is met
    complexity = "HIGH" if (is_high_token_count or has_complex_intent) else "LOW"
    
    # Generate metadata for the classification to enable debugging and analytics
    metadata = {
        "token_count": token_count,
        "has_complex_intent": has_complex_intent,
        "is_high_token_count": is_high_token_count,
        "complex_keywords_found": [kw for kw in complex_keywords if kw in prompt.lower()]
    }
    
    return complexity, metadata