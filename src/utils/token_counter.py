"""
Utility module for token counting operations.

This module provides functions to accurately count tokens in text using tiktoken,
which is the same tokenization method used by many LLMs. This ensures accurate
cost estimation and complexity classification based on actual token counts rather
than simple character or word counts.
"""

import tiktoken
from typing import Union


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text using tiktoken.
    
    Uses the cl100k_base encoding which is standard for models like GPT-4,
    GPT-3.5-Turbo, and text-embedding-ada-002. This provides accurate token
    counts that align with how major LLM providers bill for usage.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        Number of tokens in the text
    """
    encoding = tiktoken.get_encoding("cl100k_base")  # Standard for GPT/Gemini
    return len(encoding.encode(text))


def estimate_total_tokens(input_text: str, output_text: str = "") -> int:
    """
    Estimate total tokens for input and output combined.
    
    This function is useful for calculating the total token consumption of an
    LLM interaction, which directly correlates to costs in most API pricing models.
    
    Args:
        input_text: Input text to tokenize
        output_text: Output text to tokenize (optional)
        
    Returns:
        Combined token count for input and output
    """
    input_tokens = count_tokens(input_text)
    output_tokens = count_tokens(output_text) if output_text else 0
    return input_tokens + output_tokens