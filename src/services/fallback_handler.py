"""
Module for handling AI model fallback logic in the inference router.

This module implements a fallback mechanism that ensures requests are
processed even when preferred (cheaper) models fail. When a request
is classified as low complexity, the system attempts to use a cheaper
model first, falling back to an expensive model if the cheaper option
fails. This maintains reliability while optimizing costs.
"""

import asyncio
import logging
from typing import Tuple
from src.services.model_calls import call_gemini_pro, call_fine_tuned_gemma3


logger = logging.getLogger(__name__)


async def get_ai_response(prompt: str, complexity: str) -> Tuple[str, str, float, float]:
    """
    Get AI response with fallback logic based on complexity and model availability.
    
    For LOW complexity requests, this function attempts to use the cheaper
    Gemma3 model first. If that fails, it falls back to the more expensive
    Gemini-2.5-Pro model. For HIGH complexity requests, it directly uses Gemini-2.5-Pro.
    
    This approach optimizes costs while maintaining reliability through
    fallback mechanisms.
    
    Args:
        prompt: Input prompt to process
        complexity: Complexity level ('HIGH' or 'LOW') as determined by classifier
        
    Returns:
        Tuple of (response, model_name, latency, cost)
        - response: The generated text response
        - model_name: Name of the model that processed the request
        - latency: Time taken to process the request in seconds
        - cost: Estimated cost of the operation in USD
    """
    if complexity == "LOW":
        try:
            # Attempt the cost-effective route first
            response, latency = await call_fine_tuned_gemma3(prompt)
            model_name = "Gemma3"
            cost = 0.001  # Cost in USD
            
            return response, model_name, latency, cost
        except Exception as e:
            logger.warning(f"Cheap model (Gemma3) failed: {e}. Falling back to Gemini-2.5-Pro.")
            
            # Fallback to expensive route if cheap fails
            # This ensures request completion even when preferred model is unavailable
            response, latency = await call_gemini_pro(prompt)
            model_name = "Gemini-2.5-Pro (Fallback)"
            cost = 0.015  # Cost in USD
            
            return response, model_name, latency, cost
    
    # For HIGH complexity, always use expensive model as it's required for quality
    response, latency = await call_gemini_pro(prompt)
    model_name = "Gemini-2.5-Pro"
    cost = 0.015  # Cost in USD
    
    return response, model_name, latency, cost