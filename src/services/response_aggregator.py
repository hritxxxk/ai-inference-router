"""
Module for aggregating AI responses with comprehensive metadata.

This module builds the final response object that includes not just the
AI-generated content but also valuable metadata about the processing,
such as cost savings, latency, and optimization strategies applied.
This information is crucial for analytics, billing, and performance monitoring.
"""

from typing import Dict, Any
from src.utils.token_counter import estimate_total_tokens
from src.config import settings


def build_response_aggregator(
    result: str,
    model_used: str,
    latency_ms: float,
    cached: bool,
    prompt: str,
    cost_estimate: float,
    routing_metadata: Dict[str, Any],
    optimization_strategy: str
) -> Dict[str, Any]:
    """
    Build a comprehensive response with metadata including cost savings.
    
    This function aggregates the AI response with important metadata that
    provides insights into the processing, including:
    - Performance metrics (latency, token count)
    - Cost information and savings
    - Applied optimization strategies
    - Complexity analysis
    
    Args:
        result: The AI-generated result
        model_used: Name of the model that generated the result
        latency_ms: Latency in milliseconds
        cached: Whether the response was retrieved from cache
        prompt: Original input prompt
        cost_estimate: Estimated cost of the operation
        complexity_metadata: Metadata from complexity classification
        optimization_strategy: Strategy used for optimization
        
    Returns:
        Dictionary containing the complete response with metadata
    """
    # Estimate total tokens (input + output) for billing and analytics
    total_tokens = estimate_total_tokens(prompt, result)
    
    # Calculate cost savings if applicable using configurable values
    gemini_cost = settings.gemini_cost
    cost_savings_usd = max(0.0, gemini_cost - cost_estimate)
    savings_multiplier = 1.0
    if cost_estimate > 0 and cost_estimate < gemini_cost:
        savings_multiplier = round(gemini_cost / cost_estimate, 1)
    
    # Build the comprehensive response object
    response = {
        "result": result,
        "model_used": model_used,
        "latency_ms": latency_ms,
        "cached": cached,
        "cost_estimate": cost_estimate,
        "metadata": {
            "tokens": total_tokens,
            "latency_ms": latency_ms,
            "optimization_strategy": optimization_strategy,
            "cost_avoided_usd": cost_savings_usd,
            "savings_multiplier": f"{savings_multiplier}x cheaper" if savings_multiplier > 1 else "Standard cost",
            "complexity_analysis": {
                "task_type": routing_metadata.get("task_type"),
                "router_confidence": routing_metadata.get("confidence"),
                "router_features": routing_metadata.get("features"),
                "fallback_used": routing_metadata.get("fallback_used", False),
                "reasons": routing_metadata.get("reasons", []),
            }
        }
    }

    return response
