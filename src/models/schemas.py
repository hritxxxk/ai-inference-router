"""
Pydantic schemas for the AI inference router API.

This module defines the data structures for API requests and responses,
ensuring type safety and automatic validation. The schemas include
comprehensive metadata fields that provide insights into processing
metrics, costs, and optimization strategies.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any


class AIRequest(BaseModel):
    """
    Schema for incoming AI requests.
    
    This schema validates the structure of incoming requests to the AI
    inference router, ensuring that required fields are present and
    correctly typed.
    """
    prompt: str
    """The input prompt to be processed by the AI model."""
    
    client_id: str
    """Unique identifier for the requesting client, used for quota management."""


class Metadata(BaseModel):
    """
    Schema for response metadata.
    
    This schema contains additional information about the AI processing,
    including performance metrics, cost analysis, and optimization details.
    This data is valuable for analytics, billing, and performance monitoring.
    """
    tokens: int
    """Total number of tokens in the input and output combined."""
    
    latency_ms: float
    """Processing time in milliseconds."""
    
    optimization_strategy: str
    """Description of the optimization strategy applied (e.g., 'Semantic Cache Hit', 'Routed to LOW Complexity')."""
    
    cost_avoided_usd: float
    """Estimated cost avoided by using optimizations (in USD)."""
    
    savings_multiplier: str
    """Multiplier indicating how much cheaper the operation was (e.g., '15x cheaper')."""
    
    complexity_analysis: Dict[str, Any]
    """Detailed analysis of the complexity classification, including token count and detected keywords."""


class AIResponse(BaseModel):
    """
    Schema for outgoing AI responses.
    
    This schema defines the structure of responses from the AI inference
    router, including the AI-generated content and comprehensive metadata
    about the processing.
    """
    result: str
    """The AI-generated response content."""
    
    model_used: str
    """Name of the model that processed the request (e.g., 'Gemma3', 'Gemini-Pro', 'CACHE')."""
    
    latency_ms: float
    """Total processing time in milliseconds."""
    
    cached: bool
    """Whether the response was retrieved from cache."""
    
    cost_estimate: float
    """Estimated cost of the operation in USD."""
    
    metadata: Metadata
    """Additional metadata about the processing, including performance and cost metrics."""