"""
Module for simulating AI model calls in the inference router.

This module contains mock implementations of calls to different AI models
with varying computational complexity and response times. These functions
simulate the behavior of real API calls to expensive (Gemini-2.5-Pro) and 
inexpensive (Gemma3) models, allowing for testing of the routing logic 
without incurring actual costs.
"""

import asyncio
import logging
from typing import Tuple


from typing import Tuple, AsyncGenerator


logger = logging.getLogger(__name__)


async def call_gemini_pro_stream(prompt: str) -> AsyncGenerator[str, None]:
    """
    Simulate streaming response from Gemini Pro.
    """
    full_response = f"Gemini-2.5-Pro (Streaming): Detailed analysis for '{prompt[:20]}...' "
    chunks = [full_response[i:i+5] for i in range(0, len(full_response), 5)]
    
    for chunk in chunks:
        await asyncio.sleep(0.1)  # Simulate chunk latency
        yield chunk


async def call_gemini_pro(prompt: str) -> Tuple[str, float]:
    """
    Mock function to simulate calling Gemini-2.5-Pro model.
    
    This represents an expensive, high-capability model suitable for complex
    reasoning tasks. The function simulates the latency and cost characteristics
    of such models by sleeping for 1.5 seconds and returning a detailed response.
    
    Args:
        prompt: Input prompt to send to the model
        
    Returns:
        Tuple of (response text, latency in seconds)
    """
    start_time = asyncio.get_event_loop().time()
    
    # Simulate high-reasoning latency typical of expensive models
    await asyncio.sleep(1.5)
    
    end_time = asyncio.get_event_loop().time()
    latency = end_time - start_time
    
    # Generate a mock response that indicates deep analysis
    response = f"Gemini-2.5-Pro: Deep analysis of '{prompt[:30]}...' (Token count: ~{len(prompt.split())})"
    
    logger.info(f"Gemini-2.5-Pro call completed with latency: {latency:.2f}s")
    
    return response, latency


async def call_fine_tuned_gemma3(prompt: str) -> Tuple[str, float]:
    """
    Mock function to simulate calling Fine-tuned Gemma3 model.
    
    This represents a lightweight, fast model suitable for simple queries.
    The function simulates the efficiency of such models by sleeping for
    only 0.4 seconds and returning a quick response.
    
    Args:
        prompt: Input prompt to send to the model
        
    Returns:
        Tuple of (response text, latency in seconds)
    """
    start_time = asyncio.get_event_loop().time()
    
    # Simulate low-latency response from efficient model
    await asyncio.sleep(0.4)
    
    end_time = asyncio.get_event_loop().time()
    latency = end_time - start_time
    
    # Generate a mock response that indicates quick processing
    response = f"Gemma3: Fast response for '{prompt[:30]}...' (Token count: ~{len(prompt.split())})"
    
    logger.info(f"Gemma3 call completed with latency: {latency:.2f}s")
    
    return response, latency