import time
import logging
from fastapi import FastAPI, HTTPException, Depends
from src.models.schemas import AIRequest, AIResponse
from src.services.quota_store import InMemoryQuotaStore
from src.services.semantic_cache import semantic_cache_lookup, semantic_cache_store
from src.services.classifier import classify_complexity
from src.services.fallback_handler import get_ai_response
from src.services.response_aggregator import build_response_aggregator
from src.middleware.timing_middleware import TimingMiddleware


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Inference Router")

# Add timing middleware
app.add_middleware(TimingMiddleware)

# Initialize services
quota_manager = InMemoryQuotaStore()


@app.post("/generate", response_model=AIResponse)
async def handle_request(request: AIRequest):
    logger.info(f"Received request from client {request.client_id}")
    start_time = time.perf_counter()
    
    # 1. Quota Check
    if not quota_manager.check_quota(request.client_id):
        logger.warning(f"Client {request.client_id} exceeded quota limit")
        raise HTTPException(status_code=429, detail="Quota Exceeded")
    
    logger.info(f"Quota check passed for client {request.client_id}")
    
    # 2. Optimization Layer: Semantic Cache
    cached_result = await semantic_cache_lookup(request.prompt)
    if cached_result:
        latency = (time.perf_counter() - start_time) * 1000
        logger.info(f"Cache hit for client {request.client_id}, latency: {latency:.2f}ms")
        return AIResponse(
            result=cached_result,
            model_used="CACHE",
            latency_ms=latency,
            cached=True,
            cost_estimate=0.0,
            metadata={
                "tokens": len(request.prompt.split()),  # Approximate token count
                "latency_ms": latency,
                "optimization_strategy": "Semantic Cache Hit",
                "cost_avoided_usd": 0.014,  # Approximate cost avoided
                "savings_multiplier": "N/A",
                "complexity_analysis": {"token_count": len(request.prompt.split())}
            }
        )

    logger.info(f"No cache hit for client {request.client_id}, proceeding with classification")
    
    # 3. Complexity Classification
    complexity, complexity_metadata = classify_complexity(request.prompt)
    logger.info(f"Prompt classified as {complexity} complexity for client {request.client_id}")
    
    # 4. Model Selection and Execution with Fallback
    logger.info(f"Selecting model for {complexity} complexity request from client {request.client_id}")
    result, model_name, model_latency, cost = await get_ai_response(
        request.prompt, 
        complexity
    )
    
    # 5. Store in cache if not from cache
    semantic_cache_store(request.prompt, result)
    logger.info(f"Response stored in cache for client {request.client_id}")
    
    # 6. Calculate final metrics
    total_latency = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
    
    # 7. Build response with comprehensive metadata
    optimization_strategy = f"Routed to {complexity} Complexity"
    response = build_response_aggregator(
        result=result,
        model_used=model_name,
        latency_ms=total_latency,
        cached=False,
        prompt=request.prompt,
        cost_estimate=cost,
        complexity_metadata=complexity_metadata,
        optimization_strategy=optimization_strategy
    )
    
    logger.info(
        f"Request completed for client {request.client_id}, "
        f"model: {model_name}, latency: {total_latency:.2f}ms, "
        f"cost: ${cost:.4f}"
    )
    
    # Return the response
    return AIResponse(**response)


# Health check endpoint
@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "healthy"}