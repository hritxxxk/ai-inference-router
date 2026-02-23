import asyncio
import time
import logging
import uuid
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from src.models.schemas import AIRequest, AIResponse, FeedbackPayload, FeedbackResponse
from src.services.quota_store import RedisQuotaStore
from src.services.semantic_cache import semantic_cache_lookup, semantic_cache_store
from src.services.fallback_handler import get_ai_response
from src.services.model_calls import call_gemini_pro_stream
from src.services.response_aggregator import build_response_aggregator
from src.services.routing_engine import get_task_router
from src.services.embedding_provider import embed_text
from src.services.telemetry import log_routing_decision, log_routing_outcome, log_feedback_event
from src.services.telemetry_store import telemetry_store
from src.middleware.timing_middleware import TimingMiddleware
from src.utils.security import verify_feedback_api_key


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Inference Router")

# Add timing middleware
app.add_middleware(TimingMiddleware)

# Initialize services
quota_manager = RedisQuotaStore()
task_router = get_task_router()


@app.post("/generate", response_model=AIResponse)
async def handle_request(request: AIRequest):
    logger.info(f"Received request from client {request.client_id}")
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())
    
    # 1. Quota Check
    if not quota_manager.check_quota(request.client_id):
        logger.warning(f"Client {request.client_id} exceeded quota limit")
        raise HTTPException(status_code=429, detail="Quota Exceeded")
    
    logger.info(f"Quota check passed for client {request.client_id}")
    
    # 2. Shared prompt embedding
    prompt_embedding = await embed_text(request.prompt)

    # 3. Optimization Layer: Semantic Cache
    cached_result = await semantic_cache_lookup(request.prompt, prompt_embedding)
    if cached_result:
        latency = (time.perf_counter() - start_time) * 1000
        logger.info(f"Cache hit for client {request.client_id}, latency: {latency:.2f}ms")
        log_routing_outcome(
            request_id,
            request.client_id,
            request.prompt,
            {
                "cached": True,
                "model_used": "CACHE",
                "latency_ms": latency,
                "cost": 0.0,
                "fallback_used": False,
            },
        )
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
    
    # 4. Routing Decision
    decision = await task_router.route(request.prompt, request.client_id, prompt_embedding)
    logger.info(
        "Routing decision for client %s -> %s (confidence=%.2f)",
        request.client_id,
        decision.target_model,
        decision.confidence,
    )
    log_routing_decision(
        request_id,
        request.client_id,
        request.prompt,
        {
            "target_model": decision.target_model,
            "fallback_model": decision.fallback_model,
            "confidence": decision.confidence,
            "task_type": decision.task_type,
            "features": decision.features,
            "reasons": decision.reasons,
        },
        prompt_embedding,
    )
    
    # 5. Model Selection and Execution with Fallback
    logger.info(
        "Executing model %s with fallback %s for client %s",
        decision.target_model,
        decision.fallback_model,
        request.client_id,
    )
    result, model_name, model_latency, cost, fallback_used = await get_ai_response(
        request.prompt,
        decision
    )
    
    # 6. Store in cache asynchronously
    semantic_cache_store(request.prompt, result, prompt_embedding)
    logger.info(f"Response stored in cache for client {request.client_id}")
    
    # 7. Calculate final metrics
    total_latency = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
    
    # 8. Build response with comprehensive metadata
    optimization_strategy = f"{decision.task_type.upper()} via {model_name}"
    routing_metadata = {
        "task_type": decision.task_type,
        "confidence": decision.confidence,
        "features": decision.features,
        "fallback_used": fallback_used,
        "reasons": decision.reasons,
    }
    response = build_response_aggregator(
        result=result,
        model_used=model_name,
        latency_ms=total_latency,
        cached=False,
        prompt=request.prompt,
        cost_estimate=cost,
        routing_metadata=routing_metadata,
        optimization_strategy=optimization_strategy
    )
    log_routing_outcome(
        request_id,
        request.client_id,
        request.prompt,
        {
            "cached": False,
            "model_used": model_name,
            "latency_ms": total_latency,
            "model_latency_sec": model_latency,
            "cost": cost,
            "fallback_used": fallback_used,
            "routing_confidence": decision.confidence,
            "task_type": decision.task_type,
        },
    )
    
    logger.info(
        f"Request completed for client {request.client_id}, "
        f"model: {model_name}, latency: {total_latency:.2f}ms, "
        f"cost: ${cost:.4f}"
    )
    
    # Return the response
    return AIResponse(**response)


@app.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_feedback(
    payload: FeedbackPayload,
    _: None = Depends(verify_feedback_api_key),
):
    """Collect human labels for router decisions without blocking the event loop."""

    if not telemetry_store.decision_exists(payload.request_id):
        raise HTTPException(status_code=404, detail="Unknown request_id")

    log_feedback_event(payload.request_id, payload.model_dump())

    asyncio.create_task(
        asyncio.to_thread(
            telemetry_store.record_feedback,
            payload.request_id,
            payload.label,
            payload.preferred_model,
            payload.reviewer,
            payload.notes,
            payload.quality_score,
        )
    )

    return FeedbackResponse(status="accepted", request_id=payload.request_id)


@app.post("/generate-stream")
async def handle_request_stream(request: AIRequest):
    """
    Handle request with streaming response.
    Note: Metadata aggregation is limited in streaming mode.
    """
    logger.info(f"Received streaming request from client {request.client_id}")
    
    # Quota Check
    if not quota_manager.check_quota(request.client_id):
        raise HTTPException(status_code=429, detail="Quota Exceeded")
        
    return StreamingResponse(
        call_gemini_pro_stream(request.prompt),
        media_type="text/plain"
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Comprehensive health check verifying dependencies.
    """
    health_status = {
        "status": "healthy",
        "dependencies": {
            "redis": "connected",
            "chromadb": "connected"
        }
    }
    
    # Check Redis
    redis_client = getattr(quota_manager, "redis_client", None)
    if not redis_client:
        health_status["dependencies"]["redis"] = "not_initialized"
        health_status["status"] = "degraded"
    else:
        try:
            redis_client.ping()
        except Exception:
            health_status["dependencies"]["redis"] = "unreachable"
            health_status["status"] = "degraded"
            
    # Check ChromaDB (Semantic Cache)
    try:
        from src.services.semantic_cache import semantic_cache
        # Simple heartbeat or version check
        semantic_cache.client.heartbeat()
    except Exception:
        health_status["dependencies"]["chromadb"] = "unreachable"
        health_status["status"] = "degraded"
        
    return health_status
