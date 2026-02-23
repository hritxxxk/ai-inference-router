import json
import uuid

import pytest

from src.config import settings
from src.services import fallback_handler
from src.services.classifier import classify_complexity
from src.services.model_calls import (
    call_code_specialist,
    call_fine_tuned_gemma3,
    call_gemini_pro,
)
from src.services.quota_store import InMemoryQuotaStore
from src.services.routing_engine import RoutingDecision, get_task_router
from src.services.semantic_cache import SemanticCache
from src.services.telemetry_store import TelemetryStore
from src.services.weight_provider import WeightProvider
from src.utils.token_counter import count_tokens, estimate_total_tokens
from scripts.train_router import train_router

PREMIUM_MODEL = settings.gemini_model_name
LOW_MODEL = settings.gemma_model_name
CODE_MODEL = settings.code_model_name


@pytest.fixture
def sample_prompt_short():
    return "Hello, how are you?"


@pytest.fixture
def sample_prompt_long():
    return "This is a much longer prompt that exceeds 128 tokens and should be classified as high complexity due to its length. " * 10


@pytest.fixture
def sample_prompt_reasoning():
    return "Analyze the following data and explain the trends you see."


def test_count_tokens():
    """Test token counting utility"""
    text = "Hello world!"
    token_count = count_tokens(text)
    assert isinstance(token_count, int)
    assert token_count > 0


def test_estimate_total_tokens():
    """Test total token estimation"""
    input_text = "Hello"
    output_text = "Hi there"
    total_tokens = estimate_total_tokens(input_text, output_text)
    assert isinstance(total_tokens, int)
    assert total_tokens > 0


def test_quota_store_basic():
    """Test basic quota store functionality"""
    quota_store = InMemoryQuotaStore()
    
    # Test initial state
    assert quota_store.check_quota("client_001", increment=False) is True
    
    # Consume quota
    assert quota_store.check_quota("client_001") is True  # First request
    # Set usage to limit to test quota exceeded
    quota_store._usage["client_001"] = quota_store._limits["client_001"]
    
    # Should fail now
    assert quota_store.check_quota("client_001", increment=False) is False


def test_semantic_cache():
    """Test semantic cache functionality"""
    cache = SemanticCache()
    
    prompt1 = "What is the weather today?"
    response1 = "The weather is sunny."
    
    # Initially should not find anything
    assert cache.lookup(prompt1) is None
    
    # Store a response
    cache.store(prompt1, response1)
    
    # Should find the stored response
    assert cache.lookup(prompt1) == response1


def test_classify_complexity_short_prompt():
    """Test complexity classification for short prompts"""
    prompt = "Hi"
    complexity, metadata = classify_complexity(prompt)
    
    assert complexity == "LOW"
    assert "token_count" in metadata
    assert "has_complex_keywords" in metadata


def test_classify_complexity_with_reasoning_keyword():
    """Test complexity classification with reasoning keywords"""
    prompt = "Analyze this data"
    complexity, metadata = classify_complexity(prompt)
    
    assert complexity == "HIGH"
    assert metadata["has_complex_keywords"] is True


def test_classify_complexity_long_prompt():
    """Test complexity classification for long prompts"""
    long_prompt = "This is a long prompt " * 50  # This should exceed 128 tokens
    complexity, metadata = classify_complexity(long_prompt)
    
    assert complexity == "HIGH"
    assert metadata["is_high_token_count"] is True


@pytest.mark.asyncio
async def test_gemini_pro_call():
    """Test Gemini 3.1 Pro call (falls back to simulation when offline)."""
    prompt = "Test prompt"
    response, latency = await call_gemini_pro(prompt)

    assert isinstance(response, str)
    assert response.strip()
    assert latency > 0


@pytest.mark.asyncio
async def test_gemma3_call():
    """Test Gemma 3 call."""
    prompt = "Test prompt"
    response, latency = await call_fine_tuned_gemma3(prompt)

    assert isinstance(response, str)
    assert response.strip()
    assert latency > 0

@pytest.mark.asyncio
async def test_code_specialist_call():
    prompt = "def foo(x): return x * 2"
    response, latency = await call_code_specialist(prompt)
    assert isinstance(response, str)
    assert response.strip()
    assert latency > 0


@pytest.mark.asyncio
async def test_task_router_detects_code_prompt():
    router = get_task_router()
    decision = await router.route("def bug():\n    return 42")
    assert decision.target_model == CODE_MODEL
    assert "code" in decision.task_type or decision.features.get("code_signal") > 0


@pytest.mark.asyncio
async def test_task_router_high_complexity():
    router = get_task_router()
    prompt = "Explain in detail how to design a fault tolerant distributed database with consensus."
    decision = await router.route(prompt)
    assert decision.target_model in {PREMIUM_MODEL, LOW_MODEL}
    assert decision.features["reasoning_signal"] > 0.1


@pytest.mark.asyncio
async def test_get_ai_response_low_complexity():
    decision = RoutingDecision(
        target_model=LOW_MODEL,
        fallback_model=PREMIUM_MODEL,
        confidence=0.7,
        task_type="reasoning",
        reasons=[],
        features={"complexity_score": 0.2},
    )
    response, model_name, latency, cost, fallback_used = await fallback_handler.get_ai_response("Short prompt", decision)
    assert isinstance(response, str)
    assert model_name == LOW_MODEL
    assert latency > 0
    assert cost > 0
    assert fallback_used is False


@pytest.mark.asyncio
async def test_get_ai_response_with_fallback(monkeypatch):
    decision = RoutingDecision(
        target_model=LOW_MODEL,
        fallback_model=PREMIUM_MODEL,
        confidence=0.5,
        task_type="reasoning",
        reasons=[],
        features={"complexity_score": 0.9},
    )

    async def _fail(_prompt: str):  # pragma: no cover - forced failure for test
        raise RuntimeError("forced failure")

    monkeypatch.setitem(fallback_handler.MODEL_CALLS, LOW_MODEL, _fail)
    response, model_name, latency, cost, fallback_used = await fallback_handler.get_ai_response("Need reliability", decision)
    assert model_name == PREMIUM_MODEL
    assert fallback_used is True
    assert cost > 0


def test_weight_provider_hot_reload(tmp_path):
    weights_path = tmp_path / "weights.json"
    provider = WeightProvider(str(weights_path), {LOW_MODEL: {"bias": 0.1}})
    heads = provider.get_heads()
    assert heads[LOW_MODEL]["bias"] == 0.1

    weights_path.write_text(json.dumps({LOW_MODEL: {"bias": 0.9}}))
    provider.reload(force=True)
    assert provider.get_heads()[LOW_MODEL]["bias"] == 0.9


def test_training_script_generates_weights(tmp_path):
    db_path = tmp_path / "router.db"
    store = TelemetryStore(str(db_path))
    request_id = str(uuid.uuid4())
    store.persist_decision(
        request_id,
        "trainer",
        "Preview",
        "hash",
        {
            "target_model": LOW_MODEL,
            "fallback_model": PREMIUM_MODEL,
            "confidence": 0.5,
            "task_type": "reasoning",
            "features": {"normalized_tokens": 0.2, "complexity_score": 0.3},
            "reasons": ["unit-test"],
        },
        embedding=[0.1, 0.2],
    )
    store.record_feedback(request_id, "correct", None, "tester", None, 5)

    output_path = tmp_path / "weights.json"
    result = train_router(str(db_path), str(output_path))
    assert result.exists()
    payload = json.loads(result.read_text())
    assert LOW_MODEL in payload
