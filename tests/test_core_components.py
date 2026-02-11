import pytest
import asyncio
from src.utils.token_counter import count_tokens, estimate_total_tokens
from src.services.quota_store import InMemoryQuotaStore
from src.services.semantic_cache import SemanticCache, SIMILARITY_THRESHOLD
from src.services.classifier import classify_complexity
from src.services.model_calls import call_gemini_pro, call_fine_tuned_gemma3
from src.services.fallback_handler import get_ai_response


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
    assert "has_complex_intent" in metadata


def test_classify_complexity_with_reasoning_keyword():
    """Test complexity classification with reasoning keywords"""
    prompt = "Analyze this data"
    complexity, metadata = classify_complexity(prompt)
    
    assert complexity == "HIGH"
    assert metadata["has_complex_intent"] is True


def test_classify_complexity_long_prompt():
    """Test complexity classification for long prompts"""
    long_prompt = "This is a long prompt " * 50  # This should exceed 128 tokens
    complexity, metadata = classify_complexity(long_prompt)
    
    assert complexity == "HIGH"
    assert metadata["is_high_token_count"] is True


@pytest.mark.asyncio
async def test_gemini_pro_call():
    """Test Gemini Pro mock call"""
    prompt = "Test prompt"
    response, latency = await call_gemini_pro(prompt)
    
    assert isinstance(response, str)
    assert "Gemini Pro:" in response
    assert latency > 0


@pytest.mark.asyncio
async def test_gemma3_call():
    """Test Gemma3 mock call"""
    prompt = "Test prompt"
    response, latency = await call_fine_tuned_gemma3(prompt)
    
    assert isinstance(response, str)
    assert "Gemma3:" in response
    assert latency > 0


@pytest.mark.asyncio
async def test_get_ai_response_low_complexity():
    """Test AI response with low complexity (should use Gemma3)"""
    prompt = "Short prompt"
    response, model_name, latency, cost = await get_ai_response(prompt, "LOW")
    
    assert isinstance(response, str)
    assert model_name in ["Gemma3", "Gemini-Pro (Fallback)"]  # Could be fallback if Gemma3 fails
    assert latency > 0
    assert cost >= 0


@pytest.mark.asyncio
async def test_get_ai_response_high_complexity():
    """Test AI response with high complexity (should use Gemini Pro)"""
    prompt = "Analyze this complex topic in detail"
    response, model_name, latency, cost = await get_ai_response(prompt, "HIGH")
    
    assert isinstance(response, str)
    assert model_name == "Gemini-Pro"
    assert latency > 0
    assert cost > 0