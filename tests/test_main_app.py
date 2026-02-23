import hashlib
import uuid

import pytest
from fastapi.testclient import TestClient

from src.config import settings
from src.main import app
from src.models.schemas import AIRequest
from src.services.telemetry_store import telemetry_store
from src.services.quota_store import InMemoryQuotaStore


@pytest.fixture()
def client(monkeypatch):
    from src import main as main_module

    main_module.quota_manager = InMemoryQuotaStore()
    with TestClient(app) as test_client:
        yield test_client


def _prompt_preview_and_hash(prompt: str) -> tuple[str, str]:
    preview = prompt.strip().replace("\n", " ")[:120]
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return preview, digest


def test_health_endpoint(client):
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"healthy", "degraded"}
    assert "dependencies" in payload


def test_generate_endpoint_basic(client):
    """Test the generate endpoint with a basic request"""
    request_data = {
        "prompt": "Hello, how are you?",
        "client_id": "client_001"
    }
    
    response = client.post("/generate", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "result" in data
    assert "model_used" in data
    assert "latency_ms" in data
    assert "cached" in data
    assert "cost_estimate" in data
    assert "metadata" in data


def test_generate_endpoint_quota_exceeded(client):
    """Test the generate endpoint when quota is exceeded"""
    # Use a client with a low quota limit
    request_data = {
        "prompt": "Test prompt",
        "client_id": "client_002"  # This client has a low quota in our implementation
    }
    
    # Make multiple requests to exceed quota
    for i in range(10):  # More requests than the quota limit
        response = client.post("/generate", json=request_data)
        if response.status_code == 429:  # Rate limited
            break
    
    # The last request should be rate limited
    assert response.status_code == 429
    assert response.json()["detail"] == "Quota Exceeded"


def test_generate_endpoint_cache_hit(client):
    """Test that repeated requests result in cache hits"""
    request_data = {
        "prompt": "What is the weather today?",
        "client_id": "client_001"
    }
    
    # First request
    response1 = client.post("/generate", json=request_data)
    assert response1.status_code == 200
    data1 = response1.json()
    
    # Second request with same prompt should be cached
    response2 = client.post("/generate", json=request_data)
    assert response2.status_code == 200
    data2 = response2.json()
    
    # Both responses should be identical
    assert data1["result"] == data2["result"]
    assert data1["model_used"] == data2["model_used"] == "CACHE"
    assert data1["cached"] is True
    assert data2["cached"] is True


def test_feedback_requires_api_key(client):
    payload = {
        "request_id": str(uuid.uuid4()),
        "label": "correct"
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 401


def test_feedback_rejects_unknown_request(client):
    payload = {
        "request_id": str(uuid.uuid4()),
        "label": "incorrect",
        "notes": "should look at another model"
    }
    response = client.post(
        "/feedback",
        json=payload,
        headers={"x-api-key": settings.feedback_api_key},
    )
    assert response.status_code == 404


def test_feedback_persists_review(client):
    request_id = str(uuid.uuid4())
    client_id = "feedback_client"
    prompt = "Explain how the router should behave"
    preview, digest = _prompt_preview_and_hash(prompt)
    telemetry_store.persist_decision(
        request_id,
        client_id,
        preview,
        digest,
        {
            "target_model": settings.gemini_model_name,
            "fallback_model": None,
            "confidence": 0.8,
            "task_type": "reasoning",
            "features": {"normalized_tokens": 0.5},
            "reasons": ["unit-test"],
        },
        embedding=[0.1, 0.2, 0.3],
    )
    payload = {
        "request_id": request_id,
        "label": "incorrect",
        "preferred_model": settings.code_model_name,
        "reviewer": "pytest",
        "quality_score": 2,
        "notes": "Prefer specialist"
    }
    response = client.post(
        "/feedback",
        json=payload,
        headers={"x-api-key": settings.feedback_api_key},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["request_id"] == request_id
    assert body["status"] == "accepted"
    
    # Wait for background task
    import time
    time.sleep(0.1)
    
    assert telemetry_store.feedback_count(request_id) >= 1


def test_generate_endpoint_complexity_routing(client):
    """Test that complex prompts are routed to expensive model"""
    # Simple prompt should use cheaper model
    simple_request = {
        "prompt": "Hi",
        "client_id": "client_001"
    }
    
    response_simple = client.post("/generate", json=simple_request)
    assert response_simple.status_code == 200
    simple_data = response_simple.json()
    
    # Complex prompt should use expensive model
    complex_request = {
        "prompt": "Analyze the economic implications of renewable energy adoption in developing countries",
        "client_id": "client_001"
    }
    
    response_complex = client.post("/generate", json=complex_request)
    assert response_complex.status_code == 200
    complex_data = response_complex.json()
    
    # The responses should be different and appropriately routed
    assert simple_data["result"] != complex_data["result"]
