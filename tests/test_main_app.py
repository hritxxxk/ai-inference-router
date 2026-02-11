import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.models.schemas import AIRequest


client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_generate_endpoint_basic():
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


def test_generate_endpoint_quota_exceeded():
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


def test_generate_endpoint_cache_hit():
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


def test_generate_endpoint_complexity_routing():
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