import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check test passed")


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    print("✅ Root endpoint test passed")


def test_predict_endpoint():
    """Test prediction endpoint"""
    payload = {
        "product_name": "Test Product",
        "category": "electronics",
        "initial_sales": 100,
        "sentiment_score": 0.8,
        "social_mentions": 500,
        "instagram_posts": 50,
        "twitter_mentions": 100,
        "tiktok_videos": 200,
        "engagement_rate": 0.1,
        "conversion_rate": 0.05,
        "search_volume": 1000,
        "views": 5000,
        "clicks": 500
    }
    
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "virality_score" in data
    assert "is_viral" in data
    assert "recommendation" in data
    print("✅ Predict endpoint test passed")


def test_feature_importance_endpoint():
    """Test feature importance endpoint"""
    response = client.get("/api/v1/feature-importance")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    print("✅ Feature importance endpoint test passed")


if __name__ == "__main__":
    test_health_check()
    test_root_endpoint()
    test_predict_endpoint()
    test_feature_importance_endpoint()
    print("\n✅ All API tests passed!")
