"""
Main application tests
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_read_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_tea_harvest_prediction():
    """Test tea harvest prediction endpoint"""
    # Sample test data
    test_data = {
        "temperature": 25.0,
        "humidity": 60.0,
        "rainfall": 5.0,
        "soil_moisture": 70.0
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
