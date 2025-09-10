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


def test_tea_harvest_prediction(client_without_model):
    """Test tea harvest prediction endpoint without model (expects 503)"""
    # Create sample CSV data
    import io
    import pandas as pd
    
    # Sample weather data for 30 days
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    test_data = pd.DataFrame({
        'date': dates,
        'temperature': [25.0] * 30,
        'humidity': [60.0] * 30,
        'precipitation': [5.0] * 30,
        'sunshine': [8.0] * 30
    })
    
    # Convert to CSV string
    csv_data = test_data.to_csv(index=False)
    csv_file = io.BytesIO(csv_data.encode('utf-8'))
    
    # Test file upload
    response = client_without_model.post(
        "/predict",
        files={"file": ("test_data.csv", csv_file, "text/csv")},
        data={"region": "静岡", "year": 2024}
    )
    
    # Should return 503 when model is not loaded
    assert response.status_code == 503
    response_data = response.json()
    assert "error" in response_data
    assert "予測モデルが利用できません" in response_data["error"]


def test_tea_harvest_prediction_with_mock_model(client_with_mock_model):
    """Test tea harvest prediction endpoint with mocked model"""
    import io
    import pandas as pd
    
    # Sample weather data for 30 days
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    test_data = pd.DataFrame({
        'date': dates,
        'temperature': [25.0] * 30,
        'humidity': [60.0] * 30,
        'precipitation': [5.0] * 30,
        'sunshine': [8.0] * 30
    })
    
    # Convert to CSV string
    csv_data = test_data.to_csv(index=False)
    csv_file = io.BytesIO(csv_data.encode('utf-8'))
    
    # Test file upload with mocked model
    response = client_with_mock_model.post(
        "/predict",
        files={"file": ("test_data.csv", csv_file, "text/csv")},
        data={"region": "静岡", "year": 2024}
    )
    
    # With mocked model, should return 200
    assert response.status_code == 200
    response_data = response.json()
    assert "predicted_date" in response_data
    assert "confidence_score" in response_data
