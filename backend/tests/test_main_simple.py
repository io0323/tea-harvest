"""
Main application tests - Simple version without complex mocking
"""
import pytest
import io
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


def test_read_root():
    """Test root endpoint"""
    # Mock the app and model
    with patch('app.main.model', None):
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200


def test_health_check():
    """Test health check endpoint"""
    # Mock the app and model
    with patch('app.main.model', None):
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_tea_harvest_prediction_without_model():
    """Test tea harvest prediction endpoint without model (expects 503)"""
    # Mock the app and model
    with patch('app.main.model', None):
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        
        # Create sample CSV data
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
        response = client.post(
            "/predict",
            files={"file": ("test_data.csv", csv_file, "text/csv")},
            data={"region": "静岡", "year": 2024}
        )
        
        # Should return 503 when model is not loaded
        assert response.status_code == 503
        response_data = response.json()
        assert "error" in response_data
        assert "予測モデルが利用できません" in response_data["error"]


def test_tea_harvest_prediction_with_mock_model():
    """Test tea harvest prediction endpoint with mock model"""
    # Create mock model
    class MockModel:
        def predict(self, X):
            return np.array([[30.0]], dtype=np.float32)
        
        def preprocess_data(self, df):
            X = np.random.rand(1, 30, 6).astype(np.float32)
            y = np.array([30.0], dtype=np.float32)
            return X, y
    
    mock_model = MockModel()
    
    # Mock file operations
    def mock_open(file_path, mode='r'):
        if 'b' in mode:
            return MagicMock(read=MagicMock(return_value=b'mock file content'))
        else:
            return MagicMock(read=MagicMock(return_value='mock file content'))
    
    def mock_read_csv(file_path):
        return pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=30, freq='D'),
            'temperature': [25.0] * 30,
            'humidity': [60.0] * 30,
            'precipitation': [5.0] * 30,
            'sunshine': [8.0] * 30
        })
    
    # Mock the app and model
    with patch('app.main.model', mock_model):
        with patch('builtins.open', mock_open):
            with patch('pandas.read_csv', mock_read_csv):
                from fastapi.testclient import TestClient
                from app.main import app
                
                client = TestClient(app)
                
                # Create sample CSV data
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
                
                # Test file upload with mock model
                response = client.post(
                    "/predict",
                    files={"file": ("test_data.csv", csv_file, "text/csv")},
                    data={"region": "静岡", "year": 2024}
                )
                
                # With mock model, should return 200
                assert response.status_code == 200
                response_data = response.json()
                assert "predicted_date" in response_data
                assert "confidence_score" in response_data
