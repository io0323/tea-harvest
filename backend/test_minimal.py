#!/usr/bin/env python3
"""
Minimal test script - completely independent from TensorFlow
"""
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Set test environment
os.environ['PYTEST_CURRENT_TEST'] = 'true'

def test_minimal_api():
    """Minimal API test without any TensorFlow dependencies"""
    print("Starting minimal API test...")
    
    try:
        # Create a more complete numpy mock
        numpy_mock = MagicMock()
        numpy_mock.__version__ = "1.21.0"
        numpy_mock.array = lambda x, **kwargs: x
        numpy_mock.random = MagicMock()
        numpy_mock.random.rand = lambda *args: [[0.1] * args[1] if len(args) > 1 else [0.1] * args[0]]
        numpy_mock.float32 = "float32"
        
        # Mock all problematic modules
        sys.modules['numpy'] = numpy_mock
        sys.modules['tensorflow'] = MagicMock()
        sys.modules['keras'] = MagicMock()
        sys.modules['pandas'] = MagicMock()
        
        # Mock the model completely
        class MinimalMockModel:
            def predict(self, X):
                return [[30.0]]
            
            def preprocess_data(self, df):
                return [[1, 2, 3]], [30.0]
        
        mock_model = MinimalMockModel()
        
        # Patch before any imports
        with patch('app.main.model', mock_model):
            with patch('app.main.load_model', return_value=mock_model):
                # Import after patching
                from app.main import app
                print("✓ App imported successfully")
                
                # Test with FastAPI TestClient
                from fastapi.testclient import TestClient
                client = TestClient(app)
                
                # Test root endpoint
                response = client.get("/")
                print(f"✓ Root endpoint: {response.status_code}")
                assert response.status_code == 200
                
                # Test health endpoint
                response = client.get("/health")
                print(f"✓ Health endpoint: {response.status_code}")
                assert response.status_code == 200
                
                # Test prediction endpoint with mock data
                test_data = {
                    "temperature": 25.0,
                    "humidity": 60.0,
                    "rainfall": 100.0,
                    "soil_moisture": 50.0,
                    "sunlight_hours": 8.0,
                    "wind_speed": 5.0
                }
                
                response = client.post("/predict", json=test_data)
                print(f"✓ Prediction endpoint: {response.status_code}")
                assert response.status_code == 200
                
                result = response.json()
                print(f"✓ Prediction result: {result}")
                assert "prediction" in result
                
        print("✅ All minimal tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal_api()
    sys.exit(0 if success else 1)