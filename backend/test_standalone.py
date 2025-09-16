#!/usr/bin/env python3
"""
Standalone test - completely bypasses TensorFlow and model loading
"""
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Set test environment
os.environ['PYTEST_CURRENT_TEST'] = 'true'

def test_standalone_api():
    """Standalone API test - no TensorFlow dependencies at all"""
    print("Starting standalone API test...")
    
    try:
        # Create comprehensive mocks
        numpy_mock = MagicMock()
        numpy_mock.__version__ = "1.21.0"
        numpy_mock.array = lambda x, **kwargs: x
        numpy_mock.random = MagicMock()
        numpy_mock.random.rand = lambda *args: [[0.1] * args[1] if len(args) > 1 else [0.1] * args[0]]
        numpy_mock.float32 = "float32"
        
        pandas_mock = MagicMock()
        pandas_mock.DataFrame = MagicMock()
        pandas_mock.read_csv = MagicMock()
        
        tensorflow_mock = MagicMock()
        tensorflow_mock.keras = MagicMock()
        tensorflow_mock.keras.models = MagicMock()
        tensorflow_mock.keras.models.Sequential = MagicMock()
        tensorflow_mock.keras.layers = MagicMock()
        tensorflow_mock.keras.optimizers = MagicMock()
        tensorflow_mock.keras.callbacks = MagicMock()
        
        # Mock all modules before any imports
        sys.modules['numpy'] = numpy_mock
        sys.modules['pandas'] = pandas_mock
        sys.modules['tensorflow'] = tensorflow_mock
        sys.modules['tensorflow.keras'] = tensorflow_mock.keras
        sys.modules['tensorflow.keras.models'] = tensorflow_mock.keras.models
        sys.modules['tensorflow.keras.layers'] = tensorflow_mock.keras.layers
        sys.modules['tensorflow.keras.optimizers'] = tensorflow_mock.keras.optimizers
        sys.modules['tensorflow.keras.callbacks'] = tensorflow_mock.keras.callbacks
        
        # Mock the model class
        class StandaloneMockModel:
            def predict(self, X):
                return [[30.0]]
            
            def preprocess_data(self, df):
                return [[1, 2, 3]], [30.0]
            
            def save_model(self, path):
                pass
            
            def load_model(self, path):
                return self
        
        mock_model = StandaloneMockModel()
        
        # Mock the model loading function
        def mock_load_model(path):
            return mock_model
        
        # Patch everything before importing
        with patch('app.models.tea_harvest_model.TeaHarvestModel', StandaloneMockModel):
            with patch('app.main.load_model', mock_load_model):
                with patch('app.main.model', mock_model):
                    # Import after all patching
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
                    
        print("✅ All standalone tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_standalone_api()
    sys.exit(0 if success else 1)
