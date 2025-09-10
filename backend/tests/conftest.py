"""
Test configuration and fixtures
"""
import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.models.tea_harvest_model import TeaHarvestModel


@pytest.fixture
def mock_model():
    """Mock model for testing"""
    mock_model = Mock(spec=TeaHarvestModel)
    mock_model.predict.return_value = [[30.0]]  # Mock prediction result
    
    # Mock the preprocess_data method to return valid data
    def mock_preprocess_data(df):
        import numpy as np
        # Return mock preprocessed data with correct shape
        X = np.random.rand(1, 30, 6).astype('float32')
        y = np.array([30.0])
        return X, y
    
    mock_model.preprocess_data = mock_preprocess_data
    return mock_model


@pytest.fixture
def client_with_mock_model(mock_model):
    """Test client with mocked model"""
    # Patch the model at the module level before creating the client
    with patch('app.main.model', mock_model):
        # Also patch the global model variable
        import app.main
        original_model = app.main.model
        app.main.model = mock_model
        try:
            with TestClient(app) as test_client:
                yield test_client
        finally:
            app.main.model = original_model


@pytest.fixture
def client_without_model():
    """Test client without model (for testing 503 errors)"""
    with TestClient(app) as test_client:
        yield test_client
