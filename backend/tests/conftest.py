"""
Test configuration and fixtures
"""
import pytest
from unittest.mock import Mock, patch
from app.main import app
from app.models.tea_harvest_model import TeaHarvestModel


@pytest.fixture
def mock_model():
    """Mock model for testing"""
    mock_model = Mock(spec=TeaHarvestModel)
    mock_model.predict.return_value = [[30.0]]  # Mock prediction result
    return mock_model


@pytest.fixture
def client_with_mock_model(mock_model):
    """Test client with mocked model"""
    with patch('app.main.model', mock_model):
        from fastapi.testclient import TestClient
        with TestClient(app) as test_client:
            yield test_client
