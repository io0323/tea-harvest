"""
Test configuration and fixtures
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.fixture
def mock_model():
    """Provide a concrete dummy model used to bypass real TF model in tests."""
    import numpy as np

    class DummyModel:
        def predict(self, X):
            # Return shape (batch_size, 1)
            batch_size = X.shape[0] if hasattr(X, "shape") else 1
            return np.full((batch_size, 1), 30.0, dtype=np.float32)

        def preprocess_data(self, df):
            # Return (X, y) with correct shapes expected by the app/model
            X = np.random.rand(1, 30, 6).astype("float32")
            y = np.array([30.0], dtype=np.float32)
            return X, y

    return DummyModel()


@pytest.fixture
def client_with_mock_model(mock_model):
    """Test client with the dummy model injected before app startup."""
    # Patch the model before importing the app
    with patch('app.main.model', mock_model):
        # Import app after patching to ensure model is set during startup
        from app.main import app
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def client_without_model():
    """Test client without model (for testing 503 errors)."""
    # Ensure model is None before importing the app
    with patch('app.main.model', None):
        from app.main import app
        with TestClient(app) as test_client:
            yield test_client
