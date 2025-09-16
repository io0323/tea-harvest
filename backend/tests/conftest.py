"""
Test configuration and fixtures
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import os
import tempfile
import shutil


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
def mock_model_files():
    """Create mock model files for testing without TensorFlow."""
    import tempfile
    import os
    import joblib
    
    # Create temporary directory for mock model files
    temp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(temp_dir, "app", "models", "saved")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a dummy model file (empty file to simulate model existence)
    model_file = os.path.join(model_dir, "model.keras")
    with open(model_file, 'w') as f:
        f.write("# Mock model file")
    
    # Create mock preprocessors
    preprocessors_file = os.path.join(model_dir, "preprocessors.pkl")
    mock_preprocessors = {
        'scaler': None,
        'region_encoder': {'静岡': 0, '京都': 1},
        'imputer': None,
        'feature_ranges': None
    }
    
    joblib.dump(mock_preprocessors, preprocessors_file)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def client_with_mock_model(mock_model, mock_model_files):
    """Test client with the dummy model injected before app startup."""
    # Change to the temporary directory with mock model files
    original_cwd = os.getcwd()
    os.chdir(mock_model_files)
    
    try:
        # Set test environment variable to skip model loading
        os.environ['PYTEST_CURRENT_TEST'] = 'true'
        
        # Patch the model before importing the app
        with patch('app.main.model', mock_model):
            # Import app after patching to ensure model is set during startup
            from app.main import app
            with TestClient(app) as test_client:
                yield test_client
    finally:
        os.chdir(original_cwd)
        # Clean up environment variable
        if 'PYTEST_CURRENT_TEST' in os.environ:
            del os.environ['PYTEST_CURRENT_TEST']


@pytest.fixture
def client_with_real_model_files(mock_model_files):
    """Test client with real model files but mocked model object."""
    # Change to the temporary directory with mock model files
    original_cwd = os.getcwd()
    os.chdir(mock_model_files)
    
    try:
        # Import app after setting up model files
        from app.main import app
        with TestClient(app) as test_client:
            yield test_client
    finally:
        os.chdir(original_cwd)


@pytest.fixture
def client_without_model():
    """Test client without model (for testing 503 errors)."""
    # Ensure model is None before importing the app
    with patch('app.main.model', None):
        from app.main import app
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def client_with_fully_mocked_model(mock_model):
    """Test client with fully mocked model - no TensorFlow dependencies."""
    # Set test environment variable to skip model loading
    os.environ['PYTEST_CURRENT_TEST'] = 'true'
    
    try:
        # Patch the model before importing the app
        with patch('app.main.model', mock_model):
            # Import app after patching to ensure model is set during startup
            from app.main import app
            with TestClient(app) as test_client:
                yield test_client
    finally:
        # Clean up environment variable
        if 'PYTEST_CURRENT_TEST' in os.environ:
            del os.environ['PYTEST_CURRENT_TEST']


@pytest.fixture
def client_with_simple_mock():
    """Test client with simple mock - minimal dependencies."""
    import numpy as np
    
    class SimpleMockModel:
        def predict(self, X):
            return np.array([[30.0]], dtype=np.float32)
        
        def preprocess_data(self, df):
            X = np.random.rand(1, 30, 6).astype("float32")
            y = np.array([30.0], dtype=np.float32)
            return X, y
    
    mock_model = SimpleMockModel()
    
    # Set test environment variable to skip model loading
    os.environ['PYTEST_CURRENT_TEST'] = 'true'
    
    try:
        # Patch the model before importing the app
        with patch('app.main.model', mock_model):
            # Import app after patching to ensure model is set during startup
            from app.main import app
            with TestClient(app) as test_client:
                yield test_client
    finally:
        # Clean up environment variable
        if 'PYTEST_CURRENT_TEST' in os.environ:
            del os.environ['PYTEST_CURRENT_TEST']
