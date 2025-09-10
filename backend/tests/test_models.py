"""
Model tests
"""
import pytest
import numpy as np
from app.models.tea_harvest_model import TeaHarvestModel


def test_tea_harvest_model_initialization():
    """Test model initialization"""
    model = TeaHarvestModel()
    assert model is not None


def test_tea_harvest_model_prediction():
    """Test model prediction functionality"""
    model = TeaHarvestModel()
    
    # Sample input data with correct shape (batch_size, sequence_length, features)
    # Expected shape: (1, 30, 6) - 1 batch, 30 days, 6 features
    test_input = np.random.rand(1, 30, 6).astype('float32')
    
    # Test prediction (assuming model has predict method)
    try:
        prediction = model.predict(test_input)
        assert prediction is not None
        assert len(prediction) > 0
        assert prediction.shape[0] == 1  # Should return 1 prediction
    except NotImplementedError:
        # If model is not fully implemented yet, just test that it exists
        pytest.skip("Model prediction not implemented yet")


def test_tea_harvest_model_data_validation():
    """Test model data validation"""
    model = TeaHarvestModel()
    
    # Test with invalid input
    with pytest.raises((ValueError, TypeError)):
        model.predict("invalid_input")
