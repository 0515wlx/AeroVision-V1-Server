"""
Unit tests for InferenceFactory.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from app.inference.factory import (
    InferenceFactory,
    InferenceFactoryError,
    INFERENCE_AVAILABLE,
)


class TestInferenceFactory:
    """Tests for InferenceFactory."""

    def test_is_available(self):
        """Test is_available returns correct status."""
        # Just check the method exists and returns a bool
        assert isinstance(InferenceFactory.is_available(), bool)

    def test_get_model_dir_default(self):
        """Test get_model_dir returns default when not set."""
        # Unset env var if set
        os.environ.pop("MODEL_DIR", None)
        model_dir = InferenceFactory.get_model_dir()
        assert str(model_dir) == "models"

    def test_get_model_dir_from_env(self):
        """Test get_model_dir reads from environment."""
        with patch.dict(os.environ, {"MODEL_DIR": "custom_models"}):
            model_dir = InferenceFactory.get_model_dir()
            assert str(model_dir) == "custom_models"

    def test_get_device_default(self):
        """Test get_device returns default when not set."""
        os.environ.pop("DEVICE", None)
        device = InferenceFactory.get_device()
        assert device == "cuda"

    def test_get_device_from_env(self):
        """Test get_device reads from environment."""
        with patch.dict(os.environ, {"DEVICE": "cpu"}):
            device = InferenceFactory.get_device()
            assert device == "cpu"

    def test_reset(self):
        """Test reset clears cached instances."""
        # Reset should not raise any errors
        InferenceFactory.reset()
        # After reset, cached instances should be None
        assert InferenceFactory._aircraft_classifier is None
        assert InferenceFactory._airline_classifier is None
        assert InferenceFactory._registration_ocr is None
        assert InferenceFactory._quality_assessor is None

    @patch("app.inference.factory.INFERENCE_AVAILABLE", False)
    def test_get_aircraft_classifier_when_not_available(self):
        """Test error when inference package not available."""
        InferenceFactory.reset()

        with pytest.raises(InferenceFactoryError) as exc_info:
            InferenceFactory.get_aircraft_classifier()

        assert "not available" in str(exc_info.value)

    @patch("app.inference.factory.INFERENCE_AVAILABLE", False)
    def test_preload_models_when_not_available(self):
        """Test preload_models handles unavailability gracefully."""
        InferenceFactory.reset()
        # Should not raise, just log warning
        InferenceFactory.preload_models()
