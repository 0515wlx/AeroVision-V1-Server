"""
Unit tests for ReviewService.

Tests for bug fixes and optimizations:
1. Test that safe_execute return value None doesn't cause TypeError when unpacking
2. Test that image is loaded only once (not re-loaded by sub-services)
"""

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch, call

import pytest
from PIL import Image

from app.services.review_service import ReviewService
from app.schemas.quality import QualityResult, QualityDetails


class TestReviewServiceBugFixes:
    """Tests for critical bug fixes in ReviewService."""

    @pytest.fixture
    def test_image_bytes(self):
        """Generate test image bytes."""
        img = Image.new("RGB", (640, 640), color=(100, 150, 200))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    @pytest.fixture
    def test_image_base64(self, test_image_bytes):
        """Generate test image as base64 string."""
        return base64.b64encode(test_image_bytes).decode("utf-8")

    @pytest.fixture
    def sample_quality_result(self):
        """Sample quality result."""
        return QualityResult.model_validate({
            "pass": True,
            "score": 0.85,
            "details": {
                "sharpness": 0.90,
                "exposure": 0.80,
                "composition": 0.85,
                "noise": 0.88,
                "color": 0.82
            }
        })

    def test_review_handles_safe_execute_none_return(self, test_image_base64, sample_quality_result):
        """
        Test that review handles None return from safe_execute without TypeError.

        Bug: When safe_execute returns None, direct unpacking (quality_result, _ = ...)
        causes TypeError: cannot unpack non-iterable NoneType object.
        """
        service = ReviewService()

        # Mock quality_service to return None (simulating safe_execute failure)
        with patch.object(service.quality_service, 'assess', return_value=None):
            # This should not raise TypeError
            result, timing = service.review(test_image_base64, include_quality=True)

            # Quality should have default value when safe_execute returns None
            assert result.quality.score == 0.0
            assert result.quality.pass_ is False
            assert result.quality.details is None

    def test_review_handles_service_exceptions(self, test_image_base64):
        """
        Test that review handles exceptions from sub-services gracefully.
        """
        service = ReviewService()

        # Create a mock function that raises exception but has __name__
        def failing_assess(*args, **kwargs):
            raise Exception("Service failed")
        failing_assess.__name__ = 'assess'

        # Mock safe_execute to return None (simulating exception caught)
        with patch.object(service, 'safe_execute', return_value=None):
            # This should not crash
            result, timing = service.review(test_image_base64, include_quality=True)

            # Quality should have default value
            assert result.quality.score == 0.0
            assert result.quality.pass_ is False

    def test_review_image_loaded_once(self, test_image_base64, sample_quality_result):
        """
        Test that image is loaded only once in review method.

        Bug: Image is loaded in review() and then passed as image_input to sub-services,
        causing sub-services to re-load the image.
        """
        service = ReviewService()

        # Track load_image calls
        original_load_image = service.load_image
        load_image_calls = []

        def tracked_load_image(image_input):
            load_image_calls.append(image_input)
            return original_load_image(image_input)

        with patch.object(service, 'load_image', side_effect=tracked_load_image):
            with patch.object(service.quality_service, 'assess', return_value=(sample_quality_result, 100)):
                # Mock other services to return None (safe_execute behavior)
                with patch.object(service, 'safe_execute', return_value=None):
                    result, timing = service.review(
                        test_image_base64,
                        include_quality=False,  # Disable to check load_image count
                        include_aircraft=False,
                        include_airline=False,
                        include_registration=False
                    )

        # Image should be loaded exactly once
        assert len(load_image_calls) == 1, f"Expected 1 image load, got {len(load_image_calls)}"
