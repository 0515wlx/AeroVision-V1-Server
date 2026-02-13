"""
Quality assessment service.
"""

from typing import Any

from PIL import Image

from app.core.exceptions import ImageLoadError
from app.inference import InferenceFactory, wrap_quality_result
from app.schemas.quality import QualityResult
from app.services.base import BaseService


class QualityService(BaseService):
    """Service for image quality assessment."""

    def __init__(self):
        """Initialize the quality service."""
        self._assessor = None

    def _get_assessor(self):
        """Lazy load the quality assessor."""
        if self._assessor is None:
            self._assessor = InferenceFactory.get_quality_assessor()
        return self._assessor

    def assess(self, image_input: str) -> tuple[QualityResult, float]:
        """
        Assess image quality.

        Args:
            image_input: Base64 encoded image or URL

        Returns:
            Tuple of (quality result, processing time ms)

        Raises:
            ImageLoadError: If image loading fails
        """
        image = self.load_image(image_input)
        assessor = self._get_assessor()

        def do_assess():
            result = assessor.assess(image)
            return wrap_quality_result(result)

        result, timing = self.measure_time(do_assess)
        return result, timing

    def assess_batch(self, image_inputs: list[str]) -> list[dict[str, Any]]:
        """
        Assess quality of multiple images.

        Args:
            image_inputs: List of base64 encoded images or URLs

        Returns:
            List of results with index, success status, and data/error
        """
        results = []

        for idx, image_input in enumerate(image_inputs):
            try:
                result, _ = self.assess(image_input)
                results.append({
                    "index": idx,
                    "success": True,
                    "data": result.model_dump(by_alias=True),
                    "error": None
                })
            except ImageLoadError as e:
                results.append({
                    "index": idx,
                    "success": False,
                    "data": None,
                    "error": str(e)
                })
            except Exception as e:
                results.append({
                    "index": idx,
                    "success": False,
                    "data": None,
                    "error": f"Quality assessment failed: {e}"
                })

        return results
