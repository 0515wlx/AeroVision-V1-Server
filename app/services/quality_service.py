"""
Quality assessment service.
"""

from concurrent.futures import ThreadPoolExecutor
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
        return self._assess_image(image)

    def _assess_image(self, image: Image.Image) -> tuple[QualityResult, float]:
        """
        Assess quality of a pre-loaded image.

        Args:
            image: PIL Image object

        Returns:
            Tuple of (quality result, processing time ms)
        """
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
        images = []
        for image_input in image_inputs:
            try:
                image = self.load_image(image_input)
                images.append(image)
            except ImageLoadError:
                images.append(None)

        return self._assess_batch(images)

    def _assess_batch(self, images: list[Image.Image | None]) -> list[QualityResult]:
        """
        Assess quality of multiple pre-loaded images concurrently.

        Args:
            images: List of PIL Image objects (can contain None for failed loads)

        Returns:
            List of QualityResult objects
        """
        results = [None] * len(images)

        with ThreadPoolExecutor() as executor:
            def assess_with_index(idx, image):
                if image is None:
                    return idx, None
                try:
                    result, _ = self._assess_image(image)
                    return idx, result
                except Exception:
                    return idx, None

            futures = [executor.submit(assess_with_index, idx, img) for idx, img in enumerate(images)]

            for future in futures:
                idx, result = future.result()
                results[idx] = result

        return results
