"""
Registration number OCR service.
"""

from typing import Any

from PIL import Image

from app.core.exceptions import ImageLoadError
from app.inference import InferenceFactory, wrap_registration_result
from app.schemas.registration import RegistrationResult
from app.services.base import BaseService


class RegistrationService(BaseService):
    """Service for registration number OCR."""

    def __init__(self):
        """Initialize the registration service."""
        self._ocr = None

    def _get_ocr(self):
        """Lazy load the registration OCR."""
        if self._ocr is None:
            self._ocr = InferenceFactory.get_registration_ocr()
        return self._ocr

    def recognize(self, image_input: str) -> tuple[RegistrationResult, float]:
        """
        Recognize registration number.

        Args:
            image_input: Base64 encoded image or URL

        Returns:
            Tuple of (registration result, processing time ms)

        Raises:
            ImageLoadError: If image loading fails
        """
        image = self.load_image(image_input)
        ocr = self._get_ocr()

        def do_recognize():
            result = ocr.recognize(image)
            return wrap_registration_result(result)

        result, timing = self.measure_time(do_recognize)
        return result, timing

    def recognize_batch(self, image_inputs: list[str]) -> list[dict[str, Any]]:
        """
        Recognize registration numbers from multiple images.

        Args:
            image_inputs: List of base64 encoded images or URLs

        Returns:
            List of results with index, success status, and data/error
        """
        results = []

        for idx, image_input in enumerate(image_inputs):
            try:
                result, _ = self.recognize(image_input)
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
                    "error": f"Registration OCR failed: {e}"
                })

        return results
