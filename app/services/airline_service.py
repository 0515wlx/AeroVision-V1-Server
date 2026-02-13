"""
Airline classification service.
"""

from typing import Any

from PIL import Image

from app.core.exceptions import ImageLoadError
from app.inference import InferenceFactory, wrap_airline_result
from app.schemas.airline import AirlineResult
from app.services.base import BaseService


class AirlineService(BaseService):
    """Service for airline classification."""

    def __init__(self):
        """Initialize the airline service."""
        self._classifier = None

    def _get_classifier(self):
        """Lazy load the airline classifier."""
        if self._classifier is None:
            self._classifier = InferenceFactory.get_airline_classifier()
        return self._classifier

    def classify(self, image_input: str, top_k: int | None = None) -> tuple[AirlineResult, float]:
        """
        Classify airline.

        Args:
            image_input: Base64 encoded image or URL
            top_k: Number of top predictions to return

        Returns:
            Tuple of (airline result, processing time ms)

        Raises:
            ImageLoadError: If image loading fails
        """
        image = self.load_image(image_input)
        classifier = self._get_classifier()

        def do_classify():
            result = classifier.predict(image, top_k=top_k)
            return wrap_airline_result(result)

        result, timing = self.measure_time(do_classify)
        return result, timing

    def classify_batch(
        self,
        image_inputs: list[str],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Classify multiple images.

        Args:
            image_inputs: List of base64 encoded images or URLs
            top_k: Number of top predictions to return

        Returns:
            List of results with index, success status, and data/error
        """
        results = []

        for idx, image_input in enumerate(image_inputs):
            try:
                result, _ = self.classify(image_input, top_k=top_k)
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
                    "error": f"Airline classification failed: {e}"
                })

        return results
