"""
Airline classification service.
"""

from concurrent.futures import ThreadPoolExecutor
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
        return self._classify_image(image, top_k)

    def _classify_image(self, image: Image.Image, top_k: int | None = None) -> tuple[AirlineResult, float]:
        """
        Classify airline of a pre-loaded image.

        Args:
            image: PIL Image object
            top_k: Number of top predictions to return

        Returns:
            Tuple of (airline result, processing time ms)
        """
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
        images = []
        for image_input in image_inputs:
            try:
                image = self.load_image(image_input)
                images.append(image)
            except ImageLoadError:
                images.append(None)

        airline_results = self._classify_batch(images, top_k)

        results = []
        for idx, result in enumerate(airline_results):
            if result is None:
                results.append({
                    "index": idx,
                    "success": False,
                    "data": None,
                    "error": "Airline classification failed"
                })
            else:
                results.append({
                    "index": idx,
                    "success": True,
                    "data": result.model_dump(by_alias=True),
                    "error": None
                })

        return results

    def _classify_batch(self, images: list[Image.Image | None], top_k: int | None = None) -> list[AirlineResult]:
        """
        Classify airline of multiple pre-loaded images concurrently.

        Args:
            images: List of PIL Image objects (can contain None for failed loads)
            top_k: Number of top predictions to return

        Returns:
            List of AirlineResult objects
        """
        results = [None] * len(images)

        with ThreadPoolExecutor() as executor:
            def classify_with_index(idx, image):
                if image is None:
                    return idx, None
                try:
                    result, _ = self._classify_image(image, top_k)
                    return idx, result
                except Exception as e:
                    logger.error(f"Failed to classify image at index {idx}: {e}")
                    return idx, None

            futures = [executor.submit(classify_with_index, idx, img) for idx, img in enumerate(images)]

            for future in futures:
                idx, result = future.result()
                results[idx] = result

        return results
