"""
Aircraft type classification service.
"""

import asyncio
from typing import Any

from PIL import Image

from app.core.exceptions import ImageLoadError
from app.inference import InferenceFactory, wrap_aircraft_result
from app.schemas.aircraft import AircraftResult
from app.services.base import BaseService


class AircraftService(BaseService):
    """Service for aircraft type classification."""

    def __init__(self):
        """Initialize the aircraft service."""
        self._classifier = None

    def _get_classifier(self):
        """Lazy load the aircraft classifier."""
        if self._classifier is None:
            self._classifier = InferenceFactory.get_aircraft_classifier()
        return self._classifier

    def classify(self, image_input: str, top_k: int | None = None) -> tuple[AircraftResult, float]:
        """
        Classify aircraft type.

        Args:
            image_input: Base64 encoded image or URL
            top_k: Number of top predictions to return

        Returns:
            Tuple of (aircraft result, processing time ms)

        Raises:
            ImageLoadError: If image loading fails
        """
        image = self.load_image(image_input)
        return self._classify_image(image, top_k)

    def _classify_image(self, image: Image.Image, top_k: int | None = None) -> tuple[AircraftResult, float]:
        """
        Classify aircraft type of a pre-loaded image.

        Args:
            image: PIL Image object
            top_k: Number of top predictions to return

        Returns:
            Tuple of (aircraft result, processing time ms)
        """
        classifier = self._get_classifier()

        def do_classify():
            result = classifier.predict(image, top_k=top_k)
            return wrap_aircraft_result(result)

        result, timing = self.measure_time(do_classify)
        return result, timing

    async def classify_batch(
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
        # Load images in parallel using asyncio
        async def load_image_async(image_input: str):
            try:
                loop = asyncio.get_event_loop()
                image = await loop.run_in_executor(None, lambda: self.load_image(image_input))
                return image
            except ImageLoadError:
                return None

        images = await asyncio.gather(*[load_image_async(img) for img in image_inputs])

        aircraft_results = await self._classify_batch(images, top_k)

        results = []
        for idx, result in enumerate(aircraft_results):
            if result is None:
                results.append({
                    "index": idx,
                    "success": False,
                    "data": None,
                    "error": "Aircraft classification failed"
                })
            else:
                results.append({
                    "index": idx,
                    "success": True,
                    "data": result.model_dump(by_alias=True),
                    "error": None
                })

        return results

    async def _classify_batch(self, images: list[Image.Image | None], top_k: int | None = None) -> list[AircraftResult]:
        """
        Classify aircraft type of multiple pre-loaded images using batch inference.

        Args:
            images: List of PIL Image objects (can contain None for failed loads)
            top_k: Number of top predictions to return

        Returns:
            List of AircraftResult objects
        """
        # Filter out None images and keep track of original indices
        valid_images = [(idx, img) for idx, img in enumerate(images) if img is not None]

        if not valid_images:
            return [None] * len(images)

        # Extract valid images in original order
        indices, batch_images = zip(*sorted(valid_images, key=lambda x: x[0]))

        # Run batch prediction in thread pool to avoid blocking
        def run_batch_prediction():
            classifier = self._get_classifier()
            batch_results = classifier.predict(list(batch_images), top_k=top_k)
            return batch_results

        # Execute in thread pool to not block the event loop
        loop = asyncio.get_event_loop()
        batch_results = await loop.run_in_executor(None, run_batch_prediction)

        # Map results back to original indices
        results = [None] * len(images)
        for idx, result in zip(indices, batch_results):
            if result is not None:
                try:
                    wrapped_result = wrap_aircraft_result(result)
                    results[idx] = wrapped_result
                except Exception as e:
                    from app.core.logging import get_logger
                    logger = get_logger("aircraft_service")
                    logger.error(f"Failed to wrap result at index {idx}: {e}")
                    results[idx] = None

        return results
