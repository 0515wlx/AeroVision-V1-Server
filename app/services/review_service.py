"""
Aggregated review service.
"""

from typing import Any

from app.schemas.review import ReviewResult, ReviewQualityResult, ReviewAircraftResult, ReviewAirlineResult, ReviewRegistrationResult
from app.services.quality_service import QualityService
from app.services.aircraft_service import AircraftService
from app.services.airline_service import AirlineService
from app.services.registration_service import RegistrationService
from app.services.base import BaseService, ImageLoadError


class ReviewService(BaseService):
    """Service for aggregated image review."""

    def __init__(self):
        """Initialize the review service with sub-services."""
        self.quality_service = QualityService()
        self.aircraft_service = AircraftService()
        self.airline_service = AirlineService()
        self.registration_service = RegistrationService()

    def review(
        self,
        image_input: str,
        include_quality: bool = True,
        include_aircraft: bool = True,
        include_airline: bool = True,
        include_registration: bool = True
    ) -> tuple[ReviewResult, float]:
        """
        Perform a complete review of an image.

        Args:
            image_input: Base64 encoded image or URL
            include_quality: Whether to include quality assessment
            include_aircraft: Whether to include aircraft classification
            include_airline: Whether to include airline classification
            include_registration: Whether to include registration OCR

        Returns:
            Tuple of (review result, processing time ms)

        Raises:
            ImageLoadError: If image loading fails
        """
        start_time = self._now()

        # Load image once
        image = self.load_image(image_input)

        # Collect results
        quality_result = None
        aircraft_result = None
        airline_result = None
        registration_result = None

        if include_quality:
            quality_result, _ = self.safe_execute(
                self.quality_service.assess, image_input
            )

        if include_aircraft:
            aircraft_data = self.safe_execute(
                self.aircraft_service.classify, image_input
            )
            if aircraft_data:
                aircraft_result = ReviewAircraftResult(
                    type_code=aircraft_data.top1.class_,
                    confidence=aircraft_data.top1.confidence
                )

        if include_airline:
            airline_data = self.safe_execute(
                self.airline_service.classify, image_input
            )
            if airline_data:
                airline_result = ReviewAirlineResult(
                    airline_code=airline_data.top1.class_,
                    confidence=airline_data.top1.confidence
                )

        if include_registration:
            reg_data = self.safe_execute(
                self.registration_service.recognize, image_input
            )
            if reg_data:
                # Use registration confidence as clarity
                registration_result = ReviewRegistrationResult(
                    registration=reg_data.registration,
                    confidence=reg_data.confidence,
                    clarity=reg_data.confidence  # Using OCR confidence as proxy
                )

        # Build final result
        # Quality is mandatory - provide default if failed
        if quality_result is None:
            quality_result = ReviewQualityResult(
                score=0.0,
                pass_=False,
                details=None
            )

        # Aircraft is mandatory - provide default if failed
        if aircraft_result is None:
            aircraft_result = ReviewAircraftResult(
                type_code="UNKNOWN",
                confidence=0.0
            )

        result = ReviewResult(
            quality=quality_result,
            aircraft=aircraft_result,
            airline=airline_result,
            registration=registration_result
        )

        timing = (self._now() - start_time) * 1000
        return result, timing

    def review_batch(
        self,
        image_inputs: list[str],
        include_quality: bool = True,
        include_aircraft: bool = True,
        include_airline: bool = True,
        include_registration: bool = True
    ) -> list[dict[str, Any]]:
        """
        Review multiple images.

        Args:
            image_inputs: List of base64 encoded images or URLs
            include_quality: Whether to include quality assessment
            include_aircraft: Whether to include aircraft classification
            include_airline: Whether to include airline classification
            include_registration: Whether to include registration OCR

        Returns:
            List of results with index, success status, and data/error
        """
        results = []

        for idx, image_input in enumerate(image_inputs):
            try:
                result, _ = self.review(
                    image_input,
                    include_quality=include_quality,
                    include_aircraft=include_aircraft,
                    include_airline=include_airline,
                    include_registration=include_registration
                )
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
                    "error": f"Review failed: {e}"
                })

        return results

    @staticmethod
    def _now() -> float:
        """Get current time in seconds."""
        import time
        return time.perf_counter()
