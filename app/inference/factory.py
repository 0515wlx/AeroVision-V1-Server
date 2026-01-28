"""
Inference model factory for loading and managing aerovision-v1-inference components.

This module provides a singleton factory for lazy-loading inference models from
the aerovision-v1-inference package.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.core.logging import logger

# Try to import from aerovision-v1-inference
# Allow graceful degradation if not available
try:
    from infer import (
        AircraftClassifier,
        AirlineClassifier,
        RegistrationOCR,
        QualityAssessor,
    )
    INFERENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"aerovision-v1-inference not available: {e}")
    INFERENCE_AVAILABLE = False
    AircraftClassifier = None  # type: ignore
    AirlineClassifier = None  # type: ignore
    RegistrationOCR = None  # type: ignore
    QualityAssessor = None  # type: ignore


class InferenceFactoryError(Exception):
    """Exception raised when inference factory operations fail."""

    pass


class InferenceFactory:
    """
    Factory for managing inference models from aerovision-v1-inference.

    Provides lazy-loading singleton instances of all inference components.
    Models are loaded on first access and cached for subsequent use.
    """

    _aircraft_classifier: Optional[AircraftClassifier] = None
    _airline_classifier: Optional[AirlineClassifier] = None
    _registration_ocr: Optional[RegistrationOCR] = None
    _quality_assessor: Optional[QualityAssessor] = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if aerovision-v1-inference package is available."""
        return INFERENCE_AVAILABLE

    @classmethod
    def get_model_dir(cls) -> Path:
        """Get the model directory from environment or default."""
        model_dir = os.getenv("MODEL_DIR", "models")
        return Path(model_dir)

    @classmethod
    def get_device(cls) -> str:
        """Get the device for inference (cuda/cpu)."""
        return os.getenv("DEVICE", "cuda")

    @classmethod
    def get_aircraft_classifier(cls) -> AircraftClassifier:
        """
        Get or create the aircraft classifier instance.

        Returns:
            AircraftClassifier: The aircraft classifier instance.

        Raises:
            InferenceFactoryError: If inference package is not available.
        """
        if not INFERENCE_AVAILABLE:
            raise InferenceFactoryError("aerovision-v1-inference package not available")

        if cls._aircraft_classifier is None:
            model_path = cls.get_model_dir() / "aircraft" / "best.pt"
            if not model_path.exists():
                model_path = cls.get_model_dir() / "best.pt"

            logger.info(f"Loading aircraft classifier from {model_path}")
            cls._aircraft_classifier = AircraftClassifier(
                model_path=str(model_path),
                device=cls.get_device()
            )
            logger.info("Aircraft classifier loaded successfully")

        return cls._aircraft_classifier

    @classmethod
    def get_airline_classifier(cls) -> AirlineClassifier:
        """
        Get or create the airline classifier instance.

        Returns:
            AirlineClassifier: The airline classifier instance.

        Raises:
            InferenceFactoryError: If inference package is not available.
        """
        if not INFERENCE_AVAILABLE:
            raise InferenceFactoryError("aerovision-v1-inference package not available")

        if cls._airline_classifier is None:
            model_path = cls.get_model_dir() / "airline" / "best.pt"
            if not model_path.exists():
                model_path = cls.get_model_dir() / "best.pt"

            logger.info(f"Loading airline classifier from {model_path}")
            cls._airline_classifier = AirlineClassifier(
                model_path=str(model_path),
                device=cls.get_device()
            )
            logger.info("Airline classifier loaded successfully")

        return cls._airline_classifier

    @classmethod
    def get_registration_ocr(cls) -> RegistrationOCR:
        """
        Get or create the registration OCR instance.

        Returns:
            RegistrationOCR: The registration OCR instance.

        Raises:
            InferenceFactoryError: If inference package is not available.
        """
        if not INFERENCE_AVAILABLE:
            raise InferenceFactoryError("aerovision-v1-inference package not available")

        if cls._registration_ocr is None:
            ocr_mode = os.getenv("OCR_MODE", "local")
            ocr_lang = os.getenv("OCR_LANG", "ch")
            use_angle_cls = os.getenv("USE_ANGLE_CLS", "true").lower() == "true"

            logger.info(f"Loading registration OCR in {ocr_mode} mode")
            cls._registration_ocr = RegistrationOCR(
                mode=ocr_mode,
                lang=ocr_lang,
                use_angle_cls=use_angle_cls,
                enabled=True
            )
            logger.info("Registration OCR loaded successfully")

        return cls._registration_ocr

    @classmethod
    def get_quality_assessor(cls) -> QualityAssessor:
        """
        Get or create the quality assessor instance.

        Returns:
            QualityAssessor: The quality assessor instance.

        Raises:
            InferenceFactoryError: If inference package is not available.
        """
        if not INFERENCE_AVAILABLE:
            raise InferenceFactoryError("aerovision-v1-inference package not available")

        if cls._quality_assessor is None:
            sharpness_weight = float(os.getenv("SHARPNESS_WEIGHT", "0.3"))
            exposure_weight = float(os.getenv("EXPOSURE_WEIGHT", "0.2"))
            composition_weight = float(os.getenv("COMPOSITION_WEIGHT", "0.15"))
            noise_weight = float(os.getenv("NOISE_WEIGHT", "0.2"))
            color_weight = float(os.getenv("COLOR_WEIGHT", "0.15"))
            pass_threshold = float(os.getenv("QUALITY_PASS_THRESHOLD", "0.6"))

            logger.info("Loading quality assessor")
            cls._quality_assessor = QualityAssessor(
                sharpness_weight=sharpness_weight,
                exposure_weight=exposure_weight,
                composition_weight=composition_weight,
                noise_weight=noise_weight,
                color_weight=color_weight,
                pass_threshold=pass_threshold
            )
            logger.info("Quality assessor loaded successfully")

        return cls._quality_assessor

    @classmethod
    def preload_models(cls) -> None:
        """
        Preload all inference models.

        Useful for warming up the service before handling requests.
        """
        if not INFERENCE_AVAILABLE:
            logger.warning("Cannot preload models: aerovision-v1-inference not available")
            return

        logger.info("Preloading inference models...")
        cls.get_aircraft_classifier()
        cls.get_airline_classifier()
        cls.get_registration_ocr()
        cls.get_quality_assessor()
        logger.info("All inference models preloaded")

    @classmethod
    def reset(cls) -> None:
        """Reset all cached instances (mainly for testing)."""
        cls._aircraft_classifier = None
        cls._airline_classifier = None
        cls._registration_ocr = None
        cls._quality_assessor = None
