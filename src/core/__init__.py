"""Core engine for dual photography: transport matrix, SVD, BRDF, calibration, and dual image computation."""

from src.core.transport import TransportMatrix
from src.core.dual import DualPhotographer
from src.core.brdf import cook_torrance_ggx, evaluate_brdf_batch
from src.core.calibration import (
    generate_gray_code_patterns,
    decode_gray_code,
    build_correspondence_map,
)

__all__ = [
    "TransportMatrix",
    "DualPhotographer",
    "cook_torrance_ggx",
    "evaluate_brdf_batch",
    "generate_gray_code_patterns",
    "decode_gray_code",
    "build_correspondence_map",
]
