"""Core engine for dual photography: transport matrix, SVD, BRDF, and dual image computation."""

from src.core.transport import TransportMatrix
from src.core.dual import DualPhotographer
from src.core.brdf import cook_torrance_ggx, evaluate_brdf_batch

__all__ = ["TransportMatrix", "DualPhotographer", "cook_torrance_ggx", "evaluate_brdf_batch"]
