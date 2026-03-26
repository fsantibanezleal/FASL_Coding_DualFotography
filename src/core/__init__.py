"""Core engine for dual photography: transport matrix, SVD, and dual image computation."""

from src.core.transport import TransportMatrix
from src.core.dual import DualPhotographer

__all__ = ["TransportMatrix", "DualPhotographer"]
