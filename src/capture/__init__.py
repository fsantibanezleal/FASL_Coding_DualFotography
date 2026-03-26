"""Physical capture module: webcam acquisition and screen-based pattern projection."""

from src.capture.camera import CameraCapture
from src.capture.projector import ScreenProjector
from src.capture.acquisition import AcquisitionPipeline

__all__ = ["CameraCapture", "ScreenProjector", "AcquisitionPipeline"]
