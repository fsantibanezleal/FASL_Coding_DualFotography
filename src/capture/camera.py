"""Camera capture module for physical dual photography acquisition.

Provides webcam access via OpenCV for capturing scene responses
to projected illumination patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraCapture:
    """Manages webcam capture for dual photography acquisition.

    Wraps OpenCV VideoCapture with convenience methods for
    synchronized pattern-capture workflows.

    Attributes:
        device_id: Camera device index (0 for default webcam).
        resolution: Desired capture resolution as (width, height).
        warmup_frames: Number of frames to discard on startup for auto-exposure settling.
    """

    device_id: int = 0
    resolution: tuple[int, int] = (640, 480)
    warmup_frames: int = 10

    def __post_init__(self):
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open the camera device and configure resolution.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device {self.device_id}. "
                "Check that a webcam is connected and not in use by another application."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Warmup: let auto-exposure settle
        for _ in range(self.warmup_frames):
            self._cap.read()

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened: device={self.device_id}, resolution={actual_w}x{actual_h}")

    def capture(self, grayscale: bool = True) -> np.ndarray:
        """Capture a single frame from the camera.

        Args:
            grayscale: If True, convert to grayscale. Otherwise return BGR.

        Returns:
            Captured image as a numpy array. Shape (H, W) for grayscale,
            (H, W, 3) for color.

        Raises:
            RuntimeError: If capture fails or camera is not open.
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera is not open. Call open() first.")

        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera.")

        if grayscale:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        return frame.astype(np.float64) / 255.0

    def capture_averaged(
        self, n_frames: int = 5, grayscale: bool = True
    ) -> np.ndarray:
        """Capture multiple frames and return their average.

        Averaging reduces sensor noise, improving transport matrix quality.

        Args:
            n_frames: Number of frames to average.
            grayscale: If True, capture in grayscale.

        Returns:
            Averaged image as a float64 array in [0, 1].
        """
        frames = [self.capture(grayscale) for _ in range(n_frames)]
        return np.mean(frames, axis=0)

    def get_resolution(self) -> tuple[int, int]:
        """Get the actual camera resolution.

        Returns:
            (width, height) tuple.
        """
        if self._cap is None:
            return self.resolution
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def close(self) -> None:
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera closed.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def list_devices(max_devices: int = 10) -> list[dict]:
        """List available camera devices.

        Probes device indices 0..max_devices-1 and returns info
        for those that can be opened.

        Args:
            max_devices: Maximum number of device indices to probe.

        Returns:
            List of dicts with 'id', 'width', 'height' for each available camera.
        """
        devices = []
        for idx in range(max_devices):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                devices.append({"id": idx, "width": w, "height": h})
                cap.release()
        return devices
