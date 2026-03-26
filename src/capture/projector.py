"""Screen-based pattern projection for dual photography.

Uses an OpenCV fullscreen window on a secondary monitor (or projector)
to display illumination patterns. In a physical setup, this replaces
a dedicated projector with a monitor/screen.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScreenProjector:
    """Displays illumination patterns on a screen/projector via OpenCV.

    Creates a fullscreen window on the specified monitor to display
    patterns for light transport acquisition.

    Attributes:
        window_name: Name of the OpenCV display window.
        monitor_index: Index of the monitor to use (0 = primary).
        display_delay_ms: Delay after displaying a pattern (ms) to allow
            camera exposure and readout.
        resolution: Pattern display resolution as (width, height).
    """

    window_name: str = "DualPhoto_Projector"
    monitor_index: int = 1
    display_delay_ms: int = 200
    resolution: tuple[int, int] = (800, 600)

    def open(self) -> None:
        """Create and configure the fullscreen display window."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        # Display black initially
        black = np.zeros(
            (self.resolution[1], self.resolution[0]), dtype=np.uint8
        )
        cv2.imshow(self.window_name, black)
        cv2.waitKey(1)
        logger.info(
            f"Projector window opened: {self.window_name} "
            f"at {self.resolution[0]}x{self.resolution[1]}"
        )

    def display_pattern(self, pattern: np.ndarray) -> None:
        """Display a pattern on the projector screen.

        The pattern is scaled to the display resolution and converted
        to 8-bit for display.

        Args:
            pattern: 2D array with values in [0, 1], shape (height, width).
        """
        # Scale pattern to display resolution
        display = cv2.resize(
            pattern.astype(np.float32),
            self.resolution,
            interpolation=cv2.INTER_NEAREST,
        )
        # Convert to 8-bit
        display_uint8 = (display * 255).clip(0, 255).astype(np.uint8)
        cv2.imshow(self.window_name, display_uint8)
        cv2.waitKey(self.display_delay_ms)

    def display_black(self) -> None:
        """Display a black (all-off) screen."""
        black = np.zeros(
            (self.resolution[1], self.resolution[0]), dtype=np.uint8
        )
        cv2.imshow(self.window_name, black)
        cv2.waitKey(1)

    def display_white(self) -> None:
        """Display a white (all-on) screen."""
        white = np.full(
            (self.resolution[1], self.resolution[0]), 255, dtype=np.uint8
        )
        cv2.imshow(self.window_name, white)
        cv2.waitKey(1)

    def close(self) -> None:
        """Close the projector display window."""
        cv2.destroyWindow(self.window_name)
        logger.info("Projector window closed.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
