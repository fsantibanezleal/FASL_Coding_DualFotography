"""Acquisition pipeline for physical dual photography capture.

Orchestrates the synchronized projection-capture workflow: displays each
illumination pattern on the screen projector, captures the camera response,
and computes the transport matrix from the collected data.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.capture.camera import CameraCapture
from src.capture.projector import ScreenProjector
from src.core.patterns import PatternType, generate_patterns
from src.core.transport import TransportMatrix

logger = logging.getLogger(__name__)


@dataclass
class AcquisitionConfig:
    """Configuration for a physical acquisition session.

    Attributes:
        proj_shape: Pattern resolution as (height, width).
        cam_resolution: Camera capture resolution as (width, height).
        pattern_type: Type of illumination patterns.
        n_patterns: Number of patterns (for Bernoulli mode).
        seed: Random seed for reproducibility.
        camera_device: Camera device index.
        display_delay_ms: Delay after pattern display (ms).
        n_avg_frames: Number of frames to average per capture.
        capture_grayscale: Whether to capture in grayscale.
    """

    proj_shape: tuple[int, int] = (8, 8)
    cam_resolution: tuple[int, int] = (640, 480)
    pattern_type: PatternType = PatternType.HADAMARD
    n_patterns: int | None = None
    seed: int | None = 42
    camera_device: int = 0
    display_delay_ms: int = 300
    n_avg_frames: int = 3
    capture_grayscale: bool = True


@dataclass
class AcquisitionResult:
    """Results from a physical acquisition session.

    Attributes:
        patterns: List of projected patterns.
        captures: List of captured camera responses.
        transport: Computed transport matrix.
        ambient: Ambient light capture (with projector off).
        elapsed_seconds: Total acquisition time.
    """

    patterns: list[np.ndarray]
    captures: list[np.ndarray]
    transport: TransportMatrix
    ambient: np.ndarray | None = None
    elapsed_seconds: float = 0.0


class AcquisitionPipeline:
    """Manages the full physical capture workflow for dual photography.

    Coordinates pattern projection, camera capture, and transport matrix
    computation in a synchronized pipeline.

    Example:
        >>> config = AcquisitionConfig(proj_shape=(8, 8), pattern_type=PatternType.HADAMARD)
        >>> pipeline = AcquisitionPipeline(config)
        >>> result = pipeline.run()
        >>> dual = result.transport.dual(np.ones(result.transport.cam_shape))
    """

    def __init__(self, config: AcquisitionConfig):
        """Initialize the acquisition pipeline.

        Args:
            config: Acquisition configuration parameters.
        """
        self.config = config
        self.camera = CameraCapture(
            device_id=config.camera_device,
            resolution=config.cam_resolution,
        )
        self.projector = ScreenProjector(
            display_delay_ms=config.display_delay_ms,
        )
        self._progress_callback: Callable[[int, int, str], None] | None = None

    def set_progress_callback(
        self, callback: Callable[[int, int, str], None]
    ) -> None:
        """Set a callback for progress reporting.

        Args:
            callback: Function called with (current_step, total_steps, message).
        """
        self._progress_callback = callback

    def _report_progress(self, current: int, total: int, message: str) -> None:
        """Report acquisition progress."""
        if self._progress_callback:
            self._progress_callback(current, total, message)
        logger.info(f"[{current}/{total}] {message}")

    def run(self) -> AcquisitionResult:
        """Execute the full acquisition pipeline.

        Steps:
        1. Open camera and projector
        2. Capture ambient (projector off)
        3. For each pattern: display -> wait -> capture -> store
        4. Compute transport matrix from captures
        5. Clean up

        Returns:
            AcquisitionResult with patterns, captures, and transport matrix.

        Raises:
            RuntimeError: If camera or projector setup fails.
        """
        cfg = self.config
        t_start = time.time()

        # Generate patterns
        patterns = generate_patterns(
            height=cfg.proj_shape[0],
            width=cfg.proj_shape[1],
            pattern_type=cfg.pattern_type,
            n_patterns=cfg.n_patterns,
            seed=cfg.seed,
        )
        total_steps = len(patterns) + 2  # +2 for ambient capture and T computation

        self.camera.open()
        self.projector.open()

        try:
            # Step 1: Capture ambient (projector off)
            self._report_progress(1, total_steps, "Capturing ambient light...")
            self.projector.display_black()
            time.sleep(cfg.display_delay_ms / 1000.0)
            ambient = self.camera.capture_averaged(
                n_frames=cfg.n_avg_frames, grayscale=cfg.capture_grayscale
            )

            # Step 2: Capture responses to each pattern
            captures = []
            for idx, pattern in enumerate(patterns):
                self._report_progress(
                    idx + 2,
                    total_steps,
                    f"Projecting pattern {idx + 1}/{len(patterns)}...",
                )
                self.projector.display_pattern(pattern)
                time.sleep(cfg.display_delay_ms / 1000.0)
                capture = self.camera.capture_averaged(
                    n_frames=cfg.n_avg_frames, grayscale=cfg.capture_grayscale
                )
                # Subtract ambient
                capture = np.clip(capture - ambient, 0, 1)
                captures.append(capture)

            # Step 3: Compute transport matrix
            self._report_progress(total_steps, total_steps, "Computing transport matrix...")
            cam_h, cam_w = captures[0].shape[:2]
            transport = TransportMatrix.from_pattern_captures(
                patterns=patterns,
                captures=captures,
                proj_shape=cfg.proj_shape,
                cam_shape=(cam_h, cam_w),
            )

            elapsed = time.time() - t_start
            logger.info(f"Acquisition complete in {elapsed:.1f}s")

            return AcquisitionResult(
                patterns=patterns,
                captures=captures,
                transport=transport,
                ambient=ambient,
                elapsed_seconds=elapsed,
            )

        finally:
            self.projector.display_black()
            self.projector.close()
            self.camera.close()
