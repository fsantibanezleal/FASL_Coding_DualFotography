"""Transport matrix computation and manipulation.

The light transport matrix T encodes how light travels from a projector
through a scene to a camera. Element T[i, j] represents the fraction of
light from projector pixel j that reaches camera pixel i, through all
transport paths (direct, reflected, refracted, scattered).

The key equation is:
    c = T @ p

where p is the projected pattern (pq x 1), c is the captured image (mn x 1),
and T is the transport matrix (mn x pq).

References:
    Sen et al., "Dual Photography", SIGGRAPH 2005
    Sen & Darabi, "Compressive Dual Photography", Eurographics 2009
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.linalg import svd


@dataclass
class TransportMatrix:
    """Encapsulates the light transport matrix and its decompositions.

    The transport matrix T maps projector pixels to camera pixels. This class
    provides methods for computing T from pattern-capture pairs, performing
    SVD decomposition, and generating dual/relighted images.

    Attributes:
        T: The full transport matrix of shape (cam_pixels, proj_pixels).
        cam_shape: Camera image resolution as (height, width).
        proj_shape: Projector pattern resolution as (height, width).
        n_channels: Number of color channels (1 for grayscale, 3 for RGB).
        U: Left singular vectors from SVD, shape (cam_pixels, k).
        S: Singular values from SVD, shape (k,).
        Vt: Right singular vectors from SVD, shape (k, proj_pixels).
        rank: Effective rank used for truncated SVD.
    """

    T: np.ndarray
    cam_shape: tuple[int, int]
    proj_shape: tuple[int, int]
    n_channels: int = 1
    U: np.ndarray | None = field(default=None, repr=False)
    S: np.ndarray | None = field(default=None, repr=False)
    Vt: np.ndarray | None = field(default=None, repr=False)
    rank: int | None = None

    @property
    def cam_pixels(self) -> int:
        """Total number of camera pixels (height * width * channels)."""
        return self.cam_shape[0] * self.cam_shape[1] * self.n_channels

    @property
    def proj_pixels(self) -> int:
        """Total number of projector pixels (height * width)."""
        return self.proj_shape[0] * self.proj_shape[1]

    @classmethod
    def from_pattern_captures(
        cls,
        patterns: list[np.ndarray],
        captures: list[np.ndarray],
        proj_shape: tuple[int, int],
        cam_shape: tuple[int, int],
        method: Literal["pseudoinverse", "direct"] = "pseudoinverse",
    ) -> TransportMatrix:
        """Compute the transport matrix from pattern-capture pairs.

        Given N projected patterns and N corresponding camera captures,
        computes T such that C = T @ P, where C and P are matrices whose
        columns are the vectorized captures and patterns respectively.

        For the pseudoinverse method:
            T = C @ pinv(P)

        For the direct method (requires canonical patterns):
            Each capture directly becomes a column of T.

        Args:
            patterns: List of N projected patterns, each of shape proj_shape.
            captures: List of N captured images, each of shape cam_shape or
                (cam_shape[0], cam_shape[1], channels).
            proj_shape: Projector resolution as (height, width).
            cam_shape: Camera resolution as (height, width).
            method: Computation method. "pseudoinverse" works with any patterns;
                "direct" requires canonical (one-pixel-at-a-time) patterns.

        Returns:
            TransportMatrix instance with the computed T.

        Raises:
            ValueError: If patterns and captures have different lengths.
        """
        if len(patterns) != len(captures):
            raise ValueError(
                f"Number of patterns ({len(patterns)}) must match "
                f"number of captures ({len(captures)})."
            )

        # Determine number of channels from captures
        sample_capture = captures[0]
        if sample_capture.ndim == 3:
            n_channels = sample_capture.shape[2]
        else:
            n_channels = 1

        n_cam = cam_shape[0] * cam_shape[1] * n_channels
        n_proj = proj_shape[0] * proj_shape[1]

        # Build matrices: columns are vectorized patterns/captures
        P = np.column_stack([p.flatten().astype(np.float64) for p in patterns])  # (n_proj, N)
        C = np.column_stack([c.flatten().astype(np.float64) for c in captures])  # (n_cam, N)

        if method == "direct":
            # For canonical patterns, C is already T (each column = one projector pixel response)
            T = C
        else:
            # T = C @ pinv(P)
            T = C @ np.linalg.pinv(P)

        return cls(T=T, cam_shape=cam_shape, proj_shape=proj_shape, n_channels=n_channels)

    def compute_svd(self, rank: int | None = None) -> None:
        """Compute the Singular Value Decomposition of T.

        Performs economy SVD: T = U @ diag(S) @ Vt. If rank is specified,
        only the top-k components are retained (truncated SVD), which acts
        as a low-rank approximation and denoises the transport matrix.

        By the Eckart-Young-Mirsky theorem, the truncated SVD is the optimal
        rank-k approximation in both Frobenius and spectral norms.

        Args:
            rank: Number of singular values to keep. If None, all are retained.
        """
        U, S, Vt = svd(self.T, full_matrices=False)
        if rank is not None and rank < len(S):
            U = U[:, :rank]
            S = S[:rank]
            Vt = Vt[:rank, :]
            self.rank = rank
        else:
            self.rank = len(S)
        self.U = U
        self.S = S
        self.Vt = Vt

    def get_truncated_T(self, rank: int | None = None) -> np.ndarray:
        """Return a low-rank approximation of the transport matrix.

        Args:
            rank: Number of singular values to keep. Uses stored rank if None.

        Returns:
            Approximate transport matrix of shape (cam_pixels, proj_pixels).
        """
        if self.U is None or self.S is None or self.Vt is None:
            self.compute_svd(rank)

        k = rank if rank is not None else self.rank
        U_k = self.U[:, :k]
        S_k = self.S[:k]
        Vt_k = self.Vt[:k, :]
        return U_k @ np.diag(S_k) @ Vt_k

    def forward(self, pattern: np.ndarray) -> np.ndarray:
        """Compute the camera image for a given projector pattern (primal).

        Implements: c = T @ p

        Args:
            pattern: Projector pattern of shape proj_shape, values in [0, 1].

        Returns:
            Simulated camera image of shape cam_shape (or with channels).
        """
        p = pattern.flatten().astype(np.float64)
        c = self.T @ p
        c = np.clip(c, 0, None)
        if self.n_channels > 1:
            return c.reshape(self.cam_shape[0], self.cam_shape[1], self.n_channels)
        return c.reshape(self.cam_shape)

    def dual(self, illumination: np.ndarray) -> np.ndarray:
        """Compute the dual image (view from projector position).

        Implements: p_dual = T^T @ c_dual

        By Helmholtz reciprocity, transposing T swaps the roles of
        projector and camera: the projector becomes a virtual camera
        and the camera becomes a virtual light source.

        Args:
            illumination: Virtual illumination pattern from camera position,
                shape cam_shape (or with channels). Values in [0, 1].

        Returns:
            Dual image of shape proj_shape, representing the scene as
            viewed from the projector's position.
        """
        c = illumination.flatten().astype(np.float64)
        p_dual = self.T.T @ c
        p_dual = np.clip(p_dual, 0, None)
        return p_dual.reshape(self.proj_shape)

    def relight(self, new_pattern: np.ndarray) -> np.ndarray:
        """Relight the scene with a new projector illumination pattern.

        Given a new illumination pattern, synthesizes what the camera
        would see without re-capturing.

        Args:
            new_pattern: New projector pattern of shape proj_shape.

        Returns:
            Relighted camera image.
        """
        return self.forward(new_pattern)

    def singular_value_spectrum(self) -> np.ndarray:
        """Return the singular value spectrum of T.

        Useful for analyzing the effective dimensionality of the light
        transport and choosing an appropriate truncation rank.

        Returns:
            Array of singular values in descending order.
        """
        if self.S is None:
            self.compute_svd()
        return self.S.copy()

    def save(self, path: str) -> None:
        """Save the transport matrix and metadata to a .npz file.

        Args:
            path: File path for the output .npz archive.
        """
        data = {
            "T": self.T,
            "cam_shape": np.array(self.cam_shape),
            "proj_shape": np.array(self.proj_shape),
            "n_channels": np.array(self.n_channels),
        }
        if self.U is not None:
            data["U"] = self.U
            data["S"] = self.S
            data["Vt"] = self.Vt
            data["rank"] = np.array(self.rank)
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str) -> TransportMatrix:
        """Load a transport matrix from a .npz file.

        Args:
            path: File path to the .npz archive.

        Returns:
            TransportMatrix instance.
        """
        data = np.load(path)
        tm = cls(
            T=data["T"],
            cam_shape=tuple(data["cam_shape"]),
            proj_shape=tuple(data["proj_shape"]),
            n_channels=int(data["n_channels"]),
        )
        if "U" in data:
            tm.U = data["U"]
            tm.S = data["S"]
            tm.Vt = data["Vt"]
            tm.rank = int(data["rank"])
        return tm
