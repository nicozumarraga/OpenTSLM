# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Style transfer for TS-Haystack.

This module provides covariance projection (linear style transfer) to make
needle signals blend naturally with target background contexts, plus
boundary blending for smooth insertion.
"""

from typing import Literal, Tuple

import numpy as np

from opentslm.time_series_datasets.ts_haystack.core.data_structures import (
    NeedleSample,
    SignalStatistics,
)


class StyleTransfer:
    """
    Applies covariance projection (linear style transfer) to transform
    needle signals to match target context statistics.

    Mathematical formulation:
        1. Normalize needle: x_norm = (x - μ_needle) / σ_needle
        2. Project through covariance: x_proj = L_target @ L_needle^{-1} @ x_norm
        3. Denormalize: x_final = x_proj * σ_target + μ_target

    Where L is the Cholesky decomposition of the covariance matrix.

    Also provides boundary blending for smooth needle insertion.

    Example:
        >>> style_transfer = StyleTransfer(blend_mode="cosine", blend_window_samples=50)
        >>> target_stats = style_transfer.compute_statistics(bg_x, bg_y, bg_z)
        >>> transferred = style_transfer.transfer(needle, target_stats)
        >>> x, y, z = style_transfer.insert_with_blending(
        ...     background=(bg_x, bg_y, bg_z),
        ...     needle=(transferred.x, transferred.y, transferred.z),
        ...     position=1000,
        ... )
    """

    def __init__(
        self,
        blend_mode: Literal["linear", "cosine"] = "cosine",
        blend_window_samples: int = 50,
    ):
        """
        Initialize style transfer.

        Args:
            blend_mode: Blending function for boundaries ("linear" or "cosine")
            blend_window_samples: Number of samples for boundary blending (~0.5s at 100Hz)
        """
        self.blend_mode = blend_mode
        self.blend_window_samples = blend_window_samples

    def compute_statistics(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> SignalStatistics:
        """
        Compute mean, std, and covariance matrix for xyz signal.

        Args:
            x, y, z: Sensor data arrays

        Returns:
            SignalStatistics with mean, std, cov, and cholesky decomposition
        """
        data = np.stack([x, y, z], axis=0)  # (3, n_samples)

        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1)

        # Handle zero std (constant signal)
        std = np.where(std < 1e-8, 1e-8, std)

        cov = np.cov(data)

        # Regularize covariance for numerical stability
        cov = cov + np.eye(3) * 1e-6

        # Cholesky decomposition
        try:
            cholesky = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Fall back to identity if decomposition fails
            cholesky = np.eye(3)

        return SignalStatistics(
            mean=mean,
            std=std,
            cov=cov,
            cholesky=cholesky,
        )

    def transfer(
        self,
        needle: NeedleSample,
        target_stats: SignalStatistics,
    ) -> NeedleSample:
        """
        Apply covariance projection to transform needle to target style.

        Args:
            needle: Source needle sample
            target_stats: Target context statistics

        Returns:
            New NeedleSample with transformed sensor data
        """
        # Compute needle statistics
        needle_stats = self.compute_statistics(needle.x, needle.y, needle.z)

        # Stack needle data
        needle_data = np.stack([needle.x, needle.y, needle.z], axis=0)  # (3, n)

        # Normalize: (x - μ) / σ
        normalized = (needle_data - needle_stats.mean[:, None]) / needle_stats.std[:, None]

        # Project through covariance: L_target @ L_needle^{-1} @ normalized
        try:
            transform = target_stats.cholesky @ np.linalg.inv(needle_stats.cholesky)
            projected = transform @ normalized
        except np.linalg.LinAlgError:
            # Fall back to simple scaling if matrix inversion fails
            print(f"❌ Matrix inversion failed, fallback to simple scaling")
            projected = normalized

        # Denormalize with target statistics: x * σ + μ
        transferred = projected * target_stats.std[:, None] + target_stats.mean[:, None]

        return NeedleSample(
            source_pid=needle.source_pid,
            activity=needle.activity,
            start_ms=needle.start_ms,
            end_ms=needle.end_ms,
            duration_ms=needle.duration_ms,
            x=transferred[0].astype(np.float32),
            y=transferred[1].astype(np.float32),
            z=transferred[2].astype(np.float32),
        )

    def insert_with_blending(
        self,
        background: Tuple[np.ndarray, np.ndarray, np.ndarray],
        needle: Tuple[np.ndarray, np.ndarray, np.ndarray],
        position: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Insert needle into background with boundary blending.

        Args:
            background: (x, y, z) background arrays
            needle: (x, y, z) needle arrays
            position: Insertion position in samples

        Returns:
            Modified (x, y, z) arrays with needle inserted and blended
        """
        result = []

        for bg, nd in zip(background, needle):
            out = bg.copy()
            needle_len = len(nd)
            end_pos = position + needle_len

            # Ensure we don't go out of bounds
            if end_pos > len(bg):
                # Trim needle to fit
                needle_len = len(bg) - position
                nd = nd[:needle_len]
                end_pos = len(bg)

            if position < 0:
                # Trim needle start
                nd = nd[-position:]
                needle_len = len(nd)
                position = 0
                end_pos = position + needle_len

            # Calculate blend regions
            blend_len = min(self.blend_window_samples, needle_len // 4)

            if blend_len > 0:
                # Entry blend weights
                entry_weights = self._get_blend_weights(blend_len)
                # Exit blend weights (reversed)
                exit_weights = entry_weights[::-1]

                # Apply entry blending
                for i in range(blend_len):
                    idx = position + i
                    if 0 <= idx < len(out):
                        w = entry_weights[i]
                        out[idx] = (1 - w) * bg[idx] + w * nd[i]

                # Insert middle section (no blending)
                mid_start = position + blend_len
                mid_end = end_pos - blend_len
                if mid_start < mid_end:
                    out[mid_start:mid_end] = nd[blend_len : needle_len - blend_len]

                # Apply exit blending
                for i in range(blend_len):
                    idx = end_pos - blend_len + i
                    nd_idx = needle_len - blend_len + i
                    if 0 <= idx < len(out):
                        w = exit_weights[i]
                        out[idx] = w * nd[nd_idx] + (1 - w) * bg[idx]
            else:
                # No blending, just insert
                out[position:end_pos] = nd

            result.append(out.astype(np.float32))

        return tuple(result)

    def _get_blend_weights(self, n_samples: int) -> np.ndarray:
        """
        Generate blend weights based on mode.

        Args:
            n_samples: Number of samples in blend region

        Returns:
            Array of weights from 0 to 1
        """
        if n_samples <= 0:
            return np.array([])

        t = np.linspace(0, 1, n_samples)

        if self.blend_mode == "linear":
            return t
        elif self.blend_mode == "cosine":
            # Smooth S-curve: (1 - cos(π * t)) / 2
            return (1 - np.cos(np.pi * t)) / 2
        else:
            raise ValueError(f"Unknown blend mode: {self.blend_mode}")

    def compute_local_statistics(
        self,
        background: Tuple[np.ndarray, np.ndarray, np.ndarray],
        position: int,
        window_samples: int = 500,
    ) -> SignalStatistics:
        """
        Compute statistics for a local region around the insertion position.

        This provides more accurate style matching than using global background stats.

        Args:
            background: (x, y, z) background arrays
            position: Center position for local window
            window_samples: Size of local window for statistics

        Returns:
            SignalStatistics for the local region
        """
        bg_x, bg_y, bg_z = background
        n = len(bg_x)

        # Define local window
        half_window = window_samples // 2
        start = max(0, position - half_window)
        end = min(n, position + half_window)

        # Extract local region
        local_x = bg_x[start:end]
        local_y = bg_y[start:end]
        local_z = bg_z[start:end]

        # Compute statistics
        return self.compute_statistics(local_x, local_y, local_z)
