# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for LocalizationTaskGenerator.

Task 2: "When did the {activity} bout occur?"
Answer type: time_range (timestamp start-end)

Tests validate:
- Sample generation with correct timestamp answers
- Needle insertion at various positions
- Timestamp formatting correctness
"""

from pathlib import Path

import numpy as np
import pytest

from opentslm.time_series_datasets.ts_haystack.core import DifficultyConfig
from opentslm.time_series_datasets.ts_haystack.test.tasks.conftest import ensure_plot_dir


class TestLocalizationSampleGeneration:
    """Tests for localization sample generation."""

    def test_generate_single_sample(self, localization_generator, medium_difficulty, rng):
        """Test generating a single localization sample."""
        sample = localization_generator.generate_sample(medium_difficulty, rng)

        assert sample is not None
        assert sample.task_type == "localization"

        if sample.is_valid:
            assert len(sample.needles) >= 1, "Localization should have at least one needle"
            needle = sample.needles[0]

            # Check timestamps in answer
            assert needle.timestamp_start in sample.answer or needle.timestamp_end in sample.answer

            print(f"  Q: {sample.question}")
            print(f"  A: {sample.answer}")
            print(f"  Needle: {needle.activity} @ {needle.timestamp_start}-{needle.timestamp_end}")

    def test_needle_positions(self, localization_generator):
        """Test different needle position modes."""
        for position_mode in ["beginning", "middle", "end", "random"]:
            difficulty = DifficultyConfig(
                context_length_samples=10000,
                needle_position=position_mode,
                needle_length_ratio_range=(0.03, 0.15),  # 300-1500 samples for 10000 context
                background_purity="pure",
                task_specific={"margin_samples": 100},
            )

            rng = np.random.default_rng(42)
            sample = localization_generator.generate_sample(difficulty, rng)

            if sample.is_valid and sample.needles:
                needle = sample.needles[0]
                pos_frac = needle.insert_position_frac

                print(f"  Position mode '{position_mode}': needle at {pos_frac:.2%}")

                # Verify position approximately matches mode
                if position_mode == "beginning":
                    assert pos_frac < 0.4, f"Beginning should be < 40%, got {pos_frac:.2%}"
                elif position_mode == "end":
                    assert pos_frac > 0.6, f"End should be > 60%, got {pos_frac:.2%}"


class TestLocalizationVisualization:
    """Generate visualizations for human evaluation."""

    def test_visualize_localization_sample(self, localization_generator):
        """Visualize a localization sample with needle highlighted."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("localization")

        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.25),  # 500-2500 samples for 10000 context
            background_purity="pure",
        )

        # Generate sample
        sample = None
        for i in range(30):
            rng = np.random.default_rng(500 + i)
            candidate = localization_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and candidate.needles:
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        needle = sample.needles[0]

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1, 1, 0.6]})
        fig.suptitle(f"Localization Task - Find '{needle.activity}'", fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}

        for ax, (name, data, color) in zip(
            axes[:3],
            [("X", sample.x, colors["x"]), ("Y", sample.y, colors["y"]), ("Z", sample.z, colors["z"])]
        ):
            ax.plot(t, data, linewidth=0.5, color=color, alpha=0.8)
            ax.set_ylabel(f"{name}-axis (g)", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Highlight needle with boundaries
            start = needle.insert_position_samples
            end = start + needle.duration_samples
            ax.axvspan(start, end, alpha=0.3, color="red")
            ax.axvline(start, color="red", linestyle="--", alpha=0.7, linewidth=1)
            ax.axvline(end, color="red", linestyle="--", alpha=0.7, linewidth=1)

        axes[2].set_xlabel("Sample Index", fontsize=10)

        # Add position annotation on first axis
        axes[0].annotate(
            f"{needle.activity}\n{needle.timestamp_start}-{needle.timestamp_end}",
            xy=(needle.insert_position_samples + needle.duration_samples // 2, axes[0].get_ylim()[1]),
            fontsize=9, ha="center", color="red"
        )

        # Text panel
        axes[3].axis("off")
        text = (
            f"Background: {sample.background_pid}  |  "
            f"Time Range: {sample.recording_time_range[0]} - {sample.recording_time_range[1]}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}\n\n"
            f"Position: samples {needle.insert_position_samples}-{needle.insert_position_samples + needle.duration_samples} "
            f"({needle.insert_position_frac*100:.1f}% into window)"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=10,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.3))

        plt.tight_layout()
        output_path = plot_dir / "localization_sample.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")

    def test_visualize_position_modes(self, localization_generator):
        """Visualize samples with different position modes."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("localization")

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("Localization Task - Position Modes", fontsize=14, fontweight="bold")

        position_modes = ["beginning", "middle", "end", "random"]

        for ax, mode in zip(axes.flat, position_modes):
            difficulty = DifficultyConfig(
                context_length_samples=8000,
                needle_position=mode,
                needle_length_ratio_range=(0.04, 0.15),  # 320-1200 samples for 8000 context
                background_purity="pure",
                task_specific={"margin_samples": 100},
            )

            sample = None
            for i in range(20):
                rng = np.random.default_rng(600 + i + hash(mode) % 100)
                candidate = localization_generator.generate_sample(difficulty, rng)
                if candidate.is_valid and candidate.needles:
                    sample = candidate
                    break

            if sample is None:
                ax.text(0.5, 0.5, f"Could not generate\n{mode} sample",
                        transform=ax.transAxes, ha="center", va="center")
                continue

            t = np.arange(len(sample.x))
            ax.plot(t, sample.x, linewidth=0.4, alpha=0.7, color="#1f77b4")

            needle = sample.needles[0]
            start = needle.insert_position_samples
            end = start + needle.duration_samples
            ax.axvspan(start, end, alpha=0.4, color="red")

            ax.set_title(f"Position: {mode}\nNeedle at {needle.insert_position_frac*100:.1f}%", fontsize=10)
            ax.set_xlabel("Sample")
            ax.set_ylabel("X-axis (g)")

        plt.tight_layout()
        output_path = plot_dir / "localization_position_modes.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")
