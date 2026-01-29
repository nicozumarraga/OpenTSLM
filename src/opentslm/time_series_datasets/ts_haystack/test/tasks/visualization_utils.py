# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Visualization utilities for TS-Haystack task testing.

Provides functions to visualize generated samples for human evaluation.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from opentslm.time_series_datasets.ts_haystack.core import GeneratedSample


# Output directory for plots
PLOTS_BASE_DIR = Path(__file__).parent.parent / "plots"


def ensure_plot_dir(task_type: str) -> Path:
    """Create and return the plot output directory for a task type."""
    plot_dir = PLOTS_BASE_DIR / task_type
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def format_sample_for_display(sample: GeneratedSample) -> str:
    """
    Format a sample for text display, truncating signal data.

    Args:
        sample: Generated sample to format

    Returns:
        Formatted string for display
    """
    # Create signal preview
    def truncate_signal(arr: np.ndarray, max_vals: int = 5) -> str:
        if len(arr) <= max_vals * 2:
            vals = ", ".join(f"{v:.3f}" for v in arr)
        else:
            head = ", ".join(f"{v:.3f}" for v in arr[:max_vals])
            tail = ", ".join(f"{v:.3f}" for v in arr[-max_vals:])
            vals = f"{head}, ... ({len(arr)} total) ..., {tail}"
        return vals

    # Format needle info
    needle_info = []
    for i, needle in enumerate(sample.needles):
        needle_info.append(
            f"  {i+1}. {needle.activity} @ {needle.insert_position_samples}-"
            f"{needle.insert_position_samples + needle.duration_samples} "
            f"({needle.timestamp_start} - {needle.timestamp_end})"
        )
    needles_str = "\n".join(needle_info) if needle_info else "  (none)"

    return f"""
========================================
TASK: {sample.task_type.upper()}
========================================

CONTEXT:
  Background PID: {sample.background_pid}
  Time Range: {sample.recording_time_range[0]} - {sample.recording_time_range[1]}
  Context Length: {sample.context_length_samples} samples
  Answer Type: {sample.answer_type}

INSERTED NEEDLES:
{needles_str}

SIGNAL:
  X: [{truncate_signal(sample.x)}]
  Y: [{truncate_signal(sample.y)}]
  Z: [{truncate_signal(sample.z)}]

----------------------------------------
QUESTION:
{sample.question}

ANSWER:
{sample.answer}
----------------------------------------

VALID: {sample.is_valid}
{f'Notes: {sample.validation_notes}' if sample.validation_notes else ''}
========================================
"""


def create_sample_report(
    samples: List[GeneratedSample],
    task_type: str,
    output_name: str = "samples_report",
) -> Path:
    """
    Create a text report summarizing generated samples.

    Args:
        samples: List of samples
        task_type: Task name
        output_name: Output file name (without extension)

    Returns:
        Path to saved report
    """
    plot_dir = ensure_plot_dir(task_type)
    output_path = plot_dir / f"{output_name}.txt"

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"TS-Haystack Task Report: {task_type.upper()}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Samples: {len(samples)}\n")
        valid_count = sum(1 for s in samples if s.is_valid)
        f.write(f"Valid Samples: {valid_count}/{len(samples)}\n\n")

        # Aggregate statistics
        context_lengths = set(s.context_length_samples for s in samples)
        f.write(f"Context Lengths: {sorted(context_lengths)}\n")

        activities = set()
        for s in samples:
            for n in s.needles:
                activities.add(n.activity)
        f.write(f"Activities Used: {sorted(activities)}\n\n")

        # Individual samples
        f.write("-" * 80 + "\n")
        f.write("INDIVIDUAL SAMPLES\n")
        f.write("-" * 80 + "\n\n")

        for sample in samples:
            f.write(format_sample_for_display(sample))
            f.write("\n")

    return output_path
