# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
TS-Haystack QADataset implementation for OpenTSLM training.

This module provides a QADataset subclass that formats TS-Haystack benchmark
data for use with the OpenTSLM Flamingo training pipeline.

IMPORTANT: OpenTSLMFlamingo uses <|endofchunk|> as the answer terminator,
NOT the tokenizer's eos_token (<|end_of_text|>). This dataset defaults to
<|endofchunk|> to ensure proper training behavior.

Usage:
    from opentslm.time_series_datasets.ts_haystack.dataset import TSHaystackQADataset

    # Single task at single context length (uses default <|endofchunk|> EOS)
    dataset = TSHaystackQADataset(
        split="train",
        tasks=["existence"],
        context_lengths_seconds=[100],
    )

    # Multi-task training
    dataset = TSHaystackQADataset(
        split="train",
        tasks=["existence", "localization", "counting"],
        context_lengths_seconds=[100, 1000],
    )
"""

import warnings
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.ts_haystack.dataset.ts_haystack_qa_loader import (
    ALL_TASKS,
    load_ts_haystack_splits,
)


# Axis labels matching HAR/Capture24 format
TIME_SERIES_LABELS = [
    "The following is the accelerometer data on the x-axis",
    "The following is the accelerometer data on the y-axis",
    "The following is the accelerometer data on the z-axis",
]

# Answer format guidance per answer type
ANSWER_FORMAT_GUIDANCE = {
    "boolean": "Answer with 'Yes' or 'No'.",
    "integer": "Answer with a number.",
    "category": "Answer with the activity name.",
    "time_range": "Answer with the time range in the format 'HH:MM AM/PM to HH:MM AM/PM'.",
    "timestamp": "Answer with the time in the format 'HH:MM AM/PM'.",
}


class TSHaystackQADataset(QADataset):
    """
    QADataset subclass for TS-Haystack benchmark.

    This class loads TS-Haystack benchmark data and formats it for training
    with the OpenTSLM Flamingo model. It supports multiple tasks and context
    lengths for curriculum learning and multi-task training.

    Note on caching:
        The QADataset base class uses class-level caching. Once data is loaded
        for a specific configuration, it's cached for all subsequent instances.
        If you need different configurations, restart the Python session.

    Args:
        split: Dataset split to load ("train", "test", or "validation")
        EOS_TOKEN: End-of-sequence token for the model. Defaults to "<|endofchunk|>"
                   which is the correct terminator for OpenTSLMFlamingo.
                   WARNING: Do NOT use tokenizer.eos_token (<|end_of_text|>).
        tasks: List of tasks to load (e.g., ["existence", "localization"])
               Use ["all"] to load all tasks
        context_lengths_seconds: List of context lengths in seconds
                                 (e.g., [100] for 10000 samples at 100Hz)
        format_sample_str: If True, format samples as strings (default: False)
        time_series_format_function: Optional function to format time series as strings

    Example:
        >>> dataset = TSHaystackQADataset(
        ...     split="train",
        ...     tasks=["existence", "counting"],
        ...     context_lengths_seconds=[100],
        ... )
        >>> print(f"Dataset size: {len(dataset)}")
        >>> sample = dataset[0]
        >>> print(f"Answer: {sample['answer']}")
    """

    # OpenTSLMFlamingo uses <|endofchunk|> as the answer terminator
    FLAMINGO_EOS_TOKEN = "<|endofchunk|>"

    # Class-level storage for configuration caching
    _cached_config: Optional[tuple] = None

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str = None,
        tasks: List[str] = None,
        context_lengths_seconds: List[int] = None,
        format_sample_str: bool = False,
        time_series_format_function: Callable[[np.ndarray], str] | None = None,
    ):
        # Default to the correct EOS token for OpenTSLMFlamingo
        if EOS_TOKEN is None:
            EOS_TOKEN = self.FLAMINGO_EOS_TOKEN
        elif EOS_TOKEN != self.FLAMINGO_EOS_TOKEN:
            # Warn if using a different EOS token (likely tokenizer.eos_token by mistake)
            if "end_of_text" in EOS_TOKEN or EOS_TOKEN == "":
                warnings.warn(
                    f"EOS_TOKEN='{EOS_TOKEN}' may cause training issues. "
                    f"OpenTSLMFlamingo expects '<|endofchunk|>' as the answer terminator. "
                    f"Consider using the default EOS_TOKEN or explicitly set "
                    f"EOS_TOKEN='<|endofchunk|>'.",
                    UserWarning,
                    stacklevel=2,
                )
        # Set defaults
        if tasks is None:
            tasks = ["all"]
        if context_lengths_seconds is None:
            context_lengths_seconds = [100]

        # Resolve "all" tasks
        if "all" in tasks:
            self.tasks = list(ALL_TASKS)
        else:
            self.tasks = list(tasks)

        self.context_lengths_seconds = list(context_lengths_seconds)

        # Check for configuration mismatch with cached data
        self._check_cache_config()

        super().__init__(
            split, EOS_TOKEN, format_sample_str, time_series_format_function
        )

    def _check_cache_config(self) -> None:
        """
        Check if cached data exists with different configuration.

        Warns the user if they're creating an instance with different parameters
        than the originally cached data.
        """
        current_config = (
            tuple(sorted(self.tasks)),
            tuple(sorted(self.context_lengths_seconds)),
        )

        if self.__class__._cached_config is None:
            # First time loading - store the configuration
            self.__class__._cached_config = current_config
        elif self.__class__._cached_config != current_config:
            # Configuration mismatch - warn user
            warnings.warn(
                f"TSHaystackQADataset was previously loaded with config "
                f"(tasks={self.__class__._cached_config[0]}, "
                f"context_lengths_seconds={self.__class__._cached_config[1]}). "
                f"Now instantiated with (tasks={current_config[0]}, "
                f"context_lengths_seconds={current_config[1]}). "
                f"Using cached data from original config. "
                f"Restart Python session for different configuration.",
                UserWarning,
                stacklevel=3,
            )

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load the TS-Haystack dataset splits.

        Returns:
            Tuple of (train, validation, test) HuggingFace Dataset objects
        """
        return load_ts_haystack_splits(
            tasks=self.tasks,
            context_lengths_seconds=self.context_lengths_seconds,
            use_cot=False,
        )

    def _get_answer(self, row) -> str:
        """Return the answer from the row."""
        return row["answer"]

    def _get_pre_prompt(self, row) -> str:
        """
        Return the instruction text before the time series.

        Includes recording time context and the question.
        """
        time_start = row.get("recording_time_start", "unknown")
        time_end = row.get("recording_time_end", "unknown")

        return (
            f"You are given accelerometer data in all three dimensions from a "
            f"wrist-worn sensor. The recording spans from {time_start} to {time_end}.\n\n"
            f"Question: {row['question']}"
        )

    def _get_post_prompt(self, row) -> str:
        """
        Return the instruction text after the time series.

        Provides task-specific answer format guidance.
        """
        answer_type = row.get("answer_type", "category")
        guidance = ANSWER_FORMAT_GUIDANCE.get(answer_type, "Provide your answer.")

        return (
            f"\nInstructions:\n"
            f"- Analyze the accelerometer data carefully.\n"
            f"- Think step-by-step about what the signal patterns indicate.\n"
            f"- {guidance}\n"
            f'- End your response with "Answer: <your answer>"'
        )

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        Convert the time series data into TextTimeSeriesPrompt objects.

        Args:
            row: Dataset row containing x_axis, y_axis, z_axis

        Returns:
            List of TextTimeSeriesPrompt objects, one per axis
        """
        series = torch.tensor(
            [
                row["x_axis"],
                row["y_axis"],
                row["z_axis"],
            ],
            dtype=torch.float32,
        )

        return [
            TextTimeSeriesPrompt(time_series_label, time_series)
            for time_series_label, time_series in zip(
                TIME_SERIES_LABELS, series.tolist()
            )
        ]

    def get_tasks(self) -> List[str]:
        """
        Return the list of tasks loaded.

        Returns:
            List of task names
        """
        return list(self.tasks)

    def get_context_lengths(self) -> List[int]:
        """
        Return the list of context lengths (in seconds).

        Returns:
            List of context lengths in seconds
        """
        return list(self.context_lengths_seconds)

    def _format_sample(self, row):
        """
        Format a sample for training, including metadata.

        Overrides the base class method to include task metadata
        and raw accelerometer data in the formatted sample.

        Args:
            row: Dataset row

        Returns:
            Dictionary with formatted prompt and metadata
        """
        sample = super()._format_sample(row)

        # Add raw data
        sample["x_axis"] = row["x_axis"]
        sample["y_axis"] = row["y_axis"]
        sample["z_axis"] = row["z_axis"]

        # Add metadata
        sample["task_type"] = row.get("task_type", "unknown")
        sample["answer_type"] = row.get("answer_type", "unknown")
        sample["context_length_samples"] = row.get("context_length_samples", 0)
        sample["question"] = row.get("question", "")

        # Keep needles and difficulty_config for analysis
        sample["needles"] = row.get("needles", "[]")
        sample["difficulty_config"] = row.get("difficulty_config", "{}")

        return sample


if __name__ == "__main__":
    print("=" * 60)
    print("TS-Haystack QADataset Demo")
    print("=" * 60)

    # Configuration
    tasks = ["existence", "localization"]
    context_lengths = [100]

    print(f"\nConfiguration:")
    print(f"  Tasks: {tasks}")
    print(f"  Context lengths (seconds): {context_lengths}")

    # Create datasets
    print("\nCreating datasets...")
    try:
        dataset = TSHaystackQADataset(
            split="train",
            EOS_TOKEN="",
            tasks=tasks,
            context_lengths_seconds=context_lengths,
        )
        dataset_val = TSHaystackQADataset(
            split="validation",
            EOS_TOKEN="",
            tasks=tasks,
            context_lengths_seconds=context_lengths,
        )
        dataset_test = TSHaystackQADataset(
            split="test",
            EOS_TOKEN="",
            tasks=tasks,
            context_lengths_seconds=context_lengths,
        )

        print(
            f"\nDataset sizes: Train: {len(dataset)}, "
            f"Validation: {len(dataset_val)}, Test: {len(dataset_test)}"
        )

        # Show configuration
        print(f"\nLoaded tasks: {dataset.get_tasks()}")
        print(f"Context lengths: {dataset.get_context_lengths()}")

        # Test sample access
        if len(dataset) > 0:
            print("\n" + "=" * 50)
            print("Sample from training set:")
            sample = dataset[0]
            print(f"  Keys: {list(sample.keys())}")
            print(f"  Task type: {sample.get('task_type', 'N/A')}")
            print(f"  Answer type: {sample.get('answer_type', 'N/A')}")
            print(f"  Question: {sample.get('question', 'N/A')}")
            print(f"  Answer: {sample['answer']}")
            if "time_series" in sample:
                print(f"  Time series count: {len(sample['time_series'])}")
                for i, ts in enumerate(sample["time_series"]):
                    print(f"    Axis {i}: length={len(ts)}, first 3 values={ts[:3]}")

        # Test DataLoader integration
        if len(dataset_test) > 0:
            print("\n" + "=" * 50)
            print("Testing DataLoader integration...")
            from opentslm.time_series_datasets.util import (
                extend_time_series_to_match_patch_size_and_aggregate,
            )

            dataloader = DataLoader(
                dataset_test,
                batch_size=2,
                shuffle=True,
                collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                    batch, patch_size=4
                ),
            )

            for batch in tqdm(dataloader, total=1, desc="Testing batch"):
                print(f"  Batch keys: {list(batch[0].keys())}")
                print(f"  Batch answer: {batch[0]['answer']}")
                break

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure TS-Haystack datasets have been generated first.")
        print("Run: python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_dataset --help")

    print("\n" + "=" * 60)
    print("QADataset demo complete!")
    print("=" * 60)
