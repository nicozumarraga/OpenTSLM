# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Loader for TS-Haystack benchmark data in HuggingFace Dataset format.

This module bridges the generated parquet files to the QADataset interface
by converting them to HuggingFace Dataset objects with the expected schema.

Usage:
    from opentslm.time_series_datasets.ts_haystack.dataset import load_ts_haystack_splits

    # Load single task at single context length
    train, val, test = load_ts_haystack_splits(
        tasks=["existence"],
        context_lengths_seconds=[100],
    )

    # Load multiple tasks at multiple context lengths
    train, val, test = load_ts_haystack_splits(
        tasks=["existence", "localization", "counting"],
        context_lengths_seconds=[100, 1000],
    )

    # Load with CoT rationales
    train, val, test = load_ts_haystack_splits(
        tasks=["existence"],
        context_lengths_seconds=[100],
        use_cot=True,
    )
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import polars as pl
from datasets import Dataset

from opentslm.time_series_datasets.constants import RAW_DATA
from opentslm.time_series_datasets.ts_haystack.utils import format_context_dir


# Default data directories
TS_HAYSTACK_TASKS_DIR = os.path.join(RAW_DATA, "capture24", "ts_haystack", "tasks")
TS_HAYSTACK_COT_DIR = os.path.join(RAW_DATA, "capture24", "ts_haystack", "cot")

# Sentinel value for "all context lengths"
ALL_CONTEXT_LENGTHS = "all"

# All available tasks
ALL_TASKS = [
    "existence",
    "localization",
    "counting",
    "ordering",
    "state_query",
    "antecedent",
    "comparison",
    "multi_hop",
    "anomaly_detection",
    "anomaly_localization",
]

# Columns required for QADataset
REQUIRED_COLUMNS = [
    "x_axis",
    "y_axis",
    "z_axis",
    "task_type",
    "context_length_samples",
    "recording_time_start",
    "recording_time_end",
    "question",
    "answer",
    "answer_type",
    "needles",
    "difficulty_config",
]

# Additional column for CoT datasets
COT_COLUMN = "rationale"


def get_available_tasks() -> List[str]:
    """Return list of available task names."""
    return list(ALL_TASKS)


def get_available_context_lengths(use_cot: bool = False, base_dir: Optional[Path] = None) -> List[float]:
    """
    Discover available context lengths from filesystem.

    Args:
        use_cot: If True, look in CoT directory, else tasks directory
        base_dir: Optional override for base directory

    Returns:
        List of context lengths in seconds, sorted ascending
    """
    data_dir = _get_data_dir(use_cot, base_dir)

    if not data_dir.exists():
        return []

    context_lengths = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.endswith("s"):
            # Parse directory name like "100s" or "2_56s" -> 100.0 or 2.56
            try:
                # Replace underscore with decimal point for names like "2_56s"
                name = item.name[:-1]  # Remove trailing 's'
                name = name.replace("_", ".")
                context_lengths.append(float(name))
            except ValueError:
                continue

    return sorted(context_lengths)


def _get_data_dir(use_cot: bool, base_dir: Optional[Path] = None) -> Path:
    """Get the appropriate data directory based on whether CoT is used."""
    if base_dir is not None:
        return Path(base_dir)
    return Path(TS_HAYSTACK_COT_DIR if use_cot else TS_HAYSTACK_TASKS_DIR)


def _load_parquet_file(parquet_path: Path) -> Optional[pl.DataFrame]:
    """
    Load a parquet file if it exists.

    Args:
        parquet_path: Path to the parquet file

    Returns:
        Polars DataFrame or None if file doesn't exist
    """
    if not parquet_path.exists():
        return None
    return pl.read_parquet(parquet_path)


def _normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize column names to match QADataset expected schema.

    The generated parquet files use x_axis, y_axis, z_axis already,
    but we ensure consistent column naming here.
    """
    # Map potential alternative names to standard names
    column_mapping = {}

    # Handle x/y/z -> x_axis/y_axis/z_axis if needed
    if "x" in df.columns and "x_axis" not in df.columns:
        column_mapping["x"] = "x_axis"
    if "y" in df.columns and "y_axis" not in df.columns:
        column_mapping["y"] = "y_axis"
    if "z" in df.columns and "z_axis" not in df.columns:
        column_mapping["z"] = "z_axis"

    # Handle recording_time_range -> recording_time_start/end if needed
    # (Some older generated files might have this format)
    if "recording_time_range" in df.columns:
        if "recording_time_start" not in df.columns:
            # Parse JSON-like format ["start", "end"]
            df = df.with_columns([
                pl.col("recording_time_range").str.json_path_match("$[0]").alias("recording_time_start"),
                pl.col("recording_time_range").str.json_path_match("$[1]").alias("recording_time_end"),
            ])

    if column_mapping:
        df = df.rename(column_mapping)

    return df


def _select_columns(df: pl.DataFrame, use_cot: bool) -> pl.DataFrame:
    """Select only the required columns."""
    columns_to_select = list(REQUIRED_COLUMNS)

    if use_cot and COT_COLUMN in df.columns:
        columns_to_select.append(COT_COLUMN)

    # Only select columns that exist
    available_columns = [col for col in columns_to_select if col in df.columns]

    return df.select(available_columns)


def load_ts_haystack_splits(
    tasks: List[str],
    context_lengths_seconds: List[Union[str, float, int]],
    data_dir: Optional[Path] = None,
    use_cot: bool = False,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load TS-Haystack datasets for specified tasks and context lengths.

    Directory structure expected:
        data_dir/{context_seconds}s/{task}/{split}/data.parquet

    Args:
        tasks: List of task names (e.g., ["existence", "localization"])
               Use ["all"] to load all tasks
        context_lengths_seconds: List of context lengths in seconds
                                 (e.g., [100] for 10000 samples at 100Hz)
                                 Use ["all"] to load all available context lengths
        data_dir: Base data directory. If None, uses default based on use_cot
        use_cot: If True, load from cot/ directory (with rationale column)

    Returns:
        Tuple of (train, val, test) HuggingFace Dataset objects

    Example:
        >>> train, val, test = load_ts_haystack_splits(
        ...     tasks=["existence", "counting"],
        ...     context_lengths_seconds=[100],
        ... )
        >>> print(f"Train: {len(train)} samples")
    """
    # Resolve tasks
    if tasks == ["all"] or "all" in tasks:
        tasks = ALL_TASKS

    # Resolve context lengths - support "all" to auto-discover
    if context_lengths_seconds == ["all"] or "all" in context_lengths_seconds:
        context_lengths_seconds = get_available_context_lengths(use_cot, data_dir)
        if not context_lengths_seconds:
            raise ValueError(
                f"No context length directories found in "
                f"{_get_data_dir(use_cot, data_dir)}. "
                f"Make sure datasets have been generated first."
            )
        print(f"Auto-discovered context lengths: {context_lengths_seconds}")

    # Validate tasks
    for task in tasks:
        if task not in ALL_TASKS:
            raise ValueError(
                f"Unknown task: {task}. Available tasks: {ALL_TASKS}"
            )

    # Get data directory
    base_dir = _get_data_dir(use_cot, data_dir)

    # Collect dataframes for each split
    split_dfs: Dict[str, List[pl.DataFrame]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    # Load parquet files
    for ctx_seconds in context_lengths_seconds:
        ctx_dir = format_context_dir(ctx_seconds)

        for task in tasks:
            for split in ["train", "val", "test"]:
                parquet_path = base_dir / ctx_dir / task / split / "data.parquet"

                df = _load_parquet_file(parquet_path)

                if df is not None:
                    # Normalize and select columns
                    df = _normalize_columns(df)
                    df = _select_columns(df, use_cot)
                    split_dfs[split].append(df)
                    print(f"Loaded {parquet_path.relative_to(base_dir)}: {len(df)} samples")
                else:
                    print(f"Warning: {parquet_path} not found, skipping")

    # Concatenate and convert to HuggingFace Dataset
    datasets = []
    for split in ["train", "val", "test"]:
        dfs = split_dfs[split]

        if dfs:
            # Concatenate all dataframes for this split
            combined_df = pl.concat(dfs, how="diagonal")  # diagonal handles different columns

            # Convert to pandas, then to HuggingFace Dataset
            pandas_df = combined_df.to_pandas()
            hf_dataset = Dataset.from_pandas(pandas_df)
            datasets.append(hf_dataset)

            print(f"Total {split}: {len(hf_dataset)} samples")
        else:
            # Create empty dataset with correct schema
            print(f"Warning: No data found for {split} split, creating empty dataset")
            empty_schema = {col: [] for col in REQUIRED_COLUMNS}
            if use_cot:
                empty_schema[COT_COLUMN] = []
            empty_df = pd.DataFrame(empty_schema)
            datasets.append(Dataset.from_pandas(empty_df))

    return tuple(datasets)


def get_task_distribution(dataset: Dataset) -> Dict[str, int]:
    """
    Get task type distribution for a dataset.

    Args:
        dataset: HuggingFace Dataset object

    Returns:
        Dictionary mapping task types to counts
    """
    if len(dataset) == 0:
        return {}

    task_types = dataset["task_type"]
    return dict(pd.Series(task_types).value_counts())


def get_answer_type_distribution(dataset: Dataset) -> Dict[str, int]:
    """
    Get answer type distribution for a dataset.

    Args:
        dataset: HuggingFace Dataset object

    Returns:
        Dictionary mapping answer types to counts
    """
    if len(dataset) == 0:
        return {}

    answer_types = dataset["answer_type"]
    return dict(pd.Series(answer_types).value_counts())


def print_dataset_info(dataset: Dataset, name: str) -> None:
    """
    Print information about a dataset split.

    Args:
        dataset: The dataset split
        name: Name of the split (e.g., "Train")
    """
    print(f"\n{name} dataset:")
    print(f"  Total samples: {len(dataset)}")

    if len(dataset) == 0:
        return

    # Task distribution
    task_dist = get_task_distribution(dataset)
    if task_dist:
        print(f"  Task distribution:")
        for task, count in sorted(task_dist.items()):
            print(f"    {task}: {count} ({count/len(dataset)*100:.1f}%)")

    # Answer type distribution
    answer_dist = get_answer_type_distribution(dataset)
    if answer_dist:
        print(f"  Answer type distribution:")
        for answer_type, count in sorted(answer_dist.items()):
            print(f"    {answer_type}: {count} ({count/len(dataset)*100:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("TS-Haystack QA Loader Demo")
    print("=" * 60)

    # Demo configuration
    tasks = ["existence", "localization"]
    context_lengths = [100]

    print(f"\nConfiguration:")
    print(f"  Tasks: {tasks}")
    print(f"  Context lengths (seconds): {context_lengths}")

    # Load the dataset splits
    print("\nLoading dataset splits...")
    try:
        train_ds, val_ds, test_ds = load_ts_haystack_splits(
            tasks=tasks,
            context_lengths_seconds=context_lengths,
        )

        # Print dataset information
        print_dataset_info(train_ds, "Train")
        print_dataset_info(val_ds, "Validation")
        print_dataset_info(test_ds, "Test")

        # Show sample data
        if len(train_ds) > 0:
            print("\n" + "=" * 50)
            print("Sample data from training set:")
            sample = train_ds[0]
            for key, value in sample.items():
                if key in ["x_axis", "y_axis", "z_axis"]:
                    if isinstance(value, list) and len(value) > 0:
                        print(f"  {key}: {value[:5]}... (length: {len(value)})")
                    else:
                        print(f"  {key}: {value}")
                elif key in ["needles", "difficulty_config"]:
                    if isinstance(value, str) and len(value) > 80:
                        print(f"  {key}: {value[:80]}...")
                    else:
                        print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure TS-Haystack datasets have been generated first.")

    print("\n" + "=" * 60)
    print("Loader demo complete!")
    print("=" * 60)
