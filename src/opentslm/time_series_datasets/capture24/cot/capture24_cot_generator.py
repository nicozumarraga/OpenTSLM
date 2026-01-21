# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Capture24 Chain-of-Thought (CoT) Dataset Generator

This script generates chain-of-thought rationales for Capture24 accelerometer
classification using an LLM (Gemini 2.5 Flash). Following the HAR-CoT paper's
strategy, each sample uses binary classification with the correct label and
a dissimilar label.

Usage:
    python -m opentslm.time_series_datasets.capture24.cot.capture24_cot_generator \
        --train-samples 10000 --val-samples 2000 --test-samples 2000

Prerequisites:
    1. Classification data must exist at:
       data/capture24/classification/{window_size}s_{hz}hz/{label_scheme}/
    2. GOOGLE_API_KEY environment variable must be set for Gemini API access
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import polars as pl
from tqdm import tqdm

from opentslm.time_series_datasets.capture24.capture24_classification import (
    get_classification_path,
    load_classification_dataset,
    load_classification_metadata,
)
from opentslm.time_series_datasets.constants import RAW_DATA


# ---------------------------
# Constants
# ---------------------------

CAPTURE24_COT_DATA_DIR = os.path.join(RAW_DATA, "capture24_cot")

# Dissimilar label mapping for Walmsley2020 scheme (4 classes)
# Each label maps to labels that represent very different activities
CAPTURE24_DISSIMILAR_MAPPING = {
    "sleep": ["light", "moderate-vigorous"],
    "sedentary": ["light", "moderate-vigorous"],
    "light": ["sleep", "sedentary", "moderate-vigorous"],
    "moderate-vigorous": ["sleep", "sedentary"],
}

# Default configuration - matches HAR CoT format: 2.56s @ 50Hz = 128 samples
DEFAULT_WINDOW_SIZE_S = 2.56
DEFAULT_EFFECTIVE_HZ = 50
DEFAULT_LABEL_SCHEME = "Walmsley2020"
DEFAULT_SAMPLES = {"train": 10000, "val": 2000, "test": 2000}

# Prompt template for CoT generation
COT_PROMPT_TEMPLATE = """You are shown accelerometer data from a wrist-worn sensor over a {window_duration:.2f} second window.
This data corresponds to one of two possible activities:
- {correct_activity}
- {dissimilar_activity}

The accelerometer readings are provided for three axes (X, Y, Z).

X-axis data (mean={x_mean:.4f}, std={x_std:.4f}):
{x_data}

Y-axis data (mean={y_mean:.4f}, std={y_std:.4f}):
{y_data}

Z-axis data (mean={z_mean:.4f}, std={z_std:.4f}):
{z_data}

Your task is to classify the activity based on analysis of the data.

Instructions:
- Begin by analyzing the time series without assuming a specific label.
- Think step-by-step about what the observed patterns suggest regarding movement intensity and behavior.
- Write your rationale as a single, natural paragraph - do not use bullet points, numbered steps, or section headings.
- Do not refer back to the data or to the act of visual analysis in your rationale; reason about the time-series patterns directly.
- Do **not** assume any answer at the beginning - analyze as if you do not yet know which class is correct.
- Do **not** mention either class label until the final sentence.
- Make sure that your last word is the answer. You MUST end your response with "Answer: {correct_activity}"
"""


# ---------------------------
# Data Classes
# ---------------------------

@dataclass
class GenerationConfig:
    """Configuration for CoT generation."""
    window_size_s: float = DEFAULT_WINDOW_SIZE_S
    effective_hz: int = DEFAULT_EFFECTIVE_HZ
    label_scheme: str = DEFAULT_LABEL_SCHEME
    train_samples: int = DEFAULT_SAMPLES["train"]
    val_samples: int = DEFAULT_SAMPLES["val"]
    test_samples: int = DEFAULT_SAMPLES["test"]
    batch_size: int = 50
    checkpoint_every: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0
    api_delay: float = 0.1  # Delay between API calls
    output_dir: str = CAPTURE24_COT_DATA_DIR
    seed: int = 42


@dataclass
class GenerationStats:
    """Statistics for tracking generation progress."""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    validation_failed: int = 0
    api_errors: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_processed": self.total_processed,
            "successful": self.successful,
            "failed": self.failed,
            "validation_failed": self.validation_failed,
            "api_errors": self.api_errors,
            "class_counts": self.class_counts,
        }


# ---------------------------
# API Client
# ---------------------------

class GeminiClient:
    """Client for Google Gemini API calls."""

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        self.model = model
        self._client = None

    def _ensure_client(self):
        """Lazily initialize the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai

                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GOOGLE_API_KEY environment variable is not set. "
                        "Please set it to use the Gemini API."
                    )

                genai.configure(api_key=api_key)
                self._client = genai.GenerativeModel(self.model)
                print(f"Initialized Gemini client with model: {self.model}")
            except ImportError:
                raise ImportError(
                    "google-generativeai package is required. "
                    "Install with: pip install google-generativeai"
                )

    def generate(self, prompt: str, temperature: float = 0.7) -> Optional[str]:
        """
        Generate text using Gemini API.

        Args:
            prompt: The prompt to send to the model
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            Generated text or None if failed
        """
        self._ensure_client()

        try:
            response = self._client.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": 512,
                }
            )

            if response.text:
                return response.text.strip()
            return None

        except Exception as e:
            print(f"Gemini API error: {e}")
            return None


# ---------------------------
# Generator Class
# ---------------------------

class Capture24CoTGenerator:
    """
    Generator for Capture24 Chain-of-Thought dataset.

    This class handles:
    1. Stratified sampling from classification parquet files
    2. Prompt creation with time series data
    3. LLM API calls for rationale generation
    4. Validation and checkpointing
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.api_client = GeminiClient()

        # Set random seed for reproducibility
        random.seed(config.seed)

        # Validate label scheme
        if config.label_scheme not in CAPTURE24_DISSIMILAR_MAPPING.keys() | {"Walmsley2020"}:
            # For Walmsley2020, the labels match our mapping
            pass

        # Get labels for the scheme
        self.labels = list(CAPTURE24_DISSIMILAR_MAPPING.keys())

    def sample_windows(
        self,
        split: str,
        n_samples: int,
        stratified: bool = True
    ) -> pd.DataFrame:
        """
        Sample windows from classification dataset.

        Args:
            split: Dataset split ("train", "val", or "test")
            n_samples: Total number of samples to select
            stratified: If True, sample equal numbers per class

        Returns:
            DataFrame with sampled windows
        """
        # Load classification data
        df = load_classification_dataset(
            window_size_s=self.config.window_size_s,
            effective_hz=self.config.effective_hz,
            label_scheme=self.config.label_scheme,
            split=split
        )

        # Convert to pandas for easier sampling
        pdf = df.to_pandas()

        if stratified:
            # Calculate samples per class
            n_classes = len(self.labels)
            samples_per_class = n_samples // n_classes
            remainder = n_samples % n_classes

            sampled_dfs = []
            for i, label in enumerate(self.labels):
                label_df = pdf[pdf["label"] == label]

                # Add remainder samples to first few classes
                n_to_sample = samples_per_class + (1 if i < remainder else 0)
                n_to_sample = min(n_to_sample, len(label_df))

                if n_to_sample > 0:
                    sampled_dfs.append(label_df.sample(n=n_to_sample, random_state=self.config.seed + i))

            if sampled_dfs:
                result = pd.concat(sampled_dfs, ignore_index=True)
                # Shuffle the combined result
                result = result.sample(frac=1, random_state=self.config.seed).reset_index(drop=True)
                return result
            else:
                return pd.DataFrame()
        else:
            n_to_sample = min(n_samples, len(pdf))
            return pdf.sample(n=n_to_sample, random_state=self.config.seed).reset_index(drop=True)

    def get_dissimilar_label(self, correct_label: str) -> str:
        """
        Select a random dissimilar label for binary classification.

        Args:
            correct_label: The true label for the sample

        Returns:
            A dissimilar label from the mapping
        """
        dissimilar_options = CAPTURE24_DISSIMILAR_MAPPING.get(correct_label, [])
        if not dissimilar_options:
            # Fallback: pick any other label
            other_labels = [l for l in self.labels if l != correct_label]
            return random.choice(other_labels)
        return random.choice(dissimilar_options)

    def create_prompt(self, row: dict, dissimilar_label: str) -> str:
        """
        Build the full prompt with time series data and binary classification task.

        Args:
            row: Dictionary containing window data (x, y, z, label)
            dissimilar_label: The dissimilar label for binary classification

        Returns:
            Formatted prompt string
        """
        x_data = row["x"]
        y_data = row["y"]
        z_data = row["z"]
        correct_label = row["label"]

        # Calculate statistics
        import numpy as np
        x_arr = np.array(x_data)
        y_arr = np.array(y_data)
        z_arr = np.array(z_data)

        # Calculate window duration
        window_duration = len(x_data) / self.config.effective_hz

        # Format time series as compact string (first 50 values with ellipsis if longer)
        def format_series(arr, max_values=50):
            if len(arr) <= max_values:
                return ", ".join(f"{v:.4f}" for v in arr)
            else:
                return ", ".join(f"{v:.4f}" for v in arr[:max_values]) + f", ... ({len(arr)} total values)"

        prompt = COT_PROMPT_TEMPLATE.format(
            window_duration=window_duration,
            correct_activity=correct_label,
            dissimilar_activity=dissimilar_label,
            x_data=format_series(x_arr),
            y_data=format_series(y_arr),
            z_data=format_series(z_arr),
            x_mean=x_arr.mean(),
            x_std=x_arr.std(),
            y_mean=y_arr.mean(),
            y_std=y_arr.std(),
            z_mean=z_arr.mean(),
            z_std=z_arr.std(),
        )

        return prompt

    def generate_rationale(self, prompt: str) -> Optional[str]:
        """
        Call LLM API to generate rationale.

        Args:
            prompt: The full prompt to send

        Returns:
            Generated rationale or None if failed
        """
        for attempt in range(self.config.max_retries):
            result = self.api_client.generate(prompt)
            if result:
                return result

            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay * (attempt + 1))

        return None

    def validate_rationale(self, rationale: str, expected_label: str) -> bool:
        """
        Verify rationale ends with 'Answer: {label}'.

        Args:
            rationale: The generated rationale text
            expected_label: The expected answer label

        Returns:
            True if rationale is valid, False otherwise
        """
        if not rationale:
            return False

        # Check for proper ending (allow for trailing punctuation)
        rationale_lower = rationale.strip().lower()
        expected_endings = [
            f"answer: {expected_label}".lower(),
            f"answer: {expected_label}.".lower(),
        ]

        for ending in expected_endings:
            if rationale_lower.endswith(ending):
                return True

        return False

    def _save_checkpoint(
        self,
        results: List[dict],
        split: str,
        checkpoint_num: int
    ):
        """Save intermediate checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints" / split
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_{checkpoint_num:04d}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(results, f)

        print(f"  Checkpoint saved: {checkpoint_path} ({len(results)} samples)")

    def _save_final(
        self,
        results: List[dict],
        split: str,
        stats: GenerationStats
    ):
        """Save final CSV file for a split."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"capture24_cot_{split}.csv"

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Convert lists to JSON strings for CSV storage
        df["x_axis"] = df["x_axis"].apply(json.dumps)
        df["y_axis"] = df["y_axis"].apply(json.dumps)
        df["z_axis"] = df["z_axis"].apply(json.dumps)

        # Save CSV
        df.to_csv(output_path, index=False)
        print(f"Saved {split} dataset: {output_path} ({len(df)} samples)")

        return output_path

    def generate_split(
        self,
        split: str,
        n_samples: int,
        resume_from: Optional[int] = None
    ) -> Tuple[List[dict], GenerationStats]:
        """
        Generate CoT rationales for a dataset split.

        Args:
            split: Dataset split ("train", "val", or "test")
            n_samples: Number of samples to generate
            resume_from: Optional checkpoint number to resume from

        Returns:
            Tuple of (results list, generation stats)
        """
        print(f"\n{'='*60}")
        print(f"Generating {split} split ({n_samples} samples)")
        print(f"{'='*60}")

        stats = GenerationStats()
        results = []

        # Load any existing checkpoints if resuming
        if resume_from is not None:
            checkpoint_dir = Path(self.config.output_dir) / "checkpoints" / split
            for i in range(resume_from + 1):
                checkpoint_path = checkpoint_dir / f"checkpoint_{i:04d}.json"
                if checkpoint_path.exists():
                    with open(checkpoint_path) as f:
                        results.extend(json.load(f))
            print(f"Resumed from checkpoint {resume_from} with {len(results)} existing samples")

        # Sample windows
        samples_df = self.sample_windows(split, n_samples)
        print(f"Sampled {len(samples_df)} windows for processing")

        # Skip already processed samples if resuming
        start_idx = len(results)

        checkpoint_num = (start_idx // self.config.checkpoint_every) + 1

        # Process each sample
        for idx in tqdm(range(start_idx, len(samples_df)), desc=f"  {split}"):
            row = samples_df.iloc[idx].to_dict()
            stats.total_processed += 1

            # Get dissimilar label for binary classification
            dissimilar_label = self.get_dissimilar_label(row["label"])

            # Create prompt
            prompt = self.create_prompt(row, dissimilar_label)

            # Generate rationale
            rationale = self.generate_rationale(prompt)

            if rationale is None:
                stats.api_errors += 1
                stats.failed += 1
                continue

            # Validate rationale
            if not self.validate_rationale(rationale, row["label"]):
                stats.validation_failed += 1
                stats.failed += 1
                continue

            # Success - add to results
            results.append({
                "x_axis": row["x"],
                "y_axis": row["y"],
                "z_axis": row["z"],
                "label": row["label"],
                "prompt": prompt,
                "rationale": rationale,
            })

            stats.successful += 1
            stats.class_counts[row["label"]] = stats.class_counts.get(row["label"], 0) + 1

            # Checkpoint
            if len(results) % self.config.checkpoint_every == 0:
                self._save_checkpoint(results, split, checkpoint_num)
                checkpoint_num += 1

            # Rate limiting
            time.sleep(self.config.api_delay)

        # Final save
        self._save_final(results, split, stats)

        return results, stats

    def generate_all(self) -> Dict[str, GenerationStats]:
        """
        Generate CoT rationales for all splits.

        Returns:
            Dictionary mapping split names to generation stats
        """
        all_stats = {}

        splits_config = {
            "train": self.config.train_samples,
            "val": self.config.val_samples,
            "test": self.config.test_samples,
        }

        for split, n_samples in splits_config.items():
            if n_samples > 0:
                _, stats = self.generate_split(split, n_samples)
                all_stats[split] = stats

        # Save metadata
        self._save_metadata(all_stats)

        return all_stats

    def _save_metadata(self, all_stats: Dict[str, GenerationStats]):
        """Save generation metadata."""
        output_dir = Path(self.config.output_dir)
        metadata_path = output_dir / "metadata.json"

        metadata = {
            "window_size_s": self.config.window_size_s,
            "effective_hz": self.config.effective_hz,
            "samples_per_window": int(self.config.window_size_s * self.config.effective_hz),
            "label_scheme": self.config.label_scheme,
            "labels": self.labels,
            "llm_model": self.api_client.model,
            "generation_date": datetime.now().isoformat(),
            "dissimilar_mapping": CAPTURE24_DISSIMILAR_MAPPING,
            "samples": {
                split: stats.successful
                for split, stats in all_stats.items()
            },
            "class_distribution": {
                split: stats.class_counts
                for split, stats in all_stats.items()
            },
            "generation_stats": {
                split: stats.to_dict()
                for split, stats in all_stats.items()
            },
            "config": {
                "batch_size": self.config.batch_size,
                "checkpoint_every": self.config.checkpoint_every,
                "max_retries": self.config.max_retries,
                "seed": self.config.seed,
            }
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nMetadata saved: {metadata_path}")


# ---------------------------
# CLI Entry Point
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Capture24 Chain-of-Thought dataset"
    )
    parser.add_argument(
        "--window-size-s", "-w",
        type=float,
        default=DEFAULT_WINDOW_SIZE_S,
        help=f"Window size in seconds, e.g., 2.56 for HAR CoT compatibility (default: {DEFAULT_WINDOW_SIZE_S})"
    )
    parser.add_argument(
        "--effective-hz", "-e",
        type=int,
        default=DEFAULT_EFFECTIVE_HZ,
        help=f"Effective sampling frequency in Hz (default: {DEFAULT_EFFECTIVE_HZ})"
    )
    parser.add_argument(
        "--label-scheme", "-l",
        type=str,
        default=DEFAULT_LABEL_SCHEME,
        help=f"Label scheme (default: {DEFAULT_LABEL_SCHEME})"
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=DEFAULT_SAMPLES["train"],
        help=f"Number of training samples (default: {DEFAULT_SAMPLES['train']})"
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=DEFAULT_SAMPLES["val"],
        help=f"Number of validation samples (default: {DEFAULT_SAMPLES['val']})"
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=DEFAULT_SAMPLES["test"],
        help=f"Number of test samples (default: {DEFAULT_SAMPLES['test']})"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N samples (default: 100)"
    )
    parser.add_argument(
        "--api-delay",
        type=float,
        default=0.1,
        help="Delay between API calls in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=CAPTURE24_COT_DATA_DIR,
        help=f"Output directory (default: {CAPTURE24_COT_DATA_DIR})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to generate (default: all)"
    )

    args = parser.parse_args()

    # Create config
    config = GenerationConfig(
        window_size_s=args.window_size_s,
        effective_hz=args.effective_hz,
        label_scheme=args.label_scheme,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        checkpoint_every=args.checkpoint_every,
        api_delay=args.api_delay,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print("=" * 60)
    print("Capture24 CoT Dataset Generator")
    print("=" * 60)
    print(f"Window size: {config.window_size_s}s @ {config.effective_hz}Hz")
    print(f"Label scheme: {config.label_scheme}")
    print(f"Output directory: {config.output_dir}")
    print(f"Samples: train={config.train_samples}, val={config.val_samples}, test={config.test_samples}")
    print()

    # Create generator
    generator = Capture24CoTGenerator(config)

    # Generate
    if args.split == "all":
        all_stats = generator.generate_all()
    else:
        samples_map = {
            "train": config.train_samples,
            "val": config.val_samples,
            "test": config.test_samples,
        }
        _, stats = generator.generate_split(args.split, samples_map[args.split])
        all_stats = {args.split: stats}

    # Print summary
    print("\n" + "=" * 60)
    print("Generation Summary")
    print("=" * 60)
    for split, stats in all_stats.items():
        print(f"\n{split}:")
        print(f"  Total processed: {stats.total_processed}")
        print(f"  Successful: {stats.successful}")
        print(f"  Failed: {stats.failed}")
        print(f"  API errors: {stats.api_errors}")
        print(f"  Validation failed: {stats.validation_failed}")
        print(f"  Class distribution: {stats.class_counts}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
