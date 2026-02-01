# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Existence Task Generator for TS-Haystack benchmark.

Task 1: "Is there {activity} in this recording?"

This is the simplest task - a binary classification asking whether
a specific activity is present in the time series window.
"""

from typing import Optional

import numpy as np

from opentslm.time_series_datasets.ts_haystack.core import (
    DifficultyConfig,
    GeneratedSample,
)
from opentslm.time_series_datasets.ts_haystack.tasks.base_task import BaseTaskGenerator


class ExistenceTaskGenerator(BaseTaskGenerator):
    """
    Task 1: Existence - "Is there {activity} in this recording?"

    Algorithm:
    1. Sample background window
    2. Flip coin: positive (50%) or negative (50%)
    3. Positive case:
       a. With 50% probability: Ask about activity already in background
       b. With 50% probability: Insert needle of new activity and ask about it
    4. Negative case:
       - Ask about activity NOT present in background (no needle inserted)
    5. Generate Q/A pair using template bank

    Difficulty Knobs:
    - context_length_samples: Longer windows are harder to scan
    - background_purity: "mixed" backgrounds have more activity variety
    - needle_length_ratio_range: Shorter needles (smaller ratio) are harder to detect
    - needle_position: Position affects difficulty (edges vs middle)

    Answer Type: boolean (Yes/No)
    """

    @property
    def task_name(self) -> str:
        return "existence"

    @property
    def answer_type(self) -> str:
        return "boolean"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """
        Generate a single existence task sample.

        Args:
            difficulty: Difficulty configuration
            rng: Random number generator (from SeedManager)

        Returns:
            GeneratedSample with existence question and boolean answer
        """
        context_length = difficulty.context_length_samples

        # Step 1: Sample background window
        background = self.background_sampler.sample_background(
            context_length_samples=context_length,
            purity=difficulty.background_purity,
            rng=rng,
        )

        if background is None:
            return self._create_invalid_sample(
                "Failed to sample background", difficulty
            )

        # Validate annotation coverage
        is_valid, reason = self._validate_background_coverage(background, difficulty)
        if not is_valid:
            return self._create_invalid_sample(reason, difficulty)

        # Initialize output signal with background
        final_x = background.x.copy()
        final_y = background.y.copy()
        final_z = background.z.copy()

        # Step 2: Decide positive or negative sample (50/50 balance)
        is_positive = rng.random() < 0.5

        # Track needle insertion
        inserted_needle = None
        target_activity: Optional[str] = None

        if is_positive:
            # Positive case: activity IS present
            # Sub-decision: use existing activity or insert needle
            insert_new_needle = rng.random() < 0.5

            if insert_new_needle:
                # Insert a needle of an activity NOT in background
                # No PID exclusion needed - we're selecting a different activity,
                # so even if from same participant, the data won't overlap
                min_duration_ms, max_duration_ms = difficulty.get_needle_length_range_ms(
                    self.source_hz
                )
                needle = self.needle_sampler.sample_needle_for_context(
                    context_activities=background.activities_present,
                    min_duration_ms=min_duration_ms,
                    rng=rng,
                )

                if needle is not None:
                    target_activity = needle.activity

                    # Determine needle length (sample within range, capped by actual needle)
                    max_duration_ms = min(max_duration_ms, needle.duration_ms)
                    target_duration_ms = int(rng.integers(
                        min_duration_ms,
                        max_duration_ms + 1,
                    ))
                    target_samples = int(target_duration_ms * self.source_hz / 1000)
                    target_samples = min(target_samples, needle.n_samples)

                    # Trim needle to target length
                    trimmed_needle = self._trim_needle(needle, target_samples)

                    # Sample insertion position
                    position = self._sample_position(
                        context_length=context_length,
                        needle_length=trimmed_needle.n_samples,
                        position_mode=difficulty.needle_position,
                        rng=rng,
                        margin_samples=difficulty.task_specific.get("margin_samples", 100),
                    )

                    if position is not None:
                        # Insert needle with style transfer
                        final_x, final_y, final_z = self._insert_needle(
                            background=background,
                            needle=trimmed_needle,
                            position=position,
                        )

                        # Create needle metadata
                        inserted_needle = self._create_inserted_needle(
                            needle=trimmed_needle,
                            position=position,
                            context_length=context_length,
                            background=background,
                        )
                    else:
                        # Position sampling failed, fall back to existing activity
                        target_activity = None

            # Fallback or intentional: use activity already in background
            if target_activity is None:
                if background.activities_present:
                    target_activity = rng.choice(list(background.activities_present))
                else:
                    return self._create_invalid_sample(
                        "No activities in background for positive case",
                        difficulty,
                    )

        else:
            # Negative case: activity is NOT present
            all_activities = set(self.needle_sampler.get_available_activities())
            absent_activities = all_activities - background.activities_present

            if not absent_activities:
                return self._create_invalid_sample(
                    "No absent activities for negative case",
                    difficulty,
                )

            target_activity = rng.choice(list(absent_activities))

        # Step 5: Generate Q/A using template bank
        question, answer = self.template_bank.sample(
            task="existence",
            rng=rng,
            activity=target_activity,
            exists=is_positive,
        )

        # Build difficulty config with task-specific info
        full_difficulty_config = {
            **difficulty.to_dict(),
            "target_activity": target_activity,
            "is_positive": is_positive,
            "needle_inserted": inserted_needle is not None,
            "background_activities": list(background.activities_present),
        }

        return GeneratedSample(
            x=final_x,
            y=final_y,
            z=final_z,
            task_type=self.task_name,
            context_length_samples=context_length,
            background_pid=background.pid,
            recording_time_range=background.recording_time_context,
            question=question,
            answer=answer,
            answer_type=self.answer_type,
            needles=[inserted_needle] if inserted_needle else [],
            difficulty_config=full_difficulty_config,
            is_valid=True,
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Generate existence task dataset for TS-Haystack"
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[1000, 10000],
        help="Context lengths in samples",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        nargs=3,
        default=[1000, 100, 100],
        help="Samples for train, val, test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs",
    )
    parser.add_argument(
        "--needle-ratio-min",
        type=float,
        default=0.02,
        help="Minimum needle length as fraction of context (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--needle-ratio-max",
        type=float,
        default=0.10,
        help="Maximum needle length as fraction of context (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--needle-position",
        type=str,
        choices=["random", "beginning", "middle", "end"],
        default="random",
        help="Needle position mode",
    )
    parser.add_argument(
        "--background-purity",
        type=str,
        choices=["pure", "mixed"],
        default="pure",
        help="Background purity mode",
    )

    args = parser.parse_args()

    print("Creating ExistenceTaskGenerator...")
    generator = ExistenceTaskGenerator.create_with_artifacts(seed=args.seed)

    for context_length in args.context_lengths:
        print(f"\nGenerating for context_length={context_length}...")

        difficulty = DifficultyConfig(
            context_length_samples=context_length,
            needle_position=args.needle_position,
            needle_length_ratio_range=(args.needle_ratio_min, args.needle_ratio_max),
            background_purity=args.background_purity,
        )

        for split, n_samples in zip(
            ["train", "val", "test"], args.samples_per_split
        ):
            print(f"  Generating {n_samples} {split} samples...")
            samples = generator.generate_dataset(
                n_samples=n_samples,
                difficulty=difficulty,
                split=split,
                n_jobs=args.n_jobs,
            )

            output_path = generator.save_dataset(
                samples=samples,
                split=split,
                context_length=context_length,
            )
            print(f"  Saved to: {output_path}")

    print("\nDone!")
