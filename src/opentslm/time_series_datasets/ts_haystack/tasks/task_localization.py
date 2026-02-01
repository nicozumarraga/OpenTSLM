# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Localization Task Generator for TS-Haystack benchmark.

Task 2: "When did the {activity} bout occur?"

This task requires the model to identify the temporal location of a
specific activity within the time series window.
"""

import numpy as np

from opentslm.time_series_datasets.ts_haystack.core import (
    DifficultyConfig,
    GeneratedSample,
)
from opentslm.time_series_datasets.ts_haystack.tasks.base_task import BaseTaskGenerator


class LocalizationTaskGenerator(BaseTaskGenerator):
    """
    Task 2: Localization - "When did the {activity} bout occur?"

    Algorithm:
    1. Sample background window (exclude target activity if possible)
    2. Sample needle from activity NOT in background
    3. Sample insertion position based on difficulty mode
    4. Apply style transfer to match target context
    5. Insert needle with boundary blending
    6. Generate Q/A with timestamp answer

    The answer is the time range (start, end) of the inserted needle.

    Difficulty Knobs:
    - context_length_samples: Longer windows are harder to scan
    - needle_position: "beginning", "middle", "end", "random"
    - needle_length_ratio_range: Shorter needles (smaller ratio) are harder to localize
    - background_purity: "mixed" backgrounds add confusion

    Answer Type: timestamp (time range)
    """

    @property
    def task_name(self) -> str:
        return "localization"

    @property
    def answer_type(self) -> str:
        return "timestamp"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """
        Generate a single localization task sample.

        Args:
            difficulty: Difficulty configuration
            rng: Random number generator (from SeedManager)

        Returns:
            GeneratedSample with localization question and timestamp answer
        """
        context_length = difficulty.context_length_samples

        # Step 1: Determine target activity first (for background exclusion)
        all_activities = list(self.needle_sampler.get_available_activities())

        if len(all_activities) < 2:
            return self._create_invalid_sample(
                "Need at least 2 activities for localization",
                difficulty,
            )

        # Pre-select candidate target activity to exclude from background
        # This ensures the needle activity won't naturally appear in background
        candidate_target = rng.choice(all_activities)

        # Step 2: Sample background excluding target activity
        background = self.background_sampler.sample_background(
            context_length_samples=context_length,
            purity=difficulty.background_purity,
            excluded_activities={candidate_target},
            rng=rng,
        )

        if background is None:
            # Retry with any background
            print(f"Warning: No background found without {candidate_target}, sampling a random background")
            background = self.background_sampler.sample_background(
                context_length_samples=context_length,
                purity=difficulty.background_purity,
                rng=rng,
            )

            if background is None:
                return self._create_invalid_sample(
                    "Failed to sample background",
                    difficulty,
                )

        # Step 3: Sample needle from activity NOT in background
        # No PID exclusion needed - we're selecting a different activity,
        # so even if from same participant, the data won't overlap
        min_duration_ms, max_duration_ms = difficulty.get_needle_length_range_ms(
            self.source_hz
        )
        needle = self.needle_sampler.sample_needle_for_context(
            context_activities=background.activities_present,
            min_duration_ms=min_duration_ms,
            use_transition_probs=difficulty.task_specific.get(
                "use_transition_probs", False
            ),
            rng=rng,
        )

        if needle is None:
            return self._create_invalid_sample(
                "Failed to sample needle for context",
                difficulty,
            )

        target_activity = needle.activity

        # Step 4: Determine needle length (cap by actual needle duration)
        max_duration_ms = min(max_duration_ms, needle.duration_ms)

        if max_duration_ms < min_duration_ms:
            return self._create_invalid_sample(
                f"Needle too short: {needle.duration_ms}ms < {min_duration_ms}ms",
                difficulty,
            )

        target_duration_ms = int(rng.integers(min_duration_ms, max_duration_ms + 1))
        target_samples = int(target_duration_ms * self.source_hz / 1000)
        target_samples = min(target_samples, needle.n_samples)

        # Trim needle to target length
        trimmed_needle = self._trim_needle(needle, target_samples)

        # Step 5: Sample insertion position
        margin = difficulty.task_specific.get("margin_samples", 100)
        position = self._sample_position(
            context_length=context_length,
            needle_length=trimmed_needle.n_samples,
            position_mode=difficulty.needle_position,
            rng=rng,
            margin_samples=margin,
        )

        if position is None:
            return self._create_invalid_sample(
                "Failed to find valid insertion position",
                difficulty,
            )

        # Step 6: Insert needle with style transfer
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

        # Step 7: Generate Q/A using template bank
        question, answer = self.template_bank.sample(
            task="localization",
            rng=rng,
            activity=target_activity,
            start=inserted_needle.timestamp_start,
            end=inserted_needle.timestamp_end,
        )

        # Build difficulty config with task-specific info
        full_difficulty_config = {
            **difficulty.to_dict(),
            "target_activity": target_activity,
            "needle_position_samples": position,
            "needle_position_frac": position / context_length,
            "needle_duration_samples": trimmed_needle.n_samples,
            "needle_duration_ms": trimmed_needle.duration_ms,
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
            needles=[inserted_needle],
            difficulty_config=full_difficulty_config,
            is_valid=True,
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate localization task dataset for TS-Haystack"
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

    print("Creating LocalizationTaskGenerator...")
    generator = LocalizationTaskGenerator.create_with_artifacts(seed=args.seed)

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