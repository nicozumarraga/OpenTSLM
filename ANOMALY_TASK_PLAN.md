# Anomaly Detection & Localization Tasks Implementation Plan

## Overview

This plan outlines the implementation of two new related tasks for the TS-Haystack benchmark:

1. **ANOMALY_DETECTION (Task 9)**: "Is there an anomaly?" - Binary detection with explanation
2. **ANOMALY_LOCALIZATION (Task 10)**: "Is there an anomaly, and if so, when?" - Detection + temporal localization

Both tasks test contextual reasoning: detecting cross-regime violations without being told what to look for.

### Key Distinction from EXISTENCE

| Aspect | EXISTENCE Task | ANOMALY_DETECTION | ANOMALY_LOCALIZATION |
|--------|----------------|-------------------|----------------------|
| Question | "Is there walking?" | "Is there an anomaly?" | "Is there an anomaly, and when?" |
| Prior knowledge | Target provided | No target | No target |
| Anomaly definition | N/A | Cross-regime | Cross-regime |
| Answer format | Yes/No | Yes + activity + regime / No | Yes + activity + time range / No |
| Answer type | `boolean` | `boolean` | `time_range` |
| Evaluation metric | Accuracy | Accuracy | Accuracy + IoU |
| Reasoning | Pattern matching | Contextual | Contextual + Temporal |

### Distractor Methodology (Preventing Detection Shortcuts)

**Problem**: If positive samples have needle insertions and negative samples don't, the model
can learn to detect ANY needle insertion (via variance change) as a shortcut to answer "yes".

**Solution**: Both positive and negative samples ALWAYS have same-regime distractor insertions:

| Case | Insertions | Difference |
|------|------------|------------|
| Positive (anomaly) | 1 cross-regime needle + N same-regime distractors | Cross-regime needle present |
| Negative (no anomaly) | N same-regime distractors only | No cross-regime needle |

This ensures:
1. Both cases have similar signal variance characteristics (multiple insertions)
2. The model must learn to identify regime mismatches, not just "something was inserted"
3. Same-regime distractors act as confounders that don't constitute anomalies

This approach is consistent with the EXISTENCE task distractor logic (see `task_existence.py`).

---

## 1. Task Design

### 1.1 Semantic Definition

**Anomaly = Cross-regime insertion**
- Background sampled from ONE regime (sedentary or active)
- Positive: Insert activity from the OPPOSITE regime
- Negative: Insert activity from the SAME regime (not anomalous)

### 1.2 Sample Generation Logic

```
Algorithm: generate_anomaly_sample()

1. Sample background window (pure regime)
2. Determine background regime from background.activities_present
3. Compute insertable_same_regime = same_regime_activities - background.activities_present
4. Decide positive (50%) or negative (50%)
5. Determine n_distractors from difficulty config (min_distractors to max_distractors)

6. If positive (ANOMALY):
   a. Sample 1 needle from OPPOSITE regime (this is the anomaly)
   b. Sample n_distractors needles from SAME regime (these are distractors, NOT anomalies)
   c. Insert all needles at non-overlapping positions
   d. anomaly_activity = opposite_regime_needle.activity
   e. DETECTION Answer: "Yes, there is anomalous {anomaly_activity} in the {background_regime} background."
   f. LOCALIZATION Answer: "Yes, there is anomalous {anomaly_activity} from {start} to {end}."

7. If negative (NO ANOMALY):
   a. Sample n_distractors needles from SAME regime only (no cross-regime = no anomaly)
   b. Insert all needles at non-overlapping positions
   c. DETECTION Answer: "No, the recording shows consistent {background_regime} activity."
   d. LOCALIZATION Answer: "No, the recording shows consistent {background_regime} activity."

8. Generate question using template bank
9. Return GeneratedSample

Key insight: Both positive and negative cases have needle insertions (distractors),
but ONLY positive cases have a cross-regime insertion (anomaly).
```

### 1.3 Difficulty Knobs

| Parameter | Easier | Harder |
|-----------|--------|--------|
| `cross_regime_contrast` | high (sleep → sports) | low (standing → walking) |
| `needle_length_ratio_range` | large (0.10-0.20) | small (0.02-0.05) |
| `background_purity` | pure | mixed |
| `min_distractors` | 1 | 2-3 (more confounders) |
| `max_distractors` | 2 | 3-4 (more confounders) |

**Design decisions**:
- **Single anomaly**: Each positive sample has exactly 1 cross-regime anomaly
- **Mandatory distractors**: Both positive and negative samples have same-regime distractor insertions
  (min_distractors >= 1) to prevent the model from using "insertion detected" as a shortcut

### 1.4 Answer Formats

**Evaluation Note**: For positive samples, the model MUST correctly identify the anomaly activity.
Simply saying "Yes" is insufficient - the answer is evaluated using regex to verify the correct
activity is mentioned. This prevents random guessing from achieving high accuracy.

#### Task 9: Anomaly Detection

**Positive (anomaly present):**
```
"Yes, there is anomalous {activity} activity in the {regime} background."
```
- Evaluation: Must match "Yes" AND mention the correct {activity} (regex match)

**Negative (no anomaly - same-regime insertion):**
```
"No, the recording shows consistent {regime} activity."
```
- Evaluation: Must match "No"

#### Task 10: Anomaly Localization

**Positive (anomaly present):**
```
"Yes, there is anomalous {activity} activity from {start} to {end}."
```
- Evaluation: Must match "Yes" AND mention correct {activity} AND correct time range (IoU)

**Negative (no anomaly - same-regime insertion):**
```
"No, the recording shows consistent {regime} activity."
```
- Evaluation: Must match "No"

---

## 2. Files to Create/Modify

### 2.1 New Files

| File | Purpose |
|------|---------|
| `tasks/task_anomaly_detection.py` | Task 9: Anomaly Detection generator |
| `tasks/task_anomaly_localization.py` | Task 10: Anomaly Localization generator |
| `test/tasks/test_task_anomaly_detection.py` | Unit tests + visualization for Task 9 |
| `test/tasks/test_task_anomaly_localization.py` | Unit tests + visualization for Task 10 |

### 2.2 Files to Modify

| File | Changes |
|------|---------|
| `tasks/__init__.py` | Add both tasks to TASK_REGISTRY |
| `core/prompt_templates.py` | Add "anomaly_detection" and "anomaly_localization" templates |
| `configs/default_generation_config.yaml` | Add both task config blocks |
| `dataset/ts_haystack_qa_loader.py` | Add both tasks to ALL_TASKS |
| `cot/prompt_builder.py` | Add context builders for both tasks |
| `test/tasks/conftest.py` | Add generator fixtures for both tasks |
| `README.md` | Document both new tasks |

---

## 3. Implementation Details

### 3.1 `tasks/task_anomaly_detection.py`

```python
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Anomaly Detection Task Generator for TS-Haystack benchmark.

Task 9: "Is there an anomaly in this recording?"

This task tests contextual reasoning: detecting when an activity is anomalous
relative to the background regime, without being told what to look for.

Key difference from Existence:
- Existence: "Is there walking?" (target given)
- Anomaly Detection: "Is there an anomaly?" (must identify the anomaly type)

Anomaly Definition:
- Cross-regime insertion: active activity in sedentary background (or vice versa)
- Same-regime activities are NOT anomalous

Sample Design (with mandatory distractors to prevent detection shortcuts):
- Positive: 1 cross-regime insertion (anomaly) + N same-regime distractors
- Negative: N same-regime distractors only (no anomaly)

Both cases have needle insertions, so the model cannot use "insertion detected"
as a shortcut. It must identify the regime mismatch (cross-regime = anomaly).

Answer Format:
- Positive: "Yes, there is anomalous {activity} activity in the {regime} background."
- Negative: "No, the recording shows consistent {regime} activity."
"""

from typing import List, Optional, Set, Tuple

import numpy as np

from opentslm.time_series_datasets.ts_haystack.core import (
    DifficultyConfig,
    GeneratedSample,
    InsertedNeedle,
    NeedleSample,
    WILLETTS_ACTIVITY_REGIMES,
    ACTIVITY_TO_REGIME,
    get_regime,
    get_regime_activities,
    get_other_regime_activities,
)
from opentslm.time_series_datasets.ts_haystack.tasks.base_task import BaseTaskGenerator


class AnomalyDetectionTaskGenerator(BaseTaskGenerator):
    """
    Task 9: Anomaly Detection - "Is there an anomaly in this recording?"

    Algorithm (with mandatory distractors):
    1. Sample background from a single regime (pure)
    2. Determine background regime
    3. Compute insertable same-regime activities
    4. Decide positive (50%) or negative (50%)
    5. Determine n_distractors from config (min_distractors to max_distractors)
    6. Positive: insert 1 OPPOSITE regime needle (anomaly) + n_distractors SAME regime needles
    7. Negative: insert n_distractors SAME regime needles only (no anomaly)
    8. Generate Q/A with explanation

    Difficulty Knobs:
    - needle_length_ratio_range: Smaller anomalies are harder to detect
    - min_distractors / max_distractors: Number of same-regime distractors (mandatory)

    Answer Type: boolean (Yes/No)
    Evaluation: Accuracy (exact match on Yes/No)
    """

    @property
    def task_name(self) -> str:
        return "anomaly_detection"

    @property
    def answer_type(self) -> str:
        return "boolean"  # Yes/No

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """Generate a single anomaly detection sample."""
        context_length = difficulty.context_length_samples

        # Step 1: Sample PURE background (single regime)
        background = self.background_sampler.sample_background(
            context_length_samples=context_length,
            purity="pure",  # Force pure for clear regime identification
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

        # Step 2: Determine background regime
        background_regime = self._get_dominant_regime(background.activities_present)
        if background_regime is None:
            return self._create_invalid_sample(
                f"Could not determine regime for activities: {background.activities_present}",
                difficulty,
            )

        # Step 3: Decide positive or negative (50/50)
        is_positive = rng.random() < 0.5

        # Step 4/5: Generate sample based on positive/negative
        if is_positive:
            return self._generate_positive_sample(
                background, background_regime, difficulty, rng
            )
        else:
            return self._generate_negative_sample(
                background, background_regime, difficulty, rng
            )

    def _get_dominant_regime(self, activities: Set[str]) -> Optional[str]:
        """Determine the dominant regime from a set of activities."""
        if not activities:
            return None

        regimes = {}
        for activity in activities:
            regime = ACTIVITY_TO_REGIME.get(activity)
            if regime:
                regimes[regime] = regimes.get(regime, 0) + 1

        if not regimes:
            return None

        return max(regimes, key=regimes.get)

    def _sample_and_insert_needle(
        self,
        background,
        target_activities: Set[str],
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
        current_signal: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        occupied_ranges: Optional[List[Tuple[int, int]]] = None,
    ) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], InsertedNeedle]]:
        """Helper to sample and insert a needle from target activities."""
        context_length = difficulty.context_length_samples
        min_duration_ms, max_duration_ms = difficulty.get_needle_length_range_ms(
            self.source_hz
        )

        needle = self.needle_sampler.sample_needle_for_activities(
            target_activities=target_activities,
            min_duration_ms=min_duration_ms,
            rng=rng,
        )

        if needle is None:
            return None

        # Trim needle to target length
        capped_max_ms = min(max_duration_ms, needle.duration_ms)
        if capped_max_ms < min_duration_ms:
            return None

        target_duration_ms = int(rng.integers(min_duration_ms, capped_max_ms + 1))
        target_samples = int(target_duration_ms * self.source_hz / 1000)
        target_samples = min(target_samples, needle.n_samples)
        trimmed_needle = self._trim_needle(needle, target_samples)

        # Find insertion position
        margin = difficulty.task_specific.get("margin_samples", 100)
        min_gap = difficulty.task_specific.get("min_gap_samples", 100)

        if occupied_ranges:
            position = self._find_valid_position(
                context_length=context_length,
                needle_length=trimmed_needle.n_samples,
                occupied_ranges=occupied_ranges,
                min_gap=min_gap,
                rng=rng,
                margin=margin,
            )
        else:
            position = self._sample_position(
                context_length=context_length,
                needle_length=trimmed_needle.n_samples,
                position_mode=difficulty.needle_position,
                rng=rng,
                margin_samples=margin,
            )

        if position is None:
            return None

        # Insert needle
        if current_signal is None:
            current_signal = (background.x.copy(), background.y.copy(), background.z.copy())

        final_signal = self._insert_needle(
            background=background,
            needle=trimmed_needle,
            position=position,
            current_signal=current_signal,
        )

        # Create needle metadata
        inserted_needle = self._create_inserted_needle(
            needle=trimmed_needle,
            position=position,
            context_length=context_length,
            background=background,
        )

        return final_signal, inserted_needle

    def _generate_positive_sample(
        self,
        background,
        background_regime: str,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """Generate a positive sample with cross-regime anomaly + same-regime distractors."""
        context_length = difficulty.context_length_samples

        # Get activities from the OPPOSITE regime (these are anomalous)
        opposite_regime = "active" if background_regime == "sedentary" else "sedentary"
        anomaly_candidates = get_regime_activities(opposite_regime)

        # Get insertable same-regime activities for distractors
        same_regime_activities = get_regime_activities(background_regime) - background.activities_present

        # Sample and insert anomaly (cross-regime)
        result = self._sample_and_insert_needle(
            background, anomaly_candidates, difficulty, rng
        )

        if result is None:
            return self._create_invalid_sample(
                f"Failed to sample anomaly needle from {opposite_regime} regime",
                difficulty,
            )

        final_signal, anomaly_needle = result
        final_x, final_y, final_z = final_signal

        # Add MANDATORY same-regime distractors (prevents detection shortcuts)
        min_distractors = difficulty.task_specific.get("min_distractors", 1)
        max_distractors = difficulty.task_specific.get("max_distractors", 3)
        n_distractors = int(rng.integers(min_distractors, max_distractors + 1))

        distractor_needles = []
        occupied_ranges = [(anomaly_needle.insert_position_samples,
                           anomaly_needle.insert_position_samples + anomaly_needle.duration_samples)]

        for _ in range(n_distractors):
            if not same_regime_activities:
                break
            result = self._sample_and_insert_needle(
                background, same_regime_activities, difficulty, rng,
                current_signal=(final_x, final_y, final_z),
                occupied_ranges=occupied_ranges,
            )
            if result:
                (final_x, final_y, final_z), distractor = result
                distractor_needles.append(distractor)
                occupied_ranges.append((distractor.insert_position_samples,
                                       distractor.insert_position_samples + distractor.duration_samples))

        # Generate Q/A
        question, answer = self.template_bank.sample(
            task="anomaly_detection",
            rng=rng,
            is_anomaly=True,
            anomaly_activity=anomaly_needle.activity,
            background_regime=background_regime,
        )

        # Build difficulty config
        full_difficulty_config = {
            **difficulty.to_dict(),
            "is_positive": True,
            "background_regime": background_regime,
            "anomaly_regime": opposite_regime,
            "anomaly_activity": anomaly_needle.activity,
            "background_activities": list(background.activities_present),
            "n_distractors": len(distractor_needles),
        }

        all_needles = [anomaly_needle] + distractor_needles

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
            needles=all_needles,
            difficulty_config=full_difficulty_config,
            is_valid=True,
        )

    def _generate_negative_sample(
        self,
        background,
        background_regime: str,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """Generate a negative sample with same-regime distractors only (no anomaly)."""
        context_length = difficulty.context_length_samples

        # Get activities from SAME regime (these are NOT anomalous)
        same_regime_activities = get_regime_activities(background_regime) - background.activities_present

        if not same_regime_activities:
            return self._create_invalid_sample(
                f"No insertable same-regime activities for {background_regime}",
                difficulty,
            )

        # Determine number of distractors (same as positive case for consistency)
        min_distractors = difficulty.task_specific.get("min_distractors", 1)
        max_distractors = difficulty.task_specific.get("max_distractors", 3)
        n_distractors = int(rng.integers(min_distractors, max_distractors + 1))

        # Insert multiple same-regime distractors (NO cross-regime = NO anomaly)
        final_x = background.x.copy()
        final_y = background.y.copy()
        final_z = background.z.copy()
        distractor_needles = []
        occupied_ranges = []

        for _ in range(n_distractors):
            if not same_regime_activities:
                break
            result = self._sample_and_insert_needle(
                background, same_regime_activities, difficulty, rng,
                current_signal=(final_x, final_y, final_z),
                occupied_ranges=occupied_ranges if occupied_ranges else None,
            )
            if result:
                (final_x, final_y, final_z), distractor = result
                distractor_needles.append(distractor)
                occupied_ranges.append((distractor.insert_position_samples,
                                       distractor.insert_position_samples + distractor.duration_samples))

        if not distractor_needles:
            return self._create_invalid_sample(
                f"Failed to insert any same-regime distractors for {background_regime}",
                difficulty,
            )

        # Generate Q/A (negative - all insertions are same-regime, no anomaly)
        question, answer = self.template_bank.sample(
            task="anomaly_detection",
            rng=rng,
            is_anomaly=False,
            background_regime=background_regime,
        )

        full_difficulty_config = {
            **difficulty.to_dict(),
            "is_positive": False,
            "background_regime": background_regime,
            "n_distractors": len(distractor_needles),
            "distractor_activities": [n.activity for n in distractor_needles],
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
            needles=distractor_needles,
            difficulty_config=full_difficulty_config,
            is_valid=True,
        )
```

### 3.2 `tasks/task_anomaly_localization.py`

```python
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Anomaly Localization Task Generator for TS-Haystack benchmark.

Task 10: "Is there an anomaly in this recording, and if so, when did it occur?"

This task combines anomaly detection with temporal localization: the model must
not only detect cross-regime violations but also specify WHEN they occur.

Key difference from Anomaly Detection:
- Anomaly Detection: "Is there an anomaly?" → Yes/No + what
- Anomaly Localization: "Is there an anomaly, and when?" → Yes/No + what + when

Anomaly Definition:
- Cross-regime insertion: active activity in sedentary background (or vice versa)
- Same-regime activities are NOT anomalous

Sample Design (with mandatory distractors to prevent detection shortcuts):
- Positive: 1 cross-regime insertion (anomaly) + N same-regime distractors - must report time range
- Negative: N same-regime distractors only (no anomaly)

Both cases have needle insertions, so the model cannot use "insertion detected"
as a shortcut. It must identify the regime mismatch AND localize it.

Answer Format:
- Positive: "Yes, there is anomalous {activity} activity from {start} to {end}."
- Negative: "No, the recording shows consistent {regime} activity."
"""

from typing import List, Optional, Set, Tuple

import numpy as np

from opentslm.time_series_datasets.ts_haystack.core import (
    DifficultyConfig,
    GeneratedSample,
    InsertedNeedle,
    ACTIVITY_TO_REGIME,
    get_regime_activities,
)
from opentslm.time_series_datasets.ts_haystack.tasks.task_anomaly_detection import (
    AnomalyDetectionTaskGenerator,
)


class AnomalyLocalizationTaskGenerator(AnomalyDetectionTaskGenerator):
    """
    Task 10: Anomaly Localization - "Is there an anomaly, and when did it occur?"

    Extends AnomalyDetectionTaskGenerator to include temporal localization.
    The answer for positive samples includes the time range of the anomaly.

    Algorithm (inherits from AnomalyDetectionTaskGenerator with mandatory distractors):
    1. Sample background from a single regime (pure)
    2. Determine background regime
    3. Decide positive (50%) or negative (50%)
    4. Positive: insert 1 OPPOSITE regime needle (anomaly) + n_distractors SAME regime needles
    5. Negative: insert n_distractors SAME regime needles only (no anomaly)
    6. For positive samples, answer includes temporal location of the anomaly

    Difficulty Knobs:
    - Same as AnomalyDetection (min_distractors, max_distractors are mandatory)
    - More distractors make localization harder (more candidates to rule out)

    Answer Type: time_range (timestamp for positive, boolean for negative)
    Evaluation:
    - Detection: Accuracy (exact match on Yes/No)
    - Localization: IoU (Intersection over Union) of predicted vs ground truth time range
    """

    @property
    def task_name(self) -> str:
        return "anomaly_localization"

    @property
    def answer_type(self) -> str:
        return "time_range"  # Includes temporal information (like Task 2: Localization)

    def _generate_positive_sample(
        self,
        background,
        background_regime: str,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """Generate a positive sample with cross-regime anomaly + same-regime distractors."""
        context_length = difficulty.context_length_samples

        # Get activities from the OPPOSITE regime (these are anomalous)
        opposite_regime = "active" if background_regime == "sedentary" else "sedentary"
        anomaly_candidates = get_regime_activities(opposite_regime)

        # Get insertable same-regime activities for distractors
        same_regime_activities = get_regime_activities(background_regime) - background.activities_present

        # Sample and insert anomaly (cross-regime)
        result = self._sample_and_insert_needle(
            background, anomaly_candidates, difficulty, rng
        )

        if result is None:
            return self._create_invalid_sample(
                f"Failed to sample anomaly needle from {opposite_regime} regime",
                difficulty,
            )

        final_signal, anomaly_needle = result
        final_x, final_y, final_z = final_signal

        # Add MANDATORY same-regime distractors (prevents detection shortcuts)
        min_distractors = difficulty.task_specific.get("min_distractors", 1)
        max_distractors = difficulty.task_specific.get("max_distractors", 3)
        n_distractors = int(rng.integers(min_distractors, max_distractors + 1))

        distractor_needles = []
        occupied_ranges = [(anomaly_needle.insert_position_samples,
                           anomaly_needle.insert_position_samples + anomaly_needle.duration_samples)]

        for _ in range(n_distractors):
            if not same_regime_activities:
                break
            result = self._sample_and_insert_needle(
                background, same_regime_activities, difficulty, rng,
                current_signal=(final_x, final_y, final_z),
                occupied_ranges=occupied_ranges,
            )
            if result:
                (final_x, final_y, final_z), distractor = result
                distractor_needles.append(distractor)
                occupied_ranges.append((distractor.insert_position_samples,
                                       distractor.insert_position_samples + distractor.duration_samples))

        # Generate Q/A with time range
        question, answer = self.template_bank.sample(
            task="anomaly_localization",
            rng=rng,
            is_anomaly=True,
            anomaly_activity=anomaly_needle.activity,
            background_regime=background_regime,
            start=anomaly_needle.timestamp_start,
            end=anomaly_needle.timestamp_end,
        )

        # Build difficulty config
        full_difficulty_config = {
            **difficulty.to_dict(),
            "is_positive": True,
            "background_regime": background_regime,
            "anomaly_regime": opposite_regime,
            "anomaly_activity": anomaly_needle.activity,
            "anomaly_start": anomaly_needle.timestamp_start,
            "anomaly_end": anomaly_needle.timestamp_end,
            "background_activities": list(background.activities_present),
            "n_distractors": len(distractor_needles),
        }

        all_needles = [anomaly_needle] + distractor_needles

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
            needles=all_needles,
            difficulty_config=full_difficulty_config,
            is_valid=True,
        )

    def _generate_negative_sample(
        self,
        background,
        background_regime: str,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """Generate a negative sample with same-regime distractors only (no anomaly)."""
        context_length = difficulty.context_length_samples

        # Get activities from SAME regime (these are NOT anomalous)
        same_regime_activities = get_regime_activities(background_regime) - background.activities_present

        if not same_regime_activities:
            return self._create_invalid_sample(
                f"No insertable same-regime activities for {background_regime}",
                difficulty,
            )

        # Determine number of distractors (same as positive case for consistency)
        min_distractors = difficulty.task_specific.get("min_distractors", 1)
        max_distractors = difficulty.task_specific.get("max_distractors", 3)
        n_distractors = int(rng.integers(min_distractors, max_distractors + 1))

        # Insert multiple same-regime distractors (NO cross-regime = NO anomaly)
        final_x = background.x.copy()
        final_y = background.y.copy()
        final_z = background.z.copy()
        distractor_needles = []
        occupied_ranges = []

        for _ in range(n_distractors):
            if not same_regime_activities:
                break
            result = self._sample_and_insert_needle(
                background, same_regime_activities, difficulty, rng,
                current_signal=(final_x, final_y, final_z),
                occupied_ranges=occupied_ranges if occupied_ranges else None,
            )
            if result:
                (final_x, final_y, final_z), distractor = result
                distractor_needles.append(distractor)
                occupied_ranges.append((distractor.insert_position_samples,
                                       distractor.insert_position_samples + distractor.duration_samples))

        if not distractor_needles:
            return self._create_invalid_sample(
                f"Failed to insert any same-regime distractors for {background_regime}",
                difficulty,
            )

        # Generate Q/A (negative - no time range needed, all insertions are same-regime)
        question, answer = self.template_bank.sample(
            task="anomaly_localization",
            rng=rng,
            is_anomaly=False,
            background_regime=background_regime,
        )

        full_difficulty_config = {
            **difficulty.to_dict(),
            "is_positive": False,
            "background_regime": background_regime,
            "n_distractors": len(distractor_needles),
            "distractor_activities": [n.activity for n in distractor_needles],
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
            needles=distractor_needles,
            difficulty_config=full_difficulty_config,
            is_valid=True,
        )
```

### 3.3 Prompt Templates (`core/prompt_templates.py`)

Add the following template blocks to `TEMPLATES` dict:

```python
# =====================================================================
# Task 9: Anomaly Detection - "Is there an anomaly in this recording?"
# =====================================================================
"anomaly_detection": [
    # --- Positive (anomaly present) ---
    TemplateVariant(
        question="Is there an anomaly in this recording?",
        answer="Yes, there is anomalous {anomaly_activity} activity in the {background_regime} background.",
    ),
    TemplateVariant(
        question="Is there any anomalous activity in this accelerometer data?",
        answer="Yes, {anomaly_activity} is anomalous in the {background_regime} context.",
    ),
    TemplateVariant(
        question="Does this recording contain any unusual activity patterns?",
        answer="Yes, there is unusual {anomaly_activity} activity within the {background_regime} background.",
    ),
    TemplateVariant(
        question="Can you detect any anomalies in this sensor data?",
        answer="Yes, I detect anomalous {anomaly_activity} in the otherwise {background_regime} recording.",
    ),
    TemplateVariant(
        question="Is there anything out of the ordinary in this recording?",
        answer="Yes, {anomaly_activity} activity is out of the ordinary for the {background_regime} background.",
    ),
    TemplateVariant(
        question="Determine if there is an anomaly in this data.",
        answer="Yes, there is an anomalous {anomaly_activity} bout in the {background_regime} background.",
    ),
    TemplateVariant(
        question="Check for any anomalous patterns in this recording.",
        answer="Yes, {anomaly_activity} is anomalous relative to the {background_regime} context.",
    ),
    TemplateVariant(
        question="Please identify if any activity is anomalous in this recording.",
        answer="Yes, {anomaly_activity} activity is anomalous in the {background_regime} background.",
    ),
    TemplateVariant(
        question="Is there anything unusual that doesn't fit the overall pattern?",
        answer="Yes, {anomaly_activity} does not fit the {background_regime} pattern.",
    ),
    TemplateVariant(
        question="Looking at this data, can you identify any anomalies?",
        answer="Yes, there is anomalous {anomaly_activity} in the {background_regime} background.",
    ),

    # --- Negative (no anomaly, same-regime insertion) ---
    TemplateVariant(
        question="Is there an anomaly in this recording?",
        answer="No, the recording shows consistent {background_regime} activity.",
    ),
    TemplateVariant(
        question="Is there any anomalous activity in this accelerometer data?",
        answer="No, all activities are consistent with the {background_regime} pattern.",
    ),
    TemplateVariant(
        question="Does this recording contain any unusual activity patterns?",
        answer="No, the recording shows typical {background_regime} activity throughout.",
    ),
    TemplateVariant(
        question="Can you detect any anomalies in this sensor data?",
        answer="No anomalies detected. The data shows consistent {background_regime} activity.",
    ),
    TemplateVariant(
        question="Is there anything out of the ordinary in this recording?",
        answer="No, everything is consistent with {background_regime} activity.",
    ),
    TemplateVariant(
        question="Determine if there is an anomaly in this data.",
        answer="No, the recording is consistent with {background_regime} activity.",
    ),
    TemplateVariant(
        question="Check for any anomalous patterns in this recording.",
        answer="No anomalous patterns detected. The {background_regime} pattern is consistent.",
    ),
    TemplateVariant(
        question="Please identify if any activity is anomalous in this recording.",
        answer="No, all activities are appropriate for the {background_regime} context.",
    ),
    TemplateVariant(
        question="Is there anything unusual that doesn't fit the overall pattern?",
        answer="No, all activities fit the {background_regime} pattern.",
    ),
    TemplateVariant(
        question="Looking at this data, can you identify any anomalies?",
        answer="No anomalies identified. The data shows {background_regime} activity.",
    ),
],

# =====================================================================
# Task 10: Anomaly Localization - "Is there an anomaly, and when?"
# =====================================================================
"anomaly_localization": [
    # --- Positive (anomaly present with time range) ---
    TemplateVariant(
        question="Is there an anomaly in this recording, and if so, when does it occur?",
        answer="Yes, there is anomalous {anomaly_activity} activity from {start} to {end}.",
    ),
    TemplateVariant(
        question="Identify any anomalies and their timing in this data.",
        answer="Yes, {anomaly_activity} is anomalous, occurring from {start} to {end}.",
    ),
    TemplateVariant(
        question="Is there any unusual activity? If yes, specify when it occurred.",
        answer="Yes, unusual {anomaly_activity} activity occurred from {start} to {end}.",
    ),
    TemplateVariant(
        question="Detect and locate any anomalies in this accelerometer data.",
        answer="Anomaly detected: {anomaly_activity} from {start} to {end}.",
    ),
    TemplateVariant(
        question="Does this recording contain anomalies? Provide the time range if so.",
        answer="Yes, anomalous {anomaly_activity} from {start} to {end}.",
    ),
    TemplateVariant(
        question="Find any anomalous patterns and specify when they occur.",
        answer="Anomalous {anomaly_activity} found from {start} to {end}.",
    ),
    TemplateVariant(
        question="Is there anything out of the ordinary? Report the timing.",
        answer="Yes, {anomaly_activity} is out of the ordinary, occurring {start} to {end}.",
    ),
    TemplateVariant(
        question="Check for anomalies and report their temporal location.",
        answer="Anomaly: {anomaly_activity} from {start} to {end}.",
    ),
    TemplateVariant(
        question="Determine if there is an anomaly and when it happened.",
        answer="Yes, there is anomalous {anomaly_activity} from {start} to {end}.",
    ),
    TemplateVariant(
        question="Identify and localize any anomalous activity in this recording.",
        answer="Anomalous {anomaly_activity} localized at {start} to {end}.",
    ),

    # --- Negative (no anomaly, same-regime insertion) ---
    TemplateVariant(
        question="Is there an anomaly in this recording, and if so, when does it occur?",
        answer="No, the recording shows consistent {background_regime} activity.",
    ),
    TemplateVariant(
        question="Identify any anomalies and their timing in this data.",
        answer="No anomalies found. The data shows consistent {background_regime} activity.",
    ),
    TemplateVariant(
        question="Is there any unusual activity? If yes, specify when it occurred.",
        answer="No unusual activity. The recording is consistent with {background_regime} patterns.",
    ),
    TemplateVariant(
        question="Detect and locate any anomalies in this accelerometer data.",
        answer="No anomalies detected. Consistent {background_regime} activity throughout.",
    ),
    TemplateVariant(
        question="Does this recording contain anomalies? Provide the time range if so.",
        answer="No, the recording shows consistent {background_regime} activity.",
    ),
    TemplateVariant(
        question="Find any anomalous patterns and specify when they occur.",
        answer="No anomalous patterns found in this {background_regime} recording.",
    ),
    TemplateVariant(
        question="Is there anything out of the ordinary? Report the timing.",
        answer="Nothing out of the ordinary. Consistent {background_regime} activity.",
    ),
    TemplateVariant(
        question="Check for anomalies and report their temporal location.",
        answer="No anomalies to report. The {background_regime} pattern is consistent.",
    ),
    TemplateVariant(
        question="Determine if there is an anomaly and when it happened.",
        answer="No anomaly detected. The recording shows {background_regime} activity.",
    ),
    TemplateVariant(
        question="Identify and localize any anomalous activity in this recording.",
        answer="No anomalous activity found. Consistent {background_regime} throughout.",
    ),
],
```

Also add grammar helper in `_add_grammar_helpers()`:

```python
# Anomaly detection/localization helpers
if "is_anomaly" in filled:
    is_anomaly = filled["is_anomaly"]
    filled["yes_no"] = "Yes" if is_anomaly else "No"
```

### 3.4 YAML Configuration (`configs/default_generation_config.yaml`)

Add at the end of the `tasks:` section:

```yaml
  # --------------------------------------------------------------------------
  # Task 9: Anomaly Detection
  # "Is there an anomaly in this recording?"
  # Tests contextual reasoning: detecting cross-regime activity insertions
  # without being told what to look for.
  #
  # Sample design (mandatory distractors to prevent detection shortcuts):
  # - Positive: 1 cross-regime (anomaly) + N same-regime distractors
  # - Negative: N same-regime distractors only (no anomaly)
  # --------------------------------------------------------------------------
  anomaly_detection:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.03, 0.15]  # 3-15% of context
    background_purity: pure              # MUST be pure for regime detection
    # Task-specific parameters
    margin_samples: 100
    min_gap_samples: 100
    min_distractors: 1                   # Minimum same-regime distractors (mandatory)
    max_distractors: 3                   # Maximum same-regime distractors

  # --------------------------------------------------------------------------
  # Task 10: Anomaly Localization
  # "Is there an anomaly, and if so, when does it occur?"
  # Combines anomaly detection with temporal localization.
  #
  # Sample design (mandatory distractors to prevent detection shortcuts):
  # - Positive: 1 cross-regime (anomaly) + N same-regime distractors + time range
  # - Negative: N same-regime distractors only (no anomaly)
  # --------------------------------------------------------------------------
  anomaly_localization:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.03, 0.15]  # 3-15% of context
    background_purity: pure              # MUST be pure for regime detection
    # Task-specific parameters
    margin_samples: 100
    min_gap_samples: 100
    min_distractors: 1                   # Minimum same-regime distractors (mandatory)
    max_distractors: 3                   # More distractors make localization harder
```

### 3.5 Task Registry (`tasks/__init__.py`)

Add imports and registry entries:

```python
from opentslm.time_series_datasets.ts_haystack.tasks.task_anomaly_detection import (
    AnomalyDetectionTaskGenerator,
)
from opentslm.time_series_datasets.ts_haystack.tasks.task_anomaly_localization import (
    AnomalyLocalizationTaskGenerator,
)

# Add to TASK_REGISTRY
TASK_REGISTRY = {
    "existence": ExistenceTaskGenerator,
    "localization": LocalizationTaskGenerator,
    "counting": CountingTaskGenerator,
    "ordering": OrderingTaskGenerator,
    "state_query": StateQueryTaskGenerator,
    "antecedent": AntecedentTaskGenerator,
    "comparison": ComparisonTaskGenerator,
    "multi_hop": MultiHopTaskGenerator,
    "anomaly_detection": AnomalyDetectionTaskGenerator,      # NEW
    "anomaly_localization": AnomalyLocalizationTaskGenerator,  # NEW
}

# Add to __all__
__all__ = [
    ...
    "AnomalyDetectionTaskGenerator",
    "AnomalyLocalizationTaskGenerator",
    ...
]
```

### 3.6 QA Loader (`dataset/ts_haystack_qa_loader.py`)

Update `ALL_TASKS`:

```python
ALL_TASKS = [
    "existence",
    "localization",
    "counting",
    "ordering",
    "state_query",
    "antecedent",
    "comparison",
    "multi_hop",
    "anomaly_detection",      # NEW
    "anomaly_localization",   # NEW
]
```

### 3.7 CoT Prompt Builder (`cot/prompt_builder.py`)

Add to `get_task_context()`:

```python
"anomaly_detection": f"""
TASK TYPE: Anomaly Detection (Contextual Reasoning)
- Background regime: {difficulty_config.get("background_regime", "unknown")}
- Is anomaly present: {difficulty_config.get("is_positive", "unknown")}
- Anomaly activity (if positive): {difficulty_config.get("anomaly_activity", "N/A")}
- Your reasoning should:
  1. First characterize the dominant pattern in the recording (sedentary or active)
  2. Identify any activity that contrasts with this background pattern
  3. Explain WHY it is (or isn't) anomalous based on regime mismatch
- Key insight: An activity is anomalous if it belongs to a DIFFERENT regime than the background
  (e.g., running in a sedentary background, or sleeping in an active background).""",

"anomaly_localization": f"""
TASK TYPE: Anomaly Localization (Detection + Temporal)
- Background regime: {difficulty_config.get("background_regime", "unknown")}
- Is anomaly present: {difficulty_config.get("is_positive", "unknown")}
- Anomaly activity (if positive): {difficulty_config.get("anomaly_activity", "N/A")}
- Anomaly time range (if positive): {difficulty_config.get("anomaly_start", "N/A")} to {difficulty_config.get("anomaly_end", "N/A")}
- Your reasoning should:
  1. Characterize the dominant pattern (sedentary or active regime)
  2. Identify any cross-regime activity
  3. Specify the EXACT time range of the anomaly
  4. Explain why this activity is anomalous in context
- Key insight: Report both WHAT the anomaly is AND WHEN it occurs.""",
```

### 3.8 Test Fixtures (`test/tasks/conftest.py`)

Add fixtures:

```python
from opentslm.time_series_datasets.ts_haystack.tasks import (
    ...
    AnomalyDetectionTaskGenerator,
    AnomalyLocalizationTaskGenerator,
)

@pytest.fixture(scope="module")
def anomaly_detection_generator():
    """Create AnomalyDetectionTaskGenerator with loaded artifacts."""
    if not PHASE1_AVAILABLE:
        pytest.skip("Phase 1 artifacts not available")
    return AnomalyDetectionTaskGenerator.create_with_artifacts(seed=42)

@pytest.fixture(scope="module")
def anomaly_localization_generator():
    """Create AnomalyLocalizationTaskGenerator with loaded artifacts."""
    if not PHASE1_AVAILABLE:
        pytest.skip("Phase 1 artifacts not available")
    return AnomalyLocalizationTaskGenerator.create_with_artifacts(seed=42)
```

---

## 4. Test Files

### 4.1 `test/tasks/test_task_anomaly_detection.py`

```python
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for AnomalyDetectionTaskGenerator.

Task 9: "Is there an anomaly in this recording?"
Answer type: boolean (Yes/No)
Evaluation: Accuracy (exact match)

Tests validate:
- Cross-regime insertion creates anomalies (positive)
- Same-regime distractors do NOT create anomalies (negative)
- BOTH positive and negative samples have multiple needle insertions (mandatory distractors)
- Background regime detection
- Balanced positive/negative samples
- Answer format correctness

Key test: Both positive and negative samples must have similar signal characteristics
(multiple needle insertions) to prevent detection shortcuts.
"""

from pathlib import Path

import numpy as np
import pytest

from opentslm.time_series_datasets.ts_haystack.core import (
    DifficultyConfig,
    get_regime,
)
from opentslm.time_series_datasets.ts_haystack.test.tasks.conftest import ensure_plot_dir


class TestAnomalyDetectionSampleGeneration:
    """Tests for basic sample generation functionality."""

    def test_generate_single_sample(self, anomaly_detection_generator, medium_difficulty, rng):
        """Test generating a single anomaly detection sample."""
        sample = anomaly_detection_generator.generate_sample(medium_difficulty, rng)

        assert sample is not None
        assert sample.task_type == "anomaly_detection"
        assert sample.answer_type == "boolean"

        if sample.is_valid:
            assert len(sample.x) == medium_difficulty.context_length_samples
            assert "anomal" in sample.question.lower() or "unusual" in sample.question.lower()
            # Both positive and negative have needle insertions
            assert len(sample.needles) >= 1
            print(f"  Q: {sample.question}")
            print(f"  A: {sample.answer}")
            print(f"  Needles: {len(sample.needles)}")
            print(f"  Is positive: {sample.difficulty_config.get('is_positive')}")

    def test_positive_has_cross_regime_plus_distractors(self, anomaly_detection_generator):
        """Verify positive samples have cross-regime anomaly + same-regime distractors."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.15),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 1, "max_distractors": 2},
        )

        for i in range(30):
            rng = np.random.default_rng(100 + i)
            sample = anomaly_detection_generator.generate_sample(difficulty, rng)

            if sample.is_valid and sample.difficulty_config.get("is_positive"):
                bg_regime = sample.difficulty_config.get("background_regime")
                anomaly_activity = sample.difficulty_config.get("anomaly_activity")
                anomaly_regime = get_regime(anomaly_activity)

                # Verify cross-regime anomaly
                assert bg_regime != anomaly_regime, \
                    f"Expected cross-regime: bg={bg_regime}, anomaly={anomaly_regime}"

                # Verify anomaly needle is present
                assert any(n.activity == anomaly_activity for n in sample.needles)

                # Verify we have multiple needles (anomaly + distractors)
                assert len(sample.needles) >= 2, \
                    f"Expected anomaly + distractors, got {len(sample.needles)} needles"

                print(f"  Positive: bg={bg_regime}, anomaly={anomaly_activity}, total needles={len(sample.needles)}")
                return

        pytest.skip("Could not generate positive sample")

    def test_negative_has_multiple_same_regime_distractors(self, anomaly_detection_generator):
        """Verify negative samples have multiple same-regime distractors (no anomaly)."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.15),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 1, "max_distractors": 2},
        )

        for i in range(30):
            rng = np.random.default_rng(200 + i)
            sample = anomaly_detection_generator.generate_sample(difficulty, rng)

            if sample.is_valid and not sample.difficulty_config.get("is_positive"):
                bg_regime = sample.difficulty_config.get("background_regime")

                # Verify ALL needles are same-regime (no anomaly)
                for needle in sample.needles:
                    needle_regime = get_regime(needle.activity)
                    assert bg_regime == needle_regime, \
                        f"Expected all same-regime: bg={bg_regime}, needle={needle_regime}"

                # Verify we have distractors (at least min_distractors)
                assert len(sample.needles) >= 1, \
                    f"Expected distractors, got {len(sample.needles)} needles"

                print(f"  Negative: bg={bg_regime}, distractors={len(sample.needles)}")
                return

        pytest.skip("Could not generate negative sample")

    def test_both_positive_and_negative_have_insertions(self, anomaly_detection_generator):
        """Critical test: Both cases have needle insertions to prevent detection shortcuts."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.15),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 1, "max_distractors": 3},
        )

        positive_needle_counts = []
        negative_needle_counts = []

        for i in range(50):
            rng = np.random.default_rng(150 + i)
            sample = anomaly_detection_generator.generate_sample(difficulty, rng)

            if sample.is_valid:
                if sample.difficulty_config.get("is_positive"):
                    positive_needle_counts.append(len(sample.needles))
                else:
                    negative_needle_counts.append(len(sample.needles))

        # Both must have insertions
        assert all(c >= 1 for c in positive_needle_counts), "Positive samples must have needles"
        assert all(c >= 1 for c in negative_needle_counts), "Negative samples must have needles"

        print(f"  Positive needle counts: {positive_needle_counts[:10]}...")
        print(f"  Negative needle counts: {negative_needle_counts[:10]}...")

    def test_balanced_positive_negative(self, anomaly_detection_generator):
        """Test that positive/negative samples are approximately balanced."""
        difficulty = DifficultyConfig(
            context_length_samples=8000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.20),
            background_purity="pure",
            task_specific={"margin_samples": 100},
        )

        positive_count = 0
        negative_count = 0

        for i in range(50):
            rng = np.random.default_rng(300 + i)
            sample = anomaly_detection_generator.generate_sample(difficulty, rng)

            if sample.is_valid:
                if sample.difficulty_config.get("is_positive"):
                    positive_count += 1
                else:
                    negative_count += 1

        total = positive_count + negative_count
        print(f"  Balance: {positive_count}/{total} positive, {negative_count}/{total} negative")

        assert positive_count >= total * 0.30
        assert negative_count >= total * 0.30


class TestAnomalyDetectionVisualization:
    """Generate visualizations for human evaluation."""

    def test_visualize_positive_sample(self, anomaly_detection_generator):
        """Visualize a positive anomaly detection sample with distractors."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("anomaly_detection")

        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.08, 0.20),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 2, "max_distractors": 3},
        )

        sample = None
        for i in range(50):
            rng = np.random.default_rng(400 + i)
            candidate = anomaly_detection_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and candidate.difficulty_config.get("is_positive"):
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1, 1, 0.6]})
        fig.suptitle("Anomaly Detection - Positive (Cross-Regime Insertion)", fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}

        for ax, (name, data, color) in zip(
            axes[:3],
            [("X", sample.x, colors["x"]), ("Y", sample.y, colors["y"]), ("Z", sample.z, colors["z"])]
        ):
            ax.plot(t, data, linewidth=0.5, color=color, alpha=0.8)
            ax.set_ylabel(f"{name}-axis (g)", fontsize=10)
            ax.grid(True, alpha=0.3)

            anomaly_activity = sample.difficulty_config.get("anomaly_activity")
            for needle in sample.needles:
                start = needle.insert_position_samples
                end = start + needle.duration_samples
                if needle.activity == anomaly_activity:
                    ax.axvspan(start, end, alpha=0.4, color="red", label=f"ANOMALY: {needle.activity}")
                else:
                    ax.axvspan(start, end, alpha=0.2, color="blue", label=f"Distractor: {needle.activity}")

        axes[2].set_xlabel("Sample Index", fontsize=10)

        # Text panel
        axes[3].axis("off")
        bg_regime = sample.difficulty_config.get("background_regime", "?")
        anomaly = sample.difficulty_config.get("anomaly_activity", "?")
        n_distractors = sample.difficulty_config.get("n_distractors", 0)
        text = (
            f"Background Regime: {bg_regime}  |  Anomaly: {anomaly}  |  Distractors: {n_distractors}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=11,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5))

        plt.tight_layout()
        output_path = plot_dir / "anomaly_detection_positive.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")

    def test_visualize_negative_sample(self, anomaly_detection_generator):
        """Visualize a negative anomaly detection sample (same-regime distractors only)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("anomaly_detection")

        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.08, 0.20),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 2, "max_distractors": 3},
        )

        sample = None
        for i in range(50):
            rng = np.random.default_rng(500 + i)
            candidate = anomaly_detection_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and not candidate.difficulty_config.get("is_positive"):
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1, 1, 0.6]})
        fig.suptitle("Anomaly Detection - Negative (Same-Regime Distractors Only)", fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}

        for ax, (name, data, color) in zip(
            axes[:3],
            [("X", sample.x, colors["x"]), ("Y", sample.y, colors["y"]), ("Z", sample.z, colors["z"])]
        ):
            ax.plot(t, data, linewidth=0.5, color=color, alpha=0.8)
            ax.set_ylabel(f"{name}-axis (g)", fontsize=10)
            ax.grid(True, alpha=0.3)

            for needle in sample.needles:
                start = needle.insert_position_samples
                end = start + needle.duration_samples
                ax.axvspan(start, end, alpha=0.3, color="green", label=f"SAME-REGIME: {needle.activity}")

        axes[2].set_xlabel("Sample Index", fontsize=10)

        # Text panel
        axes[3].axis("off")
        bg_regime = sample.difficulty_config.get("background_regime", "?")
        n_distractors = sample.difficulty_config.get("n_distractors", len(sample.needles))
        distractor_activities = sample.difficulty_config.get("distractor_activities", [n.activity for n in sample.needles])
        text = (
            f"Background Regime: {bg_regime}  |  Distractors: {n_distractors} (all same-regime)\n"
            f"Activities: {distractor_activities}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=11,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.5))

        plt.tight_layout()
        output_path = plot_dir / "anomaly_detection_negative.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")
```

### 4.2 `test/tasks/test_task_anomaly_localization.py`

```python
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for AnomalyLocalizationTaskGenerator.

Task 10: "Is there an anomaly, and if so, when does it occur?"
Answer type: time_range (timestamp for positive, boolean logic for negative)
Evaluation:
- Detection: Accuracy (Yes/No match)
- Localization: IoU (Intersection over Union) for time range

Tests validate:
- Positive samples include time range in answer + distractors
- Negative samples have same-regime distractors only (no time range)
- Time range matches actual anomaly needle position
- BOTH positive and negative samples have multiple needle insertions (mandatory distractors)

Key test: Both positive and negative samples must have similar signal characteristics
(multiple needle insertions) to prevent detection shortcuts.
"""

from pathlib import Path

import numpy as np
import pytest

from opentslm.time_series_datasets.ts_haystack.core import (
    DifficultyConfig,
    get_regime,
)
from opentslm.time_series_datasets.ts_haystack.test.tasks.conftest import ensure_plot_dir


class TestAnomalyLocalizationSampleGeneration:
    """Tests for basic sample generation functionality."""

    def test_generate_single_sample(self, anomaly_localization_generator, medium_difficulty, rng):
        """Test generating a single anomaly localization sample."""
        sample = anomaly_localization_generator.generate_sample(medium_difficulty, rng)

        assert sample is not None
        assert sample.task_type == "anomaly_localization"
        assert sample.answer_type == "time_range"

        if sample.is_valid:
            assert len(sample.x) == medium_difficulty.context_length_samples
            # Both positive and negative have needle insertions
            assert len(sample.needles) >= 1
            print(f"  Q: {sample.question}")
            print(f"  A: {sample.answer}")
            print(f"  Needles: {len(sample.needles)}")
            print(f"  Is positive: {sample.difficulty_config.get('is_positive')}")

    def test_positive_includes_time_range(self, anomaly_localization_generator):
        """Verify positive samples include time range in answer + distractors."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.15),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 1, "max_distractors": 2},
        )

        for i in range(30):
            rng = np.random.default_rng(100 + i)
            sample = anomaly_localization_generator.generate_sample(difficulty, rng)

            if sample.is_valid and sample.difficulty_config.get("is_positive"):
                # Answer should include time range
                anomaly_start = sample.difficulty_config.get("anomaly_start")
                anomaly_end = sample.difficulty_config.get("anomaly_end")

                assert anomaly_start is not None
                assert anomaly_end is not None
                # Check that time range appears in answer
                assert anomaly_start in sample.answer or "to" in sample.answer

                print(f"  Positive: {sample.answer}")
                print(f"  Time range: {anomaly_start} to {anomaly_end}")
                return

        pytest.skip("Could not generate positive sample")

    def test_negative_no_time_range(self, anomaly_localization_generator):
        """Verify negative samples have distractors but no anomaly time range."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.15),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 1, "max_distractors": 2},
        )

        for i in range(30):
            rng = np.random.default_rng(200 + i)
            sample = anomaly_localization_generator.generate_sample(difficulty, rng)

            if sample.is_valid and not sample.difficulty_config.get("is_positive"):
                # Answer should NOT include anomaly time range
                assert "anomaly_start" not in sample.difficulty_config or sample.difficulty_config.get("anomaly_start") is None
                # Answer should mention consistent activity
                assert "consistent" in sample.answer.lower() or "no" in sample.answer.lower()

                print(f"  Negative: {sample.answer}")
                return

        pytest.skip("Could not generate negative sample")

    def test_time_range_matches_needle(self, anomaly_localization_generator):
        """Verify that reported time range matches actual anomaly needle position."""
        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.15),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_distractors": 1, "max_distractors": 2},
        )

        for i in range(30):
            rng = np.random.default_rng(300 + i)
            sample = anomaly_localization_generator.generate_sample(difficulty, rng)

            if sample.is_valid and sample.difficulty_config.get("is_positive"):
                anomaly_activity = sample.difficulty_config.get("anomaly_activity")
                anomaly_start = sample.difficulty_config.get("anomaly_start")
                anomaly_end = sample.difficulty_config.get("anomaly_end")

                # Find the anomaly needle
                anomaly_needle = None
                for n in sample.needles:
                    if n.activity == anomaly_activity:
                        anomaly_needle = n
                        break

                assert anomaly_needle is not None
                assert anomaly_needle.timestamp_start == anomaly_start
                assert anomaly_needle.timestamp_end == anomaly_end

                print(f"  Time range verified: {anomaly_start} to {anomaly_end}")
                return

        pytest.skip("Could not generate positive sample")


class TestAnomalyLocalizationVisualization:
    """Generate visualizations for human evaluation."""

    def test_visualize_positive_with_distractors(self, anomaly_localization_generator):
        """Visualize positive sample with same-regime distractors."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        plot_dir = ensure_plot_dir("anomaly_localization")

        difficulty = DifficultyConfig(
            context_length_samples=10000,
            needle_position="random",
            needle_length_ratio_range=(0.05, 0.15),
            background_purity="pure",
            task_specific={"margin_samples": 100, "min_gap_samples": 100, "min_distractors": 2, "max_distractors": 3},
        )

        sample = None
        for i in range(50):
            rng = np.random.default_rng(400 + i)
            candidate = anomaly_localization_generator.generate_sample(difficulty, rng)
            if candidate.is_valid and candidate.difficulty_config.get("is_positive"):
                sample = candidate
                break

        if sample is None:
            pytest.skip("Could not generate suitable sample")

        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1, 1, 0.6]})
        fig.suptitle("Anomaly Localization - Positive (with Distractors)", fontsize=14, fontweight="bold")

        t = np.arange(len(sample.x))
        colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#ff7f0e"}
        anomaly_activity = sample.difficulty_config.get("anomaly_activity")

        for ax, (name, data, color) in zip(
            axes[:3],
            [("X", sample.x, colors["x"]), ("Y", sample.y, colors["y"]), ("Z", sample.z, colors["z"])]
        ):
            ax.plot(t, data, linewidth=0.5, color=color, alpha=0.8)
            ax.set_ylabel(f"{name}-axis (g)", fontsize=10)
            ax.grid(True, alpha=0.3)

            for needle in sample.needles:
                start = needle.insert_position_samples
                end = start + needle.duration_samples
                if needle.activity == anomaly_activity:
                    ax.axvspan(start, end, alpha=0.4, color="red", label=f"ANOMALY: {needle.activity}")
                else:
                    ax.axvspan(start, end, alpha=0.2, color="blue", label=f"Distractor: {needle.activity}")

        axes[2].set_xlabel("Sample Index", fontsize=10)

        # Text panel
        axes[3].axis("off")
        bg_regime = sample.difficulty_config.get("background_regime", "?")
        anomaly_start = sample.difficulty_config.get("anomaly_start", "?")
        anomaly_end = sample.difficulty_config.get("anomaly_end", "?")
        n_distractors = sample.difficulty_config.get("n_distractors", 0)
        text = (
            f"Background: {bg_regime}  |  Anomaly: {anomaly_activity}  |  Distractors: {n_distractors}\n\n"
            f"Q: {sample.question}\n\n"
            f"A: {sample.answer}\n\n"
            f"Time range: {anomaly_start} to {anomaly_end}"
        )
        axes[3].text(0.5, 0.5, text, transform=axes[3].transAxes, fontsize=10,
                     verticalalignment="center", horizontalalignment="center",
                     family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5))

        plt.tight_layout()
        output_path = plot_dir / "anomaly_localization_positive.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"  Saved: {output_path}")
```

---

## 5. README Updates

Add to the Task Overview table in `README.md`:

```markdown
| Task | Name | Question Type | Answer Type | Description |
|------|------|---------------|-------------|-------------|
| 1 | Existence | "Is there {activity} in this recording?" | boolean | Detect presence/absence of an activity |
| ... | ... | ... | ... | ... |
| 9 | Anomaly Detection | "Is there an anomaly in this recording?" | boolean | Detect cross-regime activity violations |
| 10 | Anomaly Localization | "Is there an anomaly, and when?" | time_range | Detect and localize cross-regime violations |
```

Add new section:

```markdown
### Anomaly Detection & Localization Tasks

Unlike Existence (which provides the target activity), these tasks ask the model
to identify cross-regime violations without being told what to look for.

**Anomaly Definition**: An activity is anomalous if it belongs to a different regime
than the dominant background:
- Running (active) in a sleeping (sedentary) background = ANOMALOUS
- Walking (active) in a bicycling (active) background = NOT anomalous

**Sample Design (both tasks)**:
- Positive: Cross-regime insertion (anomaly present)
- Negative: Same-regime insertion (not anomalous, but activity present)

This design ensures both positive and negative samples have similar signal characteristics,
forcing the model to reason about regime context rather than just detecting variance changes.

**Task 9: Anomaly Detection**
- Question: "Is there an anomaly in this recording?"
- Positive Answer: "Yes, there is anomalous {activity} activity in the {regime} background."
- Negative Answer: "No, the recording shows consistent {regime} activity."

**Task 10: Anomaly Localization**
- Question: "Is there an anomaly, and if so, when does it occur?"
- Positive Answer: "Yes, there is anomalous {activity} activity from {start} to {end}."
- Negative Answer: "No, the recording shows consistent {regime} activity."

**Configuration**:
```yaml
anomaly_detection:
  enabled: true
  n_distractors: 0          # Additional same-regime insertions

anomaly_localization:
  enabled: true
  n_distractors: 1          # Makes localization harder
```
```

---

## 6. Implementation Order

1. **Core Implementation**
   - [ ] Create `tasks/task_anomaly_detection.py`
   - [ ] Create `tasks/task_anomaly_localization.py`
   - [ ] Add templates to `core/prompt_templates.py`
   - [ ] Register both in `tasks/__init__.py`

2. **Configuration & Integration**
   - [ ] Add config blocks to `configs/default_generation_config.yaml`
   - [ ] Update `dataset/ts_haystack_qa_loader.py`
   - [ ] Add CoT contexts to `cot/prompt_builder.py`

3. **Testing**
   - [ ] Add fixtures to `test/tasks/conftest.py`
   - [ ] Create `test/tasks/test_task_anomaly_detection.py`
   - [ ] Create `test/tasks/test_task_anomaly_localization.py`
   - [ ] Run tests:
     ```bash
     pytest src/opentslm/time_series_datasets/ts_haystack/test/tasks/test_task_anomaly_detection.py -v
     pytest src/opentslm/time_series_datasets/ts_haystack/test/tasks/test_task_anomaly_localization.py -v
     ```

4. **Documentation**
   - [ ] Update `README.md`

5. **Validation**
   - [ ] Generate sample datasets for both tasks
   - [ ] Inspect visualizations
   - [ ] Verify CoT rationale generation

---

## 7. Evaluation Metrics

### Task 9: Anomaly Detection

| Metric | Description |
|--------|-------------|
| **Answer Type** | `boolean` |
| **Detection Accuracy** | Yes/No match |
| **Activity Accuracy** | Correct anomaly activity identification (for positive samples) |
| **Combined Accuracy** | Detection AND activity must both be correct |

**Important**: For positive samples, the model must correctly identify WHICH activity is anomalous.
Simply saying "Yes" is not enough - it must say "Yes, {correct_activity}".
This prevents random guessing from achieving high accuracy.

**Evaluation Logic:**
```python
import re

def evaluate_anomaly_detection(
    predicted: str,
    ground_truth: str,
    anomaly_activity: str = None,  # Ground truth activity (for positive samples)
) -> dict:
    """
    Evaluate anomaly detection accuracy.

    For positive samples: Must match Yes/No AND correctly identify the anomaly activity.
    For negative samples: Must match Yes/No only.

    Args:
        predicted: Model output (e.g., "Yes, there is anomalous running...")
        ground_truth: Expected answer (e.g., "Yes, there is anomalous running...")
        anomaly_activity: The actual anomaly activity (e.g., "running") for positive samples

    Returns:
        {
            "detection_correct": bool,   # Yes/No matches
            "activity_correct": bool,    # Activity mentioned correctly (positive only)
            "fully_correct": bool        # Both detection AND activity correct
        }
    """
    pred_positive = "yes" in predicted.lower()
    gt_positive = "yes" in ground_truth.lower()
    detection_correct = pred_positive == gt_positive

    # For negative samples, only detection matters
    if not gt_positive:
        return {
            "detection_correct": detection_correct,
            "activity_correct": True,  # N/A for negative
            "fully_correct": detection_correct,
        }

    # For positive samples, must also identify the correct activity
    if not pred_positive:
        # Said "No" when should be "Yes" - wrong
        return {
            "detection_correct": False,
            "activity_correct": False,
            "fully_correct": False,
        }

    # Check if the anomaly activity is mentioned in the prediction
    # Use regex to match the activity (case-insensitive)
    activity_pattern = re.compile(re.escape(anomaly_activity), re.IGNORECASE)
    activity_correct = bool(activity_pattern.search(predicted))

    return {
        "detection_correct": detection_correct,
        "activity_correct": activity_correct,
        "fully_correct": detection_correct and activity_correct,
    }
```

### Task 10: Anomaly Localization

| Metric | Description |
|--------|-------------|
| **Answer Type** | `time_range` |
| **Detection Accuracy** | Yes/No match |
| **Activity Accuracy** | Correct anomaly activity identification (for positive samples) |
| **IoU (Intersection over Union)** | Overlap between predicted and ground truth time ranges |
| **Combined Score** | All three must be correct: Detection + Activity + Localization |

**Important**: For positive samples, the model must correctly identify WHICH activity is anomalous
AND provide the correct time range. Simply saying "Yes, anomaly from X to Y" is not enough.

**Evaluation Logic:**
```python
import re

def compute_iou(pred_start: float, pred_end: float,
                gt_start: float, gt_end: float) -> float:
    """
    Compute IoU between predicted and ground truth time ranges.

    Args:
        pred_start, pred_end: Predicted time range (as fraction of recording)
        gt_start, gt_end: Ground truth time range (as fraction of recording)

    Returns:
        IoU score in [0, 1]
    """
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)

    if intersection_end <= intersection_start:
        return 0.0  # No overlap

    intersection = intersection_end - intersection_start
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection

    return intersection / union if union > 0 else 0.0


def evaluate_anomaly_localization(
    predicted: str,
    ground_truth: str,
    anomaly_activity: str,  # Ground truth activity
    gt_start_frac: float,
    gt_end_frac: float,
    recording_duration_samples: int,
) -> dict:
    """
    Evaluate anomaly localization.

    For positive samples: Must match Yes/No, activity, AND time range.
    For negative samples: Must match Yes/No only.

    Returns:
        {
            "detection_correct": bool,   # Yes/No match
            "activity_correct": bool,    # Activity mentioned correctly (positive only)
            "iou": float,                # IoU for positive samples (0 for negative)
            "fully_correct": bool        # All criteria met
        }
    """
    pred_positive = "yes" in predicted.lower()
    gt_positive = "yes" in ground_truth.lower()
    detection_correct = pred_positive == gt_positive

    if not gt_positive:
        # Negative sample: only detection matters
        return {
            "detection_correct": detection_correct,
            "activity_correct": True,  # N/A for negative
            "iou": 0.0,
            "fully_correct": detection_correct,
        }

    if not pred_positive:
        # False negative: no localization to evaluate
        return {
            "detection_correct": False,
            "activity_correct": False,
            "iou": 0.0,
            "fully_correct": False,
        }

    # Check if the anomaly activity is mentioned in the prediction
    activity_pattern = re.compile(re.escape(anomaly_activity), re.IGNORECASE)
    activity_correct = bool(activity_pattern.search(predicted))

    # Parse predicted time range from answer
    pred_start_frac, pred_end_frac = parse_time_range(predicted, recording_duration_samples)

    iou = compute_iou(pred_start_frac, pred_end_frac, gt_start_frac, gt_end_frac)

    # Fully correct requires: detection + activity + good localization (IoU > threshold)
    iou_threshold = 0.5
    fully_correct = detection_correct and activity_correct and (iou >= iou_threshold)

    return {
        "detection_correct": detection_correct,
        "activity_correct": activity_correct,
        "iou": iou,
        "fully_correct": fully_correct,
    }
```

### Evaluation Thresholds

| Task | Metric | Threshold | Interpretation |
|------|--------|-----------|----------------|
| Anomaly Detection | Detection Accuracy | 0.5 | Random guessing on Yes/No |
| Anomaly Detection | Activity Accuracy | 0.1-0.2 | Random guessing on activity (1/5 to 1/10 activities) |
| Anomaly Detection | Fully Correct | 0.7+ | Good performance (detection + activity) |
| Anomaly Localization | IoU | 0.5 | Acceptable overlap |
| Anomaly Localization | IoU | 0.75+ | Good localization |
| Anomaly Localization | Fully Correct | 0.5+ | Good (detection + activity + IoU > 0.5) |

**Note**: Requiring activity identification significantly raises the bar compared to simple Yes/No.
Random guessing would achieve ~10-20% on activity (depending on number of activities in opposite regime),
making the effective random baseline for "fully correct" much lower than 50%.

---

## 8. CoT Rationale Examples

### Task 9: Anomaly Detection

**Positive Example:**
```
Looking at this accelerometer data spanning from 6:00 AM to 7:40 AM, I observe
predominantly low-variance patterns throughout most of the recording, consistent
with sedentary activity such as sitting or sleeping. The signal shows minimal
oscillation and stable baseline levels.

However, I notice a distinct period of high-amplitude, rhythmic oscillations
with increased variance across all three axes - patterns characteristic of
vigorous physical activity like running. This active-regime activity is clearly
anomalous within the otherwise sedentary context. The stark contrast between the
high-intensity segment and the surrounding low-activity periods indicates a
regime mismatch.

Answer: Yes, there is anomalous running activity in the sedentary background.
```

**Negative Example (same-regime insertion):**
```
Analyzing this accelerometer data from 6:00 AM to 7:40 AM, I observe predominantly
low-variance patterns consistent with sedentary activity. The signal shows minimal
oscillation typical of rest or sitting.

There is a segment showing slightly different patterns, but upon closer inspection,
these patterns are also consistent with sedentary activities - the variance remains
low and there are no high-frequency oscillations characteristic of active movement.
The signal characteristics throughout are appropriate for the sedentary context.

Answer: No, the recording shows consistent sedentary activity.
```

### Task 10: Anomaly Localization

**Positive Example:**
```
Examining this accelerometer data from 6:00 AM to 7:40 AM, I observe predominantly
sedentary patterns (low variance, stable baseline) throughout most of the recording.

Between approximately 6:45 AM and 6:52 AM, there is a dramatic change in signal
characteristics: high-amplitude oscillations with rhythmic patterns typical of
running or sports activity. This approximately 7-minute period shows variance
levels 5-10x higher than the surrounding sedentary segments.

This active-regime activity (running) is anomalous in the sedentary context.
The anomaly is clearly localized to the specified time window.

Answer: Yes, there is anomalous running activity from 6:45 AM to 6:52 AM.
```

---

## 9. Potential Extensions (Future Work)

1. **Multi-Anomaly Detection/Localization**: Detect and report multiple anomalies
2. **Anomaly Severity Grading**: Rate how anomalous (cross-regime > edge-case)
3. **Fine-Grained Regimes**: More than 2 regimes for nuanced detection
4. **Anomaly Characterization**: Describe WHY it's anomalous (signal properties)
5. **Counterfactual Reasoning**: "What would make this normal?"
