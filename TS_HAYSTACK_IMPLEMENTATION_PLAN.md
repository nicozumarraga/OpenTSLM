# TS-Haystack Implementation Plan

A semi-synthetic benchmark for testing retrieval and reasoning over long time series (1K–1M+ datapoints) using Capture-24 accelerometer data.

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Directory Structure](#2-directory-structure)
3. [Core Infrastructure](#3-core-infrastructure)
4. [Data Structures](#4-data-structures)
5. [Task Generators](#5-task-generators)
6. [Semi-Synthetic Bout Insertion](#6-semi-synthetic-bout-insertion)
7. [QADataset Integration](#7-qadataset-integration)
8. [Implementation Phases](#8-implementation-phases)
9. [Reproducibility & Seed Strategy](#9-reproducibility--seed-strategy)

---

## 1. Architecture Overview

### Design Philosophy

The TS-Haystack benchmark builds on the existing Capture-24 classification pipeline but introduces a fundamentally different paradigm:

| Aspect | Capture-24 Classification | TS-Haystack |
|--------|--------------------------|-------------|
| **Task Type** | Single-label classification | Multi-step retrieval + reasoning |
| **Window Selection** | Non-overlapping | Controlled selection with needle insertion |
| **Ground Truth** | Activity label | Task-specific (boolean, timestamp, count, etc.) |
| **Difficulty** | Fixed | Configurable via multiple axes |

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXISTING INFRASTRUCTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  capture24.zip → capture24_loader.py → sensor_data_100hz/pid=*/data.parquet │
│                                        participants.parquet                  │
│                                        label_mappings.parquet               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TS-HAYSTACK INFRASTRUCTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: Timeline & Index Building                                         │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌──────────────────┐│
│  │  timeline_builder   │───▶│   bout_indexer      │───▶│ transition_matrix││
│  │  (per-participant)  │    │  (cross-participant)│    │ (global stats)   ││
│  └─────────────────────┘    └─────────────────────┘    └──────────────────┘│
│           │                          │                          │           │
│           ▼                          ▼                          ▼           │
│  timelines/P001.parquet    bout_index.parquet       transition_matrix.json  │
│  timelines/P001.json       bout_index.json                                  │
│                                                                             │
│  Phase 2: Task Generation                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                       TASK GENERATORS                                   ││
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐           ││
│  │  │ existence  │ │localization│ │  counting  │ │ ordering   │           ││
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘           ││
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐           ││
│  │  │state_query │ │ antecedent │ │ comparison │ │ multi_hop  │           ││
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘           ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    BOUT INSERTION ENGINE                                ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  ││
│  │  │ background   │  │   needle     │  │    style     │                  ││
│  │  │  sampler     │  │   sampler    │  │  transfer    │                  ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│           │                                                                 │
│           ▼                                                                 │
│  tasks/{task_name}/{context_len}/{split}/data.parquet                      │
│  tasks/{task_name}/metadata.json                                           │
│                                                                             │
│  Phase 3: QADataset Integration                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  TSHaystackQADataset(QADataset)                                        ││
│  │    - _load_splits()                                                    ││
│  │    - _get_answer() → task-specific answer formatting                   ││
│  │    - _get_pre_prompt() → timestamp context + task instruction          ││
│  │    - _get_post_prompt() → answer format guidance                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

### Source Code

```
src/opentslm/time_series_datasets/
├── capture24/                          # Existing
│   ├── __init__.py
│   ├── capture24_loader.py
│   ├── capture24_windows.py
│   ├── capture24_classification.py
│   ├── capture24_qa_loader.py
│   ├── Capture24AccQADataset.py
│   └── ...
│
└── ts_haystack/                        # New
    ├── __init__.py                     # Module exports
    │
    ├── core/                           # Core infrastructure
    │   ├── __init__.py
    │   ├── seed_manager.py             # Reproducibility & seed management
    │   ├── timeline_builder.py         # Activity timeline extraction
    │   ├── bout_indexer.py             # Cross-participant bout index
    │   ├── transition_matrix.py        # Activity transition probabilities
    │   ├── background_sampler.py       # Sample background windows
    │   ├── needle_sampler.py           # Sample needles from bout index
    │   ├── style_transfer.py           # Covariance projection + blending
    │   └── prompt_templates.py         # NL template bank for Q/A diversity
    │
    ├── tasks/                          # Task generators
    │   ├── __init__.py
    │   ├── base_task.py                # Abstract base class
    │   ├── task_existence.py           # Task 1: Existence
    │   ├── task_localization.py        # Task 2: Localization
    │   ├── task_counting.py            # Task 3: Counting
    │   ├── task_ordering.py            # Task 4: Temporal Ordering
    │   ├── task_state_query.py         # Task 5: State Query
    │   ├── task_antecedent.py          # Task 6: Temporal Antecedent
    │   ├── task_comparison.py          # Task 7: Comparison & Negation
    │   └── task_multi_hop.py           # Task 8: Multi-Hop Localization
    │
    ├── dataset/                        # Dataset classes
    │   ├── __init__.py
    │   ├── ts_haystack_qa_loader.py    # Parquet → HuggingFace Dataset
    │   └── TSHaystackQADataset.py      # QADataset implementation
    │
    ├── utils/                          # Utilities
    │   ├── __init__.py
    │   ├── validation.py               # Sample validation & quality checks
    │   ├── metrics.py                  # Task-specific evaluation metrics
    │   └── visualization.py            # Plotting for debugging
    │
    └── test/                           # Tests
        ├── __init__.py
        ├── test_timeline_builder.py
        ├── test_bout_indexer.py
        ├── test_style_transfer.py
        └── test_tasks.py
```

### Data Output

```
data/capture24/
├── sensor_data_100hz/                  # Existing raw data
├── participants.parquet                # Existing metadata
├── label_mappings.parquet              # Existing label schemes
│
└── ts_haystack/                        # New
    ├── timelines/                      # Phase 1 output
    │   ├── P001.parquet
    │   ├── P001.json                   # Dev observability
    │   ├── P002.parquet
    │   └── ...
    │
    ├── bout_index.parquet              # Cross-participant bout index
    ├── bout_index.json                 # Dev observability
    │
    ├── transition_matrix.json          # Activity transition probabilities
    │
    └── tasks/                          # Phase 2 output
        ├── existence/
        │   ├── 1k/
        │   │   ├── train/data.parquet
        │   │   ├── val/data.parquet
        │   │   └── test/data.parquet
        │   ├── 10k/
        │   ├── 100k/
        │   ├── 1m/
        │   └── metadata.json
        │
        ├── localization/
        │   └── ...
        │
        └── {task_name}/
            └── ...
```

---

## 3. Core Infrastructure

### 3.1 TimelineBuilder (`core/timeline_builder.py`)

**Purpose:** Extract and pre-compute activity timelines from raw sensor data.

```python
class TimelineBuilder:
    """
    Builds activity timelines by merging consecutive same-activity annotations
    into contiguous bouts.
    """

    def __init__(
        self,
        label_scheme: str = "Walmsley2020",
        source_hz: int = 100,
        min_bout_duration_ms: int = 1000,  # 1 second minimum
    ):
        """
        Args:
            label_scheme: Which label scheme to use for activity mapping
            source_hz: Source data frequency
            min_bout_duration_ms: Minimum bout duration to keep
        """
        ...

    def build_participant_timeline(
        self,
        pid: str,
    ) -> ParticipantTimeline:
        """
        Build timeline for a single participant.

        Returns:
            ParticipantTimeline dataclass with:
            - participant_id: str
            - total_duration_ms: int
            - recording_start_ms: int
            - recording_end_ms: int
            - timeline: List[BoutRecord]  # Chronological sequence
            - bouts_by_activity: Dict[str, List[BoutRecord]]  # Indexed by activity
        """
        ...

    def build_all_timelines(
        self,
        n_jobs: int = 1,
        max_participants: Optional[int] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Build and save timelines for all participants.
        Saves both .parquet (efficient) and .json (readable) formats.
        """
        ...

    @staticmethod
    def load_timeline(pid: str) -> ParticipantTimeline:
        """Load pre-computed timeline for a participant."""
        ...

    @staticmethod
    def load_all_timelines() -> Dict[str, ParticipantTimeline]:
        """Load all pre-computed timelines."""
        ...
```

**Key Algorithm: Bout Merging**

```python
def _merge_annotations_to_bouts(
    self,
    df: pl.DataFrame,  # Columns: timestamp_ms, annotation
) -> List[BoutRecord]:
    """
    Merge consecutive same-activity samples into bouts.

    Algorithm:
    1. Map annotations to labels using label_scheme
    2. Identify activity transitions (label[i] != label[i-1])
    3. For each contiguous segment:
       - Record: start_ms, end_ms, activity, duration_ms
    4. Filter by min_bout_duration_ms
    5. Return chronological list of BoutRecord
    """
    ...
```

### 3.2 BoutIndexer (`core/bout_indexer.py`)

**Purpose:** Create a cross-participant index of all bouts for efficient needle sampling.

```python
class BoutIndexer:
    """
    Aggregates bouts across all participants into a queryable index.
    Enables efficient sampling of needles by activity type and duration.
    """

    def __init__(self, min_bout_duration_ms: int = 1000):
        self.min_bout_duration_ms = min_bout_duration_ms

    def build_index(
        self,
        timelines: Dict[str, ParticipantTimeline],
    ) -> BoutIndex:
        """
        Build cross-participant bout index.

        Returns:
            BoutIndex with:
            - by_activity: Dict[str, List[BoutRef]]
            - activity_stats: Dict[str, ActivityStats]  # count, durations
        """
        ...

    def save_index(self, index: BoutIndex, overwrite: bool = False) -> None:
        """Save index as parquet + json."""
        ...

    @staticmethod
    def load_index() -> BoutIndex:
        """Load pre-computed bout index."""
        ...

    def sample_bout(
        self,
        activity: str,
        min_duration_ms: int,
        max_duration_ms: Optional[int] = None,
        exclude_pids: Optional[Set[str]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[BoutRef]:
        """
        Sample a bout matching criteria.

        Args:
            activity: Target activity type
            min_duration_ms: Minimum bout duration
            max_duration_ms: Maximum bout duration (optional)
            exclude_pids: Participants to exclude (optional, not typically needed)
            rng: Random generator for reproducibility

        Returns:
            BoutRef with (pid, start_ms, end_ms, duration_ms) or None if no match
        """
        ...
```

### 3.3 TransitionMatrix (`core/transition_matrix.py`)

**Purpose:** Compute activity transition probabilities for realistic needle pairing.

```python
class TransitionMatrix:
    """
    Computes P(activity_next | activity_current) from timeline data.
    Used to guide plausible (needle, background) pairings.
    """

    def __init__(self):
        self.activities: List[str] = []
        self.matrix: np.ndarray = None  # Shape: (n_activities, n_activities)
        self.activity_to_idx: Dict[str, int] = {}

    def build_from_timelines(
        self,
        timelines: Dict[str, ParticipantTimeline],
    ) -> None:
        """
        Build transition matrix from all participant timelines.

        For each consecutive (bout_i, bout_{i+1}) pair:
            matrix[activity_i][activity_{i+1}] += 1

        Then row-normalize to get probabilities.
        """
        ...

    def get_transition_prob(
        self,
        from_activity: str,
        to_activity: str,
    ) -> float:
        """Get P(to_activity | from_activity)."""
        ...

    def sample_successor(
        self,
        activity: str,
        exclude: Optional[Set[str]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> str:
        """Sample a plausible successor activity."""
        ...

    def sample_predecessor(
        self,
        activity: str,
        exclude: Optional[Set[str]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> str:
        """Sample a plausible predecessor activity."""
        ...

    def save(self, path: Optional[Path] = None) -> None:
        """Save as JSON."""
        ...

    @staticmethod
    def load() -> "TransitionMatrix":
        """Load pre-computed matrix."""
        ...
```

### 3.4 BackgroundSampler (`core/background_sampler.py`)

**Purpose:** Sample background windows for needle insertion.

```python
class BackgroundSampler:
    """
    Samples background windows from participant recordings.
    Provides control over background purity and activity composition.
    """

    def __init__(
        self,
        timelines: Dict[str, ParticipantTimeline],
        bout_index: BoutIndex,
        source_hz: int = 100,
    ):
        self.timelines = timelines
        self.bout_index = bout_index
        self.source_hz = source_hz

    def sample_background(
        self,
        context_length_samples: int,
        purity: Literal["pure", "mixed"] = "pure",
        allowed_activities: Optional[Set[str]] = None,
        excluded_activities: Optional[Set[str]] = None,
        min_activity_count: int = 1,
        max_activity_count: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> BackgroundSample:
        """
        Sample a background window.

        Args:
            context_length_samples: Window size in samples
            purity: "pure" = single activity, "mixed" = multiple activities
            allowed_activities: Only sample from these activities
            excluded_activities: Never sample from these activities
            min_activity_count: Minimum distinct activities in window
            max_activity_count: Maximum distinct activities in window
            rng: Random generator

        Returns:
            BackgroundSample with:
            - pid: str
            - start_ms: int
            - end_ms: int
            - x, y, z: np.ndarray  # Sensor values
            - activities_present: Set[str]
            - activity_timeline: List[Tuple[float, float, str]]  # (start_frac, end_frac, activity)
            - recording_time_context: Tuple[str, str]  # ("6:00 AM", "8:00 AM")
        """
        ...

    def _load_sensor_window(
        self,
        pid: str,
        start_ms: int,
        end_ms: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load x, y, z arrays for a window."""
        ...
```

### 3.5 NeedleSampler (`core/needle_sampler.py`)

**Purpose:** Sample needle bouts from the bout index.

```python
class NeedleSampler:
    """
    Samples needle bouts from the cross-participant bout index.
    Handles duration constraints and participant exclusion.
    """

    def __init__(
        self,
        bout_index: BoutIndex,
        transition_matrix: TransitionMatrix,
        source_hz: int = 100,
    ):
        self.bout_index = bout_index
        self.transition_matrix = transition_matrix
        self.source_hz = source_hz

    def sample_needle(
        self,
        activity: str,
        min_duration_ms: int,
        max_duration_ms: Optional[int] = None,
        target_duration_ms: Optional[int] = None,  # Exact if possible
        exclude_pids: Optional[Set[str]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> NeedleSample:
        """
        Sample a needle bout.

        Returns:
            NeedleSample with:
            - pid: str (source participant)
            - activity: str
            - start_ms: int
            - end_ms: int
            - x, y, z: np.ndarray  # Raw sensor values
        """
        ...

    def sample_needle_for_background(
        self,
        background: BackgroundSample,
        activity: Optional[str] = None,  # None = sample from transition matrix
        use_transition_probs: bool = True,
        **kwargs,
    ) -> NeedleSample:
        """
        Sample a needle appropriate for a given background.
        Optionally uses transition matrix for realistic pairing.
        """
        ...
```

### 3.6 PromptTemplateBank (`core/prompt_templates.py`)

**Purpose:** Manage diverse natural language templates for questions and answers to prevent model overfitting on specific phrasings.

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TemplateVariant:
    """A single question/answer template with placeholders."""
    question: str
    answer: str
    requires_plural: bool = False  # Hint for grammatical handling


class PromptTemplateBank:
    """
    Manages natural language templates per task to prevent phrasing overfitting.

    Design Principles:
    - Separation of concerns: NLP phrasing is decoupled from task logic
    - Data-driven: Templates can be extended without code changes
    - Grammatically correct: Auto-handles singular/plural, yes/no variants
    - Reproducible: Uses same RNG chain for deterministic template selection

    Usage:
        bank = PromptTemplateBank()
        q, a = bank.sample("counting", rng=rng, activity="walking", count=3)
    """

    # Templates use {placeholder} syntax for variable substitution
    TEMPLATES: Dict[str, List[TemplateVariant]] = {

        # Task 1: Existence
        "existence": [
            TemplateVariant(
                question="Is there any {activity} in this recording?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Does this recording contain {activity}?",
                answer="{yes_no}, it {does_doesnt}.",
            ),
            TemplateVariant(
                question="Can you detect {activity} in the sensor data?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Is {activity} present in this accelerometer data?",
                answer="{yes_no}, {activity} is {present_absent}.",
            ),
            TemplateVariant(
                question="Did the person perform {activity} during this period?",
                answer="{yes_no}.",
            ),
        ],

        # Task 2: Localization
        "localization": [
            TemplateVariant(
                question="When did the {activity} bout occur?",
                answer="The {activity} bout occurred from {start} to {end}.",
            ),
            TemplateVariant(
                question="At what time was {activity} detected?",
                answer="{activity} was detected between {start} and {end}.",
            ),
            TemplateVariant(
                question="Identify when {activity} happened.",
                answer="From {start} to {end}.",
            ),
            TemplateVariant(
                question="During which time period did {activity} take place?",
                answer="{activity} took place from {start} to {end}.",
            ),
        ],

        # Task 3: Counting
        "counting": [
            TemplateVariant(
                question="How many {activity} bouts occurred in this recording?",
                answer="There {is_are} {count} {activity} {bout_bouts}.",
            ),
            TemplateVariant(
                question="Count the number of {activity} episodes.",
                answer="{count}.",
            ),
            TemplateVariant(
                question="How many times did {activity} occur?",
                answer="{activity} occurred {count} {time_times}.",
            ),
            TemplateVariant(
                question="What is the total count of {activity} bouts?",
                answer="The total count is {count}.",
            ),
        ],

        # Task 4: Ordering
        "ordering": [
            TemplateVariant(
                question="Did {activity_a} occur before {activity_b}?",
                answer="{yes_no}.",
            ),
            TemplateVariant(
                question="Which occurred first, {activity_a} or {activity_b}?",
                answer="{first_activity}.",
            ),
            TemplateVariant(
                question="Was {activity_a} performed before {activity_b}?",
                answer="{yes_no}, {activity_a} {came_before_after} {activity_b}.",
            ),
            TemplateVariant(
                question="In what order did {activity_a} and {activity_b} happen?",
                answer="{first_activity} occurred first, followed by {second_activity}.",
            ),
        ],

        # Task 5: State Query
        "state_query": [
            TemplateVariant(
                question="What was the overall activity level when the {needle_activity} spike occurred?",
                answer="The overall activity level was {global_state}.",
            ),
            TemplateVariant(
                question="During the {needle_activity} event, what was the general activity regime?",
                answer="{global_state}.",
            ),
            TemplateVariant(
                question="What activity state was the person in when {needle_activity} happened?",
                answer="The person was in a {global_state} state.",
            ),
        ],

        # Task 6: Antecedent
        "antecedent": [
            TemplateVariant(
                question="What activity occurred immediately before the {target_activity} bout?",
                answer="{antecedent_activity}.",
            ),
            TemplateVariant(
                question="Which activity preceded {target_activity}?",
                answer="{antecedent_activity} preceded {target_activity}.",
            ),
            TemplateVariant(
                question="What was the person doing right before {target_activity}?",
                answer="The person was {antecedent_activity}.",
            ),
        ],

        # Task 7: Comparison
        "comparison": [
            TemplateVariant(
                question="What was the {extremum} period {polarity} {activity}?",
                answer="The {extremum} period was from {start} to {end}.",
            ),
            TemplateVariant(
                question="Identify the {extremum} {activity} bout.",
                answer="From {start} to {end} ({duration}).",
            ),
            TemplateVariant(
                question="When was the {extremum} stretch {polarity} {activity}?",
                answer="The {extremum} stretch {polarity} {activity} was from {start} to {end}.",
            ),
        ],

        # Task 8: Multi-Hop
        "multi_hop": [
            TemplateVariant(
                question="When did the {ordinal} {target_activity} bout occur {direction} the {anchor_activity}?",
                answer="The {ordinal} {target_activity} bout {direction} {anchor_activity} occurred from {start} to {end}.",
            ),
            TemplateVariant(
                question="Identify the {ordinal} {target_activity} {direction} {anchor_activity}.",
                answer="From {start} to {end}.",
            ),
            TemplateVariant(
                question="After locating {anchor_activity}, when was the {ordinal} {target_activity} bout {direction} it?",
                answer="It occurred from {start} to {end}.",
            ),
        ],
    }

    # Ordinal mappings for multi-hop
    ORDINALS = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}

    def __init__(self, custom_templates: Optional[Dict[str, List[TemplateVariant]]] = None):
        """
        Initialize with default templates, optionally extended with custom ones.

        Args:
            custom_templates: Additional templates to merge with defaults
        """
        self.templates = {task: list(variants) for task, variants in self.TEMPLATES.items()}
        if custom_templates:
            for task, variants in custom_templates.items():
                self.templates.setdefault(task, []).extend(variants)

    def sample(
        self,
        task: str,
        rng: np.random.Generator,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Sample a random template and fill placeholders.

        Args:
            task: Task name (existence, localization, etc.)
            rng: Random generator for reproducible selection
            **kwargs: Placeholder values (activity, count, start, end, etc.)

        Returns:
            (question, answer) tuple with placeholders filled

        Example:
            >>> bank.sample("counting", rng, activity="walking", count=3)
            ("How many walking bouts occurred?", "There are 3 walking bouts.")
        """
        variants = self.templates.get(task)
        if not variants:
            raise ValueError(f"No templates registered for task: {task}")

        # Randomly select a template variant
        variant = variants[rng.integers(0, len(variants))]

        # Auto-generate grammatical helpers based on provided kwargs
        filled_kwargs = self._add_grammar_helpers(kwargs)

        # Fill placeholders
        try:
            question = variant.question.format(**filled_kwargs)
            answer = variant.answer.format(**filled_kwargs)
        except KeyError as e:
            raise ValueError(f"Missing placeholder {e} for task '{task}'. Provided: {list(kwargs.keys())}")

        return question, answer

    def _add_grammar_helpers(self, kwargs: Dict) -> Dict:
        """
        Auto-generate grammatical helper variables based on context.

        Handles:
        - Singular/plural: is/are, bout/bouts, time/times
        - Boolean: yes/no, does/doesn't, present/absent
        - Ordinals: 1st, 2nd, 3rd
        - Ordering: came before/after
        """
        filled = dict(kwargs)

        # Count-based singular/plural
        if "count" in filled:
            count = filled["count"]
            filled["is_are"] = "is" if count == 1 else "are"
            filled["bout_bouts"] = "bout" if count == 1 else "bouts"
            filled["time_times"] = "time" if count == 1 else "times"

        # Boolean yes/no helpers
        if "exists" in filled:
            exists = filled["exists"]
            filled["yes_no"] = "Yes" if exists else "No"
            filled["does_doesnt"] = "does" if exists else "doesn't"
            filled["present_absent"] = "present" if exists else "absent"

        # Ordering helpers
        if "a_before_b" in filled:
            a_before_b = filled["a_before_b"]
            filled["yes_no"] = "Yes" if a_before_b else "No"
            filled["came_before_after"] = "came before" if a_before_b else "came after"

        # Ordinal conversion
        if "k" in filled:
            k = filled["k"]
            filled["ordinal"] = self.ORDINALS.get(k, f"{k}th")

        # Duration formatting (if duration_ms provided)
        if "duration_ms" in filled:
            duration_ms = filled["duration_ms"]
            if duration_ms >= 60000:
                mins = duration_ms / 60000
                filled["duration"] = f"{mins:.1f} minutes"
            else:
                secs = duration_ms / 1000
                filled["duration"] = f"{secs:.1f} seconds"

        return filled

    def get_template_count(self, task: str) -> int:
        """Return number of template variants for a task."""
        return len(self.templates.get(task, []))

    def register_templates(self, task: str, variants: List[TemplateVariant]) -> None:
        """Add new templates for a task at runtime."""
        self.templates.setdefault(task, []).extend(variants)

    @classmethod
    def from_json(cls, path: str) -> "PromptTemplateBank":
        """
        Load templates from a JSON file.

        Expected format:
        {
            "existence": [
                {"question": "Is there {activity}?", "answer": "{yes_no}"},
                ...
            ],
            ...
        }
        """
        import json
        with open(path) as f:
            data = json.load(f)

        custom = {}
        for task, variants in data.items():
            custom[task] = [TemplateVariant(**v) for v in variants]

        return cls(custom_templates=custom)
```

**Integration with BaseTaskGenerator:**

Update `BaseTaskGenerator.__init__` to accept and use the template bank:

```python
class BaseTaskGenerator(ABC):
    def __init__(
        self,
        background_sampler: BackgroundSampler,
        needle_sampler: NeedleSampler,
        style_transfer: StyleTransfer,
        config: TaskConfig,
        seed_manager: Optional[SeedManager] = None,
        template_bank: Optional[PromptTemplateBank] = None,  # NEW
    ):
        self.background_sampler = background_sampler
        self.needle_sampler = needle_sampler
        self.style_transfer = style_transfer
        self.config = config
        self.seed_manager = seed_manager or SeedManager(config.seed)
        self.template_bank = template_bank or PromptTemplateBank()  # NEW

    def _format_qa(
        self,
        rng: np.random.Generator,
        **kwargs,
    ) -> Tuple[str, str]:
        """
        Generate question/answer pair using varied templates.

        Delegates to PromptTemplateBank for natural language diversity.
        """
        return self.template_bank.sample(self.task_name, rng, **kwargs)
```

**Usage in Task Generators:**

Instead of hardcoding question/answer strings, tasks call `_format_qa`:

```python
# In CountingTaskGenerator.generate_sample():
question, answer = self._format_qa(
    rng=rng,
    activity=target_activity,
    count=actual_count,
)

# In ExistenceTaskGenerator.generate_sample():
question, answer = self._format_qa(
    rng=rng,
    activity=target_activity,
    exists=is_positive,
)

# In MultiHopTaskGenerator.generate_sample():
question, answer = self._format_qa(
    rng=rng,
    target_activity=target_activity,
    anchor_activity=anchor_activity,
    k=K,
    direction=direction,
    start=answer_needle.timestamp_start,
    end=answer_needle.timestamp_end,
)
```

**Benefits:**

| Benefit | Description |
|---------|-------------|
| **Prevents overfitting** | Model sees varied phrasings, learns task semantics not surface patterns |
| **Separation of concerns** | Task logic (sampling, insertion) decoupled from NLP formatting |
| **Easy extension** | Add templates via JSON or `register_templates()` without touching task code |
| **Grammatically correct** | Auto-handles singular/plural, ordinals, yes/no variants |
| **Reproducible** | Template selection uses same RNG chain as sample generation |
| **Testable** | Templates can be unit tested independently of task logic |

---

## 4. Data Structures

### 4.1 Core Dataclasses

```python
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
import numpy as np

@dataclass
class BoutRecord:
    """A single activity bout within a participant's timeline."""
    start_ms: int
    end_ms: int
    activity: str
    duration_ms: int

    @property
    def center_ms(self) -> int:
        return (self.start_ms + self.end_ms) // 2

@dataclass
class BoutRef:
    """Reference to a bout in the cross-participant index."""
    pid: str
    start_ms: int
    end_ms: int
    duration_ms: int
    activity: str

@dataclass
class ParticipantTimeline:
    """Complete activity timeline for one participant."""
    participant_id: str
    total_duration_ms: int
    recording_start_ms: int
    recording_end_ms: int
    timeline: List[BoutRecord]  # Chronological
    bouts_by_activity: Dict[str, List[BoutRecord]]

@dataclass
class BoutIndex:
    """Cross-participant bout index."""
    by_activity: Dict[str, List[BoutRef]]
    activity_stats: Dict[str, "ActivityStats"]

@dataclass
class ActivityStats:
    """Statistics for an activity across all participants."""
    count: int
    total_duration_ms: int
    mean_duration_ms: float
    std_duration_ms: float
    min_duration_ms: int
    max_duration_ms: int
    duration_percentiles: Dict[int, float]  # {10: val, 25: val, 50: val, 75: val, 90: val}

@dataclass
class BackgroundSample:
    """A sampled background window."""
    pid: str
    start_ms: int
    end_ms: int
    duration_ms: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    activities_present: Set[str]
    activity_timeline: List[Tuple[float, float, str]]  # (start_frac, end_frac, activity)
    recording_time_context: Tuple[str, str]  # Human-readable time range

@dataclass
class NeedleSample:
    """A sampled needle bout."""
    source_pid: str
    activity: str
    start_ms: int
    end_ms: int
    duration_ms: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

@dataclass
class InsertedNeedle:
    """Record of an inserted needle with position info."""
    activity: str
    source_pid: str
    source_start_ms: int
    source_end_ms: int
    insert_position_samples: int  # Position in output sequence
    insert_position_frac: float   # Fractional position (0-1)
    duration_samples: int
    duration_ms: int
    timestamp_start: str  # Human-readable
    timestamp_end: str    # Human-readable

@dataclass
class GeneratedSample:
    """A complete generated sample for any task."""
    # Sensor data
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    # Metadata
    task_type: str
    context_length_samples: int
    background_pid: str
    recording_time_range: Tuple[str, str]

    # Question/Answer
    question: str
    answer: str
    answer_type: str  # "boolean", "timestamp", "integer", "category", "time_range"

    # Needle metadata (for analysis)
    needles: List[InsertedNeedle]

    # Difficulty indicators
    difficulty_config: Dict[str, any]

    # Validation
    is_valid: bool
    validation_notes: Optional[str] = None
```

### 4.2 Task-Specific Structures

```python
@dataclass
class ExistenceAnswer:
    """Answer for existence task."""
    exists: bool
    activity: str

@dataclass
class LocalizationAnswer:
    """Answer for localization task."""
    start_timestamp: str  # "9:15 AM"
    end_timestamp: str    # "9:20 AM"
    start_frac: float     # 0.0-1.0
    end_frac: float       # 0.0-1.0

@dataclass
class CountingAnswer:
    """Answer for counting task."""
    count: int
    activity: str
    needle_positions: List[Tuple[str, str]]  # List of (start, end) timestamps

@dataclass
class OrderingAnswer:
    """Answer for ordering task."""
    first_activity: str
    second_activity: str
    answer: str  # "Yes" / "No" or activity name

@dataclass
class StateQueryAnswer:
    """Answer for state query task."""
    local_event: str      # The needle activity
    global_state: str     # The regime at needle position
    timestamp: str        # When the local event occurred

@dataclass
class AntecedentAnswer:
    """Answer for antecedent task."""
    target_activity: str
    antecedent_activity: str

@dataclass
class ComparisonAnswer:
    """Answer for comparison task."""
    extremum: str         # "longest" / "shortest"
    polarity: str         # "with" / "without"
    activity: str
    answer_period: Tuple[str, str]  # Timestamp range
    duration_ms: int

@dataclass
class MultiHopAnswer:
    """Answer for multi-hop task."""
    anchor_activity: str
    target_activity: str
    ordinal: int          # K-th occurrence
    direction: str        # "before" / "after"
    answer_period: Tuple[str, str]
```

### 4.3 Configuration Structures

```python
@dataclass
class DifficultyConfig:
    """Difficulty configuration for sample generation."""
    # Context
    context_length_samples: int  # 1K, 10K, 100K, 1M

    # Needle characteristics
    needle_position: str = "random"  # "beginning", "middle", "end", "random"
    needle_length_distribution: str = "uniform"  # "uniform", "activity_specific"
    needle_length_range_ms: Tuple[int, int] = (5000, 300000)  # 5s to 5min

    # Distractors
    distractor_density: str = "none"  # "none", "low", "high"
    distractor_count: int = 0

    # Background
    background_purity: str = "pure"  # "pure", "mixed"

    # Task-specific
    task_specific: Dict[str, any] = field(default_factory=dict)

@dataclass
class TaskConfig:
    """Configuration for a task generator."""
    task_name: str
    samples_per_split: Dict[str, int]  # {"train": 10000, "val": 1000, "test": 1000}
    context_lengths: List[int]  # [1000, 10000, 100000, 1000000]
    difficulty_levels: List[DifficultyConfig]
    label_scheme: str = "Walmsley2020"
    seed: int = 42
```

---

## 5. Task Generators

### 5.1 Base Task Generator

```python
from abc import ABC, abstractmethod

class BaseTaskGenerator(ABC):
    """
    Abstract base class for all task generators.
    Provides common infrastructure for sample generation.
    """

    def __init__(
        self,
        background_sampler: BackgroundSampler,
        needle_sampler: NeedleSampler,
        style_transfer: StyleTransfer,
        config: TaskConfig,
    ):
        self.background_sampler = background_sampler
        self.needle_sampler = needle_sampler
        self.style_transfer = style_transfer
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Unique task identifier."""
        ...

    @property
    @abstractmethod
    def answer_type(self) -> str:
        """Type of answer: boolean, timestamp, integer, category, time_range."""
        ...

    @abstractmethod
    def generate_sample(
        self,
        difficulty: DifficultyConfig,
    ) -> GeneratedSample:
        """Generate a single sample."""
        ...

    @abstractmethod
    def format_question(self, sample: GeneratedSample) -> str:
        """Format the question string."""
        ...

    @abstractmethod
    def format_answer(self, sample: GeneratedSample) -> str:
        """Format the answer string."""
        ...

    def generate_dataset(
        self,
        n_samples: int,
        difficulty: DifficultyConfig,
        split: str,
    ) -> List[GeneratedSample]:
        """Generate multiple samples for a split."""
        samples = []
        attempts = 0
        max_attempts = n_samples * 3  # Allow retries for invalid samples

        while len(samples) < n_samples and attempts < max_attempts:
            sample = self.generate_sample(difficulty)
            if sample.is_valid:
                samples.append(sample)
            attempts += 1

        return samples

    def validate_sample(self, sample: GeneratedSample) -> Tuple[bool, str]:
        """
        Validate a generated sample.
        Override in subclasses for task-specific validation.
        """
        # Check basic properties
        if len(sample.x) != sample.context_length_samples:
            return False, "Context length mismatch"

        if not sample.question or not sample.answer:
            return False, "Missing question or answer"

        return True, "Valid"

    def save_dataset(
        self,
        samples: List[GeneratedSample],
        split: str,
        context_length: int,
    ) -> None:
        """Save generated samples as parquet."""
        ...
```

### 5.2 Task 1: Existence

```python
class ExistenceTaskGenerator(BaseTaskGenerator):
    """
    Task 1: "Is there {activity_x} in this recording?"

    PseudoCode:
    1. Sample background window
    2. Identify activities present and absent
    3. Generate balanced positive/negative questions
    4. Optionally insert needle for positive cases
    """

    task_name = "existence"
    answer_type = "boolean"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
    ) -> GeneratedSample:
        # Sample background
        background = self.background_sampler.sample_background(
            context_length_samples=difficulty.context_length_samples,
            purity=difficulty.background_purity,
        )

        # Determine if positive or negative sample (50/50)
        is_positive = self.rng.random() < 0.5

        if is_positive:
            # For positive: use activity that IS in background
            # OR insert a needle of an absent activity
            ...
        else:
            # For negative: ask about activity NOT in background
            ...

        return GeneratedSample(...)

    def format_question(self, sample: GeneratedSample) -> str:
        activity = sample.difficulty_config["target_activity"]
        return f"Is there any {activity} in this recording?"

    def format_answer(self, sample: GeneratedSample) -> str:
        return "Yes" if sample.answer == "true" else "No"
```

### 5.3 Task 2: Localization

```python
class LocalizationTaskGenerator(BaseTaskGenerator):
    """
    Task 2: "When did the {activity_x} bout occur?"

    PseudoCode:
    1. Sample background with controlled activity composition (purity/mixed)
    2. Determine needle activity from activities NOT present in background
    3. Sample needle from bout index
    4. Apply style transfer to match target participant's signal characteristics
    5. Insert at controlled position with boundary blending
    6. Ground answer in real timestamps
    """

    task_name = "localization"
    answer_type = "timestamp"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
    ) -> GeneratedSample:
        # Step 1: Sample background with n activity types based on difficulty
        # The background determines what activities are "natural" in this window
        background = self.background_sampler.sample_background(
            context_length_samples=difficulty.context_length_samples,
            purity=difficulty.background_purity,  # "pure" or "mixed"
            min_activity_count=difficulty.task_specific.get("min_bg_activities", 1),
            max_activity_count=difficulty.task_specific.get("max_bg_activities", None),
        )

        # Step 2: Determine needle activity = something NOT in background
        # This ensures the needle is truly foreign to the context
        all_activities = set(self.needle_sampler.get_available_activities())
        candidate_activities = all_activities - background.activities_present

        if not candidate_activities:
            # Fallback: resample background or use least common activity
            raise SamplingError("No candidate activities for needle insertion")

        # Sample needle activity (uniform or weighted by transition matrix)
        if difficulty.task_specific.get("use_transition_probs", False):
            # Weight by plausibility given background activities
            target_activity = self.needle_sampler.sample_activity_given_context(
                context_activities=background.activities_present,
                candidates=candidate_activities,
                rng=self.rng,
            )
        else:
            target_activity = self.rng.choice(list(candidate_activities))

        # Step 3: Sample needle bout from index
        needle = self.needle_sampler.sample_needle(
            activity=target_activity,
            min_duration_ms=difficulty.needle_length_range_ms[0],
            max_duration_ms=difficulty.needle_length_range_ms[1],
            exclude_pids=None,  # Same-participant sampling is allowed
            rng=self.rng,
        )

        # Step 4: Determine insertion position
        insert_position = self._sample_position(
            context_length=difficulty.context_length_samples,
            needle_length=len(needle.x),
            position_mode=difficulty.needle_position,  # "beginning", "middle", "end", "random"
        )

        # Step 5: Apply style transfer to match target context
        local_stats = self._compute_local_stats(background, insert_position)
        transferred_needle = self.style_transfer.transfer(
            needle=needle,
            target_stats=local_stats,
        )

        # Step 6: Insert needle with boundary blending
        x, y, z = self.style_transfer.insert_with_blending(
            background=(background.x, background.y, background.z),
            needle=(transferred_needle.x, transferred_needle.y, transferred_needle.z),
            position=insert_position,
        )

        # Build metadata
        needle_metadata = InsertedNeedle(
            activity=target_activity,
            source_pid=needle.source_pid,
            source_start_ms=needle.start_ms,
            source_end_ms=needle.end_ms,
            insert_position_samples=insert_position,
            insert_position_frac=insert_position / difficulty.context_length_samples,
            duration_samples=len(needle.x),
            duration_ms=needle.duration_ms,
            timestamp_start=self._samples_to_timestamp(insert_position, background),
            timestamp_end=self._samples_to_timestamp(insert_position + len(needle.x), background),
        )

        return GeneratedSample(
            x=x, y=y, z=z,
            task_type=self.task_name,
            context_length_samples=difficulty.context_length_samples,
            background_pid=background.pid,
            recording_time_range=background.recording_time_context,
            question=self.format_question_for_needle(needle_metadata),
            answer=self.format_answer_for_needle(needle_metadata),
            answer_type=self.answer_type,
            needles=[needle_metadata],
            difficulty_config=asdict(difficulty),
            is_valid=True,
        )

    def format_question(self, sample: GeneratedSample) -> str:
        activity = sample.needles[0].activity
        return f"When did the {activity} bout occur?"

    def format_answer(self, sample: GeneratedSample) -> str:
        needle = sample.needles[0]
        return f"The {needle.activity} bout occurred from {needle.timestamp_start} to {needle.timestamp_end}."
```

### 5.4 Task 3: Counting

```python
class CountingTaskGenerator(BaseTaskGenerator):
    """
    Task 3: "How many {activity_x} bouts occurred?"

    Core Logic:
    - Insert N bouts of the same target activity into a background
    - N is sampled from a distribution (e.g., Uniform(1, max_bouts))
    - Bouts must not overlap and must have minimum gap between them
    - Answer is the integer count

    Key Constraint: Minimum gap between bouts to ensure they are distinguishable.

    Difficulty Knobs:
    - N (number of bouts): higher = harder
    - Bout duration variability: similar lengths = harder to distinguish
    - Gap between bouts: smaller gap = harder
    - Background purity: mixed background = harder
    - Context length: longer = harder (more signal to scan)
    - Conditional variant: "How many {activity_x} bouts during {time_period}?"
    """

    task_name = "counting"
    answer_type = "integer"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        # Step 1: Sample background window
        background = self.background_sampler.sample_background(
            context_length_samples=difficulty.context_length_samples,
            purity=difficulty.background_purity,
            rng=rng,
        )

        # Step 2: Determine target activity (must NOT be in background)
        all_activities = set(self.needle_sampler.get_available_activities())
        candidate_activities = all_activities - background.activities_present
        if not candidate_activities:
            return self._create_invalid_sample("No candidate activities")

        target_activity = rng.choice(list(candidate_activities))

        # Step 3: Sample N = number of bouts to insert
        # Get available bouts for this activity with sufficient length
        min_needle_ms = difficulty.needle_length_range_ms[0]
        max_needle_ms = difficulty.needle_length_range_ms[1]
        available_bouts = self.bout_index.get_bouts_by_activity(
            target_activity, min_duration_ms=min_needle_ms
        )

        max_bouts = difficulty.task_specific.get("max_bouts", 5)
        min_bouts = difficulty.task_specific.get("min_bouts", 1)
        n_bouts = rng.integers(min_bouts, min(max_bouts, len(available_bouts)) + 1)

        # Step 4: Sample needle lengths and positions
        min_gap_samples = difficulty.task_specific.get("min_gap_samples", 100)  # ~1s at 100Hz

        needles = []
        occupied_ranges = []  # List of (start, end) tuples

        for i in range(n_bouts):
            # Sample needle duration
            needle_duration_ms = rng.integers(min_needle_ms, max_needle_ms + 1)
            needle_duration_samples = int(needle_duration_ms * self.source_hz / 1000)

            # Sample needle from bout index
            needle = self.needle_sampler.sample_needle(
                activity=target_activity,
                min_duration_ms=needle_duration_ms,
                max_duration_ms=needle_duration_ms + 5000,  # Allow some slack
                exclude_pids=None,  # Same-participant sampling is allowed
                rng=rng,
            )

            if needle is None:
                continue  # Skip if no suitable needle found

            # Trim needle to target duration
            needle = self._trim_needle(needle, needle_duration_samples)

            # Find valid insertion position (no overlap, respecting min_gap)
            position = self._find_valid_position(
                context_length=difficulty.context_length_samples,
                needle_length=len(needle.x),
                occupied_ranges=occupied_ranges,
                min_gap=min_gap_samples,
                rng=rng,
            )

            if position is None:
                continue  # No valid position found

            # Record occupied range
            occupied_ranges.append((position, position + len(needle.x)))

            # Apply style transfer
            local_stats = self._compute_local_stats(background, position)
            transferred_needle = self.style_transfer.transfer(needle, local_stats)

            needles.append((transferred_needle, position))

        if len(needles) == 0:
            return self._create_invalid_sample("Could not insert any needles")

        # Step 5: Insert all needles into background
        x, y, z = background.x.copy(), background.y.copy(), background.z.copy()
        needle_metadata = []

        for needle, position in sorted(needles, key=lambda x: x[1]):
            x, y, z = self.style_transfer.insert_with_blending(
                background=(x, y, z),
                needle=(needle.x, needle.y, needle.z),
                position=position,
            )

            needle_metadata.append(InsertedNeedle(
                activity=target_activity,
                source_pid=needle.source_pid,
                source_start_ms=needle.start_ms,
                source_end_ms=needle.end_ms,
                insert_position_samples=position,
                insert_position_frac=position / difficulty.context_length_samples,
                duration_samples=len(needle.x),
                duration_ms=needle.duration_ms,
                timestamp_start=self._samples_to_timestamp(position, background),
                timestamp_end=self._samples_to_timestamp(position + len(needle.x), background),
            ))

        # Step 6: Generate question and answer
        actual_count = len(needles)

        return GeneratedSample(
            x=x, y=y, z=z,
            task_type=self.task_name,
            context_length_samples=difficulty.context_length_samples,
            background_pid=background.pid,
            recording_time_range=background.recording_time_context,
            question=f"How many {target_activity} bouts occurred in this recording?",
            answer=str(actual_count),
            answer_type=self.answer_type,
            needles=needle_metadata,
            difficulty_config={
                **asdict(difficulty),
                "target_activity": target_activity,
                "n_bouts": actual_count,
                "min_gap_samples": min_gap_samples,
            },
            is_valid=True,
        )

    def _find_valid_position(
        self,
        context_length: int,
        needle_length: int,
        occupied_ranges: List[Tuple[int, int]],
        min_gap: int,
        rng: np.random.Generator,
        max_attempts: int = 100,
    ) -> Optional[int]:
        """Find a position that doesn't overlap with existing needles."""
        for _ in range(max_attempts):
            # Sample candidate position
            max_start = context_length - needle_length
            if max_start <= 0:
                return None
            position = rng.integers(0, max_start)

            # Check for conflicts with existing needles
            needle_end = position + needle_length
            valid = True

            for occ_start, occ_end in occupied_ranges:
                # Check overlap with gap consideration
                if not (needle_end + min_gap <= occ_start or position >= occ_end + min_gap):
                    valid = False
                    break

            if valid:
                return position

        return None

    def format_answer(self, sample: GeneratedSample) -> str:
        count = int(sample.answer)
        activity = sample.difficulty_config["target_activity"]
        if count == 1:
            return f"There is 1 {activity} bout in this recording."
        return f"There are {count} {activity} bouts in this recording."
```

### 5.5 Task 4: Temporal Ordering

```python
class OrderingTaskGenerator(BaseTaskGenerator):
    """
    Task 4: "Did {activity_x} occur before {activity_y}?" or
            "Which occurred first, {activity_x} or {activity_y}?"

    Core Logic:
    - Insert two distinct needle activities A and B into a clean background
    - Background must NOT contain A or B naturally (prevents ambiguity)
    - Randomly assign temporal order (50/50 split for balanced labels)
    - Control gap between bouts for difficulty modulation

    Why Needle Insertion (not natural bout selection):
    - Natural selection would allow solving via activity transition matrix
    - Insertion gives experimental control and prevents data leakage

    Difficulty Knobs:
    - Needle lengths: shorter = harder to detect
    - Gap between activities: smaller gap = harder
    - Context length: longer = more signal to scan
    - Question format: Boolean ("Did A occur before B?") vs Category ("Which first?")
    """

    task_name = "ordering"
    answer_type = "boolean"  # Can also be "category" for "Which occurred first?"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        # Step 1: Sample two distinct needle activities
        all_activities = list(self.needle_sampler.get_available_activities())
        if len(all_activities) < 2:
            return self._create_invalid_sample("Need at least 2 activities")

        # Sample A and B (can use transition matrix or uniform)
        if difficulty.task_specific.get("use_transition_probs", False):
            activity_A = rng.choice(all_activities)
            # Sample B from plausible successors of A
            activity_B = self.transition_matrix.sample_successor(
                activity_A, exclude={activity_A}, rng=rng
            )
        else:
            # Uniform random (prevents learning transition patterns)
            activities = rng.choice(all_activities, size=2, replace=False)
            activity_A, activity_B = activities[0], activities[1]

        # Step 2: Sample background window that does NOT contain A or B
        background = self.background_sampler.sample_background(
            context_length_samples=difficulty.context_length_samples,
            excluded_activities={activity_A, activity_B},
            purity=difficulty.background_purity,
            rng=rng,
        )

        if activity_A in background.activities_present or activity_B in background.activities_present:
            return self._create_invalid_sample("Background contains target activities")

        # Step 3: Sample needles from bout index
        needle_A = self.needle_sampler.sample_needle(
            activity=activity_A,
            min_duration_ms=difficulty.needle_length_range_ms[0],
            max_duration_ms=difficulty.needle_length_range_ms[1],
            exclude_pids=None,  # Same-participant sampling is allowed
            rng=rng,
        )

        needle_B = self.needle_sampler.sample_needle(
            activity=activity_B,
            min_duration_ms=difficulty.needle_length_range_ms[0],
            max_duration_ms=difficulty.needle_length_range_ms[1],
            exclude_pids=None,  # Same-participant sampling is allowed
            rng=rng,
        )

        if needle_A is None or needle_B is None:
            return self._create_invalid_sample("Could not sample needles")

        # Step 4: Sample insertion positions with controlled gap
        min_gap_samples = difficulty.task_specific.get("min_gap_samples", 100)
        total_needle_length = len(needle_A.x) + len(needle_B.x) + min_gap_samples

        if total_needle_length >= difficulty.context_length_samples:
            return self._create_invalid_sample("Needles too long for context")

        # Sample first position
        max_first_pos = difficulty.context_length_samples - total_needle_length
        t1 = rng.integers(0, max_first_pos + 1)

        # Second position must be after first needle + gap
        t2 = t1 + len(needle_A.x) + min_gap_samples

        # Optionally add random extra gap
        max_extra_gap = difficulty.context_length_samples - t2 - len(needle_B.x)
        if max_extra_gap > 0:
            extra_gap = rng.integers(0, min(max_extra_gap, min_gap_samples * 3) + 1)
            t2 += extra_gap

        # Step 5: Randomly assign order (50/50 for balanced labels)
        # Flip: if True, A comes first; if False, B comes first
        a_first = rng.random() < 0.5

        if a_first:
            pos_A, pos_B = t1, t2
            answer = "Yes"  # A occurred before B
            first_activity, second_activity = activity_A, activity_B
        else:
            pos_A, pos_B = t2, t1
            answer = "No"  # B occurred before A (A did NOT occur before B)
            first_activity, second_activity = activity_B, activity_A

        # Step 6: Apply style transfer and insert needles
        stats_A = self._compute_local_stats(background, pos_A)
        stats_B = self._compute_local_stats(background, pos_B)

        transferred_A = self.style_transfer.transfer(needle_A, stats_A)
        transferred_B = self.style_transfer.transfer(needle_B, stats_B)

        x, y, z = background.x.copy(), background.y.copy(), background.z.copy()

        # Insert in temporal order (earlier position first)
        for needle, pos in sorted([(transferred_A, pos_A), (transferred_B, pos_B)], key=lambda x: x[1]):
            x, y, z = self.style_transfer.insert_with_blending(
                (x, y, z), (needle.x, needle.y, needle.z), pos
            )

        # Step 7: Build metadata
        needle_metadata = [
            InsertedNeedle(
                activity=activity_A,
                source_pid=needle_A.source_pid,
                source_start_ms=needle_A.start_ms,
                source_end_ms=needle_A.end_ms,
                insert_position_samples=pos_A,
                insert_position_frac=pos_A / difficulty.context_length_samples,
                duration_samples=len(needle_A.x),
                duration_ms=needle_A.duration_ms,
                timestamp_start=self._samples_to_timestamp(pos_A, background),
                timestamp_end=self._samples_to_timestamp(pos_A + len(needle_A.x), background),
            ),
            InsertedNeedle(
                activity=activity_B,
                source_pid=needle_B.source_pid,
                source_start_ms=needle_B.start_ms,
                source_end_ms=needle_B.end_ms,
                insert_position_samples=pos_B,
                insert_position_frac=pos_B / difficulty.context_length_samples,
                duration_samples=len(needle_B.x),
                duration_ms=needle_B.duration_ms,
                timestamp_start=self._samples_to_timestamp(pos_B, background),
                timestamp_end=self._samples_to_timestamp(pos_B + len(needle_B.x), background),
            ),
        ]

        # Question format options
        question_format = difficulty.task_specific.get("question_format", "boolean")
        if question_format == "boolean":
            question = f"Did {activity_A} occur before {activity_B}?"
        else:  # "category"
            question = f"Which occurred first, {activity_A} or {activity_B}?"
            answer = first_activity

        return GeneratedSample(
            x=x, y=y, z=z,
            task_type=self.task_name,
            context_length_samples=difficulty.context_length_samples,
            background_pid=background.pid,
            recording_time_range=background.recording_time_context,
            question=question,
            answer=answer,
            answer_type="boolean" if question_format == "boolean" else "category",
            needles=needle_metadata,
            difficulty_config={
                **asdict(difficulty),
                "activity_A": activity_A,
                "activity_B": activity_B,
                "true_order": f"{first_activity} before {second_activity}",
                "gap_samples": abs(pos_B - pos_A) - len(needle_A.x if a_first else needle_B.x),
            },
            is_valid=True,
        )
```

### 5.6 Task 5: State Query (Cross-Scale)

```python
class StateQueryTaskGenerator(BaseTaskGenerator):
    """
    Task 5: "What was the activity level when the {local_event} occurred?"

    Core Logic (Cross-Scale Integration):
    - Window must contain multiple global activity states/regimes
    - Insert a short needle (local event) within one of the global states
    - Model must identify BOTH the local event AND the global context
    - Answer is the global state, not the local event

    This tests simultaneous attention to multiple temporal resolutions:
    - Local: needle-scale (seconds)
    - Global: regime-scale (minutes-hours)

    Difficulty Knobs:
    - Needle position: center of state (easy) vs near boundary (hard)
    - Number of global states in window: more = harder
    - Needle duration: shorter = harder to detect
    - State duration: shorter states = harder
    """

    task_name = "state_query"
    answer_type = "category"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        # Step 1: Sample background window with multiple global states
        min_states = difficulty.task_specific.get("min_global_states", 2)
        max_states = difficulty.task_specific.get("max_global_states", 4)

        background = self.background_sampler.sample_background(
            context_length_samples=difficulty.context_length_samples,
            purity="mixed",  # Must have multiple activities
            min_activity_count=min_states,
            max_activity_count=max_states,
            rng=rng,
        )

        # Extract global state timeline from background
        # Format: [(start_frac, end_frac, activity), ...]
        global_timeline = background.activity_timeline

        if len(global_timeline) < min_states:
            return self._create_invalid_sample(f"Need >= {min_states} global states")

        # Step 2: Identify candidate needle activities (NOT in background)
        background_activities = background.activities_present
        all_activities = set(self.needle_sampler.get_available_activities())
        candidate_needles = all_activities - background_activities

        if not candidate_needles:
            return self._create_invalid_sample("No candidate needle activities")

        # Step 3: Select target global state for the question
        # Pick a state that's large enough to contain the needle
        min_state_duration_samples = difficulty.task_specific.get("min_state_duration_samples", 500)
        context_len = difficulty.context_length_samples

        valid_states = []
        for start_frac, end_frac, activity in global_timeline:
            state_duration = int((end_frac - start_frac) * context_len)
            if state_duration >= min_state_duration_samples:
                valid_states.append((start_frac, end_frac, activity, state_duration))

        if not valid_states:
            return self._create_invalid_sample("No valid states for needle insertion")

        # Randomly select target state
        target_state = valid_states[rng.integers(0, len(valid_states))]
        state_start_frac, state_end_frac, target_global_activity, state_duration = target_state

        # Step 4: Sample needle event type
        # Can weight by plausibility P(needle | global_state) or use uniform
        if difficulty.task_specific.get("use_plausibility_weights", False):
            needle_activity = self._sample_plausible_needle(
                target_global_activity, candidate_needles, rng
            )
        else:
            needle_activity = rng.choice(list(candidate_needles))

        # Step 5: Sample needle from bout index
        needle = self.needle_sampler.sample_needle(
            activity=needle_activity,
            min_duration_ms=difficulty.needle_length_range_ms[0],
            max_duration_ms=difficulty.needle_length_range_ms[1],
            exclude_pids=None,  # Same-participant sampling is allowed
            rng=rng,
        )

        if needle is None:
            return self._create_invalid_sample(f"No needle found for {needle_activity}")

        # Step 6: Sample insertion position WITHIN target global state
        # Position parameter: "center" (easy) or "near_boundary" (hard)
        position_mode = difficulty.task_specific.get("position_mode", "random")
        margin_frac = difficulty.task_specific.get("boundary_margin_frac", 0.1)

        state_start_samples = int(state_start_frac * context_len)
        state_end_samples = int(state_end_frac * context_len)
        state_margin = int((state_end_samples - state_start_samples) * margin_frac)

        # Ensure needle fits within state with margins
        safe_start = state_start_samples + state_margin
        safe_end = state_end_samples - state_margin - len(needle.x)

        if safe_end <= safe_start:
            return self._create_invalid_sample("State too small for needle with margins")

        if position_mode == "center":
            # Insert at center of state
            position = (safe_start + safe_end) // 2
        elif position_mode == "near_boundary":
            # Insert near one of the boundaries (harder)
            if rng.random() < 0.5:
                position = safe_start + rng.integers(0, state_margin // 2 + 1)
            else:
                position = safe_end - rng.integers(0, state_margin // 2 + 1)
        else:  # "random"
            position = rng.integers(safe_start, safe_end + 1)

        # Step 7: Apply style transfer and insert needle
        local_stats = self._compute_local_stats(background, position)
        transferred_needle = self.style_transfer.transfer(needle, local_stats)

        x, y, z = self.style_transfer.insert_with_blending(
            (background.x, background.y, background.z),
            (transferred_needle.x, transferred_needle.y, transferred_needle.z),
            position,
        )

        # Step 8: Build metadata
        needle_metadata = [InsertedNeedle(
            activity=needle_activity,
            source_pid=needle.source_pid,
            source_start_ms=needle.start_ms,
            source_end_ms=needle.end_ms,
            insert_position_samples=position,
            insert_position_frac=position / context_len,
            duration_samples=len(needle.x),
            duration_ms=needle.duration_ms,
            timestamp_start=self._samples_to_timestamp(position, background),
            timestamp_end=self._samples_to_timestamp(position + len(needle.x), background),
        )]

        # Distance from nearest state boundary (difficulty indicator)
        dist_to_start = position - state_start_samples
        dist_to_end = state_end_samples - (position + len(needle.x))
        dist_to_boundary = min(dist_to_start, dist_to_end)

        return GeneratedSample(
            x=x, y=y, z=z,
            task_type=self.task_name,
            context_length_samples=context_len,
            background_pid=background.pid,
            recording_time_range=background.recording_time_context,
            question=f"What was the overall activity level when the {needle_activity} spike occurred?",
            answer=target_global_activity,
            answer_type=self.answer_type,
            needles=needle_metadata,
            difficulty_config={
                **asdict(difficulty),
                "needle_activity": needle_activity,
                "global_activity": target_global_activity,
                "global_timeline": global_timeline,
                "position_mode": position_mode,
                "dist_to_boundary_samples": dist_to_boundary,
            },
            is_valid=True,
        )

    def format_answer(self, sample: GeneratedSample) -> str:
        needle = sample.difficulty_config["needle_activity"]
        global_state = sample.answer
        return f"The overall activity level was {global_state} when the {needle} spike occurred."
```

### 5.7 Task 6: Temporal Antecedent

```python
class AntecedentTaskGenerator(BaseTaskGenerator):
    """
    Task 6: "What activity occurred immediately before {target_activity}?"

    Core Logic (Two-Needle Insertion):
    - Insert ANTECEDENT needle first, then TARGET needle adjacent to it
    - Background should be clean (low-activity) or NOT contain the antecedent
    - This prevents trivial answers (if background is homogeneous, answer = background)

    Design Choice: Two-needle insertion provides:
    - Control over antecedent-target pairing
    - Prevents learning from natural transition statistics
    - Clear adjacency relationship

    Example:
        Background:  [...sleep............................]
        + Antecedent: [...sleep...][SEDENTARY][...sleep...]
        + Target:     [...sleep...][SEDENTARY][WALKING][...sleep...]
        Q: "What occurred immediately before walking?"
        A: "Sedentary"

    Difficulty Knobs:
    - Gap between antecedent and target: smaller = easier (clear adjacency)
    - Background purity: mixed = harder
    - Use transition matrix vs uniform for pairing
    """

    task_name = "antecedent"
    answer_type = "category"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        # Step 1: Sample background window (prefer clean/low-activity)
        # Can modulate "noise" in background based on difficulty
        background_mode = difficulty.task_specific.get("background_mode", "low_activity")

        if background_mode == "low_activity":
            # Select from sleep or sedentary periods
            background = self.background_sampler.sample_background(
                context_length_samples=difficulty.context_length_samples,
                allowed_activities={"sleep", "sedentary"},
                purity="pure",
                rng=rng,
            )
        else:
            background = self.background_sampler.sample_background(
                context_length_samples=difficulty.context_length_samples,
                purity=difficulty.background_purity,
                rng=rng,
            )

        # Step 2: Sample antecedent activity A (NOT in background)
        all_activities = set(self.needle_sampler.get_available_activities())
        candidate_A = all_activities - background.activities_present

        if len(candidate_A) < 1:
            return self._create_invalid_sample("No candidate antecedent activities")

        antecedent_activity = rng.choice(list(candidate_A))

        # Step 3: Sample target activity T
        # Can use transition matrix P(T | A) for realistic pairs
        # Or uniform random for maximum diversity / leakage prevention
        candidate_T = candidate_A - {antecedent_activity}
        if len(candidate_T) < 1:
            return self._create_invalid_sample("No candidate target activities")

        if difficulty.task_specific.get("use_transition_probs", False):
            target_activity = self.transition_matrix.sample_successor(
                antecedent_activity, exclude={antecedent_activity}, rng=rng
            )
            if target_activity not in candidate_T:
                target_activity = rng.choice(list(candidate_T))
        else:
            target_activity = rng.choice(list(candidate_T))

        # Step 4: Sample antecedent needle
        antecedent_needle = self.needle_sampler.sample_needle(
            activity=antecedent_activity,
            min_duration_ms=difficulty.needle_length_range_ms[0],
            max_duration_ms=difficulty.needle_length_range_ms[1],
            exclude_pids=None,  # Same-participant sampling is allowed
            rng=rng,
        )

        # Step 5: Sample target needle
        target_needle = self.needle_sampler.sample_needle(
            activity=target_activity,
            min_duration_ms=difficulty.needle_length_range_ms[0],
            max_duration_ms=difficulty.needle_length_range_ms[1],
            exclude_pids=None,  # Same-participant sampling is allowed
            rng=rng,
        )

        if antecedent_needle is None or target_needle is None:
            return self._create_invalid_sample("Could not sample needles")

        # Step 6: Determine insertion positions with adjacency constraint
        # Antecedent must come BEFORE target with small gap
        gap_samples = difficulty.task_specific.get("adjacency_gap_samples", 10)  # Small gap
        total_length = len(antecedent_needle.x) + gap_samples + len(target_needle.x)

        context_len = difficulty.context_length_samples
        margin_samples = difficulty.task_specific.get("margin_samples", 100)

        if total_length + 2 * margin_samples >= context_len:
            return self._create_invalid_sample("Needles too long for context")

        # Sample antecedent position (ensuring space for target after)
        max_antecedent_pos = context_len - total_length - margin_samples
        antecedent_pos = rng.integers(margin_samples, max_antecedent_pos + 1)

        # Target position is immediately after antecedent + gap
        target_pos = antecedent_pos + len(antecedent_needle.x) + gap_samples

        # Step 7: Apply style transfer and insert both needles
        stats_A = self._compute_local_stats(background, antecedent_pos)
        stats_T = self._compute_local_stats(background, target_pos)

        transferred_A = self.style_transfer.transfer(antecedent_needle, stats_A)
        transferred_T = self.style_transfer.transfer(target_needle, stats_T)

        x, y, z = background.x.copy(), background.y.copy(), background.z.copy()

        # Insert antecedent first (comes earlier in time)
        x, y, z = self.style_transfer.insert_with_blending(
            (x, y, z),
            (transferred_A.x, transferred_A.y, transferred_A.z),
            antecedent_pos,
        )

        # Then insert target
        x, y, z = self.style_transfer.insert_with_blending(
            (x, y, z),
            (transferred_T.x, transferred_T.y, transferred_T.z),
            target_pos,
        )

        # Step 8: Build metadata
        needle_metadata = [
            InsertedNeedle(
                activity=antecedent_activity,
                source_pid=antecedent_needle.source_pid,
                source_start_ms=antecedent_needle.start_ms,
                source_end_ms=antecedent_needle.end_ms,
                insert_position_samples=antecedent_pos,
                insert_position_frac=antecedent_pos / context_len,
                duration_samples=len(antecedent_needle.x),
                duration_ms=antecedent_needle.duration_ms,
                timestamp_start=self._samples_to_timestamp(antecedent_pos, background),
                timestamp_end=self._samples_to_timestamp(antecedent_pos + len(antecedent_needle.x), background),
            ),
            InsertedNeedle(
                activity=target_activity,
                source_pid=target_needle.source_pid,
                source_start_ms=target_needle.start_ms,
                source_end_ms=target_needle.end_ms,
                insert_position_samples=target_pos,
                insert_position_frac=target_pos / context_len,
                duration_samples=len(target_needle.x),
                duration_ms=target_needle.duration_ms,
                timestamp_start=self._samples_to_timestamp(target_pos, background),
                timestamp_end=self._samples_to_timestamp(target_pos + len(target_needle.x), background),
            ),
        ]

        return GeneratedSample(
            x=x, y=y, z=z,
            task_type=self.task_name,
            context_length_samples=context_len,
            background_pid=background.pid,
            recording_time_range=background.recording_time_context,
            question=f"What activity occurred immediately before the {target_activity} bout?",
            answer=antecedent_activity,
            answer_type=self.answer_type,
            needles=needle_metadata,
            difficulty_config={
                **asdict(difficulty),
                "antecedent_activity": antecedent_activity,
                "target_activity": target_activity,
                "gap_samples": gap_samples,
                "background_activity": list(background.activities_present)[0] if len(background.activities_present) == 1 else "mixed",
            },
            is_valid=True,
        )

    def format_answer(self, sample: GeneratedSample) -> str:
        return sample.answer
```

### 5.8 Task 7: Comparison and Negation

```python
class ComparisonTaskGenerator(BaseTaskGenerator):
    """
    Task 7: "What was the longest/shortest period with(out) {activity_x}?"

    Core Logic:
    - Insert multiple bouts of target activity into background
    - Bouts should have DIFFERENT durations (no ties)
    - Answer is the time range of the extremum period
    - Supports both presence ("with") and absence ("without") queries

    Question Variants (2x2x2 = 8 types):
    | Dimension  | Options                                       |
    |------------|-----------------------------------------------|
    | Extremum   | Longest / Shortest                            |
    | Polarity   | With (presence) / Without (absence)           |
    | Framing    | Absolute time / Relative halves (future ext.) |

    Difficulty Knobs:
    - Number of periods: more = harder
    - Duration difference: smaller = harder (closer to ties)
    - Context length
    - Background complexity
    """

    task_name = "comparison"
    answer_type = "time_range"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        # Step 1: Sample question type
        extremum = rng.choice(["longest", "shortest"])
        polarity = rng.choice(["with", "without"])

        # Step 2: Sample background window with timestamp context
        # For negation queries, prefer backgrounds with at least 2 activity types
        # (prevents trivial solution of "no needles = entire window")
        if polarity == "without":
            min_bg_activities = 2
        else:
            min_bg_activities = 1

        background = self.background_sampler.sample_background(
            context_length_samples=difficulty.context_length_samples,
            purity="mixed" if min_bg_activities > 1 else difficulty.background_purity,
            min_activity_count=min_bg_activities,
            rng=rng,
        )

        context_len = difficulty.context_length_samples

        # Step 3: Sample target activity X (NOT in background)
        all_activities = set(self.needle_sampler.get_available_activities())
        candidate_X = all_activities - background.activities_present

        if not candidate_X:
            return self._create_invalid_sample("No candidate target activities")

        target_activity = rng.choice(list(candidate_X))

        # Step 4: Determine number of bouts to insert
        min_bouts = difficulty.task_specific.get("min_bouts", 2)
        max_bouts = difficulty.task_specific.get("max_bouts", 4)
        n_bouts = rng.integers(min_bouts, max_bouts + 1)

        # Step 5: Sample bout durations (must be different to avoid ties)
        min_duration_ms = difficulty.needle_length_range_ms[0]
        max_duration_ms = difficulty.needle_length_range_ms[1]

        # Ensure distinct durations
        min_diff_ms = difficulty.task_specific.get("min_duration_diff_ms", 2000)
        durations_ms = self._sample_distinct_durations(
            n=n_bouts,
            min_duration=min_duration_ms,
            max_duration=max_duration_ms,
            min_diff=min_diff_ms,
            rng=rng,
        )

        if durations_ms is None:
            return self._create_invalid_sample("Could not sample distinct durations")

        # Step 6: Insert bouts with non-overlapping positions
        min_gap_samples = difficulty.task_specific.get("min_gap_samples", 100)
        needles = []
        occupied_ranges = []

        for duration_ms in durations_ms:
            # Sample needle from bout index
            needle = self.needle_sampler.sample_needle(
                activity=target_activity,
                min_duration_ms=duration_ms,
                max_duration_ms=duration_ms + 5000,
                exclude_pids=None,  # Same-participant sampling is allowed
                rng=rng,
            )

            if needle is None:
                continue

            # Trim to exact duration
            target_samples = int(duration_ms * self.source_hz / 1000)
            needle = self._trim_needle(needle, target_samples)

            # Find valid position
            position = self._find_valid_position(
                context_length=context_len,
                needle_length=len(needle.x),
                occupied_ranges=occupied_ranges,
                min_gap=min_gap_samples,
                rng=rng,
            )

            if position is None:
                continue

            occupied_ranges.append((position, position + len(needle.x)))
            needles.append((needle, position, duration_ms))

        if len(needles) < 2:
            return self._create_invalid_sample("Need at least 2 bouts for comparison")

        # Step 7: Insert all needles
        x, y, z = background.x.copy(), background.y.copy(), background.z.copy()
        needle_metadata = []

        for needle, position, duration_ms in sorted(needles, key=lambda x: x[1]):
            local_stats = self._compute_local_stats(background, position)
            transferred = self.style_transfer.transfer(needle, local_stats)

            x, y, z = self.style_transfer.insert_with_blending(
                (x, y, z), (transferred.x, transferred.y, transferred.z), position
            )

            needle_metadata.append(InsertedNeedle(
                activity=target_activity,
                source_pid=needle.source_pid,
                source_start_ms=needle.start_ms,
                source_end_ms=needle.end_ms,
                insert_position_samples=position,
                insert_position_frac=position / context_len,
                duration_samples=len(needle.x),
                duration_ms=duration_ms,
                timestamp_start=self._samples_to_timestamp(position, background),
                timestamp_end=self._samples_to_timestamp(position + len(needle.x), background),
            ))

        # Step 8: Determine answer based on extremum and polarity
        if polarity == "with":
            # Find longest/shortest period WITH the activity
            periods = [(nm.timestamp_start, nm.timestamp_end, nm.duration_ms) for nm in needle_metadata]
        else:
            # Find longest/shortest period WITHOUT the activity
            # These are the gaps between needles (and before first / after last)
            periods = self._compute_gaps(needle_metadata, context_len, background)

        if not periods:
            return self._create_invalid_sample("No valid periods found")

        # Sort by duration
        periods_sorted = sorted(periods, key=lambda p: p[2], reverse=(extremum == "longest"))
        answer_period = periods_sorted[0]
        runner_up = periods_sorted[1] if len(periods_sorted) > 1 else None

        # Step 9: Generate question and answer
        question = f"What was the {extremum} period {polarity} {target_activity}?"
        answer = f"The {extremum} {target_activity} period was from {answer_period[0]} to {answer_period[1]}."

        return GeneratedSample(
            x=x, y=y, z=z,
            task_type=self.task_name,
            context_length_samples=context_len,
            background_pid=background.pid,
            recording_time_range=background.recording_time_context,
            question=question,
            answer=answer,
            answer_type=self.answer_type,
            needles=needle_metadata,
            difficulty_config={
                **asdict(difficulty),
                "target_activity": target_activity,
                "extremum": extremum,
                "polarity": polarity,
                "all_periods": periods,
                "answer_period": answer_period,
                "runner_up_period": runner_up,
                "duration_diff_to_runner_up": abs(answer_period[2] - runner_up[2]) if runner_up else None,
            },
            is_valid=True,
        )

    def _sample_distinct_durations(
        self,
        n: int,
        min_duration: int,
        max_duration: int,
        min_diff: int,
        rng: np.random.Generator,
    ) -> Optional[List[int]]:
        """Sample n distinct durations with minimum difference between them."""
        # Check if feasible
        range_needed = (n - 1) * min_diff
        if range_needed > (max_duration - min_duration):
            return None

        # Sample base durations and space them out
        base = rng.integers(min_duration, max_duration - range_needed + 1)
        durations = [base + i * min_diff + rng.integers(0, min_diff // 2 + 1) for i in range(n)]

        # Shuffle to randomize order
        rng.shuffle(durations)
        return durations

    def _compute_gaps(
        self,
        needles: List[InsertedNeedle],
        context_len: int,
        background: BackgroundSample,
    ) -> List[Tuple[str, str, int]]:
        """Compute gaps (periods without target activity)."""
        # Sort needles by position
        sorted_needles = sorted(needles, key=lambda n: n.insert_position_samples)

        gaps = []

        # Gap before first needle
        if sorted_needles[0].insert_position_samples > 0:
            gap_start = 0
            gap_end = sorted_needles[0].insert_position_samples
            gaps.append((
                self._samples_to_timestamp(gap_start, background),
                self._samples_to_timestamp(gap_end, background),
                gap_end - gap_start,
            ))

        # Gaps between needles
        for i in range(len(sorted_needles) - 1):
            gap_start = sorted_needles[i].insert_position_samples + sorted_needles[i].duration_samples
            gap_end = sorted_needles[i + 1].insert_position_samples
            if gap_end > gap_start:
                gaps.append((
                    self._samples_to_timestamp(gap_start, background),
                    self._samples_to_timestamp(gap_end, background),
                    gap_end - gap_start,
                ))

        # Gap after last needle
        last_end = sorted_needles[-1].insert_position_samples + sorted_needles[-1].duration_samples
        if last_end < context_len:
            gaps.append((
                self._samples_to_timestamp(last_end, background),
                self._samples_to_timestamp(context_len, background),
                context_len - last_end,
            ))

        return gaps
```

### 5.9 Task 8: Multi-Hop Localization

```python
class MultiHopTaskGenerator(BaseTaskGenerator):
    """
    Task 8: "When did the K-th {target_activity} bout occur after {anchor_activity}?"

    Core Logic (Multi-Step Reasoning):
    1. Insert an anchor bout at position t_A
    2. Insert K target bouts after the anchor: t_X1 < t_X2 < ... < t_XK
    3. Question asks for the K-th target after the anchor
    4. Model must: (a) find anchor, (b) count targets after it, (c) return K-th

    Optionally add distractors:
    - Target bouts BEFORE the anchor (tests "after" understanding)
    - Other activity bouts interspersed (tests activity discrimination)

    Difficulty Knobs:
    - K value: K=1 (easy) vs K=3 (hard)
    - Distractors before anchor
    - Gap between target bouts
    - Other activity distractors
    - Target bout duration
    """

    task_name = "multi_hop"
    answer_type = "timestamp"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        # Step 1: Sample background with timestamp context
        background = self.background_sampler.sample_background(
            context_length_samples=difficulty.context_length_samples,
            purity=difficulty.background_purity,
            rng=rng,
        )

        context_len = difficulty.context_length_samples

        # Step 2: Sample anchor activity A and target activity X
        all_activities = set(self.needle_sampler.get_available_activities())
        candidates = all_activities - background.activities_present

        if len(candidates) < 2:
            return self._create_invalid_sample("Need at least 2 non-background activities")

        activities = rng.choice(list(candidates), size=2, replace=False)
        anchor_activity, target_activity = activities[0], activities[1]

        # Step 3: Sample K (ordinal index)
        k_distribution = difficulty.task_specific.get("k_distribution", [0.2, 0.5, 0.3])  # P(K=1,2,3)
        K = rng.choice([1, 2, 3], p=k_distribution)

        # Step 4: Sample direction (before or after)
        direction = rng.choice(["before", "after"])

        # Step 5: Determine distractor configuration
        n_distractors_opposite = difficulty.task_specific.get("n_distractors_opposite", 0)
        n_other_distractors = difficulty.task_specific.get("n_other_distractors", 0)

        # Step 6: Sample anchor needle
        anchor_needle = self.needle_sampler.sample_needle(
            activity=anchor_activity,
            min_duration_ms=difficulty.needle_length_range_ms[0],
            max_duration_ms=difficulty.needle_length_range_ms[1],
            exclude_pids=None,  # Same-participant sampling is allowed
            rng=rng,
        )

        if anchor_needle is None:
            return self._create_invalid_sample("Could not sample anchor needle")

        # Step 7: Plan insertion layout
        # Layout: [margin] [optional before distractors] [anchor] [target bouts] [margin]
        min_gap = difficulty.task_specific.get("min_gap_samples", 50)
        margin = difficulty.task_specific.get("margin_samples", 100)

        # Calculate space needed
        target_needle_samples = int(difficulty.needle_length_range_ms[0] * self.source_hz / 1000)
        space_for_targets = K * (target_needle_samples + min_gap)
        space_for_before_distractors = n_distractors_opposite * (target_needle_samples + min_gap)

        total_min_space = (
            margin +
            space_for_before_distractors +
            len(anchor_needle.x) +
            min_gap +
            space_for_targets +
            margin
        )

        if total_min_space > context_len:
            return self._create_invalid_sample("Not enough space for all bouts")

        # Step 8: Sample anchor position
        if direction == "after":
            # Targets come after anchor, so anchor should be early enough
            max_anchor_pos = context_len - space_for_targets - len(anchor_needle.x) - margin
            min_anchor_pos = margin + space_for_before_distractors
        else:
            # Targets come before anchor, so anchor should be late enough
            min_anchor_pos = margin + space_for_targets
            max_anchor_pos = context_len - len(anchor_needle.x) - margin - space_for_before_distractors

        if max_anchor_pos <= min_anchor_pos:
            return self._create_invalid_sample("Invalid anchor position range")

        anchor_pos = rng.integers(min_anchor_pos, max_anchor_pos + 1)

        # Step 9: Sample target needle positions
        target_positions = []
        target_needles = []

        if direction == "after":
            # K targets after anchor
            current_pos = anchor_pos + len(anchor_needle.x) + min_gap
            for i in range(K):
                target_needle = self.needle_sampler.sample_needle(
                    activity=target_activity,
                    min_duration_ms=difficulty.needle_length_range_ms[0],
                    max_duration_ms=difficulty.needle_length_range_ms[1],
                    exclude_pids=None,  # Same-participant sampling is allowed
                    rng=rng,
                )
                if target_needle is None:
                    return self._create_invalid_sample(f"Could not sample target needle {i+1}")

                # Add some random gap variation
                extra_gap = rng.integers(0, min_gap * 2 + 1)
                pos = current_pos + extra_gap

                target_positions.append((pos, "after", i + 1))
                target_needles.append(target_needle)

                current_pos = pos + len(target_needle.x) + min_gap
        else:
            # K targets before anchor
            positions_reverse = []
            current_pos = anchor_pos - min_gap
            for i in range(K):
                target_needle = self.needle_sampler.sample_needle(
                    activity=target_activity,
                    min_duration_ms=difficulty.needle_length_range_ms[0],
                    max_duration_ms=difficulty.needle_length_range_ms[1],
                    exclude_pids=None,  # Same-participant sampling is allowed
                    rng=rng,
                )
                if target_needle is None:
                    return self._create_invalid_sample(f"Could not sample target needle {i+1}")

                extra_gap = rng.integers(0, min_gap * 2 + 1)
                pos = current_pos - len(target_needle.x) - extra_gap

                positions_reverse.append((pos, target_needle))
                current_pos = pos - min_gap

            # Reverse to get chronological order (earliest first)
            for idx, (pos, needle) in enumerate(reversed(positions_reverse)):
                target_positions.append((pos, "before", K - idx))
                target_needles.append(needle)

        # Step 10: Optionally add distractor targets on opposite side
        distractor_needles = []
        distractor_positions = []

        if n_distractors_opposite > 0:
            if direction == "after":
                # Add distractors BEFORE anchor
                current_pos = anchor_pos - min_gap
                for _ in range(n_distractors_opposite):
                    d_needle = self.needle_sampler.sample_needle(
                        activity=target_activity,
                        min_duration_ms=difficulty.needle_length_range_ms[0],
                        max_duration_ms=difficulty.needle_length_range_ms[1],
                        exclude_pids=None,  # Same-participant sampling is allowed
                        rng=rng,
                    )
                    if d_needle:
                        pos = current_pos - len(d_needle.x)
                        if pos >= margin:
                            distractor_positions.append((pos, "distractor_before"))
                            distractor_needles.append(d_needle)
                            current_pos = pos - min_gap
            else:
                # Add distractors AFTER anchor
                current_pos = anchor_pos + len(anchor_needle.x) + min_gap
                for _ in range(n_distractors_opposite):
                    d_needle = self.needle_sampler.sample_needle(
                        activity=target_activity,
                        min_duration_ms=difficulty.needle_length_range_ms[0],
                        max_duration_ms=difficulty.needle_length_range_ms[1],
                        exclude_pids=None,  # Same-participant sampling is allowed
                        rng=rng,
                    )
                    if d_needle:
                        pos = current_pos
                        if pos + len(d_needle.x) <= context_len - margin:
                            distractor_positions.append((pos, "distractor_after"))
                            distractor_needles.append(d_needle)
                            current_pos = pos + len(d_needle.x) + min_gap

        # Step 11: Insert all needles
        x, y, z = background.x.copy(), background.y.copy(), background.z.copy()
        all_insertions = []

        # Anchor
        anchor_stats = self._compute_local_stats(background, anchor_pos)
        transferred_anchor = self.style_transfer.transfer(anchor_needle, anchor_stats)
        all_insertions.append((transferred_anchor, anchor_pos, anchor_activity, "anchor", 0))

        # Target needles
        for (pos, side, ordinal), needle in zip(target_positions, target_needles):
            stats = self._compute_local_stats(background, pos)
            transferred = self.style_transfer.transfer(needle, stats)
            all_insertions.append((transferred, pos, target_activity, side, ordinal))

        # Distractor needles
        for (pos, label), needle in zip(distractor_positions, distractor_needles):
            stats = self._compute_local_stats(background, pos)
            transferred = self.style_transfer.transfer(needle, stats)
            all_insertions.append((transferred, pos, target_activity, label, 0))

        # Sort by position and insert
        all_insertions.sort(key=lambda x: x[1])
        needle_metadata = []

        for transferred, pos, activity, label, ordinal in all_insertions:
            x, y, z = self.style_transfer.insert_with_blending(
                (x, y, z), (transferred.x, transferred.y, transferred.z), pos
            )

            needle_metadata.append(InsertedNeedle(
                activity=activity,
                source_pid=transferred.source_pid,
                source_start_ms=transferred.start_ms,
                source_end_ms=transferred.end_ms,
                insert_position_samples=pos,
                insert_position_frac=pos / context_len,
                duration_samples=len(transferred.x),
                duration_ms=transferred.duration_ms,
                timestamp_start=self._samples_to_timestamp(pos, background),
                timestamp_end=self._samples_to_timestamp(pos + len(transferred.x), background),
            ))

        # Step 12: Determine answer (K-th target in specified direction)
        # Find the correct target
        correct_target_idx = None
        for i, (pos, side, ordinal) in enumerate(target_positions):
            if ordinal == K:
                correct_target_idx = i
                break

        if correct_target_idx is None:
            return self._create_invalid_sample("Could not find K-th target")

        # Find corresponding needle metadata (offset by 1 for anchor)
        answer_needle_idx = 1 + correct_target_idx  # +1 because anchor is first
        answer_needle = needle_metadata[answer_needle_idx]

        # Step 13: Generate question and answer
        ordinal_str = {1: "1st", 2: "2nd", 3: "3rd"}.get(K, f"{K}th")
        question = f"When did the {ordinal_str} {target_activity} bout occur {direction} the {anchor_activity}?"
        answer = f"The {ordinal_str} {target_activity} bout {direction} {anchor_activity} occurred from {answer_needle.timestamp_start} to {answer_needle.timestamp_end}."

        return GeneratedSample(
            x=x, y=y, z=z,
            task_type=self.task_name,
            context_length_samples=context_len,
            background_pid=background.pid,
            recording_time_range=background.recording_time_context,
            question=question,
            answer=answer,
            answer_type=self.answer_type,
            needles=needle_metadata,
            difficulty_config={
                **asdict(difficulty),
                "anchor_activity": anchor_activity,
                "target_activity": target_activity,
                "K": K,
                "direction": direction,
                "n_distractors_opposite": len(distractor_needles),
                "n_other_distractors": 0,  # Not implemented in this version
                "target_positions": [(p, s, o) for p, s, o in target_positions],
            },
            is_valid=True,
        )
```

---

## 6. Semi-Synthetic Bout Insertion

### 6.1 Style Transfer (`core/style_transfer.py`)

```python
class StyleTransfer:
    """
    Applies covariance projection (linear style transfer) to make
    needle signals blend naturally with the target context.

    Mathematical formulation:
        x_A_normalized = (x_A - μ_A) / σ_A
        x_transferred = L_B @ L_A^{-1} @ x_A_normalized
        x_final = x_transferred * σ_B + μ_B

    Where L is the Cholesky decomposition of the covariance matrix.
    """

    def __init__(
        self,
        blend_mode: str = "cosine",  # "linear", "cosine", "zero_crossing"
        blend_window_samples: int = 50,  # ~0.5s at 100Hz
    ):
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

        Returns:
            SignalStatistics with:
            - mean: np.ndarray shape (3,)
            - std: np.ndarray shape (3,)
            - cov: np.ndarray shape (3, 3)
            - cholesky: np.ndarray shape (3, 3)  # L where cov = L @ L.T
        """
        data = np.stack([x, y, z], axis=0)  # (3, n_samples)
        mean = np.mean(data, axis=1)
        std = np.std(data, axis=1)
        cov = np.cov(data)

        # Regularize covariance for numerical stability
        cov += np.eye(3) * 1e-6
        cholesky = np.linalg.cholesky(cov)

        return SignalStatistics(mean, std, cov, cholesky)

    def transfer(
        self,
        needle: NeedleSample,
        target_stats: SignalStatistics,
    ) -> NeedleSample:
        """
        Apply covariance projection to transform needle to target style.
        """
        # Compute needle statistics
        needle_stats = self.compute_statistics(needle.x, needle.y, needle.z)

        # Stack and normalize
        needle_data = np.stack([needle.x, needle.y, needle.z], axis=0)  # (3, n)
        normalized = (needle_data - needle_stats.mean[:, None]) / needle_stats.std[:, None]

        # Project through covariance (style transfer)
        # L_target @ L_needle^{-1} @ normalized
        transform = target_stats.cholesky @ np.linalg.inv(needle_stats.cholesky)
        projected = transform @ normalized

        # Denormalize with target statistics
        transferred = projected * target_stats.std[:, None] + target_stats.mean[:, None]

        return NeedleSample(
            source_pid=needle.source_pid,
            activity=needle.activity,
            start_ms=needle.start_ms,
            end_ms=needle.end_ms,
            duration_ms=needle.duration_ms,
            x=transferred[0],
            y=transferred[1],
            z=transferred[2],
        )

    def insert_with_blending(
        self,
        background: Tuple[np.ndarray, np.ndarray, np.ndarray],
        needle: Tuple[np.ndarray, np.ndarray, np.ndarray],
        position: int,  # Sample index for insertion start
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Insert needle into background with boundary blending.

        Args:
            background: (x, y, z) arrays
            needle: (x, y, z) arrays
            position: Insertion position in samples

        Returns:
            Modified (x, y, z) arrays with needle inserted
        """
        result = []
        for bg, nd in zip(background, needle):
            # Create output array
            out = bg.copy()

            # Insert needle with blending
            end_pos = position + len(nd)

            # Entry blend (background → needle)
            entry_start = max(0, position - self.blend_window_samples)
            entry_weights = self._get_blend_weights(position - entry_start)
            for i, weight in enumerate(entry_weights):
                idx = entry_start + i
                if idx < position:
                    out[idx] = (1 - weight) * bg[idx] + weight * nd[0]

            # Insert needle body
            out[position:end_pos] = nd

            # Exit blend (needle → background)
            exit_end = min(len(bg), end_pos + self.blend_window_samples)
            exit_weights = self._get_blend_weights(exit_end - end_pos)[::-1]
            for i, weight in enumerate(exit_weights):
                idx = end_pos + i
                if idx < len(bg):
                    out[idx] = weight * nd[-1] + (1 - weight) * bg[idx]

            result.append(out)

        return tuple(result)

    def _get_blend_weights(self, n_samples: int) -> np.ndarray:
        """Generate blend weights based on mode."""
        t = np.linspace(0, 1, n_samples)

        if self.blend_mode == "linear":
            return t
        elif self.blend_mode == "cosine":
            return (1 - np.cos(np.pi * t)) / 2
        elif self.blend_mode == "zero_crossing":
            # TODO: Implement phase-aligned blending
            return (1 - np.cos(np.pi * t)) / 2
        else:
            raise ValueError(f"Unknown blend mode: {self.blend_mode}")
```

### 6.2 Validation (`utils/validation.py`)

```python
class SampleValidator:
    """
    Validates generated samples to ensure quality.
    Includes adversarial validation to detect synthetic artifacts.
    """

    def __init__(
        self,
        boundary_discontinuity_threshold: float = 2.0,  # std devs
        adversarial_model: Optional[Any] = None,
    ):
        self.boundary_threshold = boundary_discontinuity_threshold
        self.adversarial_model = adversarial_model

    def validate_sample(
        self,
        sample: GeneratedSample,
    ) -> Tuple[bool, List[str]]:
        """
        Run all validation checks on a sample.

        Returns:
            (is_valid, list of validation issues)
        """
        issues = []

        # Check boundary discontinuities
        if sample.needles:
            for needle in sample.needles:
                disc = self._check_boundary_discontinuity(
                    sample.x, sample.y, sample.z,
                    needle.insert_position_samples,
                    needle.duration_samples,
                )
                if disc > self.boundary_threshold:
                    issues.append(f"Boundary discontinuity at needle {needle.activity}: {disc:.2f} std")

        # Check answer consistency
        answer_valid, answer_issue = self._validate_answer_consistency(sample)
        if not answer_valid:
            issues.append(answer_issue)

        # Adversarial validation (if model available)
        if self.adversarial_model:
            synth_prob = self._adversarial_check(sample)
            if synth_prob > 0.7:
                issues.append(f"High synthetic detection probability: {synth_prob:.2f}")

        return len(issues) == 0, issues

    def _check_boundary_discontinuity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        position: int,
        duration: int,
    ) -> float:
        """Check for discontinuities at insertion boundaries."""
        # Compute local statistics around insertion point
        window = 50  # samples

        # Before insertion
        before_start = max(0, position - window)
        before_data = np.stack([
            x[before_start:position],
            y[before_start:position],
            z[before_start:position],
        ])

        # After insertion
        end_pos = position + duration
        after_end = min(len(x), end_pos + window)
        after_data = np.stack([
            x[end_pos:after_end],
            y[end_pos:after_end],
            z[end_pos:after_end],
        ])

        # Compute first derivative discontinuity
        if position > 0 and end_pos < len(x):
            entry_jump = np.abs(np.stack([x, y, z])[:, position] - np.stack([x, y, z])[:, position-1])
            exit_jump = np.abs(np.stack([x, y, z])[:, end_pos] - np.stack([x, y, z])[:, end_pos-1])

            # Normalize by local std
            local_std = np.std(before_data, axis=1) + 1e-6
            entry_disc = np.mean(entry_jump / local_std)
            exit_disc = np.mean(exit_jump / local_std)

            return max(entry_disc, exit_disc)

        return 0.0
```

---

## 7. QADataset Integration

### 7.1 QADataset Loader (`dataset/ts_haystack_qa_loader.py`)

```python
def load_ts_haystack_splits(
    task: str,
    context_length: int,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load TS-Haystack task splits as HuggingFace Datasets.

    Args:
        task: Task name (existence, localization, etc.)
        context_length: Context length in samples (1000, 10000, etc.)

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    base_path = DATA_DIR / "ts_haystack" / "tasks" / task / f"{context_length}"

    splits = {}
    for split in ["train", "val", "test"]:
        path = base_path / split / "data.parquet"
        if path.exists():
            df = pl.read_parquet(path)
            splits[split] = Dataset.from_pandas(df.to_pandas())
        else:
            splits[split] = Dataset.from_dict({})

    return splits["train"], splits["val"], splits["test"]


def get_available_tasks() -> List[str]:
    """List all available tasks."""
    tasks_dir = DATA_DIR / "ts_haystack" / "tasks"
    if tasks_dir.exists():
        return [d.name for d in tasks_dir.iterdir() if d.is_dir()]
    return []


def get_available_context_lengths(task: str) -> List[int]:
    """List available context lengths for a task."""
    task_dir = DATA_DIR / "ts_haystack" / "tasks" / task
    if task_dir.exists():
        return sorted([int(d.name) for d in task_dir.iterdir() if d.is_dir()])
    return []
```

### 7.2 TSHaystackQADataset (`dataset/TSHaystackQADataset.py`)

```python
class TSHaystackQADataset(QADataset):
    """
    QADataset implementation for TS-Haystack benchmark.

    Supports all 8 task types with consistent interface.
    """

    # Class-level cache for configuration
    _cached_config: Optional[Tuple[str, int]] = None
    _cached_splits: Optional[Tuple[Dataset, Dataset, Dataset]] = None

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        task: str = "existence",
        context_length: int = 1000,
        format_sample_str: bool = False,
        time_series_format_function: Optional[Callable] = None,
    ):
        self.task = task
        self.context_length = context_length

        super().__init__(
            split=split,
            EOS_TOKEN=EOS_TOKEN,
            format_sample_str=format_sample_str,
            time_series_format_function=time_series_format_function,
        )

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load task splits from parquet files."""
        config = (self.task, self.context_length)

        if TSHaystackQADataset._cached_config != config:
            TSHaystackQADataset._cached_splits = load_ts_haystack_splits(
                task=self.task,
                context_length=self.context_length,
            )
            TSHaystackQADataset._cached_config = config

        return TSHaystackQADataset._cached_splits

    def _get_answer(self, row: Dict[str, Any]) -> str:
        """Extract answer from row."""
        return row["answer"]

    def _get_pre_prompt(self, row: Dict[str, Any]) -> str:
        """
        Generate pre-prompt with recording context.

        Example:
        "You are given accelerometer data from a wrist-worn sensor
         recorded between 6:00 AM and 8:00 AM."
        """
        time_start, time_end = row["recording_time_range"]

        return (
            f"You are given accelerometer data from a wrist-worn sensor "
            f"recorded between {time_start} and {time_end}. "
            f"The data includes measurements on X, Y, and Z axes at 100Hz sampling rate."
        )

    def _get_post_prompt(self, row: Dict[str, Any]) -> str:
        """
        Generate post-prompt with question and answer format guidance.
        """
        question = row["question"]
        answer_format = self._get_answer_format_guidance(row["answer_type"])

        return f"{question}\n\n{answer_format}"

    def _get_answer_format_guidance(self, answer_type: str) -> str:
        """Get format guidance based on answer type."""
        guidance = {
            "boolean": "Answer with 'Yes' or 'No'.",
            "timestamp": "Provide your answer as a time range (e.g., 'from 9:15 AM to 9:20 AM').",
            "integer": "Provide your answer as a number.",
            "category": "Choose from the available activity categories.",
            "time_range": "Provide your answer as a time range (e.g., 'from 9:15 AM to 11:45 AM').",
        }
        return guidance.get(answer_type, "")

    def _get_text_time_series_prompt_list(
        self,
        row: Dict[str, Any],
    ) -> List[TextTimeSeriesPrompt]:
        """Convert time series columns to prompt objects."""
        return [
            TextTimeSeriesPrompt(
                label_text="X-axis accelerometer data:",
                time_series=row["x_axis"],
            ),
            TextTimeSeriesPrompt(
                label_text="Y-axis accelerometer data:",
                time_series=row["y_axis"],
            ),
            TextTimeSeriesPrompt(
                label_text="Z-axis accelerometer data:",
                time_series=row["z_axis"],
            ),
        ]

    def _format_sample(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Preserve raw data in formatted sample."""
        sample = super()._format_sample(row)

        # Preserve metadata for analysis
        sample["task_type"] = row["task_type"]
        sample["answer_type"] = row["answer_type"]
        sample["context_length"] = row["context_length_samples"]
        sample["needle_metadata"] = row.get("needles", [])
        sample["difficulty_config"] = row.get("difficulty_config", {})

        return sample

    def get_task_info(self) -> Dict[str, Any]:
        """Get information about the current task configuration."""
        return {
            "task": self.task,
            "context_length": self.context_length,
            "answer_type": self._get_answer_type_for_task(self.task),
            "train_size": len(self._train_dataset),
            "val_size": len(self._val_dataset),
            "test_size": len(self._test_dataset),
        }

    def _get_answer_type_for_task(self, task: str) -> str:
        """Map task to answer type."""
        mapping = {
            "existence": "boolean",
            "localization": "timestamp",
            "counting": "integer",
            "ordering": "boolean",
            "state_query": "category",
            "antecedent": "category",
            "comparison": "time_range",
            "multi_hop": "timestamp",
        }
        return mapping.get(task, "unknown")
```

---

## 8. Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

**Goal:** Build timeline extraction, bout indexing, and seed management.

| Component | File | Priority | Dependencies |
|-----------|------|----------|--------------|
| Seed Manager | `core/seed_manager.py` | P0 | None |
| Data Structures | `core/data_structures.py` | P0 | None |
| Timeline Builder | `core/timeline_builder.py` | P0 | capture24_loader |
| Bout Indexer | `core/bout_indexer.py` | P0 | timeline_builder |
| Transition Matrix | `core/transition_matrix.py` | P1 | timeline_builder |
| Unit Tests | `test/test_core.py` | P0 | All core |

**Deliverables:**
- `SeedManager` with reproducibility guarantees
- `timelines/P*.parquet` for all 151 participants
- `bout_index.parquet` with cross-participant index
- `transition_matrix.json` with activity transition probabilities

### Phase 2: Sampling & Style Transfer (Week 2-3)

**Goal:** Build sampling infrastructure and style transfer.

| Component | File | Priority | Dependencies |
|-----------|------|----------|--------------|
| Background Sampler | `core/background_sampler.py` | P0 | bout_index |
| Needle Sampler | `core/needle_sampler.py` | P0 | bout_index |
| Style Transfer | `core/style_transfer.py` | P0 | None |
| Prompt Templates | `core/prompt_templates.py` | P0 | None |
| Sample Validator | `utils/validation.py` | P1 | style_transfer |
| Unit Tests | `test/test_sampling.py` | P0 | All sampling |

**Deliverables:**
- Working sampling pipeline
- Style transfer with boundary blending
- PromptTemplateBank with templates for all 8 tasks
- Validation metrics

### Phase 3: Task Generators (Week 3-5)

**Goal:** Implement all 8 task generators.

| Task | File | Priority | Difficulty |
|------|------|----------|------------|
| 1. Existence | `tasks/task_existence.py` | P0 | Easy |
| 2. Localization | `tasks/task_localization.py` | P0 | Medium |
| 3. Counting | `tasks/task_counting.py` | P1 | Medium |
| 4. Ordering | `tasks/task_ordering.py` | P1 | Medium |
| 5. State Query | `tasks/task_state_query.py` | P2 | Hard |
| 6. Antecedent | `tasks/task_antecedent.py` | P2 | Medium |
| 7. Comparison | `tasks/task_comparison.py` | P2 | Hard |
| 8. Multi-Hop | `tasks/task_multi_hop.py` | P2 | Hard |

**Deliverables:**
- Working generators for all 8 tasks
- Generated datasets at multiple context lengths

### Phase 4: QADataset Integration (Week 5-6)

**Goal:** Integrate with OpenTSLM training pipeline.

| Component | File | Priority | Dependencies |
|-----------|------|----------|--------------|
| QA Loader | `dataset/ts_haystack_qa_loader.py` | P0 | task generators |
| QADataset | `dataset/TSHaystackQADataset.py` | P0 | qa_loader |
| Metrics | `utils/metrics.py` | P1 | None |
| Integration Tests | `test/test_integration.py` | P0 | All |

**Deliverables:**
- TSHaystackQADataset compatible with CurriculumTrainer
- Task-specific evaluation metrics
- End-to-end integration tests

### Phase 5: Validation & Baselines (Week 6-7)

**Goal:** Validate benchmark quality and establish baselines.

| Task | Priority |
|------|----------|
| Adversarial validation (AUC < 0.55) | P0 |
| Short-context baseline (>80% accuracy @ 128 samples) | P0 |
| Scaling analysis (1K → 1M) | P1 |
| Ablation studies (difficulty knobs) | P2 |

**Success Criteria:**
1. All 8 tasks generate valid samples at all context lengths
2. Adversarial validation AUC < 0.55
3. Generation completes in < 4 hours on single GPU node
4. 128-sample context achieves > 80% accuracy

---

## 9. Reproducibility & Seed Strategy

### 9.1 Design Principles

Reproducibility is critical for benchmarking. Our seed strategy ensures:

| Requirement | Solution |
|-------------|----------|
| **Cross-task consistency** | Participant splits use a dedicated seed, shared across all tasks |
| **Per-sample reproducibility** | Each sample has a deterministic seed derived from (task, context, split, index) |
| **Parallel-safe generation** | Pre-computed sample seeds enable safe parallel execution |
| **Metadata tracking** | All seeds recorded in output metadata for full traceability |

### 9.2 Seed Hierarchy

```
master_seed (e.g., 42)
    │
    ├── participant_split_seed ──────► Train/Val/Test participant assignment
    │       (deterministic, shared across ALL tasks)
    │
    └── task_seed(task, context_length, split)
            │
            ├── sample_seed[0] ──► Background selection, needle selection, position
            ├── sample_seed[1]
            ├── sample_seed[2]
            └── ...
```

### 9.3 SeedManager Implementation (`core/seed_manager.py`)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import hashlib
import json
import numpy as np


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducible benchmark generation."""
    master_seed: int = 42

    # Derived seeds (computed automatically)
    participant_split_seed: int = field(init=False)

    def __post_init__(self):
        """Derive dependent seeds from master seed."""
        self.participant_split_seed = self._derive_seed("participant_split")

    def _derive_seed(self, component: str) -> int:
        """Derive a deterministic seed for a named component."""
        h = hashlib.sha256(f"{self.master_seed}:{component}".encode())
        return int.from_bytes(h.digest()[:4], "big")

    def to_dict(self) -> Dict:
        """Serialize for metadata storage."""
        return {
            "master_seed": self.master_seed,
            "participant_split_seed": self.participant_split_seed,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ReproducibilityConfig":
        """Deserialize from metadata."""
        return cls(master_seed=d["master_seed"])


class SeedManager:
    """
    Manages deterministic seed generation for reproducible task generation.

    Key features:
    - Hierarchical seed derivation from a single master seed
    - Consistent participant splits across all tasks
    - Per-sample seeds for parallel-safe generation
    - Full metadata tracking for reproducibility

    Example:
        >>> seed_mgr = SeedManager(master_seed=42)
        >>>
        >>> # Split participants (same for all tasks)
        >>> split_rng = seed_mgr.get_participant_split_rng()
        >>> train_pids, val_pids, test_pids = split_participants(all_pids, split_rng)
        >>>
        >>> # Generate samples for a specific task/context/split
        >>> sample_seeds = seed_mgr.get_sample_seeds("existence", 1000, "train", n_samples=10000)
        >>> for i, seed in enumerate(sample_seeds):
        ...     rng = np.random.default_rng(seed)
        ...     sample = generate_sample(rng, ...)
    """

    def __init__(self, master_seed: int = 42):
        self.config = ReproducibilityConfig(master_seed=master_seed)
        self._cache: Dict[str, np.random.Generator] = {}

    @property
    def master_seed(self) -> int:
        return self.config.master_seed

    def _derive_seed(self, *components: str) -> int:
        """
        Derive a deterministic seed from components.

        Uses SHA-256 hash of "master_seed:component1:component2:..." to get
        a deterministic integer seed.
        """
        key = ":".join(str(c) for c in [self.master_seed, *components])
        h = hashlib.sha256(key.encode())
        return int.from_bytes(h.digest()[:8], "big")

    def get_rng(self, *components: str) -> np.random.Generator:
        """
        Get a fresh RNG for the given components.

        Each unique combination of components produces a unique,
        deterministic RNG. Calling with the same components always
        returns an RNG initialized to the same state.

        Args:
            *components: Strings identifying the RNG purpose
                         e.g., ("existence", "1000", "train")

        Returns:
            A new numpy Generator initialized with the derived seed
        """
        seed = self._derive_seed(*components)
        return np.random.default_rng(seed)

    def get_participant_split_rng(self) -> np.random.Generator:
        """
        Get RNG for participant train/val/test splitting.

        This RNG is shared across ALL tasks to ensure consistent
        participant assignment regardless of which task is generated first.
        """
        return np.random.default_rng(self.config.participant_split_seed)

    def get_task_rng(
        self,
        task: str,
        context_length: int,
        split: str,
    ) -> np.random.Generator:
        """
        Get RNG for a specific (task, context_length, split) combination.

        Use this for sequential sample generation when not parallelizing.
        """
        return self.get_rng("task", task, str(context_length), split)

    def get_sample_seeds(
        self,
        task: str,
        context_length: int,
        split: str,
        n_samples: int,
    ) -> List[int]:
        """
        Pre-generate seeds for each sample in a task/context/split.

        This is the key method for parallel-safe generation. By pre-generating
        all sample seeds from a single RNG, we ensure:
        1. Reproducibility regardless of parallelization
        2. Each sample has a unique, deterministic seed
        3. The order of samples is consistent

        Args:
            task: Task name (e.g., "existence")
            context_length: Context length in samples (e.g., 1000)
            split: Data split ("train", "val", "test")
            n_samples: Number of samples to generate

        Returns:
            List of integer seeds, one per sample
        """
        rng = self.get_task_rng(task, context_length, split)
        # Generate seeds using 64-bit integers for maximum entropy
        return [int(rng.integers(0, 2**63)) for _ in range(n_samples)]

    def get_sample_rng(
        self,
        task: str,
        context_length: int,
        split: str,
        sample_index: int,
    ) -> np.random.Generator:
        """
        Get RNG for a specific sample by index.

        Useful for regenerating a single sample without generating all prior seeds.
        Note: This is less efficient than get_sample_seeds() for batch generation.
        """
        return self.get_rng("sample", task, str(context_length), split, str(sample_index))

    def get_metadata(self) -> Dict:
        """
        Get seed metadata for storage with generated datasets.

        This should be saved alongside generated data to enable reproduction.
        """
        return {
            "seed_config": self.config.to_dict(),
            "seed_manager_version": "1.0",
        }

    @classmethod
    def from_metadata(cls, metadata: Dict) -> "SeedManager":
        """Reconstruct SeedManager from saved metadata."""
        config = ReproducibilityConfig.from_dict(metadata["seed_config"])
        return cls(master_seed=config.master_seed)
```

### 9.4 Parallel Generation Pattern

```python
from joblib import Parallel, delayed

def generate_task_parallel(
    task: str,
    context_length: int,
    split: str,
    n_samples: int,
    n_jobs: int = 4,
    seed_manager: Optional[SeedManager] = None,
) -> List[GeneratedSample]:
    """
    Generate samples in parallel with guaranteed reproducibility.
    """
    seed_manager = seed_manager or SeedManager()

    # Pre-generate all sample seeds (fast, sequential)
    sample_seeds = seed_manager.get_sample_seeds(
        task, context_length, split, n_samples
    )

    # Generate samples in parallel (each worker gets its own seed)
    def generate_one(seed: int, index: int) -> GeneratedSample:
        rng = np.random.default_rng(seed)
        return generate_sample(rng, task, context_length, index)

    samples = Parallel(n_jobs=n_jobs)(
        delayed(generate_one)(seed, i)
        for i, seed in enumerate(sample_seeds)
    )

    return samples
```

### 9.5 Integration with Task Generators

Update `BaseTaskGenerator` to use `SeedManager`:

```python
class BaseTaskGenerator(ABC):
    """Base class with integrated seed management."""

    def __init__(
        self,
        background_sampler: BackgroundSampler,
        needle_sampler: NeedleSampler,
        style_transfer: StyleTransfer,
        config: TaskConfig,
        seed_manager: Optional[SeedManager] = None,
    ):
        self.background_sampler = background_sampler
        self.needle_sampler = needle_sampler
        self.style_transfer = style_transfer
        self.config = config
        self.seed_manager = seed_manager or SeedManager(config.seed)

    def generate_dataset(
        self,
        n_samples: int,
        difficulty: DifficultyConfig,
        split: str,
        n_jobs: int = 1,
    ) -> List[GeneratedSample]:
        """Generate samples with reproducibility guarantees."""

        # Pre-generate seeds for all samples
        sample_seeds = self.seed_manager.get_sample_seeds(
            task=self.task_name,
            context_length=difficulty.context_length_samples,
            split=split,
            n_samples=n_samples * 2,  # Extra for retries
        )

        if n_jobs == 1:
            return self._generate_sequential(sample_seeds, difficulty, n_samples)
        else:
            return self._generate_parallel(sample_seeds, difficulty, n_samples, n_jobs)

    def _generate_sequential(
        self,
        sample_seeds: List[int],
        difficulty: DifficultyConfig,
        n_samples: int,
    ) -> List[GeneratedSample]:
        """Sequential generation with retry support."""
        samples = []
        seed_iter = iter(sample_seeds)

        while len(samples) < n_samples:
            try:
                seed = next(seed_iter)
            except StopIteration:
                raise RuntimeError(f"Exhausted seeds, only generated {len(samples)}/{n_samples}")

            rng = np.random.default_rng(seed)
            sample = self.generate_sample(difficulty, rng)

            if sample.is_valid:
                samples.append(sample)

        return samples

    @abstractmethod
    def generate_sample(
        self,
        difficulty: DifficultyConfig,
        rng: np.random.Generator,
    ) -> GeneratedSample:
        """Generate a single sample using the provided RNG."""
        ...
```

### 9.6 Metadata Recording

Each generated dataset includes full seed information:

```python
def save_task_metadata(
    task: str,
    context_length: int,
    seed_manager: SeedManager,
    generation_stats: Dict,
    output_dir: Path,
) -> None:
    """Save metadata alongside generated data."""
    metadata = {
        # Seed information (for reproduction)
        **seed_manager.get_metadata(),

        # Task configuration
        "task": task,
        "context_length": context_length,

        # Generation statistics
        "generation_stats": generation_stats,

        # Timestamp
        "generated_at": datetime.utcnow().isoformat(),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
```

**Example metadata.json:**

```json
{
  "seed_config": {
    "master_seed": 42,
    "participant_split_seed": 3405691582
  },
  "seed_manager_version": "1.0",
  "task": "existence",
  "context_length": 10000,
  "generation_stats": {
    "train": {"requested": 10000, "generated": 10000, "attempts": 10234},
    "val": {"requested": 1000, "generated": 1000, "attempts": 1012},
    "test": {"requested": 1000, "generated": 1000, "attempts": 1008}
  },
  "generated_at": "2024-01-15T10:30:00Z"
}
```

### 9.7 Verification & Testing

```python
def test_seed_reproducibility():
    """Verify that seed management produces reproducible results."""

    # Test 1: Same seeds produce same samples
    mgr1 = SeedManager(master_seed=42)
    mgr2 = SeedManager(master_seed=42)

    seeds1 = mgr1.get_sample_seeds("existence", 1000, "train", 100)
    seeds2 = mgr2.get_sample_seeds("existence", 1000, "train", 100)
    assert seeds1 == seeds2, "Sample seeds should be deterministic"

    # Test 2: Different tasks get different seeds
    seeds_existence = mgr1.get_sample_seeds("existence", 1000, "train", 10)
    seeds_localization = mgr1.get_sample_seeds("localization", 1000, "train", 10)
    assert seeds_existence != seeds_localization, "Different tasks should have different seeds"

    # Test 3: Participant split is consistent
    split_rng1 = mgr1.get_participant_split_rng()
    split_rng2 = mgr2.get_participant_split_rng()
    vals1 = [split_rng1.random() for _ in range(10)]
    vals2 = [split_rng2.random() for _ in range(10)]
    assert vals1 == vals2, "Participant split RNG should be deterministic"

    # Test 4: Parallel vs sequential produces same results
    seeds = mgr1.get_sample_seeds("existence", 1000, "train", 10)

    # Sequential
    seq_results = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        seq_results.append(rng.random())

    # "Parallel" (simulated with different order)
    par_results = [None] * len(seeds)
    for i in [5, 2, 8, 0, 3, 7, 1, 9, 4, 6]:  # Random order
        rng = np.random.default_rng(seeds[i])
        par_results[i] = rng.random()

    assert seq_results == par_results, "Parallel execution should match sequential"


def test_participant_split_consistency():
    """Verify participant splits are identical across tasks."""
    mgr = SeedManager(master_seed=42)

    # Simulate splitting for different tasks
    def do_split(task: str):
        rng = mgr.get_participant_split_rng()
        pids = [f"P{i:03d}" for i in range(151)]
        rng.shuffle(pids)
        return pids[:100], pids[100:120], pids[120:]

    train1, val1, test1 = do_split("existence")
    train2, val2, test2 = do_split("localization")

    assert train1 == train2, "Train split should be identical across tasks"
    assert val1 == val2, "Val split should be identical across tasks"
    assert test1 == test2, "Test split should be identical across tasks"
```

### 9.8 CLI Integration

```bash
# All commands accept --seed for reproducibility
python -m opentslm.time_series_datasets.ts_haystack.generate_all \
    --seed 42 \
    --context-lengths 1000 10000 100000 \
    --n-jobs 8

# Regenerate a specific task with same seed
python -m opentslm.time_series_datasets.ts_haystack.tasks.task_existence \
    --seed 42 \
    --context-lengths 10000 \
    --samples-per-split 10000 1000 1000

# Verify reproducibility
python -m opentslm.time_series_datasets.ts_haystack.utils.verify_reproducibility \
    --seed 42 \
    --task existence \
    --context-length 10000 \
    --n-samples 100
```

---

## Appendix: CLI Commands

```bash
# Phase 1: Build timelines and index
python -m opentslm.time_series_datasets.ts_haystack.core.timeline_builder \
    --n-jobs 8 --label-scheme Walmsley2020

python -m opentslm.time_series_datasets.ts_haystack.core.bout_indexer \
    --min-bout-duration-ms 1000

python -m opentslm.time_series_datasets.ts_haystack.core.transition_matrix

# Phase 3: Generate task datasets
python -m opentslm.time_series_datasets.ts_haystack.tasks.task_existence \
    --context-lengths 1000 10000 100000 \
    --samples-per-split 10000 1000 1000 \
    --n-jobs 4

python -m opentslm.time_series_datasets.ts_haystack.tasks.task_localization \
    --context-lengths 1000 10000 100000 \
    --samples-per-split 10000 1000 1000 \
    --n-jobs 4

# Generate all tasks
python -m opentslm.time_series_datasets.ts_haystack.generate_all \
    --context-lengths 1000 10000 100000 1000000 \
    --n-jobs 8

# Validation
python -m opentslm.time_series_datasets.ts_haystack.utils.validation \
    --task existence --context-length 10000 --run-adversarial
```

---

## Appendix: Parquet Schemas

### Timeline Parquet Schema

```
participant_id: string
recording_start_ms: int64
recording_end_ms: int64
total_duration_ms: int64
bout_start_ms: list<int64>
bout_end_ms: list<int64>
bout_activity: list<string>
bout_duration_ms: list<int64>
```

### Bout Index Parquet Schema

```
activity: string
pid: string
start_ms: int64
end_ms: int64
duration_ms: int64
```

### Task Sample Parquet Schema

```
# Sensor data
x_axis: list<float32>
y_axis: list<float32>
z_axis: list<float32>

# Task metadata
task_type: string
context_length_samples: int32
answer_type: string

# Question/Answer
question: string
answer: string

# Recording context
background_pid: string
recording_time_start: string
recording_time_end: string

# Needle metadata (JSON string)
needles: string  # JSON array of InsertedNeedle

# Difficulty config (JSON string)
difficulty_config: string  # JSON object
```
