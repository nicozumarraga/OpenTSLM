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
│  │  (per-participant)  │    │   (cross-participant)│    │ (global stats)   ││
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
    │   ├── timeline_builder.py         # Activity timeline extraction
    │   ├── bout_indexer.py             # Cross-participant bout index
    │   ├── transition_matrix.py        # Activity transition probabilities
    │   ├── background_sampler.py       # Sample background windows
    │   ├── needle_sampler.py           # Sample needles from bout index
    │   └── style_transfer.py           # Covariance projection + blending
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
            exclude_pids: Participants to exclude (e.g., background participant)
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
    1. Sample pure background (single activity, different from target)
    2. Sample needle from bout index
    3. Apply style transfer
    4. Insert at controlled position
    5. Ground answer in real timestamps
    """

    task_name = "localization"
    answer_type = "timestamp"

    def generate_sample(
        self,
        difficulty: DifficultyConfig,
    ) -> GeneratedSample:
        # Sample background WITHOUT target activity
        background = self.background_sampler.sample_background(
            context_length_samples=difficulty.context_length_samples,
            purity="pure",
            excluded_activities={difficulty.task_specific["target_activity"]},
        )

        # Sample needle
        needle = self.needle_sampler.sample_needle(
            activity=difficulty.task_specific["target_activity"],
            min_duration_ms=difficulty.needle_length_range_ms[0],
            max_duration_ms=difficulty.needle_length_range_ms[1],
            exclude_pids={background.pid},
        )

        # Apply style transfer
        transferred_needle = self.style_transfer.transfer(
            needle=needle,
            target_statistics=self._compute_local_stats(background, insert_position),
        )

        # Insert needle
        x, y, z = self._insert_needle(
            background=(background.x, background.y, background.z),
            needle=(transferred_needle.x, transferred_needle.y, transferred_needle.z),
            position=self._sample_position(difficulty.needle_position),
        )

        return GeneratedSample(...)

    def format_question(self, sample: GeneratedSample) -> str:
        activity = sample.needles[0].activity
        return f"When did the {activity} bout occur?"

    def format_answer(self, sample: GeneratedSample) -> str:
        needle = sample.needles[0]
        return f"The {needle.activity} bout occurred from {needle.timestamp_start} to {needle.timestamp_end}."
```

### 5.4 Remaining Tasks (Structure Only)

```python
class CountingTaskGenerator(BaseTaskGenerator):
    """Task 3: How many {activity_x} bouts occurred?"""
    task_name = "counting"
    answer_type = "integer"

class OrderingTaskGenerator(BaseTaskGenerator):
    """Task 4: Did {activity_x} occur before {activity_y}?"""
    task_name = "ordering"
    answer_type = "boolean"  # or "category" for "Which occurred first?"

class StateQueryTaskGenerator(BaseTaskGenerator):
    """Task 5: What was the activity level when the {event} occurred?"""
    task_name = "state_query"
    answer_type = "category"

class AntecedentTaskGenerator(BaseTaskGenerator):
    """Task 6: What activity preceded {activity_x}?"""
    task_name = "antecedent"
    answer_type = "category"

class ComparisonTaskGenerator(BaseTaskGenerator):
    """Task 7: What was the longest/shortest period with(out) {activity_x}?"""
    task_name = "comparison"
    answer_type = "time_range"

class MultiHopTaskGenerator(BaseTaskGenerator):
    """Task 8: When did the K-th {activity_x} occur before/after {anchor}?"""
    task_name = "multi_hop"
    answer_type = "timestamp"
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

**Goal:** Build timeline extraction and bout indexing.

| Component | File | Priority | Dependencies |
|-----------|------|----------|--------------|
| Timeline Builder | `core/timeline_builder.py` | P0 | capture24_loader |
| Data Structures | `core/data_structures.py` | P0 | None |
| Bout Indexer | `core/bout_indexer.py` | P0 | timeline_builder |
| Transition Matrix | `core/transition_matrix.py` | P1 | timeline_builder |
| Unit Tests | `test/test_core.py` | P0 | All core |

**Deliverables:**
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
| Sample Validator | `utils/validation.py` | P1 | style_transfer |
| Unit Tests | `test/test_sampling.py` | P0 | All sampling |

**Deliverables:**
- Working sampling pipeline
- Style transfer with boundary blending
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