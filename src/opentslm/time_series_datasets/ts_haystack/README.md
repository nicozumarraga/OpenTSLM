# TS-Haystack

A semi-synthetic benchmark for testing retrieval and reasoning over long time series (1K–1M+ datapoints) using Capture-24 accelerometer data.

See `TS_HAYSTACK_IMPLEMENTATION_PLAN.md` in project root for full design.

## Structure

```
ts_haystack/
├── core/
│   ├── data_structures.py     # Dataclasses (BoutRecord, NeedleSample, BackgroundSample, etc.)
│   ├── seed_manager.py        # Reproducibility & deterministic seeds
│   ├── timeline_builder.py    # Extract activity bouts from Capture24
│   ├── bout_indexer.py        # Cross-participant bout index
│   ├── transition_matrix.py   # Activity transition probabilities
│   ├── background_sampler.py  # Sample background windows
│   ├── needle_sampler.py      # Sample needles from bout index
│   ├── style_transfer.py      # Covariance projection + blending
│   └── prompt_templates.py    # NL templates for Q/A diversity
├── tasks/
│   ├── base_task.py           # Abstract base class for all tasks
│   ├── task_existence.py      # Task 1: Existence detection
│   ├── task_localization.py   # Task 2: Temporal localization
│   ├── task_counting.py       # Task 3: Bout counting
│   ├── task_ordering.py       # Task 4: Temporal ordering
│   ├── task_state_query.py    # Task 5: Cross-scale state query
│   ├── task_antecedent.py     # Task 6: Temporal antecedent
│   ├── task_comparison.py     # Task 7: Comparison & negation
│   └── task_multi_hop.py      # Task 8: Multi-hop localization
├── utils/
│   ├── timestamp_utils.py     # Timestamp conversion utilities
│   └── position_utils.py      # Position sampling utilities
├── scripts/
│   └── build_core_artifacts.py  # CLI to build timelines, index, matrix
└── test/
    └── test_imports.py        # Verify module imports
```

## Status

- [x] Phase 1: Core infrastructure (timelines, bout index, transition matrix)
- [x] Phase 2: Sampling & style transfer (background/needle samplers, style transfer, prompts)
- [x] Phase 3: Task generators (all 8 tasks implemented)
- [ ] Phase 4: QADataset integration

## Task Overview

| Task | Name | Question Type | Answer Type | Description |
|------|------|---------------|-------------|-------------|
| 1 | Existence | "Is there {activity} in this recording?" | boolean | Detect presence/absence of an activity |
| 2 | Localization | "When did the {activity} bout occur?" | time_range | Find temporal location of an activity |
| 3 | Counting | "How many {activity} bouts occurred?" | integer | Count occurrences of an activity |
| 4 | Ordering | "Did {activity_a} occur before {activity_b}?" | boolean/category | Determine temporal order of two activities |
| 5 | State Query | "What was the activity level when {event} occurred?" | category | Cross-scale integration (local event + global state) |
| 6 | Antecedent | "What activity occurred before {target}?" | category | Identify preceding activity |
| 7 | Comparison | "What was the longest/shortest period with/without {activity}?" | time_range | Find extremum periods |
| 8 | Multi-Hop | "When did the Kth {target} occur after {anchor}?" | time_range | Multi-step reasoning with anchor reference |

## Quick Start

### 1. Build Core Artifacts

Requires Capture24 data extracted (see `capture24/README.md`).

```bash
python -m opentslm.time_series_datasets.ts_haystack.scripts.build_core_artifacts \
    --n-jobs 8 \
    --label-scheme WillettsSpecific2018
```

### 2. Generate Task Datasets

Each task can be run as a standalone script:

```bash
# Generate existence task samples
python -m opentslm.time_series_datasets.ts_haystack.tasks.task_existence \
    --context-lengths 10000 50000 \
    --samples-per-split 1000 100 100 \
    --seed 42 \
    --n-jobs 4

# Generate multi-hop task samples
python -m opentslm.time_series_datasets.ts_haystack.tasks.task_multi_hop \
    --context-lengths 10000 50000 \
    --samples-per-split 1000 100 100 \
    --seed 42 \
    --n-jobs 4 \
    --direction-mode random \
    --n-distractors 1
```

## Output

Artifacts are saved to `data/capture24/ts_haystack/`:

```
data/capture24/ts_haystack/
├── timelines/P*.parquet       # Per-participant activity timelines
├── bout_index.parquet         # Cross-participant bout index
├── transition_matrix.json     # Activity transition probabilities
└── tasks/
    ├── existence/
    │   ├── 10000/
    │   │   ├── train/data.parquet
    │   │   ├── val/data.parquet
    │   │   └── test/data.parquet
    │   └── metadata.json
    ├── localization/
    ├── counting/
    ├── ordering/
    ├── state_query/
    ├── antecedent/
    ├── comparison/
    └── multi_hop/
```

## Programmatic Usage

### Using Task Generators

```python
from opentslm.time_series_datasets.ts_haystack.tasks import (
    ExistenceTaskGenerator,
    LocalizationTaskGenerator,
    CountingTaskGenerator,
    OrderingTaskGenerator,
    StateQueryTaskGenerator,
    AntecedentTaskGenerator,
    ComparisonTaskGenerator,
    MultiHopTaskGenerator,
    TASK_REGISTRY,
    list_available_tasks,
)
from opentslm.time_series_datasets.ts_haystack.core import DifficultyConfig

# List available tasks
print(list_available_tasks())
# ['antecedent', 'comparison', 'counting', 'existence', 'localization',
#  'multi_hop', 'ordering', 'state_query']

# Create a task generator with loaded artifacts
generator = MultiHopTaskGenerator.create_with_artifacts(seed=42)

# Configure difficulty
difficulty = DifficultyConfig(
    context_length_samples=10000,
    needle_position="random",
    needle_length_ratio_range=(0.03, 0.30),  # 3-30% of context (300-3000 samples)
    background_purity="pure",
    task_specific={
        "k_distribution": [0.4, 0.4, 0.2],  # P(K=1,2,3)
        "direction_mode": "random",
        "n_distractors_opposite": 0,
        "min_gap_samples": 100,
    },
)

# Generate samples for a split
samples = generator.generate_dataset(
    n_samples=100,
    difficulty=difficulty,
    split="train",
    n_jobs=4,
)

# Save to parquet
output_path = generator.save_dataset(
    samples=samples,
    split="train",
    context_length=10000,
)
print(f"Saved to: {output_path}")
```

### Using the Task Registry

```python
from opentslm.time_series_datasets.ts_haystack.tasks import get_task_generator

# Get generator class by name
TaskClass = get_task_generator("multi_hop")
generator = TaskClass.create_with_artifacts(seed=42)
```

### Low-Level Component Usage

```python
from opentslm.time_series_datasets.ts_haystack.core import (
    TimelineBuilder,
    BoutIndexer,
    TransitionMatrix,
    BackgroundSampler,
    NeedleSampler,
    StyleTransfer,
    PromptTemplateBank,
    SeedManager,
)

# Load Phase 1 artifacts
timelines = TimelineBuilder.load_all_timelines()
bout_index = BoutIndexer.load_index()
transition_matrix = TransitionMatrix.load()

# Initialize Phase 2 components
seed_manager = SeedManager(master_seed=42)
background_sampler = BackgroundSampler(timelines, bout_index)
needle_sampler = NeedleSampler(bout_index, transition_matrix)
style_transfer = StyleTransfer(blend_mode="cosine")
template_bank = PromptTemplateBank()

# Sample a pure background
rng = seed_manager.get_sample_rng("existence", 10000, "train", sample_index=0)
background = background_sampler.sample_background(
    context_length_samples=10000,
    purity="pure",
    rng=rng,
)

# Sample a needle (different activity than background)
needle = needle_sampler.sample_needle_for_context(
    context_activities=background.activities_present,
    min_duration_ms=5000,
    rng=rng,
)

# Trim needle to desired length
needle = needle.trim(n_samples=500)

# Apply style transfer
local_stats = style_transfer.compute_local_statistics(
    (background.x, background.y, background.z),
    position=5000,
)
transferred_needle = style_transfer.transfer(needle, local_stats)

# Insert with blending
x, y, z = style_transfer.insert_with_blending(
    (background.x, background.y, background.z),
    (transferred_needle.x, transferred_needle.y, transferred_needle.z),
    position=5000,
)

# Generate Q/A using templates
question, answer = template_bank.sample(
    "existence",
    rng,
    activity=needle.activity,
    exists=True,
)
```

## Task-Specific Configuration

Each task supports `task_specific` parameters in `DifficultyConfig`:

### Existence
- No additional parameters required

### Localization
- `min_bg_activities`: Minimum activities in background (default: 1)

### Counting
- `min_bouts`: Minimum bouts to insert (default: 1)
- `max_bouts`: Maximum bouts to insert (default: 5)
- `min_gap_samples`: Gap between bouts (default: 100)

### Ordering
- `min_gap_samples`: Gap between activities (default: 100)
- `question_format`: "boolean" or "category" (default: "boolean")

### State Query
- `min_global_states`: Minimum activity states in background (default: 2)
- `max_global_states`: Maximum activity states (default: 5)
- `position_mode`: "center", "near_boundary", or "random" (default: "random")

### Antecedent
- `adjacency_gap_samples`: Gap between antecedent and target (default: 10)
- `background_mode`: "low_activity" or "mixed" (default: "low_activity")
- `use_transition_probs`: Use transition matrix for pairing (default: False)

### Comparison
- `min_bouts`: Minimum bouts to insert (default: 2)
- `max_bouts`: Maximum bouts (default: 4)
- `min_duration_diff_ms`: Minimum duration difference to avoid ties (default: 2000)

### Multi-Hop
- `k_distribution`: Probability distribution for K values [P(K=1), P(K=2), P(K=3)]
- `direction_mode`: "random", "after_only", or "before_only"
- `n_distractors_opposite`: Distractor targets on opposite side of anchor
- `min_gap_samples`: Gap between bouts (default: 100)

## Test Suite

  Comprehensive tests validate all task generators, with optional plot generation for visual inspection.

  ```bash
  # Run all tests
  pytest src/opentslm/time_series_datasets/ts_haystack/test/ -v

  # Run only task tests (no plots)
  pytest src/opentslm/time_series_datasets/ts_haystack/test/tasks/ -v -k "not visualize"

  # Generate sample plots only
  pytest src/opentslm/time_series_datasets/ts_haystack/test/tasks/ -v -k "visualize"

  Plots are saved to test/plots/<task_name>/.

  Note: Tests require Phase 1 artifacts to be built first.
