# TS-Haystack Dataset Generator, CoT Pipeline & QADataset Implementation Plan

## Overview

This plan describes the implementation of three interconnected components for the TS-Haystack benchmark:

1. **Centralized Dataset Generator** - Generate task samples at configurable context lengths
2. **CoT Rationale Generator** - Generate LLM-based chain-of-thought rationales using rich sample metadata
3. **TSHaystackCoTQADataset** - QADataset implementation for training OpenTSLM models

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXISTING INFRASTRUCTURE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Phase 1: Timelines, BoutIndex, TransitionMatrix                            │
│  Phase 2: BackgroundSampler, NeedleSampler, StyleTransfer                   │
│  Phase 3: 8 Task Generators (existence, localization, counting, etc.)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NEW COMPONENTS (This Plan)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Component 1: Dataset Generator Script                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  scripts/generate_ts_haystack_dataset.py                                ││
│  │    - CLI for centralized dataset generation                             ││
│  │    - Configurable: tasks, context lengths, samples per split            ││
│  │    - Outputs parquet files with full metadata                           ││
│  │    - Supports parallel generation and resumption                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│           │                                                                 │
│           ▼                                                                 │
│  data/capture24/ts_haystack/tasks/{context_len}/{task}/{split}/data.parquet │
│                                                                             │
│  Component 2: CoT Rationale Generator (LLM-based)                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  scripts/generate_ts_haystack_cot.py                                    ││
│  │    - Reads parquet files with metadata                                  ││
│  │    - Generates plots + passes rich metadata to LLM                      ││
│  │    - LLM produces fluid, grounded reasoning                             ││
│  │    - Outputs enriched parquet with 'rationale' column                   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│           │                                                                 │
│           ▼                                                                 │
│  data/capture24/ts_haystack/cot/{context_len}/{task}/{split}/data.parquet   │
│                                                                             │
│  Component 3: TSHaystackCoTQADataset                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  dataset/TSHaystackCoTQADataset.py                                      ││
│  │    - Extends QADataset base class                                       ││
│  │    - Configurable task selection and context length                     ││
│  │    - Multi-task training support (mixed task batches)                   ││
│  │    - Loads CoT-enriched parquet files                                   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Dataset Generator Script

### Location
`src/opentslm/time_series_datasets/ts_haystack/scripts/generate_ts_haystack_dataset.py`

### Purpose
Centralized CLI for generating TS-Haystack datasets across all tasks and context lengths.

### Design

```python
"""
Centralized dataset generator for TS-Haystack benchmark.

Usage:
    # Generate all tasks at multiple context lengths
    python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_dataset \
        --tasks all \
        --context-lengths 1000 10000 100000 \
        --train-samples 10000 \
        --val-samples 1000 \
        --test-samples 1000 \
        --n-jobs 8 \
        --seed 42

    # Generate specific tasks
    python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_dataset \
        --tasks existence localization counting \
        --context-lengths 10000 \
        --train-samples 5000 \
        --seed 42
"""

@dataclass
class GenerationConfig:
    """Configuration for dataset generation."""
    tasks: List[str]                          # Task names or ["all"]
    context_lengths: List[int]                # e.g., [1000, 10000, 100000]
    samples_per_split: Dict[str, int]         # {"train": 10000, "val": 1000, "test": 1000}
    seed: int = 42
    n_jobs: int = 1
    output_dir: Optional[Path] = None
    overwrite: bool = False

    # Task-specific difficulty overrides
    difficulty_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

def generate_dataset(config: GenerationConfig) -> Dict[str, Path]:
    """
    Generate datasets for all specified tasks and context lengths.

    Returns:
        Dictionary mapping (context_len, task_name) -> output_path
    """
    ...
```

### YAML Configuration File

The dataset generator uses a YAML config file that provides full visibility and control over all generation parameters. Each task has its own section with task-specific defaults extracted from the actual implementation.

#### Default Config Location
`src/opentslm/time_series_datasets/ts_haystack/configs/default_generation_config.yaml`

#### Example Config File

```yaml
# TS-Haystack Dataset Generation Configuration
# =============================================
# All values shown are the actual defaults from the implementation.
# Modify as needed for your experiment.

# ------------------------------------------------------------------------------
# Global Settings
# ------------------------------------------------------------------------------
global:
  seed: 42                              # Master seed for reproducibility
  n_jobs: 4                             # Parallel workers for generation
  output_dir: data/capture24/ts_haystack/tasks
  overwrite: false                      # Overwrite existing files

# ------------------------------------------------------------------------------
# Context Lengths (in samples at 100Hz)
# ------------------------------------------------------------------------------
# 1000 = 10s, 10000 = 100s (~1.7min), 100000 = 1000s (~17min)
context_lengths:
  - 10000

# ------------------------------------------------------------------------------
# Samples per Split
# ------------------------------------------------------------------------------
samples:
  train: 10000
  val: 1000
  test: 1000

# ------------------------------------------------------------------------------
# Style Transfer Settings (applied during needle insertion)
# ------------------------------------------------------------------------------
style_transfer:
  transfer_mode: mean_only              # "mean_only" (recommended) or "full"
  blend_mode: cosine                    # "cosine" or "linear"
  blend_window_samples: 50              # Samples for boundary blending (~0.5s)

# ------------------------------------------------------------------------------
# Task Configurations
# ------------------------------------------------------------------------------
# Each task has its own difficulty settings.
# Set enabled: false to skip a task.

tasks:

  # --------------------------------------------------------------------------
  # Task 1: Existence
  # "Is there {activity} in this recording?"
  # --------------------------------------------------------------------------
  existence:
    enabled: true
    needle_position: random             # "random", "beginning", "middle", "end"
    needle_length_ratio_range: [0.02, 0.10]  # 2-10% of context  # 5s to 5min
    background_purity: pure             # "pure" or "mixed"
    # task_specific
    margin_samples: 100                 # Position margin from edges

  # --------------------------------------------------------------------------
  # Task 2: Localization
  # "When did the {activity} bout occur?"
  # --------------------------------------------------------------------------
  localization:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.02, 0.10]  # 2-10% of context
    background_purity: pure
    # task_specific
    use_transition_probs: false         # Use transition matrix for needle selection
    margin_samples: 100

  # --------------------------------------------------------------------------
  # Task 3: Counting
  # "How many {activity} bouts occurred?"
  # --------------------------------------------------------------------------
  counting:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.02, 0.10]  # 2-10% of context
    background_purity: pure
    # task_specific
    min_bouts: 1                        # Minimum bouts to insert
    max_bouts: 5                        # Maximum bouts to insert
    min_gap_samples: 100                # Minimum gap between bouts
    margin_samples: 100

  # --------------------------------------------------------------------------
  # Task 4: Ordering
  # "Did {activity_a} occur before {activity_b}?"
  # --------------------------------------------------------------------------
  ordering:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.02, 0.10]  # 2-10% of context
    background_purity: pure
    # task_specific
    min_gap_samples: 100                # Gap between the two activities
    margin_samples: 100
    question_format: boolean            # "boolean" or "category"

  # --------------------------------------------------------------------------
  # Task 5: State Query (Cross-Scale)
  # "What was the activity level when {event} occurred?"
  # NOTE: background_purity is HARDCODED to "mixed" for this task
  # --------------------------------------------------------------------------
  state_query:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.02, 0.10]  # 2-10% of context
    # background_purity: mixed          # HARDCODED - cannot be changed
    # task_specific
    min_global_states: 2                # Minimum activity states in background
    max_global_states: 5                # Maximum activity states
    min_state_duration_samples: 500     # Minimum state size for needle insertion
    position_mode: random               # "random", "center", "near_boundary"
    boundary_margin_frac: 0.1           # Margin from state boundaries (fraction)

  # --------------------------------------------------------------------------
  # Task 6: Antecedent
  # "What activity occurred before {target}?"
  # NOTE: background_purity depends on background_mode setting
  # --------------------------------------------------------------------------
  antecedent:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.02, 0.10]  # 2-10% of context
    background_purity: pure             # Used only if background_mode != "low_activity"
    # task_specific
    background_mode: low_activity       # "low_activity" (forces pure) or "mixed"
    adjacency_gap_samples: 10           # Gap between antecedent and target
    margin_samples: 100
    use_transition_probs: false         # Use transition matrix for activity pairing

  # --------------------------------------------------------------------------
  # Task 7: Comparison
  # "What was the longest/shortest period with/without {activity}?"
  # --------------------------------------------------------------------------
  comparison:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.02, 0.10]  # 2-10% of context
    background_purity: pure
    # task_specific
    min_bouts: 2                        # Minimum bouts to compare
    max_bouts: 4                        # Maximum bouts
    min_duration_diff_ms: 2000          # Minimum duration difference (avoid ties)
    min_gap_samples: 100
    margin_samples: 100

  # --------------------------------------------------------------------------
  # Task 8: Multi-Hop
  # "When did the Kth {target} occur after {anchor}?"
  # --------------------------------------------------------------------------
  multi_hop:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.02, 0.10]  # 2-10% of context
    background_purity: pure
    # task_specific
    k_distribution: [0.4, 0.4, 0.2]     # P(K=1), P(K=2), P(K=3)
    direction_mode: random              # "random", "after_only", "before_only"
    n_distractors_opposite: 0           # Distractor targets on opposite side
    min_gap_samples: 100
    margin_samples: 100
```

#### Config Loading

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any
import yaml

@dataclass
class TaskDifficultyConfig:
    """Difficulty config for a single task."""
    enabled: bool = True
    needle_position: str = "random"
    needle_length_ratio_range: tuple = (0.02, 0.10)
    background_purity: str = "pure"
    task_specific: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GenerationConfig:
    """Configuration for dataset generation, loaded from YAML."""

    # Global settings
    seed: int = 42
    n_jobs: int = 4
    output_dir: Path = Path("data/capture24/ts_haystack/tasks")
    overwrite: bool = False

    # What to generate
    context_lengths: List[int] = field(default_factory=lambda: [10000])
    samples_per_split: Dict[str, int] = field(
        default_factory=lambda: {"train": 10000, "val": 1000, "test": 1000}
    )

    # Style transfer
    style_transfer: Dict[str, Any] = field(default_factory=lambda: {
        "transfer_mode": "mean_only",
        "blend_mode": "cosine",
        "blend_window_samples": 50,
    })

    # Per-task configs
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "GenerationConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            seed=data.get("global", {}).get("seed", 42),
            n_jobs=data.get("global", {}).get("n_jobs", 4),
            output_dir=Path(data.get("global", {}).get("output_dir", "data/capture24/ts_haystack/tasks")),
            overwrite=data.get("global", {}).get("overwrite", False),
            context_lengths=data.get("context_lengths", [10000]),
            samples_per_split=data.get("samples", {"train": 10000, "val": 1000, "test": 1000}),
            style_transfer=data.get("style_transfer", {}),
            tasks=data.get("tasks", {}),
        )

    def get_enabled_tasks(self) -> List[str]:
        """Get list of enabled tasks."""
        return [name for name, cfg in self.tasks.items() if cfg.get("enabled", True)]

    def get_difficulty_config(self, task_name: str, context_length: int) -> "DifficultyConfig":
        """Build DifficultyConfig for a specific task."""
        from opentslm.time_series_datasets.ts_haystack.core import DifficultyConfig

        task_cfg = self.tasks.get(task_name, {})

        # Extract task_specific keys (everything except standard DifficultyConfig fields)
        standard_fields = {"enabled", "needle_position", "needle_length_ratio_range", "background_purity"}
        task_specific = {k: v for k, v in task_cfg.items() if k not in standard_fields}

        return DifficultyConfig(
            context_length_samples=context_length,
            needle_position=task_cfg.get("needle_position", "random"),
            needle_length_ratio_range=tuple(task_cfg.get("needle_length_ratio_range", [5000, 300000])),
            background_purity=task_cfg.get("background_purity", "pure"),
            task_specific=task_specific,
        )
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | Path | Required | Path to YAML configuration file |
| `--override` | str | None | Override specific values (e.g., `--override global.seed=123`) |
| `--dry-run` | flag | False | Print configuration and exit without generating |
| `--print-default-config` | flag | False | Print default config YAML and exit |

### CLI Usage

```bash
# Generate using config file
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_dataset \
    --config configs/my_experiment.yaml

# Print default config to create a starting point
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_dataset \
    --print-default-config > my_config.yaml

# Dry run to validate config
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_dataset \
    --config configs/my_experiment.yaml \
    --dry-run
```

### SBATCH Integration

```bash
#!/bin/bash
#SBATCH --job-name=ts-haystack-gen
#SBATCH --output=logs/ts_haystack_gen_%j.out
#SBATCH --error=logs/ts_haystack_gen_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

source ~/.bashrc
conda activate opentslm

python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_dataset \
    --config configs/production_config.yaml
```

### Output Structure

```
data/capture24/ts_haystack/tasks/
├── 10s/                               # Context length in seconds (1000 samples at 100Hz)
│   ├── existence/
│   │   ├── train/data.parquet
│   │   ├── val/data.parquet
│   │   └── test/data.parquet
│   ├── localization/
│   │   └── ...
│   └── {all 8 tasks}/
├── 100s/                              # 100 seconds (10000 samples)
│   └── ...
├── 1000s/                             # ~17 minutes (100000 samples)
│   └── ...
└── metadata.json                      # Global generation metadata
```

### Parquet Schema (existing, preserved)

```python
schema = {
    # Sensor data (stored as lists)
    "x": List[float],
    "y": List[float],
    "z": List[float],

    # Task metadata
    "task_type": str,
    "context_length_samples": int,
    "background_pid": str,
    "recording_time_range": str,  # JSON: ["6:00 AM", "8:00 AM"]

    # Q/A
    "question": str,
    "answer": str,
    "answer_type": str,

    # Rich metadata for CoT (passed to LLM)
    "needles": str,           # JSON: List[InsertedNeedle.to_dict()]
    "difficulty_config": str, # JSON: full difficulty config

    # Validation
    "is_valid": bool,
    "validation_notes": Optional[str],
}
```

---

## Component 2: CoT Rationale Generator (LLM-based)

### Location
`src/opentslm/time_series_datasets/ts_haystack/scripts/generate_ts_haystack_cot.py`

### Purpose
Generate fluid chain-of-thought rationales by passing rich metadata + time series plots to an LLM.

**Key Approach (similar to Capture24 CoT but with structured metadata):**
- Generate a 3-axis plot of the accelerometer data
- Extract and format all metadata (needle positions, timestamps, activities, etc.)
- Pass both plot + structured metadata to LLM
- LLM generates natural, grounded reasoning that references the actual data

### Why LLM-based?

While we have all the ground truth metadata, an LLM is needed to:
1. **Generate fluid, natural language** - templates would be too rigid/repetitive
2. **Adapt phrasing to task complexity** - multi-hop requires different reasoning flow than existence
3. **Produce varied expressions** - prevent model overfitting on specific phrasings
4. **Reference visual patterns** - describe what the signal "looks like" at needle locations

### LLM Prompt Design

The prompt provides the LLM with:
1. **The plot** - 3-axis accelerometer visualization
2. **Ground truth metadata** - needle positions, activities, timestamps
3. **Task-specific instructions** - what reasoning pattern to follow

```python
def create_cot_prompt(sample: Dict, task_type: str) -> str:
    """Create LLM prompt with rich metadata context."""

    # Parse metadata
    time_range = json.loads(sample["recording_time_range"])
    needles = json.loads(sample["needles"])
    difficulty_config = json.loads(sample["difficulty_config"])

    # Format needle information
    needle_info = format_needle_metadata(needles)

    # Task-specific context
    task_context = get_task_context(task_type, difficulty_config)

    prompt = f"""You are shown a time-series plot of accelerometer data (X, Y, Z axes) from a wrist-worn sensor.

RECORDING CONTEXT:
- Time span: {time_range[0]} to {time_range[1]}
- Context length: {sample["context_length_samples"]} samples

GROUND TRUTH METADATA (use this to ground your reasoning):
{needle_info}

{task_context}

QUESTION: {sample["question"]}
CORRECT ANSWER: {sample["answer"]}

YOUR TASK:
Write a step-by-step reasoning that:
1. Describes what you observe in the accelerometer patterns
2. Identifies relevant activity bouts using the timestamps provided
3. Explains how you arrive at the answer
4. Ends with "Answer: {sample["answer"]}"

Write your rationale as a natural, flowing paragraph. Reference specific timestamps and signal characteristics.
Do NOT mention that you were given ground truth metadata - reason as if discovering the patterns yourself.

Rationale:"""

    return prompt
```

### Task-Specific Metadata Formatting

```python
def format_needle_metadata(needles: List[Dict]) -> str:
    """Format needle metadata for LLM context."""
    if not needles:
        return "No inserted activity bouts (natural recording)"

    lines = ["INSERTED ACTIVITY BOUTS:"]
    for i, needle in enumerate(needles, 1):
        lines.append(f"""
Bout {i}:
  - Activity: {needle["activity"]}
  - Time: {needle["timestamp_start"]} to {needle["timestamp_end"]}
  - Position: {needle["insert_position_frac"]*100:.1f}% into recording
  - Duration: {needle["duration_ms"]/1000:.1f} seconds""")

    return "\n".join(lines)

def get_task_context(task_type: str, config: Dict) -> str:
    """Get task-specific context for the LLM."""

    contexts = {
        "existence": f"""
TASK TYPE: Existence Detection
- Target activity: {config.get("target_activity", "unknown")}
- Is positive sample: {config.get("is_positive", "unknown")}
- Background activities: {config.get("background_activities", [])}""",

        "localization": f"""
TASK TYPE: Temporal Localization
- Target activity to locate: {config.get("target_activity", "unknown")}
- The model needs to identify WHEN this activity occurred""",

        "counting": f"""
TASK TYPE: Bout Counting
- Target activity: {config.get("target_activity", "unknown")}
- True count: {config.get("count", "unknown")}
- The model needs to count all occurrences""",

        "ordering": f"""
TASK TYPE: Temporal Ordering
- Activity A: {config.get("activity_a", "unknown")}
- Activity B: {config.get("activity_b", "unknown")}
- True order: {config.get("true_order", "unknown")}""",

        "state_query": f"""
TASK TYPE: State Query (Cross-Scale)
- Local event (needle): {config.get("needle_activity", "unknown")}
- Global state at event time: {config.get("global_state_activity", "unknown")}
- The model must identify the background activity when the local event occurred""",

        "antecedent": f"""
TASK TYPE: Temporal Antecedent
- Target activity: {config.get("target_activity", "unknown")}
- Antecedent activity: {config.get("antecedent_activity", "unknown")}
- The model must identify what occurred BEFORE the target""",

        "comparison": f"""
TASK TYPE: Comparison (Extremum Finding)
- Target activity: {config.get("target_activity", "unknown")}
- Query type: {config.get("extremum", "unknown")} period {config.get("polarity", "unknown")} activity
- All periods: {config.get("all_periods", [])}""",

        "multi_hop": f"""
TASK TYPE: Multi-Hop Localization
- Anchor activity: {config.get("anchor_activity", "unknown")}
- Target activity: {config.get("target_activity", "unknown")}
- K (which occurrence): {config.get("K", "unknown")}
- Direction: {config.get("direction", "unknown")}
- The model must: (1) find anchor, (2) find Kth target in direction""",
    }

    return contexts.get(task_type, "TASK TYPE: Unknown")
```

### Plot Generation

```python
def create_accelerometer_plot(
    x: List[float],
    y: List[float],
    z: List[float],
    time_range: Tuple[str, str],
    needles: List[Dict],
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100,
) -> bytes:
    """
    Create a 3-axis accelerometer plot with needle annotations.

    Returns:
        PNG image as bytes
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    n_samples = len(x)
    time_axis = np.linspace(0, 1, n_samples)  # Normalized time

    data = [x, y, z]
    labels = ["X-axis", "Y-axis", "Z-axis"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, d, label, color in zip(axes, data, labels, colors):
        ax.plot(time_axis, d, color=color, linewidth=0.5, alpha=0.8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

        # Annotate needle regions
        for needle in needles:
            start_frac = needle["insert_position_frac"]
            end_frac = start_frac + (needle["duration_samples"] / n_samples)
            ax.axvspan(start_frac, end_frac, alpha=0.2, color="red")
            ax.axvline(start_frac, color="red", linestyle="--", alpha=0.5)

    axes[-1].set_xlabel(f"Time ({time_range[0]} to {time_range[1]})")
    axes[0].set_title("Accelerometer Data")

    plt.tight_layout()

    # Convert to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return buf.read()
```

### LLM Client (Gemini)

```python
class GeminiCoTClient:
    """Client for generating CoT rationales using Gemini."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash-preview-05-20",
        temperature: float = 0.4,
        max_retries: int = 5,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

    def generate_rationale(
        self,
        prompt: str,
        image_bytes: bytes,
    ) -> str:
        """Generate a CoT rationale given prompt and image."""

        image = PIL.Image.open(io.BytesIO(image_bytes))

        for attempt in range(self.max_retries):
            try:
                response = self.client.generate_content(
                    [prompt, image],
                    generation_config=genai.GenerationConfig(
                        temperature=self.temperature,
                    ),
                )
                return response.text
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = min(60, 2 ** attempt)
                    time.sleep(delay)
                else:
                    raise
```

### Main Generator Class

```python
class TSHaystackCoTGenerator:
    """Generate LLM-based CoT rationales for TS-Haystack samples."""

    def __init__(
        self,
        llm_client: GeminiCoTClient,
        include_needle_annotations: bool = True,
        max_workers: int = 4,
    ):
        self.llm_client = llm_client
        self.include_needle_annotations = include_needle_annotations
        self.max_workers = max_workers

    def generate_rationale(self, sample: Dict) -> str:
        """Generate rationale for a single sample."""
        # Parse data
        x = sample["x"] if isinstance(sample["x"], list) else json.loads(sample["x"])
        y = sample["y"] if isinstance(sample["y"], list) else json.loads(sample["y"])
        z = sample["z"] if isinstance(sample["z"], list) else json.loads(sample["z"])
        time_range = json.loads(sample["recording_time_range"])
        needles = json.loads(sample["needles"])

        # Create plot
        plot_bytes = create_accelerometer_plot(
            x, y, z, time_range,
            needles if self.include_needle_annotations else [],
        )

        # Create prompt
        prompt = create_cot_prompt(sample, sample["task_type"])

        # Generate rationale
        rationale = self.llm_client.generate_rationale(prompt, plot_bytes)

        return rationale

    def process_dataset(
        self,
        input_parquet: Path,
        output_parquet: Path,
        resume: bool = True,
    ) -> None:
        """Process entire parquet file, adding rationale column."""
        df = pl.read_parquet(input_parquet)

        # Resume support
        processed_indices = set()
        if resume and output_parquet.exists():
            existing_df = pl.read_parquet(output_parquet)
            processed_indices = set(range(len(existing_df)))

        rationales = [""] * len(df)

        # Load existing rationales
        if processed_indices:
            existing_df = pl.read_parquet(output_parquet)
            for i, row in enumerate(existing_df.iter_rows(named=True)):
                if i < len(rationales):
                    rationales[i] = row.get("rationale", "")

        # Process remaining samples
        to_process = [i for i in range(len(df)) if i not in processed_indices]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for i in to_process:
                row = df.row(i, named=True)
                future = executor.submit(self.generate_rationale, row)
                futures[future] = i

            for future in tqdm(as_completed(futures), total=len(futures)):
                i = futures[future]
                try:
                    rationales[i] = future.result()
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    rationales[i] = ""

                # Incremental save every 100 samples
                if sum(1 for r in rationales if r) % 100 == 0:
                    self._save_partial(df, rationales, output_parquet)

        # Final save
        df = df.with_columns(pl.Series("rationale", rationales))
        df.write_parquet(output_parquet)

    def _save_partial(self, df, rationales, output_path):
        """Save partial results for resume capability."""
        temp_df = df.with_columns(pl.Series("rationale", rationales))
        temp_df.write_parquet(output_path)
```

### CLI

```bash
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_cot \
    --context-lengths 10000 \
    --tasks all \
    --input-dir data/capture24/ts_haystack/tasks \
    --output-dir data/capture24/ts_haystack/cot \
    --max-workers 4 \
    --model gemini-2.5-flash-preview-05-20 \
    --temperature 0.4
```

### Output Structure

```
data/capture24/ts_haystack/cot/
├── 100s/                              # Context length in seconds (10000 samples)
│   ├── existence/
│   │   ├── train/data.parquet         # Same schema + "rationale" column
│   │   ├── val/data.parquet
│   │   └── test/data.parquet
│   ├── localization/
│   │   └── ...
│   └── {all 8 tasks}/
├── 1000s/                             # ~17 minutes (100000 samples)
│   └── ...
└── metadata.json                      # Generation metadata (model, params, etc.)
```

---

## Component 3: TSHaystackCoTQADataset

### Location
`src/opentslm/time_series_datasets/ts_haystack/dataset/TSHaystackCoTQADataset.py`

### Purpose
QADataset implementation for training OpenTSLM models on TS-Haystack data with CoT rationales.

### Design

```python
class TSHaystackCoTQADataset(QADataset):
    """
    QADataset for TS-Haystack benchmark with chain-of-thought rationales.

    Supports:
    - Single task or multi-task training
    - Configurable context lengths (in seconds for readability)
    - CoT rationales as answers

    Usage:
        # Single task (context length in seconds)
        dataset = TSHaystackCoTQADataset(
            split="train",
            EOS_TOKEN="<eos>",
            tasks=["existence"],
            context_lengths_seconds=[100],    # 100s = 10000 samples at 100Hz
        )

        # Multi-task
        dataset = TSHaystackCoTQADataset(
            split="train",
            EOS_TOKEN="<eos>",
            tasks=["existence", "localization", "counting"],
            context_lengths_seconds=[100, 1000],  # 100s and ~17min
        )
    """

    # Class-level caching
    _cached_config: Optional[Tuple] = None
    _train_dataset: Optional[Dataset] = None
    _validation_dataset: Optional[Dataset] = None
    _test_dataset: Optional[Dataset] = None
    loaded: bool = False

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        tasks: List[str] = ["existence"],
        context_lengths_seconds: List[int] = [100],  # Context length in seconds
        data_dir: Optional[Path] = None,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        # Configuration
        self.tasks = tasks if tasks != ["all"] else list(TASK_REGISTRY.keys())
        self.context_lengths_seconds = context_lengths_seconds
        self.data_dir = data_dir or Path("data/capture24/ts_haystack/cot")

        # Check config consistency
        config = (tuple(sorted(self.tasks)), tuple(sorted(context_lengths_seconds)))
        if TSHaystackCoTQADataset._cached_config is not None:
            if TSHaystackCoTQADataset._cached_config != config:
                warnings.warn(
                    f"TSHaystackCoTQADataset already loaded with different config. "
                    f"Existing: {TSHaystackCoTQADataset._cached_config}, "
                    f"Requested: {config}. Using existing."
                )

        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load train, validation, and test datasets."""
        return load_ts_haystack_cot_splits(
            tasks=self.tasks,
            context_lengths_seconds=self.context_lengths_seconds,
            data_dir=self.data_dir,
        )

    def _get_answer(self, row) -> str:
        """Get the answer from the row (rationale with embedded answer)."""
        return row["rationale"]

    def _get_pre_prompt(self, row) -> str:
        """Get the pre-prompt text with task-specific instructions."""
        time_range = json.loads(row["recording_time_range"])

        return f"""You are given accelerometer data in all three dimensions from a wrist-worn sensor.
The recording spans from {time_range[0]} to {time_range[1]}.

Your task is to analyze the time series and answer the following question.

Instructions:
- Examine the accelerometer patterns carefully
- Think step-by-step about what the signal patterns indicate
- Write your reasoning as a natural paragraph
- End with "Answer: " followed by your answer

Question: {row["question"]}
"""

    def _get_post_prompt(self, row) -> str:
        """Get the post-prompt text."""
        return "Rationale:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """Convert time series data to TextTimeSeriesPrompt objects."""
        # Parse time series data
        x = row["x"] if isinstance(row["x"], list) else json.loads(row["x"])
        y = row["y"] if isinstance(row["y"], list) else json.loads(row["y"])
        z = row["z"] if isinstance(row["z"], list) else json.loads(row["z"])

        series = torch.tensor([x, y, z], dtype=torch.float32)

        # Normalize per axis
        means = series.mean(dim=1, keepdim=True)
        stds = series.std(dim=1, keepdim=True).clamp(min=1e-6)
        series_norm = (series - means) / stds

        TIME_SERIES_LABELS = [
            "The following is the accelerometer data on the x-axis",
            "The following is the accelerometer data on the y-axis",
            "The following is the accelerometer data on the z-axis",
        ]

        prompts = []
        for label, ts, mean, std in zip(
            TIME_SERIES_LABELS,
            series_norm.tolist(),
            means.squeeze().tolist(),
            stds.squeeze().tolist(),
        ):
            text = f"{label}, it has mean {mean:.4f} and std {std:.4f}:"
            prompts.append(TextTimeSeriesPrompt(text, ts))

        return prompts

    @staticmethod
    def get_labels() -> List[str]:
        """Return possible activity labels."""
        return [
            "sleep", "sedentary", "light", "moderate-vigorous",
            "walking", "bicycling", "vehicle", "mixed",
        ]

    def _format_sample(self, row) -> Dict:
        """Format a sample with additional metadata."""
        sample = super()._format_sample(row)
        sample["task_type"] = row["task_type"]
        sample["answer_type"] = row["answer_type"]
        sample["context_length"] = row["context_length_samples"]
        sample["direct_answer"] = row["answer"]  # Keep original answer for evaluation
        return sample
```

### Loader Function

```python
def load_ts_haystack_cot_splits(
    tasks: List[str],
    context_lengths_seconds: List[int],
    data_dir: Path = Path("data/capture24/ts_haystack/cot"),
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load TS-Haystack CoT datasets for specified tasks and context lengths.

    Directory structure expected:
        data_dir/{context_seconds}s/{task}/{split}/data.parquet

    Args:
        tasks: List of task names
        context_lengths_seconds: List of context lengths in seconds
        data_dir: Base data directory (should point to cot/ directory)

    Returns:
        Tuple of (train, validation, test) HuggingFace Datasets
    """
    train_dfs, val_dfs, test_dfs = [], [], []

    for ctx_seconds in context_lengths_seconds:
        ctx_dir = f"{ctx_seconds}s"  # e.g., "100s", "1000s"
        for task in tasks:
            base_path = data_dir / ctx_dir / task

            for split, dfs in [("train", train_dfs), ("val", val_dfs), ("test", test_dfs)]:
                parquet_path = base_path / split / "data.parquet"
                if parquet_path.exists():
                    df = pl.read_parquet(parquet_path)
                    dfs.append(df)
                else:
                    print(f"Warning: {parquet_path} not found, skipping")

    # Concatenate all dataframes
    train_df = pl.concat(train_dfs) if train_dfs else pl.DataFrame()
    val_df = pl.concat(val_dfs) if val_dfs else pl.DataFrame()
    test_df = pl.concat(test_dfs) if test_dfs else pl.DataFrame()

    # Convert to HuggingFace Dataset
    return (
        Dataset.from_pandas(train_df.to_pandas()) if len(train_df) > 0 else Dataset.from_dict({}),
        Dataset.from_pandas(val_df.to_pandas()) if len(val_df) > 0 else Dataset.from_dict({}),
        Dataset.from_pandas(test_df.to_pandas()) if len(test_df) > 0 else Dataset.from_dict({}),
    )
```

---

## File Structure (Final)

```
src/opentslm/time_series_datasets/ts_haystack/
├── core/                              # Existing
│   └── ...
├── tasks/                             # Existing
│   └── ...
├── dataset/                           # NEW
│   ├── __init__.py
│   ├── ts_haystack_cot_loader.py      # load_ts_haystack_cot_splits()
│   └── TSHaystackCoTQADataset.py      # QADataset implementation
├── cot/                               # NEW
│   ├── __init__.py
│   ├── cot_generator.py               # TSHaystackCoTGenerator
│   ├── llm_client.py                  # GeminiCoTClient
│   ├── prompt_builder.py              # create_cot_prompt(), format_needle_metadata()
│   └── plot_generator.py              # create_accelerometer_plot()
├── scripts/
│   ├── build_core_artifacts.py        # Existing
│   ├── generate_ts_haystack_dataset.py # NEW - Centralized generator
│   └── generate_ts_haystack_cot.py     # NEW - CoT rationale generator
└── test/
    ├── ...                            # Existing
    ├── test_cot_generator.py          # NEW
    └── test_cot_qa_dataset.py         # NEW
```

---

## Data Directory Structure

```
data/capture24/ts_haystack/
├── timelines/                         # Phase 1 (existing)
│   └── P*.parquet
├── bout_index.parquet                 # Phase 1 (existing)
├── transition_matrix.json             # Phase 1 (existing)
│
├── tasks/                             # Component 1 output
│   ├── 10s/                           # Context length in seconds (1000 samples)
│   │   ├── existence/
│   │   │   ├── train/data.parquet
│   │   │   ├── val/data.parquet
│   │   │   └── test/data.parquet
│   │   ├── localization/
│   │   └── ...
│   ├── 100s/                          # 10000 samples
│   │   └── ...
│   ├── 1000s/                         # 100000 samples (~17 min)
│   │   └── ...
│   └── metadata.json
│
└── cot/                               # Component 2 output
    ├── 10s/
    │   ├── existence/
    │   │   ├── train/data.parquet     # + rationale column
    │   │   ├── val/data.parquet
    │   │   └── test/data.parquet
    │   └── ...
    ├── 100s/
    │   └── ...
    ├── 1000s/
    │   └── ...
    └── metadata.json                  # LLM model, params, generation stats
```

---

## Pre-Implementation: Code Updates Required

### ~~Issue 1: Needle Length as Ratio (not absolute)~~ ✅ COMPLETED

This change has been implemented. `DifficultyConfig` now uses `needle_length_ratio_range` instead of `needle_length_range_ms`. All task generators and tests have been updated.

---

### Issue 2: Directory Naming Convention

**Problem:** Current uses samples (`10000/`) which is not human-readable.

**Solution:** Use seconds like Capture24 (`100s/`).

**Directory structure:**
```
data/capture24/ts_haystack/
├── tasks/
│   ├── 10s/                           # 1000 samples at 100Hz
│   │   ├── existence/
│   │   │   ├── train/data.parquet
│   │   │   └── ...
│   │   └── ...
│   ├── 100s/                          # 10000 samples
│   ├── 1000s/                         # 100000 samples (~17min)
│   └── metadata.json
└── cot/
    ├── 10s/
    ├── 100s/
    └── ...
```

**Config uses seconds:**
```yaml
# Human-readable seconds (converted to samples internally)
context_lengths_seconds:
  - 100     # 100s = 10000 samples at 100Hz
  - 1000    # 1000s = 100000 samples
```

---

### Updated YAML Config with Ratio-Based Needle Lengths

```yaml
# TS-Haystack Dataset Generation Configuration
# =============================================

global:
  seed: 42
  n_jobs: 4
  output_dir: data/capture24/ts_haystack/tasks
  overwrite: false
  source_hz: 100                        # Capture24 sampling rate

# Context lengths in SECONDS (more readable)
# Converted internally: seconds × source_hz = samples
context_lengths_seconds:
  - 100                                 # 100s = 10000 samples
  - 1000                                # 1000s = 100000 samples (~17min)

samples:
  train: 10000
  val: 1000
  test: 1000

style_transfer:
  transfer_mode: mean_only
  blend_mode: cosine
  blend_window_samples: 50

# Task Configurations
# needle_length_ratio_range: [min_ratio, max_ratio] as fraction of context length
tasks:

  existence:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.02, 0.10]   # 2-10% of context
    background_purity: pure
    margin_samples: 100

  localization:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.02, 0.10]
    background_purity: pure
    use_transition_probs: false
    margin_samples: 100

  counting:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.02, 0.08]   # Smaller range for multiple needles
    background_purity: pure
    min_bouts: 1
    max_bouts: 5
    min_gap_samples: 100
    margin_samples: 100

  ordering:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.02, 0.10]
    background_purity: pure
    min_gap_samples: 100
    margin_samples: 100
    question_format: boolean

  state_query:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.01, 0.05]   # Smaller - local event in global context
    # background_purity: mixed (HARDCODED)
    min_global_states: 2
    max_global_states: 5
    min_state_duration_samples: 500
    position_mode: random
    boundary_margin_frac: 0.1

  antecedent:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.02, 0.08]   # Two needles need to fit
    background_purity: pure
    background_mode: low_activity
    adjacency_gap_samples: 10
    margin_samples: 100
    use_transition_probs: false

  comparison:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.02, 0.08]   # Multiple needles
    background_purity: pure
    min_bouts: 2
    max_bouts: 4
    min_duration_diff_ms: 2000
    min_gap_samples: 100
    margin_samples: 100

  multi_hop:
    enabled: true
    needle_position: random
    needle_length_ratio_range: [0.02, 0.06]   # Multiple needles + anchor
    background_purity: pure
    k_distribution: [0.4, 0.4, 0.2]
    direction_mode: random
    n_distractors_opposite: 0
    min_gap_samples: 100
    margin_samples: 100
```

---

## Implementation Order

### ~~Phase 0: Update Core Code~~ ✅ COMPLETED
The needle length ratio change has been implemented:
- `DifficultyConfig` now uses `needle_length_ratio_range` (ratio of context length)
- Added `get_needle_length_range_samples()` and `get_needle_length_range_ms()` helper methods
- All 8 task generators updated to use the new API
- All tests updated and passing

### Phase 1: Dataset Generator Script (Priority: High)
1. Create `scripts/generate_ts_haystack_dataset.py`
2. Implement `GenerationConfig` with seconds-based context lengths
3. Add CLI argument parsing
4. Implement batch generation with progress tracking
5. Add resume capability (skip existing files)
6. Use `{context_seconds}s/{task}` directory structure
7. Test with small sample counts

### Phase 2: CoT Rationale Generator (Priority: High)
1. Create `cot/llm_client.py` with `GeminiCoTClient`
2. Create `cot/plot_generator.py` with `create_accelerometer_plot()`
3. Create `cot/prompt_builder.py` with metadata formatting functions
4. Implement `cot/cot_generator.py` with `TSHaystackCoTGenerator`
5. Create `scripts/generate_ts_haystack_cot.py` CLI
6. Test with a few samples to verify rationale quality
7. Add resume support for long-running generation

### Phase 3: TSHaystackCoTQADataset (Priority: High)
1. Create `dataset/ts_haystack_cot_loader.py`
2. Implement `dataset/TSHaystackCoTQADataset.py`
3. Add to `dataset/__init__.py` exports
4. Test with training pipeline
5. Verify DataLoader compatibility

### Phase 4: Testing & Validation (Priority: Medium)
1. Unit tests for CoT generator
2. Integration tests for QADataset
3. End-to-end training test
4. Documentation and README updates

---

## Usage Examples

### Generate Dataset
```bash
# First, create or copy a config file
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_dataset \
    --print-default-config > configs/my_experiment.yaml

# Edit the config as needed, then run:
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_dataset \
    --config configs/my_experiment.yaml

# Dry run to validate config
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_dataset \
    --config configs/my_experiment.yaml \
    --dry-run
```

### Generate CoT Rationales
```bash
# Generate rationales for a specific context length
# Requires that the task data already exists in data/capture24/ts_haystack/tasks/
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_cot \
    --config configs/cot_generation_config.yaml

# Or with CLI args for quick runs
python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_cot \
    --input-dir data/capture24/ts_haystack/tasks/100s \
    --output-dir data/capture24/ts_haystack/cot/100s \
    --tasks all \
    --max-workers 4
```

### Training with TSHaystackCoTQADataset
```python
from opentslm.time_series_datasets.ts_haystack import TSHaystackCoTQADataset

# Single task training (context length in seconds)
dataset = TSHaystackCoTQADataset(
    split="train",
    EOS_TOKEN="<eos>",
    tasks=["existence"],
    context_lengths_seconds=[100],       # 100s = 10000 samples
)

# Multi-task training
dataset = TSHaystackCoTQADataset(
    split="train",
    EOS_TOKEN="<eos>",
    tasks=["existence", "localization", "counting", "ordering"],
    context_lengths_seconds=[100, 1000],  # 100s and 1000s (~17min)
)

# Access sample
sample = dataset[0]
print(sample["pre_prompt"])
print(sample["time_series"])
print(sample["answer"])  # Contains full rationale
print(sample["direct_answer"])  # Original answer for evaluation
```

---

## Key Design Decisions

1. **Directory Structure `{context_len}/{task}`**: This allows:
   - Generating CoT for a specific context length independently
   - Easy iteration over all tasks at a fixed context length
   - Natural grouping for curriculum learning by length

2. **LLM-based CoT Generation**: Unlike template-based approaches, using an LLM:
   - Produces natural, varied language
   - Can describe visual patterns in the signal
   - Adapts reasoning flow to task complexity
   - Still grounded by explicit metadata passed in prompt

3. **Rich Metadata in LLM Prompt**: The LLM receives:
   - Ground truth needle positions and timestamps
   - Task-specific context (activities, counts, etc.)
   - Visual plot with optional needle annotations
   - This ensures factual correctness while allowing natural expression

4. **Separate CoT Directory**: Allows re-generation of rationales without re-running expensive task generation.

5. **Class-Level Caching**: Following existing QADataset pattern for efficiency.

---

## Example LLM-Generated Rationale

**Task: Counting**
**Question:** "How many walking bouts occurred in this recording?"
**Metadata passed to LLM:**
- Time span: 6:00 AM to 7:40 AM
- 3 walking bouts at: 6:12-6:18 AM, 6:35-6:42 AM, 7:15-7:22 AM

**Generated Rationale:**
```
Looking at the accelerometer data spanning from 6:00 AM to 7:40 AM, I need to
identify all walking bouts. The signal shows predominantly low-variance patterns
consistent with sedentary activity, but I can identify three distinct periods of
rhythmic, moderate-intensity oscillations characteristic of walking gait.

The first walking bout appears around 6:12 AM and continues until approximately
6:18 AM, showing the typical regular patterns of heel-strike and toe-off. A second
walking period begins at 6:35 AM, lasting until 6:42 AM with similar oscillatory
characteristics. Finally, a third walking bout is visible from 7:15 AM to 7:22 AM.

The intervals between these bouts show the low-activity baseline of sitting or
standing. Counting all the distinct walking periods: 3.

Answer: 3
```

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM API rate limits | Slow generation | Retry logic, parallel workers, resume support |
| Rationale-answer mismatch | Training corruption | Post-process validation, reject mismatches |

---

## Success Criteria

1. **Dataset Generator**: Can generate 10K+ samples per task at 100K context length in reasonable time

2. **CoT Generator**: Rationales are:
   - Grammatically correct and natural
   - Factually grounded (use provided timestamps/activities)
   - End with correct answer format
   - Diverse in phrasing across samples

3. **QADataset**:
   - Loads correctly into DataLoader
   - Compatible with existing training pipeline
   - Supports multi-task training
   - Handles variable context lengths
