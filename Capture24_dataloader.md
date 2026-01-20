**Final Goal of this project section:** Implement a finetuning and evaluation pipeline of the OpenTSLM-Flamingo architecture to the Capture-24 dataset.

# The dataset:

Downloads a capture24.zip which contains the following:

-- annotation-label-dictionary.csv
-- metadata.csv
-- PXXX.csv.gz

# annotation-label-dictionary.csv
```sample
annotation	label:WillettsSpecific2018	label:WillettsMET2018	label:DohertySpecific2018	label:Willetts2018	label:Doherty2018	label:Walmsley2020
7030 sleeping;MET 0.95	sleep	sleep	sleep	sleep	sleep	sleep
occupation;office and administrative support;11580 office/computer work general;MET 1.5	sitting	sitstand+lowactivity	sedentary-screen	sit-stand	sedentary	sedentary
home activity;household chores;preparing meals/cooking/washing dishes;5035 kitchen activity general cooking/washing/dishes/cleaning up;MET 3.3	household-chores	sitstand+activity	tasks-moderate	mixed	moderate	light
occupation;office and administrative support;11580 office wok/computer work general;MET 1.5	sitting	sitstand+lowactivity	sedentary-screen	sit-stand	sedentary	sedentary
home activity;miscellaneous;sitting;9060 sitting/lying reading or without observable/identifiable activities;MET 1.3	sitting	sitstand+lowactivity	sedentary-non-screen	sit-stand	sedentary	sedentary
home activity;miscellaneous;walking;17150 walking household without observable loads;MET 2.0	walking	walking	walking	walking	walking	light
transportation;private transportation;16010 driving automobile or light truck (not a semi);MET 2.5	vehicle	vehicle	vehicle	vehicle	sedentary	sedentary
home activity;miscellaneous;sitting;7010 sitting/lying and watching television with TV on as the primary activity;MET 1.0	sitting	sitting	sedentary-screen	sit-stand	sedentary	sedentary
home activity;miscellaneous;sitting;11580 office/computer work general;MET 1.5	sitting	sitstand+lowactivity	sedentary-screen	sit-stand	sedentary	sedentary
home activity;miscellaneous;sitting;9055 sitting/lying talking in person/using a mobile phone/smartphone/tablet or talking on the phone/computer (skype chatting);MET 1.5	sitting	sitstand+lowactivity	sedentary-screen	sit-stand	sedentary	sedentary
home activity;miscellaneous;sitting;11580 office work such as writing and typing (with or without eating at the same time);MET 1.5	sitting	sitstand+lowactivity	sedentary-screen	sit-stand	sedentary	sedentary
home activity;miscellaneous;walking;5165 (generic) walking non-cleaning task such as closing windows lock door putting away items;MET 3.5	walking	walking	walking	walking	walking	light
home activity;eating;13030 eating sitting alone or with someone;MET 1.5	sitting	sitstand+activity	sedentary-non-screen	sit-stand	sedentary	sedentary
leisure;miscellaneous;walking;21070 (generic) walking and occasional standing (no more than two consecutive images);MET 2.5	walking	walking+activity	walking	walking	walking	light
transportation;walking;17161 walking not as the single means of transports e.g.from house to transports or vice versa/from car to places or vice versa/between transports;MET 2.5	walking	walking	walking	walking	walking	light
home activity;miscellaneous;sitting;9030 sitting desk entertainment/hobby (with or without eating at the same time);MET
```

# metadata.csv
```sample
pid	age	sex
P001	38-52	F
P002	30-37	F
P003	30-37	F
P004	53+	F
P005	38-52	F
P006	18-29	F
P007	53+	M
P008	53+	F
P009	53+	F
P010	18-29	F
```

# PXXX file
Each P{id}.csv.gz is present in the file, this is the sample content:

```sample
time,x,y,z,annotation
2016-10-17 03:53:00.000000,-0.38611975,0.48305458,-0.79465705,7030 sleeping;MET 0.95
2016-10-17 03:53:00.010000,-0.38611975,0.48305458,-0.79465705,7030 sleeping;MET 0.95
2016-10-17 03:53:00.020000,-0.38611975,0.48305458,-0.79465705,7030 sleeping;MET 0.95
2016-10-17 03:53:00.030000,-0.38611975,0.48305458,-0.79465705,7030 sleeping;MET 0.95
2016-10-17 03:53:00.040000,-0.38611975,0.48305458,-0.79465705,7030 sleeping;MET 0.95
2016-10-17 03:53:00.050000,-0.38611975,0.48305458,-0.7785672,7030 sleeping;MET 0.95
2016-10-17 03:53:00.060000,-0.38611975,0.48305458,-0.7785672,7030 sleeping;MET 0.95
```

Some of the data does not have annotation.


# 1. Dataset loading and first evaluation.

Our first goal is to load the raw dataset and be able to create a set of training and evaluation pairs.
The way i was thinking about it is as follows:

We should be able to select a timeframe for this dataset (example: 1 second, 5s, 10s, 1m, 5m, 30m, 60m). Then we should check if the datset already exists locally considering the path.

We should also be able to select which of the annotations labels we want to use. So, based on this annotation, we need to make sure that we find samples of data that are fully annotated into these.

We also want to add the option to downsample the readings.

The file that will do this should be created in src/opentslm/time_series_datasets/capture24.

---

# Implementation Plan: Step 1 - Data Loading into Parquet

## Goal
Extract the Capture-24 dataset from ZIP and load into Parquet files stored at `data/capture24/` for efficient querying.

## Dataset Scale
- **151 participants** with ~7 days of wrist-worn accelerometer data each
- **100Hz sampling rate** = ~60 million rows per participant
- **Total**: ~9 billion rows of sensor data

## Files to Create

```
src/opentslm/time_series_datasets/capture24/
├── __init__.py              # Module exports
├── capture24_loader.py      # ZIP extraction and Parquet conversion
├── README.md                # (existing - update with usage)
├── annotation-label-dictionary.csv  # (existing)
└── metadata.csv             # (existing)

data/capture24/
├── participants.parquet     # Metadata (151 rows)
├── label_mappings.parquet   # Annotation dictionary (~207 rows)
└── sensor_data/             # Partitioned by participant
    ├── pid=P001/
    │   └── data.parquet
    ├── pid=P002/
    │   └── data.parquet
    └── ...
```

## Implementation Steps

### Step 1: Create `capture24_loader.py`
Data extraction and Parquet conversion:

**Constants:**
- `CAPTURE24_DATA_DIR` - `data/capture24/`
- `CAPTURE24_ZIP_PATH` - `data/capture24.zip`
- `SENSOR_DATA_DIR` - `data/capture24/sensor_data/`

**Functions:**
- `ensure_capture24_data()` - main entry point, ensures data is extracted
- `extract_and_convert_to_parquet()` - orchestrate full extraction
- `process_participant_file()` - convert single gzipped CSV to Parquet
- `parse_timestamp()` - convert datetime strings to Unix milliseconds
- `load_participants()` - load participant metadata as DataFrame
- `load_label_mappings()` - load annotation dictionary as DataFrame
- `is_data_ready()` - check if Parquet files exist

**Key implementation details:**
- Use `pyarrow` for Parquet writing with snappy compression
- Partition sensor data by `pid` for efficient per-participant queries
- Store timestamps as `int64` (milliseconds) for fast range queries
- Process each participant sequentially to manage memory

### Step 2: Create `__init__.py`
Export main functions:
- `ensure_capture24_data()`
- `load_participants()`
- `load_label_mappings()`
- `load_participant_sensor_data(pid: str)`
- Constants: `CAPTURE24_DATA_DIR`, `SENSOR_DATA_DIR`

### Step 3: Update README.md
Add usage examples for the Parquet loader.

## Parquet Schema

**participants.parquet:**
```
pid: string
age: string
sex: string
```

**label_mappings.parquet:**
```
annotation: string
label_willetts_specific_2018: string
label_willetts_met_2018: string
label_doherty_specific_2018: string
label_willetts_2018: string
label_doherty_2018: string
label_walmsley_2020: string
```

**sensor_data/pid=PXXX/data.parquet:**
```
timestamp_ms: int64
x: float32
y: float32
z: float32
annotation: string (nullable)
```

## Key Design Decisions

1. **Parquet over SQLite**: Better compression (~3-5x), columnar reads, native pandas/polars support
2. **Partition by participant**: Each participant's data in separate directory for efficient loading
3. **Timestamps as int64 (ms)**: Fast range filtering, smaller than datetime
4. **Snappy compression**: Good balance of speed and compression ratio
5. **Float32 for accelerometer**: Sufficient precision, halves storage vs float64

## Storage Estimate
- Raw CSV data: ~450GB uncompressed
- Parquet with snappy: ~30-50GB (columnar + compression)

## Query Patterns

```python
import polars as pl

# Load single participant
df = pl.read_parquet("data/capture24/sensor_data/pid=P001/data.parquet")

# Load with time filtering (lazy evaluation)
df = (
    pl.scan_parquet("data/capture24/sensor_data/pid=P001/data.parquet")
    .filter(pl.col("timestamp_ms") >= start)
    .filter(pl.col("timestamp_ms") < end)
    .collect()
)

# Load multiple participants (lazy, efficient)
df = (
    pl.scan_parquet("data/capture24/sensor_data/*/data.parquet")
    .filter(pl.col("pid").is_in(["P001", "P002", "P003"]))
    .collect()
)
```

## Verification
After implementation, verify with:
```python
from opentslm.time_series_datasets.capture24 import (
    ensure_capture24_data,
    load_participants,
    load_participant_sensor_data,
)

ensure_capture24_data()

# Check participants loaded
participants = load_participants()
assert len(participants) == 151

# Check sensor data for one participant
df = load_participant_sensor_data("P001")
print(f"P001 has {len(df)} samples")
assert "timestamp_ms" in df.columns
assert "x" in df.columns
```

## Dependencies
Add to pyproject.toml if not present:
- `polars>=1.0`
- `pyarrow>=14.0`

# Implementation Plan: Step 2A - Window Extraction

## Goal
Extract non-overlapping windows of sensor data with their raw annotations preserved.

## Output Structure
```
data/capture24/
├── sensor_data/                         # Step 1 (raw parquet)
└── windows/                             # Step 2A (extracted windows)
    └── {window_size_s}s_{downsample_hz}hz/
        ├── train/
        │   └── data.parquet
        ├── val/
        │   └── data.parquet
        └── test/
            └── data.parquet
```

## Window Schema
```
window_id: string           # "{pid}_{start_ms}" for uniqueness
pid: string
start_ms: int64
end_ms: int64
x: list[float32]            # length = window_size_s * downsample_hz
y: list[float32]
z: list[float32]
annotations: list[string]   # same length as x/y/z, one annotation per sample
```

**Note**: Sensor sample timestamps are uniformly distributed from `start_ms` to `end_ms`, so per-sample timestamps are not stored.

## Files to Create
```
src/opentslm/time_series_datasets/capture24/
├── __init__.py                  # (update exports)
├── capture24_loader.py          # (existing - Step 1)
└── capture24_windows.py         # Window extraction
```

## Functions in `capture24_windows.py`

**Constants:**
- `WINDOWS_DIR` - `data/capture24/windows/`

**Functions:**
- `extract_windows(window_size_s: int, downsample_hz: int, label_scheme: str, seed: int = 42)` - main entry point
- `get_windows_path(window_size_s: int, downsample_hz: int) -> Path` - returns path for given config
- `split_participants(seed: int) -> tuple[list, list, list]` - returns (train_pids, val_pids, test_pids)
- `extract_participant_windows(pid: str, window_size_s: int, downsample_hz: int) -> pl.DataFrame` - extract windows for one participant
- `downsample_window(data: pl.DataFrame, target_hz: int) -> pl.DataFrame` - downsample from 100Hz to target
- `load_windows(window_size_s: int, downsample_hz: int, split: str) -> pl.DataFrame` - load extracted windows

## Key Design Decisions

1. **Downsampling**: Take every Nth sample where N = 100 / target_hz (e.g., 100Hz → 10Hz means take every 10th sample)
2. **Participant split**: 100 train / 25 val / 26 test, random with fixed seed for reproducibility
3. **Discard incomplete windows**: Windows with any missing annotations are discarded
4. **Flat arrays over nested structs**: Store x, y, z as separate lists for easier tensor conversion
5. **Per-sample annotations**: Each sample has its corresponding annotation string (same length as sensor arrays)

## Verification
```python
from opentslm.time_series_datasets.capture24 import (
    extract_windows,
    load_windows,
)

# Extract 10-second windows at 30Hz
extract_windows(window_size_s=10, downsample_hz=30, label_scheme="Walmsley2020")

# Load training set
train_df = load_windows(window_size_s=10, downsample_hz=30, split="train")
assert len(train_df.columns) == 7  # window_id, pid, start_ms, end_ms, x, y, z, annotations
assert len(train_df["x"][0]) == 300  # 10s * 30Hz

# Verify annotations length matches sensor data
assert len(train_df["x"][0]) == len(train_df["annotations"][0])
```

# Implementation Plan: Step 2B - Task-Specific Formatting

In phase 2A we kept all raw annotations per window. This next phase converts windows into supervised tasks for our TSLM models. We will have two types of tasks – classification and QA pairs.

---

## Classification Dataset

### Design Decision: Store Mapped Labels Directly

We store the **final mapped labels** (e.g., "sleep", "sedentary") rather than raw annotations in the classification dataset. Rationale:

1. **Different schemes = different models** - A 4-class Walmsley2020 model is fundamentally different from a 12-class WillettsSpecific2018 model
2. **Phase 2A preserves raw annotations** - Users can regenerate classification datasets with different schemes from the phase 2A windows
3. **Self-contained datasets** - Standard ML practice; no runtime mapping needed
4. **Path encodes the scheme** - Clear which label scheme was used

### How Label Mapping Works

Raw annotations in sensor data (e.g., `"7030 sleeping;MET 0.95"`) are mapped to simplified labels using `annotation-label-dictionary.csv`:

```
Raw annotation                                              → Walmsley2020
─────────────────────────────────────────────────────────────────────────
"7030 sleeping;MET 0.95"                                    → "sleep"
"occupation;office and administrative support;11580..."     → "sedentary"
"home activity;miscellaneous;walking;17150..."              → "light"
"sports/gym;MET 8.0"                                        → "moderate-vigorous"
```

### Available Label Schemes

| Scheme | Classes | Labels |
|--------|---------|--------|
| **Walmsley2020** | 4 | `sleep`, `sedentary`, `light`, `moderate-vigorous` |
| **Doherty2018** | 5 | `sleep`, `sedentary`, `tasks-light`, `walking`, `moderate` |
| **Willetts2018** | 5 | `sleep`, `sit-stand`, `mixed`, `walking`, `vehicle` |
| **WillettsSpecific2018** | ~12 | `sleep`, `sitting`, `standing`, `walking`, `sports`, `manual-work`, `household-chores`, `mixed-activity`, `bicycling`, `vehicle` |
| **DohertySpecific2018** | ~10 | `sleep`, `sedentary-screen`, `sedentary-non-screen`, `tasks-light`, `tasks-moderate`, `walking`, `sports-continuous`, `sport-interrupted`, `bicycling`, `vehicle` |
| **WillettsMET2018** | ~10 | `sleep`, `sitting`, `sitstand+lowactivity`, `sitstand+activity`, `walking`, `walking+activity`, `gym`, `sports`, `bicycling`, `vehicle` |

### Window-Level Label Assignment

Each window contains multiple samples (e.g., 300 samples for 10s @ 30Hz), each with its own annotation. To get a single label per window:

1. **Map** each raw annotation to the chosen label scheme
2. **Mode** (most frequent label) becomes the window label
3. **Filter** windows where mode doesn't meet a confidence threshold (optional)

```python
# Example: 10s window with 300 samples
annotations = ["7030 sleeping;MET 0.95"] * 250 + ["sitting;..."] * 50
mapped = ["sleep"] * 250 + ["sedentary"] * 50
label = mode(mapped)  # → "sleep" (83% confidence)
```

### Output Structure

```
data/capture24/
├── windows/                                    # Phase 2A
│   └── {window_size_s}s_{hz}hz/
│       ├── train/data.parquet
│       ├── val/data.parquet
│       └── test/data.parquet
└── classification/                             # Phase 2B
    └── {window_size_s}s_{hz}hz/{label_scheme}/
        ├── train/data.parquet
        ├── val/data.parquet
        ├── test/data.parquet
        └── metadata.json                       # class names, counts, etc.
```

### Classification Schema

```
window_id: string           # "{pid}_{start_ms}"
pid: string                 # Participant ID
start_ms: int64             # Window start timestamp
end_ms: int64               # Window end timestamp
x: list[float32]            # Accelerometer x-axis
y: list[float32]            # Accelerometer y-axis
z: list[float32]            # Accelerometer z-axis
label: string               # Mapped label (e.g., "sleep", "sedentary")
label_id: int32             # Integer encoding of label (for training)
confidence: float32         # Fraction of samples with the mode label
```

### Files to Create

```
src/opentslm/time_series_datasets/capture24/
├── capture24_classification.py     # Classification dataset creation
└── __init__.py                     # Update exports
```

### Functions in `capture24_classification.py`

**Constants:**
- `CLASSIFICATION_DIR` - `data/capture24/classification/`
- `LABEL_SCHEMES` - List of valid scheme names

**Functions:**
- `load_label_mapping(label_scheme: str) -> dict[str, str]` - Load annotation → label mapping
- `get_window_label(annotations: list[str], mapping: dict) -> tuple[str, float]` - Returns (mode_label, confidence)
- `create_classification_dataset(window_size_s: int, effective_hz: int, label_scheme: str, min_confidence: float = 0.0)` - Main entry point
- `get_classification_path(window_size_s: int, effective_hz: int, label_scheme: str) -> Path`
- `load_classification_dataset(window_size_s: int, effective_hz: int, label_scheme: str, split: str) -> pl.DataFrame`
- `get_class_names(label_scheme: str) -> list[str]` - Returns ordered list of class names
- `get_class_distribution(window_size_s: int, effective_hz: int, label_scheme: str, split: str) -> dict[str, int]`

### Implementation Steps

1. **Load phase 2A windows** for each split (train/val/test)
2. **Load label mapping** from `annotation-label-dictionary.csv`
3. **For each window:**
   - Map all raw annotations to labels using the chosen scheme
   - Compute mode (most frequent label)
   - Compute confidence (fraction of samples with mode label)
   - Filter if confidence < min_confidence (optional)
4. **Encode labels** to integers (alphabetical order for consistency)
5. **Save** to parquet with metadata.json containing class info
6. **Report** class distribution statistics

### Verification

```python
from opentslm.time_series_datasets.capture24 import (
    create_classification_dataset,
    load_classification_dataset,
    get_class_names,
)

# Create classification dataset with Walmsley2020 labels
create_classification_dataset(
    window_size_s=10,
    effective_hz=30,
    label_scheme="Walmsley2020",
    min_confidence=0.5  # Optional: require 50%+ agreement
)

# Load and verify
train_df = load_classification_dataset(
    window_size_s=10, effective_hz=30, label_scheme="Walmsley2020", split="train"
)

# Check schema
assert "label" in train_df.columns
assert "label_id" in train_df.columns
assert "confidence" in train_df.columns

# Check class names
classes = get_class_names("Walmsley2020")
assert classes == ["light", "moderate-vigorous", "sedentary", "sleep"]

# Verify label_id encoding
assert train_df["label_id"].max() == len(classes) - 1
```

### CLI Usage

```bash
python -m opentslm.time_series_datasets.capture24.capture24_classification \
    --window-size-s 10 \
    --effective-hz 30 \
    --label-scheme Walmsley2020 \
    --min-confidence 0.5
```

---

## QA Pairs

(Will be implemented later - requires LLM-based question generation using annotations, similar to SensorLLM from Google.)

---

# Implementation Plan: Step 3 - OpenTSLM QADataset Integration

## Goal

Create a QADataset subclass that makes Capture-24 classification data compatible with OpenTSLM Flamingo training pipeline. This enables finetuning and evaluation of the OpenTSLM model on human activity recognition from wrist-worn accelerometer data.

## Reference Files (Use as Templates)

The HAR (Human Activity Recognition) datasets are the most similar to Capture-24 and should be used as implementation templates:

| File | Purpose | Key Patterns |
|------|---------|--------------|
| `src/opentslm/time_series_datasets/QADataset.py` | **Base class** - defines the interface all datasets must implement | Abstract methods, `PromptWithAnswer` format, caching |
| `src/opentslm/time_series_datasets/har_cot/HARAccQADataset.py` | **Best template** - accelerometer activity classification | Prompt structure, time series labels, `get_labels()` |
| `src/opentslm/time_series_datasets/har_cot/har_cot_loader.py` | **Loader pattern** - loads CSV → HuggingFace Dataset | `load_har_cot_splits()` returns `(train, val, test)` |
| `src/opentslm/time_series_datasets/pamap2/PAMAP2AccQADataset.py` | **Alternative template** - similar HAR task with downsampling | Shows per-axis TextTimeSeriesPrompt creation |

## Key Architecture Understanding

### QADataset Base Class

The `QADataset` base class (`src/opentslm/time_series_datasets/QADataset.py`) requires subclasses to implement:

```python
class QADataset(Dataset, ABC):
    @abstractmethod
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Return (train, val, test) as HuggingFace Dataset objects."""
        pass

    @abstractmethod
    def _get_answer(self, row) -> str:
        """Return the label/answer string."""
        pass

    @abstractmethod
    def _get_pre_prompt(self, row) -> str:
        """Return instruction text BEFORE the time series."""
        pass

    @abstractmethod
    def _get_post_prompt(self, row) -> str:
        """Return instruction text AFTER the time series (include possible labels)."""
        pass

    @abstractmethod
    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """Return list of TextTimeSeriesPrompt objects, one per axis."""
        pass
```

### Data Flow

```
Classification Parquet (Phase 2B)
         ↓
capture24_qa_loader.py (load → HuggingFace Dataset)
         ↓
Capture24AccQADataset (format for model)
         ↓
PromptWithAnswer → Training Loop
```

## Files to Create

```
src/opentslm/time_series_datasets/capture24/
├── capture24_qa_loader.py       # NEW: Load classification parquet → HuggingFace Dataset
├── Capture24AccQADataset.py     # NEW: QADataset subclass for activity classification
├── __init__.py                  # UPDATE: Add new exports
└── test/
    └── test_capture24_qa.py     # NEW: Verification script
```

## Implementation Details

### 1. `capture24_qa_loader.py`

Loads Phase 2B classification parquet files and returns HuggingFace Dataset objects.

**Functions:**

```python
def load_capture24_classification_splits(
    window_size_s: int = 10,
    effective_hz: int = 100,
    label_scheme: str = "Walmsley2020"
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load Capture-24 classification data as HuggingFace Dataset objects.

    Returns:
        Tuple of (train, val, test) Dataset objects with schema:
        - x_axis: list[float]
        - y_axis: list[float]
        - z_axis: list[float]
        - label: str
    """
    pass

def get_label_list(label_scheme: str) -> List[str]:
    """Return alphabetically sorted list of labels for a scheme."""
    pass

def print_dataset_info(dataset: Dataset, name: str):
    """Print dataset statistics (size, label distribution)."""
    pass
```

**Key Implementation Notes:**
- Load from `data/capture24/classification/{window_size_s}s_{effective_hz}hz/{label_scheme}/`
- Rename columns: `x` → `x_axis`, `y` → `y_axis`, `z` → `z_axis` (to match HAR format)
- Handle missing splits gracefully (val may not exist in test runs)
- Convert Polars DataFrame → Pandas → HuggingFace Dataset

### 2. `Capture24AccQADataset.py`

QADataset subclass for activity classification (modeled after `HARAccQADataset.py`).

**Class Structure:**

```python
# Time series labels (one per axis)
TIME_SERIES_LABELS = [
    "The following is the accelerometer data on the x-axis",
    "The following is the accelerometer data on the y-axis",
    "The following is the accelerometer data on the z-axis",
]

class Capture24AccQADataset(QADataset):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        window_size_s: int = 10,
        effective_hz: int = 100,
        label_scheme: str = "Walmsley2020",
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        # Store config before calling super().__init__
        self.window_size_s = window_size_s
        self.effective_hz = effective_hz
        self.label_scheme = label_scheme
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        return load_capture24_classification_splits(
            self.window_size_s, self.effective_hz, self.label_scheme
        )

    def _get_answer(self, row) -> str:
        return row["label"]

    def _get_pre_prompt(self, _row) -> str:
        return "You are given accelerometer data in all three dimensions from a wrist-worn sensor. Your task is to predict the person's activity."

    def _get_post_prompt(self, _row) -> str:
        activities = ", ".join(self.get_labels())
        return f"""
Instructions:
- Analyze the accelerometer patterns to determine the activity.
- Consider movement intensity, periodicity, and axis relationships.
The following activities are possible: {activities}
- You MUST end your response with "Answer: <class label>"
"""

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        series = torch.tensor(
            [row["x_axis"], row["y_axis"], row["z_axis"]],
            dtype=torch.float32,
        )
        return [
            TextTimeSeriesPrompt(label, ts.tolist())
            for label, ts in zip(TIME_SERIES_LABELS, series)
        ]

    @staticmethod
    def get_labels() -> List[str]:
        # Return labels for the configured scheme
        # For Walmsley2020: ["light", "moderate-vigorous", "sedentary", "sleep"]
        pass
```

**Key Implementation Notes:**
- Store `window_size_s`, `effective_hz`, `label_scheme` as instance attributes BEFORE calling `super().__init__()` (required because `_load_splits` is called in parent constructor)
- Use class-level caching pattern from base class (datasets loaded once per class)
- Match prompt style from `HARAccQADataset.py`
- `get_labels()` should return the labels for the configured scheme

### 3. Update `__init__.py`

Add exports:

```python
from .capture24_qa_loader import (
    load_capture24_classification_splits,
    get_label_list,
)
from .Capture24AccQADataset import Capture24AccQADataset

__all__ = [
    # ... existing exports ...
    # QA Dataset
    "load_capture24_classification_splits",
    "get_label_list",
    "Capture24AccQADataset",
]
```

## Expected Data Schema

**Input (from Phase 2B parquet):**
```python
{
    "window_id": "P001_1476676380000",
    "pid": "P001",
    "x": [0.38, 0.39, ...],        # list[float32]
    "y": [0.48, 0.49, ...],        # list[float32]
    "z": [-0.79, -0.78, ...],      # list[float32]
    "label": "sleep",
    "label_id": 3,
    "confidence": 1.0,
}
```

**Output (HuggingFace Dataset for QADataset):**
```python
{
    "x_axis": [0.38, 0.39, ...],   # renamed from x
    "y_axis": [0.48, 0.49, ...],   # renamed from y
    "z_axis": [-0.79, -0.78, ...], # renamed from z
    "label": "sleep",
}
```

**Final format (after QADataset processing):**
```python
{
    "pre_prompt": "You are given accelerometer data...",
    "time_series": [[x_values], [y_values], [z_values]],
    "post_prompt": "Instructions: ... Answer: <class label>",
    "answer": "sleep",
}
```

## Verification

### Test Script (`test/test_capture24_qa.py`)

```python
from opentslm.time_series_datasets.capture24 import (
    load_capture24_classification_splits,
    Capture24AccQADataset,
)
from torch.utils.data import DataLoader
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

def main():
    # 1. Test loader
    train_ds, val_ds, test_ds = load_capture24_classification_splits(
        window_size_s=10, effective_hz=100, label_scheme="Walmsley2020"
    )
    print(f"Loaded: train={len(train_ds)}, val={len(val_ds) if val_ds else 0}, test={len(test_ds)}")

    # 2. Test QADataset
    dataset = Capture24AccQADataset(
        split="train",
        EOS_TOKEN="",
        window_size_s=10,
        effective_hz=100,
        label_scheme="Walmsley2020"
    )
    print(f"QADataset size: {len(dataset)}")

    # 3. Test sample format
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Answer: {sample['answer']}")

    # 4. Test DataLoader integration
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=4
        ),
    )

    for batch in dataloader:
        print(f"Batch keys: {batch[0].keys()}")
        print(f"Time series shape: {len(batch[0]['time_series'])}")
        break

    print("✓ All verification checks passed!")

if __name__ == "__main__":
    main()
```

### CLI Verification

```bash
# Run test script
python -m opentslm.time_series_datasets.capture24.test.test_capture24_qa

# Run QADataset directly (if __main__ implemented)
python -m opentslm.time_series_datasets.capture24.Capture24AccQADataset
```

## Label Scheme Mapping

For `get_labels()` method:

| Scheme | Labels (alphabetically sorted) |
|--------|-------------------------------|
| **Walmsley2020** | `["light", "moderate-vigorous", "sedentary", "sleep"]` |
| **Doherty2018** | `["moderate", "sedentary", "sleep", "tasks-light", "walking"]` |
| **Willetts2018** | `["bicycling", "mixed", "sit-stand", "sleep", "vehicle", "walking"]` |

## Integration with Training

After implementation, the dataset can be used in training:

```python
from opentslm.time_series_datasets.capture24 import Capture24AccQADataset

# Create datasets
train_dataset = Capture24AccQADataset(
    split="train",
    EOS_TOKEN=tokenizer.eos_token,
    window_size_s=10,
    effective_hz=100,
    label_scheme="Walmsley2020"
)
val_dataset = Capture24AccQADataset(split="validation", ...)
test_dataset = Capture24AccQADataset(split="test", ...)

# Use with trainer
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    ...
)
```

## Dependencies

No new dependencies required - uses existing:
- `datasets` (HuggingFace)
- `torch`
- `polars`
