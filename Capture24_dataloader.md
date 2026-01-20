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

In the phase 2A we kept all labels per window. This next phase is about converting windows into supervised tasks for our TSLM models. We will have only two types of tasks at first – first is classification, second is QA pairs.

## Classification

User can choose label to generate the dataset. The label assignment of each section is mode (most common) annotation.

## QA Pairs

(Will be done later - need to use another GPT model for example + annotations extraction used by SensorLM from Google for example.)
