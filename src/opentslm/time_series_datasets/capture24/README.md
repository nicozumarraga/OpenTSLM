# Capture-24 Dataset

## Dataset Overview

- 151 participants with ~7 days of wrist-worn accelerometer data each
- 100Hz sampling rate = ~60 million rows per participant
- Total: ~9 billion rows of sensor data
- Files in ZIP: PXXX.csv.gz (one per participant), metadata.csv, annotation-label-dictionary.csv

## Getting Started

### 1. Download the Dataset

Download `capture24.zip` from the Oxford Research Archive:
https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/files/rpr76f381b

### 2. Place the ZIP File

Place the downloaded `capture24.zip` in the `data/` directory at the project root:

```
OpenTSLM/
├── data/
│   └── capture24.zip    <-- place here
├── src/
└── ...
```

### 3. Extract and Convert to Parquet

Run the loader to extract the ZIP and convert CSV files to Parquet format:

```bash
# Basic extraction (100Hz, single-threaded)
python -m opentslm.time_series_datasets.capture24.capture24_loader

# With parallel processing (recommended for faster extraction)
python -m opentslm.time_series_datasets.capture24.capture24_loader --n-jobs 4

# Downsample to 25Hz during extraction
python -m opentslm.time_series_datasets.capture24.capture24_loader --downsample-hz 25 --n-jobs 4

# Test with a few participants first
python -m opentslm.time_series_datasets.capture24.capture24_loader --max-participants 3

# Submit as SLURM job (recommended if working on a computing cluster -- adjust partition to your needs)
sbatch src/opentslm/time_series_datasets/capture24/extract_data.sbatch
```

This creates:
```
data/capture24/
├── participants.parquet         # Participant metadata
├── label_mappings.parquet       # Annotation-to-label mappings
└── sensor_data_{hz}hz/          # Sensor data directory
    ├── pid=P001/data.parquet
    ├── pid=P002/data.parquet
    └── ...
```

### 4. Extract Windows

After extraction, split the data into non-overlapping windows for training:

```bash
# Basic window extraction (10s windows at 100Hz)
python -m opentslm.time_series_datasets.capture24.capture24_windows

# 10s windows downsampled to 25Hz with parallel processing
python -m opentslm.time_series_datasets.capture24.capture24_windows \
    --window-size-s 10 \
    --source-hz 100 \
    --downsample-hz 25 \
    --n-jobs 4

# Custom train/val/test split ratios
python -m opentslm.time_series_datasets.capture24.capture24_windows \
    --train-ratio 0.8 \
    --val-ratio 0.1

# Test with a few participants
python -m opentslm.time_series_datasets.capture24.capture24_windows --max-participants 5
```

This creates:
```
data/capture24/windows/{window_size}s_{hz}hz/
├── train/data.parquet
├── val/data.parquet
└── test/data.parquet
```

## Programmatic Usage

### Loading Raw Sensor Data

```python
from opentslm.time_series_datasets.capture24.capture24_loader import (
    ensure_capture24_data,
    load_participants,
    load_label_mappings,
    load_participant_sensor_data,
)

# Ensure data is extracted (no-op if already done)
ensure_capture24_data(downsample_hz=100)

# Load metadata
participants = load_participants()
label_mappings = load_label_mappings()

# Load sensor data for a specific participant
df = load_participant_sensor_data("P001", downsample_hz=100)
# Columns: timestamp_ms, x, y, z, annotation
```

### Loading Pre-extracted Windows

```python
from opentslm.time_series_datasets.capture24.capture24_windows import (
    extract_windows,
    load_windows,
)

# Extract windows (no-op if already done)
extract_windows(window_size_s=10, source_hz=100, downsample_hz=25)

# Load windows for a split
train_windows = load_windows(window_size_s=10, effective_hz=25, split="train")
val_windows = load_windows(window_size_s=10, effective_hz=25, split="val")
test_windows = load_windows(window_size_s=10, effective_hz=25, split="test")

# Window schema: window_id, pid, start_ms, end_ms, x, y, z, annotations
```

## Command-Line Options

### capture24_loader.py

| Option | Default | Description |
|--------|---------|-------------|
| `--max-participants, -n` | None | Limit participants (for testing) |
| `--downsample-hz, -d` | 100 | Target sampling frequency |
| `--n-jobs, -j` | 1 | Parallel jobs |
| `--overwrite` | False | Force re-extraction |

### capture24_windows.py

| Option | Default | Description |
|--------|---------|-------------|
| `--window-size-s, -w` | 10 | Window size in seconds |
| `--source-hz` | 100 | Source data frequency |
| `--downsample-hz, -d` | None | Target frequency |
| `--annotation-threshold, -a` | 0.6 | Min annotation coverage |
| `--seed, -s` | 42 | Random seed for splits |
| `--train-ratio, -t` | 0.7 | Training set fraction |
| `--val-ratio, -v` | 0.15 | Validation set fraction |
| `--n-jobs, -j` | 1 | Parallel jobs |
| `--max-participants, -n` | None | Limit participants |
| `--overwrite` | False | Force re-extraction |

## Label Scheme Summary

| Scheme | Unique Labels | Description |
|--------|---------------|-------------|
| WillettsSpecific2018 | 12 | Most granular activity types |
| WillettsMET2018 | 10 | Activity + intensity combinations |
| DohertySpecific2018 | 11 | Screen/task-based distinction |
| Willetts2018 | 6 | Simplified posture-based |
| Doherty2018 | 6 | Intensity-based |
| Walmsley2020 | 5 | Most simplified |

### Labels per Scheme

**WillettsSpecific2018 (12 labels):**
sleep, sitting, standing, walking, vehicle, bicycling, sports, household-chores, mixed-activity, manual-work

**WillettsMET2018 (10 labels):**
sleep, sitting, sitstand+lowactivity, sitstand+activity, walking, walking+activity, vehicle, bicycling, sports, gym

**DohertySpecific2018 (11 labels):**
sleep, sedentary-screen, sedentary-non-screen, tasks-light, tasks-moderate, walking, vehicle, bicycling, sports-continuous, sport-interrupted

**Willetts2018 (6 labels):**
sleep, sit-stand, mixed, walking, vehicle, bicycling

**Doherty2018 (6 labels):**
sleep, sedentary, tasks-light, moderate, walking

**Walmsley2020 (5 labels):**
sleep, sedentary, light, moderate-vigorous
