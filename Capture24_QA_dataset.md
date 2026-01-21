**Final Goal of this project section:** Implement a QA CoT dataset for training and evaluation on the capture24 dataset, compatible with existing datasets in the OpenTSLM repository.

---

## Current State of the Dataloading Pipeline

### Phase 1: Raw Data Extraction (COMPLETE)
**File:** `src/opentslm/time_series_datasets/capture24/capture24_loader.py`

Converts Capture-24 ZIP archive to Parquet format:
- Extracts gzipped CSV files from the ZIP archive
- Converts timestamps to milliseconds
- Optional downsampling (100Hz -> target Hz)
- Outputs: `data/capture24/sensor_data_{hz}hz/pid=P*/data.parquet`
- Also extracts `participants.parquet` and `label_mappings.parquet`

### Phase 2A: Window Extraction (COMPLETE)
**File:** `src/opentslm/time_series_datasets/capture24/capture24_windows.py`

Converts continuous sensor data into non-overlapping windows:
- Splits participants into train/val/test sets (70/15/15 by default)
- Extracts fixed-size windows (default: 10s at 100Hz = 1000 samples)
- Filters by annotation threshold (minimum % of samples with labels)
- Outputs: `data/capture24/windows/{window_size}s_{hz}hz/{split}/data.parquet`
- Schema: `window_id, pid, start_ms, end_ms, x[], y[], z[], annotations[]`

### Phase 2B: Classification Dataset (COMPLETE)
**File:** `src/opentslm/time_series_datasets/capture24/capture24_classification.py`

Converts windows with raw annotations to classification-ready format:
- Maps raw annotations to class labels using label schemes (Walmsley2020, Doherty2018, etc.)
- Computes window label via mode (most frequent label in window)
- Computes confidence score (fraction of samples with mode label)
- Optional filtering by minimum confidence threshold
- Outputs: `data/capture24/classification/{window_size}s_{hz}hz/{scheme}/{split}/data.parquet`
- Schema: `window_id, pid, start_ms, end_ms, x[], y[], z[], label, label_id, confidence`

**Available Label Schemes:**
- `Walmsley2020` (4 classes): sleep, sedentary, light, moderate-vigorous
- `Doherty2018` (7 classes): sleep, sit-stand, vehicle, walking, mixed, bicycling
- `Willetts2018` (5 classes): sleep, sit-stand, vehicle, walking, bicycling
- And more specific variants

### Phase 3A: HuggingFace Loader (COMPLETE)
**File:** `src/opentslm/time_series_datasets/capture24/capture24_qa_loader.py`

Bridges Phase 2B parquet to HuggingFace Dataset format:
- Renames columns: `x -> x_axis, y -> y_axis, z -> z_axis`
- Returns `(train, val, test)` Dataset tuple
- Schema: `x_axis[], y_axis[], z_axis[], label`

### Phase 3B: Accelerometer QA Dataset (COMPLETE)
**File:** `src/opentslm/time_series_datasets/capture24/Capture24AccQADataset.py`

QADataset subclass similar to `HARAccQADataset`:
- Inherits from `QADataset` base class
- Formats accelerometer data for OpenTSLM Flamingo training
- **Answer:** Label only (e.g., "sleep", "moderate-vigorous")
- **Prompt structure:**
  ```
  [PRE_PROMPT] You are given accelerometer data... predict the activity.
  [TIME_SERIES] x-axis data, y-axis data, z-axis data
  [POST_PROMPT] Instructions: analyze patterns... Answer: <class label>
  ```
- Supports configurable window_size, effective_hz, and label_scheme

---

## Next Phase: Implement Chain-of-Thought classification Dataset similar to HAR-CoT

**Reference:** `src/opentslm/time_series_datasets/har_cot/HARCoTQADataset.py`

### Goal
Create `Capture24CoTQADataset` that includes chain-of-thought reasoning, where the model learns to generate analysis rationale before the final answer.

We need to make a step by step plan of 1) how to generate these CoT samples for training similar to the HAR dataset and 2) store it in a way that can we accessed by other researchers like the other datasets are being stored and 3) the compatible dataloaders for OpenTSLM training.

---

## Implementation Plan for Capture24 CoT Dataset

### Configuration Decisions

| Setting | Value | Rationale |
|---------|-------|-----------|
| Label Scheme | `Walmsley2020` | 4 classes (sleep, sedentary, light, moderate-vigorous) |
| Window Size | 2.56s @ 50Hz = 128 samples | Match HAR CoT for compatibility |
| Dataset Size | Train: 10K, Val: 2K, Test: 2K | Initial proof-of-concept |
| LLM Model | `gemini-2.5-flash` | Cost-effective, high-quality rationale generation |

---

### Phase 4A: CoT Rationale Generation Pipeline

**File to create:** `src/opentslm/time_series_datasets/capture24/cot/capture24_cot_generator.py`

#### 4A.1 Dissimilarity Mapping for Capture24 (Walmsley2020 labels)

Following the HAR-CoT paper's strategy, each sample uses binary classification with the correct label and a dissimilar label:

```python
CAPTURE24_DISSIMILAR_MAPPING = {
    "sleep": ["light", "moderate-vigorous"],           # Active activities are dissimilar to sleep
    "sedentary": ["light", "moderate-vigorous"],       # Active activities are dissimilar to sedentary
    "light": ["sleep", "sedentary", "moderate-vigorous"],
    "moderate-vigorous": ["sleep", "sedentary"],       # Inactive activities are dissimilar to vigorous
}
```

#### 4A.2 Prompt Template for CoT Generation

```
You are shown a time-series plot of accelerometer over a 2.56 second window.
This data corresponds to one of two possible activities:
{CORRECT_ACTIVITY}
{DISSIMILAR_ACTIVITY}

Your task is to classify the activity based on analysis of the data.

Instructions:
- Begin by analyzing the time series without assuming a specific label.
- Think step-by-step about what the observed patterns suggest regarding movement intensity and behavior.
- Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
- Do not refer back to the plot or to the act of visual analysis in your rationale; the plot is only for reference but you should reason about the time-series data.
- Do **not** assume any answer at the beginning — analyze as if you do not yet know which class is correct.
- Do **not** mention either class label until the final sentence.
- Make sure that your last word is the answer. You MUST end your response with "Answer: {CORRECT_ACTIVITY}":
```

#### 4A.3 Generator Script Structure

```python
class Capture24CoTGenerator:
    def __init__(
        self,
        window_size_s: float = 2.56,
        effective_hz: int = 50,
        label_scheme: str = "Walmsley2020",
        api_client: str = "google",  # gemini-2.5-flash
        batch_size: int = 50,
        checkpoint_every: int = 100,
    ):
        ...

    def sample_windows(self, split: str, n_samples: int) -> pd.DataFrame:
        """Stratified sampling from classification parquet files."""
        # Load from: data/capture24/classification/2s_50hz/Walmsley2020/{split}/
        # Sample equal numbers per class
        ...

    def get_dissimilar_label(self, correct_label: str) -> str:
        """Select random dissimilar label from mapping."""
        return random.choice(CAPTURE24_DISSIMILAR_MAPPING[correct_label])

    def create_prompt(self, row: dict, dissimilar_label: str) -> str:
        """Build full prompt with time series data and binary classification task."""
        ...

    def generate_rationale(self, prompt: str) -> str:
        """Call LLM API to generate rationale."""
        ...

    def validate_rationale(self, rationale: str, expected_label: str) -> bool:
        """Verify rationale ends with 'Answer: {label}.'"""
        return rationale.strip().endswith(f"Answer: {expected_label}.")

    def generate_and_save(self, output_dir: str):
        """Main generation loop with checkpointing."""
        for split in ["train", "val", "test"]:
            samples = self.sample_windows(split, self.max_samples[split])
            results = []
            for i, row in enumerate(samples.iterrows()):
                dissimilar = self.get_dissimilar_label(row["label"])
                prompt = self.create_prompt(row, dissimilar)
                rationale = self.generate_rationale(prompt)

                if self.validate_rationale(rationale, row["label"]):
                    results.append({
                        "x_axis": json.dumps(row["x"]),
                        "y_axis": json.dumps(row["y"]),
                        "z_axis": json.dumps(row["z"]),
                        "label": row["label"],
                        "prompt": prompt,
                        "rationale": rationale,
                    })

                if i % self.checkpoint_every == 0:
                    self._save_checkpoint(results, split)

            self._save_final(results, split, output_dir)
```

#### 4A.4 Prerequisite: Generate 2.56s @ 50Hz Classification Data

Before generating CoT, we need classification data at the HAR-compatible window size:

```bash
# Run window extraction at 2.56s @ 50Hz
python -m opentslm.time_series_datasets.capture24.capture24_windows \
    --window_size_s 2.56 \
    --effective_hz 50

# Run classification with Walmsley2020 scheme
python -m opentslm.time_series_datasets.capture24.capture24_classification \
    --window_size_s 2.56 \
    --effective_hz 50 \
    --label_scheme Walmsley2020
```

---

### Phase 4B: Data Storage Format

**Output directory:** `data/capture24/cot/`

#### 4B.1 File Structure

```
data/capture24/cot/
├── capture24_cot_train.csv
├── capture24_cot_val.csv
├── capture24_cot_test.csv
└── metadata.json
```

#### 4B.2 CSV Schema (matches HAR CoT format)

| Column | Type | Description |
|--------|------|-------------|
| x_axis | string | JSON array of 128 floats |
| y_axis | string | JSON array of 128 floats |
| z_axis | string | JSON array of 128 floats |
| label | string | Correct class label |
| prompt | string | Full LLM prompt used for generation |
| rationale | string | Generated CoT ending with "Answer: {label}." |

#### 4B.3 Sample Rationale Format

```
The accelerometer data over the 2.56 second window shows relatively low variability
across all three axes with values clustering tightly around their means. The X and Y
axes exhibit minimal fluctuation, suggesting little lateral or forward-backward movement.
The Z-axis maintains steady values consistent with a stationary orientation against
gravity. There are no periodic patterns that would indicate repetitive motion like
walking or running, nor sharp transitions suggesting postural changes. The overall
stability and low amplitude of the signals indicate a period of minimal physical
activity where the body remains mostly still. Answer: sedentary.
```

#### 4B.4 metadata.json

```json
{
    "window_size_s": 2.56,
    "effective_hz": 50,
    "samples_per_window": 128,
    "label_scheme": "Walmsley2020",
    "labels": ["sleep", "sedentary", "light", "moderate-vigorous"],
    "llm_model": "gemini-2.5-flash",
    "generation_date": "2026-01-XX",
    "dissimilar_mapping": {
        "sleep": ["light", "moderate-vigorous"],
        "sedentary": ["light", "moderate-vigorous"],
        "light": ["sleep", "sedentary", "moderate-vigorous"],
        "moderate-vigorous": ["sleep", "sedentary"]
    },
    "samples": {"train": 10000, "val": 2000, "test": 2000},
    "class_distribution": {
        "train": {"sleep": 2500, "sedentary": 2500, "light": 2500, "moderate-vigorous": 2500}
    }
}
```

---

### Phase 4C: DataLoader Implementation

#### 4C.1 Loader Module

**File:** `src/opentslm/time_series_datasets/capture24/cot/capture24_cot_loader.py`

```python
import ast
import os
import pandas as pd
from datasets import Dataset
from typing import Tuple

CAPTURE24_COT_DATA_DIR = os.path.join("data", "capture24_cot")

def parse_time_series(series_str: str) -> list:
    """Parse JSON array string to list of floats."""
    return ast.literal_eval(series_str)

def load_capture24_cot_csv(csv_path: str) -> pd.DataFrame:
    """Load and preprocess a CoT CSV file."""
    df = pd.read_csv(csv_path)
    # Filter out any duplicate header rows
    df = df[df["x_axis"] != "x_axis"]
    # Parse time series columns
    df["x_axis"] = df["x_axis"].apply(parse_time_series)
    df["y_axis"] = df["y_axis"].apply(parse_time_series)
    df["z_axis"] = df["z_axis"].apply(parse_time_series)
    return df

def load_capture24_cot_splits() -> Tuple[Dataset, Dataset, Dataset]:
    """Load train/val/test splits as HuggingFace Datasets."""
    train_df = load_capture24_cot_csv(os.path.join(CAPTURE24_COT_DATA_DIR, "capture24_cot_train.csv"))
    val_df = load_capture24_cot_csv(os.path.join(CAPTURE24_COT_DATA_DIR, "capture24_cot_val.csv"))
    test_df = load_capture24_cot_csv(os.path.join(CAPTURE24_COT_DATA_DIR, "capture24_cot_test.csv"))
    return (
        Dataset.from_pandas(train_df),
        Dataset.from_pandas(val_df),
        Dataset.from_pandas(test_df),
    )
```

#### 4C.2 QADataset Class

**File:** `src/opentslm/time_series_datasets/capture24/cot/Capture24CoTQADataset.py`

```python
from typing import List, Literal, Tuple
import torch
from datasets import Dataset
from opentslm.time_series_datasets.QADataset import QADataset, TextTimeSeriesPrompt
from .capture24_cot_loader import load_capture24_cot_splits

TIME_SERIES_LABELS = [
    "The following is the accelerometer data on the x-axis",
    "The following is the accelerometer data on the y-axis",
    "The following is the accelerometer data on the z-axis",
]

class Capture24CoTQADataset(QADataset):
    """Chain-of-Thought QA Dataset for Capture24 accelerometer data."""

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        return load_capture24_cot_splits()

    def _get_answer(self, row) -> str:
        """Return full rationale as the answer (CoT)."""
        return row["rationale"]

    def _get_pre_prompt(self, _row) -> str:
        return """You are given accelerometer data from a wrist-worn sensor.
Your task is to classify the activity and explain your reasoning.

Possible activity labels: sleep, sedentary, light, moderate-vigorous

Instructions:
- Analyze the time series patterns
- Think step-by-step about movement intensity
- Write rationale as a single paragraph
- End with "Answer: <label>"
"""

    def _get_post_prompt(self, _row) -> str:
        return "Rationale:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """Convert time series to normalized TextTimeSeriesPrompt objects."""
        series = torch.tensor(
            [row["x_axis"], row["y_axis"], row["z_axis"]],
            dtype=torch.float32
        )

        # Z-score normalization
        means = series.mean(dim=1, keepdim=True)
        stds = series.std(dim=1, keepdim=True).clamp(min=1e-6)
        series_norm = (series - means) / stds

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
        return ["sleep", "sedentary", "light", "moderate-vigorous"]

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["label"] = row["label"]
        sample["x_axis"] = row["x_axis"]
        sample["y_axis"] = row["y_axis"]
        sample["z_axis"] = row["z_axis"]
        return sample
```

#### 4C.3 Module Init

**File:** `src/opentslm/time_series_datasets/capture24/cot/__init__.py`

```python
from .Capture24CoTQADataset import Capture24CoTQADataset
from .cot.capture24_cot_loader import load_capture24_cot_splits

__all__ = ["Capture24CoTQADataset", "load_capture24_cot_splits"]
```

---

### Phase 4D: Files to Create Summary

| File | Purpose |
|------|---------|
| `src/opentslm/time_series_datasets/capture24/cot/__init__.py` | Module exports |
| `src/opentslm/time_series_datasets/capture24/cot/capture24_cot_generator.py` | Gemini 2.5 Flash rationale generation script |
| `src/opentslm/time_series_datasets/capture24/cot/capture24_cot_loader.py` | CSV loader for HuggingFace |
| `src/opentslm/time_series_datasets/capture24/cot/Capture24CoTQADataset.py` | QADataset subclass |
| `src/opentslm/time_series_datasets/capture24/cot/test/test_capture24_cot.py` | Unit tests |
| `data/capture24/cot/capture24_cot_{train,val,test}.csv` | Generated CoT data |
| `data/capture24/cot/metadata.json` | Generation parameters |

**Note:** All CoT-related code lives inside the existing `capture24/` module as a `cot/` subfolder, keeping the codebase organized.

---

### Phase 4E: Testing & Validation

**File:** `src/opentslm/time_series_datasets/capture24/cot/test/test_capture24_cot.py`

```python
def test_loader_parses_csv():
    """Test CSV loading and parsing."""
    train_ds, val_ds, test_ds = load_capture24_cot_splits()
    assert len(train_ds) > 0
    sample = train_ds[0]
    assert all(k in sample for k in ["x_axis", "y_axis", "z_axis", "label", "rationale"])

def test_time_series_length():
    """Verify 128 samples per axis (2.56s @ 50Hz)."""
    train_ds, _, _ = load_capture24_cot_splits()
    assert len(train_ds[0]["x_axis"]) == 128

def test_rationale_format():
    """All rationales end with 'Answer: <label>.'"""
    train_ds, _, _ = load_capture24_cot_splits()
    labels = ["sleep", "sedentary", "light", "moderate-vigorous"]
    for sample in train_ds:
        assert any(f"Answer: {l}." in sample["rationale"] for l in labels)

def test_qa_dataset_integration():
    """Test with QADataset base class."""
    dataset = Capture24CoTQADataset(split="train", EOS_TOKEN="</s>")
    assert len(dataset) > 0
    sample = dataset[0]
    assert "answer" in sample
    assert "Answer:" in sample["answer"]

def test_dataloader_batching():
    """Test with PyTorch DataLoader."""
    from torch.utils.data import DataLoader
    dataset = Capture24CoTQADataset(split="test", EOS_TOKEN="</s>")
    loader = DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))
    assert len(batch["answer"]) == 4
```

---

### Implementation Order

1. **Prerequisite:** Generate 2.56s @ 50Hz classification data (may need to add support for fractional window sizes)
2. Create `capture24/cot/` directory structure with `__init__.py`
3. Implement `capture24_cot_generator.py` with dissimilarity mapping
4. Generate small test batch (100 samples) to validate pipeline
5. Implement `capture24_cot_loader.py`
6. Implement `Capture24CoTQADataset.py`
7. Write unit tests
8. Generate full dataset
9. Run test suite and validate integration

note: update model to gemini-2.5-flash-lite
