# OpenTSLM evaluation on Capture24

**Goal:** Evaluate how the PerceiverResampler's fixed-size compression bottleneck affects classification performance as context length increases.

**Hypothesis:** As window length L increases, the compression ratio (L/p)/N_latent grows linearly, causing degradation in classification accuracy.

# Expected Results

Plot the different Accuracy and F1 results on the Y axis and Context Length in the X axis.

# Experiments to Run

Window lengths (seconds): [2.56, 10, 30, 60, 300, 900, 1800, 3600]

At 50Hz sampling, this translates to: [128, 500, 1500, 3000, 15000, 45000, 90000, 180000] datapoints

Approach: Use Zero-shot (HAR-CoT checkpoint on direct classification, no CoT generated at each timestep)

Model checkpoint: Llama3.2-1B (~21GB)

### Sampling
- Target: 250 samples per class per window length (1000 total)
- Stratified by class, fixed seed (42)
- Fallback: If class has < 250 samples, use all available and document

### Metrics
- Macro F1 (primary)
- Accuracy (secondary)
- Per-class F1 (diagnostic)

# Matching labels from HAR dataset to Capture24

Capture 24 has the following labels:

The closest thing in Capture24 is the `label:WillettsSpecific2018`. Classes (10): ['bicycling', 'household-chores', 'manual-work', 'mixed-activity', 'sitting', 'sleep', 'sports', 'standing', 'vehicle', 'walking']

In the HAR dataset the classes are ["biking", "lying", "running", "sitting", "standing", "walking", "walking_down", "walking_up"]

We can use this dataset at this path: @data/capture24/classification/2_56s_50hz/WillettsSpecific2018/test , which has this metadata @data/capture24/classification/2_56s_50hz/WillettsSpecific2018/metadata.json

Hence we will do a Subset Evaluation using only the 4 directly matching classes.

label_map = {
    "sitting": "sitting",
    "standing": "standing",
    "walking": "walking",
    "bicycling": "biking",
}

**Note**: Prompt includes all 8 HAR-CoT classes to match training distribution.

# Implementation Plan

### Phase 0: Feasibility Check
1. Download HAR-CoT Flamingo checkpoint from HuggingFace
2. Extract config: `patch_size`, `N_latent`, model size
3. Verify window lengths are divisible by `patch_size`


### Phase 0: Results

@scripts/phase0_feasibility_check_results.txt ✅

### Phase 1: Dataset Preparation
1. Create Capture24 datasets for all window lengths
   - Reference: src/opentslm/time_series_datasets/capture24/README.md
   - Ensure 50Hz sampling during window creation, 3-axis accelerometer
2. Filter to 4-class subset
3. Sample 250 per class (or max available) with seed=42
4. Verify normalization matches OpenTSLM expectations ([-1, 1])

@data/capture24/eval_context_length/summary.json ✅

### Phase 2: Validation Run
1. Run inference on 2.56s dataset (matches HAR-CoT training)
2. Compare to expected HAR-CoT performance (~65% F1)
3. Debug any prompt/data format misalignments
4. Run memory/timing pilot (10 samples at 2.56s, 900s, 3600s)

_[PHASE 2 and 3 CURRENTLY RUNNING ON GPU]_

### Phase 3: Full Evaluation
1. Run classification eval across all window lengths
2. Log: predictions, ground truth, inference time, memory usage
3. Generate plots: F1 vs window length, accuracy vs window length

_[PHASE 2 and 3 CURRENTLY RUNNING ON GPU]_

### Phase 4: Analysis
1. Identify "knee" where performance drops sharply
2. Calculate correlation between compression ratio and F1
3. Document failure modes (confusion matrices per window length)

# Relevant files

@src/opentslm/time_series_datasets/har_cot/HARAccQADataset.py
- Checkpoint loading: demo/huggingface/03_test_hf_har_cot.py
- Evaluation logic: evaluation/baseline/evaluate_ecg_qa.py
- Dataset format: src/opentslm/time_series_datasets/har_cot/HARAccQADataset.py
- Capture24 creation: src/opentslm/time_series_datasets/capture24/README.md
