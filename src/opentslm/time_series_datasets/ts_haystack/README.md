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
│   ├── background_sampler.py  # Sample background windows (Phase 2)
│   ├── needle_sampler.py      # Sample needles from bout index (Phase 2)
│   ├── style_transfer.py      # Covariance projection + blending (Phase 2)
│   └── prompt_templates.py    # NL templates for Q/A diversity (Phase 2)
├── scripts/
│   └── build_core_artifacts.py  # CLI to build timelines, index, matrix
└── test/
    └── test_imports.py        # Verify module imports
```

## Quick Start

Build Core artifacts (requires Capture24 data extracted):

```bash
python -m opentslm.time_series_datasets.ts_haystack.scripts.build_core_artifacts \
    --n-jobs 8 \
    --label-scheme WillettsSpecific2018
```

## Output

Artifacts are saved to `data/capture24/ts_haystack/`:
- `timelines/P*.parquet` — Per-participant activity timelines
- `bout_index.parquet` — Cross-participant bout index
- `transition_matrix.json` — Activity transition probabilities

## Status

- [x] Phase 1: Core infrastructure (timelines, bout index, transition matrix)
- [x] Phase 2: Sampling & style transfer (background/needle samplers, style transfer, prompts)
- [ ] Phase 3: Task generators (8 tasks: existence, localization, counting, etc.)
- [ ] Phase 4: QADataset integration

## Phase 2 Usage

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
