# TS-Haystack

A semi-synthetic benchmark for testing retrieval and reasoning over long time series (1K–1M+ datapoints) using Capture-24 accelerometer data.

See `TS_HAYSTACK_IMPLEMENTATION_PLAN.md` in project root for full design.

## Structure

```
ts_haystack/
├── core/                    # Phase 1: Core infrastructure
│   ├── data_structures.py   # Dataclasses (BoutRecord, ParticipantTimeline, BoutIndex, etc.)
│   ├── seed_manager.py      # Reproducibility & deterministic seeds
│   ├── timeline_builder.py  # Extract activity bouts from Capture24
│   ├── bout_indexer.py      # Cross-participant bout index
│   └── transition_matrix.py # Activity transition probabilities
├── scripts/
│   └── build_phase1_artifacts.py  # CLI to build timelines, index, matrix
└── test/
    └── test_imports.py      # Verify module imports
```

## Quick Start

Build Phase 1 artifacts (requires Capture24 data extracted):

```bash
python -m opentslm.time_series_datasets.ts_haystack.scripts.build_phase1_artifacts \
    --n-jobs 8 \
    --label-scheme WillettsSpecific2018
```

## Output

Artifacts are saved to `data/capture24/ts_haystack/`:
- `timelines/P*.parquet` — Per-participant activity timelines
- `bout_index.parquet` — Cross-participant bout index
- `transition_matrix.json` — Activity transition probabilities

## Status

- [x] Phase 1: Core infrastructure
- [ ] Phase 2: Sampling & style transfer
- [ ] Phase 3: Task generators
- [ ] Phase 4: QADataset integration
