# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Tests for TS-Haystack benchmark.

Test modules:
- test_style_transfer.py: StyleTransfer unit tests (synthetic data)
- test_prompt_templates.py: PromptTemplateBank unit tests (no data needed)
- test_needle_sampler.py: NeedleSampler integration tests (requires data)
- test_background_sampler.py: BackgroundSampler integration tests (requires data)
- test_phase2_integration.py: Full pipeline integration tests (requires data)

Running tests:

    # Run all tests (some will skip if data unavailable)
    pytest src/opentslm/time_series_datasets/ts_haystack/test/ -v

    # Run only unit tests (no data required)
    pytest src/opentslm/time_series_datasets/ts_haystack/test/test_style_transfer.py -v
    pytest src/opentslm/time_series_datasets/ts_haystack/test/test_prompt_templates.py -v

    # Run with plot generation
    pytest src/opentslm/time_series_datasets/ts_haystack/test/ -v -k "visualization"

    # Run specific test class
    pytest src/opentslm/time_series_datasets/ts_haystack/test/test_style_transfer.py::TestVisualization -v

Plots are saved to:
    src/opentslm/time_series_datasets/ts_haystack/test/plots/

Prerequisites for integration tests:
1. Capture24 sensor data extracted (run capture24_loader.py)
2. Phase 1 artifacts built (run build_phase1_artifacts.py)
"""
