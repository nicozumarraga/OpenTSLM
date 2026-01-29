# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
TS-Haystack: A semi-synthetic benchmark for testing retrieval and reasoning
over long time series (1K-1M+ datapoints) using Capture-24 accelerometer data.
"""

from opentslm.time_series_datasets.ts_haystack.core import (
    # Data structures
    ActivityStats,
    BoutIndex,
    BoutRecord,
    BoutRef,
    DifficultyConfig,
    GeneratedSample,
    InsertedNeedle,
    ParticipantTimeline,
    TaskConfig,
    # Seed management
    ReproducibilityConfig,
    SeedManager,
    # Timeline builder
    TimelineBuilder,
    # Bout indexer
    BoutIndexer,
    # Transition matrix
    TransitionMatrix,
)

__all__ = [
    # Data structures
    "BoutRecord",
    "ParticipantTimeline",
    "BoutRef",
    "ActivityStats",
    "BoutIndex",
    "DifficultyConfig",
    "TaskConfig",
    "InsertedNeedle",
    "GeneratedSample",
    # Seed management
    "SeedManager",
    "ReproducibilityConfig",
    # Core components
    "TimelineBuilder",
    "BoutIndexer",
    "TransitionMatrix",
]
