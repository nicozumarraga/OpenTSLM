# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Core infrastructure for TS-Haystack benchmark.

This module provides the foundational components:
- Data structures for timelines, bouts, and indices
- Seed management for reproducibility
- Timeline building from Capture-24 data
- Cross-participant bout indexing
- Activity transition matrix
"""

from opentslm.time_series_datasets.ts_haystack.core.data_structures import (
    ActivityStats,
    BoutIndex,
    BoutRecord,
    BoutRef,
    DifficultyConfig,
    GeneratedSample,
    InsertedNeedle,
    ParticipantTimeline,
    TaskConfig,
)
from opentslm.time_series_datasets.ts_haystack.core.seed_manager import (
    ReproducibilityConfig,
    SeedManager,
)
from opentslm.time_series_datasets.ts_haystack.core.timeline_builder import (
    TimelineBuilder,
    get_timeline_path,
    get_timelines_dir,
)
from opentslm.time_series_datasets.ts_haystack.core.bout_indexer import (
    BoutIndexer,
    get_bout_index_path,
)
from opentslm.time_series_datasets.ts_haystack.core.transition_matrix import (
    TransitionMatrix,
    get_transition_matrix_path,
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
    # Timeline builder
    "TimelineBuilder",
    "get_timeline_path",
    "get_timelines_dir",
    # Bout indexer
    "BoutIndexer",
    "get_bout_index_path",
    # Transition matrix
    "TransitionMatrix",
    "get_transition_matrix_path",
]
