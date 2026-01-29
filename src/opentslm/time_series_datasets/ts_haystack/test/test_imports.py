# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""Verify Phase 1 module imports."""

def test_imports():
    from opentslm.time_series_datasets.ts_haystack.core.data_structures import (
        BoutRecord, ParticipantTimeline, BoutRef, BoutIndex, ActivityStats
    )
    print("data_structures: OK")

    from opentslm.time_series_datasets.ts_haystack.core.seed_manager import SeedManager
    print("seed_manager: OK")

    from opentslm.time_series_datasets.ts_haystack.core.timeline_builder import TimelineBuilder
    print("timeline_builder: OK")

    from opentslm.time_series_datasets.ts_haystack.core.bout_indexer import BoutIndexer
    print("bout_indexer: OK")

    from opentslm.time_series_datasets.ts_haystack.core.transition_matrix import TransitionMatrix
    print("transition_matrix: OK")

    from opentslm.time_series_datasets.ts_haystack import (
        SeedManager, TimelineBuilder, BoutIndexer, TransitionMatrix
    )
    print("Main module imports: OK")

    print("\nAll Phase 1 modules imported successfully!")


if __name__ == "__main__":
    test_imports()
