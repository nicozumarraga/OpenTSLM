# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
TS-Haystack Dataset module for OpenTSLM training integration.

This module provides QADataset implementations for training OpenTSLM models
on TS-Haystack benchmark data.

Components:
    - TSHaystackQADataset: QADataset for direct answer training
    - TSHaystackCoTQADataset: QADataset for chain-of-thought training
    - load_ts_haystack_splits: Loader function for parquet data
"""

from .ts_haystack_qa_loader import load_ts_haystack_splits, get_available_tasks
from .TSHaystackQADataset import TSHaystackQADataset
from .TSHaystackCoTQADataset import TSHaystackCoTQADataset

__all__ = [
    "TSHaystackQADataset",
    "TSHaystackCoTQADataset",
    "load_ts_haystack_splits",
    "get_available_tasks",
]
