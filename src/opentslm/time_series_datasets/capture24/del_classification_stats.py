# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Simple script to print statistics for a Capture-24 classification dataset.

Usage:
    python -m opentslm.time_series_datasets.capture24.classification_stats --window-size-s 2.56 --effective-hz 50 --label-scheme Walmsley2020
"""

import argparse

from opentslm.time_series_datasets.capture24.capture24_classification import (
    LABEL_SCHEMES,
    get_classification_path,
    load_classification_dataset,
    load_classification_metadata,
)


def print_stats(
    window_size_s: float,
    effective_hz: int,
    label_scheme: str,
) -> None:
    """Print statistics for a classification dataset."""

    dataset_path = get_classification_path(window_size_s, effective_hz, label_scheme)

    print("=" * 70)
    print("Capture-24 Classification Dataset Statistics")
    print("=" * 70)
    print()

    # Dataset configuration
    print("Configuration:")
    print(f"  Path: {dataset_path}")
    print(f"  Window size: {window_size_s}s")
    print(f"  Effective Hz: {effective_hz}")
    print(f"  Samples per window: {int(window_size_s * effective_hz)}")
    print(f"  Label scheme: {label_scheme}")
    print()

    # Load metadata
    try:
        metadata = load_classification_metadata(window_size_s, effective_hz, label_scheme)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    print(f"Classes ({metadata['num_classes']}): {metadata['class_names']}")
    print(f"Min confidence threshold: {metadata['min_confidence']:.0%}")
    print()

    # Load each split and compute stats
    total_samples = 0
    total_participants = set()

    print("-" * 70)
    print(f"{'Split':<12} {'Samples':>10} {'Participants':>14} {'Avg Conf':>10}")
    print("-" * 70)

    split_data = {}
    for split in ["train", "val", "test"]:
        try:
            df = load_classification_dataset(window_size_s, effective_hz, label_scheme, split)
            n_samples = len(df)
            participants = df["pid"].unique().to_list()
            n_participants = len(participants)
            avg_confidence = df["confidence"].mean()

            total_samples += n_samples
            total_participants.update(participants)
            split_data[split] = df

            print(f"{split:<12} {n_samples:>10,} {n_participants:>14} {avg_confidence:>10.2%}")
        except FileNotFoundError:
            print(f"{split:<12} {'N/A':>10} {'N/A':>14} {'N/A':>10}")

    print("-" * 70)
    print(f"{'TOTAL':<12} {total_samples:>10,} {len(total_participants):>14}")
    print()

    # Window length verification
    if split_data:
        sample_df = next(iter(split_data.values()))
        if len(sample_df) > 0:
            x_len = len(sample_df["x"][0])
            y_len = len(sample_df["y"][0])
            z_len = len(sample_df["z"][0])
            print(f"Sensor data shape: x={x_len}, y={y_len}, z={z_len} samples/axis")
            print()

    # Label distribution per split
    print("=" * 70)
    print("Label Distribution")
    print("=" * 70)

    class_names = metadata["class_names"]

    # Header
    header = f"{'Label':<20}"
    for split in ["train", "val", "test"]:
        header += f" {split:>12}"
    header += f" {'Total':>12}"
    print(header)
    print("-" * 70)

    # Per-class counts
    class_totals = {c: 0 for c in class_names}
    for class_name in class_names:
        row = f"{class_name:<20}"
        for split in ["train", "val", "test"]:
            if split in split_data:
                count = (split_data[split]["label"] == class_name).sum()
                class_totals[class_name] += count
                row += f" {count:>12,}"
            else:
                row += f" {'N/A':>12}"
        row += f" {class_totals[class_name]:>12,}"
        print(row)

    print("-" * 70)

    # Print percentages
    print()
    print("Label Distribution (%)")
    print("-" * 70)

    header = f"{'Label':<20}"
    for split in ["train", "val", "test"]:
        header += f" {split:>12}"
    print(header)
    print("-" * 70)

    for class_name in class_names:
        row = f"{class_name:<20}"
        for split in ["train", "val", "test"]:
            if split in split_data:
                df = split_data[split]
                count = (df["label"] == class_name).sum()
                pct = 100 * count / len(df) if len(df) > 0 else 0
                row += f" {pct:>11.1f}%"
            else:
                row += f" {'N/A':>12}"
        print(row)

    print("-" * 70)
    print()

    # Participant distribution
    print("=" * 70)
    print("Participant Distribution")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        if split in split_data:
            df = split_data[split]
            participants = df["pid"].unique().sort().to_list()
            windows_per_pid = df.group_by("pid").len()
            min_windows = windows_per_pid["len"].min()
            max_windows = windows_per_pid["len"].max()
            avg_windows = windows_per_pid["len"].mean()

            print(f"{split}:")
            print(f"  Participants: {len(participants)}")
            print(f"  Windows per participant: min={min_windows:,}, max={max_windows:,}, avg={avg_windows:,.0f}")
            if len(participants) <= 20:
                print(f"  PIDs: {participants}")
            else:
                print(f"  PIDs: {participants[:5]} ... {participants[-5:]}")
            print()

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print statistics for a Capture-24 classification dataset"
    )
    parser.add_argument(
        '--window-size-s', '-w',
        type=float,
        default=2.56,
        help='Window size in seconds (default: 2.56)'
    )
    parser.add_argument(
        '--effective-hz', '-e',
        type=int,
        default=50,
        help='Effective sampling frequency in Hz (default: 50)'
    )
    parser.add_argument(
        '--label-scheme', '-l',
        type=str,
        default="Walmsley2020",
        choices=list(LABEL_SCHEMES.keys()),
        help='Label scheme (default: Walmsley2020)'
    )

    args = parser.parse_args()

    print_stats(
        window_size_s=args.window_size_s,
        effective_hz=args.effective_hz,
        label_scheme=args.label_scheme,
    )