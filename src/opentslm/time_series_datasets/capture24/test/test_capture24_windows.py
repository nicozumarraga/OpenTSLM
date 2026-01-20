#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Verification script for Capture-24 window extraction."""

from opentslm.time_series_datasets.capture24 import (
    extract_windows,
    load_windows,
    split_participants,
    get_windows_path,
)


def main():
    print("=" * 60)
    print("Capture-24 Window Extraction Verification")
    print("=" * 60)

    # Test configuration
    window_size_s = 10
    source_hz = 10  # Use 100Hz source data
    downsample_hz = None  # No downsampling
    annotation_threshold = 0.6

    # Step 1: Check participant split
    print("\n[1/5] Testing participant split...")
    train_pids, val_pids, test_pids = split_participants(seed=42, source_hz=source_hz)
    print(f"  ✓ Train: {len(train_pids)} participants")
    print(f"  ✓ Val: {len(val_pids)} participants")
    print(f"  ✓ Test: {len(test_pids)} participants")
    print(f"  Total: {len(train_pids) + len(val_pids) + len(test_pids)} participants")

    # Verify split is deterministic
    train_pids2, val_pids2, test_pids2 = split_participants(seed=42, source_hz=source_hz)
    assert train_pids == train_pids2, "Split is not deterministic!"
    print("  ✓ Split is deterministic (same seed produces same split)")

    # Step 2: Extract windows (limited for testing)
    hz_display = f"{downsample_hz}Hz" if downsample_hz else "(no downsampling)"
    print(f"\n[2/5] Extracting windows ({window_size_s}s @ {hz_display})...")
    print(f"  Annotation threshold: {annotation_threshold:.0%}")

    extract_windows(
        window_size_s=window_size_s,
        source_hz=source_hz,
        downsample_hz=downsample_hz,
        annotation_threshold=annotation_threshold,
        seed=42,
        n_jobs=1,
        max_participants=3,
        overwrite=True
    )

    # Step 3: Verify window path
    print("\n[3/5] Verifying window path...")
    effective_hz = downsample_hz if downsample_hz is not None else source_hz
    windows_path = get_windows_path(window_size_s, effective_hz)
    print(f"  ✓ Windows path: {windows_path}")
    assert windows_path.exists(), f"Windows path does not exist: {windows_path}"

    # Step 4: Load and verify windows
    print("\n[4/5] Loading and verifying windows...")
    for split in ["train", "val", "test"]:
        try:
            windows = load_windows(window_size_s, effective_hz, split)
            print(f"\n  {split.upper()} split:")
            print(f"    ✓ {len(windows):,} windows")
            print(f"    Columns: {windows.columns}")

            if len(windows) > 0:
                # Check first window
                first_window = windows[0]
                x_len = len(first_window["x"][0])
                y_len = len(first_window["y"][0])
                z_len = len(first_window["z"][0])
                ann_len = len(first_window["annotations"][0])

                expected_len = window_size_s * effective_hz
                print(f"    Sample window shape: x={x_len}, y={y_len}, z={z_len}, annotations={ann_len}")
                print(f"    Expected length: {expected_len}")

                assert x_len == expected_len, f"X length mismatch: {x_len} != {expected_len}"
                assert y_len == expected_len, f"Y length mismatch: {y_len} != {expected_len}"
                assert z_len == expected_len, f"Z length mismatch: {z_len} != {expected_len}"
                assert ann_len == expected_len, f"Annotations length mismatch: {ann_len} != {expected_len}"

                # Check annotation threshold
                sample_annotations = first_window["annotations"][0]
                non_null_count = sum(1 for a in sample_annotations if a is not None)
                annotation_ratio = non_null_count / len(sample_annotations)
                print(f"    Annotation ratio: {annotation_ratio:.2%} (threshold: {annotation_threshold:.0%})")
                assert annotation_ratio >= annotation_threshold, \
                    f"Annotation ratio {annotation_ratio:.2%} below threshold {annotation_threshold:.0%}"

                # Check window metadata
                print(f"    Window ID: {first_window['window_id'][0]}")
                print(f"    Participant: {first_window['pid'][0]}")
                print(f"    Start timestamp: {first_window['start_ms'][0]}")
                print(f"    End timestamp: {first_window['end_ms'][0]}")

        except FileNotFoundError:
            print(f"\n  {split.upper()} split: No windows (expected if no participants in this split)")

    # Step 5: Test with downsampling
    downsample_target_hz = int(source_hz / 2)
    print(f"\n[5/5] Testing window extraction with downsampling to {downsample_target_hz}Hz...")
    extract_windows(
        window_size_s=5,
        source_hz=source_hz,
        downsample_hz=downsample_target_hz,
        annotation_threshold=0.8,
        seed=42,
        n_jobs=1,
        max_participants=2,
        overwrite=True
    )

    windows_downsampled = load_windows(window_size_s=5, effective_hz=downsample_target_hz, split="train")
    if len(windows_downsampled) > 0:
        sample_len = len(windows_downsampled["x"][0])
        expected_len = 5 * downsample_target_hz  # 5s at downsampled rate
        print(f"  ✓ Downsampled to {downsample_target_hz}Hz: {sample_len} samples (expected {expected_len})")
        assert sample_len == expected_len, f"Sample length mismatch: {sample_len} != {expected_len}"

    print("\n" + "=" * 60)
    print("✓ All verification checks passed!")
    print("=" * 60)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Phase 2A window extraction is working correctly:")
    print("  ✓ Participant splitting (train/val/test)")
    print(f"  ✓ Window extraction at source frequency ({source_hz}Hz)")
    print(f"  ✓ Window extraction with downsampling ({int(source_hz/2)}Hz)")
    print("  ✓ Annotation threshold filtering")
    print("  ✓ Window schema validation")
    print("  ✓ Metadata preservation")
    print("  ✓ Hz-specific directory structure")
    print("\nYou can now use:")
    print("  - extract_windows(source_hz=..., downsample_hz=...) to create window datasets")
    print("  - load_windows(effective_hz=...) to load pre-extracted windows")
    print("  - split_participants(source_hz=...) to get train/val/test splits")


if __name__ == "__main__":
    main()
