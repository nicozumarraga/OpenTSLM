#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Verification script for Capture-24 classification dataset creation."""

from opentslm.time_series_datasets.capture24 import (
    create_classification_dataset,
    load_classification_dataset,
    load_classification_metadata,
    get_classification_path,
    get_class_names,
    get_class_distribution,
    load_label_mapping,
    LABEL_SCHEMES,
)


def main():
    print("=" * 60)
    print("Capture-24 Classification Dataset Verification")
    print("=" * 60)

    # Test configuration (default parameters)
    window_size_s = 10
    effective_hz = 100
    label_scheme = "Walmsley2020"
    min_confidence = 0.0

    # Step 1: Verify label schemes and mappings
    print("\n[1/6] Verifying label schemes...")
    print(f"  Available schemes: {list(LABEL_SCHEMES.keys())}")
    assert label_scheme in LABEL_SCHEMES, f"Unknown label scheme: {label_scheme}"
    print(f"  ✓ Label scheme '{label_scheme}' is valid")

    # Step 2: Test label mapping loading
    print(f"\n[2/6] Loading label mapping for {label_scheme}...")
    mapping = load_label_mapping(label_scheme)
    print(f"  ✓ Loaded {len(mapping)} annotation-to-label mappings")

    # Show sample mappings
    print("  Sample mappings:")
    for i, (annotation, label) in enumerate(list(mapping.items())[:3]):
        print(f"    '{annotation[:50]}...' -> '{label}'")

    # Step 3: Test class names
    print(f"\n[3/6] Getting class names for {label_scheme}...")
    class_names = get_class_names(label_scheme)
    print(f"  ✓ {len(class_names)} classes: {class_names}")

    # Step 4: Create classification dataset
    print(f"\n[4/6] Creating classification dataset...")
    print(f"  Window size: {window_size_s}s")
    print(f"  Effective Hz: {effective_hz}")
    print(f"  Label scheme: {label_scheme}")
    print(f"  Min confidence: {min_confidence}")

    create_classification_dataset(
        window_size_s=window_size_s,
        effective_hz=effective_hz,
        label_scheme=label_scheme,
        min_confidence=min_confidence,
        overwrite=True
    )

    # Step 5: Verify classification path and load data
    print("\n[5/6] Loading and verifying classification dataset...")
    classification_path = get_classification_path(window_size_s, effective_hz, label_scheme)
    print(f"  Classification path: {classification_path}")
    assert classification_path.exists(), f"Classification path does not exist: {classification_path}"

    for split in ["train", "val", "test"]:
        try:
            df = load_classification_dataset(window_size_s, effective_hz, label_scheme, split)
            print(f"\n  {split.upper()} split:")
            print(f"    ✓ {len(df):,} samples")
            print(f"    Columns: {df.columns}")

            if len(df) > 0:
                # Verify schema
                assert "window_id" in df.columns, "Missing window_id"
                assert "pid" in df.columns, "Missing pid"
                assert "x" in df.columns, "Missing x"
                assert "y" in df.columns, "Missing y"
                assert "z" in df.columns, "Missing z"
                assert "label" in df.columns, "Missing label"
                assert "label_id" in df.columns, "Missing label_id"
                assert "confidence" in df.columns, "Missing confidence"

                # Check sensor data shape
                first_sample = df[0]
                x_len = len(first_sample["x"][0])
                expected_len = window_size_s * effective_hz
                print(f"    Sensor samples per window: {x_len} (expected: {expected_len})")
                assert x_len == expected_len, f"X length mismatch: {x_len} != {expected_len}"

                # Check label encoding
                unique_labels = df["label"].unique().sort().to_list()
                unique_label_ids = df["label_id"].unique().sort().to_list()
                print(f"    Unique labels: {unique_labels}")
                print(f"    Label IDs: {unique_label_ids}")
                assert max(unique_label_ids) < len(class_names), "Label ID out of range"

                # Check confidence range
                min_conf = df["confidence"].min()
                max_conf = df["confidence"].max()
                print(f"    Confidence range: [{min_conf:.2f}, {max_conf:.2f}]")
                assert min_conf >= 0.0 and max_conf <= 1.0, "Confidence out of range"

        except FileNotFoundError:
            print(f"\n  {split.upper()} split: No data (expected if no windows passed filtering)")

    # Step 6: Verify metadata and class distribution
    print("\n[6/6] Verifying metadata...")
    metadata = load_classification_metadata(window_size_s, effective_hz, label_scheme)
    print(f"  ✓ Label scheme: {metadata['label_scheme']}")
    print(f"  ✓ Window size: {metadata['window_size_s']}s")
    print(f"  ✓ Effective Hz: {metadata['effective_hz']}")
    print(f"  ✓ Num classes: {metadata['num_classes']}")
    print(f"  ✓ Class names: {metadata['class_names']}")
    print(f"  ✓ Total windows processed: {metadata['total_windows_processed']:,}")
    print(f"  ✓ Windows filtered: {metadata['windows_filtered']:,}")

    # Test get_class_distribution helper
    print("\n  Class distribution:")
    for split in ["train", "val", "test"]:
        dist = get_class_distribution(window_size_s, effective_hz, label_scheme, split)
        if dist:
            total = sum(dist.values())
            print(f"    {split}:")
            for class_name in class_names:
                count = dist.get(class_name, 0)
                pct = 100 * count / total if total > 0 else 0
                print(f"      {class_name}: {count:,} ({pct:.1f}%)")

    # Log sample windows
    print("\n" + "=" * 60)
    print("SAMPLE WINDOWS")
    print("=" * 60)

    train_df = load_classification_dataset(window_size_s, effective_hz, label_scheme, "train")
    if len(train_df) >= 2:
        for i in range(2):
            sample = train_df[i]
            print(f"\n[Sample {i+1}]")
            print(f"  Window ID: {sample['window_id'][0]}")
            print(f"  Participant: {sample['pid'][0]}")
            print(f"  Time range: {sample['start_ms'][0]} - {sample['end_ms'][0]}")
            print(f"  Label: '{sample['label'][0]}' (ID: {sample['label_id'][0]})")
            print(f"  Confidence: {sample['confidence'][0]:.2%}")
            print(f"  Sensor data shape: x={len(sample['x'][0])}, y={len(sample['y'][0])}, z={len(sample['z'][0])}")
            print(f"  X values (first 5): {sample['x'][0][:5]}")
            print(f"  Y values (first 5): {sample['y'][0][:5]}")
            print(f"  Z values (first 5): {sample['z'][0][:5]}")

    print("\n" + "=" * 60)
    print("✓ All verification checks passed!")
    print("=" * 60)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Phase 2B classification dataset creation is working correctly:")
    print("  ✓ Label scheme validation")
    print("  ✓ Annotation-to-label mapping")
    print("  ✓ Class name extraction (alphabetically sorted)")
    print("  ✓ Classification dataset creation")
    print("  ✓ Schema validation (window_id, pid, x, y, z, label, label_id, confidence)")
    print("  ✓ Metadata and class distribution")
    print("  ✓ Sample logging")
    print("\nYou can now use:")
    print("  - create_classification_dataset() to create labeled datasets")
    print("  - load_classification_dataset() to load pre-created datasets")
    print("  - get_class_names() to get class labels for a scheme")
    print("  - load_classification_metadata() to get dataset metadata")
    print("  - get_class_distribution() to get class counts per split")


if __name__ == "__main__":
    main()
