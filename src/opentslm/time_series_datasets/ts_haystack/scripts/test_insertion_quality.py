# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Classifier-based insertion quality test for TS-Haystack.

Tests whether style transfer + cosine blending produces clean insertions
by training a classifier to distinguish inserted vs non-inserted samples.

Design:
- Class 0: Pure background (no insertion)
- Class 1: Pure background + same-activity needle from different participant

Using same-activity needles isolates insertion artifacts from activity differences.
If the classifier can distinguish classes, it's detecting blending imperfections.

Expected Results:
- AUC ≈ 0.50 → insertions are undetectable (good)
- AUC >> 0.50 → classifier detects artifacts (needs investigation)

Usage:
    python -m opentslm.time_series_datasets.ts_haystack.scripts.test_insertion_quality \
        --n-train 5000 --n-test 500 --context-seconds 3.0 --seed 42
"""

import argparse
import sys
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: xgboost is required. Install with: pip install xgboost")
    sys.exit(1)

from opentslm.time_series_datasets.ts_haystack.core import (
    BackgroundSampler,
    BoutIndexer,
    NeedleSampler,
    StyleTransfer,
    TimelineBuilder,
)


# ==============================================================================
# Configuration
# ==============================================================================

SOURCE_HZ = 100  # Capture-24 sampling rate
NEEDLE_LENGTH_RATIO_MIN = 0.02  # 2% of context
NEEDLE_LENGTH_RATIO_MAX = 0.08  # 8% of context
MARGIN_RATIO = 0.05  # 5% margin from edges


# ==============================================================================
# Sample Generation
# ==============================================================================


def generate_samples(
    n_samples: int,
    context_length_samples: int,
    seed: int,
    positive_ratio: float = 0.5,
    verbose: bool = True,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[int]]:
    """
    Generate samples with/without same-activity needle insertion.

    Args:
        n_samples: Number of samples to generate
        context_length_samples: Window size in samples
        seed: Random seed
        positive_ratio: Fraction of samples with insertion (default 0.5)
        verbose: Show progress bar

    Returns:
        Tuple of (samples, labels) where samples are (x, y, z) tuples
    """
    # Load artifacts
    if verbose:
        print("Loading artifacts...")
    timelines = TimelineBuilder.load_all_timelines()
    bout_index = BoutIndexer.load_index()

    # Initialize components
    background_sampler = BackgroundSampler(timelines, bout_index, source_hz=SOURCE_HZ)
    needle_sampler = NeedleSampler(bout_index, transition_matrix=None, source_hz=SOURCE_HZ)
    style_transfer = StyleTransfer(
        transfer_mode="mean_only",
        blend_mode="cosine",
        blend_window_samples=50,
    )

    # RNG
    rng = np.random.default_rng(seed)

    # Compute needle length range in samples
    min_needle_samples = int(context_length_samples * NEEDLE_LENGTH_RATIO_MIN)
    max_needle_samples = int(context_length_samples * NEEDLE_LENGTH_RATIO_MAX)
    min_needle_ms = int(min_needle_samples * 1000 / SOURCE_HZ)

    # Margin from edges
    margin = int(context_length_samples * MARGIN_RATIO)

    samples = []
    labels = []
    failed_attempts = 0
    max_failed = n_samples * 2  # Allow some failures

    iterator = tqdm(range(n_samples), desc="Generating samples", disable=not verbose)

    while len(samples) < n_samples and failed_attempts < max_failed:
        try:
            # Sample pure background
            background = background_sampler.sample_background(
                context_length_samples=context_length_samples,
                purity="pure",
                rng=rng,
            )

            # Decide if positive (insertion) or negative (no insertion)
            is_positive = rng.random() < positive_ratio

            if is_positive:
                # Get the activity from the background
                activity = list(background.activities_present)[0]

                # Sample needle from SAME activity but DIFFERENT participant
                needle = needle_sampler.sample_needle(
                    activity=activity,
                    min_duration_ms=min_needle_ms,
                    exclude_pids={background.pid},
                    rng=rng,
                )

                if needle is None:
                    # No needle found for this activity from different participant
                    failed_attempts += 1
                    continue

                # Random needle length within range
                target_samples = rng.integers(min_needle_samples, max_needle_samples + 1)
                if target_samples > needle.n_samples:
                    target_samples = needle.n_samples

                # Trim needle
                trimmed_needle = needle.trim(target_samples)

                # Random position with margin
                max_position = context_length_samples - trimmed_needle.n_samples - margin
                if max_position <= margin:
                    # Context too short for this needle
                    failed_attempts += 1
                    continue
                position = rng.integers(margin, max_position)

                # Apply style transfer
                local_stats = style_transfer.compute_local_statistics(
                    (background.x, background.y, background.z),
                    position=position,
                )
                transferred = style_transfer.transfer(trimmed_needle, local_stats)

                # Insert with blending
                x, y, z = style_transfer.insert_with_blending(
                    (background.x, background.y, background.z),
                    (transferred.x, transferred.y, transferred.z),
                    position=position,
                )
                label = 1
            else:
                # No insertion - use background as-is
                x, y, z = background.x, background.y, background.z
                label = 0

            samples.append((x, y, z))
            labels.append(label)
            iterator.update(1)

        except Exception as e:
            failed_attempts += 1
            if failed_attempts % 100 == 0 and verbose:
                print(f"Warning: {failed_attempts} failed attempts (last error: {e})")
            continue

    iterator.close()

    if len(samples) < n_samples:
        print(f"Warning: Only generated {len(samples)}/{n_samples} samples")

    return samples, labels


# ==============================================================================
# Feature Preparation (Raw Time Series)
# ==============================================================================


def flatten_samples(
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> np.ndarray:
    """
    Flatten 3-axis time series into feature matrix.

    Args:
        samples: List of (x, y, z) tuples

    Returns:
        Feature matrix of shape (n_samples, 3 * context_length)
    """
    return np.array([np.concatenate([x, y, z]) for x, y, z in samples])


# ==============================================================================
# Training & Evaluation
# ==============================================================================


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple["XGBClassifier", float]:
    """
    Train XGBoost classifier and compute AUC.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        Tuple of (trained classifier, AUC score)
    """
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    print("Training XGBoost classifier...")
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    return clf, auc


# ==============================================================================
# Analysis
# ==============================================================================


def analyze_position_importance(
    clf: "XGBClassifier",
    context_length: int,
    top_k: int = 10,
) -> None:
    """
    Show which time positions the classifier focuses on.

    If the classifier detects insertion artifacts, high importance positions
    near needle boundaries indicate blending issues.

    Args:
        clf: Trained classifier
        context_length: Number of samples per axis
        top_k: Number of top positions to show
    """
    importances = clf.feature_importances_

    # Split importances by axis
    x_imp = importances[:context_length]
    y_imp = importances[context_length : 2 * context_length]
    z_imp = importances[2 * context_length :]

    # Combined importance across axes
    combined = x_imp + y_imp + z_imp

    # Find peak importance positions
    peak_positions = np.argsort(combined)[::-1][:top_k]

    print("\n" + "-" * 60)
    print("POSITION IMPORTANCE (where classifier detects differences)")
    print("-" * 60)
    print(f"{'Rank':<6} {'Position':<12} {'% into window':<15} {'Importance':<12}")
    print("-" * 60)

    for rank, pos in enumerate(peak_positions, 1):
        pct = 100 * pos / context_length
        imp = combined[pos]
        print(f"{rank:<6} {pos:<12} {pct:>6.1f}%{'':<8} {imp:.6f}")

    # Analyze distribution of high-importance positions
    print("\n" + "-" * 60)
    print("POSITIONAL DISTRIBUTION ANALYSIS")
    print("-" * 60)

    # Check if importance is concentrated in certain regions
    n_bins = 5
    bin_size = context_length // n_bins
    bin_importances = []
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else context_length
        bin_imp = np.sum(combined[start:end])
        bin_importances.append(bin_imp)
        pct_start = 100 * start / context_length
        pct_end = 100 * end / context_length
        print(f"Region {pct_start:5.1f}%-{pct_end:5.1f}%: {bin_imp:.4f}")

    # Check for boundary concentration
    boundary_region = context_length // 10  # 10% at each edge
    edge_importance = np.sum(combined[:boundary_region]) + np.sum(combined[-boundary_region:])
    middle_importance = np.sum(combined[boundary_region:-boundary_region])
    total_importance = np.sum(combined)

    print(f"\nEdge regions (0-10%, 90-100%): {100*edge_importance/total_importance:.1f}% of importance")
    print(f"Middle region (10-90%): {100*middle_importance/total_importance:.1f}% of importance")

    if edge_importance > middle_importance:
        print("\n⚠️  High importance at edges may indicate blending artifacts at boundaries.")
    else:
        print("\n✓  Importance is distributed across the window (no obvious boundary artifacts).")


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test insertion quality using classifier-based detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default settings (3 seconds, 5000 train, 500 test)
  python -m opentslm.time_series_datasets.ts_haystack.scripts.test_insertion_quality

  # Custom context length (2-5 seconds recommended)
  python -m opentslm.time_series_datasets.ts_haystack.scripts.test_insertion_quality \\
      --context-seconds 5.0

  # Quick test with fewer samples
  python -m opentslm.time_series_datasets.ts_haystack.scripts.test_insertion_quality \\
      --n-train 500 --n-test 100
        """,
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=5000,
        help="Number of training samples (default: 5000)",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=500,
        help="Number of test samples (default: 500)",
    )
    parser.add_argument(
        "--context-seconds",
        type=float,
        default=3.0,
        help="Context window size in seconds (default: 3.0, recommended: 2-5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--auc-threshold",
        type=float,
        default=0.55,
        help="AUC threshold for investigation (default: 0.55)",
    )
    args = parser.parse_args()

    # Convert to samples
    context_length = int(args.context_seconds * SOURCE_HZ)

    print("=" * 60)
    print("TS-HAYSTACK INSERTION QUALITY TEST")
    print("=" * 60)
    print(f"Context: {args.context_seconds}s ({context_length} samples @ {SOURCE_HZ}Hz)")
    print(f"Needle length: {NEEDLE_LENGTH_RATIO_MIN*100:.0f}-{NEEDLE_LENGTH_RATIO_MAX*100:.0f}% of context")
    print(f"Train samples: {args.n_train}")
    print(f"Test samples: {args.n_test}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Generate training samples
    print("\n[1/4] Generating training samples...")
    train_samples, train_labels = generate_samples(
        n_samples=args.n_train,
        context_length_samples=context_length,
        seed=args.seed,
        positive_ratio=0.5,
    )

    # Generate test samples (different seed)
    print("\n[2/4] Generating test samples...")
    test_samples, test_labels = generate_samples(
        n_samples=args.n_test,
        context_length_samples=context_length,
        seed=args.seed + 1000,  # Different seed for test
        positive_ratio=0.5,
    )

    # Flatten to feature matrices
    print("\n[3/4] Preparing features...")
    X_train = flatten_samples(train_samples)
    X_test = flatten_samples(test_samples)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    print(f"Train shape: {X_train.shape} (flattened x,y,z)")
    print(f"Test shape: {X_test.shape}")
    print(f"Train class balance: {np.mean(y_train):.2%} positive")
    print(f"Test class balance: {np.mean(y_test):.2%} positive")

    # Train and evaluate
    print("\n[4/4] Training and evaluating...")
    clf, auc = train_and_evaluate(X_train, y_train, X_test, y_test)

    # Report results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"XGBoost AUC: {auc:.4f}")

    if auc < 0.45:
        status = "UNEXPECTED"
        interpretation = "AUC below 0.45 suggests label issues or data problems."
    elif auc <= args.auc_threshold:
        status = "PASS"
        interpretation = "Insertions are effectively undetectable by the classifier."
    elif auc <= 0.65:
        status = "MARGINAL"
        interpretation = "Weak signal detected. May warrant investigation."
    else:
        status = "INVESTIGATE"
        interpretation = "Classifier detects insertion artifacts."

    print(f"Status: {status}")
    print(f"Interpretation: {interpretation}")

    # Show position importance if AUC is above threshold
    if auc > args.auc_threshold:
        analyze_position_importance(clf, context_length)

    print("\n" + "=" * 60)
    if status == "PASS":
        print("✓ Style transfer + cosine blending produces clean insertions.")
    else:
        print("⚠️  Review the position importance analysis above.")
    print("=" * 60)

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
