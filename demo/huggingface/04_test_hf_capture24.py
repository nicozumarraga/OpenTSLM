# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
"""
Demo script for testing Capture24 evaluation dataset with HAR model from HuggingFace.

This script:
1. Loads a HAR pretrained model from HuggingFace Hub (SP or Flamingo)
2. Loads the Capture24 evaluation dataset (2.56s window by default)
3. Generates predictions on a few test samples
4. Prints model outputs with ground truth comparison

Usage:
    python 04_test_hf_capture24.py
    python 04_test_hf_capture24.py --window-size 10
    python 04_test_hf_capture24.py --num-samples 10 --shuffle
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.model_config import PATCH_SIZE
from opentslm.time_series_datasets.capture24.Capture24EvalQADataset import (
    Capture24EvalQADataset,
)
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

# Model repository ID - HAR-Flamingo checkpoint (for zero-shot on Capture24)
REPO_ID = "OpenTSLM/llama-3.2-1b-har-sp"


def main():
    parser = argparse.ArgumentParser(
        description="Test HAR-Flamingo model on Capture24 evaluation dataset"
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=2.56,
        help="Window size in seconds (default: 2.56)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to test (default: 5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum new tokens to generate (default: 500)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples to see different classes",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Capture24 Evaluation Demo (HAR-Flamingo)")
    print("=" * 60)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load model from HuggingFace
    print(f"\nLoading model from {REPO_ID}...")
    enable_lora = "-sp" in REPO_ID
    model = OpenTSLM.load_pretrained(REPO_ID, enable_lora=enable_lora, device=device)
    model.eval()
    print("Model loaded successfully!")

    # Create dataset
    print(f"\nLoading Capture24 test dataset ({args.window_size}s windows)...")
    try:
        test_dataset = Capture24EvalQADataset(
            split="test",
            EOS_TOKEN=model.get_eos_token(),
            window_size_s=args.window_size,
        )
    except FileNotFoundError as e:
        print(f"\nERROR: Dataset not found!")
        print(f"  {e}")
        print("\nRun scripts/phase1_dataset_preparation.py first to create the dataset.")
        sys.exit(1)

    print(f"Dataset size: {len(test_dataset)}")
    print(f"Eval labels: {test_dataset.get_eval_labels()}")

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        shuffle=args.shuffle,
        batch_size=1,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        ),
    )

    print(f"\nRunning inference on {min(args.num_samples, len(test_dataset))} test samples...")
    print("=" * 60)

    # Track accuracy
    correct = 0
    total = 0

    # Iterate over evaluation set
    for i, batch in enumerate(test_loader):
        if i >= args.num_samples:
            break

        # Generate predictions
        with torch.no_grad():
            predictions = model.generate(batch, max_new_tokens=args.max_tokens)

        # Print results
        for sample, pred in zip(batch, predictions):
            total += 1
            ground_truth = sample.get("label", sample.get("answer", "N/A"))

            # Extract predicted label (last word after "Answer:" or just last word)
            pred_lower = pred.lower()
            if "answer:" in pred_lower:
                # Find last occurrence of "answer:"
                idx = pred_lower.rfind("answer:")
                after_answer = pred[idx + 7:].strip()
                predicted_label = after_answer.split()[0] if after_answer.split() else ""
            else:
                predicted_label = pred.split()[-1] if pred.split() else ""

            # Clean up punctuation
            predicted_label = predicted_label.lower().rstrip(".,;:!?")

            is_correct = predicted_label == ground_truth.lower()
            if is_correct:
                correct += 1

            status = "CORRECT" if is_correct else "WRONG"

            print(f"\nSample {i + 1}:")
            print(f"  Ground Truth: {ground_truth}")
            print(f"  Predicted: {predicted_label} [{status}]")
            print(f"  Full Output: {pred}")
            print("-" * 60)

    # Print summary
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\nSummary: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
