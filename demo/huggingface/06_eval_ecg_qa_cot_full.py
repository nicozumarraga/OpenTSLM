# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
"""
Full evaluation script for ECG QA CoT model from HuggingFace.

This script:
1. Loads a pretrained model from HuggingFace Hub
2. Loads the ECG QA CoT test dataset
3. Generates predictions on the full test set (or a sample)
4. Saves predictions in JSONL format compatible with parse_ecg_qa_cot_data.py

Usage:
    # Quick test with 5 samples
    python 06_eval_ecg_qa_cot_full.py --sample 5

    # Full evaluation (41k samples)
    python 06_eval_ecg_qa_cot_full.py --full

    # Custom sample size
    python 06_eval_ecg_qa_cot_full.py --sample 100
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE


# Model repository ID - change this to test different models
# NOTE: Use -flamingo for Perceiver-based models, -sp for SoftPrompt models
REPO_ID = "OpenTSLM/llama-3.2-1b-ecg-flamingo"


def main():
    parser = argparse.ArgumentParser(description="ECG QA CoT Full Evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sample",
        type=int,
        help="Run on N samples (for quick testing, e.g., --sample 5)"
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="Run on full test set (~41k samples)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path (default: auto-generated based on mode)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=REPO_ID,
        help=f"HuggingFace model repo ID (default: {REPO_ID})"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Max new tokens for generation (default: 500)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh, ignoring any existing progress"
    )
    args = parser.parse_args()

    # Determine model type from repo_id for output naming
    model_type = "flamingo" if "flamingo" in args.repo_id.lower() else "sp"

    # Determine output path
    script_dir = Path(__file__).parent
    if args.output:
        output_path = Path(args.output)
    else:
        if args.full:
            output_path = script_dir / f"ecg_qa_cot_predictions_full_{model_type}.jsonl"
        else:
            output_path = script_dir / f"ecg_qa_cot_predictions_sample_{args.sample}_{model_type}.jsonl"

    print("=" * 60)
    print("ECG QA CoT Full Evaluation")
    print("=" * 60)
    print(f"Model: {args.repo_id}")
    print(f"Device: {args.device}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Output: {output_path}")
    if args.sample:
        print(f"Mode: SAMPLE ({args.sample} samples)")
    else:
        print("Mode: FULL EVALUATION")
    print("=" * 60)

    # Load model from HuggingFace
    print(f"\n📥 Loading model from {args.repo_id}...")
    enable_lora = "-sp" in args.repo_id
    model = OpenTSLM.load_pretrained(args.repo_id, enable_lora=enable_lora, device=args.device)

    # Create dataset
    print("\n📊 Loading ECG QA CoT test dataset...")
    test_dataset = ECGQACoTQADataset("test", EOS_TOKEN=model.get_eos_token())
    total_samples = len(test_dataset)
    print(f"   Total test samples: {total_samples}")

    # Determine number of samples to process
    num_samples = args.sample if args.sample else total_samples

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        ),
    )

    # Check for existing progress (resume capability)
    start_index = 0
    if output_path.exists() and not args.fresh:
        with open(output_path, "r", encoding="utf-8") as f:
            start_index = sum(1 for _ in f)
        if start_index > 0:
            print(f"\n📂 Found existing progress: {start_index} samples already completed")
            print(f"   Resuming from sample {start_index + 1}...")
            if args.sample and start_index >= args.sample:
                print(f"   Already completed {args.sample} samples. Nothing to do.")
                return
    elif args.fresh and output_path.exists():
        print(f"\n🗑️  --fresh specified, removing existing progress file")
        output_path.unlink()

    remaining = num_samples - start_index
    print(f"\n🔍 Running inference on {remaining} remaining samples (of {num_samples} total)...")
    print("=" * 60)

    # Open output file in append mode for resume capability
    results_count = start_index
    with open(output_path, "a", encoding="utf-8") as f:
        for i, batch in enumerate(tqdm(test_loader, total=num_samples, desc="Evaluating", initial=start_index)):
            # Skip already processed samples
            if i < start_index:
                continue

            # Check if we've reached the sample limit
            if args.sample and i >= args.sample:
                break

            # Generate predictions
            predictions = model.generate(batch, max_new_tokens=args.max_new_tokens)

            # Process results
            for sample, pred in zip(batch, predictions):
                result = {
                    # Fields expected by parse_ecg_qa_cot_data.py
                    "generated_answer": pred,
                    "target_answer": sample.get("answer", ""),
                    "template_id": sample.get("template_id"),
                    "ecg_id": sample.get("ecg_id"),
                    # Additional fields for debugging/analysis
                    "pre_prompt": sample.get("pre_prompt", ""),
                }

                # Write to file immediately (streaming)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                results_count += 1

    print("\n" + "=" * 60)
    print(f"✅ Evaluation complete!")
    print(f"   Processed: {results_count} samples")
    print(f"   Output saved to: {output_path}")
    print("=" * 60)

    # Show sample of output for verification
    print("\n📝 Sample output (first 2 entries):")
    print("-" * 60)
    with open(output_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            data = json.loads(line)
            print(f"\nEntry {i + 1}:")
            print(f"  Template ID: {data.get('template_id')}")
            print(f"  ECG ID: {data.get('ecg_id')}")
            print(f"  Target Answer (truncated): {data.get('target_answer', '')[:100]}...")
            print(f"  Generated Answer (truncated): {data.get('generated_answer', '')[:100]}...")

    print("\n" + "=" * 60)
    print("📊 To calculate metrics, run:")
    print(f"   cd evaluation/opentslm/ecg_qa_cot")
    print(f"   python parse_ecg_qa_cot_data.py --input {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
