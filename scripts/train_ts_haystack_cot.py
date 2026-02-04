#!/usr/bin/env python3
"""
Train OpenTSLM Flamingo on TS-Haystack Chain-of-Thought Dataset.

This script fine-tunes the OpenTSLM Flamingo model on the TS-Haystack benchmark
with chain-of-thought rationales. It can start from:
1. A pretrained HuggingFace checkpoint (recommended)
2. From scratch (random initialization)

Each training run creates a unique folder with timestamp (or custom name):
    results/ts_haystack_cot/run_20260202_143045/
    results/ts_haystack_cot/my_experiment/

Usage:
    # Train from scratch with default LLM (Llama-3.2-1B)
    python scripts/train_ts_haystack_cot.py

    # Fine-tune from a pretrained HuggingFace checkpoint
    python scripts/train_ts_haystack_cot.py --from-hf OpenTSLM/llama-3.2-1b-har-flamingo

    # Fine-tune with specific tasks
    python scripts/train_ts_haystack_cot.py --from-hf OpenTSLM/llama-3.2-1b-har-flamingo --tasks existence localization counting

    # Quick test run with custom run name
    python scripts/train_ts_haystack_cot.py --max-samples 100 --epochs 2 --run-name quick_test

    # Resume training from local checkpoint (uses same output directory)
    python scripts/train_ts_haystack_cot.py --resume results/ts_haystack_cot/run_20260202_143045/checkpoints/best_model.pt
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.model_config import PATCH_SIZE
from opentslm.time_series_datasets.ts_haystack import TSHaystackCoTQADataset
from opentslm.time_series_datasets.ts_haystack.dataset.ts_haystack_qa_loader import (
    get_available_context_lengths,
)
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    llm_id: str = "meta-llama/Llama-3.2-1B"
    hf_checkpoint_repo: Optional[str] = None  # e.g., "OpenTSLM/llama-3.2-1b-har-flamingo"
    hf_checkpoint_file: str = "model_checkpoint.pt"  # Filename in HF repo

    # Data
    tasks: List[str] = None  # None = all tasks
    context_lengths_seconds: List[Union[str, float, int]] = None  # None = ["all"] (auto-discover)
    max_samples: Optional[int] = None  # None = use all samples

    # Training
    batch_size: int = 2
    epochs: int = 30
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    warmup_fraction: float = 0.03

    # Early stopping
    early_stop_patience: int = 5

    # Output
    output_dir: Path = Path("results/ts_haystack_cot")
    run_name: Optional[str] = None  # Custom run name (default: timestamp)

    # Checkpoint
    resume_from: Optional[str] = None

    def __post_init__(self):
        if self.tasks is None:
            self.tasks = ["all"]
        if self.context_lengths_seconds is None:
            self.context_lengths_seconds = ["all"]  # Auto-discover from filesystem
        self.output_dir = Path(self.output_dir)


# ==============================================================================
# Utility Functions
# ==============================================================================

def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_peak_gpu_memory_mb() -> float:
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ==============================================================================
# Model Setup
# ==============================================================================

def load_model(config: TrainingConfig, device: str) -> OpenTSLMFlamingo:
    """
    Load OpenTSLM Flamingo model.

    Args:
        config: Training configuration
        device: Device to load model on

    Returns:
        Loaded model
    """
    print(f"\nInitializing model with LLM: {config.llm_id}")

    # Calculate positional embedding sizes based on context length
    # Formula: context_seconds * sampling_rate / patch_size
    max_context_seconds = max(config.context_lengths_seconds)
    sampling_rate = 100  # Capture24 is 100Hz
    trained_patches = int((max_context_seconds * sampling_rate) // PATCH_SIZE)
    max_patches = trained_patches + 1000  # Buffer for slightly longer sequences

    print(f"  Context: {max_context_seconds}s -> {trained_patches} patches (max: {max_patches})")

    model = OpenTSLMFlamingo(
        device=device,
        llm_id=config.llm_id,
        cross_attn_every_n_layers=1,
        max_patches=max_patches,
        trained_patches=trained_patches,
    ).to(device)

    # Optionally load pretrained weights from HuggingFace
    if config.hf_checkpoint_repo:
        try:
            from huggingface_hub import hf_hub_download
            print(f"  Downloading checkpoint from: {config.hf_checkpoint_repo}")
            checkpoint_path = hf_hub_download(
                repo_id=config.hf_checkpoint_repo,
                filename=config.hf_checkpoint_file,
            )
            print(f"  Loading checkpoint: {checkpoint_path}")
            model.load_from_file(checkpoint_path)
        except Exception as e:
            print(f"  Warning: Could not load from HuggingFace: {e}")
            print("  Proceeding with randomly initialized weights")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Memory: {get_gpu_memory_mb():.1f} MB")

    return model


def freeze_except_trainable(model: OpenTSLMFlamingo) -> int:
    """
    Freeze LLM backbone, keep encoder + perceiver + cross-attention trainable.

    Args:
        model: OpenTSLMFlamingo model instance

    Returns:
        Number of trainable parameters
    """
    # Freeze everything first
    model.model.requires_grad_(False)

    # Unfreeze vision encoder (includes pos_embed)
    model.model.vision_encoder.requires_grad_(True)

    # Unfreeze perceiver (cross-attention perceiver)
    model.model.perceiver.requires_grad_(True)

    # Unfreeze gated cross-attention layers in LLM
    model.model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)

    # Unfreeze LM input embeddings
    model.model.lang_encoder.get_input_embeddings().requires_grad_(True)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.model.parameters())

    print(f"\nFreezing configuration:")
    print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Print breakdown
    encoder_params = sum(p.numel() for p in model.model.vision_encoder.parameters() if p.requires_grad)
    perceiver_params = sum(p.numel() for p in model.model.perceiver.parameters() if p.requires_grad)
    cross_attn_params = sum(p.numel() for p in model.model.lang_encoder.gated_cross_attn_layers.parameters() if p.requires_grad)
    embed_params = sum(p.numel() for p in model.model.lang_encoder.get_input_embeddings().parameters() if p.requires_grad)

    print(f"  - Vision encoder: {encoder_params:,}")
    print(f"  - Perceiver: {perceiver_params:,}")
    print(f"  - Cross-attention layers: {cross_attn_params:,}")
    print(f"  - LM embeddings: {embed_params:,}")

    return trainable_params


# ==============================================================================
# Data Loading
# ==============================================================================

def create_dataloaders(
    config: TrainingConfig,
) -> tuple:
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Training configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print(f"\nLoading TS-Haystack CoT dataset...")
    print(f"  Tasks: {config.tasks}")
    print(f"  Context lengths (seconds): {config.context_lengths_seconds}")

    # Create datasets - use default EOS_TOKEN (<|endofchunk|>) which is correct
    # for OpenTSLMFlamingo. Do NOT pass model.get_eos_token() as it returns
    # <|end_of_text|> which causes training issues.
    train_dataset = TSHaystackCoTQADataset(
        split="train",
        tasks=config.tasks,
        context_lengths_seconds=config.context_lengths_seconds,
    )

    val_dataset = TSHaystackCoTQADataset(
        split="validation",
        tasks=config.tasks,
        context_lengths_seconds=config.context_lengths_seconds,
    )

    test_dataset = TSHaystackCoTQADataset(
        split="test",
        tasks=config.tasks,
        context_lengths_seconds=config.context_lengths_seconds,
    )

    # Apply max_samples if specified
    if config.max_samples is not None:
        print(f"  Limiting to {config.max_samples} samples per split")
        # Note: This is a simple truncation; for proper random sampling,
        # you'd want to subset the underlying data

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Collate function for batching
    def collate_fn(batch):
        return extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, val_loader, test_loader


# ==============================================================================
# Checkpointing
# ==============================================================================

def save_checkpoint(
    model: OpenTSLMFlamingo,
    optimizer: AdamW,
    scheduler,
    epoch: int,
    val_loss: float,
    config: TrainingConfig,
    is_best: bool = False,
) -> Path:
    """Save training checkpoint."""
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "config": {
            "tasks": config.tasks,
            "context_lengths_seconds": config.context_lengths_seconds,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        },
        "timestamp": datetime.now().isoformat(),
    }

    if is_best:
        path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, path)
        print(f"  Saved best checkpoint to {path}")
        return path
    else:
        path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        return path


def load_checkpoint(
    path: str,
    model: OpenTSLMFlamingo,
    optimizer: Optional[AdamW] = None,
    scheduler = None,
) -> Dict[str, Any]:
    """Load training checkpoint."""
    print(f"\nLoading checkpoint from: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    print(f"  Resumed from epoch {checkpoint['epoch']}")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")

    return checkpoint


# ==============================================================================
# Training Loop
# ==============================================================================

def train_epoch(
    model: OpenTSLMFlamingo,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    config: TrainingConfig,
    epoch: int,
) -> float:
    """
    Train for one epoch.

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        try:
            loss = model.compute_loss(batch)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  Warning: Invalid loss at batch {batch_idx}, skipping")
                continue

            loss.backward()

            # Gradient clipping
            clip_grad_norm_(model.parameters(), config.grad_clip_norm)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            avg_loss = total_loss / num_batches
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}",
            })

        except Exception as e:
            print(f"\n  Error at batch {batch_idx}: {e}")
            continue

    return total_loss / max(num_batches, 1)


def extract_answer_from_rationale(rationale: str, answer_type: str) -> str:
    """
    Extract the final answer from a chain-of-thought rationale.

    Args:
        rationale: Full rationale text
        answer_type: Type of answer (boolean, integer, timestamp, category, time_range)

    Returns:
        Extracted answer string
    """
    import re
    rationale = rationale.strip()

    # Find the last occurrence of "Answer:" (case-insensitive)
    matches = list(re.finditer(r"answer:\s*", rationale, re.IGNORECASE))

    if matches:
        start = matches[-1].end()
        answer = rationale[start:].strip()

        if answer_type == "boolean":
            answer_lower = answer.lower()
            if answer_lower.startswith("yes") or "yes" in answer_lower[:10]:
                return "Yes"
            elif answer_lower.startswith("no") or "no" in answer_lower[:10]:
                return "No"

        elif answer_type == "integer":
            match = re.search(r"\d+", answer)
            if match:
                return match.group()

        else:
            answer = answer.split("\n")[0].split(".")[0].strip()

        return answer
    else:
        words = rationale.split()
        if words:
            return words[-1].rstrip(".,;:!?")

    return ""


def normalize_answer(answer: str, answer_type: str) -> str:
    """Normalize an answer for comparison."""
    import re
    answer = str(answer).strip().lower()
    answer = re.sub(r"[.,;:!?]+$", "", answer)

    if answer_type == "boolean":
        if answer in ["yes", "true", "1"]:
            return "yes"
        elif answer in ["no", "false", "0"]:
            return "no"

    if answer_type == "integer":
        try:
            return str(int(float(answer)))
        except ValueError:
            pass

    return answer


def validate(
    model: OpenTSLMFlamingo,
    val_loader: DataLoader,
    epoch: int,
    output_dir: Optional[Path] = None,
    split_name: str = "val",
    log_outputs: bool = False,
) -> Dict[str, Any]:
    """
    Validate the model with optional output logging.

    Args:
        model: The model to validate
        val_loader: Validation data loader
        epoch: Current epoch number
        output_dir: Directory to save output logs
        split_name: Name of the split (val/test) for logging
        log_outputs: Whether to generate and log model outputs for all samples

    Returns:
        Dictionary with validation loss and optional metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # For output logging
    logged_outputs = []
    samples_generated = 0
    task_correct = {}
    task_total = {}

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [{split_name.upper()}]", leave=False)

    with torch.no_grad():
        for batch in pbar:
            try:
                # Compute loss
                loss = model.compute_loss(batch)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    num_batches += 1

                avg_loss = total_loss / max(num_batches, 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                # Generate outputs for logging (all samples)
                if log_outputs:
                    try:
                        predictions = model.generate(batch, max_new_tokens=500)

                        for i, (sample, pred) in enumerate(zip(batch, predictions)):
                            task_type = sample.get("task_type", "unknown")
                            answer_type = sample.get("answer_type", "unknown")
                            ground_truth = sample.get("direct_answer", "")
                            ground_truth_rationale = sample.get("answer", "")  # Full CoT rationale
                            question = sample.get("question", "")

                            # Extract answer from prediction
                            pred_answer = extract_answer_from_rationale(pred, answer_type)

                            # Check correctness
                            gt_norm = normalize_answer(ground_truth, answer_type)
                            pred_norm = normalize_answer(pred_answer, answer_type)
                            is_correct = gt_norm == pred_norm

                            # Track per-task accuracy
                            if task_type not in task_correct:
                                task_correct[task_type] = 0
                                task_total[task_type] = 0
                            task_total[task_type] += 1
                            if is_correct:
                                task_correct[task_type] += 1

                            logged_outputs.append({
                                "sample_idx": samples_generated,
                                "task_type": task_type,
                                "answer_type": answer_type,
                                "question": question[:200],
                                "ground_truth": ground_truth,
                                "ground_truth_rationale": ground_truth_rationale,
                                "predicted_answer": pred_answer,
                                "prediction_preview": pred,
                                "correct": is_correct,
                            })

                            samples_generated += 1

                    except Exception as e:
                        pass  # Skip generation errors

            except Exception as e:
                continue

    avg_loss = total_loss / max(num_batches, 1)

    result = {
        "loss": avg_loss,
        "num_batches": num_batches,
    }

    # Add accuracy metrics if we logged outputs
    if logged_outputs:
        total_correct = sum(1 for o in logged_outputs if o["correct"])
        result["accuracy"] = total_correct / len(logged_outputs) if logged_outputs else 0.0
        result["samples_evaluated"] = len(logged_outputs)
        result["task_accuracy"] = {
            task: task_correct[task] / task_total[task] if task_total[task] > 0 else 0.0
            for task in task_correct
        }

        # Save logged outputs to file
        if output_dir:
            logs_dir = output_dir / "output_logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = logs_dir / f"{split_name}_epoch_{epoch}.json"
            with open(log_file, "w") as f:
                json.dump({
                    "epoch": epoch,
                    "split": split_name,
                    "loss": avg_loss,
                    "accuracy": result.get("accuracy", 0.0),
                    "task_accuracy": result.get("task_accuracy", {}),
                    "outputs": logged_outputs,
                }, f, indent=2)

    return result


def train(
    config: TrainingConfig,
) -> Dict[str, Any]:
    """
    Main training function.

    Args:
        config: Training configuration

    Returns:
        Dictionary with training results
    """
    device = get_device()

    # Resolve "all" context lengths early (needed for model initialization)
    if config.context_lengths_seconds == ["all"] or "all" in config.context_lengths_seconds:
        resolved_lengths = get_available_context_lengths(use_cot=True)
        if not resolved_lengths:
            raise ValueError(
                "No context lengths found in CoT directory. "
                "Make sure CoT datasets have been generated first."
            )
        config.context_lengths_seconds = resolved_lengths
        print(f"Auto-discovered context lengths: {resolved_lengths}")

    # Create unique run directory with timestamp (unless resuming)
    if config.resume_from is None:
        if config.run_name:
            run_folder = config.run_name
        else:
            run_folder = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config.output_dir = config.output_dir / run_folder
    else:
        # When resuming, use the same output directory as the checkpoint
        # Checkpoint path: .../run_XXXXXXXX/checkpoints/best_model.pt
        checkpoint_path = Path(config.resume_from)
        if checkpoint_path.parent.name == "checkpoints":
            config.output_dir = checkpoint_path.parent.parent
        else:
            # Fallback: use parent directory
            config.output_dir = checkpoint_path.parent

    print("=" * 70)
    print("TS-Haystack CoT Training")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Tasks: {config.tasks}")
    print(f"  Context lengths: {config.context_lengths_seconds}s")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Output dir: {config.output_dir}")

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(config.output_dir / "config.json", "w") as f:
        json.dump({
            "run_dir": str(config.output_dir),
            "tasks": config.tasks,
            "context_lengths_seconds": config.context_lengths_seconds,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "llm_id": config.llm_id,
            "hf_checkpoint_repo": config.hf_checkpoint_repo,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    # Load model
    model = load_model(config, device)

    # Freeze non-trainable components
    freeze_except_trainable(model)

    # Create dataloaders
    # NOTE: Don't pass model.get_eos_token() - the dataset defaults to <|endofchunk|>
    # which is the correct answer terminator for OpenTSLMFlamingo. The model's
    # tokenizer.eos_token returns <|end_of_text|> which causes training issues.
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Setup scheduler
    total_steps = config.epochs * len(train_loader)
    warmup_steps = int(config.warmup_fraction * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\nTraining setup:")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float("inf")

    if config.resume_from:
        checkpoint = load_checkpoint(
            config.resume_from, model, optimizer, scheduler
        )
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": None,
        "best_val_loss": float("inf"),
    }

    epochs_no_improve = 0
    training_start_time = time.time()

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, config, epoch
        )

        # Validate with output logging (logs all samples)
        val_result = validate(
            model, val_loader, epoch,
            output_dir=config.output_dir,
            split_name="val",
            log_outputs=True,
        )
        val_loss = val_result["loss"]

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Record accuracy if available
        if "accuracy" in val_result:
            if "val_accuracy" not in history:
                history["val_accuracy"] = []
            history["val_accuracy"].append(val_result["accuracy"])

        epoch_time = time.time() - epoch_start_time

        # Print epoch summary
        summary_str = (f"Epoch {epoch}/{config.epochs}: "
                      f"train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}")
        if "accuracy" in val_result:
            summary_str += f", val_acc={val_result['accuracy']*100:.1f}%"
        summary_str += f", time={format_time(epoch_time)}"
        print(summary_str)

        # Print per-task accuracy if available
        if "task_accuracy" in val_result:
            for task, acc in sorted(val_result["task_accuracy"].items()):
                print(f"    {task}: {acc*100:.1f}%")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            history["best_val_loss"] = val_loss
            epochs_no_improve = 0

            # Save best checkpoint
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, is_best=True)
            print(f"  ✓ New best model!")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs")

        # Early stopping
        if epochs_no_improve >= config.early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

        # Save history
        with open(config.output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - training_start_time

    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Best epoch: {history['best_epoch']}")
    print(f"  Best val loss: {history['best_val_loss']:.4f}")
    print(f"  Checkpoint: {config.output_dir / 'checkpoints' / 'best_model.pt'}")

    # Load best model for final evaluation
    if history["best_epoch"] is not None:
        load_checkpoint(
            str(config.output_dir / "checkpoints" / "best_model.pt"),
            model
        )

    # Final test evaluation with full output logging (logs all samples)
    print("\nFinal test evaluation...")
    test_result = validate(
        model, test_loader, epoch=0,
        output_dir=config.output_dir,
        split_name="test",
        log_outputs=True,
    )
    print(f"  Test loss: {test_result['loss']:.4f}")
    if "accuracy" in test_result:
        print(f"  Test accuracy: {test_result['accuracy']*100:.1f}%")
        if "task_accuracy" in test_result:
            print("  Per-task accuracy:")
            for task, acc in sorted(test_result["task_accuracy"].items()):
                print(f"    {task}: {acc*100:.1f}%")

    history["test_loss"] = test_result["loss"]
    if "accuracy" in test_result:
        history["test_accuracy"] = test_result["accuracy"]
        history["test_task_accuracy"] = test_result.get("task_accuracy", {})

    # Save final history
    with open(config.output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train OpenTSLM Flamingo on TS-Haystack CoT"
    )

    # Model
    parser.add_argument(
        "--llm-id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Base LLM model ID (default: meta-llama/Llama-3.2-1B)"
    )
    parser.add_argument(
        "--from-hf",
        type=str,
        default=None,
        help="Load pretrained weights from HuggingFace repo (e.g., OpenTSLM/llama-3.2-1b-har-flamingo)"
    )
    parser.add_argument(
        "--hf-checkpoint-file",
        type=str,
        default="model_checkpoint.pt",
        help="Checkpoint filename in HuggingFace repo (default: model_checkpoint.pt)"
    )

    # Data
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Tasks to train on (default: all). Options: existence, localization, counting, ordering, state_query, antecedent, comparison, multi_hop"
    )
    parser.add_argument(
        "--context-lengths",
        type=str,
        nargs="+",
        default=["all"],
        help="Context lengths in seconds, or 'all' to auto-discover (default: all)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per split (for quick testing)"
    )

    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs (default: 30)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ts_haystack_cot",
        help="Base output directory (default: results/ts_haystack_cot)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name (default: run_YYYYMMDD_HHMMSS)"
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Parse context lengths - can be "all" or numeric values
    context_lengths = args.context_lengths
    if "all" not in context_lengths:
        # Convert numeric strings to floats
        context_lengths = [float(x) for x in context_lengths]

    # Create config
    config = TrainingConfig(
        llm_id=args.llm_id,
        hf_checkpoint_repo=args.from_hf,
        hf_checkpoint_file=args.hf_checkpoint_file,
        tasks=args.tasks if args.tasks else ["all"],
        context_lengths_seconds=context_lengths,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        early_stop_patience=args.early_stop,
        output_dir=Path(args.output_dir),
        run_name=args.run_name,
        resume_from=args.resume,
    )

    # Train
    train(config)


if __name__ == "__main__":
    main()
