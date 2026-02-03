#!/bin/bash
#SBATCH --job-name=ts-haystack-cot
#SBATCH --output=ts_haystack_cot_%j.log
#SBATCH --error=ts_haystack_cot_%j.log
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-a100-80g

# ==============================================================================
# Train OpenTSLM Flamingo on TS-Haystack Chain-of-Thought Dataset
# ==============================================================================
#
# Usage:
#   # Default training (from scratch, all tasks, 100s context)
#   sbatch scripts/train_ts_haystack_cot.sbatch
#
#   # Fine-tune from pretrained HuggingFace checkpoint
#   sbatch --export=FROM_HF="OpenTSLM/llama-3.2-1b-har-flamingo" scripts/train_ts_haystack_cot.sbatch
#
#   # Train on specific tasks
#   sbatch --export=TASKS="existence localization counting" scripts/train_ts_haystack_cot.sbatch
#
#   # Quick test run
#   sbatch --export=MAX_SAMPLES=100,EPOCHS=2 scripts/train_ts_haystack_cot.sbatch
#
#   # Custom context length
#   sbatch --export=CONTEXT_LENGTHS="100 1000" scripts/train_ts_haystack_cot.sbatch
#
#   # Use different base LLM
#   sbatch --export=LLM_ID="meta-llama/Llama-3.2-3B" scripts/train_ts_haystack_cot.sbatch
#
#   # Resume from local checkpoint
#   sbatch --export=RESUME_FROM="results/ts_haystack_cot/checkpoints/best_model.pt" scripts/train_ts_haystack_cot.sbatch
#
# ==============================================================================

set -e

echo "========================================"
echo "TS-Haystack CoT Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

cd /zfsstore/user/s4350529/projects/OpenTSLM-clone
source .venv/bin/activate

# Print environment info
echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" = "True" ]; then
    echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "  GPU Memory: $(python -c 'import torch; mem=torch.cuda.get_device_properties(0).total_memory / 1024**3; print(f"{mem:.1f} GB")')"
fi
echo ""

# Build command
CMD="python scripts/train_ts_haystack_cot.py"

# Base LLM ID
if [ -n "$LLM_ID" ]; then
    CMD="$CMD --llm-id $LLM_ID"
    echo "LLM ID: $LLM_ID"
else
    echo "LLM ID: meta-llama/Llama-3.2-1B (default)"
fi

# Load pretrained weights from HuggingFace
if [ -n "$FROM_HF" ]; then
    CMD="$CMD --from-hf $FROM_HF"
    echo "Loading pretrained from: $FROM_HF"
else
    echo "Training from scratch (no pretrained weights)"
fi

# Tasks
if [ -n "$TASKS" ]; then
    CMD="$CMD --tasks $TASKS"
    echo "Tasks: $TASKS"
else
    echo "Tasks: all (default)"
fi

# Context lengths
if [ -n "$CONTEXT_LENGTHS" ]; then
    CMD="$CMD --context-lengths $CONTEXT_LENGTHS"
    echo "Context lengths: $CONTEXT_LENGTHS"
else
    echo "Context lengths: 100s (default)"
fi

# Batch size
if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch-size $BATCH_SIZE"
    echo "Batch size: $BATCH_SIZE"
else
    echo "Batch size: 2 (default)"
fi

# Epochs
if [ -n "$EPOCHS" ]; then
    CMD="$CMD --epochs $EPOCHS"
    echo "Epochs: $EPOCHS"
else
    echo "Epochs: 30 (default)"
fi

# Learning rate
if [ -n "$LR" ]; then
    CMD="$CMD --lr $LR"
    echo "Learning rate: $LR"
else
    echo "Learning rate: 2e-4 (default)"
fi

# Early stopping
if [ -n "$EARLY_STOP" ]; then
    CMD="$CMD --early-stop $EARLY_STOP"
    echo "Early stop patience: $EARLY_STOP"
else
    echo "Early stop patience: 5 (default)"
fi

# Max samples (for testing)
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
    echo "Max samples: $MAX_SAMPLES (testing mode)"
fi

# Output directory
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
    echo "Output dir: $OUTPUT_DIR"
else
    echo "Output dir: results/ts_haystack_cot (default)"
fi

# Resume from checkpoint
if [ -n "$RESUME_FROM" ]; then
    CMD="$CMD --resume $RESUME_FROM"
    echo "Resuming from: $RESUME_FROM"
fi

echo ""
echo "========================================"
echo "Running: $CMD"
echo "========================================"
echo ""

# Run training
$CMD

echo ""
echo "========================================"
echo "Training complete: $(date)"
echo "========================================"
