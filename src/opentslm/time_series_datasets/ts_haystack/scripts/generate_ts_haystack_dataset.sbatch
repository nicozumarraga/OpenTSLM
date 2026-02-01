#!/bin/bash
#SBATCH --job-name=ts-haystack-gen
#SBATCH --output=logs/ts_haystack_gen_%j.out
#SBATCH --error=logs/ts_haystack_gen_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# TS-Haystack Dataset Generation SBATCH Script
# =============================================
# Generates TS-Haystack benchmark datasets using YAML configuration.
#
# Usage:
#   # Default config
#   sbatch generate_ts_haystack_dataset.sbatch
#
#   # Custom config
#   sbatch --export=CONFIG=configs/my_experiment.yaml generate_ts_haystack_dataset.sbatch

set -e

# Setup environment
source ~/.bashrc
conda activate opentslm

# Create logs directory
mkdir -p logs

# Use custom config if provided, otherwise use default
CONFIG_PATH="${CONFIG:-}"

echo "========================================"
echo "TS-Haystack Dataset Generation"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Config: ${CONFIG_PATH:-default}"
echo "========================================"

if [ -n "$CONFIG_PATH" ]; then
    python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_dataset \
        --config "$CONFIG_PATH"
else
    python -m opentslm.time_series_datasets.ts_haystack.scripts.generate_ts_haystack_dataset
fi

echo "========================================"
echo "Generation complete!"
echo "========================================"
