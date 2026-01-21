#!/bin/bash
#SBATCH --job-name=capture24-extract
#SBATCH --partition=cpu-skylake
#SBATCH --output=capture24_extract_%j.log
#SBATCH --error=capture24_extract_%j.log
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

# Capture-24 Data Extraction Job
# Extracts and converts the capture24.zip to Parquet format

set -e

echo "========================================"
echo "Capture-24 Data Extraction"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "========================================"

cd /zfsstore/user/s4350529/projects/OpenTSLM-clone
source .venv/bin/activate

# Run extraction with parallel processing
python -m opentslm.time_series_datasets.capture24.capture24_loader --n-jobs 6 --overwrite

echo ""
echo "========================================"
echo "Extraction complete: $(date)"
echo "========================================"
