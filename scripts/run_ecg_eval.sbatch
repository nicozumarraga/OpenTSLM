#!/bin/bash
#SBATCH --job-name=ecg_eval
#SBATCH --partition=gpu-a100-80g       # 7-day limit, L4 24GB GPU
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00               # 72 hours for full evaluation
#SBATCH --output=logs/ecg_eval_%j.out
#SBATCH --error=logs/ecg_eval_%j.err
# #SBATCH --mail-type=END,FAIL          # Email when done or failed (optional)
# #SBATCH --mail-user=your@email.com  # Uncomment and set your email

# Model type: "flamingo" or "sp" (passed as first argument, defaults to "flamingo")
MODEL_TYPE=${1:-flamingo}

# Print job info
echo "=========================================="
echo "ECG QA CoT EVALUATION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model Type: $MODEL_TYPE"
echo "Started: $(date)"
echo "=========================================="

# Navigate to project directory
cd /zfsstore/user/s4350529/projects/OpenTSLM

# Activate virtual environment
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the evaluation (auto-resumes if interrupted)
# Model type is auto-detected from repo_id, output file named accordingly
python demo/huggingface/06_eval_ecg_qa_cot_full.py --full

# Parse results when done
echo ""
echo "=========================================="
echo "Parsing results..."
echo "=========================================="
python evaluation/opentslm/ecg_qa_cot/parse_ecg_qa_cot_data.py \
    --input demo/huggingface/ecg_qa_cot_predictions_full_${MODEL_TYPE}.jsonl

echo ""
echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
