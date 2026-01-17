#!/bin/bash
#SBATCH --job-name=curriculum_train
#SBATCH --partition=gpu-a100-80g        # A100 for faster training
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00                # 7 days (max allowed)
#SBATCH --output=logs/curriculum_train_%j.out
#SBATCH --error=logs/curriculum_train_%j.err
#SBATCH --mem=64G                        # Request sufficient RAM
#SBATCH --cpus-per-task=8                # CPUs for data loading
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your@email.com

# Print job info
echo "=========================================="
echo "CURRICULUM LEARNING TRAINING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Started: $(date)"
echo "=========================================="

# Navigate to project directory
cd /zfsstore/user/s4350529/projects/OpenTSLM

# Activate virtual environment
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Print Python and torch info
echo ""
echo "Environment:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo ""

# Run full curriculum training (all 5 stages)
# This will:
# - Stage 1 (MCQ): ~30 epochs
# - Stage 2 (Captioning): ~20 epochs
# - Stage 3 (HAR CoT): ~30 epochs
# - Stage 4 (Sleep CoT): ~60 epochs
# - Stage 5 (ECG CoT): ~60 epochs
#
# Each stage automatically loads the previous stage's best checkpoint
# Training resumes from last checkpoint if interrupted

echo "=========================================="
echo "Starting curriculum training..."
echo "Model: OpenTSLMFlamingo"
echo "LLM: meta-llama/Llama-3.2-1B"
echo "Stages: stage1_mcq -> stage2_captioning -> stage3_cot -> stage4_sleep_cot -> stage5_ecg_cot"
echo "=========================================="

python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --llm_id meta-llama/Llama-3.2-1B \
    --device cuda \
    --gradient_checkpointing \
    --batch_size 4

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Training failed or was interrupted"
    echo "You can resume by running the same command"
    echo "=========================================="
fi

echo ""
echo "Checkpoints saved to:"
echo "  results/Llama_3_2_1B/OpenTSLMFlamingo/stage1_mcq/checkpoints/"
echo "  results/Llama_3_2_1B/OpenTSLMFlamingo/stage2_captioning/checkpoints/"
echo "  results/Llama_3_2_1B/OpenTSLMFlamingo/stage3_cot/checkpoints/"
echo "  results/Llama_3_2_1B/OpenTSLMFlamingo/stage4_sleep_cot/checkpoints/"
echo "  results/Llama_3_2_1B/OpenTSLMFlamingo/stage5_ecg_cot/checkpoints/"
echo ""
echo "Finished: $(date)"
echo "=========================================="
