#!/bin/bash
#SBATCH --job-name=wan-step1-similarity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0:30:00
#SBATCH --output=wan-step1-similarity-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week11/wan_step1_similarity/${SLURM_JOB_ID:-manual}"
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
SEED=42
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FLOW_SHIFT=5.0

setup_sglang_env

# The Python script fixes num_inference_steps=1 and skips video decoding.
python3 -u "$REPO_DIR/slurm/week11/wan_step1_similarity.py" \
  --model-path "$MODEL_PATH" \
  --output-csv "$OUTPUT_DIR/similarity.csv" \
  --seed "$SEED" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num-frames "$NUM_FRAMES" \
  --flow-shift "$FLOW_SHIFT" 2>&1 | tee "$OUTPUT_DIR/experiment.log"

echo "Results: $OUTPUT_DIR/similarity.csv"
