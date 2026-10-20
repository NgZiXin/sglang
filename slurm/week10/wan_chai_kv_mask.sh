#!/bin/bash
#SBATCH --job-name=wan-chai-mask
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=01:00:00
#SBATCH --output=wan-chai-mask-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"
export OUTPUT_DIR="$SCRATCH/sglang/outputs/week10/wan_chai_kv_mask/${SLURM_JOB_ID:-manual}/16_balloons"
setup_sglang_env

# One A and three B videos. All B variants use identical sampling settings.
# Lower thresholds exclude more cached tokens. 0.5 is an initial experiment.
python3 -u "$REPO_DIR/slurm/week10/wan_chai_kv_mask.py" \
  --model-path "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" \
  --prompt-a "A balloon floats above a grassy field in a gentle breeze, low-angle shot." \
  --prompt-b "A kite floats above a grassy field in a gentle breeze, low-angle shot." \
  --exclude-word balloon \
  --mask-threshold 0.5 \
  --steps-a 40 \
  --steps-b 10 \
  --seed 42 \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --fps 16 \
  --guidance-scale 5.0 \
  --output-dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/experiment.log"
