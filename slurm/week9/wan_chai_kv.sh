#!/bin/bash
#SBATCH --job-name=wan-chai-kv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=wan-chai-kv-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"
export OUTPUT_DIR="$SCRATCH/sglang/outputs/week9/wan_chai_kv/${SLURM_JOB_ID:-manual}"

# Experiment settings. STEPS_B applies to both cached and baseline B.
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
PROMPT_A="A party on a sandy beach beside gentle waves, wide shot."
PROMPT_B="Gentle waves roll onto an empty sandy beach, wide shot."
SEED=42
STEPS_A=40
STEPS_B=10
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
GUIDANCE_SCALE=5.0

setup_sglang_env

python3 -u "$REPO_DIR/slurm/week9/wan_chai_kv.py" \
  --model-path "$MODEL_PATH" \
  --prompt-a "$PROMPT_A" \
  --prompt-b "$PROMPT_B" \
  --seed "$SEED" \
  --steps-a "$STEPS_A" \
  --steps-b "$STEPS_B" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num-frames "$NUM_FRAMES" \
  --fps "$FPS" \
  --guidance-scale "$GUIDANCE_SCALE" \
  --output-dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/experiment.log"
