#!/bin/bash
#SBATCH --job-name=wan-request-caching
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=wan-request-caching-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"
export OUTPUT_DIR="$SCRATCH/sglang/outputs/week9/wan_request_caching/${SLURM_JOB_ID:-manual}"
# Experiment settings.
MODEL_PATH="Wan-AI/Wan2.1-T2V-14B-Diffusers"
PROMPT_A="A city square"
PROMPT_B="A tram moves through a city square"
SEED=42
NUM_INFERENCE_STEPS=40
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
GUIDANCE_SCALE=5.0
CLIP_MODEL="openai/clip-vit-large-patch14"
# Paper Figure 10; initial thresholds, not calibrated for Wan video generation.
SIMILARITY_TO_K='[[0.95,25],[0.90,20],[0.85,15],[0.75,10],[0.65,5]]'
K_OVERRIDE="" # Empty uses similarity; set e.g. 15 for a fixed-k experiment.

setup_sglang_env

EXTRA_ARGS=()
if [[ -n "$K_OVERRIDE" ]]; then
  EXTRA_ARGS+=(--k-override "$K_OVERRIDE")
fi

python3 -u "$REPO_DIR/slurm/week9/wan_request_caching.py" \
  --model-path "$MODEL_PATH" \
  --prompt-a "$PROMPT_A" \
  --prompt-b "$PROMPT_B" \
  --seed "$SEED" \
  --num-inference-steps "$NUM_INFERENCE_STEPS" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num-frames "$NUM_FRAMES" \
  --fps "$FPS" \
  --guidance-scale "$GUIDANCE_SCALE" \
  --clip-model "$CLIP_MODEL" \
  --similarity-to-k "$SIMILARITY_TO_K" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/experiment.log"
