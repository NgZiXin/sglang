#!/bin/bash
#SBATCH --job-name=wan-request-caching-pairs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=2:30:00
#SBATCH --output=wan-request-caching-pairs-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"
export OUTPUT_DIR="$SCRATCH/sglang/outputs/week11/wan_request_caching_prompt_pairs/${SLURM_JOB_ID:-manual}"

# One seed per pair, shared by A, cached B, and the full B baseline.
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
SEED=42
NUM_INFERENCE_STEPS=50
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
# Required by the Python CLI; the runner currently hardcodes k=5.
CLIP_MODEL="openai/clip-vit-large-patch14"
SIMILARITY_TO_K='[[0.95,25],[0.90,20],[0.85,15],[0.75,10],[0.65,5]]'

setup_sglang_env

# Each row below is pair_id|prompt_A|prompt_B. Keep pipes out of prompts.
while IFS='|' read -r PAIR_ID PROMPT_A PROMPT_B; do
  PAIR_OUTPUT_DIR="$OUTPUT_DIR/${PAIR_ID}/seed_$SEED"
  mkdir -p "$PAIR_OUTPUT_DIR"
  printf '\nRunning pair %s with seed %s\nA: %s\nB: %s\nOutputs: %s\n' \
    "$PAIR_ID" "$SEED" "$PROMPT_A" "$PROMPT_B" "$PAIR_OUTPUT_DIR"

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
    --clip-model "$CLIP_MODEL" \
    --similarity-to-k "$SIMILARITY_TO_K" \
    --output-dir "$PAIR_OUTPUT_DIR" 2>&1 | tee "$PAIR_OUTPUT_DIR/experiment.log"
done <<'PROMPT_PAIRS'
01_add_vase|Sunlight falls across a wooden table in a quiet dining room, fixed camera.|Sunlight falls across a wooden table with a ceramic vase in a quiet dining room, fixed camera.
02_city_tram|A tram moves slowly through a cobblestone city square in morning light, wide shot.|A bus moves slowly through a cobblestone city square in morning light, wide shot.
03_beach_dog|A golden retriever runs along a sandy beach beside gentle waves at sunset, side view.|A golden retriever digs a hole in a sandy beach beside gentle waves at sunset, side view.
04_park_fountain|Water splashes from a stone fountain in a quiet park lined with trees, fixed camera.|Water splashes from a stone fountain in a quiet courtyard lined with arches, fixed camera.
05_park_cyclist|A cyclist rides along a tree-lined park path in daylight, tracking shot.|A cyclist rides along a canal towpath in daylight, tracking shot.
06_snowy_cabin|Snow falls around a wooden cabin in a pine forest, fixed camera.|Snow falls around a camping tent in a pine forest, fixed camera.
07_refine_car_colour|A silver car drives along a coastal road beneath a clear sky, side tracking shot.|A red car drives along a coastal road beneath a clear sky, side tracking shot.
08_remove_umbrella|Gentle waves roll onto a sandy beach with a red beach umbrella at sunset, wide shot.|Gentle waves roll onto a sandy beach at sunset, wide shot.
PROMPT_PAIRS
