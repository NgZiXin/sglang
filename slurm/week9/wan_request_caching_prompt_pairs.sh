#!/bin/bash
#SBATCH --job-name=wan-request-caching-pairs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=10:00:00
#SBATCH --output=wan-request-caching-pairs-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"
export OUTPUT_DIR="$SCRATCH/sglang/outputs/week9/wan_request_caching_prompt_pairs/${SLURM_JOB_ID:-manual}"

# One seed per pair, shared by A, cached B, and the full B baseline.
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
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

# Each row below is pair_id|prompt_A|prompt_B. Keep pipes out of prompts.
# B changes one detail while preserving the scene, action, and framing otherwise.
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
    --guidance-scale "$GUIDANCE_SCALE" \
    --clip-model "$CLIP_MODEL" \
    --similarity-to-k "$SIMILARITY_TO_K" \
    --output-dir "$PAIR_OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$PAIR_OUTPUT_DIR/experiment.log"
done <<'PROMPT_PAIRS'
01_city_tram|A red tram moves slowly through a cobblestone city square in morning light, wide shot.|A blue tram moves slowly through a cobblestone city square in morning light, wide shot.
02_beach_dog|A golden retriever walks along a sandy beach beside gentle waves at sunset, side view.|A golden retriever runs along a sandy beach beside gentle waves at sunset, side view.
03_window_cat|An orange cat sits on a windowsill watching rain fall outside, close-up shot.|A gray cat sits on a windowsill watching rain fall outside, close-up shot.
04_park_cyclist|A cyclist wearing a red jacket rides along a tree-lined park path, tracking shot.|A cyclist wearing a yellow jacket rides along a tree-lined park path, tracking shot.
05_coffee_cup|Steam rises from a white ceramic cup of coffee on a wooden cafe table, close-up shot.|Steam rises from a blue ceramic cup of coffee on a wooden cafe table, close-up shot.
06_lake_boat|A small wooden boat drifts across a calm lake beneath misty mountains at sunrise, wide shot.|A small wooden boat drifts across a calm lake beneath misty mountains at sunset, wide shot.
07_garden_butterfly|A yellow butterfly flutters above purple lavender flowers in a sunlit garden, close-up shot.|A white butterfly flutters above purple lavender flowers in a sunlit garden, close-up shot.
08_market_walk|A woman carrying a canvas bag walks through an outdoor fruit market in daylight, tracking shot.|A woman carrying a wicker basket walks through an outdoor fruit market in daylight, tracking shot.
09_park_fountain|Water splashes from a stone fountain surrounded by red flowers in a quiet park, fixed camera.|Water splashes from a stone fountain surrounded by white flowers in a quiet park, fixed camera.
10_river_ducks|Two ducks paddle across a calm river beneath an old stone bridge, wide shot.|Three ducks paddle across a calm river beneath an old stone bridge, wide shot.
11_mountain_hiker|A hiker with a blue backpack walks along a rocky mountain trail under a clear sky, rear view.|A hiker with a green backpack walks along a rocky mountain trail under a clear sky, rear view.
12_kitchen_chef|A chef chops a carrot on a wooden cutting board in a bright kitchen, close-up shot.|A chef chops a cucumber on a wooden cutting board in a bright kitchen, close-up shot.
13_meadow_horse|A brown horse walks through a grassy meadow with distant hills, side view.|A white horse walks through a grassy meadow with distant hills, side view.
14_rainy_street|A person holding a red umbrella walks along a rain-soaked city sidewalk at night, tracking shot.|A person holding a blue umbrella walks along a rain-soaked city sidewalk at night, tracking shot.
15_forest_stream|A shallow stream flows over smooth rocks through a green forest in soft morning light, fixed camera.|A shallow stream flows over smooth rocks through a green forest in soft evening light, fixed camera.
16_balloons|A red balloon floats slowly above a grassy field under a clear blue sky, low-angle shot.|A yellow balloon floats slowly above a grassy field under a clear blue sky, low-angle shot.
17_beach_waves|Gentle waves roll onto an empty sandy beach beneath a clear sky, wide shot.|Gentle waves roll onto an empty sandy beach beneath a cloudy sky, wide shot.
18_street_musician|A musician plays an acoustic guitar beside a brick wall on a quiet street, medium shot.|A musician plays a violin beside a brick wall on a quiet street, medium shot.
19_snowy_cabin|Snow falls around a wooden cabin with a glowing window in a pine forest, fixed camera.|Snow falls around a wooden cabin with a dark window in a pine forest, fixed camera.
20_garden_watering|A person waters potted red flowers on a sunny balcony with a metal watering can, medium shot.|A person waters potted pink flowers on a sunny balcony with a metal watering can, medium shot.
PROMPT_PAIRS
