#!/bin/bash
#SBATCH --job-name=wan-request-caching-pairs-neg-k5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=4:00:00
#SBATCH --output=wan-request-caching-pairs-neg-k5-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"
export OUTPUT_DIR="$SCRATCH/sglang/outputs/week9/wan_request_caching_prompt_pairs_negative_k5/${SLURM_JOB_ID:-manual}"

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
# Legacy selection arguments are accepted but ignored: the runner hard-codes k=5.
SIMILARITY_TO_K='[[0.95,25],[0.90,20],[0.85,15],[0.75,10],[0.65,5]]'

setup_sglang_env

# Each row: pair_id|prompt_A|prompt_B|negative_append.
# Append only the replaced A terms to B negatives; A keeps the model default.
# Use the original A phrase without adding synonyms (tram -> bus appends tram).
# B changes the motion, an object, or the scene while keeping related visual context.
while IFS='|' read -r PAIR_ID PROMPT_A PROMPT_B NEGATIVE_APPEND; do
  PAIR_OUTPUT_DIR="$OUTPUT_DIR/${PAIR_ID}/seed_$SEED"
  mkdir -p "$PAIR_OUTPUT_DIR"
  printf '\nRunning pair %s with seed %s\nA: %s\nB: %s\nOutputs: %s\n' \
    "$PAIR_ID" "$SEED" "$PROMPT_A" "$PROMPT_B" "$PAIR_OUTPUT_DIR"

  printf 'Append to B negative: %s\nk: 5\n' "$NEGATIVE_APPEND"

  python3 -u "$REPO_DIR/slurm/week9/wan_request_caching.py" \
    --model-path "$MODEL_PATH" \
    --prompt-a "$PROMPT_A" \
    --prompt-b "$PROMPT_B" \
    --negative-prompt-b-append "$NEGATIVE_APPEND" \
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
    --k-override 5 2>&1 | tee "$PAIR_OUTPUT_DIR/experiment.log"
done <<'PROMPT_PAIRS'
01_city_tram|A tram moves slowly through a cobblestone city square in morning light, wide shot.|A bus moves slowly through a cobblestone city square in morning light, wide shot.|tram
02_beach_dog|A golden retriever runs along a sandy beach beside gentle waves at sunset, side view.|A golden retriever digs a hole in a sandy beach beside gentle waves at sunset, side view.|runs along
03_window_cat|A cat sits on a windowsill watching rain fall outside, medium shot.|A cat jumps down from a windowsill while rain falls outside, medium shot.|sits on a windowsill watching rain
04_park_cyclist|A cyclist rides along a tree-lined park path in daylight, tracking shot.|A cyclist rides along a canal towpath in daylight, tracking shot.|tree-lined park path
05_coffee_cup|A hand pours coffee from a glass pot into a ceramic cup on a wooden cafe table, close-up shot.|A hand stirs coffee with a spoon in a ceramic cup on a wooden cafe table, close-up shot.|pours coffee from a glass pot
06_lake_boat|A small wooden boat drifts across a calm lake beneath misty mountains, wide shot.|A small wooden boat drifts along a narrow river beneath misty mountains, wide shot.|calm lake
07_garden_butterfly|A butterfly flutters above lavender flowers in a sunlit garden, close-up shot.|A bee flutters above lavender flowers in a sunlit garden, close-up shot.|butterfly
08_market_walk|A woman walks past a fruit stall at an outdoor market in daylight, medium shot.|A woman picks up an apple at a fruit stall at an outdoor market in daylight, medium shot.|walks past a fruit stall
09_park_fountain|Water splashes from a stone fountain in a quiet park lined with trees, fixed camera.|Water splashes from a stone fountain in a quiet courtyard lined with arches, fixed camera.|quiet park lined with trees
10_river_ducks|Two ducks paddle across a calm river beneath an old stone bridge, wide shot.|Two ducks take flight from a calm river beneath an old stone bridge, wide shot.|paddle across
11_mountain_hiker|A hiker with a backpack walks along a rocky mountain trail under a clear sky, rear view.|A hiker with a backpack walks along a wooden suspension bridge under a clear sky, rear view.|rocky mountain trail
12_kitchen_chef|A chef slices a loaf of bread on a wooden cutting board in a bright kitchen, close-up shot.|A chef slices a tomato on a wooden cutting board in a bright kitchen, close-up shot.|loaf of bread
13_meadow_horse|A horse grazes in a grassy meadow with distant hills, side view.|A horse gallops through a grassy meadow with distant hills, side view.|grazes
14_rainy_street|A person holding an umbrella walks along a rain-soaked city sidewalk at night, tracking shot.|A person holding an umbrella walks across a rain-soaked pedestrian bridge at night, tracking shot.|city sidewalk
15_forest_stream|A shallow stream flows over smooth rocks through a dense forest in soft morning light, fixed camera.|A shallow stream flows over smooth rocks through an open meadow in soft morning light, fixed camera.|dense forest
16_balloons|A balloon floats above a grassy field in a gentle breeze, low-angle shot.|A kite floats above a grassy field in a gentle breeze, low-angle shot.|balloon
17_beach_waves|Gentle waves roll onto an empty sandy beach beneath a clear sky, wide shot.|Powerful waves crash onto an empty sandy beach beneath a clear sky, wide shot.|gentle waves roll
18_street_musician|A musician plays a flute beside a brick wall on a quiet street, medium shot.|A musician plays a saxophone beside a brick wall on a quiet street, medium shot.|flute
19_snowy_cabin|Snow falls around a wooden cabin in a pine forest, fixed camera.|Snow falls around a camping tent in a pine forest, fixed camera.|wooden cabin
20_garden_watering|A person waters potted flowers with a watering can on a sunny balcony, medium shot.|A person trims potted flowers with pruning shears on a sunny balcony, medium shot.|waters potted flowers with a watering can
PROMPT_PAIRS
