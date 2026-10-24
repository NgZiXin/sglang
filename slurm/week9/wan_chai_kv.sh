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
export OUTPUT_DIR="$SCRATCH/sglang/outputs/week10/wan_chai_kv_stepb_25/${SLURM_JOB_ID:-manual}"

# Experiment settings. STEPS_B applies to both cached and baseline B.
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
SEED=42
STEPS_A=50
STEPS_B=25
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16

setup_sglang_env

# Each row: pair_id|prompt_A|prompt_B.
while IFS='|' read -r PAIR_ID PROMPT_A PROMPT_B; do
  PAIR_OUTPUT_DIR="$OUTPUT_DIR/${PAIR_ID}/seed_$SEED"
  mkdir -p "$PAIR_OUTPUT_DIR"
  printf '\nRunning pair %s with seed %s\nA: %s\nB: %s\nOutputs: %s\n' \
    "$PAIR_ID" "$SEED" "$PROMPT_A" "$PROMPT_B" "$PAIR_OUTPUT_DIR"

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
    --output-dir "$PAIR_OUTPUT_DIR" 2>&1 | tee "$PAIR_OUTPUT_DIR/experiment.log"
done <<'PROMPT_PAIRS'
01_city_tram|A tram moves slowly through a cobblestone city square in morning light, wide shot.|A bus moves slowly through a cobblestone city square in morning light, wide shot.
02_beach_dog|A golden retriever runs along a sandy beach beside gentle waves at sunset, side view.|A golden retriever digs a hole in a sandy beach beside gentle waves at sunset, side view.
03_window_cat|A cat sits on a windowsill watching rain fall outside, medium shot.|A cat jumps down from a windowsill while rain falls outside, medium shot.
04_park_cyclist|A cyclist rides along a tree-lined park path in daylight, tracking shot.|A cyclist rides along a canal towpath in daylight, tracking shot.
05_coffee_cup|A hand pours coffee from a glass pot into a ceramic cup on a wooden cafe table, close-up shot.|A hand stirs coffee with a spoon in a ceramic cup on a wooden cafe table, close-up shot.
06_lake_boat|A small wooden boat drifts across a calm lake beneath misty mountains, wide shot.|A small wooden boat drifts along a narrow river beneath misty mountains, wide shot.
07_garden_butterfly|A butterfly flutters above lavender flowers in a sunlit garden, close-up shot.|A bee flutters above lavender flowers in a sunlit garden, close-up shot.
08_market_walk|A woman walks past a fruit stall at an outdoor market in daylight, medium shot.|A woman picks up an apple at a fruit stall at an outdoor market in daylight, medium shot.
09_park_fountain|Water splashes from a stone fountain in a quiet park lined with trees, fixed camera.|Water splashes from a stone fountain in a quiet courtyard lined with arches, fixed camera.
10_river_ducks|Two ducks paddle across a calm river beneath an old stone bridge, wide shot.|Two ducks take flight from a calm river beneath an old stone bridge, wide shot.
11_mountain_hiker|A hiker with a backpack walks along a rocky mountain trail under a clear sky, rear view.|A hiker with a backpack walks along a wooden suspension bridge under a clear sky, rear view.
12_kitchen_chef|A chef slices a loaf of bread on a wooden cutting board in a bright kitchen, close-up shot.|A chef slices a tomato on a wooden cutting board in a bright kitchen, close-up shot.
13_meadow_horse|A horse grazes in a grassy meadow with distant hills, side view.|A horse gallops through a grassy meadow with distant hills, side view.
14_rainy_street|A person holding an umbrella walks along a rain-soaked city sidewalk at night, tracking shot.|A person holding an umbrella walks across a rain-soaked pedestrian bridge at night, tracking shot.
15_forest_stream|A shallow stream flows over smooth rocks through a dense forest in soft morning light, fixed camera.|A shallow stream flows over smooth rocks through an open meadow in soft morning light, fixed camera.
16_balloons|A balloon floats above a grassy field in a gentle breeze, low-angle shot.|A kite floats above a grassy field in a gentle breeze, low-angle shot.
17_beach_waves|Gentle waves roll onto an empty sandy beach beneath a clear sky, wide shot.|Powerful waves crash onto an empty sandy beach beneath a clear sky, wide shot.
18_street_musician|A musician plays a flute beside a brick wall on a quiet street, medium shot.|A musician plays a saxophone beside a brick wall on a quiet street, medium shot.
19_snowy_cabin|Snow falls around a wooden cabin in a pine forest, fixed camera.|Snow falls around a camping tent in a pine forest, fixed camera.
20_garden_watering|A person waters potted flowers with a watering can on a sunny balcony, medium shot.|A person trims potted flowers with pruning shears on a sunny balcony, medium shot.
21_beach_party|A party on a sandy beach beside gentle waves, wide shot.|Gentle waves roll onto an empty sandy beach, wide shot.
PROMPT_PAIRS

# 01_add_vase|Sunlight falls across a wooden table in a quiet dining room, fixed camera.|Sunlight falls across a wooden table with a ceramic vase in a quiet dining room, fixed camera.
# 02_add_boat|Gentle ripples move across a calm lake beneath misty mountains, wide shot.|Gentle ripples move across a calm lake with a small wooden boat beneath misty mountains, wide shot.
# 03_add_bench|Leaves sway beside a tree-lined park path in soft morning light, wide shot.|Leaves sway beside a tree-lined park path with a wooden bench in soft morning light, wide shot.
# 04_add_basket|A cyclist rides along a quiet city street in daylight, side tracking shot.|A cyclist with a wicker basket on the bicycle rides along a quiet city street in daylight, side tracking shot.
# 05_add_spoon|Steam rises from a ceramic cup of coffee on a wooden cafe table, close-up shot.|Steam rises from a ceramic cup of coffee on a wooden cafe table with a silver spoon, close-up shot.
# 06_remove_umbrella|Gentle waves roll onto a sandy beach with a red beach umbrella at sunset, wide shot.|Gentle waves roll onto a sandy beach at sunset, wide shot.
# 07_remove_backpack|A hiker with a backpack walks along a rocky mountain trail under a clear sky, rear view.|A hiker walks along a rocky mountain trail under a clear sky, rear view.
# 08_remove_lamp|Rain falls outside a window beside an armchair with a floor lamp in a cozy living room, fixed camera.|Rain falls outside a window beside an armchair in a cozy living room, fixed camera.
# 09_remove_ball|A golden retriever sits beside a tennis ball on a grassy lawn in daylight, medium shot.|A golden retriever sits on a grassy lawn in daylight, medium shot.
# 10_remove_hat|A woman wearing a straw hat walks past a fruit stall at an outdoor market in daylight, medium shot.|A woman walks past a fruit stall at an outdoor market in daylight, medium shot.
# 11_refine_tram_colour|A red tram moves slowly through a cobblestone city square in morning light, wide shot.|A blue tram moves slowly through a cobblestone city square in morning light, wide shot.
# 12_refine_cup_colour|Steam rises from a white ceramic cup of coffee on a wooden cafe table, close-up shot.|Steam rises from a black ceramic cup of coffee on a wooden cafe table, close-up shot.
# 13_refine_umbrella_colour|A person holding a yellow umbrella walks along a rain-soaked city sidewalk at night, tracking shot.|A person holding a purple umbrella walks along a rain-soaked city sidewalk at night, tracking shot.
# 14_refine_balloon_colour|A green balloon floats above a grassy field in a gentle breeze, low-angle shot.|A pink balloon floats above a grassy field in a gentle breeze, low-angle shot.
# 15_refine_car_colour|A silver car drives along a coastal road beneath a clear sky, side tracking shot.|A red car drives along a coastal road beneath a clear sky, side tracking shot.
