#!/bin/bash
#SBATCH --job-name=wan-request-caching-pairs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:30:00
#SBATCH --output=wan-request-caching-pairs-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"
export OUTPUT_DIR="$SCRATCH/sglang/outputs/week12/official/k5/wan_request_caching_prompt_pairs/${SLURM_JOB_ID:-manual}"

# One seed per pair, shared by A and cached B; skip the full B baseline.
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
SEED=42
NUM_INFERENCE_STEPS=50
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
FLOW_SHIFT=8
GUIDANCE_SCALE=6
K_OVERRIDE=5
# Required by the Python CLI; similarity selection is currently disabled.
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
    --flow-shift "$FLOW_SHIFT" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --clip-model "$CLIP_MODEL" \
    --similarity-to-k "$SIMILARITY_TO_K" \
    --k-override "$K_OVERRIDE" \
    --skip-b-full \
    --output-dir "$PAIR_OUTPUT_DIR" 2>&1 | tee "$PAIR_OUTPUT_DIR/experiment.log"
done <<'PROMPT_PAIRS'
p01_subject_consistency|a horse walking across an open field|a horse galloping across an open field
p02_subject_consistency|a dog digging happily|a dog running happily
p03_scene|underwater rocky seabed|underwater coral reef
p05_temporal_flickering|In a still frame, a speed limit sign|In a still frame, a stop sign
p11_human_action|A person is playing harp|A person is playing piano
p13_color|a blue car|a red car
p14_color|a red bicycle|a blue bicycle
p17_appearance_style|The bund Shanghai, watercolor painting|The bund Shanghai, Van Gogh style
p18_appearance_style|A beautiful coastal beach in spring, waves lapping on sand, oil painting|A beautiful coastal beach in spring, waves lapping on sand, watercolor painting
p22_overall_consistency|Two pandas discussing an academic paper beside a laptop.|Two pandas discussing an academic paper.
p23_subject_consistency|a person holding a burger|a person eating a burger
p24_scene|indoor botanical garden|botanical garden
p29_color|a yellow car|a green car
p31_appearance_style|a shark is swimming in the ocean, watercolor painting|a shark is swimming in the ocean, oil painting
p32_temporal_style|a shark and a dolphin are swimming in the ocean, tilt up|a shark is swimming in the ocean, tilt up
PROMPT_PAIRS

# 23_city_pigeons|A flock of pigeons gathers on a cobblestone city square in daylight, wide shot.|A flock of pigeons gathers on a grassy hillside in daylight, wide shot.
# 06_snowy_cabin|Snow falls around a wooden cabin in a pine forest, fixed camera.|Snow falls around a camping tent in a pine forest, fixed camera.
# 04_add_basket|A cyclist rides along a quiet city street in daylight, side tracking shot.|A cyclist with a wicker basket on the bicycle rides along a quiet city street in daylight, side tracking shot.
# 07_refine_car_colour|A silver car drives along a coastal road beneath a clear sky, side tracking shot.|A red car drives along a coastal road beneath a clear sky, side tracking shot.
# 08_beach_party|A party on a sandy beach beside gentle waves, wide shot.|Gentle waves roll onto an empty sandy beach, wide shot.
# 10_window_cat|A cat sits on a windowsill watching rain fall outside, medium shot.|A cat jumps down from a windowsill while rain falls outside, medium shot.

# 22_forest_runner|A runner jogs along a shaded forest trail in morning light, tracking shot.|A runner jogs along a seaside promenade in morning light, tracking shot.
# 23_city_pigeons|A flock of pigeons gathers on a cobblestone city square in daylight, wide shot.|A flock of pigeons gathers on a grassy hillside in daylight, wide shot.
# 01_add_vase|Sunlight falls across a wooden table in a quiet dining room, fixed camera.|Sunlight falls across a wooden table with a ceramic vase in a quiet dining room, fixed camera.
# 02_city_tram|A tram moves slowly through a cobblestone city square in morning light, wide shot.|A bus moves slowly through a cobblestone city square in morning light, wide shot.
# 03_beach_dog|A golden retriever runs along a sandy beach beside gentle waves at sunset, side view.|A golden retriever digs a hole in a sandy beach beside gentle waves at sunset, side view.
# 04_add_basket|A cyclist rides along a quiet city street in daylight, side tracking shot.|A cyclist with a wicker basket on the bicycle rides along a quiet city street in daylight, side tracking shot.
# 06_snowy_cabin|Snow falls around a wooden cabin in a pine forest, fixed camera.|Snow falls around a camping tent in a pine forest, fixed camera.
# 07_refine_car_colour|A silver car drives along a coastal road beneath a clear sky, side tracking shot.|A red car drives along a coastal road beneath a clear sky, side tracking shot.
# 08_beach_party|A party on a sandy beach beside gentle waves, wide shot.|Gentle waves roll onto an empty sandy beach, wide shot.
# 09_add_boat|Gentle ripples move across a calm empty lake beneath misty mountains, wide shot.|Gentle ripples move across a calm lake with a small wooden boat beneath misty mountains, wide shot.
# 10_window_cat|A cat sits on a windowsill watching rain fall outside, medium shot.|A cat jumps down from a windowsill while rain falls outside, medium shot.
