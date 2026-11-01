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
mkdir -p "$OUTPUT_DIR"

# The Python script fixes num_inference_steps=1 and skips video decoding.
python3 -u "$REPO_DIR/slurm/week11/wan_step1_similarity.py" \
  --model-path "$MODEL_PATH" \
  --prompt-pairs - \
  --output-csv "$OUTPUT_DIR/similarity.csv" \
  --seed "$SEED" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num-frames "$NUM_FRAMES" \
  --flow-shift "$FLOW_SHIFT" 2>&1 <<'PROMPT_PAIRS' | tee "$OUTPUT_DIR/experiment.log"
22_forest_runner|A runner jogs along a shaded forest trail in morning light, tracking shot.|A runner jogs along a seaside promenade in morning light, tracking shot.
23_city_pigeons|A flock of pigeons gathers on a cobblestone city square in daylight, wide shot.|A flock of pigeons gathers on a grassy hillside in daylight, wide shot.
01_add_vase|Sunlight falls across a wooden table in a quiet dining room, fixed camera.|Sunlight falls across a wooden table with a ceramic vase in a quiet dining room, fixed camera.
02_city_tram|A tram moves slowly through a cobblestone city square in morning light, wide shot.|A bus moves slowly through a cobblestone city square in morning light, wide shot.
03_beach_dog|A golden retriever runs along a sandy beach beside gentle waves at sunset, side view.|A golden retriever digs a hole in a sandy beach beside gentle waves at sunset, side view.
04_add_basket|A cyclist rides along a quiet city street in daylight, side tracking shot.|A cyclist with a wicker basket on the bicycle rides along a quiet city street in daylight, side tracking shot.
06_snowy_cabin|Snow falls around a wooden cabin in a pine forest, fixed camera.|Snow falls around a camping tent in a pine forest, fixed camera.
07_refine_car_colour|A silver car drives along a coastal road beneath a clear sky, side tracking shot.|A red car drives along a coastal road beneath a clear sky, side tracking shot.
08_beach_party|A party on a sandy beach beside gentle waves, wide shot.|Gentle waves roll onto an empty sandy beach, wide shot.
09_add_boat|Gentle ripples move across a calm empty lake beneath misty mountains, wide shot.|Gentle ripples move across a calm lake with a small wooden boat beneath misty mountains, wide shot.
10_window_cat|A cat sits on a windowsill watching rain fall outside, medium shot.|A cat jumps down from a windowsill while rain falls outside, medium shot.
PROMPT_PAIRS

echo "Results: $OUTPUT_DIR/similarity.csv"
