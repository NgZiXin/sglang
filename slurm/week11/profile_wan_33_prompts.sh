#!/bin/bash
#SBATCH --job-name=wan-block-profile
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:30:00
#SBATCH --output=wan-block-profile-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week11/block_step_profile"
RUN_DIR="$OUTPUT_DIR/wan_50_steps_${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
PROFILER="$REPO_DIR/slurm/week11/profile_wan.py"
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
NUM_INFERENCE_STEPS=50

setup_sglang_env
mkdir -p "$RUN_DIR"/{videos,raw,perf,logs}
cp "$REPO_DIR/slurm/week10/vbench_33_prompts_wan_chai_kv.json" "$RUN_DIR/prompts.json"

PROMPTS=$(python3 - "$RUN_DIR/prompts.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as file:
    manifest = json.load(file)
for prompt in manifest["prompts"]:
    print(prompt["prompt_id"], prompt["prompt_en"], manifest["seed"], sep="\t")
PY
)

while IFS=$'\t' read -r LABEL PROMPT SEED; do
  echo "Profiling $LABEL: $NUM_INFERENCE_STEPS steps"
  WAN_PROFILE_PATH="$RUN_DIR/raw/$LABEL.json" python3 "$PROFILER" generate \
    --model-path "$MODEL_PATH" \
    --encoder-parallel replicate \
    --prompt "$PROMPT" \
    --height "$HEIGHT" --width "$WIDTH" \
    --num-frames "$NUM_FRAMES" --fps "$FPS" \
    --num-inference-steps "$NUM_INFERENCE_STEPS" \
    --seed "$SEED" --save-output \
    --output-file-path "$RUN_DIR/videos/$LABEL.mp4" \
    --perf-dump-path "$RUN_DIR/perf/$LABEL.json" \
    2>&1 | tee "$RUN_DIR/logs/$LABEL.log"
  test -s "$RUN_DIR/raw/$LABEL.json"
  test -s "$RUN_DIR/perf/$LABEL.json"
  test -s "$RUN_DIR/videos/$LABEL.mp4"
done <<< "$PROMPTS"

python3 "$PROFILER" summarize "$RUN_DIR"
