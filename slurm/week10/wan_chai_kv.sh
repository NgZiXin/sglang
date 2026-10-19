#!/bin/bash
#SBATCH --job-name=wan-chai-kv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=3:00:00
#SBATCH --output=wan-chai-kv-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"
export OUTPUT_DIR="$SCRATCH/sglang/outputs/week10/wan_chai_kv/${SLURM_JOB_ID:-manual}"

# Fixed experiment settings. A records the cache; B reuses it.
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
STEPS_A=40
STEPS_B=10
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
GUIDANCE_SCALE=5.0

setup_sglang_env

PROMPT_MANIFEST="$OUTPUT_DIR/prompts.json"
cp "$REPO_DIR/slurm/week10/vbench_22_prompts_wan_chai_kv.json" "$PROMPT_MANIFEST"

# prompt_en is the target B prompt used by VBench; prompt_a supplies the cache.
PROMPTS=$(python3 - "$PROMPT_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    manifest = json.load(file)
for prompt in manifest["prompts"]:
    print(prompt["prompt_id"], prompt["prompt_a"], prompt["prompt_en"],
          manifest["seed"], manifest["sample_index"], sep="\t")
PY
)

for FLOW_SHIFT in 3 5 7 9; do
  RUN_DIR="$OUTPUT_DIR/flow_shift_$FLOW_SHIFT"
  mkdir -p "$RUN_DIR/videos"

  while IFS=$'\t' read -r PAIR_ID PROMPT_A PROMPT_B SEED SAMPLE_INDEX; do
    PAIR_OUTPUT_DIR="$RUN_DIR/${PAIR_ID}/seed_$SEED"
    mkdir -p "$PAIR_OUTPUT_DIR"
    printf '\nRunning pair %s with seed %s\nA: %s\nB: %s\nOutputs: %s\n' \
      "$PAIR_ID" "$SEED" "$PROMPT_A" "$PROMPT_B" "$PAIR_OUTPUT_DIR"

    python3 -u "$REPO_DIR/slurm/week10/wan_chai_kv.py" \
      --model-path "$MODEL_PATH" \
      --prompt-a "$PROMPT_A" \
      --prompt-b "$PROMPT_B" \
      --seed "$SEED" \
      --steps-a "$STEPS_A" \
      --steps-b "$STEPS_B" \
      --flow-shift "$FLOW_SHIFT" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --num-frames "$NUM_FRAMES" \
      --fps "$FPS" \
      --guidance-scale "$GUIDANCE_SCALE" \
      --output-dir "$PAIR_OUTPUT_DIR" 2>&1 | tee "$PAIR_OUTPUT_DIR/experiment.log"

    GENERATED_VIDEO="$PAIR_OUTPUT_DIR/b_chai_kv.mp4"
    if ! test -s "$GENERATED_VIDEO"; then
      echo "ERROR: Generated video is missing or empty: $GENERATED_VIDEO" >&2
      exit 1
    fi
    # Only cached B videos go into the evaluation directory, using VBench names.
    cp "$GENERATED_VIDEO" "$RUN_DIR/videos/${PROMPT_B}-${SAMPLE_INDEX}.mp4"
  done <<< "$PROMPTS"

  run_vbench_evaluation \
    --video-dir "$RUN_DIR/videos" \
    --manifest "$PROMPT_MANIFEST" \
    --output-file "$RUN_DIR/vbench_scores.json"

  echo "Experiment completed. Outputs and VBench scores: $RUN_DIR"
done
