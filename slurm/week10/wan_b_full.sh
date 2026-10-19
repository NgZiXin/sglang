#!/bin/bash
#SBATCH --job-name=wan-b-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=2:00:00
#SBATCH --output=wan-b-full-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"
export OUTPUT_DIR="$SCRATCH/sglang/outputs/week10/wan_b_full/${SLURM_JOB_ID:-manual}"

# Generate only the target B prompts, without cache reuse.
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
NUM_INFERENCE_STEPS=40
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
GUIDANCE_SCALE=5.0

setup_sglang_env

PROMPT_MANIFEST="$OUTPUT_DIR/prompts.json"
cp "$REPO_DIR/slurm/week10/vbench_22_prompts_wan_chai_kv.json" "$PROMPT_MANIFEST"

# prompt_en is the same target B prompt evaluated in wan_chai_kv.sh.
PROMPTS=$(python3 - "$PROMPT_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    manifest = json.load(file)
for prompt in manifest["prompts"]:
    print(prompt["prompt_id"], prompt["prompt_en"],
          manifest["seed"], manifest["sample_index"], sep="\t")
PY
)

for FLOW_SHIFT in 5; do
  RUN_DIR="$OUTPUT_DIR/flow_shift_$FLOW_SHIFT"
  mkdir -p "$RUN_DIR/videos" "$RUN_DIR/perf" "$RUN_DIR/logs"

  while IFS=$'\t' read -r LABEL PROMPT SEED SAMPLE_INDEX; do
    FILE_NAME="${LABEL}_seed${SEED}"
    GENERATED_VIDEO="$RUN_DIR/videos/$FILE_NAME.mp4"
    VIDEO_PATH="$RUN_DIR/videos/${PROMPT}-${SAMPLE_INDEX}.mp4"

    echo "Generating ${LABEL}, seed=${SEED}, flow_shift=${FLOW_SHIFT}: ${PROMPT}"
    sglang generate \
      --model-path "$MODEL_PATH" \
      --num-gpus 1 \
      --sp-degree 1 \
      --ulysses-degree 1 \
      --ring-degree 1 \
      --attention-backend fa \
      --warmup-mode off \
      --prompt "$PROMPT" \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --num-frames "$NUM_FRAMES" \
      --fps "$FPS" \
      --num-inference-steps "$NUM_INFERENCE_STEPS" \
      --flow-shift "$FLOW_SHIFT" \
      --guidance-scale "$GUIDANCE_SCALE" \
      --cfg-gate-step 1.0 \
      --seed "$SEED" \
      --enable-cache-dit false \
      --save-output \
      --perf-dump-path "$RUN_DIR/perf/$FILE_NAME.json" \
      --output-file-path "$GENERATED_VIDEO" 2>&1 | tee "$RUN_DIR/logs/$FILE_NAME.log"

    if ! test -s "$GENERATED_VIDEO"; then
      echo "ERROR: Generated video is missing or empty: $GENERATED_VIDEO" >&2
      exit 1
    fi
    # Rename after saving because SGLang replaces spaces and punctuation.
    mv "$GENERATED_VIDEO" "$VIDEO_PATH"
  done <<< "$PROMPTS"

  run_vbench_evaluation \
    --video-dir "$RUN_DIR/videos" \
    --manifest "$PROMPT_MANIFEST" \
    --output-file "$RUN_DIR/vbench_scores.json"

  echo "Experiment completed. Outputs and VBench scores: $RUN_DIR"
done
