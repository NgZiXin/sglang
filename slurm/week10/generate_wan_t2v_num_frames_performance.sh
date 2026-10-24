#!/bin/bash
#SBATCH --job-name=wan-t2v-num-frames
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=wan-t2v-num-frames-%j.out

set -euo pipefail

source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week10/num_frames_performance"
RUN_PREFIX="wan_t2v_num_frames_${SLURM_JOB_ID:-manual}"
RUN_DIR="$OUTPUT_DIR/$RUN_PREFIX"
PROMPT_MANIFEST="$RUN_DIR/prompts.json"
SUMMARY_CSV="$RUN_DIR/perf_summary.csv"

MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
HEIGHT=480
WIDTH=832
NUM_FRAMES_LIST=(49 81 113 145 177)
FPS=16
NUM_INFERENCE_STEPS=40
NUM_GPUS=1
REPEATS=1

setup_sglang_env
mkdir -p "$RUN_DIR/videos" "$RUN_DIR/perf"
cp "$REPO_DIR/slurm/week10/vbench_33_prompts_wan_chai_kv.json" "$PROMPT_MANIFEST"

PROMPTS=$(python3 - "$PROMPT_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    manifest = json.load(file)

prompts = manifest.get("prompts")
if not isinstance(prompts, list) or not prompts:
    raise SystemExit(f"ERROR: No prompts found in {sys.argv[1]}")

for prompt in prompts:
    print(
        prompt["prompt_id"],
        prompt["prompt_en"],
        manifest["seed"],
        sep="\t",
    )
PY
)

for NUM_FRAMES in "${NUM_FRAMES_LIST[@]}"; do
  while IFS=$'\t' read -r PROMPT_ID PROMPT SEED; do
    for RUN_ID in $(seq 1 "$REPEATS"); do
      LABEL="${PROMPT_ID}_frames_${NUM_FRAMES}"
      FILE_NAME="${LABEL}_run_${RUN_ID}"
      PERF_PATH="$RUN_DIR/perf/${FILE_NAME}.json"
      OUTPUT_PATH="$RUN_DIR/videos/${FILE_NAME}.mp4"

      echo "Generating ${PROMPT_ID} with ${NUM_FRAMES} frames (run ${RUN_ID}/${REPEATS}): ${PROMPT}"
      sglang generate \
        --model-path "$MODEL_PATH" \
        --num-gpus "$NUM_GPUS" \
        --encoder-parallel replicate \
        --cfg-parallel-size 1 \
        --prompt "$PROMPT" \
        --height "$HEIGHT" \
        --width "$WIDTH" \
        --num-frames "$NUM_FRAMES" \
        --fps "$FPS" \
        --num-inference-steps "$NUM_INFERENCE_STEPS" \
        --seed "$SEED" \
        --save-output \
        --output-file-path "$OUTPUT_PATH" \
        --perf-dump-path "$PERF_PATH"

      append_perf_summary
    done
  done <<< "$PROMPTS"
done

echo "Experiment completed. Performance summary: $SUMMARY_CSV"
