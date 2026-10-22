#!/bin/bash
#SBATCH --job-name=wan-t2v
#SBATCH --partition=gpu
#SBATCH --gpus=h100-96
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=7:00:00
#SBATCH --output=wan-t2v-%j.out

set -euo pipefail

source "$HOME/cp4101/sglang/slurm/common.sh"

# Scheduler params
FLOW_SHIFT=5 # Model default; sets the range before Beta sampling.

OUTPUT_DIR="$SCRATCH/sglang/outputs/week10/vary_denoise_2"
RUN_ID=1 # Required by perf summary csv

MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
NUM_GPUS=1
ULYSSES_DEGREE=1
RING_DEGREE=1
WAN_TIMESTEP_SCHEDULE=0 # 0=flow shift, 1=early-fine, 2=Beta(0.6, 0.6), 3=Beta(0.6, 0.4).

setup_sglang_env

for NUM_INFERENCE_STEPS in 5 10 20 30 40 50; do
RUN_DIR="$OUTPUT_DIR/steps_${NUM_INFERENCE_STEPS}_schedule_${WAN_TIMESTEP_SCHEDULE}_shift_${FLOW_SHIFT}_${SLURM_JOB_ID}"
PROMPT_MANIFEST="$RUN_DIR/prompts.json"
SUMMARY_CSV="$RUN_DIR/perf_summary.csv"
mkdir "$RUN_DIR"
mkdir "$RUN_DIR/videos" "$RUN_DIR/perf" "$RUN_DIR/logs"
# Save the prompt manifest to the run directory for reproducibility.
cp "$REPO_DIR/slurm/week10/vbench_33_prompts_wan_chai_kv.json" "$PROMPT_MANIFEST"

# Bash reads the prompt rows directly from this inline JSON reader.
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

while IFS=$'\t' read -r LABEL PROMPT SEED SAMPLE_INDEX; do
  FILE_NAME="${LABEL}_seed${SEED}"
  PERF_PATH="$RUN_DIR/perf/$FILE_NAME.json"
  GENERATED_VIDEO="$RUN_DIR/videos/$FILE_NAME.mp4"
  VIDEO_PATH="$RUN_DIR/videos/${PROMPT}-${SAMPLE_INDEX}.mp4" # VBench filename format.

  echo "Generating ${LABEL}, seed=${SEED}: ${PROMPT}"
  sglang generate \
    --model-path "$MODEL_PATH" \
    --num-gpus "$NUM_GPUS" \
    --sp-degree "$NUM_GPUS" \
    --ulysses-degree "$ULYSSES_DEGREE" \
    --ring-degree "$RING_DEGREE" \
    --prompt "$PROMPT" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num-frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --num-inference-steps "$NUM_INFERENCE_STEPS" \
    --flow-shift "$FLOW_SHIFT" \
    --wan-timestep-schedule "$WAN_TIMESTEP_SCHEDULE" \
    --seed "$SEED" \
    --enable-cache-dit false \
    --save-output \
    --perf-dump-path "$PERF_PATH" \
    --output-file-path "$GENERATED_VIDEO" 2>&1 | tee "$RUN_DIR/logs/$FILE_NAME.log"
  if ! test -s "$GENERATED_VIDEO"; then
    echo "ERROR: Generated video is missing or empty: $GENERATED_VIDEO" >&2
    exit 1
  fi  
  # Rename after saving because SGLang replaces spaces and punctuation.
  mv "$GENERATED_VIDEO" "$VIDEO_PATH"
  append_perf_summary
done <<< "$PROMPTS"

# Keep evaluation failures local to this batch. The subshell retains errexit
# so an environment/setup failure cannot fall through into the evaluator.
set +e
(
  set -e
  run_vbench_evaluation \
    --video-dir "$RUN_DIR/videos" \
    --manifest "$PROMPT_MANIFEST" \
    --output-file "$RUN_DIR/vbench_scores.json"
) 2>&1 | tee "$RUN_DIR/logs/vbench.log"
EVALUATION_STATUS=("${PIPESTATUS[@]}")
set -e

if (( EVALUATION_STATUS[0] == 0 && EVALUATION_STATUS[1] == 0 )); then
  printf '{"status":"completed","evaluation_exit_code":0,"log_exit_code":0,"log_file":"logs/vbench.log"}\n' \
    > "$RUN_DIR/vbench_status.json"
  echo "Generation and evaluation completed: $RUN_DIR"
else
  printf '{"status":"failed","evaluation_exit_code":%d,"log_exit_code":%d,"log_file":"logs/vbench.log"}\n' \
    "${EVALUATION_STATUS[0]}" "${EVALUATION_STATUS[1]}" > "$RUN_DIR/vbench_status.json"
  echo "WARNING: Evaluation failed for ${NUM_INFERENCE_STEPS} steps; generated videos are saved. See $RUN_DIR/logs/vbench.log. Continuing the sweep." >&2
fi
done
