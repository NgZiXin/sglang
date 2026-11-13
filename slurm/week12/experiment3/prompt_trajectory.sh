#!/bin/bash
#SBATCH --job-name=wan-t2v-prompt-similarity
#SBATCH --partition=gpu
#SBATCH --gpus=h100-96
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:30:00
#SBATCH --output=wan-t2v-prompt-similarity-%j.out

set -euo pipefail

source "$HOME/cp4101/sglang/slurm/common.sh"

FLOW_SHIFT=8
GUIDANCE_SCALE=6
NUM_INFERENCE_STEPS=50
OUTPUT_DIR="$SCRATCH/sglang/outputs/week12/experiment3/compare_trajectory"
RUN_ID=1 # Required by perf summary csv

MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
NUM_GPUS=1
ULYSSES_DEGREE=1
RING_DEGREE=1

setup_sglang_env

RUN_DIR="$OUTPUT_DIR/steps_${NUM_INFERENCE_STEPS}_shift_${FLOW_SHIFT}_${SLURM_JOB_ID}"
PROMPT_MANIFEST="$RUN_DIR/prompts.json"
SUMMARY_CSV="$RUN_DIR/perf_summary.csv"
mkdir "$RUN_DIR"
mkdir "$RUN_DIR/videos" "$RUN_DIR/perf" "$RUN_DIR/logs"
cp "$REPO_DIR/slurm/week12/experiment1/wan_mix_cache_11_prompts.json" "$PROMPT_MANIFEST"

python3 -u "$REPO_DIR/slurm/week12/experiment3/compare_prompt_trajectories.py" \
  --model-path "$MODEL_PATH" \
  --prompt-manifest "$PROMPT_MANIFEST" \
  --run-dir "$RUN_DIR" \
  --height "$HEIGHT" --width "$WIDTH" --num-frames "$NUM_FRAMES" --fps "$FPS" \
  --num-inference-steps "$NUM_INFERENCE_STEPS" \
  --flow-shift "$FLOW_SHIFT" --guidance-scale "$GUIDANCE_SCALE" \
  2>&1 | tee "$RUN_DIR/logs/experiment.log"

for PERF_PATH in "$RUN_DIR"/perf/*.json; do
  FILE_NAME=${PERF_PATH##*/}
  LABEL=${FILE_NAME%_seed*.json}
  SEED=${FILE_NAME##*_seed}
  SEED=${SEED%.json}
  append_perf_summary
done

echo "Results: $RUN_DIR/step_similarity.csv"
