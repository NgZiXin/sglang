#!/bin/bash
#SBATCH --job-name=wan-t2v-prompt-trajectory-chai-kv
#SBATCH --partition=gpu
#SBATCH --gpus=h100-96
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=2:30:00
#SBATCH --output=wan-t2v-prompt-trajectory-chai-kv-%j.out

set -euo pipefail

source "$HOME/cp4101/sglang/slurm/common.sh"

FLOW_SHIFT=8
GUIDANCE_SCALE=6
STEPS_A=50
STEPS_EN=10
OUTPUT_DIR="$SCRATCH/sglang/outputs/week12/experiment3/compare_trajectory_chai_kv"
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

RUN_DIR="$OUTPUT_DIR/steps_${STEPS_A}_${STEPS_EN}_shift_${FLOW_SHIFT}_${SLURM_JOB_ID}"
PROMPT_MANIFEST="$RUN_DIR/prompts.json"
SUMMARY_CSV="$RUN_DIR/perf_summary.csv"
mkdir "$RUN_DIR"
mkdir "$RUN_DIR/videos" "$RUN_DIR/perf" "$RUN_DIR/logs" "$RUN_DIR/kv"
cp "$REPO_DIR/slurm/week10/vbench_33_prompts_wan_chai_kv.json" "$PROMPT_MANIFEST"

python3 -u "$REPO_DIR/slurm/week12/experiment3/compare_prompt_trajectories_chai_kv.py" \
  --model-path "$MODEL_PATH" \
  --prompt-manifest "$PROMPT_MANIFEST" \
  --run-dir "$RUN_DIR" \
  --height "$HEIGHT" --width "$WIDTH" --num-frames "$NUM_FRAMES" --fps "$FPS" \
  --steps-a "$STEPS_A" --steps-en "$STEPS_EN" \
  --flow-shift "$FLOW_SHIFT" --guidance-scale "$GUIDANCE_SCALE" \
  2>&1 | tee "$RUN_DIR/logs/experiment.log"

for PERF_PATH in "$RUN_DIR"/perf/*.json; do
  FILE_NAME=${PERF_PATH##*/}
  LABEL=${FILE_NAME%_seed*.json}
  SEED=${FILE_NAME##*_seed}
  SEED=${SEED%.json}
  case "$LABEL" in
    *_prompt_a) NUM_INFERENCE_STEPS=$STEPS_A ;;
    *_prompt_en_chai_kv) NUM_INFERENCE_STEPS=$STEPS_EN ;;
    *) echo "Unexpected perf file: $PERF_PATH" >&2; exit 1 ;;
  esac
  append_perf_summary
done

echo "Results: $RUN_DIR/step_similarity.csv"
