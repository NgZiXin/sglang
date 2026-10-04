#!/bin/bash
#SBATCH --job-name=wan-t2v-outputs
#SBATCH --partition=gpu
#SBATCH --gpus=h100-96
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=wan-t2v-outputs-%j.out

set -euo pipefail

source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week8/num-outputs"
RUN_PREFIX="wan_t2v_${SLURM_JOB_ID:-manual}"
SUMMARY_CSV="$OUTPUT_DIR/${RUN_PREFIX}_summary.csv"

MODEL_PATH="Wan-AI/Wan2.1-T2V-14B-Diffusers"
PROMPT="A red tram moves slowly through a sunlit city square"
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
NUM_INFERENCE_STEPS=50
SEED=42 # + 1 for each output
REPEATS=4
NUM_GPUS=1
ULYSSES_DEGREE=1
RING_DEGREE=1

setup_sglang_env

# Multi-output generation; this does not imply batched DiT denoising.
for NUM_OUTPUTS in 1 2 3; do
  LABEL="outputs_${NUM_OUTPUTS}"

  for RUN_ID in $(seq 1 "$REPEATS"); do
    PERF_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_perf_run_${RUN_ID}.json"
    OUTPUT_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_run_${RUN_ID}.mp4"

    echo "Starting ${LABEL} run ${RUN_ID}/${REPEATS}"

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
      --num-outputs-per-prompt "$NUM_OUTPUTS" \
      --seed "$SEED" \
      --enable-cache-dit false \
      --save-output \
      --perf-dump-path "$PERF_PATH" \
      --output-file-path "$OUTPUT_PATH"

    append_perf_summary
  done
done

echo "Experiment completed."
