#!/bin/bash
#SBATCH --job-name=wan-t2v-sp-speed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=8:00:00
#SBATCH --output=wan-t2v-sp-speed-%j.out

set -euo pipefail

source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week4/experiement3"
RUN_PREFIX="wan_t2v_sp_speedup_${SLURM_JOB_ID:-manual}"
SUMMARY_CSV="$OUTPUT_DIR/${RUN_PREFIX}_summary.csv"

MODEL_PATH="Wan-AI/Wan2.1-T2V-14B-Diffusers"
PROMPT="A red tram moves slowly through a sunlit city square"
HEIGHT=480
WIDTH=832
NUM_FRAMES_LIST=(81 97 113 129 145)
FPS=16
NUM_INFERENCE_STEPS=50
SEED=42
REPEATS=4

# label num_gpus ulysses_degree ring_degree
RUN_CONFIGS=(
  "baseline 1 1 1"
)

setup_sglang_env

for NUM_FRAMES in "${NUM_FRAMES_LIST[@]}"; do
  for CONFIG in "${RUN_CONFIGS[@]}"; do
    read -r LABEL NUM_GPUS ULYSSES_DEGREE RING_DEGREE <<< "$CONFIG"

    for RUN_ID in $(seq 1 "$REPEATS"); do
      PERF_PATH="$OUTPUT_DIR/${RUN_PREFIX}_frames_${NUM_FRAMES}_${LABEL}_perf_run_${RUN_ID}.json"
      OUTPUT_PATH="$OUTPUT_DIR/${RUN_PREFIX}_frames_${NUM_FRAMES}_${LABEL}_run_${RUN_ID}.mp4"

      echo "Starting ${LABEL} run ${RUN_ID}/${REPEATS}: num_frames=${NUM_FRAMES}, num_gpus=${NUM_GPUS}, ulysses_degree=${ULYSSES_DEGREE}, ring_degree=${RING_DEGREE}"

      # --no-save-output
      sglang generate \
        --model-path "$MODEL_PATH" \
        --num-gpus "$NUM_GPUS" \
        --sp-degree "$NUM_GPUS" \
        --ulysses-degree "$ULYSSES_DEGREE" \
        --ring-degree "$RING_DEGREE" \
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
  done
done

echo "Experiment completed."
