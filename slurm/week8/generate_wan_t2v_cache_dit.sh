#!/bin/bash
#SBATCH --job-name=wan-t2v
#SBATCH --partition=gpu
#SBATCH --gpus=h100-96
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=wan-t2v-%j.out

set -euo pipefail

source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week8/cache-dit"
RUN_PREFIX="wan_t2v_${SLURM_JOB_ID:-manual}"
SUMMARY_CSV="$OUTPUT_DIR/${RUN_PREFIX}_summary.csv"

MODEL_PATH="Wan-AI/Wan2.1-T2V-14B-Diffusers"
PROMPT="A red tram moves slowly through a sunlit city square"
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
NUM_INFERENCE_STEPS=50
SEED=42
REPEATS=1

# label num_gpus ulysses_degree ring_degree
RUN_CONFIGS=(
  "cache_dit 1 1 1"
)

# Fixed cache settings
export SGLANG_CACHE_DIT_FN=1
export SGLANG_CACHE_DIT_BN=1
export SGLANG_CACHE_DIT_WARMUP=4
export SGLANG_CACHE_DIT_MC=-1
export SGLANG_CACHE_DIT_RDT=0.24
export SGLANG_CACHE_DIT_TAYLORSEER=false
export SGLANG_CACHE_DIT_TS_ORDER=1
export SGLANG_CACHE_DIT_SCM_PRESET=none
export SGLANG_CACHE_DIT_SCM_POLICY=dynamic
unset SGLANG_CACHE_DIT_SCM_COMPUTE_BINS SGLANG_CACHE_DIT_SCM_CACHE_BINS

setup_sglang_env

for CONFIG in "${RUN_CONFIGS[@]}"; do
  read -r LABEL NUM_GPUS ULYSSES_DEGREE RING_DEGREE <<< "$CONFIG"

  for RUN_ID in $(seq 1 "$REPEATS"); do
    PERF_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_perf_run_${RUN_ID}.json"
    OUTPUT_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_run_${RUN_ID}.mp4"
    LOG_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_run_${RUN_ID}.log"

    echo "Starting ${LABEL} run ${RUN_ID}/${REPEATS}: num_gpus=${NUM_GPUS}, ulysses_degree=${ULYSSES_DEGREE}, ring_degree=${RING_DEGREE}"

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
      --seed "$SEED" \
      --enable-cache-dit \
      --save-output \
      --perf-dump-path "$PERF_PATH" \
      --output-file-path "$OUTPUT_PATH" 2>&1 | tee "$LOG_PATH"

    append_perf_summary
  done
done

echo "Experiment completed."
