#!/bin/bash
#SBATCH --job-name=wan-t2v-cache-dit-nsys
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=wan-t2v-cache-dit-nsys-%j.out

set -euo pipefail

source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week7/nsys/cache_dit_nsys"
RUN_PREFIX="wan_t2v_cache_dit_nsys_${SLURM_JOB_ID:-manual}"
SUMMARY_CSV="$OUTPUT_DIR/${RUN_PREFIX}_summary.csv"

MODEL_PATH="Wan-AI/Wan2.1-T2V-14B-Diffusers"
PROMPT="A red tram moves slowly through a sunlit city square"
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
NUM_INFERENCE_STEPS=50
SEED=42
REPEATS=2

# Maximum consecutive cached steps; -1 means no cap.
MAX_CONTINUOUS_CACHED_STEPS_VALUES=(3 -1)

# Fixed cache settings
export SGLANG_CACHE_DIT_FN=1
export SGLANG_CACHE_DIT_BN=0
export SGLANG_CACHE_DIT_WARMUP=4
export SGLANG_CACHE_DIT_RDT=0.24
export SGLANG_CACHE_DIT_TAYLORSEER=false
export SGLANG_CACHE_DIT_TS_ORDER=1
export SGLANG_CACHE_DIT_SCM_PRESET=none
export SGLANG_CACHE_DIT_SCM_POLICY=dynamic
unset SGLANG_CACHE_DIT_SCM_COMPUTE_BINS SGLANG_CACHE_DIT_SCM_CACHE_BINS

# label num_gpus ulysses_degree ring_degree
RUN_CONFIGS=(
  "cache_dit 1 1 1"
)

setup_sglang_env

for CONFIG in "${RUN_CONFIGS[@]}"; do
  read -r BASE_LABEL NUM_GPUS ULYSSES_DEGREE RING_DEGREE <<< "$CONFIG"

  for MAX_CONTINUOUS_CACHED_STEPS in "${MAX_CONTINUOUS_CACHED_STEPS_VALUES[@]}"; do
    export SGLANG_CACHE_DIT_MC="$MAX_CONTINUOUS_CACHED_STEPS"
    LABEL="${BASE_LABEL}_mc${MAX_CONTINUOUS_CACHED_STEPS}"

    for ((RUN_ID = 1; RUN_ID <= REPEATS; RUN_ID++)); do
      PERF_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_perf_run_${RUN_ID}.json"
      OUTPUT_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_run_${RUN_ID}.mp4"
      NSYS_OUTPUT_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_run_${RUN_ID}"

      echo "Starting ${LABEL} run ${RUN_ID}/${REPEATS}: max_continuous_cached_steps=${MAX_CONTINUOUS_CACHED_STEPS}, num_gpus=${NUM_GPUS}, ulysses_degree=${ULYSSES_DEGREE}, ring_degree=${RING_DEGREE}"

      # --no-save-output
      nsys profile \
        -t cuda,nvtx \
        --sample=process-tree \
        --backtrace=dwarf \
        --gpu-metrics-devices=cuda-visible \
        --gpu-metrics-frequency=1000 \
        --force-overwrite=true \
        --stats=false \
        -o "$NSYS_OUTPUT_PATH" \
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
        --enable-cache-dit \
        --save-output \
        --output-file-path "$OUTPUT_PATH" \
        --perf-dump-path "$PERF_PATH" \
        --enable-layerwise-nvtx-marker

      append_perf_summary
    done
  done
done

echo "Experiment completed."
