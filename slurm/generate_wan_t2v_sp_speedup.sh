#!/bin/bash
#SBATCH --job-name=wan-t2v-sp-speed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=0:45:00
#SBATCH --output=wan-t2v-sp-speed-%j.out

set -euo pipefail

source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week12/slides/lossless/"
RUN_PREFIX="wan_t2v_sp_speedup_${SLURM_JOB_ID:-manual}"
SUMMARY_CSV="$OUTPUT_DIR/${RUN_PREFIX}_summary.csv"

MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
PROMPT="A golden retriever runs along a sandy beach beside gentle waves at sunset, side view."
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
NUM_INFERENCE_STEPS=50
SEED=42
REPEATS=4

# label num_gpus sp_degree ulysses_degree ring_degree cfg_parallel_size
RUN_CONFIGS=(
  "cfg_parallel_2gpu 2 1 1 1 2"
  "ulysses_2gpu 2 2 2 1 1"
  "sequential_2gpu 2 1 1 1 1"
)

setup_sglang_env

for CONFIG in "${RUN_CONFIGS[@]}"; do
  read -r LABEL NUM_GPUS SP_DEGREE ULYSSES_DEGREE RING_DEGREE CFG_PARALLEL_SIZE <<< "$CONFIG"

  for RUN_ID in $(seq 1 "$REPEATS"); do
    PERF_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_perf_run_${RUN_ID}.json"
    OUTPUT_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_run_${RUN_ID}.mp4"

    echo "Starting ${LABEL} run ${RUN_ID}/${REPEATS}: num_gpus=${NUM_GPUS}, sp_degree=${SP_DEGREE}, ulysses_degree=${ULYSSES_DEGREE}, ring_degree=${RING_DEGREE}, cfg_parallel_size=${CFG_PARALLEL_SIZE}"

    # --no-save-output
    sglang generate \
      --model-path "$MODEL_PATH" \
      --num-gpus "$NUM_GPUS" \
      --sp-degree "$SP_DEGREE" \
      --ulysses-degree "$ULYSSES_DEGREE" \
      --ring-degree "$RING_DEGREE" \
      --encoder-parallel replicate \
      --cfg-parallel-size "$CFG_PARALLEL_SIZE" \
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

echo "Experiment completed."
