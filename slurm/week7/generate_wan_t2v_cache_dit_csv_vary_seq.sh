#!/bin/bash
#SBATCH --job-name=wan-t2v-cache-dit-csv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --output=wan-t2v-cache-dit-csv-%j.out

set -euo pipefail

source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week7/cache_dit_csv_2"
RUN_PREFIX="wan_t2v_cache_dit_csv_${SLURM_JOB_ID:-manual}"
SUMMARY_CSV="$OUTPUT_DIR/${RUN_PREFIX}_summary.csv"
export CACHE_DIT_DECISION_CSV="$OUTPUT_DIR/${RUN_PREFIX}_cache_decisions.csv"

MODEL_PATH="Wan-AI/Wan2.1-T2V-14B-Diffusers"
HEIGHT=480
WIDTH=832
NUM_FRAMES_LIST=(81 97 113 129 145)
FPS=16
NUM_INFERENCE_STEPS=50
SEED=42
REPEATS=2

# prompt_id|prompt
PROMPT_CONFIGS=(
  "p01_tram|A red tram moves slowly through a sunlit city square"
  "p02_static_landscape|A locked-off camera shows a quiet mountain lake reflecting snow-covered peaks with only gentle ripples on the water"
  "p03_fast_motion|A cheetah sprints across a grassy savannah while the camera tracks alongside it and dust rises behind its paws"
  "p04_camera_orbit|The camera makes a smooth orbit around a stationary stone statue in a courtyard revealing the buildings and trees behind it"
  "p05_human_action|A dancer performs a full-body spin and then leaps across a wooden stage while a fixed camera keeps the dancer in view"
  "p06_multiple_objects|Three brightly colored toy cars cross a tabletop intersection in different directions passing in front of one another"
  "p07_fine_detail|A close-up view of a butterfly slowly opening and closing its intricately patterned wings on a flower swaying in a light breeze"
  "p08_fluid_motion|A wave crashes against dark coastal rocks sending water droplets and white foam into the air before the water recedes"
)

# Fixed cache settings
export SGLANG_CACHE_DIT_MC=3
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

  for NUM_FRAMES in "${NUM_FRAMES_LIST[@]}"; do
    for PROMPT_CONFIG in "${PROMPT_CONFIGS[@]}"; do
      IFS='|' read -r PROMPT_ID PROMPT <<< "$PROMPT_CONFIG"

      for ((RUN_ID = 1; RUN_ID <= REPEATS; RUN_ID++)); do
        LABEL="${BASE_LABEL}_${PROMPT_ID}_frames${NUM_FRAMES}_seed${SEED}_mc${SGLANG_CACHE_DIT_MC}"
        PERF_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_perf_run_${RUN_ID}.json"
        OUTPUT_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_run_${RUN_ID}.mp4"
        export CACHE_DIT_RUN_LABEL="$LABEL"
        export CACHE_DIT_RUN_ID="$RUN_ID"

        echo "Starting ${LABEL} run ${RUN_ID}/${REPEATS}: num_frames=${NUM_FRAMES}, num_gpus=${NUM_GPUS}, ulysses_degree=${ULYSSES_DEGREE}, ring_degree=${RING_DEGREE}"

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
          --perf-dump-path "$PERF_PATH"

        append_perf_summary
      done
    done
  done
done

echo "Experiment completed."
