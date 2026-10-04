#!/bin/bash
#SBATCH --job-name=wan-refinement-experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --output=wan-refinement-experiment-%j.out

set -euo pipefail

source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week8/wan_refinement_experiment"
RUN_PREFIX="wan_refinement_experiment_${SLURM_JOB_ID:-manual}"
SUMMARY_CSV="$OUTPUT_DIR/${RUN_PREFIX}_summary.csv"

FASTWAN_MODEL_PATH="$SCRATCH/models/FastWan2.1-T2V-14B-Diffusers"
MODEL_ID="Wan-AI/Wan2.1-T2V-14B-Diffusers"
PROMPT="A red tram moves slowly through a sunlit city square"
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FPS=16
SEED=42
NUM_GPUS=1
ULYSSES_DEGREE=1
RING_DEGREE=1
DMD_DENOISING_STEPS="1000,757,522"
REFINE_STEPS=50
REFINE_SIGMA=0.4
DRAFT_PATH="$OUTPUT_DIR/${RUN_PREFIX}_draft.pt"
BASE_PATH="$OUTPUT_DIR/${RUN_PREFIX}_base.pt"
REFINE_SCHEDULE="rescale"

# Fixed cache settings, matching the week7 experiments.
export SGLANG_CACHE_DIT_FN=1
export SGLANG_CACHE_DIT_BN=0
export SGLANG_CACHE_DIT_WARMUP=4
export SGLANG_CACHE_DIT_RDT=0.24
export SGLANG_CACHE_DIT_MC=3
export SGLANG_CACHE_DIT_TAYLORSEER=false
export SGLANG_CACHE_DIT_SCM_PRESET=none
export SGLANG_CACHE_DIT_SCM_POLICY=dynamic
unset SGLANG_CACHE_DIT_SCM_COMPUTE_BINS SGLANG_CACHE_DIT_SCM_CACHE_BINS

RUN_CONFIGS=(
  draft
  baseline_full
  baseline_cache
  draft_refine_full
  draft_refine_cache
  base_refine_cache
)

setup_sglang_env

RUN_ID=0

for LABEL in "${RUN_CONFIGS[@]}"; do
  RUN_ID=$((RUN_ID + 1))
  MODEL_PATH="$MODEL_ID"
  NUM_INFERENCE_STEPS="$REFINE_STEPS"
  GUIDANCE_SCALE=5
  ENABLE_CACHE=false
  EXTRA_ARGS=()

  case "$LABEL" in
    draft)
      MODEL_PATH="$FASTWAN_MODEL_PATH"
      NUM_INFERENCE_STEPS=3
      GUIDANCE_SCALE=1
      EXTRA_ARGS=(
        --model-id "$MODEL_ID"
        --pipeline WanDMDPipeline
        --dmd-denoising-steps "$DMD_DENOISING_STEPS"
        --wan-save-latent-path "$DRAFT_PATH"
      )
      ;;
    baseline_full)
      EXTRA_ARGS=(--wan-save-latent-path "$BASE_PATH")
      ;;
    baseline_cache)
      ENABLE_CACHE=true
      ;;
    draft_refine_full|draft_refine_cache|base_refine_cache)
      INIT_PATH="$DRAFT_PATH"
      if [[ "$LABEL" == "base_refine_cache" ]]; then
        INIT_PATH="$BASE_PATH"
      fi
      if [[ "$LABEL" != "draft_refine_full" ]]; then
        ENABLE_CACHE=true
      fi
      test -s "$INIT_PATH"
      EXTRA_ARGS=(
        --wan-init-latent-path "$INIT_PATH"
        --wan-refine-sigma "$REFINE_SIGMA"
        --wan-refine-schedule "$REFINE_SCHEDULE"
      )
      ;;
  esac

  PERF_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}_perf.json"
  OUTPUT_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}.mp4"
  LOG_PATH="$OUTPUT_DIR/${RUN_PREFIX}_${LABEL}.log"

  echo "Starting ${LABEL}"

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
    --cfg-gate-step 1.0 \
    --guidance-scale "$GUIDANCE_SCALE" \
    --enable-cache-dit "$ENABLE_CACHE" \
    "${EXTRA_ARGS[@]}" \
    --save-output \
    --output-file-path "$OUTPUT_PATH" \
    --perf-dump-path "$PERF_PATH" 2>&1 | tee "$LOG_PATH"

  append_perf_summary
done

echo "Experiment completed. Outputs: $OUTPUT_DIR"
