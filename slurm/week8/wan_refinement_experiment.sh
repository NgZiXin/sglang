#!/bin/bash
# Run inside an existing GPU allocation with the project's environment active.
# Does not submit jobs or select a cluster partition.
set -euo pipefail

: "${FASTWAN_MODEL_PATH:?Set FASTWAN_MODEL_PATH to your local FastWan 2.1 14B checkpoint}"
BASE_MODEL="${BASE_MODEL:-Wan-AI/Wan2.1-T2V-14B-Diffusers}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/week8/wan_refinement}"
PROMPT="${PROMPT:-A red tram moves slowly through a sunlit city square}"
SEED="${SEED:-42}"
SIGMA="${SIGMA:-0.4}"
SCHEDULE="${SCHEDULE:-rescale}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
STEPS="${STEPS:-50}"
mkdir -p "$OUTPUT_DIR"

# One GPU initially isolates the initialization experiment from parallelism.
COMMON=(--num-gpus 1 --sp-degree 1 --ulysses-degree 1 --ring-degree 1
        --cfg-parallel-size 1 --prompt "$PROMPT" --height "$HEIGHT"
        --width "$WIDTH" --num-frames "$NUM_FRAMES" --fps 16 --seed "$SEED"
        --cfg-gate-step 1.0 --save-output)

# Explicit overrides keep threshold/cache policy identical in all cache-on runs.
CACHE_PARAMS='{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"residual_diff_threshold":0.24,"max_continuous_cached_steps":3,"enable_taylorseer":false,"scm_preset":"none","scm_policy":"dynamic"}'

run_case() {
    local label="$1"
    shift
    # These hooks already appear in this checkout's week7 scripts. They require
    # the user's instrumented cache-dit; upstream builds may not write the CSV.
    CACHE_DIT_DECISION_CSV="$OUTPUT_DIR/${label}_cache.csv" \
    CACHE_DIT_RUN_LABEL="$label" CACHE_DIT_RUN_ID=1 \
      sglang generate "${COMMON[@]}" \
        --output-file-path "$OUTPUT_DIR/${label}.mp4" \
        --perf-dump-path "$OUTPUT_DIR/${label}_perf.json" \
        "$@" 2>&1 | tee "$OUTPUT_DIR/${label}.log"
}

# Match the existing FastWan launch recipe in this repository. guidance=1
# makes the distilled draft's CFG behavior explicit.
run_case draft --model-path "$FASTWAN_MODEL_PATH" --model-id "$BASE_MODEL" \
    --pipeline WanDMDPipeline --num-inference-steps 3 \
    --dmd-denoising-steps 1000,757,522 --guidance-scale 1 \
    --enable-cache-dit false --wan-save-latent-path "$OUTPUT_DIR/draft.pt"

# Base-generated clean video is a control for the effect of the draft source.
run_case baseline_full --model-path "$BASE_MODEL" --num-inference-steps "$STEPS" \
    --guidance-scale 5 --enable-cache-dit false \
    --wan-save-latent-path "$OUTPUT_DIR/base.pt"
run_case baseline_cache --model-path "$BASE_MODEL" --num-inference-steps "$STEPS" \
    --guidance-scale 5 --enable-cache-dit true --cache-dit-params "$CACHE_PARAMS"

REFINE=(--model-path "$BASE_MODEL" --num-inference-steps "$STEPS"
        --guidance-scale 5 --wan-refine-sigma "$SIGMA"
        --wan-refine-schedule "$SCHEDULE")
run_case draft_refine_full "${REFINE[@]}" \
    --wan-init-latent-path "$OUTPUT_DIR/draft.pt" --enable-cache-dit false
run_case draft_refine_cache "${REFINE[@]}" \
    --wan-init-latent-path "$OUTPUT_DIR/draft.pt" --enable-cache-dit true \
    --cache-dit-params "$CACHE_PARAMS"
run_case base_refine_cache "${REFINE[@]}" \
    --wan-init-latent-path "$OUTPUT_DIR/base.pt" --enable-cache-dit true \
    --cache-dit-params "$CACHE_PARAMS"
