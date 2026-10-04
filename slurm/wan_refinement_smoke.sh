#!/usr/bin/env bash
# Run from the repo root inside an active GPU allocation and SGLang environment.
# Usage: FASTWAN_MODEL_PATH=/path/to/FastWan bash slurm/wan_refinement_smoke.sh
set -euo pipefail

: "${FASTWAN_MODEL_PATH:?Set FASTWAN_MODEL_PATH to your FastWan 2.1 14B checkpoint}"
BASE_MODEL="${BASE_MODEL:-Wan-AI/Wan2.1-T2V-14B-Diffusers}"
OUT="outputs/wan_refinement_smoke"
mkdir -p "$OUT"

COMMON=(--num-gpus 1 --height 480 --width 832 --num-frames 81 --fps 16
        --prompt "A red tram moves slowly through a sunlit city square"
        --seed 42 --cfg-gate-step 1.0 --save-output)

echo "1/2: Generate a three-step FastWan draft"
sglang generate "${COMMON[@]}" \
    --model-path "$FASTWAN_MODEL_PATH" --model-id "$BASE_MODEL" \
    --pipeline WanDMDPipeline --num-inference-steps 3 \
    --dmd-denoising-steps 1000,757,522 --guidance-scale 1 \
    --enable-cache-dit false \
    --wan-save-latent-path "$OUT/draft.pt" \
    --output-file-path "$OUT/draft.mp4" 2>&1 | tee "$OUT/draft.log"
test -s "$OUT/draft.pt"

echo "2/2: Refine the draft for 50 steps with cache-dit"
# Default refinement schedule preserves 50 steps; cache-dit uses its defaults.
sglang generate "${COMMON[@]}" \
    --model-path "$BASE_MODEL" --num-inference-steps 50 --guidance-scale 5 \
    --wan-init-latent-path "$OUT/draft.pt" --wan-refine-sigma 0.4 \
    --enable-cache-dit true \
    --output-file-path "$OUT/refined.mp4" 2>&1 | tee "$OUT/refined.log"

echo "Done. Videos and logs: $OUT"
