#!/bin/bash
#SBATCH --job-name=wan-t2v
#SBATCH --partition=gpu
#SBATCH --gpus=h100-96
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=wan-t2v-%j.out

set -e

cd ~/cp4101/week2/sglang

# Avoid cuDNN v8 frontend failures seen on some cluster CUDA/cuDNN stacks.
export TORCH_CUDNN_V8_API_DISABLED=1

export SCRATCH=/mnt/scratch/n/$USER
export HF_HOME=$SCRATCH/sglang/hf-cache
export TMPDIR=$SCRATCH/sglang/tmp
export XDG_CACHE_HOME=$SCRATCH/sglang/cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/sglang/torch-cache/inductor
export TRITON_CACHE_DIR=$SCRATCH/sglang/torch-cache/triton
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1

OUTPUT_DIR="$SCRATCH/sglang/outputs"
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

mkdir -p "$HF_HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$OUTPUT_DIR"

append_csv_row() {
  python3 - "$SUMMARY_CSV" "$PERF_PATH" "$RUN_ID" "$SEED" "$HEIGHT" "$WIDTH" "$NUM_FRAMES" "$FPS" "$NUM_INFERENCE_STEPS" "$OUTPUT_PATH" <<'PY'
import csv
import json
import os
import sys

summary_csv, perf_path, run_id, seed, height, width, num_frames, fps, num_steps, output_path = sys.argv[1:]

columns = [
    "run_id",
    "seed",
    "height",
    "width",
    "num_frames",
    "fps",
    "num_inference_steps",
    "total_duration_ms",
    "input_validation_ms",
    "text_encoding_ms",
    "latent_preparation_ms",
    "timestep_preparation_ms",
    "denoising_ms",
    "decoding_ms",
    "stage_durations_json",
    "perf_path",
    "output_path",
]

with open(perf_path, "r", encoding="utf-8") as f:
    perf = json.load(f)

stage_durations = {
    item["name"]: item.get("duration_ms", "")
    for item in perf.get("steps", [])
    if item.get("name")
}

row = {
    "run_id": run_id,
    "seed": seed,
    "height": height,
    "width": width,
    "num_frames": num_frames,
    "fps": fps,
    "num_inference_steps": num_steps,
    "total_duration_ms": perf.get("total_duration_ms", ""),
    "input_validation_ms": stage_durations.get("InputValidationStage", ""),
    "text_encoding_ms": stage_durations.get("TextEncodingStage", ""),
    "latent_preparation_ms": stage_durations.get("LatentPreparationStage", ""),
    "timestep_preparation_ms": stage_durations.get("TimestepPreparationStage", ""),
    "denoising_ms": stage_durations.get("DenoisingStage", ""),
    "decoding_ms": stage_durations.get("DecodingStage", ""),
    "stage_durations_json": json.dumps(stage_durations, sort_keys=True),
    "perf_path": perf_path,
    "output_path": output_path,
}

write_header = not os.path.exists(summary_csv) or os.path.getsize(summary_csv) == 0
with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

source "$SCRATCH/sglang/venv/bin/activate"

nvidia-smi

for RUN_ID in 1 2 3; do
  PERF_PATH="$OUTPUT_DIR/${RUN_PREFIX}_perf_run_${RUN_ID}.json"
  OUTPUT_PATH="$OUTPUT_DIR/${RUN_PREFIX}_run_${RUN_ID}.mp4"

  echo "Starting Wan T2V run ${RUN_ID}/3"

  sglang generate \
    --model-path "$MODEL_PATH" \
    --num-gpus 1 \
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
    --output-file-path "$OUTPUT_PATH"

  append_csv_row
done

echo "Summary CSV: $SUMMARY_CSV"
