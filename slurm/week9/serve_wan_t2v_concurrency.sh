#!/bin/bash
#SBATCH --job-name=wan-t2v-ncu
#SBATCH --partition=gpu
#SBATCH --gpus=h100-96:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=wan-t2v-ncu-%j.out

set -euo pipefail

source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week9/serving-concurrency"
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
HEIGHT=480
WIDTH=832
NUM_FRAMES=33
FPS=16
NUM_INFERENCE_STEPS=50
NUM_PROMPTS=2 # Requests after the single benchmark warmup.
PORT=30000

setup_sglang_env
export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=0

RUN_DIR="$OUTPUT_DIR/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
ncu --version > "$RUN_DIR/ncu.log"
cp "$REPO_DIR/slurm/vbench_11_prompts.json" "$RUN_DIR/prompts.json"

# The serving benchmark expects one prompt per line.
python3 - "$RUN_DIR/prompts.json" > "$RUN_DIR/prompts.txt" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    for prompt in json.load(file)["prompts"]:
        print(prompt["prompt_en"])
PY

NCU_PID=""
SERVER_PID_FILE=""

# Stop the server and its workers, then let ncu finish writing its report.
# The server runs in its own process group; ncu stays outside that group.
cleanup() {
  if [[ -n "$SERVER_PID_FILE" && -s "$SERVER_PID_FILE" ]]; then
    SERVER_PID=$(cat "$SERVER_PID_FILE")
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
  elif [[ -n "$NCU_PID" ]]; then
    kill -TERM "$NCU_PID" 2>/dev/null || true
  fi
  if [[ -n "$NCU_PID" ]]; then
    wait "$NCU_PID" 2>/dev/null || true
  fi
  NCU_PID=""
  SERVER_PID_FILE=""
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Four independent captures. Profiling overhead makes their latency unsuitable
# for normal throughput comparisons.
for KERNEL in attention matmul; do
  if [[ "$KERNEL" == attention ]]; then
    KERNEL_FILTER="regex:FlashAttnFwd"
    LAUNCH_COUNT=1
  else
    KERNEL_FILTER="regex:nvjet_"
    LAUNCH_COUNT=20
  fi

  for CONCURRENCY in 1 2; do
    PROFILE_DIR="$RUN_DIR/${KERNEL}_c${CONCURRENCY}"
    mkdir -p "$PROFILE_DIR"
    SERVER_PID_FILE="$PROFILE_DIR/server.pid"
    echo "Profiling $KERNEL, concurrency=$CONCURRENCY"

    # Startup warmup emits no stage NVTX markers in this repo.
    # Range 1 is the benchmark warmup; range 2 is the first measured request/batch.
    # setsid gives only the server and its workers a separate cleanup group.
    ncu \
      --target-processes all \
      --nvtx \
      --nvtx-include "stage_denoising_stage/" \
      --range-filter "::2" \
      --kernel-name-base demangled \
      --rename-kernels off \
      --kernel-name "$KERNEL_FILTER" \
      --launch-count "$LAUNCH_COUNT" \
      --set full \
      --output "$PROFILE_DIR/profile" \
      setsid bash -c 'echo "$$" > "$1"; shift; exec "$@"' _ "$SERVER_PID_FILE" \
      sglang serve \
        --model-path "$MODEL_PATH" \
        --num-gpus 1 \
        --host 127.0.0.1 \
        --port "$PORT" \
        --batching-mode dynamic \
        --batching-max-size 3 \
        --batching-delay-ms 5 \
        --enable-batching-metrics \
        --enable-layerwise-nvtx-marker \
        > "$PROFILE_DIR/server.log" 2>&1 &
    NCU_PID=$!

    # Wait up to 20 minutes for loading; fail early if the profiler exits.
    python3 - "$PORT" "$NCU_PID" <<'PY'
import os
import sys
import time
import requests

port, pid = map(int, sys.argv[1:])
deadline = time.monotonic() + 1200
while time.monotonic() < deadline:
    os.kill(pid, 0)  # Raises if ncu has already exited.
    try:
        if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).ok:
            break
    except requests.RequestException:
        pass
    time.sleep(2)
else:
    raise TimeoutError("Server did not become ready; check server.log.")
PY

    python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
      --model "$MODEL_PATH" \
      --host 127.0.0.1 \
      --port "$PORT" \
      --task text-to-video \
      --dataset vbench \
      --dataset-path "$RUN_DIR/prompts.txt" \
      --num-prompts "$NUM_PROMPTS" \
      --max-concurrency "$CONCURRENCY" \
      --request-rate inf \
      --height "$HEIGHT" \
      --width "$WIDTH" \
      --num-frames "$NUM_FRAMES" \
      --fps "$FPS" \
      --num-inference-steps "$NUM_INFERENCE_STEPS" \
      --extra-body '{"seed": 42, "enable_cache_dit": false}' \
      --warmup-requests 1 \
      --warmup-inference-steps "$NUM_INFERENCE_STEPS" \
      --output-file "$PROFILE_DIR/benchmark.json" \
      2>&1 | tee "$PROFILE_DIR/benchmark.log"

    cleanup
    if [[ ! -s "$PROFILE_DIR/profile.ncu-rep" ]]; then
      echo "No Nsight Compute report. Check $PROFILE_DIR/server.log"
      exit 1
    fi
    grep -q "Warmup 1/1:.*success=True" "$PROFILE_DIR/benchmark.log"
    python3 - "$PROFILE_DIR/benchmark.json" "$NUM_PROMPTS" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    result = json.load(file)
if result["completed_requests"] != int(sys.argv[2]) or result["failed_requests"]:
    raise RuntimeError("Some requests failed; check benchmark.log and server.log.")
PY
    if [[ "$CONCURRENCY" == 2 ]] &&
       ! grep -q "Processed dynamic batch of 2/" "$PROFILE_DIR/server.log"; then
      echo "No batch of two was observed. Check $PROFILE_DIR/server.log"
      exit 1
    fi
  done
done

echo "Profiling completed. Nsight Compute reports and logs: $RUN_DIR"
