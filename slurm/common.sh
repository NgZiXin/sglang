#!/bin/bash

REPO_DIR="$HOME/cp4101/sglang"

export SCRATCH="/mnt/scratch/n/$USER"

setup_sglang_env() {
  # cd to the repo directory to ensure relative paths work correctly
  cd "$REPO_DIR"

  # Avoid cuDNN v8 frontend failures seen on some cluster CUDA/cuDNN stacks.
  export TORCH_CUDNN_V8_API_DISABLED=1

  export HF_HOME="$SCRATCH/sglang/hf-cache"
  export TMPDIR="$SCRATCH/sglang/tmp"
  export XDG_CACHE_HOME="$SCRATCH/sglang/cache"
  export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/sglang/torch-cache/inductor"
  export TRITON_CACHE_DIR="$SCRATCH/sglang/torch-cache/triton"
  export SGLANG_DIFFUSION_SYNC_STAGE_PROFILING=1

  mkdir -p "$HF_HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$OUTPUT_DIR"

  # activate virtual environment
  source "$SCRATCH/sglang/venv/bin/activate"
  
  nvidia-smi
}

append_perf_summary() {
  python3 "$REPO_DIR/slurm/perf_to_csv_util.py" \
    "$SUMMARY_CSV" \
    "$PERF_PATH" \
    "$LABEL" \
    "$NUM_GPUS" \
    "$ULYSSES_DEGREE" \
    "$RING_DEGREE" \
    "$RUN_ID" \
    "$SEED" \
    "$HEIGHT" \
    "$WIDTH" \
    "$NUM_FRAMES" \
    "$FPS" \
    "$NUM_INFERENCE_STEPS"
}
