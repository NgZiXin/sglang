#!/bin/bash
#SBATCH --job-name=wan-step1-similarity-vbench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0:30:00
#SBATCH --output=wan-step1-similarity-vbench-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"

OUTPUT_DIR="$SCRATCH/sglang/outputs/week11/wan_step1_similarity_vbench/${SLURM_JOB_ID:-manual}"
PROMPT_JSON="$REPO_DIR/slurm/week10/vbench_33_prompts_wan_chai_kv.json"
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
SEED=42
HEIGHT=480
WIDTH=832
NUM_FRAMES=81
FLOW_SHIFT=5.0

setup_sglang_env
mkdir -p "$OUTPUT_DIR"

# Convert the JSON manifest to the existing runner's prompt-pair format.
PROMPT_PAIRS="$OUTPUT_DIR/prompt_pairs.txt"
python3 - "$PROMPT_JSON" "$PROMPT_PAIRS" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
pairs = []
seen_ids = set()
for entry in manifest["prompts"]:
    fields = [entry["prompt_id"], entry["prompt_a"], entry["prompt_en"]]
    if any(not isinstance(value, str) or not value.strip() or
           any(char in value for char in "|\r\n") for value in fields):
        raise ValueError(f"Invalid prompt pair: {fields!r}")
    fields = [value.strip() for value in fields]
    if fields[0] in seen_ids:
        raise ValueError(f"Duplicate prompt_id: {fields[0]}")
    seen_ids.add(fields[0])
    pairs.append("|".join(fields))
if not pairs:
    raise ValueError("No prompt pairs in manifest")
Path(sys.argv[2]).write_text("\n".join(pairs) + "\n", encoding="utf-8")
print(f"Loaded {len(pairs)} prompt pairs from {sys.argv[1]}", flush=True)
PY

# The Python script fixes num_inference_steps=1 and skips video decoding.
python3 -u "$REPO_DIR/slurm/week11/wan_step1_similarity.py" \
  --model-path "$MODEL_PATH" \
  --prompt-pairs "$PROMPT_PAIRS" \
  --output-csv "$OUTPUT_DIR/similarity.csv" \
  --seed "$SEED" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num-frames "$NUM_FRAMES" \
  --flow-shift "$FLOW_SHIFT" 2>&1 | tee "$OUTPUT_DIR/experiment.log"

echo "Results: $OUTPUT_DIR/similarity.csv"
