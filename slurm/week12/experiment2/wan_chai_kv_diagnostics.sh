#!/bin/bash
#SBATCH --job-name=wan-chai-kv-diagnostics
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=wan-chai-kv-diagnostics-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"
export OUTPUT_DIR="$SCRATCH/sglang/outputs/week12/experiment2/wan_chai_kv_diagnostics/${SLURM_JOB_ID:-manual}"
export WAN_KV_SAME_PROMPT_CONTROL="${WAN_KV_SAME_PROMPT_CONTROL:-0}"

setup_sglang_env

PROMPT_MANIFEST="$OUTPUT_DIR/prompts.json"
PROMPTS=$(python3 - "$REPO_DIR/slurm/week10/vbench_33_prompts_wan_chai_kv.json" \
  "$PROMPT_MANIFEST" "${WAN_KV_PAIR_LIMIT:-0}" <<'PY'
import json
import sys
from pathlib import Path

source, destination, limit = sys.argv[1:]
manifest = json.loads(Path(source).read_text(encoding="utf-8-sig"))
limit = int(limit)
if limit < 0:
    raise ValueError("WAN_KV_PAIR_LIMIT must be nonnegative")
if limit:
    manifest["prompts"] = manifest["prompts"][:limit]
Path(destination).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
for prompt in manifest["prompts"]:
    print(prompt["prompt_id"], prompt["prompt_a"], prompt["prompt_en"],
          manifest["seed"], manifest["sample_index"], sep="\t")
PY
)

mkdir -p "$OUTPUT_DIR/videos_reuse" "$OUTPUT_DIR/videos_baseline"
while IFS=$'\t' read -r PAIR_ID PROMPT_A PROMPT_B SEED SAMPLE_INDEX; do
  PAIR_OUTPUT_DIR="$OUTPUT_DIR/${PAIR_ID}/seed_$SEED"
  mkdir -p "$PAIR_OUTPUT_DIR"
  python3 -u "$REPO_DIR/slurm/week12/experiment2/wan_chai_kv_diagnostics.py" \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --prompt-a "$PROMPT_A" --prompt-b "$PROMPT_B" --seed "$SEED" \
    --steps-a 50 --steps-b 10 --flow-shift 8 --guidance-scale 6 \
    --height 480 --width 832 --num-frames 81 --fps 16 \
    --output-dir "$PAIR_OUTPUT_DIR" 2>&1 | tee "$PAIR_OUTPUT_DIR/experiment.log"

  VIDEO_NAME="${PROMPT_B}-${SAMPLE_INDEX}.mp4"
  cp "$PAIR_OUTPUT_DIR/b_chai_kv.mp4" "$OUTPUT_DIR/videos_reuse/$VIDEO_NAME"
  cp "$PAIR_OUTPUT_DIR/b_full.mp4" "$OUTPUT_DIR/videos_baseline/$VIDEO_NAME"
done <<< "$PROMPTS"

python3 - "$OUTPUT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "prompts.json").read_text(encoding="utf-8"))
with (root / "kv_diagnostics.csv").open("w", newline="", encoding="utf-8") as output:
    writer = None
    for prompt in manifest["prompts"]:
        pair_id = prompt["prompt_id"]
        path = root / pair_id / f"seed_{manifest['seed']}" / "kv_diagnostics.csv"
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if writer is None:
                writer = csv.DictWriter(output, fieldnames=["pair_id", *reader.fieldnames])
                writer.writeheader()
            for row in reader:
                writer.writerow({"pair_id": pair_id, **row})
PY

if [[ "${WAN_KV_RUN_VBENCH:-1}" == 1 ]]; then
  for variant in baseline reuse; do
    run_vbench_evaluation \
      --video-dir "$OUTPUT_DIR/videos_$variant" \
      --manifest "$PROMPT_MANIFEST" \
      --output-file "$OUTPUT_DIR/vbench_${variant}.json"
  done
fi

echo "Experiment completed. Outputs: $OUTPUT_DIR"
