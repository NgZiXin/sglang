#!/bin/bash
#SBATCH --job-name=wan-cache-vbench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=wan-cache-vbench-%j.out

set -euo pipefail

# Paths: change the environment path here if you installed elsewhere.
export SCRATCH="/mnt/scratch/n/$USER"
source "$SCRATCH/vbench/bin/activate"

VIDEO_DIR="$SCRATCH/sglang/outputs/week8/flow-shift"
RESULT_DIR="$SCRATCH/sglang/outputs/week8/vbench"

# Evaluate the folder, then combine individual video scores into one CSV.
python - "$VIDEO_DIR" "$RESULT_DIR" <<'PY'
import csv
import json
import subprocess
import sys
from pathlib import Path

video_dir, result_dir = map(Path, sys.argv[1:])
dimensions = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]

# One CSV row per MP4, with its filename and metric scores.
rows = {video.name: {"video": video.name} for video in sorted(video_dir.glob("*.mp4"))}
if not rows:
    raise ValueError(f"No MP4 videos found in {video_dir}")

# Each metric processes all videos together, loading its model once.
for dimension in dimensions:
    metric_dir = result_dir / dimension
    subprocess.run([
        "vbench", "evaluate", "--ngpus", "1",
        "--videos_path", str(video_dir), "--mode", "custom_input",
        "--dimension", dimension, "--output_path", str(metric_dir),
    ], check=True)

    # Require exactly one result file and a score for every input video.
    result_file, = metric_dir.glob("*_eval_results.json")
    results = json.loads(result_file.read_text())[dimension][1]
    scores = {Path(item["video_path"]).name: float(item["video_results"])
              for item in results}
    if scores.keys() != rows.keys():
        raise ValueError(f"{dimension}: results do not match the input videos")
    for name, score in scores.items():
        # VBench's per-video imaging scores need /100 to match its aggregate scale.
        rows[name][dimension] = score / 100 if dimension == "imaging_quality" else score

csv_path = result_dir / "video_scores.csv"
with csv_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["video"] + dimensions)
    writer.writeheader()
    writer.writerows(rows.values())
print(f"Saved {len(rows)} video scores to {csv_path}")
PY
