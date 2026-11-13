#!/bin/bash
#SBATCH --job-name=wan-clip-prompt-sim
#SBATCH --partition=gpu
#SBATCH --gpus=h100-96
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=wan-clip-prompt-sim-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"
OUTPUT_DIR="$SCRATCH/sglang/outputs/week12/experiment3/prompt_similarity"
setup_sglang_env

RESULT_CSV="$OUTPUT_DIR/clip_prompt_similarity_${SLURM_JOB_ID}.csv"
python3 -u - "$RESULT_CSV" <<'PY'
import csv
import sys

import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, CLIPTextModelWithProjection

pairs = """\
23_city_pigeons|A flock of pigeons gathers on a cobblestone city square in daylight, wide shot.|A flock of pigeons gathers on a grassy hillside in daylight, wide shot.
06_snowy_cabin|Snow falls around a wooden cabin in a pine forest, fixed camera.|Snow falls around a camping tent in a pine forest, fixed camera.
04_add_basket|A cyclist rides along a quiet city street in daylight, side tracking shot.|A cyclist with a wicker basket on the bicycle rides along a quiet city street in daylight, side tracking shot.
07_refine_car_colour|A silver car drives along a coastal road beneath a clear sky, side tracking shot.|A red car drives along a coastal road beneath a clear sky, side tracking shot.
08_beach_party|A party on a sandy beach beside gentle waves, wide shot.|Gentle waves roll onto an empty sandy beach, wide shot.
10_window_cat|A cat sits on a windowsill watching rain fall outside, medium shot.|A cat jumps down from a windowsill while rain falls outside, medium shot.
"""
rows = [line.split("|", 2) for line in pairs.splitlines()]
prompts = [prompt for _, prompt_a, prompt_b in rows for prompt in (prompt_a, prompt_b)]

checkpoint = "openai/clip-vit-large-patch14"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
encoder = CLIPTextModelWithProjection.from_pretrained(checkpoint).eval().to("cuda")
inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to("cuda")

with torch.inference_mode():
    embeddings = encoder(**inputs).text_embeds.float()
    scores = cosine_similarity(embeddings[0::2], embeddings[1::2], dim=-1).tolist()

with open(sys.argv[1], "w", newline="", encoding="utf-8") as result:
    writer = csv.writer(result)
    writer.writerow(["pair_id", "prompt_a", "prompt_b", "clip_cosine_similarity"])
    for (pair_id, prompt_a, prompt_b), score in zip(rows, scores, strict=True):
        print(f"{pair_id}: {score:.6f}")
        writer.writerow([pair_id, prompt_a, prompt_b, f"{score:.8f}"])
PY

echo "Results: $RESULT_CSV"
