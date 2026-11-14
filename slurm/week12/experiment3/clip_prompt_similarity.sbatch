#!/bin/bash
#SBATCH --job-name=wan-clip-prompt-sim
#SBATCH --partition=gpu
#SBATCH --gpus=h100-47
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=wan-clip-prompt-sim-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"
OUTPUT_DIR="$SCRATCH/sglang/outputs/week12/experiment3/prompt_similarity"
setup_sglang_env

RESULT_CSV="$OUTPUT_DIR/clip_prompt_similarity_${SLURM_JOB_ID}.csv"
PROMPT_MANIFEST="$REPO_DIR/slurm/week10/vbench_33_prompts_wan_chai_kv.json"
python3 -u - "$RESULT_CSV" "$PROMPT_MANIFEST" <<'PY'
import csv
import json
import sys

import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, CLIPTextModelWithProjection

with open(sys.argv[2], encoding="utf-8") as manifest_file:
    manifest = json.load(manifest_file)

rows = [(pair["prompt_id"], pair["prompt_a"], pair["prompt_en"]) for pair in manifest["prompts"]]
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
