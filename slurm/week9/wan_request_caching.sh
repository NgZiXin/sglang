#!/bin/bash
#SBATCH --job-name=wan-request-caching
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=00:45:00
#SBATCH --output=wan-request-caching-%j.out

set -euo pipefail
source "$HOME/cp4101/sglang/slurm/common.sh"
export OUTPUT_DIR="$SCRATCH/sglang/outputs/week9/wan_request_caching/${SLURM_JOB_ID:-manual}"
setup_sglang_env

# Spawned SGLang workers need a real Python file; keep its source in this script.
cat > "$OUTPUT_DIR/experiment.py" <<'PY'
import csv
import json
import os
import time
from pathlib import Path

# Experiment settings: edit these here.
MODEL_PATH = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
PROMPT_A = "A city square"
PROMPT_B = "A tram moves through a city square"
SEED = 42
NUM_STEPS = 50
K_OVERRIDE = None  # None uses similarity; set e.g. 15 for a fixed-k experiment.
CLIP_MODEL = "openai/clip-vit-large-patch14"
# Paper Figure 10; initial thresholds, not calibrated for Wan video generation.
SIMILARITY_TO_K = [(0.95, 25), (0.90, 20), (0.85, 15), (0.75, 10), (0.65, 5)]


def choose_k(similarity):
    return next((k for threshold, k in SIMILARITY_TO_K if similarity > threshold), 0)


def prompt_similarity(prompt_a, prompt_b):
    import torch
    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL)
    model = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL).eval()
    inputs = tokenizer(
        [prompt_a, prompt_b], padding=True, truncation=True, return_tensors="pt"
    )
    with torch.inference_mode():
        embeddings = torch.nn.functional.normalize(model(**inputs).text_embeds, dim=-1)
    return (embeddings[0] @ embeddings[1]).item()


def save_checkpoint(result, path, server_args, k):
    """Save only the selected post-step latent from the returned trajectory."""
    import torch
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
        FlowUniPCMultistepScheduler,
    )
    from sglang.multimodal_gen.runtime.utils.wan_refinement import INTERMEDIATE_FORMAT

    # Rebuild the same schedule to retain exact sigmas; returned timesteps alone
    # are rounded and must not be divided by 1000 to recover noise levels.
    config = server_args.pipeline_config
    scheduler = FlowUniPCMultistepScheduler(
        shift=config.flow_shift, timestep_schedule=config.wan_timestep_schedule
    )
    scheduler.set_timesteps(NUM_STEPS, device="cpu")
    torch.testing.assert_close(
        result.trajectory_timesteps, scheduler.timesteps, rtol=0, atol=0
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, num_frames = result.size
    torch.save(
        {
            "format": INTERMEDIATE_FORMAT,
            # Entry k-1 is AFTER k updates. Clone so torch.save doesn't serialize
            # the entire trajectory storage backing this tensor view.
            "latents": result.trajectory_latents[:, k - 1].clone(),
            "completed_steps": k,
            "timesteps": scheduler.timesteps,
            "sigmas": scheduler.sigmas,
            "model_path": str(server_args.model_path),
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "prompt": result.prompt,
        },
        path,
    )


def main():
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    output_dir = Path(os.environ["OUTPUT_DIR"])
    similarity = prompt_similarity(PROMPT_A, PROMPT_B)
    k = choose_k(similarity) if K_OVERRIDE is None else K_OVERRIDE
    assert k in [0] + [step for _, step in SIMILARITY_TO_K]
    print(
        f"Similarity={similarity:.6f}; skip k={k}; B runs {NUM_STEPS - k} steps",
        flush=True,
    )

    generator = DiffGenerator.from_pretrained(
        model_path=MODEL_PATH,
        num_gpus=1,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        encoder_parallel="replicate",
    )
    settings = dict(
        height=480,
        width=832,
        num_frames=81,
        fps=16,
        seed=SEED,
        num_inference_steps=NUM_STEPS,
        guidance_scale=5.0,
        progressive_mode="fullres",
        enable_cache_dit=False,
        enable_teacache=False,
        cfg_gate_step=1.0,
        save_output=True,
        output_path=str(output_dir),
    )
    rows = []

    def run(label, prompt, skipped=0, **kwargs):
        started = time.perf_counter()
        result = generator.generate(
            dict(
                settings,
                prompt=prompt,
                output_file_name=f"{label}.mp4",
                perf_dump_path=str(output_dir / f"{label}_perf.json"),
                **kwargs,
            )
        )
        elapsed = time.perf_counter() - started
        if result is None:
            raise RuntimeError(f"Generation failed: {label}")
        perf = json.loads((output_dir / f"{label}_perf.json").read_text())
        stages = {entry["name"]: entry["duration_ms"] for entry in perf["steps"]}
        actual_steps = len(perf["denoise_steps_ms"])
        assert actual_steps == NUM_STEPS - skipped, (label, actual_steps)
        rows.append(
            dict(
                label=label,
                similarity=similarity,
                k=skipped,
                actual_steps=actual_steps,
                wall_seconds=elapsed,
                denoising_ms=stages["DenoisingStage"],
                decoding_ms=stages["DecodingStage"],
            )
        )
        return result

    try:
        # 1. Complete A; collect its trajectory only when B will reuse a latent.
        source = run("a_full", PROMPT_A, return_trajectory_latents=k > 0)
        checkpoint = None
        if k > 0:
            checkpoint = output_dir / "checkpoints" / f"step_{k}.pt"
            save_checkpoint(source, checkpoint, generator.server_args, k)
        del source

        # 2. Resume B at k, then generate the full B baseline for comparison.
        run(
            "b_cached",
            PROMPT_B,
            skipped=k,
            wan_resume_latent_path=str(checkpoint) if checkpoint else None,
        )
        run("b_full", PROMPT_B)
    finally:
        generator.shutdown()

    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "experiment.json").write_text(
        json.dumps(
            dict(
                model_path=MODEL_PATH,
                prompt_a=PROMPT_A,
                prompt_b=PROMPT_B,
                settings=settings,
                clip_model=CLIP_MODEL,
                similarity=similarity,
                similarity_to_k=SIMILARITY_TO_K,
                selected_k=k,
                manual_k=K_OVERRIDE,
            ),
            indent=2,
        )
    )
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
PY

python3 -u "$OUTPUT_DIR/experiment.py" 2>&1 | tee "$OUTPUT_DIR/experiment.log"
