"""Compare first-step guided Wan flow predictions for the prompt pairs below.

Lower mean_l1 / relative_l1 means closer outputs; higher cosine_similarity
means closer directions. These scores do not establish a safe reuse depth.
Uses Diffusers directly; no SGLang server or video decoding is needed.
"""

import argparse
import csv
import math
from pathlib import Path


# Edit or add (pair_id, prompt_a, prompt_b) entries here.
PROMPT_PAIRS = [
    (
        '01_add_vase',
        'Sunlight falls across a wooden table in a quiet dining room, fixed camera.',
        'Sunlight falls across a wooden table with a ceramic vase in a quiet dining room, fixed camera.',
    ),
    (
        '02_city_tram',
        'A tram moves slowly through a cobblestone city square in morning light, wide shot.',
        'A bus moves slowly through a cobblestone city square in morning light, wide shot.',
    ),
    (
        '03_beach_dog',
        'A golden retriever runs along a sandy beach beside gentle waves at sunset, side view.',
        'A golden retriever digs a hole in a sandy beach beside gentle waves at sunset, side view.',
    ),
    (
        '04_park_fountain',
        'Water splashes from a stone fountain in a quiet park lined with trees, fixed camera.',
        'Water splashes from a stone fountain in a quiet courtyard lined with arches, fixed camera.',
    ),
    (
        '05_park_cyclist',
        'A cyclist rides along a tree-lined park path in daylight, tracking shot.',
        'A cyclist rides along a canal towpath in daylight, tracking shot.',
    ),
    (
        '06_snowy_cabin',
        'Snow falls around a wooden cabin in a pine forest, fixed camera.',
        'Snow falls around a camping tent in a pine forest, fixed camera.',
    ),
    (
        '07_refine_car_colour',
        'A silver car drives along a coastal road beneath a clear sky, side tracking shot.',
        'A red car drives along a coastal road beneath a clear sky, side tracking shot.',
    ),
    (
        '08_remove_umbrella',
        'Gentle waves roll onto a sandy beach with a red beach umbrella at sunset, wide shot.',
        'Gentle waves roll onto a sandy beach at sunset, wide shot.',
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--flow-shift", type=float, default=5.0)
    # Same negative prompt as the repo's Wan sampling defaults.
    parser.add_argument("--negative-prompt", default=(
        "Bright tones, overexposed, static, blurred details, subtitles, style, "
        "works, paintings, images, static, overall gray, worst quality, low quality, "
        "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn "
        "hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused "
        "fingers, still picture, messy background, three legs, many people in the "
        "background, walking backwards"
    ))
    return parser.parse_args()


def first_prediction(pipe, prompt, args):
    import torch

    captured = []
    original_step = pipe.scheduler.step

    def capture_step(model_output, timestep, sample, **kwargs):
        # WanPipeline has already applied CFG here. Save the flow prediction
        # BEFORE the scheduler updates the latent, and compare in float32.
        captured.append((model_output.detach().float().cpu(), float(timestep)))
        return original_step(model_output, timestep, sample, **kwargs)

    pipe.scheduler.step = capture_step
    try:
        pipe(
            prompt=prompt, negative_prompt=args.negative_prompt,
            height=args.height, width=args.width, num_frames=args.num_frames,
            num_inference_steps=1,
            # Reset for EVERY prompt, so A and B get identical starting noise.
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            output_type="latent",  # Skip VAE decoding and video saving.
        )
    finally:
        pipe.scheduler.step = original_step
    if len(captured) != 1:
        raise RuntimeError(f"Expected one scheduler step, got {len(captured)}")
    return captured[0]


def main(args):
    import torch
    from diffusers import UniPCMultistepScheduler, WanPipeline

    pipe = WanPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift,
    )
    pipe.to("cuda")
    print(f"Model={args.model_path}, seed={args.seed}, steps=1, "
          f"size={args.width}x{args.height}, frames={args.num_frames}, "
          f"flow_shift={args.flow_shift}", flush=True)
    print(f"Negative prompt: {args.negative_prompt}", flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["pair_id", "seed", "timestep", "mean_l1", "relative_l1",
                         "cosine_similarity", "prompt_a", "prompt_b"])
        for pair_id, prompt_a, prompt_b in PROMPT_PAIRS:
            print(f"\n{pair_id}\nA: {prompt_a}\nB: {prompt_b}", flush=True)
            a, timestep_a = first_prediction(pipe, prompt_a, args)
            b, timestep_b = first_prediction(pipe, prompt_b, args)
            if timestep_a != timestep_b or a.shape != b.shape:
                raise RuntimeError("A and B must have the same timestep and output shape")

            mean_l1 = (a - b).abs().mean()
            scale = 0.5 * (a.abs().mean() + b.abs().mean())
            relative_l1 = mean_l1 / scale.clamp_min(1e-8)
            cosine = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
            scores = [mean_l1.item(), relative_l1.item(), cosine.item()]
            if not all(math.isfinite(score) for score in scores):
                raise RuntimeError(f"Non-finite scores for {pair_id}: {scores}")
            writer.writerow([pair_id, args.seed, timestep_a, *scores, prompt_a, prompt_b])
            file.flush()  # Keep completed pairs if the Slurm job is interrupted.
            print(f"t={timestep_a:g}, mean_l1={scores[0]:.6f}, "
                  f"relative_l1={scores[1]:.6f}, cosine={scores[2]:.6f}", flush=True)

    print(f"\nResults: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main(parse_args())
