import argparse
import csv
import json
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Wan request caching experiment")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-a", required=True)
    parser.add_argument("--prompt-b", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--clip-model", required=True)
    parser.add_argument("--similarity-to-k", type=json.loads, required=True)
    parser.add_argument("--k-override", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def choose_k(similarity, similarity_to_k):
    return next((k for threshold, k in similarity_to_k if similarity > threshold), 0)


def prompt_similarity(prompt_a, prompt_b, clip_model):
    import torch
    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    tokenizer = AutoTokenizer.from_pretrained(clip_model)
    model = CLIPTextModelWithProjection.from_pretrained(clip_model).eval()
    inputs = tokenizer(
        [prompt_a, prompt_b], padding=True, truncation=True, return_tensors="pt"
    )
    with torch.inference_mode():
        embeddings = torch.nn.functional.normalize(model(**inputs).text_embeds, dim=-1)
    return (embeddings[0] @ embeddings[1]).item()


def make_checkpoint(result, server_args, k, num_steps):
    """Keep only the selected post-step latent in CPU memory."""
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
    scheduler.set_timesteps(num_steps, device="cpu")
    torch.testing.assert_close(
        result.trajectory_timesteps, scheduler.timesteps, rtol=0, atol=0
    )
    height, width, num_frames = result.size
    return {
        "format": INTERMEDIATE_FORMAT,
        # Entry k-1 is AFTER k updates. Clone to release the full trajectory.
        "latents": result.trajectory_latents[:, k - 1].cpu().clone(),
        "completed_steps": k,
        "timesteps": scheduler.timesteps,
        "sigmas": scheduler.sigmas,
        "model_path": str(server_args.model_path),
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "prompt": result.prompt,
    }


def main(args):
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    output_dir = args.output_dir
    similarity = prompt_similarity(args.prompt_a, args.prompt_b, args.clip_model)
    k = (
        choose_k(similarity, args.similarity_to_k)
        if args.k_override is None
        else args.k_override
    )
    assert k in [0] + [step for _, step in args.similarity_to_k]
    print(
        f"Similarity={similarity:.6f}; skip k={k}; B runs {args.num_inference_steps - k} steps",
        flush=True,
    )

    generator = DiffGenerator.from_pretrained(
        model_path=args.model_path,
        num_gpus=1,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        encoder_parallel="replicate",
    )
    settings = dict(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
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
        assert actual_steps == args.num_inference_steps - skipped, (label, actual_steps)
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
        source = run("a_full", args.prompt_a, return_trajectory_latents=k > 0)
        checkpoint = None
        if k > 0:
            checkpoint = make_checkpoint(
                source, generator.server_args, k, args.num_inference_steps
            )
        del source

        # 2. Resume B at k, then generate the full B baseline for comparison.
        run(
            "b_cached",
            args.prompt_b,
            skipped=k,
            wan_resume_latent=checkpoint,
        )
        del checkpoint
        run("b_full", args.prompt_b)
    finally:
        generator.shutdown()

    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "experiment.json").write_text(
        json.dumps(
            dict(
                model_path=args.model_path,
                prompt_a=args.prompt_a,
                prompt_b=args.prompt_b,
                settings=settings,
                clip_model=args.clip_model,
                similarity=similarity,
                similarity_to_k=args.similarity_to_k,
                selected_k=k,
                manual_k=args.k_override,
            ),
            indent=2,
        )
    )
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main(parse_args())
