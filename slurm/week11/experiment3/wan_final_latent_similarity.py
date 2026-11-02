"""Compare final Wan latents from full 50-step and mixed 22-step generation.

Each manifest prompt uses the same seed and generation settings for both runs.
VAE decoding is skipped. Metrics use float32 CPU arrays before VAE scaling:
  mean_l1 = mean(abs(full - mixed))
  relative_l1 = mean_l1 / max((mean(abs(full)) + mean(abs(mixed))) / 2, 1e-8)
  cosine_distance = 1 - cosine_similarity
Lower L1 / cosine distance means closer latents. Relative L1 is not pixel L1
normalized by 255, and these metrics do not establish perceptual similarity.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


# Exact schedule from experiment2/generate_wan_t2v_mix_step.py, including zero.
MIXED_SIGMAS = [
    1.00000000, 0.99745547, 0.99481865, 0.99208443, 0.98924731,
    0.98630137, 0.98324022, 0.98005698, 0.97674419, 0.97329377,
    0.96969697, 0.96594427, 0.96202532, 0.95792880, 0.95364238,
    0.94915254, 0.92307692, 0.88888889, 0.84210526, 0.77419355,
    0.66666667, 0.47058824, 0.00000000,
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    args = parser.parse_args()
    try:
        with args.prompt_manifest.open(encoding="utf-8-sig") as file:
            manifest = json.load(file)
        args.seed = int(manifest["seed"])
        args.prompts = manifest["prompts"]
        if not args.prompts:
            raise ValueError("Manifest has no prompts")
        seen = set()
        for prompt in args.prompts:
            if not prompt["prompt_id"] or not prompt["prompt_en"].strip():
                raise ValueError("Each prompt needs a nonempty prompt_id and prompt_en")
            if prompt["prompt_id"] in seen:
                raise ValueError(f"Duplicate prompt_id: {prompt['prompt_id']}")
            seen.add(prompt["prompt_id"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.error(f"Invalid prompt manifest: {error}")
    return args


def final_latent(worker, prompt, args, *, mixed):
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

    steps = len(MIXED_SIGMAS) - 1 if mixed else 50
    # Fresh requests recreate the same starting noise from the manifest seed.
    # Leave the negative prompt at Wan's default, as in both generation scripts.
    params = SamplingParams.from_user_sampling_params_args(
        args.model_path, server_args=worker.server_args,
        prompt=prompt, seed=args.seed, height=args.height, width=args.width,
        num_frames=args.num_frames, fps=16, num_inference_steps=steps,
        guidance_scale=args.guidance_scale, progressive_mode="fullres",
        save_output=False, enable_cache_dit=False, enable_teacache=False,
        enable_spectrum=False,
    )
    request = prepare_request(worker.server_args, params)
    stage = worker.pipeline.get_stage("TimestepPreparationStage")
    original_forward = stage.forward

    def prepare_mixed_timesteps(batch, server_args):
        # Set NumPy sigmas inside forward, as in experiment2: the stage's input
        # validator only accepts None/list, while UniPC needs a NumPy array.
        sigmas = np.asarray(MIXED_SIGMAS[:-1], dtype=np.float64)
        shift = stage.scheduler.config.shift
        batch.sigmas = sigmas / (shift - (shift - 1) * sigmas)
        # UniPC applies the shift and appends zero; start below one as in experiment2.
        batch.sigmas[0] = stage.scheduler.sigma_max
        return original_forward(batch, server_args)

    try:
        if mixed:
            # Override this instance for this request only, leaving full runs native.
            stage.forward = prepare_mixed_timesteps
        result = worker.pipeline.forward(request, worker.server_args)
    finally:
        stage.forward = original_forward
    if len(result.timesteps) != steps or result.latents is None:
        raise RuntimeError(f"Expected final latents after {steps} denoising steps")
    return result.latents.detach().float().cpu().numpy()


def latent_metrics(a, b):
    """Same symmetric relative-L1 and flattened cosine formulas as step1."""
    if a.shape != b.shape or a.size == 0:
        raise ValueError("Latents must have matching, nonempty shapes")
    a, b = a.astype(np.float32, copy=False), b.astype(np.float32, copy=False)
    mean_l1 = float(np.abs(a - b).mean())
    scale = 0.5 * (float(np.abs(a).mean()) + float(np.abs(b).mean()))
    relative_l1 = mean_l1 / max(scale, 1e-8)
    a, b = a.flatten(), b.flatten()
    cosine = float(np.dot(
        a / max(float(np.linalg.norm(a)), 1e-8),
        b / max(float(np.linalg.norm(b)), 1e-8),
    ))
    if not all(math.isfinite(value) for value in (mean_l1, relative_l1, cosine)):
        raise ValueError("Non-finite latent metrics")
    cosine = min(1.0, max(-1.0, cosine))  # Guard float32 rounding at endpoints.
    return mean_l1, relative_l1, cosine, 1.0 - cosine


def log_results(worker, args):
    print(f"Model={args.model_path}, seed={args.seed}, full_steps=50, mix_steps=22, "
          f"size={args.width}x{args.height}, frames={args.num_frames}, "
          f"flow_shift={args.flow_shift}, guidance_scale={args.guidance_scale}", flush=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["prompt_id", "seed", "full_steps", "mix_steps", "mean_l1",
                         "relative_l1", "cosine_similarity", "cosine_distance", "prompt"])
        for index, row in enumerate(args.prompts, start=1):
            prompt = row["prompt_en"]
            print(f"\n[{index}/{len(args.prompts)}] {row['prompt_id']}: {prompt}", flush=True)
            full = final_latent(worker, prompt, args, mixed=False)
            mixed = final_latent(worker, prompt, args, mixed=True)
            scores = latent_metrics(full, mixed)
            writer.writerow([row["prompt_id"], args.seed, 50, 22, *scores, prompt])
            file.flush()  # Retain completed prompts if the Slurm job is interrupted.
            print(f"mean_l1={scores[0]:.6f}, relative_l1={scores[1]:.6f}, "
                  f"cosine_similarity={scores[2]:.6f}, cosine_distance={scores[3]:.6f}",
                  flush=True)
    print(f"\nResults: {args.output_csv}", flush=True)


def main(args):
    from sglang.multimodal_gen.runtime.distributed import cleanup_dist_env_and_memory
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args

    server_args = ServerArgs.from_kwargs(
        model_path=args.model_path, backend="sglang",
        num_gpus=1, tp_size=1, sp_degree=1, ulysses_degree=1, ring_degree=1,
        encoder_parallel="replicate", enable_cfg_parallel=False,
        enable_torch_compile=False, enable_breakable_cuda_graph=False,
        warmup_mode="off", attention_backend="fa", flow_shift=args.flow_shift,
        wan_timestep_schedule=0,
    )
    set_global_server_args(server_args)
    try:
        # Load once; compare both schedules in the same single-GPU worker.
        worker = GPUWorker(local_rank=0, rank=0,
                           master_port=server_args.master_port, server_args=server_args)
        decoding_stage = worker.pipeline.get_stage("decoding_stage")
        worker.pipeline.stages.remove(decoding_stage)
        log_results(worker, args)
    finally:
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main(parse_args())
