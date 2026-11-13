"""Compare Wan's guided model outputs and post-update latents for prompt pairs.

L1 is the mean absolute elementwise difference. Cosine is computed over the
flattened tensor. Each row uses a one-based step and its scheduler timestep.
"""

import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np


def tensor_metrics(a, b):
    if a.shape != b.shape or a.numel() == 0:
        raise ValueError("Expected matching, nonempty tensors")
    a = a.float().numpy().reshape(-1)
    b = b.float().numpy().reshape(-1)

    mean_l1 = float(np.abs(a - b).mean())
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        cosine = float(norm_a == norm_b)
    else:
        cosine = float(np.dot(a, b)) / (norm_a * norm_b)
    if not np.isfinite([mean_l1, norm_a, norm_b, cosine]).all():
        raise ValueError("Non-finite tensor metrics")
    return min(1.0, max(-1.0, cosine)), mean_l1


def generate(worker, prompt, label, args):
    from sglang.multimodal_gen.configs.sample.sampling_params import (
        SamplingParams,
        generate_request_id,
    )
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

    name = f"{label}_seed{args.seed}"
    perf_path = args.run_dir / "perf" / f"{name}.json"
    params = SamplingParams.from_user_sampling_params_args(
        args.model_path,
        server_args=worker.server_args,
        request_id=generate_request_id(),
        prompt=prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        progressive_mode="fullres",
        enable_cache_dit=False,
        enable_teacache=False,
        enable_spectrum=False,
        return_trajectory_latents=True,
        save_output=True,
        return_file_paths_only=True,
        output_path=str(args.run_dir / "videos"),
        output_file_name=f"{name}.mp4",
        perf_dump_path=str(perf_path),
    )
    request = prepare_request(worker.server_args, params)
    denoising = worker.pipeline.get_stage("denoising_stage")
    predict = denoising._predict_noise_with_cfg
    outputs = []

    def capture_prediction(*positional, **kwargs):
        prediction = predict(*positional, **kwargs)
        outputs.append(prediction.detach().to(device="cpu", copy=True))
        return prediction

    with patch.object(denoising, "_predict_noise_with_cfg", capture_prediction):
        result = worker.execute_forward([request])
    if result.error:
        raise RuntimeError(f"{label}: {result.error}")

    latents = result.trajectory_latents
    timesteps = result.trajectory_timesteps
    steps = args.num_inference_steps
    if (
        len(outputs) != steps
        or latents is None
        or latents.shape[1] != steps
        or timesteps is None
        or timesteps.shape != (steps,)
    ):
        raise RuntimeError(f"{label}: expected {steps} model outputs, latents and timesteps")
    for path in (Path(request.output_file_path()), perf_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"{label}: missing or empty output {path}")
    return outputs, latents, timesteps.tolist(), Path(request.output_file_path())


def compare_pair(worker, pair, args):
    prompt_id = pair["prompt_id"]
    print(f"Comparing {prompt_id}, seed={args.seed}", flush=True)
    a_outputs, a_latents, a_times, _ = generate(
        worker, pair["prompt_a"], f"{prompt_id}_prompt_a", args
    )
    en_outputs, en_latents, en_times, en_video = generate(
        worker, pair["prompt_en"], f"{prompt_id}_prompt_en", args
    )
    if a_times != en_times:
        raise RuntimeError(f"{prompt_id}: scheduler timesteps differ between prompts")

    rows = []
    for index, timestep in enumerate(a_times):
        output_scores = tensor_metrics(a_outputs[index], en_outputs[index])
        latent_scores = tensor_metrics(a_latents[:, index], en_latents[:, index])
        rows.append(
            (prompt_id, args.seed, index + 1, timestep, *output_scores, *latent_scores)
        )

    # Keep the baseline's VBench filename for the English prompt.
    en_video.rename(en_video.with_name(f"{pair['prompt_en']}-{args.sample_index}.mp4"))
    return rows


def main(args):
    from sglang.multimodal_gen.runtime.distributed import cleanup_dist_env_and_memory
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args

    server_args = ServerArgs.from_kwargs(
        model_path=args.model_path,
        backend="sglang",
        num_gpus=1,
        tp_size=1,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        flow_shift=args.flow_shift,
        enable_cfg_parallel=False,
        enable_torch_compile=False,
        enable_breakable_cuda_graph=False,
        warmup_mode="off",
    )
    set_global_server_args(server_args)
    try:
        worker = GPUWorker(
            local_rank=0, rank=0, master_port=server_args.master_port, server_args=server_args
        )
        with (args.run_dir / "step_similarity.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow([
                "prompt_id", "seed", "step", "timestep",
                "output_cosine_similarity", "output_mean_l1",
                "latent_cosine_similarity", "latent_mean_l1",
            ])
            for pair in args.prompts:
                writer.writerows(compare_pair(worker, pair, args))
                file.flush()
    finally:
        cleanup_dist_env_and_memory()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--num-inference-steps", type=int, required=True)
    parser.add_argument("--flow-shift", type=float, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    args = parser.parse_args()
    if args.num_inference_steps < 1:
        parser.error("--num-inference-steps must be positive")
    with args.prompt_manifest.open(encoding="utf-8-sig") as file:
        manifest = json.load(file)
    args.seed = manifest["seed"]
    args.sample_index = manifest["sample_index"]
    args.prompts = manifest["prompts"]
    if not args.prompts:
        parser.error("prompt manifest is empty")
    return args


if __name__ == "__main__":
    main(parse_args())
