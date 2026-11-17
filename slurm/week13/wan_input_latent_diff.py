"""Compare Wan 2.1 input latents for each prompt_a / prompt_en pair.

Step 1 is the shared initial noise. At every step, measure the latent
after the model dtype cast and scheduler input scaling, before CFG prediction.
Relative mean L1 divides mean(abs(a - en)) by the average of mean(abs(a))
and mean(abs(en)), floored at 1e-8, as in the week11 experiments.
"""

import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "vbench_33_prompts_wan_chai_kv.json",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("input_latent_diff.csv"))
    parser.add_argument("--model-path", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--prompt-id", help="Run just this pair; defaults to all pairs")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        args.prompts = load_pairs(args.prompt_manifest, args.prompt_id)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Invalid prompt selection: {error}")
    if args.num_inference_steps < 1:
        parser.error("--num-inference-steps must be positive")
    return args


def load_pairs(path, prompt_id):
    with path.open(encoding="utf-8-sig") as file:
        pairs = json.load(file)["prompts"]
    seen = set()
    for row in pairs:
        for key in ("prompt_id", "prompt_a", "prompt_en"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Expected a nonempty string for {key}")
        if row["prompt_id"] in seen:
            raise ValueError(f"Duplicate prompt_id: {row['prompt_id']}")
        seen.add(row["prompt_id"])
    selected = [row for row in pairs if prompt_id is None or row["prompt_id"] == prompt_id]
    if not selected:
        raise ValueError("Manifest has no prompts" if prompt_id is None else f"Unknown prompt_id: {prompt_id}")
    return selected


def latent_metrics(a, en):
    if a.shape != en.shape or a.size == 0:
        raise RuntimeError(f"Expected nonempty latents of matching shape, got {a.shape} and {en.shape}")
    mean_l1 = float(np.abs(a - en).mean())
    scale = 0.5 * (float(np.abs(a).mean()) + float(np.abs(en).mean()))
    if not np.isfinite(mean_l1) or not np.isfinite(scale):
        raise RuntimeError("Non-finite input latent metrics")
    return mean_l1, mean_l1 / max(scale, 1e-8)


def run_prompt(worker, stage, prompt, args, on_latent):
    import torch

    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

    params = SamplingParams.from_user_sampling_params_args(
        args.model_path, server_args=worker.server_args,
        prompt=prompt, seed=args.seed, height=args.height, width=args.width,
        num_frames=args.num_frames, fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale, progressive_mode="fullres",
        save_output=False, enable_cache_dit=False, enable_teacache=False,
        enable_spectrum=False,
    )
    request = prepare_request(worker.server_args, params)
    predict = stage._predict_noise_with_cfg
    count = 0

    def capture(*, latent_model_input, timestep_index, **kwargs):
        nonlocal count
        if timestep_index != count or count >= args.num_inference_steps:
            raise RuntimeError(f"Unexpected denoising step index: {timestep_index}")
        # This hook runs once per step, before the positive and negative CFG branches.
        latent = latent_model_input.detach().to(device="cpu", dtype=torch.float32, copy=True)
        on_latent(count, latent.numpy())
        count += 1
        return predict(latent_model_input=latent_model_input, timestep_index=timestep_index, **kwargs)

    with patch.object(stage, "_predict_noise_with_cfg", capture):
        result = worker.pipeline.forward(request, worker.server_args)
    if count != args.num_inference_steps or len(result.timesteps) != count:
        raise RuntimeError(f"Captured {count} inputs, expected {args.num_inference_steps}")
    return result.timesteps.tolist()


def compare_pair(worker, stage, row, args):
    reference = []
    scores = []

    def compare(index, latent):
        if index == 0 and not np.array_equal(reference[index], latent):
            raise RuntimeError(f"{row['prompt_id']}: initial latents differ despite matching seeds")
        scores.append(latent_metrics(reference[index], latent))

    schedule_a = run_prompt(worker, stage, row["prompt_a"], args,
                            lambda _, latent: reference.append(latent))
    schedule_en = run_prompt(worker, stage, row["prompt_en"], args, compare)
    if schedule_a != schedule_en:
        raise RuntimeError(f"{row['prompt_id']}: denoising schedules differ")
    return zip(schedule_a, scores)


def main(args):
    from sglang.multimodal_gen.runtime.distributed import cleanup_dist_env_and_memory
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args

    server_args = ServerArgs.from_kwargs(
        model_path=args.model_path, backend="sglang", num_gpus=1, tp_size=1,
        sp_degree=1, ulysses_degree=1, ring_degree=1,
        encoder_parallel="replicate", enable_cfg_parallel=False,
        enable_torch_compile=False, enable_breakable_cuda_graph=False,
        warmup_mode="off", attention_backend="fa", flow_shift=args.flow_shift,
        wan_timestep_schedule=0,
    )
    set_global_server_args(server_args)
    try:
        worker = GPUWorker(local_rank=0, rank=0,
                           master_port=server_args.master_port, server_args=server_args)
        pipeline = worker.pipeline
        route = pipeline.get_stage("denoising_stage")
        stage = getattr(route, "standard_stage", route)
        pipeline.stages.remove(pipeline.get_stage("decoding_stage"))

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["prompt_id", "seed", "step", "timestep", "mean_l1",
                             "relative_mean_l1", "prompt_a", "prompt_en"])
            for index, row in enumerate(args.prompts, start=1):
                print(f"[{index}/{len(args.prompts)}] {row['prompt_id']}", flush=True)
                for step, (timestep, scores) in enumerate(compare_pair(worker, stage, row, args), start=1):
                    writer.writerow([row["prompt_id"], args.seed, step, timestep, *scores,
                                     row["prompt_a"], row["prompt_en"]])
                file.flush()
        print(f"Results: {args.output_csv}", flush=True)
    finally:
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main(parse_args())
