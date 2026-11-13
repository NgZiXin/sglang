"""Compare Wan prompt A with a shorter English trajectory using CHAI K/V reuse."""

import argparse
import csv
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

from compare_prompt_trajectories import tensor_metrics


# One-based guided-output comparisons, independent of CHAI's K/V mapping.
COMPARISON_STEPS = ((1, 5), (2, 10), (3, 15))  # (English step, A step)
CHAI_TARGET_STEPS = [2, 3, 4]


def generate(worker, prompt, label, num_steps, capture_steps, cache, args):
    from sglang.multimodal_gen.configs.sample.sampling_params import (
        SamplingParams,
        generate_request_id,
    )
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
    from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.denoising import (
        ProgressiveDenoisingStageRouter,
    )

    name = f"{label}_seed{args.seed}"
    perf_path = args.run_dir / "perf" / f"{name}.json"
    report_path = args.run_dir / "kv" / f"{name}.json"
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
        num_inference_steps=num_steps,
        guidance_scale=args.guidance_scale,
        progressive_mode="fullres",
        enable_cache_dit=False,
        enable_teacache=False,
        enable_spectrum=False,
        cfg_gate_step=1.0,
        wan_chai_kv=dict(cache, report_path=str(report_path)),
        return_trajectory_latents=True,
        save_output=True,
        return_file_paths_only=True,
        output_path=str(args.run_dir / "videos"),
        output_file_name=f"{name}.mp4",
        perf_dump_path=str(perf_path),
    )
    request = prepare_request(worker.server_args, params)
    denoising = worker.pipeline.get_stage("denoising_stage")
    if isinstance(denoising, ProgressiveDenoisingStageRouter):
        denoising = denoising.standard_stage
    predict = denoising._predict_noise_with_cfg
    outputs = {}
    step = 0

    def capture_prediction(*positional, **kwargs):
        nonlocal step
        prediction = predict(*positional, **kwargs)
        step += 1
        if step in capture_steps:
            outputs[step] = prediction.detach().to(device="cpu", copy=True)
        return prediction

    with patch.object(denoising, "_predict_noise_with_cfg", capture_prediction):
        result = worker.execute_forward([request])
    if result.error:
        raise RuntimeError(f"{label}: {result.error}")

    latents = result.trajectory_latents
    timesteps = result.trajectory_timesteps
    if (
        step != num_steps
        or outputs.keys() != capture_steps
        or latents is None
        or latents.shape[1] != num_steps
        or timesteps is None
        or timesteps.shape != (num_steps,)
    ):
        raise RuntimeError(f"{label}: incomplete {num_steps}-step trajectory")

    video = Path(request.output_file_path())
    for path in (video, perf_path, report_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"{label}: missing or empty output {path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    captures, replacements = (
        (2 * len(set(cache["source_steps"])), 0)
        if cache["mode"] == "record"
        else (0, 2 * len(CHAI_TARGET_STEPS))
    )
    expected = {
        "mode": cache["mode"],
        "cache_id": cache["cache_id"],
        "reuse_strategy": "replace",
        "num_steps": num_steps,
        "block0_calls": 2 * num_steps,
        "source_steps": [index + 1 for index in cache["source_steps"]],
        "target_steps": CHAI_TARGET_STEPS,
        "captures": captures,
        "replacements": replacements,
    }
    for key, value in expected.items():
        if report[key] != value:
            raise RuntimeError(f"{label}: expected K/V {key}={value}, got {report[key]}")

    snapshots = {
        index: (outputs[index], latents[:, index - 1].clone(), timesteps[index - 1].item())
        for index in capture_steps
    }
    return snapshots, video


def compare_pair(worker, pair, source_steps, args):
    prompt_id = pair["prompt_id"]
    print(f"Comparing {prompt_id}, seed={args.seed}", flush=True)
    cache = {
        "cache_id": uuid.uuid4().hex,
        "source_steps": source_steps,
        "target_steps": [step - 1 for step in CHAI_TARGET_STEPS],
        "reuse_strategy": "replace",
    }
    a_snapshots, _ = generate(
        worker, pair["prompt_a"], f"{prompt_id}_prompt_a", args.steps_a,
        {a_step for _, a_step in COMPARISON_STEPS}, dict(cache, mode="record"), args,
    )
    en_snapshots, en_video = generate(
        worker, pair["prompt_en"], f"{prompt_id}_prompt_en_chai_kv", args.steps_en,
        {en_step for en_step, _ in COMPARISON_STEPS}, dict(cache, mode="reuse"), args,
    )

    rows = []
    for en_step, a_step in COMPARISON_STEPS:
        a_output, a_latent, a_timestep = a_snapshots[a_step]
        en_output, en_latent, en_timestep = en_snapshots[en_step]
        rows.append((
            prompt_id, args.seed, en_step, a_step, en_timestep, a_timestep,
            *tensor_metrics(a_output, en_output),
            *tensor_metrics(a_latent, en_latent),
        ))

    en_video.rename(en_video.with_name(f"{pair['prompt_en']}-{args.sample_index}.mp4"))
    return rows


def main(args):
    from sglang.multimodal_gen.runtime.distributed import cleanup_dist_env_and_memory
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args

    # Reuse the week 11 scheduler-sigma mapping.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week11"))
    from wan_chai_kv import build_cache_mapping

    server_args = ServerArgs.from_kwargs(
        model_path=args.model_path,
        backend="sglang",
        num_gpus=1,
        tp_size=1,
        sp_degree=1,
        ulysses_degree=1,
        ring_degree=1,
        encoder_parallel="replicate",
        flow_shift=args.flow_shift,
        enable_cfg_parallel=False,
        enable_torch_compile=False,
        enable_breakable_cuda_graph=False,
        warmup_mode="off",
        attention_backend="fa",
    )
    set_global_server_args(server_args)
    try:
        worker = GPUWorker(
            local_rank=0, rank=0, master_port=server_args.master_port, server_args=server_args
        )
        source_steps, mapping = build_cache_mapping(
            worker.server_args.pipeline_config, args.steps_a, args.steps_en,
        )
        print(f"K/V mapping (A -> English): {mapping}", flush=True)
        with (args.run_dir / "step_similarity.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow([
                "prompt_id", "seed", "step_en", "step_a", "timestep_en", "timestep_a",
                "output_cosine_similarity", "output_mean_l1",
                "latent_cosine_similarity", "latent_mean_l1",
            ])
            for pair in args.prompts:
                writer.writerows(compare_pair(worker, pair, source_steps, args))
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
    parser.add_argument("--steps-a", type=int, choices=[50], required=True)
    parser.add_argument("--steps-en", type=int, choices=[10], required=True)
    parser.add_argument("--flow-shift", type=float, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    args = parser.parse_args()
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
