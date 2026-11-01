"""Compare first-step guided Wan flow predictions for supplied prompt pairs.

Lower mean_l1 / relative_l1 means closer outputs; higher cosine_similarity
means closer directions. These scores do not establish a safe reuse depth.
Uses SGLang's native model loader in one process; video decoding is skipped.
"""

import argparse
import csv
import math
import sys
from functools import wraps
from pathlib import Path


def load_prompt_pairs(source):
    """Read nonempty pair_id|prompt_a|prompt_b lines from a file or stdin."""
    def read_pairs(lines):
        pairs = []
        seen_ids = set()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            fields = [field.strip() for field in line.rstrip("\r\n").split("|")]
            if len(fields) != 3 or not all(fields):
                raise ValueError(
                    f"{source}:{line_number}: expected pair_id|prompt_a|prompt_b "
                    "with three nonempty fields"
                )
            if fields[0] in seen_ids:
                raise ValueError(f"{source}:{line_number}: duplicate pair_id {fields[0]!r}")
            seen_ids.add(fields[0])
            pairs.append(tuple(fields))
        if not pairs:
            raise ValueError(f"{source}: no prompt pairs supplied")
        return pairs

    if source == "-":
        return read_pairs(sys.stdin)
    with Path(source).open(encoding="utf-8-sig") as file:
        return read_pairs(file)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--prompt-pairs", required=True, metavar="PATH",
        help="UTF-8 file containing pair_id|prompt_a|prompt_b lines; use - for stdin",
    )
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
    args = parser.parse_args()
    try:
        args.prompt_pairs = load_prompt_pairs(args.prompt_pairs)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    return args


def first_prediction(worker, prompt, args):
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

    # A fresh request with the same seed recreates the same starting noise.
    params = SamplingParams.from_user_sampling_params_args(
        args.model_path, server_args=worker.server_args,
        prompt=prompt, negative_prompt=args.negative_prompt, seed=args.seed,
        height=args.height, width=args.width, num_frames=args.num_frames,
        num_inference_steps=1, progressive_mode="fullres", save_output=False,
        enable_cache_dit=False, enable_teacache=False, enable_spectrum=False,
    )
    request = prepare_request(worker.server_args, params)
    pipe = worker.pipeline
    scheduler = pipe.get_module("scheduler")
    captured = []
    original_step = scheduler.step

    @wraps(original_step)
    def capture_step(model_output, timestep, sample, **kwargs):
        # SGLang has already applied CFG here. Save the flow prediction
        # BEFORE the scheduler updates the latent, and compare in float32.
        captured.append((model_output.detach().float().cpu(), float(timestep)))
        return original_step(model_output, timestep, sample, **kwargs)

    scheduler.step = capture_step
    try:
        pipe.forward(request, worker.server_args)
    finally:
        scheduler.step = original_step
    if len(captured) != 1:
        raise RuntimeError(f"Expected one scheduler step, got {len(captured)}")
    return captured[0]


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
    )
    set_global_server_args(server_args)
    try:
        # Same native worker/loader as SGLang generation, kept in this process
        # so the scheduler prediction can be captured without server IPC.
        worker = GPUWorker(local_rank=0, rank=0,
                           master_port=server_args.master_port, server_args=server_args)
        # Stop after denoising; there is no need to decode or save a video.
        decoding_stage = worker.pipeline.get_stage("decoding_stage")
        worker.pipeline.stages.remove(decoding_stage)
        log_pairs(worker, args)
    finally:
        cleanup_dist_env_and_memory()


def log_pairs(worker, args):
    import torch

    print(f"Model={args.model_path}, seed={args.seed}, steps=1, "
          f"size={args.width}x{args.height}, frames={args.num_frames}, "
          f"flow_shift={args.flow_shift}", flush=True)
    print(f"Negative prompt: {args.negative_prompt}", flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["pair_id", "seed", "timestep", "mean_l1", "relative_l1",
                         "cosine_similarity", "prompt_a", "prompt_b"])
        for pair_id, prompt_a, prompt_b in args.prompt_pairs:
            print(f"\n{pair_id}\nA: {prompt_a}\nB: {prompt_b}", flush=True)
            a, timestep_a = first_prediction(worker, prompt_a, args)
            b, timestep_b = first_prediction(worker, prompt_b, args)
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
