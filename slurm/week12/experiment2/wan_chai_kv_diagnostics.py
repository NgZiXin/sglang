"""Measure cross-request Wan K/V replacement on selected denoising steps."""

import csv
import json
import math
import uuid
from pathlib import Path
from unittest.mock import patch

from wan_chai_kv_vbench import build_cache_mapping, parse_args


def tensor_metrics(name, baseline, replacement):
    import torch
    import torch.nn.functional as F

    baseline = baseline.detach().float().reshape(-1)
    replacement = replacement.detach().float().reshape(-1)
    baseline_norm = torch.linalg.vector_norm(baseline)
    replacement_norm = torch.linalg.vector_norm(replacement)
    error = torch.linalg.vector_norm(replacement - baseline)
    values = torch.stack((
        error / baseline_norm.clamp_min(1e-8),
        F.cosine_similarity(baseline, replacement, dim=0),
        replacement_norm / baseline_norm.clamp_min(1e-8),
    )).tolist()
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"Non-finite {name} metric")
    return dict(zip((f"{name}_{metric}" for metric in (
        "relative_l2", "cosine", "norm_ratio",
    )), values))


def write_metrics(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure(worker, args, source_steps, mapping, output_dir, *, control=False):
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
    from sglang.multimodal_gen.runtime.utils.wan_chai_kv import WanChaiKVCache

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_id = uuid.uuid4().hex
    source_prompt = args.prompt_b if control else args.prompt_a
    target_steps = [row["b_step"] - 1 for row in mapping]
    mapping_by_target = dict(zip(target_steps, mapping))
    attention = worker.pipeline.get_module("transformer").blocks[0].attn1
    original_apply = WanChaiKVCache.apply
    cached = {}
    pending = None
    rows = []

    def observe_apply(cache, request, step, key, value):
        nonlocal pending
        live_key, live_value = original_apply(cache, request, step, key, value)
        mode = request.wan_chai_kv["mode"]
        negative = bool(request.is_cfg_negative)
        if mode == "record":
            entry = cache.entries.get((step, negative))
            if entry is not None:
                cached[step, negative] = entry
        elif step in mapping_by_target:
            matched = mapping_by_target[step]
            source_key, source_value = cached[matched["a_step"] - 1, negative]
            pending = (mode, matched, negative, key, value, source_key, source_value)
        return live_key, live_value

    def measure_attention(module, inputs, output):
        nonlocal pending
        if pending is None:
            return
        mode, matched, negative, key, value, source_key, source_value = pending
        pending = None
        query, live_key, live_value = inputs
        if mode == "reuse":
            if live_key is not source_key or live_value is not source_value:
                raise RuntimeError("Expected cached A K/V in the replacement run")
            baseline = module.forward(query, key, value)
            replacement = output
        else:
            if live_key is not key or live_value is not value:
                raise RuntimeError("Expected native B K/V in the baseline run")
            baseline = output
            replacement = module.forward(query, source_key, source_value)

        row = dict(
            comparison="same_prompt" if control else "cross_prompt",
            prompt_a=source_prompt,
            prompt_b=args.prompt_b,
            trajectory="b_reuse" if mode == "reuse" else "b_clean",
            seed=args.seed,
            **matched,
            branch="negative" if negative else "positive",
        )
        row.update(tensor_metrics("key", key, source_key))
        row.update(tensor_metrics("value", value, source_value))
        row.update(tensor_metrics("attention", baseline, replacement))
        rows.append(row)

    def run(label, prompt, mode, steps, *, save_video=False):
        report_path = output_dir / f"{label}_kv.json"
        settings = dict(
            request_id=f"{cache_id}-{label}", prompt=prompt,
            height=args.height, width=args.width, num_frames=args.num_frames,
            fps=args.fps, seed=args.seed, num_inference_steps=steps,
            guidance_scale=args.guidance_scale, cfg_gate_step=1.0,
            progressive_mode="fullres", enable_cache_dit=False,
            enable_teacache=False, enable_spectrum=False,
            save_output=save_video, return_file_paths_only=save_video,
            output_path=str(output_dir), output_file_name=f"{label}.mp4",
            wan_chai_kv=dict(
                mode=mode, reuse_strategy="replace", cache_id=cache_id,
                source_steps=source_steps, target_steps=target_steps,
                report_path=str(report_path),
            ),
        )
        if args.negative_prompt is not None:
            settings["negative_prompt"] = args.negative_prompt
        params = SamplingParams.from_user_sampling_params_args(
            args.model_path, server_args=worker.server_args, **settings,
        )
        request = prepare_request(worker.server_args, params)
        result = worker.execute_forward([request])
        if result is None or getattr(result, "error", None):
            raise RuntimeError(f"{label} failed: {getattr(result, 'error', result)}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected = {
            "record": (2 * len(set(source_steps)), 0),
            "reuse": (0, 2 * len(target_steps)),
            "off": (0, 0),
        }[mode]
        if (report["cache_id"] != cache_id or report["mode"] != mode
                or report["block0_calls"] != 2 * steps
                or (report["captures"], report["replacements"]) != expected):
            raise RuntimeError(f"Unexpected {label} cache report: {report}")
        if save_video:
            video = output_dir / f"{label}.mp4"
            if not video.is_file() or video.stat().st_size == 0:
                raise RuntimeError(f"Missing video: {video}")

    hook = attention.register_forward_hook(measure_attention)
    try:
        with patch.object(WanChaiKVCache, "apply", observe_apply):
            run("a_full", source_prompt, "record", args.steps_a)
            if not control:
                run("b_chai_kv", args.prompt_b, "reuse", args.steps_b, save_video=True)
            # Off mode clears the runtime cache. The experiment keeps its own
            # references for counterfactual attention on the clean B trajectory.
            run("b_full", args.prompt_b, "off", args.steps_b, save_video=not control)
    finally:
        hook.remove()

    trajectories = {"b_clean"} if control else {"b_clean", "b_reuse"}
    expected_rows = {(trajectory, row["b_step"], branch)
                     for trajectory in trajectories for row in mapping
                     for branch in ("positive", "negative")}
    actual_rows = {(row["trajectory"], row["b_step"], row["branch"]) for row in rows}
    if actual_rows != expected_rows or len(rows) != len(expected_rows):
        raise RuntimeError(f"Unexpected diagnostics rows: {actual_rows}")
    return rows


def main(args):
    import os

    from sglang.multimodal_gen.runtime.distributed import cleanup_dist_env_and_memory
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    same_prompt_control = os.environ.get("WAN_KV_SAME_PROMPT_CONTROL") == "1"
    step2_only = os.environ.get("WAN_KV_STEP2_ONLY") == "1"
    server_args = ServerArgs.from_kwargs(
        model_path=args.model_path, backend="sglang", num_gpus=1, tp_size=1,
        sp_degree=1, ulysses_degree=1, ring_degree=1,
        encoder_parallel="replicate", enable_cfg_parallel=False,
        enable_torch_compile=False, enable_breakable_cuda_graph=False,
        warmup_mode="off", attention_backend="fa", flow_shift=args.flow_shift,
        wan_timestep_schedule=args.wan_timestep_schedule,
    )
    set_global_server_args(server_args)
    try:
        worker = GPUWorker(local_rank=0, rank=0, master_port=server_args.master_port,
                           server_args=server_args)
        _, mapping = build_cache_mapping(
            server_args.pipeline_config, args.steps_a, args.steps_b,
        )
        if step2_only:
            mapping = [match for match in mapping if match["b_step"] == 2]
        source_steps = [match["a_step"] - 1 for match in mapping]
        metadata = {**vars(args), "output_dir": str(output_dir),
                    "cache_mapping_policy": "nearest_sigma", "cache_mapping": mapping,
                    "replacement_steps": [match["b_step"] for match in mapping],
                    "same_prompt_control": same_prompt_control}
        (output_dir / "experiment.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8",
        )
        print(f"K/V mapping (A -> B): {mapping}", flush=True)
        rows = measure(worker, args, source_steps, mapping, output_dir)
        write_metrics(output_dir / "kv_diagnostics.csv", rows)
        if same_prompt_control:
            rows.extend(measure(worker, args, source_steps, mapping,
                                output_dir / "same_prompt_control", control=True))
            write_metrics(output_dir / "kv_diagnostics.csv", rows)
    finally:
        cleanup_dist_env_and_memory()
    print(f"Diagnostics and B videos: {output_dir}", flush=True)


if __name__ == "__main__":
    main(parse_args())
