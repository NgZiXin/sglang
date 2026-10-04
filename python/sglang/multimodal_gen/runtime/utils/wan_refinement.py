# SPDX-License-Identifier: Apache-2.0
"""Portable normalized Wan 2.1 latents and explicit refinement schedules.

Keep scheduler math independent of the GPU runtime so it can be tested on CPU.
"""

import math
from pathlib import Path

import numpy as np

LATENT_FORMAT = "sglang-wan21-clean-latent-v1"
BASE_CONFIGS = {"WanT2V480PConfig", "WanT2V720PConfig"}
EXPORT_CONFIGS = BASE_CONFIGS | {"FastWan2_1_T2V_480P_Config"}


def validate_refinement_options(params, config_name=None):
    source = params.wan_init_latent_path
    destination = params.wan_save_latent_path
    sigma = params.wan_refine_sigma
    mode = params.wan_refine_schedule
    if mode not in ("rescale", "suffix"):
        raise ValueError("wan_refine_schedule must be rescale or suffix")
    if bool(source) != (sigma is not None):
        raise ValueError("wan_init_latent_path and wan_refine_sigma must be set together")
    if sigma is not None and (
        isinstance(sigma, bool)
        or not isinstance(sigma, (int, float))
        or not math.isfinite(sigma)
        or not 0 < sigma <= 1
    ):
        raise ValueError("wan_refine_sigma must be finite and in (0, 1]")
    if not source and mode != "rescale":
        raise ValueError("wan_refine_schedule requires wan_init_latent_path")
    if not source and not destination:
        return
    for value in (source, destination):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("Wan latent paths must be non-empty strings")
    if (
        params.num_outputs_per_prompt != 1
        or (isinstance(params.prompt, list) and len(params.prompt) != 1)
        or params.prompt_path
        or (isinstance(params.seed, list) and len(params.seed) != 1)
    ):
        raise ValueError("Wan latent experiments support one prompt and one output per run")
    if params.progressive_mode != "fullres" or params.rollout:
        raise ValueError("Wan latent experiments require fullres mode and rollout=False")
    if source and destination and Path(source).resolve() == Path(destination).resolve():
        raise ValueError("Use different input and output latent paths to preserve the draft")
    if config_name is not None:
        if source and config_name not in BASE_CONFIGS:
            raise ValueError("Latent refinement supports only base Wan 2.1 T2V pipelines")
        if destination and config_name not in EXPORT_CONFIGS:
            raise ValueError("Latent export supports only Wan 2.1 T2V and FastWan 2.1 T2V")


def refinement_sigmas(full_sigmas, start_sigma, mode):
    """Return *already shifted* positive sigmas, excluding terminal zero.

    rescale: retain the full step count and relative spacing, scaling the entire
    schedule to the requested maximum sigma. suffix: retain only existing steps
    at or below that sigma. Neither mode equates step fraction with noise level.
    """
    sigmas = np.asarray(full_sigmas, dtype=np.float64)
    if (
        sigmas.ndim != 1
        or len(sigmas) < 2
        or not np.isfinite(sigmas).all()
        or sigmas[-1] != 0
        or not np.all(np.diff(sigmas) < 0)
        or not 0 < sigmas[0] < 1
    ):
        raise ValueError("Expected descending Wan sigmas below 1 ending at zero")
    if not math.isfinite(start_sigma) or not 0 < start_sigma <= 1:
        raise ValueError("start_sigma must be finite and in (0, 1]")
    positive = sigmas[:-1]
    if mode == "rescale":
        # Do not introduce sigma=1: UniPC's flow alpha would be zero.
        return positive * (min(start_sigma, positive[0]) / positive[0])
    if mode == "suffix":
        selected = positive[positive <= start_sigma]
        if not len(selected):
            raise ValueError("No schedule step at this sigma; increase sigma or use rescale")
        return selected.copy()
    raise ValueError("Unknown Wan refinement schedule")


def vae_signature(config):
    arch = config.vae_config.arch_config
    return {
        "z_dim": arch.z_dim,
        "spatial_compression_ratio": arch.spatial_compression_ratio,
        "temporal_compression_ratio": arch.temporal_compression_ratio,
        "latents_mean": list(arch.latents_mean),
        "latents_std": list(arch.latents_std),
    }


def make_latent_payload(latents, batch, server_args):
    return {
        "format": LATENT_FORMAT,
        "layout": "BCTHW",
        "latents": latents.detach().float().cpu().contiguous(),
        "vae": vae_signature(server_args.pipeline_config),
        "height": batch.height,
        "width": batch.width,
        "num_frames": batch.num_frames,
        "prompt": batch.prompt,
        "seed": batch.seed,
        "model_path": str(server_args.model_path),
    }


def load_clean_latent(path, noise, batch, config):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != LATENT_FORMAT:
        raise ValueError("Expected a latent exported with --wan-save-latent-path")
    if payload.get("layout") != "BCTHW" or payload.get("vae") != vae_signature(config):
        raise ValueError("Draft latent layout or VAE normalization does not match Wan")
    for name in ("height", "width", "num_frames"):
        if payload.get(name) != getattr(batch, name):
            raise ValueError(f"Draft {name} does not match the refinement request")
    clean = payload.get("latents")
    if (
        not isinstance(clean, torch.Tensor)
        or clean.ndim != 5
        or clean.shape != noise.shape
        or not clean.is_floating_point()
        or not torch.isfinite(clean).all().item()
    ):
        raise ValueError("Draft must be finite floating-point latents matching the requested shape")
    return clean.to(device=noise.device, dtype=noise.dtype)
