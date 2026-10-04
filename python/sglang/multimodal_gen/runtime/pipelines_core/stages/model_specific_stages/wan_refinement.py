# SPDX-License-Identifier: Apache-2.0
"""Opt-in, file-backed Wan 2.1 draft/refinement experiment stages."""

import os
import tempfile
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_world_rank,
    world_group_is_initialized,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.utils.wan_refinement import (
    load_clean_latent,
    make_latent_payload,
    refinement_sigmas,
    validate_refinement_options,
)


class WanRefinementPreparationStage(PipelineStage):
    """Run after ordinary noise/timestep preparation and before SP sharding."""

    def __init__(self, allow_refinement=True):
        super().__init__()
        self.allow_refinement = allow_refinement

    @property
    def parallelism_type(self):
        return StageParallelismType.REPLICATED

    def forward(self, batch, server_args):
        if not batch.wan_init_latent_path:
            return batch
        if not self.allow_refinement:
            raise ValueError("Use base WanPipeline, not WanDMDPipeline, for refinement")
        if batch.is_warmup:
            return batch
        validate_refinement_options(
            batch.sampling_params, type(server_args.pipeline_config).__name__
        )
        scheduler = batch.scheduler
        if not isinstance(scheduler, FlowUniPCMultistepScheduler):
            raise ValueError("Wan refinement requires FlowUniPCMultistepScheduler")
        if scheduler.config.use_dynamic_shifting or scheduler.solver_p is not None:
            raise ValueError("Wan refinement requires static shifting and the native UniPC solver")
        noise = batch.latents
        clean = load_clean_latent(
            batch.wan_init_latent_path, noise, batch, server_args.pipeline_config
        )
        original_steps = batch.num_inference_steps
        selected = refinement_sigmas(
            scheduler.sigmas.detach().cpu().numpy(),
            batch.wan_refine_sigma,
            batch.wan_refine_schedule,
        )
        # selected contains shifted sigmas already. shift=1 avoids shifting twice.
        # set_timesteps also clears model_outputs, last_sample and solver history.
        scheduler.set_timesteps(sigmas=selected, device=noise.device, shift=1.0)
        scheduler.set_begin_index(0)
        sigma = scheduler.sigmas[0].to(device=noise.device, dtype=torch.float32)
        batch.latents = ((1 - sigma) * clean.float() + sigma * noise.float()).to(noise.dtype)
        batch.timesteps = scheduler.timesteps
        batch.sigmas = scheduler.sigmas[:-1].detach().cpu().tolist()
        # Cache warmup, SCM and refresh must use the actual refinement step count.
        batch.num_inference_steps = len(batch.timesteps)
        batch.raw_latent_shape = batch.latents.shape
        self.log_info(
            "Wan refinement: schedule=%s requested_sigma=%.6f actual_sigma=%.6f "
            "steps=%d/%d source=%s",
            batch.wan_refine_schedule,
            batch.wan_refine_sigma,
            sigma.item(),
            batch.num_inference_steps,
            original_steps,
            batch.wan_init_latent_path,
        )
        return batch


class WanLatentExportStage(PipelineStage):
    """Save gathered, unpadded, normalized final latents before VAE decode."""

    @property
    def role_affinity(self):
        return RoleType.DECODER

    @property
    def parallelism_type(self):
        return StageParallelismType.REPLICATED

    def forward(self, batch, server_args):
        if not batch.wan_save_latent_path or batch.is_warmup:
            return batch
        validate_refinement_options(
            batch.sampling_params, type(server_args.pipeline_config).__name__
        )
        if world_group_is_initialized() and get_world_rank() != 0:
            return batch
        destination = Path(batch.wan_save_latent_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = make_latent_payload(batch.latents, batch, server_args)
        # An interrupted export must not leave a valid-looking partial draft.
        fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".pt.tmp")
        os.close(fd)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.log_info("Saved normalized Wan latent: %s", destination)
        return batch
