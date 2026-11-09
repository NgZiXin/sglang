"""Run the generation CLI with the hardcoded combined Wan schedule."""

import numpy as np

from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
    TimestepPreparationStage,
)


# Start, 10-step targets 1-2, 50-step targets 11-15,
# then 10-step targets 4-10, including terminal zero.
MIXED_SIGMAS = [
    1.00000000,
    0.98630137,
    0.96969697,
    0.96594427,
    0.96202532,
    0.95792880,
    0.95364238,
    0.94915254,
    0.92307692,
    0.88888889,
    0.84210526,
    0.77419355,
    0.66666667,
    0.47058824,
    0.00000000,
]


_original_forward = TimestepPreparationStage.forward


def prepare_mixed_timesteps(self, batch, server_args):
    if not batch.is_warmup:
        # UniPC appends terminal zero and applies flow shift itself.
        sigmas = np.asarray(MIXED_SIGMAS[:-1], dtype=np.float64)
        shift = self.scheduler.config.shift
        batch.sigmas = sigmas / (shift - (shift - 1) * sigmas)
        # Preserve UniPC's native start below one to keep log(1 - sigma) finite.
        batch.sigmas[0] = self.scheduler.sigma_max
        batch.num_inference_steps = len(MIXED_SIGMAS) - 1
    return _original_forward(self, batch, server_args)


# Spawned GPU workers import this module too, so they receive the same override.
TimestepPreparationStage.forward = prepare_mixed_timesteps


if __name__ == "__main__":
    from sglang.cli.main import main

    main()
