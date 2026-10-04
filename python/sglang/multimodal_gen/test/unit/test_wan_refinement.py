# SPDX-License-Identifier: Apache-2.0
"""Run directly on a CPU with numpy; tensor checks additionally require torch."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Load the numerical/serialization helper without importing SGLang's GPU stack.
HELPER_PATH = Path(__file__).resolve().parents[2] / "runtime/utils/wan_refinement.py"
spec = importlib.util.spec_from_file_location("wan_refinement_helpers", HELPER_PATH)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

try:
    import torch
except ImportError:
    torch = None


def options(**overrides):
    values = dict(
        wan_init_latent_path=None, wan_save_latent_path=None,
        wan_refine_sigma=None, wan_refine_schedule="rescale",
        num_outputs_per_prompt=1, prompt="A moving tram", prompt_path=None,
        seed=42, progressive_mode="fullres", rollout=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestWanRefinementSchedule(unittest.TestCase):
    def test_rescale_keeps_fifty_steps_and_spacing(self):
        unshifted = np.linspace(0.999, 0, 51)[:-1]
        shifted = 3 * unshifted / (1 + 2 * unshifted)
        result = helpers.refinement_sigmas(np.append(shifted, 0), 0.4, "rescale")
        self.assertEqual(len(result), 50)
        self.assertAlmostEqual(result[0], 0.4)
        np.testing.assert_allclose(result / result[0], shifted / shifted[0])
        self.assertTrue(np.all(np.diff(result) < 0))
        self.assertGreater(result[-1], 0)

    def test_suffix_preserves_existing_noise_levels(self):
        result = helpers.refinement_sigmas([0.99, 0.8, 0.5, 0.2, 0], 0.6, "suffix")
        np.testing.assert_array_equal(result, [0.5, 0.2])

    def test_sigma_one_does_not_create_unipc_singularity(self):
        result = helpers.refinement_sigmas([0.999, 0.5, 0], 1, "rescale")
        np.testing.assert_array_equal(result, [0.999, 0.5])

    def test_empty_suffix_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "No schedule step"):
            helpers.refinement_sigmas([0.99, 0.2, 0], 0.1, "suffix")

    def test_bad_schedules_and_sigmas(self):
        for schedule in ([1, 0], [0.5, 0.7, 0], [0.5, 0.5, 0], [0.5], [np.nan, 0]):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                helpers.refinement_sigmas(schedule, 0.4, "rescale")
        for sigma in (0, -1, 1.1, np.nan, np.inf):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                helpers.refinement_sigmas([0.99, 0.2, 0], sigma, "rescale")


class TestWanRefinementValidation(unittest.TestCase):
    def test_disabled_options_allow_other_models(self):
        helpers.validate_refinement_options(options(), "FluxPipelineConfig")

    def test_sigma_and_source_required_together(self):
        for kwargs in ({"wan_refine_sigma": 0.4}, {"wan_init_latent_path": "draft.pt"}):
            with self.assertRaisesRegex(ValueError, "set together"):
                helpers.validate_refinement_options(options(**kwargs))

    def test_only_base_wan_can_refine(self):
        params = options(wan_init_latent_path="draft.pt", wan_refine_sigma=0.4)
        helpers.validate_refinement_options(params, "WanT2V720PConfig")
        with self.assertRaisesRegex(ValueError, "only base Wan"):
            helpers.validate_refinement_options(params, "FastWan2_1_T2V_480P_Config")

    def test_export_rejects_unsupported_models(self):
        with self.assertRaisesRegex(ValueError, "Latent export"):
            helpers.validate_refinement_options(options(wan_save_latent_path="x.pt"), "Wan2_2_TI2V_5B_Config")

    def test_rejects_ambiguous_and_conflicting_experiments(self):
        for extra in (
            dict(num_outputs_per_prompt=2), dict(prompt=["a", "b"]),
            dict(prompt_path="prompts.txt"), dict(progressive_mode="dct"),
            dict(rollout=True), dict(wan_save_latent_path="draft.pt"),
            dict(wan_refine_sigma=float("nan")), dict(wan_refine_sigma=True),
        ):
            kwargs = dict(wan_init_latent_path="draft.pt", wan_refine_sigma=0.4)
            kwargs.update(extra)
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                helpers.validate_refinement_options(options(**kwargs))


@unittest.skipIf(torch is None, "PyTorch is required for latent file validation")
class TestWanLatentRoundTrip(unittest.TestCase):
    def test_roundtrip_and_mismatches(self):
        arch = SimpleNamespace(z_dim=2, spatial_compression_ratio=8,
                               temporal_compression_ratio=4,
                               latents_mean=[0, 0], latents_std=[1, 1])
        config = SimpleNamespace(vae_config=SimpleNamespace(arch_config=arch))
        batch = SimpleNamespace(height=32, width=32, num_frames=5, prompt="tram", seed=42)
        server = SimpleNamespace(pipeline_config=config, model_path="draft-model")
        clean = torch.randn(1, 2, 2, 4, 4)
        noise = torch.zeros_like(clean)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draft.pt"
            payload = helpers.make_latent_payload(clean, batch, server)
            torch.save(payload, path)
            torch.testing.assert_close(helpers.load_clean_latent(path, noise, batch, config), clean)
            batch.num_frames = 9
            with self.assertRaisesRegex(ValueError, "num_frames"):
                helpers.load_clean_latent(path, noise, batch, config)
            batch.num_frames = 5
            with self.assertRaisesRegex(ValueError, "shape"):
                helpers.load_clean_latent(path, noise[:, :, :1], batch, config)
            payload["latents"][0, 0, 0, 0, 0] = float("nan")
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "finite"):
                helpers.load_clean_latent(path, noise, batch, config)


@unittest.skipIf(
    torch is None or importlib.util.find_spec("diffusers") is None,
    "Native stage checks require the SGLang PyTorch/Diffusers environment",
)
class TestWanRefinementStage(unittest.TestCase):
    def test_solver_reset_noise_alignment_and_next_request(self):
        from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
            FlowUniPCMultistepScheduler,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_refinement import (
            WanRefinementPreparationStage,
        )

        arch = SimpleNamespace(
            z_dim=2, spatial_compression_ratio=8, temporal_compression_ratio=4,
            latents_mean=[0, 0], latents_std=[1, 1],
        )
        config = type("WanT2V720PConfig", (), {})()
        config.vae_config = SimpleNamespace(arch_config=arch)
        server = SimpleNamespace(pipeline_config=config, model_path="base-model")
        stage = WanRefinementPreparationStage.__new__(WanRefinementPreparationStage)
        stage.allow_refinement = True
        stage.log_info = lambda *args: None
        for mode in ("rescale", "suffix"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = str(Path(directory) / "draft.pt")
                params = options(wan_init_latent_path=path, wan_refine_sigma=0.4,
                                 wan_refine_schedule=mode)
                scheduler = FlowUniPCMultistepScheduler(shift=3)
                scheduler.set_timesteps(50, device="cpu")
                original_sigmas = scheduler.sigmas.clone()
                scheduler.model_outputs = [torch.ones(1)] * scheduler.config.solver_order
                scheduler.last_sample = torch.ones(1)
                scheduler.lower_order_nums = 2
                noise = torch.randn(1, 2, 2, 4, 4)
                clean = torch.ones_like(noise)
                batch = SimpleNamespace(
                    **vars(params), sampling_params=params, scheduler=scheduler,
                    num_inference_steps=50, latents=noise.clone(), is_warmup=False,
                    height=32, width=32, num_frames=5,
                )
                torch.save(helpers.make_latent_payload(clean, batch, server), path)
                expected = helpers.refinement_sigmas(original_sigmas.numpy(), 0.4, mode)
                stage.forward(batch, server)
                np.testing.assert_allclose(scheduler.sigmas[:-1].numpy(), expected)
                sigma = scheduler.sigmas[0]
                torch.testing.assert_close(batch.latents, (1 - sigma) * clean + sigma * noise)
                self.assertEqual(batch.num_inference_steps, len(expected))
                self.assertEqual(scheduler.begin_index, 0)
                self.assertEqual(scheduler.model_outputs, [None] * scheduler.config.solver_order)
                self.assertIsNone(scheduler.last_sample)
                self.assertEqual(scheduler.lower_order_nums, 0)
                # Refinement must not poison the reusable scheduler's flow shift.
                scheduler.set_timesteps(50, device="cpu")
                torch.testing.assert_close(scheduler.sigmas, original_sigmas)


if __name__ == "__main__":
    unittest.main()
