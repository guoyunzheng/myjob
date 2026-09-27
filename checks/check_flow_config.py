"""CPU configuration checks; no CLIP, RLBench or diffusion dependencies needed."""

from contextlib import redirect_stderr
from dataclasses import asdict
import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import parse_arguments as parse_train
from online_evaluation_rlbench.evaluate_policy import parse_arguments as parse_eval
from modeling.flow_config import FlowConfig, flow_model_kwargs, resolve_flow_config, configure_action_precision
from modeling.noise_scheduler import fetch_schedulers
from modeling.noise_scheduler.meanflow import MFScheduler


class FlowConfigChecks(unittest.TestCase):
    def test_cli_defaults_and_legacy_alias(self):
        for parse in (parse_train, parse_eval):
            for argv in ([], ["--denoise_model", "meanflow"]):
                args = parse(argv)
                self.assertEqual(flow_model_kwargs(args), asdict(FlowConfig()))
                self.assertEqual(args.denoise_model, "meanflow")

    def test_explicit_config_reaches_both_schedulers(self):
        argv = ["--action_head", "film_tcn", "--flow_objective", "meanflow",
                "--attention_backend", "math", "--time_sampler", "uniform",
                "--time_sampler_mean", "-0.4", "--time_sampler_std", "1.0",
                "--meanflow_offdiag_ratio", "0.6", "--flow_loss_type", "l1"]
        for parse in (parse_train, parse_eval):
            args = parse(argv)
            config = resolve_flow_config(**flow_model_kwargs(args))
            self.assertEqual(config.attention_backend, "math")
            for scheduler in fetch_schedulers("meanflow", 2, flow_config=config):
                self.assertEqual(scheduler.noise_sampler, "uniform")
                self.assertEqual(scheduler.meanflow_r_ne_t_ratio, 0.6)
                self.assertEqual(scheduler.noise_sampler_config, {"mean": -0.4, "std": 1.0})

    def test_cli_rejects_unimplemented_or_incompatible_config(self):
        cases = [
            ["--action_head", "transformer", "--action_hidden_dim", "15"],
            ["--action_head", "transformer", "--num_attn_heads", "0"],
            ["--action_head", "transformer", "--action_num_blocks", "0"],
            ["--flow_objective", "imf", "--guidance_scale", "2"],
            ["--flow_objective", "fm", "--denoise_model", "meanflow"],
            ["--flow_objective", "fm", "--meanflow_offdiag_ratio", "0.25"],
            ["--denoise_model", "rectified_flow"],
            ["--denoise_model", "ddpm", "--flow_objective", "meanflow"],
            ["--action_head", "typo"], ["--attention_backend", "flash"],
            ["--flow_loss_type", "mse"], ["--time_sampler", "pi0"],
            ["--time_sampler_std", "0"], ["--time_sampler_std", "nan"],
            ["--time_sampler_mean", "inf"], ["--meanflow_offdiag_ratio", "1.1"],
            ["--meanflow_offdiag_ratio", "nan"],
            ["--model_type", "denoise2d", "--action_head", "film_tcn"],
            ["--model_type", "denoise2d", "--denoise_model", "fm"],
        ]
        for parse in (parse_train, parse_eval):
            for argv in cases:
                with self.subTest(parse=parse.__module__, argv=argv):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        parse(argv)
                    self.assertEqual(error.exception.code, 2)

    def test_legacy_2d_scheduler_flag_is_unchanged(self):
        for parse in (parse_train, parse_eval):
            args = parse(["--model_type", "denoise2d", "--denoise_model", "ddpm"])
            self.assertEqual(args.denoise_model, "ddpm")

    def test_programmatic_calls_cannot_bypass_validation(self):
        for config in ({"flow_objective": "unknown"}, {"meanflow_offdiag_ratio": -0.1}):
            with self.assertRaises(ValueError):
                resolve_flow_config(**config)
        with self.assertRaises(ValueError):
            fetch_schedulers("ddpm", 2, flow_config=FlowConfig())

    def test_fm_configuration_and_scheduler_are_live(self):
        for parse in (parse_train, parse_eval):
            args = parse(["--flow_objective", "fm"])
            self.assertEqual(args.denoise_model, "fm")
            self.assertEqual(args.meanflow_offdiag_ratio, 0.)
            config = resolve_flow_config(**flow_model_kwargs(args))
            for scheduler in fetch_schedulers("fm", 2, flow_config=config):
                t, r = scheduler.sample_noise_step(20, "cpu")
                self.assertTrue(torch.equal(t, r))
            self.assertEqual(parse(["--denoise_model", "fm"]).flow_objective, "fm")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_train(["--flow_objective", "fm", "--ivc_loss_weight", "0.5"])

    def test_transformer_configuration_for_both_objectives(self):
        for parse in (parse_train, parse_eval):
            for objective in ("fm", "meanflow", "imf"):
                for backend in ("auto", "math"):
                    args = parse(["--action_head", "transformer", "--flow_objective", objective,
                                  "--attention_backend", backend, "--num_attn_heads", "8"])
                    config = resolve_flow_config(**flow_model_kwargs(args))
                    self.assertEqual(config.action_head, "transformer")
                    self.assertEqual(config.flow_objective, objective)

    def test_transformer_precision_disables_tf32_without_changing_tcn_default(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            configure_action_precision("film_tcn")
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            configure_action_precision("transformer")
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    def test_default_sampler_matches_previous_algorithm_exactly(self):
        # Original algorithm, including RNG ordering and diagonal mask.
        torch.manual_seed(42)
        r_samples = torch.empty(2048, dtype=torch.float32).normal_(0.0, 1.5).sigmoid()
        t_samples = torch.empty_like(r_samples).normal_(0.0, 1.5).sigmoid()
        expected_t = torch.maximum(r_samples, t_samples)
        expected_r = torch.minimum(r_samples, t_samples)
        mask = torch.rand(2048) < 0.25
        expected_r = torch.where(mask, expected_r, expected_t)
        expected_rng = torch.get_rng_state()
        for config in (None, FlowConfig()):
            scheduler = fetch_schedulers("meanflow", 2, flow_config=config)[0]
            torch.manual_seed(42)
            t, r = scheduler.sample_noise_step(2048, "cpu")
            self.assertTrue(torch.equal(t, expected_t))
            self.assertTrue(torch.equal(r, expected_r))
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_sampler_ratio_endpoints_and_distribution(self):
        for sampler in ("uniform", "logit_normal"):
            for ratio in (0.0, 0.25, 1.0):
                torch.manual_seed(7)
                scheduler = MFScheduler(sampler, meanflow_r_ne_t_ratio=ratio)
                t, r = scheduler.sample_noise_step(20000, "cpu")
                self.assertTrue(torch.all((r >= 0) & (r <= t) & (t <= 1)))
                actual = (r < t).float().mean().item()
                self.assertAlmostEqual(actual, ratio, delta=0.015)
                if sampler == "uniform":
                    # t is max of two uniform draws, NOT a uniform marginal.
                    self.assertAlmostEqual(t.mean().item(), 2 / 3, delta=0.015)

    def test_logit_normal_parameters_are_used(self):
        torch.manual_seed(5)
        scheduler = MFScheduler("logit_normal", {"mean": -2.0, "std": 0.1})
        t, _ = scheduler.sample_noise_step(1000, "cpu")
        self.assertLess(t.mean().item(), 0.2)

    def test_scheduler_rejects_invalid_config(self):
        for kwargs in ({"noise_sampler": "typo"}, {"meanflow_r_ne_t_ratio": float("nan")},
                       {"noise_sampler_config": {"std": -1}},
                       {"noise_sampler_config": {"mean": float("inf")}}):
            with self.assertRaises(ValueError):
                MFScheduler(**kwargs)

    def test_integration_and_interpolation_unchanged(self):
        scheduler = fetch_schedulers("meanflow", 2, flow_config=FlowConfig())[0]
        data, noise = torch.zeros(2, 1, 1, 9), torch.ones(2, 1, 1, 9)
        z = scheduler.add_noise(data, noise, torch.tensor([0.25, 0.75]))
        torch.testing.assert_close(z[:, 0, 0, 0], torch.tensor([0.25, 0.75]))
        scheduler.set_timesteps(2)
        z = noise
        for t, r in zip(scheduler.timesteps, scheduler.prev_timesteps):
            z = scheduler.step(noise - data, t, r, z).prev_sample
        torch.testing.assert_close(z, data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
