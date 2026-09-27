"""CPU analytic and production-head checks for opt-in boundary-reuse iMF."""

from contextlib import redirect_stderr
from dataclasses import replace
import io
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import MethodType
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.func import jvp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import parse_arguments as parse_train
from online_evaluation_rlbench.evaluate_policy import parse_arguments as parse_eval
from modeling.flow_config import FlowConfig, resolve_flow_config
from modeling.flow_objectives import ImprovedMeanFlowObjective as IMF, MeanFlowObjective
from modeling.noise_scheduler import fetch_schedulers
from checks.check_flow_objectives import Actor, make_actor, poses
from checks.check_transformer_action_head import make_context
from modeling.transformer_action_head import TransformerActionHead
from utils.checkpoint_utils import load_model_state_strict
from utils.training_checkpoint import validate_init_config, validate_evaluation_config, validate_resume
from checks.check_training_resume import components, sample_checkpoint


class ImprovedMeanFlowChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(91)
        self.z = torch.randn(5, 2, 2, 9)
        self.r = torch.tensor([.1, .7, .2, .3, .8])
        self.t = torch.tensor([.8, .7, .6, .9, .8])
        self.v = torch.randn_like(self.z, requires_grad=True)
        self.condition = torch.randn(5, 3, requires_grad=True)

    @staticmethod
    def field(z, r, t, condition):
        r4, t4 = r[:, None, None, None], t[:, None, None, None]
        return .3*z.square() + .4*t4.square() + 1.3*r4 + .2*(t4-r4).square() + condition[:, :1, None, None]

    def correction(self, chunk=0):
        return IMF.prediction_correction(self.field, self.z, self.r, self.t, self.condition, chunk)

    def test_closed_form_uses_predicted_boundary_tangent_and_fixed_r(self):
        tangent = self.field(self.z, self.t, self.t, self.condition)
        derivative = .6*self.z*tangent + (.8*self.t + .4*(self.t-self.r))[:, None, None, None]
        expected = (self.t-self.r)[:, None, None, None] * derivative
        torch.testing.assert_close(self.correction(2), expected)
        partial_only = (self.t-self.r)[:, None, None, None] * (.8*self.t+.4*(self.t-self.r))[:, None, None, None]
        self.assertFalse(torch.allclose(expected, partial_only))
        old_target = MeanFlowObjective.target(self.field, self.z, self.r, self.t, self.v, self.condition)
        self.assertFalse(torch.allclose(self.v-self.correction(), old_target))

    def test_full_directional_finite_difference_with_fixed_tangent(self):
        tangent = self.field(self.z, self.t, self.t, self.condition).detach()
        eps = 1e-3
        plus = self.field(self.z+eps*tangent, self.r, self.t+eps, self.condition)
        minus = self.field(self.z-eps*tangent, self.r, self.t-eps, self.condition)
        expected = (self.t-self.r)[:, None, None, None] * (plus-minus)/(2*eps)
        torch.testing.assert_close(self.correction(), expected, atol=3e-4, rtol=2e-3)

    def test_target_is_fixed_velocity_and_never_calls_field(self):
        target = IMF.target(lambda *args: self.fail("iMF target called field"),
                            self.z, self.r, self.t, self.v, self.condition, 2)
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(target, self.v, rtol=0, atol=0)
        # Correction has no velocity input. Changing the regression target
        # cannot leak the sampled noise-data into the prediction.
        expected = self.correction()
        with torch.no_grad():
            self.v.add_(100.)
        torch.testing.assert_close(self.correction(), expected, rtol=0, atol=0)

    def test_diagonal_is_exact_fm_without_any_extra_forward(self):
        with patch("modeling.flow_objectives.jvp", side_effect=AssertionError("diagonal JVP")):
            result = IMF.prediction_correction(lambda *args: self.fail("diagonal forward"),
                                               self.z, self.t, self.t, self.condition, 1)
        torch.testing.assert_close(result, torch.zeros_like(self.z), rtol=0, atol=0)
        self.assertFalse(result.requires_grad)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            IMF.prediction_correction(self.field, self.z[:0], self.r[:0], self.t[:0], self.condition[:0])
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, "forward AD is disabled"):
            self.correction()

    def test_chunking_autocast_no_gradients_and_rng_preservation(self):
        expected = self.correction()
        rng = torch.get_rng_state()
        for chunk in (None, 1, 2, 9):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = self.correction(chunk)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertFalse(actual.requires_grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIsNone(self.condition.grad)
        self.assertIsNone(self.v.grad)

    def test_only_offdiagonal_rows_enter_jvp_and_boundary_forward(self):
        calls = []
        def field(z, r, t, condition):
            calls.append((len(z), r.detach().clone(), t.detach().clone()))
            return self.field(z, r, t, condition)
        with patch("modeling.flow_objectives.jvp", wraps=jvp) as counted:
            IMF.prediction_correction(field, self.z, self.r, self.t, self.condition, 2)
        self.assertEqual(counted.call_count, 2)  # 3 off-diagonal samples in chunks 2+1
        self.assertEqual([item[0] for item in calls], [2, 2, 1, 1])
        for _, r, t in calls[::2]:
            self.assertTrue(torch.equal(r, t))

    def test_semigradient_matches_manual_detached_correction(self):
        parameter = torch.tensor(.4, requires_grad=True)
        def field(z, r, t, condition):
            return parameter*z.square() + condition[:, :1, None, None] + t[:, None, None, None]
        correction = IMF.prediction_correction(field, self.z, self.r, self.t, self.condition, 2)
        self.assertFalse(correction.requires_grad)
        prediction = field(self.z, self.r, self.t, self.condition)
        residual = prediction + correction - self.v.detach()
        residual.square().mean().backward()
        expected = (2 * residual.detach() * self.z.square()).mean()
        torch.testing.assert_close(parameter.grad, expected)
        self.assertGreater(self.condition.grad.abs().sum().item(), 0.)
        self.assertIsNone(self.v.grad)

    def test_actual_transformer_context_gradients_and_chunk_equivalence(self):
        head = TransformerActionHead(12, 16, 16, 4, 2, 2)
        context = make_context(grad=True)
        z, r, t = self.z[:3], self.r[:3], self.t[:3]
        field = lambda z,r,t,c: head(z, r, t, c)[-1][..., :9]
        expected = IMF.prediction_correction(field, z, r, t, context)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = IMF.prediction_correction(field, z, r, t, context, 1)
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
        self.assertFalse(actual.requires_grad)
        (field(z,r,t,context)+actual-self.v[:3].detach()).square().mean().backward()
        for tensor in (context.scene_tokens, context.scene_xyz, context.language_tokens,
                       context.proprio_tokens, context.global_condition):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0.)

    def test_actor_regression_uses_compound_prediction_with_l1_and_l2(self):
        for objective in ("fm", "meanflow", "imf"):
            for metric in ("l1", "l2"):
                actor = make_actor(objective)
                actor.flow_config = replace(actor.flow_config, flow_loss_type=metric)
                actor._lv2_batch_size = 1
                observations = []
                original = actor.policy_forward_pass
                def capture(z, r, t, c):
                    result = original(z, r, t, c)
                    if torch.is_grad_enabled():
                        observations.append(result[-1][..., :9])
                    return result
                v = self.v[:3].detach()
                with patch.object(actor, "policy_forward_pass", side_effect=capture), \
                     patch.object(actor, "compute_flow_target", return_value=v), \
                     patch.object(IMF, "prediction_correction", return_value=torch.full_like(v, 2.)):
                    loss = actor.compute_loss(poses(), None, None, None, None, poses(steps=3))
                prediction = observations[0] + (2. if objective == "imf" else 0.)
                function = F.l1_loss if metric == "l1" else F.mse_loss
                expected = 30*function(prediction[..., :3], v[..., :3]) + 10*function(prediction[..., 3:], v[..., 3:])
                expected = expected + actor.loss_diagnostics["gripper"]
                torch.testing.assert_close(loss, expected)
                loss.backward()
                for module in (actor.encoder, actor.condition_pooler, actor.prediction_head):
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))

    def test_actual_tcn_correction_chunking_and_training_gradients(self):
        actor = make_actor("imf", endpoint=.1, ivc=.2)
        condition = torch.randn(5, 16)
        expected = IMF.prediction_correction(actor.pose_velocity_field, self.z, self.r, self.t, condition)
        actual = IMF.prediction_correction(actor.pose_velocity_field, self.z, self.r, self.t, condition, 2)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = actor.compute_loss(poses(), None, None, None, None, poses(steps=3))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(loss, sum(actor.loss_diagnostics.values()))
        self.assertEqual(set(actor.flow_diagnostics), {"imf_correction_rms", "imf_offdiag_fraction"})
        for module in (actor.encoder, actor.condition_pooler, actor.prediction_head, actor.gripper_state_head):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)

    def test_inference_integrates_u_without_boundary_prediction_or_jvp(self):
        actor = make_actor("imf")
        for steps in (1, 2, 5):
            actor.n_steps = steps
            calls = []
            def field(self, z, r, t, c):
                calls.append((r.clone(), t.clone()))
                return [torch.cat((torch.ones_like(z), z.new_zeros(*z.shape[:-1], 1)), -1)]
            actor.policy_forward_pass = MethodType(field, actor)
            with patch("modeling.flow_objectives.jvp", side_effect=AssertionError("inference JVP")):
                result, _ = actor.denoise_trajectory(torch.ones(3, 2, 2, 9), torch.zeros(3,16))
            torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-6, rtol=0)
            self.assertEqual(len(calls), steps)
            self.assertTrue(all(torch.all(r < t) for r,t in calls))

    def test_cli_defaults_overrides_and_scheduler_match_meanflow(self):
        for parse in (parse_train, parse_eval):
            args = parse(["--flow_objective", "imf"])
            self.assertEqual((args.flow_loss_type, args.meanflow_offdiag_ratio, args.denoise_model), ("l2", .25, "imf"))
            self.assertEqual(parse(["--flow_objective", "imf", "--flow_loss_type", "l1"]).flow_loss_type, "l1")
            self.assertEqual(parse(["--denoise_model", "imf"]).flow_objective, "imf")
            # Both canonical fields are forwarded by the real train/eval
            # model factories, not just the explicit objective used in tests.
            actor = Actor(embedding_dim=16, action_hidden_dim=16, action_num_blocks=1,
                          flow_objective=args.flow_objective, denoise_model=args.denoise_model)
            self.assertEqual(actor.flow_objective.name, "imf")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse(["--model_type", "denoise2d", "--denoise_model", "imf"])
        for objective in ("fm", "meanflow"):
            self.assertEqual(resolve_flow_config(flow_objective=objective).flow_loss_type, "l1")
        for sampler in ("uniform", "logit_normal"):
            expected = None
            for objective in ("meanflow", "imf"):
                config = FlowConfig(flow_objective=objective, time_sampler=sampler)
                torch.manual_seed(21)
                pair = fetch_schedulers(objective, 2, flow_config=config)[0].sample_noise_step(100, "cpu")
                if expected is not None:
                    for a,b in zip(pair, expected):
                        torch.testing.assert_close(a,b, rtol=0, atol=0)
                expected = pair

    def test_guidance_and_dropout_rejected_at_all_public_boundaries(self):
        for parse in (parse_train, parse_eval):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse(["--flow_objective", "imf", "--guidance_scale", "2"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_train(["--flow_objective", "imf", "--condition_dropout_prob", ".1"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_train(["--flow_objective", "imf", "--use_compile", "true"])
        for kwargs in ({"guidance_scale": 2.}, {"condition_dropout_prob": .1}):
            with self.assertRaisesRegex(ValueError, "flexible CFG"):
                Actor(flow_objective="imf", **kwargs)
        with self.assertRaisesRegex(ValueError, "flexible CFG"):
            make_actor("imf").denoise_trajectory(self.z, torch.randn(5,16), guidance_scale=2.)

    def test_checkpoint_weight_transfer_is_explicit_and_parameter_shapes_unchanged(self):
        old = make_actor("meanflow")
        new = make_actor("imf")
        load_model_state_strict(new, old.state_dict())
        for key, value in old.state_dict().items():
            torch.testing.assert_close(new.state_dict()[key], value, rtol=0, atol=0)
        with patch("torch.cuda.is_available", return_value=False):
            checkpoint = sample_checkpoint(components())
        config = dict(checkpoint["config"], flow_objective="imf", denoise_model="imf", flow_loss_type="l2")
        validate_init_config(checkpoint, config)
        with self.assertRaisesRegex(ValueError, "config mismatch"):
            validate_resume(checkpoint, config, 1)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_evaluation_config(checkpoint, config)
        same_objective_new_metric = dict(checkpoint["config"], flow_loss_type="l2")
        with self.assertRaisesRegex(ValueError, "config mismatch"):
            validate_resume(checkpoint, same_objective_new_metric, 1)

    def test_bash_recipes_round_trip_to_parser_without_launching_training(self):
        bash = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
        if not Path(bash).is_file():
            self.skipTest("Bash unavailable")
        root = Path(__file__).resolve().parents[1]
        clean_env = dict(os.environ)
        for key in ("ENDPOINT_LOSS_WEIGHT", "IVC_LOSS_WEIGHT", "FLOW_LOSS_TYPE", "RESUME", "INIT_FROM",
                    "MEANFLOW_OFFDIAG_RATIO", "GRIPPER_TRANSITION_WEIGHT", "GRIPPER_CLOSED_HOLD_WEIGHT"):
            clean_env.pop(key, None)
        clean_env.update(PYTHON_BIN="echo", SEED="17", RUN_LOG_DIR="step9_test_preview", GRIPPER_LOSS_TYPE="bce")
        for objective, metric, endpoint, ivc in (("fm", "l1", .25, 0.), ("meanflow", "l1", .25, .5),
                                                ("imf", "l2", 0., 0.)):
            environment = dict(clean_env, FLOW_OBJECTIVE=objective)
            output = subprocess.check_output([bash, "train.sh"], cwd=root, env=environment, text=True)
            tokens = shlex.split(output)
            args = parse_train(tokens[tokens.index("main.py")+1:])
            self.assertEqual((args.flow_objective, args.flow_loss_type, args.endpoint_loss_weight,
                              args.ivc_loss_weight, args.seed), (objective, metric, endpoint, ivc, 17))
        output = subprocess.check_output([bash, "train.sh"], cwd=root,
                                          env=dict(clean_env, FLOW_OBJECTIVE="imf", FLOW_LOSS_TYPE="l1",
                                                   ENDPOINT_LOSS_WEIGHT=".1", IVC_LOSS_WEIGHT=".2"), text=True)
        tokens = shlex.split(output)
        args = parse_train(tokens[tokens.index("main.py")+1:])
        self.assertEqual((args.flow_loss_type, args.endpoint_loss_weight, args.ivc_loss_weight), ("l1", .1, .2))

    def test_systemd_wrapper_defers_objective_defaults_without_starting_service(self):
        bash = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
        if not Path(bash).is_file():
            self.skipTest("Bash unavailable")
        environment = dict(os.environ, FLOW_OBJECTIVE="imf")
        for key in ("GYZ_LIMITED_CHILD", "ENDPOINT_LOSS_WEIGHT", "FLOW_LOSS_TYPE"):
            environment.pop(key, None)
        # Intercept exec itself: Bash's exec builtin bypasses shell functions
        # named systemd-run. Exit immediately after printing the command so
        # neither the service nor the child training branch can run.
        command = ('systemctl() { return 1; }\n'
                   'exec() { printf "%s\\n" "$@"; exit 0; }\n'
                   'export -f systemctl exec\n'
                   'bash train_systemd_limited.sh')
        output = subprocess.check_output([bash, "-c", command], cwd=Path(__file__).resolve().parents[1],
                                          env=environment, text=True)
        lines = output.splitlines()
        self.assertIn("FLOW_OBJECTIVE=imf", lines)
        self.assertIn("FLOW_LOSS_TYPE=", lines)
        self.assertIn("ENDPOINT_LOSS_WEIGHT=", lines)


if __name__ == "__main__":
    unittest.main(verbosity=2)
