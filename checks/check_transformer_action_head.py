"""CPU checks of the production Transformer, exact JVP and actor wiring.

Only the heavyweight observation encoder is replaced in the integration check.
The actor class is loaded by the existing dependency-light AST harness; its
pooler, new decoder, objectives, gripper loss and solvers are the actual code.
"""

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.func import jvp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modeling.action_context import ActionContext
from modeling.flow_objectives import MeanFlowObjective
from modeling.transformer_action_head import TransformerActionHead
from checks.check_flow_objectives import Actor, poses
from utils.checkpoint_utils import load_model_state_strict
from utils.training_checkpoint import validate_init_config, validate_evaluation_config


def make_context(batch=3, grad=False):
    def random(*shape):
        return torch.randn(batch, *shape, requires_grad=grad)
    return ActionContext(
        global_condition=random(16), scene_tokens=random(7, 12),
        scene_xyz=random(7, 3), language_tokens=random(4, 12),
        language_padding_mask=torch.tensor([[False, False, True, True]]).expand(batch, -1),
        proprio_tokens=random(6, 12),
        workspace_bounds=torch.tensor([[-.4, -.8, .1], [.6, .7, 1.5]]).expand(batch, -1, -1),
    )


class TinyObservationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 16)
        self.calls = 0

    @staticmethod
    def instruction_padding_mask(instruction):
        mask = torch.zeros(instruction.shape[:2], dtype=torch.bool)
        mask[:, -1] = True
        return mask

    def forward(self, rgb3d, rgb2d, pcd, instruction, proprio):
        self.calls += 1
        dense = self.projection(rgb3d)
        language = self.projection(instruction)
        return (dense, pcd, None, None, language,
                proprio[:, -1:, :3].expand(-1, instruction.shape[1], -1),
                self.projection(proprio), dense[:, :2], pcd[:, :2])


class TransformerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(29)
        self.head = TransformerActionHead(12, hidden_dim=16, condition_dim=16,
                                          num_heads=4, num_blocks=2, nhand=2)
        self.context = make_context()
        self.z = torch.randn(3, 2, 2, 9)
        self.v = torch.randn_like(self.z)
        self.r = torch.tensor([.1, .2, .3])
        self.t = torch.tensor([.8, .7, .6])

    def field(self, z, r, t, context):
        return self.head(z, r, t, context)[-1][..., :9]

    def test_training_is_deterministic_without_consuming_rng(self):
        self.head.train()
        rng = torch.get_rng_state()
        first = self.head(self.z, self.r, self.t, self.context)[0]
        second = self.head(self.z, self.r, self.t, self.context)[0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.head.eval()
        torch.testing.assert_close(first, self.head(self.z, self.r, self.t, self.context)[0], rtol=0, atol=0)
        self.assertFalse(any(isinstance(module, nn.Dropout) for module in self.head.modules()))
        self.assertEqual(self.head.attention_backend, "math")
        math_head = TransformerActionHead(12, 16, 16, 4, 2, 2, "math")
        math_head.load_state_dict(self.head.state_dict())
        torch.testing.assert_close(first, math_head(self.z, self.r, self.t, self.context)[0], rtol=0, atol=0)

    def test_exact_jvp_matches_full_directional_difference(self):
        _, derivative = jvp(lambda z, r, t: self.field(z, r, t, self.context),
                            (self.z, self.r, self.t),
                            (self.v, torch.zeros_like(self.r), torch.ones_like(self.t)))
        eps = 1e-3
        finite_difference = (self.field(self.z+eps*self.v, self.r, self.t+eps, self.context)
                             - self.field(self.z-eps*self.v, self.r, self.t-eps, self.context)) / (2*eps)
        torch.testing.assert_close(derivative, finite_difference, atol=4e-4, rtol=5e-3)
        actual = MeanFlowObjective.target(self.field, self.z, self.r, self.t, self.v, self.context)
        expected = self.v + (self.r-self.t)[:, None, None, None]*derivative
        torch.testing.assert_close(actual, expected)
        self.assertFalse(actual.requires_grad)

    def test_context_chunking_fp32_and_diagonal(self):
        expected = MeanFlowObjective.target(self.field, self.z, self.r, self.t, self.v, self.context)
        for chunk in (1, 2, 0, 8):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = MeanFlowObjective.target(self.field, self.z, self.r, self.t, self.v, self.context, chunk)
                output = self.head(self.z, self.r, self.t, self.context)[0]
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(output.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=1e-5)
            torch.testing.assert_close(output, self.head(self.z, self.r, self.t, self.context)[0], rtol=0, atol=0)
        diagonal = MeanFlowObjective.target(self.field, self.z, self.t, self.t, self.v, self.context)
        torch.testing.assert_close(diagonal, self.v, rtol=0, atol=0)

    def test_geometry_keeps_candidate_derivative_and_world_units(self):
        # Disable xyz input projection: any xyz JVP now comes ONLY through
        # differentiable query-to-scene geometry, not the pose feature path.
        with torch.no_grad():
            self.head.input_projection.weight[:, :3].zero_()
        tangent = torch.zeros_like(self.z)
        tangent[..., :3] = self.v[..., :3]
        _, derivative = jvp(lambda z: self.field(z, self.r, self.t, self.context), (self.z,), (tangent,))
        self.assertGreater(derivative.abs().max().item(), 1e-7)
        eps = 1e-3
        finite_difference = (self.field(self.z+eps*tangent, self.r, self.t, self.context)
                             - self.field(self.z-eps*tangent, self.r, self.t, self.context)) / (2*eps)
        torch.testing.assert_close(derivative, finite_difference, atol=3e-4, rtol=1e-2)
        shift = torch.tensor([2., -3., .7])
        translated = replace(self.context, scene_xyz=self.context.scene_xyz+shift,
                             workspace_bounds=self.context.workspace_bounds+shift)
        torch.testing.assert_close(self.field(self.z, self.r, self.t, translated),
                                   self.field(self.z, self.r, self.t, self.context), atol=3e-7, rtol=2e-6)

    def test_masks_and_all_padded_language_are_finite_and_invariant(self):
        context = make_context(grad=True)
        mask = context.language_padding_mask.clone()
        mask[1] = True
        context = replace(context, language_padding_mask=mask)
        changed = replace(context, language_tokens=context.language_tokens.detach().masked_fill(mask[..., None], 1e4))
        output = self.field(self.z, self.r, self.t, context)
        torch.testing.assert_close(output, self.field(self.z, self.r, self.t, changed), rtol=0, atol=0)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(context.language_tokens.grad[mask].abs().sum().item(), 0.)
        self.assertGreater(context.language_tokens.grad[~mask].abs().sum().item(), 0.)
        self.assertEqual(context.detach().float()[1:2].language_padding_mask.dtype, torch.bool)
        empty = replace(context, language_tokens=context.language_tokens[:, :0], language_padding_mask=mask[:, :0])
        self.assertTrue(torch.isfinite(self.field(self.z, self.r, self.t, empty)).all())

    def test_tokens_have_direct_gradient_routes_and_target_is_detached(self):
        context = make_context(grad=True)
        target = MeanFlowObjective.target(self.field, self.z, self.r, self.t, self.v, context, 2)
        self.assertFalse(target.requires_grad)
        prediction = self.field(self.z, self.r, self.t, context)
        (prediction-target).square().mean().backward()
        for tensor in (context.scene_tokens, context.scene_xyz, context.language_tokens,
                       context.proprio_tokens, context.global_condition):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0.)

    def test_ivc_context_mask_preserves_selected_encoder_gradients(self):
        context = make_context(grad=True)
        r = self.r.clone()
        r[1] = self.t[1]
        loss = MeanFlowObjective.ivc_loss(self.field, self.z, r, self.t, self.v, context)
        loss.backward()
        self.assertEqual(context.scene_tokens.grad[1].abs().sum().item(), 0.)
        self.assertGreater(context.scene_tokens.grad[[0, 2]].abs().sum().item(), 0.)

    def test_single_and_multi_step_single_and_bimanual_shapes(self):
        for hands in (1, 2):
            head = TransformerActionHead(12, 16, 16, 4, 1, hands)
            for steps in (1, 5):
                z = torch.randn(3, steps, hands, 9)
                out = head(z, self.r, self.t, self.context)[0]
                self.assertEqual(out.shape, (3, steps, hands, 10))
                target = MeanFlowObjective.target(lambda z, r, t, c: head(z, r, t, c)[0][..., :9],
                                                  z, self.r, self.t, torch.randn_like(z), self.context, 2)
                self.assertTrue(torch.isfinite(target).all())

    def test_actor_training_endpoint_inference_and_encoder_once(self):
        for objective in ("fm", "meanflow", "imf"):
            for gripper in ("direct", "legacy_denoise"):
                with self.subTest(objective=objective, gripper=gripper):
                    actor = Actor(embedding_dim=16, num_attn_heads=4, nhist=3, nhand=2,
                                  action_hidden_dim=16, action_num_blocks=2, action_head="transformer",
                                  flow_objective=objective, attention_backend="math",
                                  lv2_batch_size=2, jvp_microbatch_size=2, endpoint_loss_weight=.25,
                                  ivc_loss_weight=.5 if objective != "fm" else 0.,
                                  meanflow_offdiag_ratio=1. if objective != "fm" else 0.,
                                  gripper_prediction_mode=gripper)
                    actor.encoder = TinyObservationEncoder()
                    rgb, pcd, instruction, proprio = torch.randn(3, 7, 8), torch.randn(3, 7, 3), torch.randn(3, 4, 8), poses(steps=3)
                    pool_calls = []
                    handle = actor.condition_pooler.register_forward_hook(lambda *args: pool_calls.append(1))
                    with patch("modeling.flow_objectives.jvp", wraps=jvp) as counted:
                        with torch.autocast("cpu", dtype=torch.bfloat16):
                            loss = actor.compute_loss(poses(), rgb, None, pcd, instruction, proprio)
                        loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertEqual(actor.encoder.calls, 1)
                    self.assertEqual(len(pool_calls), 1)
                    self.assertEqual(counted.call_count, 4 if objective != "fm" else 0)
                    for module in (actor.encoder, actor.condition_pooler, actor.prediction_head):
                        gradients = [p.grad for p in module.parameters() if p.grad is not None]
                        self.assertTrue(gradients)
                        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)
                    actor.eval()
                    for steps in (1, 2, 5):
                        actor.n_steps = steps
                        with torch.no_grad():
                            output = actor.compute_trajectory(torch.ones(3, 2, 2, dtype=torch.bool),
                                                              rgb, None, pcd, instruction, proprio)
                        self.assertEqual(output.shape, (3, 2, 2, 8))
                        self.assertTrue(torch.isfinite(output).all())
                    handle.remove()

    def test_invalid_head_config_is_rejected(self):
        for kwargs in ({"hidden_dim": 15}, {"num_heads": 3}, {"num_heads": 0},
                       {"num_blocks": 0}, {"attention_backend": "flash"}):
            with self.assertRaises(ValueError):
                TransformerActionHead(12, **kwargs)

    def test_checkpoint_transfer_requires_same_architecture(self):
        tcn = Actor(embedding_dim=12, action_hidden_dim=16, action_num_blocks=2, nhand=2)
        before = {key: value.clone() for key, value in self.head.state_dict().items()}
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            load_model_state_strict(self.head, tcn.prediction_head.state_dict())
        for key, value in self.head.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "architecture mismatch"):
            validate_init_config({"config": {"action_head": "film_tcn"}}, {"action_head": "transformer"})
        source = {"config": {"action_head": "transformer", "flow_objective": "fm"}, "weight": before}
        destination = {"action_head": "transformer", "flow_objective": "meanflow"}
        validate_init_config(source, destination)
        load_model_state_strict(self.head, source)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_evaluation_config(source, destination)

    def test_script_width_and_depth_forward_ad_smoke(self):
        # Production head dimensions, with reduced observation count/batch.
        # This is NOT a full-sized scene or a GPU throughput measurement.
        head = TransformerActionHead(120, nhand=1)
        context = ActionContext(
            torch.randn(1, 256), torch.randn(1, 24, 120), torch.randn(1, 24, 3),
            torch.randn(1, 8, 120), None, torch.randn(1, 3, 120),
            torch.tensor([[[0., 0., 0.], [1., 1., 1.]]]),
        )
        z = torch.randn(1, 1, 1, 9)
        output, derivative = jvp(lambda z, t: head(z, t*.0, t, context)[0],
                                 (z, torch.tensor([.7])), (torch.randn_like(z), torch.ones(1)))
        self.assertEqual(output.shape, (1, 1, 1, 10))
        self.assertTrue(torch.isfinite(derivative).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
