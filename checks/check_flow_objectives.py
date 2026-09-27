"""Analytic, derivative and actor-integration checks for FM / MeanFlow on CPU."""

import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import jvp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.flow_config import resolve_flow_config
from modeling.action_context import ActionContext
from modeling.transformer_action_head import TransformerActionHead
from modeling.loss_config import LossConfig
from modeling.flow_objectives import (
    FlowMatchingObjective, MeanFlowObjective, fetch_flow_objective, interval_coordinates,
)
from modeling.noise_scheduler import fetch_schedulers
from utils.action_contract import current_proprio, validate_action_config, validate_proprio_shape


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rearrange_layout(tensor, pattern, **dims):
    # The only two einops expressions in the tested TCN; no arithmetic is
    # replaced. This permits real head/JVP tests without installing einops/CLIP.
    if pattern == 'b l h c -> b c (l h)':
        return tensor.permute(0, 3, 1, 2).flatten(2)
    if pattern == 'b c (l h) -> b l h c':
        return tensor.transpose(1, 2).reshape(tensor.shape[0], dims['l'], dims['h'], tensor.shape[1])
    raise AssertionError(f"Unexpected layout expression: {pattern}")


namespace = dict(vars(load_file("flow_rotation_utils", "modeling/utils/utils.py")))
namespace.update(
    nn=nn, F=F, torch=torch, jvp=jvp, einops=SimpleNamespace(rearrange=rearrange_layout),
    fetch_schedulers=fetch_schedulers, resolve_flow_config=resolve_flow_config,
    fetch_flow_objective=fetch_flow_objective, interval_coordinates=interval_coordinates,
    ActionContext=ActionContext, TransformerActionHead=TransformerActionHead,
    LossConfig=LossConfig,
    current_proprio=current_proprio, validate_action_config=validate_action_config,
    validate_proprio_shape=validate_proprio_shape,
    SinusoidalPosEmb=load_file("flow_position_utils", "modeling/utils/position_encodings.py").SinusoidalPosEmb,
    GripperStateHead=load_file("flow_gripper_head", "modeling/policy/gripper_head.py").GripperStateHead,
)
actor_path = ROOT / "modeling/policy/base_denoise_actor.py"
tree = ast.parse(actor_path.read_text(encoding="utf-8"))
tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
exec(compile(tree, str(actor_path), "exec"), namespace)
Actor = namespace["DenoiseActor"]


def make_actor(objective, endpoint=0., ivc=0.):
    actor = Actor(embedding_dim=16, nhist=3, nhand=2, action_hidden_dim=16,
                  action_num_blocks=2, jvp_microbatch_size=2, denoise_timesteps=2,
                  flow_objective=objective, endpoint_loss_weight=endpoint,
                  ivc_loss_weight=ivc, gripper_prediction_mode="direct")
    # Replace only observation encoding/pooling with trainable lightweight modules.
    actor.encoder = nn.Linear(8, 16)
    actor.condition_pooler = nn.Linear(16, 16)
    def encode(self, rgb3d, rgb2d, pcd, instruction, proprio):
        return current_proprio(proprio), self.encoder(proprio.mean(dim=(1, 2)))
    actor.encode_inputs = MethodType(encode, actor)
    actor.encode_condition = MethodType(lambda self, fixed: self.condition_pooler(fixed[1]), actor)
    # Residual branches are zero-initialized in production. Make them nonzero
    # here so gradient-to-condition checks exercise a trained-like network.
    for block in actor.prediction_head.blocks:
        nn.init.normal_(block.conv2.weight, std=0.03)
    actor.collect_training_diagnostics = True
    return actor


def poses(batch=3, steps=2):
    data = torch.rand(batch, steps, 2, 8)
    data[..., 3:7] = F.normalize(data[..., 3:7], dim=-1)
    data[..., 7] = (data[..., 7] > 0.5).float()
    return data


class FlowObjectiveChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(15)
        self.z = torch.randn(7, 2, 2, 9)
        self.v = torch.randn_like(self.z)
        self.r = torch.linspace(.05, .3, 7)
        self.t = self.r + .5
        self.condition = torch.randn(7, 3, requires_grad=True)

    @staticmethod
    def analytic_field(z, r, t, condition):
        t4, r4 = t[:, None, None, None], r[:, None, None, None]
        return .7*z.square() + .4*t4.square() + 1.3*r4 + .2*(t4-r4).square() + condition[:, :1, None, None]

    def test_meanflow_total_derivative_matches_closed_form(self):
        target = MeanFlowObjective.target(self.analytic_field, self.z, self.r, self.t, self.v, self.condition, 3)
        derivative = 1.4*self.z*self.v + (.8*self.t + .4*(self.t-self.r))[:, None, None, None]
        expected = self.v + (self.r-self.t)[:, None, None, None]*derivative
        torch.testing.assert_close(target, expected)
        self.assertFalse(target.requires_grad)
        self.assertIsNone(self.condition.grad)
        # Omitting D_z u @ v (the archived finite-difference bug) must fail.
        partial_only = self.v + (self.r-self.t)[:, None, None, None]*(.8*self.t+.4*(self.t-self.r))[:, None, None, None]
        self.assertFalse(torch.allclose(target, partial_only))

    def test_directional_finite_difference_is_validation_only(self):
        eps = 1e-3
        plus = self.analytic_field(self.z+eps*self.v, self.r, self.t+eps, self.condition)
        minus = self.analytic_field(self.z-eps*self.v, self.r, self.t-eps, self.condition)
        expected = self.v + (self.r-self.t)[:, None, None, None]*(plus-minus)/(2*eps)
        actual = MeanFlowObjective.target(self.analytic_field, self.z, self.r, self.t, self.v, self.condition)
        torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-3)

    def test_chunking_and_outer_bfloat16_preserve_target(self):
        expected = MeanFlowObjective.target(self.analytic_field, self.z, self.r, self.t, self.v, self.condition, 0)
        for chunk in (None, 1, 3, 20):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = MeanFlowObjective.target(self.analytic_field, self.z, self.r, self.t, self.v, self.condition, chunk)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_diagonal_meanflow_target_is_instantaneous_velocity(self):
        target = MeanFlowObjective.target(self.analytic_field, self.z, self.t, self.t, self.v, self.condition)
        torch.testing.assert_close(target, self.v, rtol=0, atol=0)

    def test_fm_has_no_target_forward_or_jvp(self):
        with patch("modeling.flow_objectives.jvp", side_effect=AssertionError("FM called JVP")):
            target = FlowMatchingObjective.target(lambda *args: self.fail("FM target called field"),
                                                   self.z, self.t, self.t, self.v.requires_grad_(), self.condition)
        torch.testing.assert_close(target, self.v, rtol=0, atol=0)
        self.assertFalse(target.requires_grad)

    def test_time_interval_characteristic_direction(self):
        (time, interval), (dt, dh) = jvp(lambda r,t: interval_coordinates(r,t),
                                       (self.r,self.t), (torch.zeros_like(self.r),torch.ones_like(self.t)))
        torch.testing.assert_close(time, self.t)
        torch.testing.assert_close(interval, self.t-self.r)
        torch.testing.assert_close(dt, torch.ones_like(dt))
        torch.testing.assert_close(dh, torch.ones_like(dh))

    def test_fm_single_time_sampling_not_sorted_pair_marginal(self):
        for sampler in ("uniform", "logit_normal"):
            config = resolve_flow_config(flow_objective="fm", time_sampler=sampler)
            scheduler = fetch_schedulers("fm", 2, flow_config=config)[0]
            t, r = scheduler.sample_noise_step(20000, "cpu")
            self.assertTrue(torch.equal(t, r))
            self.assertAlmostEqual(t.mean().item(), .5, delta=.015)
            self.assertTrue(torch.all((t >= 0) & (t <= 1)))

    def test_actual_solver_uses_diagonal_only_for_fm(self):
        for objective in ("fm", "meanflow"):
            actor = make_actor(objective)
            for steps in (1, 2, 5):
                actor.n_steps = steps
                calls = []
                def field(self, z, r, t, condition):
                    calls.append((r.clone(), t.clone()))
                    return [torch.cat((torch.ones_like(z), z.new_zeros(*z.shape[:-1], 1)), -1)]
                actor.policy_forward_pass = MethodType(field, actor)
                actual, _ = actor.denoise_trajectory(torch.ones(2, 2, 2, 9), torch.zeros(2,16))
                torch.testing.assert_close(actual, torch.zeros_like(actual), atol=1e-6, rtol=0)
                self.assertEqual(len(calls), steps)
                for r, t in calls:
                    self.assertTrue(torch.equal(r,t) if objective == "fm" else torch.all(r < t))

    def test_real_tcn_jvp_chunking_and_training_gradients(self):
        actor = make_actor("meanflow")
        condition = torch.randn(7, 16)
        actor._jvp_microbatch_size = 0
        expected = actor.compute_flow_target(self.z, self.r, self.t, self.v, condition)
        for chunk in (1, 3):
            actor._jvp_microbatch_size = chunk
            actual = actor.compute_flow_target(self.z, self.r, self.t, self.v, condition)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        for objective in ("fm", "meanflow"):
            actor = make_actor(objective)
            data, proprio = poses(), poses(steps=3)
            loss = actor.compute_loss(data, None, None, None, None, proprio)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for module in (actor.encoder, actor.condition_pooler, actor.prediction_head):
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters()))

    def test_fm_training_and_endpoint_never_call_jvp(self):
        actor = make_actor("fm", endpoint=.25)
        with patch("modeling.flow_objectives.jvp", side_effect=AssertionError("FM called JVP")):
            loss = actor.compute_loss(poses(), None, None, None, None, poses(steps=3))
            loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("endpoint_position", actor.loss_diagnostics)

    def test_meanflow_ivc_keeps_main_and_condition_gradients(self):
        actor = make_actor("meanflow", ivc=.5)
        condition = torch.randn(7,16, requires_grad=True)
        r = self.r.clone()
        r[0] = self.t[0]
        loss = actor.compute_ivc_loss(self.z, r, self.t, self.v, condition)
        mask = r != self.t
        expected = F.mse_loss(actor.pose_velocity_field(self.z[mask], self.t[mask], self.t[mask], condition[mask]), self.v[mask])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(condition.grad[0].abs().sum().item(), 0.)
        self.assertGreater(condition.grad[1:].abs().sum().item(), 0.)

    def test_default_meanflow_loss_and_gradients_match_archived_implementation(self):
        # Stable baseline from before the extraction, not a moving HEAD reference.
        try:
            source = subprocess.check_output(["git", "show", "c80b1bd:modeling/policy/base_denoise_actor.py"],
                                             cwd=ROOT, text=True, encoding="utf-8", stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("Archived Git reference c80b1bd is unavailable")
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DenoiseActor")
        cls.bases = []
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in
                    {"compute_loss", "compute_meanflow_target", "compute_ivc_loss"}]
        tree.body = [cls]
        legacy_namespace = dict(namespace)
        exec(compile(tree, "archived_meanflow", "exec"), legacy_namespace)
        new = make_actor("meanflow", endpoint=.25, ivc=.5)
        old = deepcopy(new)
        for name in ("compute_loss", "compute_meanflow_target", "compute_ivc_loss"):
            setattr(old, name, MethodType(getattr(legacy_namespace["DenoiseActor"], name), old))
        data, proprio = poses(), poses(steps=3)
        torch.manual_seed(37)
        old_loss = old.compute_loss(data, None, None, None, None, proprio)
        old_loss.backward()
        old_rng = torch.get_rng_state()
        torch.manual_seed(37)
        new_loss = new.compute_loss(data, None, None, None, None, proprio)
        new_loss.backward()
        torch.testing.assert_close(new_loss, old_loss, atol=0, rtol=0)
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        for (name, p), (old_name, old_p) in zip(new.named_parameters(), old.named_parameters()):
            self.assertEqual(name, old_name)
            if p.grad is None:
                self.assertIsNone(old_p.grad)
            else:
                torch.testing.assert_close(p.grad, old_p.grad, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
