"""CPU loss/recipe and execution ablation checks, not robot-success evidence."""
import ast
from contextlib import redirect_stderr
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import parse_arguments
from modeling.loss_config import LossConfig
from checks.check_flow_objectives import Actor, make_actor, poses
from checks.check_transformer_action_head import TinyObservationEncoder
from checks.check_training_resume import sample_checkpoint, components
from checks.check_action_contract import load_methods
from utils.training_checkpoint import validate_resume, validate_init_config
from online_evaluation_rlbench.evaluate_policy import parse_arguments as parse_eval, evaluation_metadata
from online_evaluation_rlbench.gripper_control import (
    gripper_command, hysteresis_gripper_command, should_execute_gripper_change,
)


class LossAblationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(65)

    def test_cli_defaults_and_plain_bce_cost_defaults(self):
        args = parse_arguments([])
        self.assertEqual(args.gripper_loss_type, "weighted_bce")
        self.assertEqual((args.pose_position_weight, args.pose_rotation_weight, args.gripper_loss_weight), (30., 10., 1.))
        self.assertEqual((args.gripper_transition_weight, args.gripper_closed_hold_weight, args.gripper_hold_prior_logit), (2., 2., 2.))
        bce = parse_arguments(["--gripper_loss_type", "bce"])
        self.assertEqual((bce.gripper_transition_weight, bce.gripper_closed_hold_weight), (0., 0.))
        actor = Actor(embedding_dim=16, action_hidden_dim=16, action_num_blocks=1, gripper_loss_type="bce")
        self.assertEqual(actor.loss_config.gripper_transition_weight, 0.)

    def test_invalid_losses_fail_before_training(self):
        for name in ("pose_position_weight", "pose_rotation_weight", "gripper_loss_weight",
                     "gripper_transition_weight", "gripper_closed_hold_weight", "gripper_hold_prior_logit",
                     "endpoint_loss_weight", "ivc_loss_weight"):
            for invalid in ("nan", "inf", "-1"):
                with self.subTest(name=name, value=invalid), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_arguments([f"--{name}", invalid])
                with self.assertRaises(ValueError):
                    LossConfig(**{name: float(invalid)}).validate()
        for argv in (["--gripper_loss_type", "bce", "--gripper_transition_weight", "2"],
                     ["--model_type", "denoise2d", "--pose_position_weight", "1"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(argv)

    def test_training_factory_forwards_resolved_ablation_options(self):
        Trainer = load_methods("utils/trainers/base.py", "BaseTrainTester", {"get_model"},
                               {"dist": SimpleNamespace(get_rank=lambda: 1)})
        trainer = Trainer()
        trainer.args = parse_arguments(["--gripper_loss_type", "bce", "--gripper_hold_prior_logit", "0",
                                        "--pose_position_weight", "7", "--pose_rotation_weight", "3",
                                        "--gripper_loss_weight", ".4"])
        captured = {}
        def factory(**kwargs):
            captured.update(kwargs)
            return torch.nn.Linear(2, 2)
        trainer.model_cls = factory
        trainer.get_model()
        for name, value in (("gripper_loss_type", "bce"), ("gripper_transition_weight", 0.),
                            ("gripper_closed_hold_weight", 0.), ("gripper_hold_prior_logit", 0.),
                            ("pose_position_weight", 7.), ("pose_rotation_weight", 3.), ("gripper_loss_weight", .4)):
            self.assertEqual(captured[name], value)

    def test_plain_and_weighted_bce_analytic_loss_and_gradients(self):
        logits = torch.tensor([-.8, .1, .9, -.3], requires_grad=True).reshape(1, 4, 1, 1)
        target = torch.tensor([0., 0., 1., 1.]).reshape_as(logits)
        current = torch.tensor([1., 0., 0., 1.]).reshape_as(logits)
        for kind in ("bce", "weighted_bce"):
            actor = Actor(embedding_dim=16, action_hidden_dim=16, action_num_blocks=1,
                          gripper_loss_type=kind, gripper_loss_weight=.7,
                          gripper_transition_weight=0. if kind == "bce" else 4.,
                          gripper_closed_hold_weight=0. if kind == "bce" else 2.)
            elementwise = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            weights = torch.tensor([5., 3., 5., 1.]).reshape_as(logits)
            expected = .7 * (elementwise.mean() if kind == "bce" else (weights*elementwise).sum()/weights.sum())
            actual = actor.compute_gripper_loss(logits, target, current)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                                       torch.autograd.grad(expected, logits, retain_graph=True)[0], rtol=0, atol=0)

    def test_zero_gripper_weight_retains_zero_gradient_edges(self):
        actor = Actor(embedding_dim=16, action_hidden_dim=16, action_num_blocks=1, gripper_loss_weight=0.)
        logits = torch.randn(2, 1, 1, 1, requires_grad=True)
        loss = actor.compute_gripper_loss(logits, torch.ones_like(logits), torch.zeros_like(logits))
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.abs().sum().item(), 0.)

    def test_zero_hold_prior_is_a_separate_explicit_recipe(self):
        for prior in (0., 2.):
            actor = Actor(embedding_dim=16, action_hidden_dim=16, action_num_blocks=1,
                          gripper_prediction_mode="direct", gripper_hold_prior_logit=prior)
            states = torch.tensor([[[0.]], [[1.]]])
            actual = actor.predict_gripper_logits(torch.zeros(2, 16), states, 3)
            expected = torch.tensor([-prior, prior])[:, None, None, None].expand_as(actual)
            torch.testing.assert_close(actual, expected)
        with self.assertRaisesRegex(ValueError, "architecture mismatch"):
            validate_init_config({"config": {"gripper_hold_prior_logit": 2.}}, {"gripper_hold_prior_logit": 0.})

    def test_loss_scaling_does_not_change_field_target_or_solver(self):
        actor = make_actor("meanflow", endpoint=.25, ivc=.5)
        changed = deepcopy(actor)
        changed.loss_config = replace(actor.loss_config, pose_position_weight=90., pose_rotation_weight=30., gripper_loss_weight=.3)
        z, condition = torch.randn(3, 2, 2, 9), torch.randn(3, 16)
        r, t, v = torch.full((3,), .2), torch.full((3,), .8), torch.randn_like(z)
        torch.testing.assert_close(actor.pose_velocity_field(z, r, t, condition), changed.pose_velocity_field(z, r, t, condition), rtol=0, atol=0)
        torch.testing.assert_close(actor.compute_flow_target(z, r, t, v, condition), changed.compute_flow_target(z, r, t, v, condition), rtol=0, atol=0)
        for a, b in zip(actor.denoise_trajectory(z, condition), changed.denoise_trajectory(z, condition)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        gt = actor.convert_rot(poses()[..., :-1])
        old_endpoint = actor.compute_endpoint_loss(z, condition, gt[..., :3], gt)
        new_endpoint = changed.compute_endpoint_loss(z, condition, gt[..., :3], gt)
        torch.testing.assert_close(new_endpoint, 3*old_endpoint)

    def test_all_head_objective_and_gripper_combinations_train_with_bce(self):
        for head in ("film_tcn", "transformer"):
            for objective in ("fm", "meanflow", "imf"):
                for gripper in ("direct", "legacy_denoise"):
                    with self.subTest(head=head, objective=objective, gripper=gripper):
                        actor = Actor(embedding_dim=16, action_hidden_dim=16, num_attn_heads=4,
                                      action_num_blocks=1, action_head=head, flow_objective=objective,
                                      gripper_prediction_mode=gripper, gripper_loss_type="bce",
                                      gripper_hold_prior_logit=0., pose_position_weight=7.,
                                      pose_rotation_weight=3., gripper_loss_weight=.4,
                                      nhist=3, nhand=2, lv2_batch_size=1, endpoint_loss_weight=.1,
                                      ivc_loss_weight=.2 if objective == "meanflow" else 0.)
                        actor.encoder = TinyObservationEncoder()
                        actor.collect_training_diagnostics = True
                        with torch.autocast("cpu", dtype=torch.bfloat16):
                            loss = actor.compute_loss(poses(), torch.randn(3, 7, 8), None,
                                                      torch.randn(3, 7, 3), torch.randn(3, 4, 8), poses(steps=3))
                        loss.backward()
                        self.assertTrue(torch.isfinite(loss))
                        torch.testing.assert_close(loss, sum(actor.loss_diagnostics.values()), rtol=1e-5, atol=1e-5)
                        self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None))

    def test_resume_rejects_loss_changes_but_init_allows_them(self):
        with patch("torch.cuda.is_available", return_value=False):
            checkpoint = sample_checkpoint(components())
        for name, value in (("pose_position_weight", 7.), ("pose_rotation_weight", 3.),
                            ("gripper_loss_weight", .2), ("gripper_loss_type", "bce")):
            config = dict(checkpoint["config"], **{name: value})
            with self.assertRaisesRegex(ValueError, "config mismatch"):
                validate_resume(checkpoint, config, 1)
            validate_init_config(checkpoint, config)

    def test_execution_threshold_is_independent_of_current_state(self):
        for probability in (0., .25, .49, .5, .51, .75, 1.):
            for current in (0., 1.):
                self.assertEqual(gripper_command(probability, current), hysteresis_gripper_command(probability, current))
                self.assertEqual(gripper_command(probability, current, mode="threshold"), float(probability >= .5))
        self.assertFalse(should_execute_gripper_change(0., 1., False, True))
        self.assertTrue(should_execute_gripper_change(0., 1., False, False))
        with self.assertRaises(ValueError):
            gripper_command(.5, 0., mode="typo")

    def test_execution_cli_and_sidecar_metadata(self):
        args = parse_eval(["--gripper_command_mode", "threshold", "--require_pose_for_gripper_open", "false"])
        model = SimpleNamespace(evaluation_checkpoint={"training_config": {"gripper_loss_type": "bce"}})
        metadata = evaluation_metadata(args, model)
        serialized = json.loads(json.dumps(metadata))
        self.assertEqual(serialized["evaluation_config"]["gripper_command_mode"], "threshold")
        self.assertEqual(serialized["checkpoint"]["training_config"]["gripper_loss_type"], "bce")
        for argv in (["--gripper_command_mode", "typo"],
                     ["--gripper_command_mode", "threshold", "--bimanual", "true"],
                     ["--gripper_command_mode", "threshold", "--gripper_open_threshold", ".8"],
                     ["--dataset", "Hiveformer", "--require_pose_for_gripper_open", "false"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_eval(argv)

    def test_real_mover_command_and_open_gate_are_independent(self):
        path = ROOT / "online_evaluation_rlbench/utils_with_rlbench.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Mover"]
        namespace = dict(np=np, gripper_command=gripper_command, should_execute_gripper_change=should_execute_gripper_change)
        exec(compile(tree, str(path), "exec"), namespace)
        Mover = namespace["Mover"]
        initial = np.array([0., 0., 0., 0., 0., 0., 1., 0.])
        class Task:
            def __init__(self):
                self.actions = []
            def step(self, action):
                self.actions.append(action.copy())
                # Deliberately fail to reach the requested xyz.
                return SimpleNamespace(gripper_pose=initial[:7], gripper_open=float(action[7])), 0, False
        for mode, gate, expected in (("hysteresis", False, 0.), ("threshold", True, 0.), ("threshold", False, 1.)):
            task = Task()
            move = Mover(task, initial_action=initial, gripper_command_mode=mode, require_pose_for_gripper_open=gate)
            target = initial.copy()
            target[0], target[7] = 1., .6
            move(target)
            self.assertEqual(task.actions[-1][7], expected)
            self.assertEqual(move.last_gripper_change_executed, expected == 1.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
