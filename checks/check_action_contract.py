"""CPU tests of the actual offline/online adapters and pose conversion code."""

import ast
from contextlib import redirect_stderr
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import parse_arguments as parse_train
from online_evaluation_rlbench.evaluate_policy import parse_arguments as parse_eval
from utils.action_contract import (
    current_proprio, select_proprio_history, validate_action_config, validate_proprio_shape,
)


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_methods(relative, class_name, methods, namespace):
    """Use actual methods without importing optional CLIP/robotics dependencies."""
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    tree.body = [cls]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[class_name]


preprocessor = load_file("base_preprocessor", "utils/data_preprocessors/base.py")
rotations = load_file("rotation_utils", "modeling/utils/utils.py")
Actor = load_methods(
    "modeling/policy/base_denoise_actor.py", "DenoiseActor",
    {"normalize_pos", "unnormalize_pos", "convert_rot", "unconvert_rot", "encode_inputs"},
    dict(vars(rotations), current_proprio=current_proprio, validate_proprio_shape=validate_proprio_shape),
)


def history(batch=2, hands=1):
    result = torch.zeros(batch, 3, hands, 8)
    result[..., 6] = 1  # Identity xyzw quaternion.
    for b in range(batch):
        for h in range(hands):
            result[b, :, h, 0] = torch.tensor([10., 20., 30.]) + b * 100 + h * 1000
            result[b, :, h, 7] = torch.tensor([1., 1., 0.])
    return result


class ActionContractChecks(unittest.TestCase):
    def test_training_and_online_use_same_recent_history(self):
        for hands in (1, 2):
            states = history(hands=hands)
            for count in (1, 2, 3):
                # Only device transport is replaced; execute process_proprio itself.
                with patch.object(torch.Tensor, "cuda", lambda tensor, **kw: tensor):
                    offline = preprocessor.DataPreprocessor(num_history=count).process_proprio(states)
                online = select_proprio_history(states.flatten(2), count, pad=True).unflatten(-1, (hands, 8))
                torch.testing.assert_close(offline, online, rtol=0, atol=0)
                torch.testing.assert_close(current_proprio(offline), states[:, -1:])
                self.assertTrue(torch.equal(offline[..., -1][:, -1], torch.zeros(2, hands)))

    def test_three_history_default_is_unchanged(self):
        states = history()
        self.assertTrue(torch.equal(select_proprio_history(states, 3), states[:, :3]))
        self.assertFalse(torch.equal(select_proprio_history(states, 1), states[:, :1]))
        self.assertEqual(select_proprio_history(states, 1)[0, 0, 0, 0].item(), 30.)

    def test_episode_start_padding_matches_existing_online_rule(self):
        for length in (1, 2, 3):
            available = history()[:, :length, 0]
            expected = torch.nn.functional.pad(available, (0, 0, 3 - length, 0), mode="replicate")
            actual = select_proprio_history(available, 3, pad=True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual.dtype, available.dtype)
            self.assertEqual(actual.device, available.device)
        with self.assertRaisesRegex(ValueError, "Not enough"):
            select_proprio_history(history()[:, :1], 3)

    def test_invalid_history_and_missing_gripper_are_rejected(self):
        for count in (0, -1):
            with self.assertRaises(ValueError):
                select_proprio_history(history(), count, pad=True)
        for states in (history()[:, :0], torch.zeros(3, 8)):
            with self.assertRaises(ValueError):
                current_proprio(states)
        for states in (history()[..., :7], history()[:, :2], history(hands=2)):
            with self.assertRaises(ValueError):
                validate_proprio_shape(states, num_history=3, nhand=1)

    def test_actual_converter_history_order(self):
        for filename, hands in (("peract_to_zarr.py", 1), ("hiveformer_to_zarr.py", 1),
                                ("peract2_to_zarr.py", 2)):
            path = ROOT / "data_processing" / filename
            tree = ast.parse(path.read_text(encoding="utf-8"))
            assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id in {"prop", "prop_1", "prop_2"}
                                   for target in node.targets)]
            tree.body = sorted(assignments, key=lambda node: node.lineno)
            states = np.zeros((4, hands * 8), dtype=np.float32)
            states[:, 0] = [10., 20., 30., 40.]
            if hands == 2:
                states[:, 8] = [110., 120., 130., 140.]
            content = {4: [state.reshape(1, hands * 8) for state in states[:-1]]}
            namespace = dict(np=np, NHAND=hands, content=content, states=states,
                             to_numpy=lambda x: x)
            exec(compile(tree, str(path), "exec"), namespace)
            actual = namespace["prop"]
            np.testing.assert_array_equal(actual[2, :, 0, 0], [10., 20., 30.])
            np.testing.assert_array_equal(actual[0, :, 0, 0], [10., 10., 10.])
            if hands == 2:
                np.testing.assert_array_equal(actual[2, :, 1, 0], [110., 120., 130.])

    def test_actual_online_actioners_preserve_gripper_and_hand_axes(self):
        for filename, hands in (("utils_with_rlbench.py", 1),
                                ("utils_with_hiveformer_rlbench.py", 1),
                                ("utils_with_bimanual_rlbench.py", 2)):
            Actioner = load_methods("online_evaluation_rlbench/" + filename, "Actioner", {"predict"}, {"torch": torch})
            actioner = Actioner()
            states = history(batch=1, hands=hands)
            received = []
            def policy(*args, **kwargs):
                received.append(args[6])
                return torch.zeros(1, 1, hands, 8)
            actioner._policy = policy
            actioner._instr = None
            with patch.object(torch.Tensor, "cuda", lambda tensor, **kw: tensor):
                actioner.predict(None, None, states.flatten(2), prediction_len=1)
            torch.testing.assert_close(received[0], states)

    def test_actor_current_anchor_uses_actual_latest_gripper(self):
        class Encoder:
            instruction_padding_mask = staticmethod(lambda instruction: None)
            def __call__(self, rgb, rgb2d, pcd, instruction, states):
                self.received = states
                return ()
        actor = Actor()
        actor.encoder = Encoder()
        actor._num_history, actor._nhand = 3, 2
        actor.flow_config = SimpleNamespace(action_head="film_tcn")
        states = history(hands=2)
        fixed = actor.encode_inputs(None, None, None, None, states)
        torch.testing.assert_close(fixed[0], states[:, -1:])
        torch.testing.assert_close(actor.encoder.received, states.flatten(1, 2))

    def test_quaternion_6d_roundtrip_and_sign_invariance(self):
        actor = Actor()
        actor._rotation_format = "quat_xyzw"
        torch.manual_seed(9)
        signal = torch.randn(2, 4, 2, 8)
        signal[..., 3:7] = torch.nn.functional.normalize(signal[..., 3:7], dim=-1)
        original = signal.clone()
        converted = actor.convert_rot(signal)
        opposite = signal.clone()
        opposite[..., 3:7] *= -1
        torch.testing.assert_close(converted, actor.convert_rot(opposite))
        restored = actor.unconvert_rot(converted)
        dot = (restored[..., 3:7] * signal[..., 3:7]).sum(-1).abs()
        torch.testing.assert_close(dot, torch.ones_like(dot))
        torch.testing.assert_close(restored[..., :3], signal[..., :3])
        torch.testing.assert_close(restored[..., 7:], signal[..., 7:])
        self.assertTrue(torch.equal(signal, original))
        identity = torch.tensor([[[[0., 0., 0., 0., 0., 0., 1., 1.]]]])
        torch.testing.assert_close(actor.convert_rot(identity)[..., 3:9],
                                   torch.tensor([[[[1., 0., 0., 0., 1., 0.]]]]))
        quarter_turn = identity.clone()
        quarter_turn[..., 5:7] = 2 ** -0.5  # +90 degrees around world z.
        torch.testing.assert_close(actor.convert_rot(quarter_turn)[..., 3:9],
                                   torch.tensor([[[[0., 1., 0., -1., 0., 0.]]]]), atol=1e-6, rtol=0)

    def test_encoder_pose_features_match_action_rotation_convention(self):
        Encoder = load_methods("modeling/encoder/multimodal/encoder3d.py", "Encoder",
                               {"encode_proprio"}, dict(vars(rotations)))
        encoder = Encoder()
        encoder.curr_gripper_embed = torch.nn.Embedding(6, 4)
        captured = []
        def state_encoder(state):
            captured.append(state)
            return torch.zeros(*state.shape[:-1], 4)
        encoder.proprio_state_encoder = state_encoder
        encoder.relative_pe_layer = lambda xyz: xyz
        encoder.gripper_context_head = lambda seq1, seq2, **kwargs: [seq1]
        states = history(hands=2).flatten(1, 2)
        states[..., 5:7] = 2 ** -0.5
        encoder.encode_proprio(states, torch.zeros(2, 4, 4), torch.zeros(2, 4, 3))
        actor = Actor()
        actor._rotation_format = "quat_xyzw"
        torch.testing.assert_close(captured[0], actor.convert_rot(states))
        opposite = states.clone()
        opposite[..., 3:7] *= -1
        encoder.encode_proprio(opposite, torch.zeros(2, 4, 4), torch.zeros(2, 4, 3))
        torch.testing.assert_close(captured[0], captured[1])

    def test_peract_observation_and_keypose_units_are_preserved(self):
        Peract = load_methods("utils/data_preprocessors/peract.py", "PeractDataPreprocessor",
                              {"process_obs"}, {"torch": torch})
        rgb = torch.tensor([0, 127, 255], dtype=torch.uint8).reshape(1, 1, 3, 1, 1)
        pcd = torch.tensor([-1.2, 0.5, 1.4]).reshape(1, 1, 3, 1, 1)
        with patch.object(torch.Tensor, "cuda", lambda tensor, **kw: tensor):
            actual_rgb, actual_pcd = Peract().process_obs(rgb, pcd, augment=False)
            actions = history(hands=2)
            chosen = preprocessor.DataPreprocessor(keypose_only=True).process_actions(actions)
        torch.testing.assert_close(actual_rgb, rgb.float() / 255.)
        torch.testing.assert_close(actual_pcd, pcd)
        torch.testing.assert_close(chosen, actions[:, -1:])

    def test_world_position_normalizer_roundtrip_no_clipping(self):
        actor = Actor()
        actor.nrm_dim = 3
        actor.workspace_normalizer = torch.tensor([[-1., -2., 0.], [3., 2., 2.]])
        world = torch.tensor([[-1., -2., 0., 0., 0., 0., 1., 0.],
                              [3., 2., 2., 0., 0., 0., 1., 1.],
                              [5., 0., 1., 0., 0., 0., 1., 0.]]).reshape(1, 3, 1, 8)
        original = world.clone()
        normalized = actor.normalize_pos(world)
        torch.testing.assert_close(normalized[0, :2, 0, :3], torch.tensor([[-1., -1., -1.], [1., 1., 1.]]))
        self.assertGreater(normalized[0, 2, 0, 0].item(), 1.)
        torch.testing.assert_close(actor.unnormalize_pos(normalized), original)
        self.assertTrue(torch.equal(world, original))
        torch.testing.assert_close(normalized[..., 3:], world[..., 3:])

    def test_cli_rejects_inconsistent_pose_modes(self):
        for parse in (parse_train, parse_eval):
            self.assertEqual(parse([]).input_contract_version, 1)
            for argv in (["--num_history", "0"], ["--num_history", "-1"],
                         ["--rotation_format", "euler"], ["--rotation_format", "quat_wxyz"],
                         ["--relative_action", "true"]):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse(argv)
        for values in ((0, "quat_xyzw", False), (3, "euler", False), (3, "quat_xyzw", True)):
            with self.assertRaises(ValueError):
                validate_action_config(*values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
