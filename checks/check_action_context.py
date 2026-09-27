"""Named token alignment, source views, history contract and step-5 parity."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks.check_transformer_action_head import make_context, TinyObservationEncoder
from checks.check_flow_objectives import Actor, poses
from modeling.transformer_action_head import TransformerActionHead
from modeling.flow_objectives import MeanFlowObjective


class ActionContextChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(53)

    def test_source_views_slice_and_detach_preserve_alignment(self):
        context = replace(make_context(grad=True), dense_scene_count=5,
                          proprio_xyz=torch.randn(3, 6, 3), proprio_layout=(3, 2)).validate()
        self.assertEqual(context.dense_scene.tokens.data_ptr(), context.scene_tokens.data_ptr())
        torch.testing.assert_close(context.sparse_scene.xyz, context.scene_xyz[:, 5:])
        selected = context[torch.tensor([True, False, True])].detach().float().validate()
        self.assertEqual(selected.dense_scene_count, 5)
        self.assertEqual(selected.proprio_layout, (3, 2))
        self.assertFalse(selected.scene_tokens.requires_grad)
        self.assertEqual(selected.language.padding_mask.dtype, torch.bool)
        torch.testing.assert_close(selected.proprio.xyz, context.proprio_xyz[[0, 2]])
        self.assertEqual(context.scene_tokens.shape[1], 7)

    def test_invalid_alignment_masks_and_bounds_rejected(self):
        base = make_context()
        for changes in ({"dense_scene_count": 8}, {"scene_xyz": torch.zeros(3, 6, 3)},
                        {"language_padding_mask": torch.zeros(3, 4)},
                        {"proprio_layout": (3, 2)}, {"workspace_bounds": torch.zeros(2, 3)}):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                replace(base, **changes).validate()

    def test_actor_history_major_hand_major_world_alignment(self):
        for hands in (1, 2):
            actor = Actor(embedding_dim=16, action_hidden_dim=16, num_attn_heads=4,
                          action_num_blocks=1, action_head="transformer", nhist=3, nhand=hands)
            actor.encoder = TinyObservationEncoder()
            proprio = poses(steps=3)[:, :, :hands]
            pcd = torch.randn(3, 7, 3)
            fixed = actor.encode_inputs(torch.randn(3, 7, 8), None, pcd, torch.randn(3, 4, 8), proprio)
            context = actor.encode_condition(fixed)
            self.assertEqual(context.proprio_layout, (3, hands))
            torch.testing.assert_close(context.proprio.xyz, proprio[..., :3].flatten(1, 2))
            torch.testing.assert_close(context.dense_scene.xyz, pcd)
            torch.testing.assert_close(context.sparse_scene.xyz, pcd[:, :2])
            with self.assertRaisesRegex(ValueError, "world proprio"):
                actor.encode_condition(fixed[:11])

    def test_metadata_does_not_change_step5_field_jvp_or_gradients(self):
        head = TransformerActionHead(12, 16, 16, 4, 2, 2)
        base = make_context(grad=True)
        named = replace(base, dense_scene_count=5, proprio_xyz=torch.randn(3, 6, 3), proprio_layout=(3, 2)).validate()
        z = torch.randn(3, 2, 2, 9)
        r, t = torch.full((3,), .2), torch.full((3,), .8)
        field = lambda z, r, t, c: head(z, r, t, c)[0][..., :9]
        old, new = field(z, r, t, base), field(z, r, t, named)
        torch.testing.assert_close(old, new, atol=0, rtol=0)
        v = torch.randn_like(z)
        torch.testing.assert_close(MeanFlowObjective.target(field, z, r, t, v, base, 2),
                                   MeanFlowObjective.target(field, z, r, t, v, named, 2), atol=0, rtol=0)
        parameters = tuple(head.parameters())
        old_grad = torch.autograd.grad(old.square().mean(), parameters, retain_graph=True)
        new_grad = torch.autograd.grad(new.square().mean(), parameters)
        for a, b in zip(old_grad, new_grad):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
