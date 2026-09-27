"""CPU regression checks for checkpoint integrity and next-update continuity."""

from contextlib import redirect_stderr
from copy import deepcopy
import ast
import io
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import parse_arguments
from utils.checkpoint_utils import load_model_state_strict
from utils.ema import EMA
from utils.training_checkpoint import (
    config_snapshot, read_checkpoint, validate_resume, validate_init_config, validate_evaluation_config,
    initialize_weights, restore_training_state, capture_rng_state, restore_rng_state,
    build_training_checkpoint, atomic_save_checkpoint, make_run_metadata,
    check_output_directory, write_run_manifest,
)
from utils.schedulers import fetch_scheduler


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Linear(2, 2)
        self.workspace_normalizer = nn.Parameter(torch.tensor([[0., 0.], [1., 1.]]), requires_grad=False)

    def forward(self, x):
        return self.core(x)


def components(use_ema=True):
    model = TinyModel()
    shadow = deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler("cpu")
    ema = EMA()
    return model, shadow, optimizer, scheduler, scaler, ema


def update(parts, step, use_ema=True):
    model, shadow, optimizer, scheduler, scaler, ema = parts
    optimizer.zero_grad(set_to_none=True)
    target = random.random() + np.random.random()
    loss = (model(torch.randn(3, 2)) - target).square().mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    ema.step(model, shadow, use_ema, step)
    return loss.detach().clone()


def sample_checkpoint(parts, use_ema=True):
    config = config_snapshot(parse_arguments(["--train_iters", "10", "--use_ema", str(use_ema)]))
    return deepcopy(build_training_checkpoint(
        *parts, config=config, run_metadata={"run_id": "test-run", "git": {}},
        step=3, best_loss=0.5, steps_per_epoch=2,
        rank_states=[{"rng": capture_rng_state(), "loader_epoch_generator": torch.Generator().get_state()}],
    ))


class TrainingResumeChecks(unittest.TestCase):
    def setUp(self):
        cuda_patch = patch("torch.cuda.is_available", return_value=False)
        cuda_patch.start()
        self.addCleanup(cuda_patch.stop)
        torch.manual_seed(3)
        random.seed(4)
        np.random.seed(5)

    def test_next_update_matches_uninterrupted_training(self):
        original = components()
        for step in range(3):
            update(original, step)
        checkpoint = sample_checkpoint(original)
        expected_loss = update(original, 3)
        resumed = components()
        validate_resume(checkpoint, checkpoint["config"], 1)
        step, best = restore_training_state(checkpoint, *resumed)
        self.assertEqual((step, best), (3, 0.5))
        self.assertEqual(resumed[2].param_groups[0]["lr"], checkpoint["optimizer"]["param_groups"][0]["lr"])
        restore_rng_state(checkpoint["rank_states"][0]["rng"])
        actual_loss = update(resumed, step)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        for old, new in zip(original[:2], resumed[:2]):
            for key, value in old.state_dict().items():
                torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(original[3].state_dict(), resumed[3].state_dict())
        self.assertEqual(original[4].state_dict(), resumed[4].state_dict())

    def test_serialization_is_weights_only_safe_and_complete(self):
        checkpoint = sample_checkpoint(components())
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("last.pth", "best.pth", "interm3.pth"):
                path = Path(tmp) / name
                atomic_save_checkpoint(checkpoint, path)
                loaded = read_checkpoint(path)
                validate_resume(loaded, checkpoint["config"], 1)
                self.assertEqual(loaded["iter"], 3)
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

    def test_init_resets_training_and_synchronizes_ema(self):
        original = components()
        update(original, 0)
        checkpoint = sample_checkpoint(original)
        for selection in ("raw", "ema"):
            fresh = components()
            before_scheduler = deepcopy(fresh[3].state_dict())
            initialize_weights(fresh[0], fresh[1], checkpoint, selection)
            chosen = checkpoint["weight" if selection == "raw" else "ema_weight"]
            for model in fresh[:2]:
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, chosen[key])
            self.assertEqual(fresh[2].state_dict()["state"], {})
            self.assertEqual(fresh[3].state_dict(), before_scheduler)

    def test_no_ema_resume_uses_raw_for_shadow(self):
        parts = components()
        checkpoint = sample_checkpoint(parts, use_ema=False)
        fresh = components()
        validate_resume(checkpoint, checkpoint["config"], 1)
        restore_training_state(checkpoint, *fresh)
        for key, value in fresh[0].state_dict().items():
            torch.testing.assert_close(value, fresh[1].state_dict()[key])
        with self.assertRaisesRegex(ValueError, "no ema_weight"):
            initialize_weights(fresh[0], fresh[1], checkpoint, "ema")

    def test_rejects_config_mismatches_including_same_shape_semantics(self):
        checkpoint = sample_checkpoint(components())
        for key, value in (("flow_objective", "fm"), ("num_attn_heads", 4),
                           ("relative_action", True), ("gripper_hold_prior_logit", 0.),
                           ("lr", 0.2), ("train_iters", 20), ("use_ema", False),
                           ("meanflow_offdiag_ratio", 0.5), ("batch_size", 32)):
            config = dict(checkpoint["config"], **{key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "config mismatch"):
                validate_resume(checkpoint, config, 1)
        config = dict(checkpoint["config"], run_log_dir="new-location", resume="last.pth")
        validate_resume(checkpoint, config, 1)

    def test_init_allows_objective_change_not_architecture_change(self):
        checkpoint = sample_checkpoint(components())
        validate_init_config(checkpoint, dict(checkpoint["config"], flow_objective="fm", lr=0.2))
        with self.assertRaisesRegex(ValueError, "architecture mismatch"):
            validate_init_config(checkpoint, dict(checkpoint["config"], num_attn_heads=4))

    def test_evaluation_rejects_objective_change_but_initialization_allows_it(self):
        checkpoint = sample_checkpoint(components())
        config = dict(checkpoint["config"], flow_objective="fm", denoise_model="fm")
        validate_init_config(checkpoint, config)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_evaluation_config(checkpoint, config)
        legacy = deepcopy(checkpoint)
        legacy["config"].pop("flow_objective")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_evaluation_config(legacy, config)

    def test_incomplete_legacy_and_wrong_world_size_fail(self):
        checkpoint = sample_checkpoint(components())
        for key in ("optimizer", "lr_scheduler", "scaler", "ema_state", "rank_states", "workspace_normalizer"):
            incomplete = dict(checkpoint)
            incomplete.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_resume(incomplete, checkpoint["config"], 1)
        with self.assertRaisesRegex(ValueError, "Legacy checkpoints"):
            validate_resume({"weight": checkpoint["weight"]}, checkpoint["config"], 1)
        with self.assertRaisesRegex(ValueError, "world_size"):
            validate_resume(checkpoint, checkpoint["config"], 2)
        with self.assertRaises(FileNotFoundError):
            read_checkpoint(Path(tempfile.gettempdir()) / "codex-nonexistent-checkpoint.pth")

    def test_incompatible_weights_do_not_partially_mutate_model(self):
        model = TinyModel()
        before = deepcopy(model.state_dict())
        for bad in ({"core.weight": torch.ones(2, 2)},
                    dict(before, **{"core.weight": torch.ones(3, 3)})):
            with self.assertRaises(RuntimeError):
                load_model_state_strict(model, bad)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_normalizer_cannot_be_silently_replaced(self):
        checkpoint = sample_checkpoint(components())
        bad = deepcopy(checkpoint)
        bad["workspace_normalizer"] += 1
        with self.assertRaisesRegex(ValueError, "normalizer metadata"):
            validate_resume(bad, checkpoint["config"], 1)
        bad = deepcopy(checkpoint)
        bad["weight"]["workspace_normalizer"][1] = -1
        with self.assertRaisesRegex(ValueError, "increasing bounds"):
            initialize_weights(*components()[:2], bad)

    def test_cli_modes_are_explicit_and_exclusive(self):
        self.assertEqual(parse_arguments(["--checkpoint", "old.pth"]).resume, "old.pth")
        self.assertIsNone(parse_arguments([]).resume)
        cases = (["--resume", "a", "--init_from", "b"],
                 ["--checkpoint", "a", "--resume", "b"],
                 ["--init_weights", "ema"], ["--eval_only", "true"],
                 ["--eval_only", "true", "--resume", "a"])
        for argv in cases:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(argv)
        args = parse_arguments(["--eval_only", "true", "--checkpoint", "a"])
        self.assertIsNone(args.resume)

    def test_run_identity_and_output_protection(self):
        config = config_snapshot(parse_arguments([]))
        source = {"run_metadata": {"run_id": "parent"}}
        init = make_run_metadata(config, source, "init", "old.pth")
        resumed = make_run_metadata(config, source, "resume", "last.pth")
        self.assertNotEqual(init["run_id"], "parent")
        self.assertEqual(init["parent_run_id"], "parent")
        self.assertEqual(resumed["run_id"], "parent")
        self.assertIn("commit", init["git"])
        with tempfile.TemporaryDirectory() as tmp:
            check_output_directory(tmp)
            write_run_manifest(tmp, init)
            with self.assertRaises(ValueError):
                check_output_directory(tmp)
            last = Path(tmp) / "last.pth"
            atomic_save_checkpoint({}, last)
            check_output_directory(tmp, last)
            with self.assertRaises(ValueError):
                check_output_directory(tmp, Path(tmp) / "best.pth")

    def test_real_trainer_orchestration_resume_and_init_on_cpu(self):
        # Execute the actual trainer class without importing optional CLIP/RLBench
        # dependencies. Only its environment/data/forward adapters are replaced.
        path = Path(__file__).resolve().parents[1] / "utils/trainers/base.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        dist = Mock()
        dist.get_rank.return_value = 0
        dist.get_world_size.return_value = 1
        dist.gather_object.side_effect = lambda state, output, dst: output.__setitem__(0, state)

        class Wrapper(nn.Module):
            def __init__(self, model, **kwargs):
                super().__init__()
                self.module = model

            def forward(self, x):
                return self.module(x)

        namespace = dict(globals(), dist=dist, DistributedDataParallel=Wrapper,
                         SummaryWriter=Mock(), trange=lambda start, end: range(start, end),
                         fetch_tokenizers=lambda _: None,
                         fetch_data_preprocessor=lambda _: lambda *a, **k: None,
                         fetch_depth2cloud=lambda _: None)
        exec(compile(tree, str(path), "exec"), namespace)
        base = namespace["BaseTrainTester"]

        class Trainer(base):
            def get_loaders(self):
                self.train_generator = torch.Generator().manual_seed(0)
                data = [torch.tensor([float(i), float(i)]) for i in range(4)]
                sampler = torch.utils.data.DistributedSampler(data, num_replicas=1, rank=0)
                loader = torch.utils.data.DataLoader(data, batch_size=2, sampler=sampler,
                                                   generator=self.train_generator)
                self.unique_train_samples = self.validation_samples = 4
                self.global_batch_size = 2
                self.seen = []
                return loader, loader, sampler

            def get_model(self):
                return TinyModel()

            def get_optimizer(self, model):
                return torch.optim.AdamW(model.parameters(), lr=self.args.lr)

            def get_workspace_normalizer(self):
                return torch.tensor([[0., 0.], [1., 1.]])

            def train_one_step(self, model, optimizer, scaler, scheduler, sample, step_id):
                self.seen.append((step_id, sample.clone()))
                optimizer.zero_grad(set_to_none=True)
                loss = (model(sample + torch.randn_like(sample)) - random.random() - np.random.random()).square().mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            def evaluate_nsteps(self, *args, **kwargs):
                return 0.3

        with tempfile.TemporaryDirectory() as tmp, patch("torch.cuda.current_device", return_value=0), \
                patch("torch.GradScaler", side_effect=lambda: torch.amp.GradScaler("cpu")):
            def args_for(name, extra=()):
                args = parse_arguments(["--train_iters", "3", "--val_freq", "1",
                                        "--interm_ckpt_freq", "1", "--use_ema", "true",
                                        "--lr_scheduler", "cosine", *extra])
                args.log_dir = Path(tmp) / name
                args.log_dir.mkdir()
                args.local_rank = 0
                return args

            original = Trainer(args_for("original"), None, None)
            expected = original.main()
            checkpoint = Path(tmp) / "original/interm2.pth"
            self.assertTrue(checkpoint.is_file())
            for saved_step in (1, 2):  # Mid-epoch and epoch-boundary restarts.
                source = Path(tmp) / f"original/interm{saved_step}.pth"
                resumed = Trainer(args_for(f"resumed{saved_step}", ["--resume", str(source)]), None, None)
                actual = resumed.main()
                self.assertEqual([step for step, _ in resumed.seen], list(range(saved_step, 3)))
                for (step, actual_batch), (_, expected_batch) in zip(resumed.seen, original.seen[saved_step:]):
                    torch.testing.assert_close(expected_batch, actual_batch)
                for key, value in expected.state_dict().items():
                    torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)
                full = read_checkpoint(Path(tmp) / f"resumed{saved_step}/last.pth")
                self.assertEqual(full["run_metadata"]["run_id"], original.run_metadata["run_id"])
                self.assertEqual(full["iter"], 3)
            initialized = Trainer(args_for("initialized", ["--init_from", str(checkpoint)]), None, None)
            initialized.main()
            self.assertEqual([step for step, _ in initialized.seen], [0, 1, 2])
            self.assertNotEqual(initialized.run_metadata["run_id"], original.run_metadata["run_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
