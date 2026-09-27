"""Step-8 recipe, seeding, safe launch and synthetic-result reporting checks."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import ast
import io
import json
import os
import pickle
from pathlib import Path
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.flow_matrix import build_suite, load_suite, write_json, launch, report, verify_checkpoint, digest
from main import parse_arguments
from modeling.flow_config import configure_action_precision
from utils.reproducibility import seed_training, data_seed


class FlowMatrixChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.suite = build_suite("test_matrix", ["close_jar", "open_drawer"], "plain",
                                 seeds=[0, 1], eval_seeds=[2], eval_steps=[1, 2], project_dir=self.root)
        self.suite["source_sha256"] = "fixture-source"
        self.suite.pop("manifest_sha256")
        self.suite["manifest_sha256"] = digest(self.suite)

    def fixture_results(self):
        for job in self.suite["jobs"]:
            for evaluation in job["evaluations"]:
                task = evaluation["config"]["task"]
                # 1/3 is deliberately rounded in the legacy score JSON.
                write_json(evaluation["output"], {task: {"0": .33, "mean": .33}})
                meta = dict(
                    evaluation_config=evaluation["config"], evaluation_source_sha256="fixture-source",
                    checkpoint=dict(path=job["checkpoint"], iteration=self.suite["spec"]["train_iters"],
                                    selected_weights="EMA", training_config=job["config"], source_sha256="fixture-source",
                                    run_id=job["id"], workspace_normalizer=[[0., 0., 0.], [1., 1., 1.]],
                                    runtime={"torch": "fixture-version"}, dataset_counts={"train_samples": 100, "validation_samples": 20},
                                    parameter_counts={"total": 1000, "trainable": 700, "action_head": 200}),
                    coverage={task: {"0": {"successes": 1, "episodes": 3, "episode_ids": ["episode0", "episode1", "episode2"]}}},
                )
                write_json(evaluation["output"] + ".config.json", meta)

    def test_four_cells_differ_only_in_declared_axes(self):
        self.assertEqual(len(self.suite["jobs"]), 8)
        self.assertEqual(sum(len(j["evaluations"]) for j in self.suite["jobs"]), 32)
        allowed = {"action_head", "flow_objective", "denoise_model", "meanflow_offdiag_ratio", "seed", "run_log_dir"}
        reference = self.suite["jobs"][0]["config"]
        for job in self.suite["jobs"]:
            changes = {k for k in reference if job["config"][k] != reference[k]}
            self.assertLessEqual(changes, allowed)
            self.assertEqual(job["config"]["ivc_loss_weight"], 0.)
            self.assertEqual(job["config"]["endpoint_loss_weight"], 0.)
            self.assertEqual(job["config"]["matmul_precision"], "ieee")
            self.assertEqual(job["config"]["gripper_loss_type"], "bce")
            self.assertEqual(job["config"]["gripper_hold_prior_logit"], 0.)
        compat = build_suite("compat", ["close_jar"], "compat", seeds=[0], project_dir=self.root)
        self.assertEqual(compat["jobs"][0]["config"]["gripper_hold_prior_logit"], 2.)

    def test_invalid_specs_and_tampered_manifest_are_rejected(self):
        for changed in ({"name": "../escape"}, {"seeds": [0, 0]}, {"seeds": [-1]}, {"tasks": []},
                        {"eval_steps": [0]}, {"batch_size": 0}, {"lr": float("nan")}, {"gripper_profile": "best"}):
            args = dict(name="valid", tasks=["close_jar"], gripper_profile="plain", project_dir=self.root)
            args.update(changed)
            with self.assertRaises(ValueError):
                build_suite(**args)
        path = self.root / "suite.json"
        write_json(path, self.suite)
        self.assertEqual(load_suite(path)["jobs"], self.suite["jobs"])
        tampered = deepcopy(self.suite)
        tampered["jobs"][0]["argv"].extend(["--ivc_loss_weight", "999"])
        write_json(self.root / "tampered.json", tampered)
        with self.assertRaises(ValueError):
            load_suite(self.root / "tampered.json")
        tampered.pop("manifest_sha256")
        tampered["manifest_sha256"] = digest(tampered)
        write_json(self.root / "retagged.json", tampered)
        with self.assertRaisesRegex(ValueError, "shared recipe"):
            load_suite(self.root / "retagged.json")

    def test_plan_and_preview_never_start_subprocesses_or_outputs(self):
        with patch("subprocess.run") as run, redirect_stdout(io.StringIO()):
            job = self.suite["jobs"][0]
            launch(self.suite, job["id"])
            launch(self.suite, job["id"], job["evaluations"][0]["id"])
            run.assert_not_called()
        self.assertFalse((self.root / "train_logs").exists())
        self.assertFalse((self.root / "eval_logs").exists())
        with self.assertRaises(ValueError):
            launch(self.suite, "nonexistent")

    def test_execute_checks_source_and_missing_inputs(self):
        job = self.suite["jobs"][0]
        with patch("subprocess.run") as run, redirect_stdout(io.StringIO()):
            with patch("utils.training_checkpoint.git_identity", return_value={"source_sha256": None}):
                with self.assertRaisesRegex(ValueError, "Source changed"):
                    launch(self.suite, job["id"], execute=True)
            with patch("utils.training_checkpoint.git_identity", return_value={"source_sha256": "fixture-source"}):
                with self.assertRaisesRegex(ValueError, "Missing training input"):
                    launch(self.suite, job["id"], execute=True)
            run.assert_not_called()

    def test_launch_requires_final_step_matching_ema_checkpoint(self):
        job = self.suite["jobs"][0]
        checkpoint = dict(iter=300000, ema_weight={"fixture": 1}, config=job["config"],
                          run_metadata={"git": {"source_sha256": "fixture-source"}})
        verify_checkpoint(self.suite, job, checkpoint)
        for change in ({"iter": 1000}, {"ema_weight": None}, {"config": {}}):
            with self.assertRaises(ValueError):
                verify_checkpoint(self.suite, job, dict(checkpoint, **change))

    def test_missing_results_have_no_aggregate(self):
        result = report(self.suite)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len(result["missing"]), 32)
        self.assertEqual(result["aggregates"], [])

    def test_complete_report_uses_integer_counts_not_rounded_scores(self):
        self.fixture_results()
        result = report(self.suite)
        self.assertEqual(result["status"], "complete", result["rejected"])
        self.assertEqual(len(result["aggregates"]), 8)
        for aggregate in result["aggregates"]:
            self.assertAlmostEqual(aggregate["macro_success_rate"], 1/3)
            self.assertEqual(aggregate["std_across_training_seeds"], 0.)

    def test_report_rejects_mixed_recipe_counts_bounds_or_episode_identities(self):
        self.fixture_results()
        path = Path(self.suite["jobs"][-1]["evaluations"][-1]["output"] + ".config.json")
        original = json.loads(path.read_text())
        modifications = [
            lambda m: m["evaluation_config"].update(seed=19),
            lambda m: m["checkpoint"].update(iteration=100),
            lambda m: m["checkpoint"].update(run_id="different-run"),
            lambda m: m["checkpoint"].update(workspace_normalizer=[[0., 0., 0.], [2., 2., 2.]]),
            lambda m: m.update(evaluation_source_sha256="other-source"),
            lambda m: m["checkpoint"]["dataset_counts"].update(train_samples=999),
            lambda m: m["coverage"]["open_drawer"]["0"].update(episode_ids=["episode0", "episode1", "episode3"]),
        ]
        # Mock only the file read of one fixture; no rewriting production files.
        real_read = Path.read_text
        for mutate in modifications:
            changed = deepcopy(original)
            mutate(changed)
            def read(p, *a, **kw):
                return json.dumps(changed) if p == path else real_read(p, *a, **kw)
            with patch.object(Path, "read_text", read):
                result = report(self.suite)
            self.assertEqual(result["status"], "incomplete")
            self.assertTrue(result["rejected"])
            self.assertFalse(result["aggregates"])

    def test_process_seed_and_data_seed_are_repeatable_and_opt_in(self):
        def sample():
            return random.random(), np.random.rand(), torch.rand(3)
        seed_training(17)
        first = sample()
        seed_training(17)
        second = sample()
        self.assertEqual(first[:2], second[:2])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)
        state = torch.get_rng_state()
        seed_training(None)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(data_seed(None), 0)
        self.assertEqual(data_seed(17), 17)
        seed_training(17, 1)
        self.assertFalse(torch.equal(first[2], torch.rand(3)))

    def test_precision_and_legacy_cli_defaults(self):
        self.assertIsNone(parse_arguments([]).seed)
        self.assertEqual(parse_arguments([]).matmul_precision, "legacy")
        for argv in (["--seed", "-1"], ["--seed", str(2**32)], ["--matmul_precision", "typo"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(argv)
        matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
            configure_action_precision("film_tcn", "legacy")
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            configure_action_precision("film_tcn", "ieee")
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn

    def test_writing_refuses_to_replace_existing_artifacts(self):
        path = self.root / "suite.json"
        write_json(path, self.suite)
        with self.assertRaises(FileExistsError):
            write_json(path, {"replacement": True})

    def test_demo_loader_retains_default_list_and_optional_episode_ids(self):
        path = ROOT / "online_evaluation_rlbench/get_stored_demos.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_stored_demos"]
        namespace = dict(pickle=pickle, np=np, listdir=os.listdir, join=os.path.join, exists=os.path.exists,
                         natsorted=sorted, VARIATIONS_FOLDER="variation%d", EPISODES_FOLDER="episodes",
                         LOW_DIM_PICKLE="low_dim_obs.pkl")
        exec(compile(tree, str(path), "exec"), namespace)
        for identity in ("episode0", "episode1"):
            folder = self.root / "close_jar/variation0/episodes" / identity
            folder.mkdir(parents=True)
            (folder / "low_dim_obs.pkl").write_bytes(pickle.dumps(SimpleNamespace(fixture=identity)))
        function = namespace["get_stored_demos"]
        original = function(amount=-1, dataset_root=str(self.root))
        demos, identities = function(amount=-1, dataset_root=str(self.root), return_identifiers=True)
        self.assertIsInstance(original, list)
        self.assertEqual([d.fixture for d in original], identities)
        self.assertEqual([d.fixture for d in demos], identities)

    def test_loader_generator_and_sampler_share_the_explicit_seed(self):
        from checks.check_action_contract import load_methods
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        def sampler(dataset, **kwargs):
            return DistributedSampler(dataset, num_replicas=1, rank=0, **kwargs)
        def loader(dataset, **kwargs):
            kwargs.pop("prefetch_factor", None)
            kwargs.pop("persistent_workers", None)
            kwargs["num_workers"] = 0
            kwargs["pin_memory"] = False
            return DataLoader(dataset, **kwargs)
        Trainer = load_methods("utils/trainers/base.py", "BaseTrainTester", {"get_loaders"},
                               dict(torch=torch, np=np, random=random, DistributedSampler=sampler,
                                    DataLoader=loader, base_collate_fn=lambda values: torch.stack(values),
                                    dist=SimpleNamespace(get_rank=lambda: 0, get_world_size=lambda: 1)))
        class Dataset(torch.utils.data.Dataset):
            annos = {"action": list(range(20))}
            def __len__(self): return 20
            def __getitem__(self, index): return torch.tensor(index)
        sequences = []
        for seed in (7, 7, 8):
            trainer = Trainer()
            trainer.args = parse_arguments(["--seed", str(seed), "--batch_size", "4"])
            trainer.get_datasets = lambda: (Dataset(), Dataset())
            train_loader, _, train_sampler = trainer.get_loaders()
            self.assertEqual(train_sampler.seed, seed)
            self.assertEqual(trainer.train_generator.initial_seed(), seed)
            sequences.append(torch.cat(list(train_loader)))
        self.assertTrue(torch.equal(sequences[0], sequences[1]))
        self.assertFalse(torch.equal(sequences[0], sequences[2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
