"""Check stage launch arguments without starting training or a simulator."""
from contextlib import redirect_stderr
import ast
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import parse_arguments


class FlowStageChecks(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.bash = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
        if not Path(self.bash).is_file():
            self.skipTest("Bash unavailable")
        # These previews print the command; they never launch torchrun.
        self.env = dict(os.environ, PYTHON_BIN="echo")
        for key in ("FLOW_OBJECTIVE", "TRAIN_ITERS", "MEANFLOW_OFFDIAG_RATIO",
                    "ENDPOINT_LOSS_WEIGHT", "IVC_LOSS_WEIGHT", "MILESTONE_CKPT_STEPS",
                    "RUN_LOG_DIR", "RESUME", "INIT_FROM", "INIT_WEIGHTS", "CHECKPOINT",
                    "EVAL_ONLY", "SEED"):
            self.env.pop(key, None)

    def preview(self, script, *args, **env):
        output = subprocess.check_output([self.bash, script, *args], cwd=self.root,
                                         env=dict(self.env, **env), text=True)
        tokens = shlex.split(output)
        return parse_arguments(tokens[tokens.index("main.py") + 1:])

    def test_fm_pretrain_retains_both_requested_snapshots(self):
        args = self.preview("train_flow_stages.sh", "fm")
        self.assertEqual((args.flow_objective, args.train_iters), ("fm", 300000))
        self.assertEqual(args.milestone_ckpt_steps, (240000, 300000))
        self.assertEqual((args.meanflow_offdiag_ratio, args.endpoint_loss_weight, args.ivc_loss_weight), (0, 0, 0))
        self.assertIsNone(args.init_from)

    def test_mf_initializes_for_60000_new_updates(self):
        args = self.preview("train_flow_stages.sh", "meanflow", INIT_FROM="fixed/step240000.pth")
        self.assertEqual((args.flow_objective, args.train_iters, args.meanflow_offdiag_ratio), ("meanflow", 60000, .75))
        self.assertEqual(args.init_from, "fixed/step240000.pth")
        self.assertIsNone(args.resume)
        self.assertEqual(args.milestone_ckpt_steps, (60000,))
        self.assertEqual((args.endpoint_loss_weight, args.ivc_loss_weight), (0, .5))

    def test_fm_offline_evaluation_uses_fm(self):
        args = self.preview("train.sh", EVAL_ONLY="true", CHECKPOINT="fixed/step300000.pth", FLOW_OBJECTIVE="fm")
        self.assertTrue(args.eval_only)
        self.assertEqual(args.flow_objective, "fm")
        self.assertEqual(args.checkpoint, "fixed/step300000.pth")
        self.assertIsNone(args.resume)

    def test_missing_mf_source_fails_before_launch(self):
        result = subprocess.run([self.bash, "train_flow_stages.sh", "meanflow"], cwd=self.root,
                                env=self.env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("INIT_FROM", result.stderr)

    def test_invalid_milestones_fail_at_parse(self):
        for steps in ("0", "-2", "oops", "300001"):
            with self.subTest(steps=steps), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(["--train_iters", "300000", "--milestone_ckpt_steps", steps])

    def test_actual_validation_exports_milestone_and_eval_reports(self):
        # Run the real validation method with one CPU sample and fixed metrics;
        # neither CLIP nor the simulator is needed to verify report persistence.
        tree = ast.parse((self.root / "utils/trainers/base.py").read_text(encoding="utf-8"))
        actor = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        method = next(node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate_nsteps")
        accumulator = Mock()
        accumulator.task_samples = {"test_task": 1}
        accumulator.summarize.return_value = {"val-loss/test_task/traj_score": .4}
        namespace = dict(torch=torch, json=json, dist=Mock(get_rank=lambda: 0),
                         ActionValidation=lambda: accumulator, tqdm=lambda x: x,
                         compute_task_balanced_selection=lambda *a, **k: (.4, .4, .4))
        exec(compile(ast.Module(body=[method], type_ignores=[]), "validation", "exec"), namespace)
        sample = {"action": torch.zeros(1, 1, 1, 8),
                  "proprioception": torch.zeros(1, 3, 1, 8), "task": ["test_task"]}
        with tempfile.TemporaryDirectory() as tmp, patch.object(torch.Tensor, "cuda", lambda self, **kw: self):
            args = parse_arguments(["--flow_objective", "fm", "--train_iters", "300000",
                                    "--milestone_ckpt_steps", "240000,300000", "--use_ema", "true"])
            args.log_dir = Path(tmp)
            args.validation_probe_batches = 0
            args.checkpoint = "fixed/step240000.pth"
            trainer = SimpleNamespace(args=args, writer=Mock(), _model_forward=lambda *a, **kw: sample["action"])
            model = torch.nn.Linear(1, 1)
            for index, name in ((239999, "validation_step240000.json"), (-1, "evaluation.json")):
                namespace["evaluate_nsteps"](trainer, model, [sample], index, -1)
                report = json.loads((Path(tmp) / name).read_text(encoding="utf-8"))
                self.assertEqual((report["flow_objective"], report["weights"]), ("fm", "ema"))
                self.assertEqual(report["selection_score"], .4)
                self.assertEqual(report["checkpoint"], "step240000.pth" if index >= 0 else args.checkpoint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
