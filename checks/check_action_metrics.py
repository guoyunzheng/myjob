import ast
import importlib.util
from pathlib import Path

import torch


_MODULE_PATH = Path(__file__).parents[1] / "utils" / "trainers" / "utils.py"
_SPEC = importlib.util.spec_from_file_location("trainer_metrics", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
compute_metrics = _MODULE.compute_metrics
compute_task_balanced_selection = _MODULE.compute_task_balanced_selection

_VALIDATION_PATH = (
    Path(__file__).parents[1] / "utils" / "trainers" / "validation.py"
)
_VALIDATION_TREE = ast.parse(_VALIDATION_PATH.read_text(encoding="utf-8"))
_VALIDATION_TREE.body = [
    node for node in _VALIDATION_TREE.body
    if not isinstance(node, ast.ImportFrom) or node.level == 0
]
_VALIDATION_NAMESPACE = {
    "np": __import__("numpy"),
    "torch": torch,
    "defaultdict": __import__("collections").defaultdict,
    "action_metric_values": _MODULE.action_metric_values,
}
exec(
    compile(_VALIDATION_TREE, str(_VALIDATION_PATH), "exec"),
    _VALIDATION_NAMESPACE,
)
ActionValidation = _VALIDATION_NAMESPACE["ActionValidation"]
noise_sensitivity = _VALIDATION_NAMESPACE["noise_sensitivity"]


def _action(position, quaternion, openness):
    return torch.tensor(
        [*position, *quaternion, openness], dtype=torch.float32
    ).reshape(1, 1, 1, 8)


def check_quaternion_sign_has_zero_rotation_error():
    prediction = _action((0, 0, 0), (0, 0, 0, 1), 1)
    target = _action((0, 0, 0), (0, 0, 0, -1), 1)
    metrics, _ = compute_metrics(prediction, target)
    torch.testing.assert_close(metrics["traj_rot_deg"], torch.tensor(0.0))
    torch.testing.assert_close(metrics["traj_score"], torch.tensor(0.0))


def check_selection_score_penalizes_pose_and_gripper_errors():
    prediction = _action((0.01, 0, 0), (0, 0, 0, 1), 0)
    target = _action((0, 0, 0), (0, 0, 0, 1), 1)
    metrics, _ = compute_metrics(prediction, target)
    torch.testing.assert_close(metrics["traj_score"], torch.tensor(0.02))


def check_false_open_is_penalized_more_than_false_close():
    false_open = _action((0, 0, 0), (0, 0, 0, 1), 1)
    closed_target = _action((0, 0, 0), (0, 0, 0, 1), 0)
    false_close = _action((0, 0, 0), (0, 0, 0, 1), 0)
    open_target = _action((0, 0, 0), (0, 0, 0, 1), 1)
    open_metrics, _ = compute_metrics(false_open, closed_target)
    close_metrics, _ = compute_metrics(false_close, open_target)
    torch.testing.assert_close(
        open_metrics["traj_gripper_false_open"], torch.tensor(1.0)
    )
    assert open_metrics["traj_score"] > close_metrics["traj_score"]


def check_closed_hold_false_open_is_reported():
    prediction = _action((0, 0, 0), (0, 0, 0, 1), 1)
    target = _action((0, 0, 0), (0, 0, 0, 1), 0)
    current = torch.tensor([[[[0.0]]]])
    metrics, _ = compute_metrics(
        prediction, target, current_openess=current
    )
    torch.testing.assert_close(
        metrics["traj_closed_hold_false_open"], torch.tensor(1.0)
    )


def check_task_balanced_selection_keeps_worst_tasks_visible():
    macro, worst, combined = compute_task_balanced_selection([0.1, 0.1, 0.1, 0.9])
    assert abs(macro - 0.3) < 1e-8
    assert abs(worst - 0.9) < 1e-8
    assert abs(combined - 0.45) < 1e-8


def check_validation_is_sample_weighted_and_phase_aware():
    good = _action((0, 0, 0), (0, 0, 0, 1), 0)
    bad = _action((0.1, 0, 0), (0, 0, 0, 1), 1)
    current_open = _action((0, 0, 0), (0, 0, 0, 1), 1)
    validation = ActionValidation()
    validation.update(
        good.expand(2, -1, -1, -1),
        good.expand(2, -1, -1, -1),
        current_open.expand(2, -1, -1, -1),
        ["easy", "easy"],
    )
    validation.update(bad, good, current_open, ["hard"])
    metrics = validation.summarize("val")
    torch.testing.assert_close(
        torch.tensor(metrics["val-losses/mean/traj_action_acc_hysteresis"]),
        torch.tensor(2 / 3),
    )
    assert metrics["val-losses/mean/count_grasp_action_acc_hysteresis"] == 3
    assert metrics["val-coverage/easy"] == 2
    assert metrics["val-coverage/hard"] == 1


def check_noise_probe_reports_output_instability():
    left = _action((0, 0, 0), (0, 0, 0, 1), 0)
    right = _action((0.1, 0, 0), (0, 0, 0, 1), 1)
    metrics = noise_sensitivity([left, right])
    torch.testing.assert_close(
        metrics["position_spread_m"], torch.tensor(0.1)
    )
    torch.testing.assert_close(
        metrics["rotation_spread_deg"], torch.tensor(0.0)
    )
    torch.testing.assert_close(
        metrics["gripper_disagreement"], torch.tensor(1.0)
    )


if __name__ == "__main__":
    check_quaternion_sign_has_zero_rotation_error()
    check_selection_score_penalizes_pose_and_gripper_errors()
    check_false_open_is_penalized_more_than_false_close()
    check_closed_hold_false_open_is_reported()
    check_task_balanced_selection_keeps_worst_tasks_visible()
    check_validation_is_sample_weighted_and_phase_aware()
    check_noise_probe_reports_output_instability()
    print("action metric checks passed")
