import importlib.util
from pathlib import Path

import torch


_MODULE_PATH = Path(__file__).parents[1] / "utils" / "trainers" / "utils.py"
_SPEC = importlib.util.spec_from_file_location("trainer_metrics", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
compute_metrics = _MODULE.compute_metrics
compute_task_balanced_selection = _MODULE.compute_task_balanced_selection


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


if __name__ == "__main__":
    check_quaternion_sign_has_zero_rotation_error()
    check_selection_score_penalizes_pose_and_gripper_errors()
    check_false_open_is_penalized_more_than_false_close()
    check_closed_hold_false_open_is_reported()
    check_task_balanced_selection_keeps_worst_tasks_visible()
    print("action metric checks passed")
