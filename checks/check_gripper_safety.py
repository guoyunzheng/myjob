import importlib.util
from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).parents[1]))
from online_evaluation_rlbench.gripper_control import (  # noqa: E402
    hysteresis_gripper_command,
    should_execute_gripper_change,
)

_HEAD_PATH = Path(__file__).parents[1] / "modeling" / "policy" / "gripper_head.py"
_HEAD_SPEC = importlib.util.spec_from_file_location("gripper_head", _HEAD_PATH)
_HEAD_MODULE = importlib.util.module_from_spec(_HEAD_SPEC)
_HEAD_SPEC.loader.exec_module(_HEAD_MODULE)
GripperStateHead = _HEAD_MODULE.GripperStateHead


def check_hysteresis_holds_uncertain_commands():
    assert hysteresis_gripper_command(0.51, 0.0) == 0.0
    assert hysteresis_gripper_command(0.49, 1.0) == 1.0
    assert hysteresis_gripper_command(0.80, 0.0) == 1.0
    assert hysteresis_gripper_command(0.20, 1.0) == 0.0


def check_direct_head_initializes_to_hold_current_state():
    head = GripperStateHead(
        condition_dim=4,
        hidden_dim=8,
        hold_prior_logit=2.0,
    )
    condition = torch.zeros(2, 4)
    current = torch.tensor([[[0.0]], [[1.0]]])
    logits = head(condition, current, traj_len=3)
    assert logits.shape == (2, 3, 1, 1)
    torch.testing.assert_close(
        logits[:, :, 0, 0],
        torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]]),
    )


def check_only_opening_waits_for_pose():
    assert not should_execute_gripper_change(0.0, 1.0, False)
    assert should_execute_gripper_change(0.0, 1.0, True)
    assert should_execute_gripper_change(1.0, 0.0, False)
    assert not should_execute_gripper_change(1.0, 1.0, True)


if __name__ == "__main__":
    check_hysteresis_holds_uncertain_commands()
    check_direct_head_initializes_to_hold_current_state()
    check_only_opening_waits_for_pose()
    print("gripper safety checks passed")
