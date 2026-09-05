import math

import torch


def compute_task_balanced_selection(task_scores, worst_fraction=0.25):
    """Return macro, worst-slice and combined checkpoint scores.

    Scores are losses, so lower is better. The worst-task component prevents
    high-volume or easy tasks from hiding a model that never becomes precise on
    insertion and stacking tasks.
    """
    if not task_scores:
        raise ValueError("task_scores must not be empty.")
    if not 0.0 < worst_fraction <= 1.0:
        raise ValueError("worst_fraction must be in (0, 1].")
    scores = [float(score) for score in task_scores]
    macro_score = sum(scores) / len(scores)
    num_worst = max(1, math.ceil(worst_fraction * len(scores)))
    worst_score = sum(sorted(scores, reverse=True)[:num_worst]) / num_worst
    combined_score = 0.75 * macro_score + 0.25 * worst_score
    return macro_score, worst_score, combined_score


def compute_metrics(pred, gt, current_openess=None):
    # pred/gt are (B, L, 3+rot+1)
    pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
    # symmetric quaternion eval
    quat_l1 = (pred[..., 3:-1] - gt[..., 3:-1]).abs().sum(-1)
    quat_l1_ = (pred[..., 3:-1] + gt[..., 3:-1]).abs().sum(-1)
    select_mask = (quat_l1 < quat_l1_).float()
    quat_l1 = (select_mask * quat_l1 + (1 - select_mask) * quat_l1_)
    pred_quat = torch.nn.functional.normalize(pred[..., 3:-1], dim=-1)
    gt_quat = torch.nn.functional.normalize(gt[..., 3:-1], dim=-1)
    quat_dot = (pred_quat * gt_quat).sum(-1).abs().clamp(max=1.0)
    rot_rad = 2.0 * torch.acos(quat_dot)
    rot_deg = torch.rad2deg(rot_rad)
    # gripper openess
    pred_is_open = pred[..., -1:] >= 0.5
    gt_is_open = gt[..., -1:] >= 0.5
    openess = (pred_is_open == gt_is_open).bool()
    false_open = (pred_is_open & ~gt_is_open).squeeze(-1).float()
    false_close = (~pred_is_open & gt_is_open).squeeze(-1).float()
    # Releasing a carried object is usually irreversible, so checkpoint
    # selection penalizes false opening more than a delayed opening.
    score = pos_l2 + 0.01 * rot_rad + 0.05 * false_open + 0.01 * false_close
    closed_hold_false_open = None
    closed_hold_fraction = None
    if current_openess is not None:
        current_is_open = current_openess >= 0.5
        closed_hold = (~current_is_open & ~gt_is_open)
        closed_hold_false_open = (
            pred_is_open & closed_hold
        ).squeeze(-1).float()
        closed_hold_fraction = closed_hold.squeeze(-1).float()
    tr = 'traj_'

    # Trajectory metrics
    ret_1, ret_2 = {
        tr + 'pos_l2': pos_l2.mean(),
        tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(),
        tr + 'pos_acc_005': (pos_l2 < 0.005).float().mean(),
        tr + 'pos_acc_002': (pos_l2 < 0.002).float().mean(),
        tr + 'rot_l1': quat_l1.mean(),
        tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(),
        tr + 'rot_deg': rot_deg.mean(),
        tr + 'rot_acc_5deg': (rot_deg < 5.0).float().mean(),
        tr + 'gripper': openess.flatten().float().mean(),
        tr + 'gripper_false_open': false_open.mean(),
        tr + 'gripper_false_close': false_close.mean(),
        tr + 'score': score.mean(),
    }, {
        tr + 'pos_l2': pos_l2.mean(-1),
        tr + 'pos_acc_001': (pos_l2 < 0.01).float().mean(-1),
        tr + 'pos_acc_005': (pos_l2 < 0.005).float().mean(-1),
        tr + 'pos_acc_002': (pos_l2 < 0.002).float().mean(-1),
        tr + 'rot_l1': quat_l1.mean(-1),
        tr + 'rot_acc_0025': (quat_l1 < 0.025).float().mean(-1),
        tr + 'rot_deg': rot_deg.mean(-1),
        tr + 'rot_acc_5deg': (rot_deg < 5.0).float().mean(-1),
        tr + 'gripper': openess.flatten(-2).float().mean(-1),
        tr + 'gripper_false_open': false_open.mean(-1),
        tr + 'gripper_false_close': false_close.mean(-1),
        tr + 'score': score.mean(-1),
    }
    if closed_hold_false_open is not None:
        ret_1[tr + 'closed_hold_false_open'] = closed_hold_false_open.mean()
        ret_1[tr + 'closed_hold_fraction'] = closed_hold_fraction.mean()
        ret_2[tr + 'closed_hold_false_open'] = closed_hold_false_open.mean(-1)
        ret_2[tr + 'closed_hold_fraction'] = closed_hold_fraction.mean(-1)

    return ret_1, ret_2
