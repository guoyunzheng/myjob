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


def action_metric_values(pred, gt, current_openess=None):
    # pred/gt are (B, L, 8) or (B, L, nhand, 8).
    pos_l2 = ((pred[..., :3] - gt[..., :3]) ** 2).sum(-1).sqrt()
    # symmetric quaternion eval
    quat_l1 = (pred[..., 3:-1] - gt[..., 3:-1]).abs().sum(-1)
    quat_l1_ = (pred[..., 3:-1] + gt[..., 3:-1]).abs().sum(-1)
    quat_l1 = torch.minimum(quat_l1, quat_l1_)
    pred_quat = torch.nn.functional.normalize(pred[..., 3:-1], dim=-1)
    gt_quat = torch.nn.functional.normalize(gt[..., 3:-1], dim=-1)
    quat_dot = (pred_quat * gt_quat).sum(-1).abs().clamp(max=1.0)
    rot_rad = 2.0 * torch.acos(quat_dot)
    rot_deg = torch.rad2deg(rot_rad)
    # gripper openess
    pred_is_open = pred[..., -1:] >= 0.5
    gt_is_open = gt[..., -1:] >= 0.5
    false_open = (pred_is_open & ~gt_is_open).squeeze(-1).float()
    false_close = (~pred_is_open & gt_is_open).squeeze(-1).float()
    # Releasing a carried object is usually irreversible, so checkpoint
    # selection penalizes false opening more than a delayed opening.
    score = pos_l2 + 0.01 * rot_rad + 0.05 * false_open + 0.01 * false_close
    metrics = {
        'pos_l2': pos_l2,
        'pos_acc_001': (pos_l2 < 0.01).float(),
        'pos_acc_005': (pos_l2 < 0.005).float(),
        'pos_acc_002': (pos_l2 < 0.002).float(),
        'rot_l1': quat_l1,
        'rot_acc_0025': (quat_l1 < 0.025).float(),
        'rot_deg': rot_deg,
        'rot_acc_5deg': (rot_deg < 5.0).float(),
        'gripper': (pred_is_open == gt_is_open).squeeze(-1).float(),
        'gripper_false_open': false_open,
        'gripper_false_close': false_close,
        'score': score,
    }
    if current_openess is not None:
        closed_hold = ~(current_openess >= 0.5) & ~gt_is_open
        metrics['closed_hold_false_open'] = (
            pred_is_open & closed_hold
        ).squeeze(-1).float()
        metrics['closed_hold_fraction'] = closed_hold.squeeze(-1).float()

    return metrics


def compute_metrics(pred, gt, current_openess=None):
    metrics = action_metric_values(pred, gt, current_openess)
    return (
        {f'traj_{name}': value.mean() for name, value in metrics.items()},
        {
            f'traj_{name}': value.flatten(1).mean(-1)
            for name, value in metrics.items()
        },
    )
