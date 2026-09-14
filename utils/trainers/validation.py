"""Sample-weighted offline action validation for training."""

from collections import defaultdict

import numpy as np
import torch

from .utils import action_metric_values


class ActionValidation:
    """Aggregate task and phase metrics without batch-size weighting bias."""

    def __init__(self):
        self.totals = defaultdict(lambda: [0.0, 0])
        self.error_samples = defaultdict(list)
        self.task_samples = defaultdict(int)

    def update(self, prediction, target, current_pose, tasks):
        if prediction.shape != target.shape or prediction.shape[-1] != 8:
            raise ValueError(
                "Expected matching quaternion actions (..., 8); got "
                f"{tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        if len(tasks) != len(prediction):
            raise ValueError("One task label is required per validation sample.")
        tensors = (prediction, target, current_pose)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise ValueError("Non-finite action or proprioception in validation.")
        if (
            (prediction[..., 3:7].norm(dim=-1) < 1e-6).any()
            or (target[..., 3:7].norm(dim=-1) < 1e-6).any()
        ):
            raise ValueError("Zero quaternion in validation.")
        if (
            (prediction[..., -1] < 0).any()
            or (prediction[..., -1] > 1).any()
        ):
            raise ValueError("Gripper probability outside [0, 1].")

        prediction = prediction.float()
        target = target.float()
        current_pose = current_pose.float()
        current_open = current_pose[..., -1:] >= 0.5
        raw = action_metric_values(prediction, target, current_open.float())
        position_ok = raw["pos_l2"] < 0.005
        rotation_ok = raw["rot_deg"] < 5.0
        pose_ok = position_ok & rotation_ok
        target_open = target[..., -1] >= 0.5
        current_open = current_open[..., 0].expand_as(target_open)

        # Match Mover's default 0.25/0.75 state-dependent hysteresis.
        gripper_command = torch.where(
            current_open,
            prediction[..., -1] > 0.25,
            prediction[..., -1] >= 0.75,
        )
        raw.update(
            pose_acc_005_5deg=pose_ok.float(),
            pose_acc_001_5deg=(
                (raw["pos_l2"] < 0.01) & rotation_ok
            ).float(),
            action_acc_005_5deg=(pose_ok & (raw["gripper"] > 0)).float(),
            action_acc_hysteresis=(
                pose_ok & (gripper_command == target_open)
            ).float(),
            gripper_hysteresis=(gripper_command == target_open).float(),
            gripper_hold_baseline=(current_open == target_open).float(),
            pos_hold_baseline=(
                current_pose[..., :3] - target[..., :3]
            ).norm(dim=-1),
        )
        phase_masks = {
            "grasp": current_open & ~target_open,
            "release": ~current_open & target_open,
            "closed_hold": ~current_open & ~target_open,
        }
        metrics = {
            name: (value, torch.ones_like(value, dtype=torch.bool))
            for name, value in raw.items()
        }
        phase_names = (
            "pos_l2",
            "rot_deg",
            "pose_acc_005_5deg",
            "action_acc_hysteresis",
            "gripper_hysteresis",
        )
        for phase, mask in phase_masks.items():
            for name in phase_names:
                metrics[f"{phase}_{name}"] = (raw[name], mask)

        task_array = np.asarray(tasks)
        unique_tasks = np.unique(task_array)
        for task in unique_tasks:
            self.task_samples[str(task)] += int((task_array == task).sum())

        metric_names = list(metrics)
        values = torch.stack([
            metrics[name][0].expand_as(pose_ok) for name in metric_names
        ]).detach().cpu().numpy()
        valid = torch.stack([
            metrics[name][1].expand_as(pose_ok) for name in metric_names
        ]).cpu().numpy()
        for index, name in enumerate(metric_names):
            for task in (None, *unique_tasks):
                rows = slice(None) if task is None else task_array == task
                selected = values[index, rows][valid[index, rows]]
                total = self.totals[(task, name)]
                total[0] += float(selected.sum(dtype=np.float64))
                total[1] += selected.size
            if name in ("pos_l2", "rot_deg"):
                self.error_samples[name].append(values[index].reshape(-1))

    def summarize(self, split):
        result = {}
        for (task, name), (total, count) in self.totals.items():
            prefix = (
                f"{split}-losses/mean"
                if task is None
                else f"{split}-loss/{task}"
            )
            if count:
                result[f"{prefix}/traj_{name}"] = total / count
            if name.endswith("_action_acc_hysteresis"):
                result[f"{prefix}/count_{name}"] = count
        for name, chunks in self.error_samples.items():
            errors = np.concatenate(chunks)
            for quantile in (50, 90, 95):
                result[f"{split}-diagnostics/{name}_p{quantile}"] = float(
                    np.percentile(errors, quantile)
                )
        result[f"{split}-diagnostics/samples"] = sum(
            self.task_samples.values()
        )
        result[f"{split}-diagnostics/tasks"] = len(self.task_samples)
        for task, count in self.task_samples.items():
            result[f"{split}-coverage/{task}"] = count
        return result


def noise_sensitivity(predictions):
    """Measure output spread across repeated inference noise samples."""
    pairs = [
        action_metric_values(left, right)
        for index, left in enumerate(predictions)
        for right in predictions[index + 1:]
    ]
    if not pairs:
        return {}
    return {
        "position_spread_m": torch.stack([
            pair["pos_l2"].mean() for pair in pairs
        ]).mean(),
        "rotation_spread_deg": torch.stack([
            pair["rot_deg"].mean() for pair in pairs
        ]).mean(),
        "gripper_disagreement": 1.0 - torch.stack([
            pair["gripper"].mean() for pair in pairs
        ]).mean(),
    }
