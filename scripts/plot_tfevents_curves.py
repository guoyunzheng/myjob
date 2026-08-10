import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


HIGHER_BETTER = ("traj_pos_acc_001", "traj_rot_acc_0025")
LOWER_BETTER = ("traj_pos_l2", "traj_rot_l1", "traj_gripper")


def load_scalars(event_file):
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    data = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        data[tag] = {
            "steps": [event.step for event in events],
            "values": [event.value for event in events],
        }
    return data


def metric_name(tag):
    return tag.rsplit("/", 1)[-1]


def split_name(tag):
    return tag.split("/", 1)[0]


def task_name(tag):
    parts = tag.split("/")
    if len(parts) >= 3:
        return parts[1]
    return ""


def is_mean_tag(tag):
    return "/mean/" in tag


def plot_mean_metrics(data, metrics, title, output):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 3.2 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for axis, metric in zip(axes, metrics):
        matching = [
            tag for tag in data
            if is_mean_tag(tag) and metric_name(tag) == metric
        ]
        for tag in sorted(matching):
            axis.plot(data[tag]["steps"], data[tag]["values"], linewidth=2, label=split_name(tag))
        axis.set_title(metric)
        axis.set_ylabel("value")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

    axes[-1].set_xlabel("step")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_task_metrics(data, metrics, title, output):
    rows = len(metrics)
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.2 * rows), sharex=False)
    if rows == 1:
        axes = [axes]

    for row, metric in enumerate(metrics):
        for col, prefix in enumerate(("train-loss", "val-loss")):
            axis = axes[row][col]
            matching = [
                tag for tag in data
                if tag.startswith(prefix + "/")
                and not is_mean_tag(tag)
                and metric_name(tag) == metric
            ]
            for tag in sorted(matching):
                axis.plot(
                    data[tag]["steps"],
                    data[tag]["values"],
                    linewidth=1.2,
                    alpha=0.78,
                    label=task_name(tag),
                )
            axis.set_title(f"{prefix} / {metric}")
            axis.set_xlabel("step")
            axis.set_ylabel("value")
            axis.grid(True, alpha=0.22)
            if matching:
                axis.legend(
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.5),
                    fontsize=7,
                    frameon=False,
                    ncol=1,
                )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.86, 0.97))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_summary(data, output):
    rows = ["tag,points,first_step,last_step,first_value,last_value,best_value"]
    for tag in sorted(data):
        values = data[tag]["values"]
        steps = data[tag]["steps"]
        metric = metric_name(tag)
        if metric in HIGHER_BETTER:
            best = max(values)
        else:
            best = min(values)
        rows.append(
            f"{tag},{len(values)},{steps[0]},{steps[-1]},"
            f"{values[0]:.8g},{values[-1]:.8g},{best:.8g}"
        )
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("event_file", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("fig/tfevents_1783573763"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_scalars(args.event_file)

    plot_mean_metrics(
        data,
        HIGHER_BETTER,
        "Mean metrics - higher is better",
        args.out_dir / "mean_higher_better.png",
    )
    plot_mean_metrics(
        data,
        LOWER_BETTER,
        "Mean metrics - lower is better",
        args.out_dir / "mean_lower_better.png",
    )
    plot_task_metrics(
        data,
        HIGHER_BETTER,
        "Per-task metrics - higher is better",
        args.out_dir / "tasks_higher_better.png",
    )
    plot_task_metrics(
        data,
        ("traj_pos_l2", "traj_rot_l1"),
        "Per-task metrics - lower is better",
        args.out_dir / "tasks_lower_better.png",
    )
    write_summary(data, args.out_dir / "scalar_summary.csv")

    print(f"Wrote plots and summary to {args.out_dir.resolve()}")
    print(f"Scalar tags: {len(data)}")


if __name__ == "__main__":
    main()
