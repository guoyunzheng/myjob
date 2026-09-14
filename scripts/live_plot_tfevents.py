"""Live plot TensorBoard scalars while a training job is running.

Edit EVENT_FILE_NAME below when a new training run creates a new event file,
then run:

    python scripts/live_plot_tfevents.py

The figure is refreshed periodically and is also saved as a PNG next to the
event file.  Command-line arguments can override all common settings.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

import matplotlib
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ---------------------------------------------------------------------------
# Usually this is the only line that needs to be changed for a new run.
EVENT_FILE_NAME = "events.out.tfevents.1781088099.adminpc-TU528V3.311155.0"
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = (
    PROJECT_ROOT
    / "train_logs"
    / "Peract"
    / "denoise3d-Peract-C120-B64-lr1e-4-constant-H3-meanflow"
)
DEFAULT_EVENT_FILE = LOG_DIR / EVENT_FILE_NAME

METRICS = (
    ("traj_pos_l2", "Position L2 error (lower is better)"),
    ("traj_rot_l1", "Rotation L1 error (lower is better)"),
    ("traj_pos_acc_001", "Position accuracy within 1 cm (higher is better)"),
    ("traj_rot_acc_0025", "Rotation accuracy (higher is better)"),
    ("traj_gripper", "Gripper accuracy (higher is better)"),
)

SPLITS = (
    ("train-losses", "Train", "tab:blue"),
    ("val-losses", "Validation", "tab:orange"),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously plot mean train/validation TensorBoard metrics."
    )
    parser.add_argument(
        "event_file",
        nargs="?",
        type=Path,
        default=DEFAULT_EVENT_FILE,
        help="TensorBoard event file (defaults to EVENT_FILE_NAME in this script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: live_training_curves.png beside event file).",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=10.0,
        help="Refresh interval in seconds (default: 10).",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="EMA smoothing factor in (0, 1]; 0 disables smoothing (default: 0).",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open a GUI window; only refresh the output PNG.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Draw once and exit instead of continuously monitoring the file.",
    )
    args = parser.parse_args()

    if not math.isfinite(args.refresh) or args.refresh <= 0:
        parser.error("--refresh must be finite and greater than 0")
    if not 0 <= args.smooth <= 1:
        parser.error("--smooth must be between 0 and 1")
    return args


def should_use_gui(no_window: bool) -> bool:
    """Return whether a display window is likely to be available."""
    if no_window:
        return False
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False
    return True


def load_mean_scalars(
    accumulator: EventAccumulator,
) -> dict[str, tuple[list[int], list[float]]]:
    """Load all mean metrics currently available in an event file."""
    accumulator.Reload()
    available_tags = set(accumulator.Tags().get("scalars", []))

    series = {}
    for metric, _ in METRICS:
        for prefix, _, _ in SPLITS:
            tag = f"{prefix}/mean/{metric}"
            if tag not in available_tags:
                continue
            events = accumulator.Scalars(tag)
            series[tag] = (
                [event.step for event in events],
                [event.value for event in events],
            )
    return series


def exponential_moving_average(values: list[float], alpha: float) -> list[float]:
    if not values or alpha == 0:
        return values
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1 - alpha) * result[-1])
    return result


def draw_figure(
    fig,
    axes,
    series,
    event_file: Path,
    output: Path,
    smooth: float,
    refresh: float,
):
    latest_step = -1

    for axis, (metric, title) in zip(axes, METRICS):
        axis.clear()
        has_data = False

        for prefix, split_label, color in SPLITS:
            tag = f"{prefix}/mean/{metric}"
            if tag not in series:
                continue

            steps, raw_values = series[tag]
            if not steps:
                continue

            has_data = True
            latest_step = max(latest_step, steps[-1])
            values = exponential_moving_average(raw_values, smooth)
            if smooth:
                axis.plot(steps, raw_values, color=color, alpha=0.20, linewidth=1)
            axis.plot(
                steps,
                values,
                color=color,
                marker="o",
                markersize=3,
                linewidth=2,
                label=f"{split_label}: {raw_values[-1]:.4f}",
            )

        axis.set_title(title)
        axis.set_xlabel("Training step")
        axis.set_ylabel("Value")
        axis.grid(True, alpha=0.25)
        if "accuracy" in title.lower():
            axis.set_ylim(-0.02, 1.02)
        if has_data:
            axis.legend(loc="best")
        else:
            axis.text(
                0.5,
                0.5,
                "Waiting for this metric...",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="gray",
            )

    # The sixth panel is reserved for live status information.
    status_axis = axes[-1]
    status_axis.clear()
    status_axis.axis("off")
    refreshed_at = time.strftime("%Y-%m-%d %H:%M:%S")
    status_axis.text(
        0.02,
        0.95,
        "Live training monitor\n\n"
        f"Latest step: {latest_step if latest_step >= 0 else 'waiting'}\n"
        f"Updated: {refreshed_at}\n"
        f"Refresh: every {refresh:g} s\n\n"
        f"Event file:\n{event_file.name}\n\n"
        f"Saved figure:\n{output}",
        transform=status_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        wrap=True,
    )

    fig.suptitle("3D FlowMatch Actor - Live Training Metrics", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    return latest_step, refreshed_at


def main() -> int:
    args = parse_arguments()

    event_file = args.event_file.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else event_file.with_name("live_training_curves.png")
    )

    if not event_file.is_file():
        print(f"Event file does not exist: {event_file}", file=sys.stderr)
        print("Update EVENT_FILE_NAME at the top of this script, or pass a path.", file=sys.stderr)
        return 2

    use_gui = should_use_gui(args.no_window)
    if not use_gui:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if use_gui:
        plt.ion()

    fig, axes_grid = plt.subplots(3, 2, figsize=(14, 12))
    axes = list(axes_grid.flat)
    try:
        fig.canvas.manager.set_window_title("Live TensorBoard training curves")
    except AttributeError:
        pass

    print(f"Monitoring: {event_file}")
    print(f"Saving PNG: {output}")
    print(f"Refresh interval: {args.refresh:g} seconds")
    print("Press Ctrl+C to stop.")

    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    previous_series = None
    try:
        while True:
            try:
                series = load_mean_scalars(accumulator)
                if series != previous_series:
                    latest_step, refreshed_at = draw_figure(
                        fig,
                        axes,
                        series,
                        event_file,
                        output,
                        args.smooth,
                        args.refresh,
                    )
                    print(
                        f"[{refreshed_at}] Updated at step "
                        f"{latest_step if latest_step >= 0 else 'waiting'}"
                    )
                    previous_series = series

                if args.once:
                    break
                if use_gui:
                    plt.show(block=False)
                    plt.pause(args.refresh)
                    if not plt.fignum_exists(fig.number):
                        print("Plot window closed; monitor stopped.")
                        break
                else:
                    time.sleep(args.refresh)
            except (OSError, RuntimeError, ValueError) as error:
                # A writer may briefly leave an incomplete event record while it
                # is appending.  Keep monitoring and try again on the next tick.
                print(f"Read/update failed, retrying: {error}", file=sys.stderr)
                if args.once:
                    return 1
                time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nMonitor stopped by user.")
    finally:
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
