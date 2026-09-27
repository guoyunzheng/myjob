"""Shared RLBench input convention: world xyz, xyzw quaternion, gripper open.

History is chronological (oldest -> current); the final timestep is current.
Helpers preserve dtype/device and never normalize, reorder hands or change grip.
"""

INPUT_CONTRACT_VERSION = 1


def validate_action_config(num_history, rotation_format, relative_action):
    if type(num_history) is not int or num_history < 1:
        raise ValueError("denoise3d num_history must be positive; 0 is not a supported no-history mode.")
    if rotation_format != "quat_xyzw":
        raise ValueError("denoise3d RLBench inputs require rotation_format=quat_xyzw. "
                         "Other action formats are not aligned with the observation encoder.")
    if relative_action:
        raise ValueError("denoise3d relative_action is not supported end-to-end: "
                         "dataset conversion, quaternion validation and online world-pose execution "
                         "are not aligned. Use relative_action=false.")


def select_proprio_history(proprio, num_history, *, pad=False):
    """Select the newest N states, keeping chronology; optionally left-pad.

    Training has a fixed stored history and must not silently fabricate missing
    timesteps. Online episode startup may repeat its earliest available state.
    Accept (B, T, D) and (B, T, nhand, D).
    """
    if type(num_history) is not int or num_history < 1:
        raise ValueError("num_history must be a positive integer.")
    if proprio.ndim not in (3, 4) or proprio.shape[1] < 1:
        raise ValueError("Proprioception needs a non-empty history axis at dimension 1.")
    history = proprio[:, -num_history:]
    missing = num_history - history.shape[1]
    if missing:
        if not pad:
            raise ValueError(f"Not enough proprio timesteps: requested {num_history}, got {proprio.shape[1]}.")
        import torch

        prefix = history[:, :1].expand(-1, missing, *history.shape[2:])
        history = torch.cat((prefix, history), dim=1)
    return history


def current_proprio(proprio):
    """Keep a singleton time axis so batch and hand axes cannot be confused."""
    return select_proprio_history(proprio, 1)


def validate_proprio_shape(proprio, *, num_history, nhand):
    if (proprio.ndim != 4 or proprio.shape[1] != num_history
            or proprio.shape[2] != nhand or proprio.shape[3] != 8):
        raise ValueError(
            f"Expected proprio (B, {num_history}, {nhand}, 8) in oldest-to-current "
            f"order, with xyz+quat_xyzw+gripper_open; got {tuple(proprio.shape)}."
        )
