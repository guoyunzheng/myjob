"""Pure gripper-command helpers shared by RLBench evaluation and checks."""


def hysteresis_gripper_command(
    open_probability,
    current_state,
    close_threshold=0.25,
    open_threshold=0.75,
):
    """Convert an open probability to a discrete command with hysteresis.

    RLBench uses 0 for closed and 1 for open. Values inside the uncertainty
    band preserve the measured current state, preventing a probability close
    to 0.5 from releasing a carried object.
    """
    if not 0.0 <= close_threshold < open_threshold <= 1.0:
        raise ValueError(
            "Expected 0 <= close_threshold < open_threshold <= 1."
        )
    probability = float(open_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("open_probability must be in [0, 1].")
    state = 1.0 if float(current_state) >= 0.5 else 0.0
    if state == 0.0 and probability >= open_threshold:
        return 1.0
    if state == 1.0 and probability <= close_threshold:
        return 0.0
    return state
