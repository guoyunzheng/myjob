"""Utilities for loading model weights without silently changing architectures."""

from collections.abc import Mapping


def extract_weight_state(checkpoint):
    """Return the model state mapping from a training or weights-only file."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "Checkpoint must be a mapping, "
            f"but received {type(checkpoint).__name__}."
        )
    state = checkpoint.get("weight", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint does not contain a non-empty model state.")
    return state


def align_module_prefix(state, expected_keys):
    """Align a DDP ``module.`` prefix with the receiving model."""
    state = dict(state)
    expected_keys = tuple(expected_keys)
    state_keys = tuple(state)
    if not state_keys or not expected_keys:
        return state

    state_prefixed = all(key.startswith("module.") for key in state_keys)
    expected_prefixed = all(key.startswith("module.") for key in expected_keys)
    if state_prefixed and not expected_prefixed:
        return {key.removeprefix("module."): value for key, value in state.items()}
    if expected_prefixed and not state_prefixed:
        return {f"module.{key}": value for key, value in state.items()}
    return state


def load_model_state_strict(
    model,
    checkpoint,
    checkpoint_name="checkpoint",
    allowed_missing_prefixes=(),
):
    """Load weights and reject every unapproved architecture difference.

    ``allowed_missing_prefixes`` is reserved for explicitly zero-initialized,
    behavior-preserving upgrade layers. It must never be used for a prediction
    head or another layer whose random initialization would alter inference.
    """
    state = extract_weight_state(checkpoint)
    state = align_module_prefix(state, model.state_dict().keys())
    try:
        incompatible = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{checkpoint_name} is incompatible with the requested model architecture. "
            "Refusing to evaluate or resume with randomly initialized missing layers.\n"
            f"{exc}"
        ) from exc

    def allowed(key):
        key = key.removeprefix("module.")
        return any(key.startswith(prefix) for prefix in allowed_missing_prefixes)

    missing = [key for key in incompatible.missing_keys if not allowed(key)]
    if missing or incompatible.unexpected_keys:
        details = []
        if missing:
            details.append(f"Missing keys: {missing}")
        if incompatible.unexpected_keys:
            details.append(f"Unexpected keys: {incompatible.unexpected_keys}")
        raise RuntimeError(
            f"{checkpoint_name} is incompatible with the requested model architecture. "
            "Refusing to evaluate or resume with randomly initialized layers.\n"
            + "\n".join(details)
        )

    approved_missing = [key for key in incompatible.missing_keys if allowed(key)]
    if approved_missing:
        print(
            "Initialized behavior-preserving precision upgrade layers: "
            + ", ".join(approved_missing)
        )
    return state
