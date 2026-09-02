from pathlib import Path
import sys

import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.checkpoint_utils import load_model_state_strict  # noqa: E402


class TinyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.core = nn.Linear(2, 2)
        self.precision_upgrade = nn.Linear(2, 2)


def check_ddp_prefix_is_aligned_without_relaxing_key_checks():
    model = TinyModel()
    prefixed = {
        f"module.{key}": value.clone()
        for key, value in model.state_dict().items()
    }
    load_model_state_strict(model, prefixed)


def check_only_explicit_precision_upgrade_keys_may_be_missing():
    model = TinyModel()
    old_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if not key.startswith("precision_upgrade.")
    }
    load_model_state_strict(
        model,
        old_state,
        allowed_missing_prefixes=("precision_upgrade.",),
    )


def check_prediction_architecture_mismatch_is_rejected():
    model = TinyModel()
    bad_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if not key.startswith("core.")
    }
    try:
        load_model_state_strict(model, bad_state)
    except RuntimeError as error:
        assert "Refusing to evaluate or resume" in str(error)
    else:
        raise AssertionError("An incompatible prediction architecture was accepted.")


if __name__ == "__main__":
    check_ddp_prefix_is_aligned_without_relaxing_key_checks()
    check_only_explicit_precision_upgrade_keys_may_be_missing()
    check_prediction_architecture_mismatch_is_rejected()
    print("checkpoint loading checks passed")
