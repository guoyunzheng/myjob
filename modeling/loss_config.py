"""Explicit loss ablations; historical recipe remains the default, not a claim of quality."""
from dataclasses import dataclass, asdict
import math
from typing import Optional


@dataclass(frozen=True)
class LossConfig:
    pose_position_weight: float = 30.0
    pose_rotation_weight: float = 10.0
    gripper_loss_weight: float = 1.0
    gripper_loss_type: str = "weighted_bce"
    gripper_transition_weight: Optional[float] = None
    gripper_closed_hold_weight: Optional[float] = None
    gripper_hold_prior_logit: float = 2.0
    endpoint_loss_weight: float = 0.25
    ivc_loss_weight: float = 0.0

    def __post_init__(self):
        for name in ("gripper_transition_weight", "gripper_closed_hold_weight"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, 0.0 if self.gripper_loss_type == "bce" else 2.0)

    def validate(self):
        if self.gripper_loss_type not in ("bce", "weighted_bce"):
            raise ValueError("gripper_loss_type must be bce or weighted_bce.")
        for name, value in asdict(self).items():
            if name != "gripper_loss_type" and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.gripper_loss_type == "bce" and (self.gripper_transition_weight or self.gripper_closed_hold_weight):
            raise ValueError("Plain BCE requires transition/closed-hold weights=0; do not silently ignore them.")
        return self


EXTRA_LOSS_FIELDS = ("pose_position_weight", "pose_rotation_weight", "gripper_loss_weight", "gripper_loss_type")


def add_loss_arguments(parser):
    group = parser.add_argument_group("3D loss ablations (no automatic recipe replacement)")
    for name in EXTRA_LOSS_FIELDS[:3]:
        group.add_argument(f"--{name}", type=float, default=None)
    group.add_argument("--gripper_loss_type", choices=("bce", "weighted_bce"), default=None,
                       help="Default weighted_bce; bce defaults transition/closed-hold weights to zero.")


def normalize_loss_arguments(args, parser):
    if args.model_type != "denoise3d":
        if any(getattr(args, name, None) is not None for name in EXTRA_LOSS_FIELDS):
            parser.error("Loss ablations are implemented only for denoise3d.")
        return args
    try:
        config = LossConfig(**{name: getattr(args, name) for name in LossConfig.__dataclass_fields__
                               if getattr(args, name, None) is not None}).validate()
    except ValueError as error:
        parser.error(str(error))
    for name, value in asdict(config).items():
        setattr(args, name, value)
    return args


def extra_loss_model_kwargs(args):
    return {name: getattr(args, name) for name in EXTRA_LOSS_FIELDS}
