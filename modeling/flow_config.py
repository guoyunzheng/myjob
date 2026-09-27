"""Dependency-free configuration for explicit action heads and flow objectives."""

from dataclasses import asdict, dataclass
import math
from typing import Optional
from utils.action_contract import INPUT_CONTRACT_VERSION, validate_action_config


@dataclass(frozen=True)
class FlowConfig:
    action_head: str = "film_tcn"
    flow_objective: str = "meanflow"
    # Action head only: FiLM-TCN has no attention; the encoder is unchanged.
    attention_backend: str = "auto"
    time_sampler: str = "logit_normal"
    time_sampler_mean: float = 0.0
    time_sampler_std: float = 1.5
    meanflow_offdiag_ratio: Optional[float] = None
    flow_loss_type: Optional[str] = None

    def __post_init__(self):
        if self.flow_loss_type is None:
            object.__setattr__(self, "flow_loss_type",
                               "l2" if self.flow_objective == "imf" else "l1")
        if self.meanflow_offdiag_ratio is None:
            object.__setattr__(self, "meanflow_offdiag_ratio",
                               0.0 if self.flow_objective == "fm" else 0.25)

    def validate(self, require_implemented=True):
        choices = {
            "action_head": ("film_tcn", "transformer"),
            "flow_objective": ("fm", "meanflow", "imf"),
            "attention_backend": ("auto", "math"),
            "time_sampler": ("logit_normal", "uniform"),
            "flow_loss_type": ("l1", "l2"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}.")
        if not math.isfinite(self.time_sampler_mean):
            raise ValueError("time_sampler_mean must be finite.")
        if not math.isfinite(self.time_sampler_std) or self.time_sampler_std <= 0:
            raise ValueError("time_sampler_std must be finite and positive.")
        if not 0.0 <= self.meanflow_offdiag_ratio <= 1.0:
            raise ValueError("meanflow_offdiag_ratio must be in [0, 1].")
        if self.flow_objective == "fm" and self.meanflow_offdiag_ratio != 0:
            raise ValueError("FM always uses r=t; meanflow_offdiag_ratio must be 0.")
        return self

    def validate_conditioning(self, guidance_scale=1.0, condition_dropout_prob=0.0):
        # Step 9 implements the conditional boundary-reuse objective only, not
        # the paper's guidance-conditioned architecture or CFG distillation.
        if self.flow_objective == "imf" and (guidance_scale != 1.0 or condition_dropout_prob != 0.0):
            raise ValueError("iMF currently requires guidance_scale=1 and condition_dropout_prob=0; flexible CFG is not implemented.")
        return self


FLOW_CONFIG_FIELDS = tuple(FlowConfig.__dataclass_fields__)


def resolve_flow_config(*, denoise_model=None, **values):
    # Do not reinterpret legacy RF runs as genuine FM training.
    if denoise_model not in (None, "meanflow", "fm", "imf"):
        raise ValueError(
            f"denoise_model={denoise_model!r} only selects a legacy scheduler; "
            "the denoise3d actor does not implement that training objective. "
            "Use flow_objective=fm, meanflow or imf explicitly."
        )
    if denoise_model is not None:
        if values.get("flow_objective") not in (None, denoise_model):
            raise ValueError("Conflicting denoise_model and flow_objective; use the explicit objective alone.")
        values["flow_objective"] = denoise_model
    return FlowConfig(**{k: v for k, v in values.items() if v is not None}).validate()


def add_flow_arguments(parser):
    group = parser.add_argument_group("3D action architecture and flow objective")
    group.description = (
        "Both heads use action_hidden_dim/action_num_blocks. Transformer uses "
        "num_attn_heads and FP32 math attention; num_shared_attn_layers is a legacy decoder option."
    )
    group.add_argument("--action_head", choices=("film_tcn", "transformer"),
                       help="Default: film_tcn; transformer uses deterministic token cross-attention.")
    group.add_argument("--flow_objective", choices=("fm", "meanflow", "imf"),
                       help="Default: meanflow; imf opts into boundary-reuse Improved MeanFlow (no flexible CFG).")
    group.add_argument("--attention_backend", choices=("auto", "math"),
                       help="Action head only: transformer auto/math both use math; no effect on TCN or encoder.")
    group.add_argument("--time_sampler", choices=("logit_normal", "uniform"))
    group.add_argument("--time_sampler_mean", type=float)
    group.add_argument("--time_sampler_std", type=float)
    group.add_argument("--meanflow_offdiag_ratio", type=float,
                       help="Default: 0.25 for MeanFlow/iMF, 0 for FM. FM requires 0.")
    group.add_argument("--flow_loss_type", choices=("l1", "l2"),
                       help="Base pose-flow regression only. Default: l1 for FM/MeanFlow, l2 for iMF; use the same metric in comparisons.")


def normalize_flow_arguments(args, parser):
    values = {name: getattr(args, name, None) for name in FLOW_CONFIG_FIELDS}
    if args.model_type != "denoise3d":
        if args.denoise_model in ("fm", "imf"):
            parser.error("The new FM/iMF objectives are implemented only for denoise3d.")
        if any(value is not None for value in values.values()):
            parser.error("Independent flow configuration currently applies only to denoise3d.")
        if args.denoise_model is None:
            args.denoise_model = "meanflow"  # Preserve historical CLI default.
        return args
    try:
        config = resolve_flow_config(denoise_model=args.denoise_model, **values)
        config.validate_conditioning(getattr(args, "guidance_scale", 1.0),
                                     getattr(args, "condition_dropout_prob", 0.0))
        if config.flow_objective == "imf" and getattr(args, "use_compile", False):
            raise ValueError("iMF currently requires use_compile=false; dynamic off-diagonal JVP chunking is not fullgraph-compatible.")
        validate_action_config(args.num_history, args.rotation_format, args.relative_action)
        if config.action_head == "transformer":
            hidden, heads = args.action_hidden_dim, args.num_attn_heads
            if hidden < 4 or hidden % 2 or heads < 1 or hidden % heads:
                raise ValueError("Transformer action_hidden_dim must be even, >=4, and divisible by num_attn_heads.")
            if args.action_num_blocks < 1:
                raise ValueError("Transformer action_num_blocks must be positive.")
        if config.flow_objective == "fm" and getattr(args, "ivc_loss_weight", 0.) != 0:
            raise ValueError("FM already supervises instantaneous velocity; set ivc_loss_weight=0.")
    except ValueError as error:
        parser.error(str(error))
    for name, value in asdict(config).items():
        setattr(args, name, value)
    args.denoise_model = config.flow_objective  # Canonical value for checkpoint/log consumers.
    args.input_contract_version = INPUT_CONTRACT_VERSION
    return args


def flow_model_kwargs(args):
    return {name: getattr(args, name, None) for name in FLOW_CONFIG_FIELDS}


def configure_action_precision(action_head, matmul_precision="legacy"):
    """Apply once at train/eval startup, before any forward or backward.

    FP32 tensors alone do not disable CUDA TF32 matmul. Do not toggle global
    backend state inside JVP forwards. TCN/legacy settings stay untouched unless
    the caller explicitly requests IEEE precision for a controlled comparison.
    Parsing remains dependency-free; torch is imported only for execution.
    """
    if matmul_precision not in ("legacy", "ieee"):
        raise ValueError("matmul_precision must be legacy or ieee.")
    if action_head == "transformer" or matmul_precision == "ieee":
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        if matmul_precision == "ieee":
            torch.backends.cudnn.allow_tf32 = False  # TCN convolutions also participate in the comparison.
        print(f"Action head={action_head}: CUDA matmul TF32 disabled (FP32 GEMM).")
