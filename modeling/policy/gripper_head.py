"""Diffusion-independent gripper-state prediction."""

import torch
from torch import nn


class GripperStateHead(nn.Module):
    """Predict the next discrete gripper state from observation state only.

    The residual is initialized to zero, so an untrained model holds the
    measured current state instead of opening unpredictably.
    """

    def __init__(self, condition_dim=256, hidden_dim=256, hold_prior_logit=2.0):
        super().__init__()
        self.register_buffer(
            "hold_prior_logit",
            torch.tensor(float(hold_prior_logit), dtype=torch.float32),
        )
        self.network = nn.Sequential(
            nn.LayerNorm(condition_dim + 1),
            nn.Linear(condition_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, condition, current_openess):
        if current_openess.ndim != 3 or current_openess.shape[-1] != 1:
            raise ValueError(
                "current_openess must have shape (B, nhand, 1); got "
                f"{tuple(current_openess.shape)}."
            )
        nhand = current_openess.shape[1]
        condition = condition.unsqueeze(1).expand(-1, nhand, -1)
        residual = self.network(
            torch.cat((condition, current_openess), dim=-1)
        )
        current_is_open = current_openess >= 0.5
        prior = torch.where(
            current_is_open,
            self.hold_prior_logit.to(dtype=residual.dtype),
            -self.hold_prior_logit.to(dtype=residual.dtype),
        )
        return prior + residual
