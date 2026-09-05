"""Diffusion-independent gripper-state prediction."""

import torch
from torch import nn


class GripperStateHead(nn.Module):
    """Predict discrete gripper states from observation and step identity.

    The residual is initialized to zero, so an untrained model holds the
    measured current state instead of opening unpredictably.
    """

    def __init__(
        self,
        condition_dim=256,
        hidden_dim=256,
        hold_prior_logit=2.0,
        max_prediction_len=64,
    ):
        super().__init__()
        if max_prediction_len < 1:
            raise ValueError("max_prediction_len must be positive.")
        self.max_prediction_len = max_prediction_len
        step_embedding_dim = min(32, hidden_dim)
        self.register_buffer(
            "hold_prior_logit",
            torch.tensor(float(hold_prior_logit), dtype=torch.float32),
        )
        self.step_embedding = nn.Embedding(
            max_prediction_len, step_embedding_dim
        )
        self.network = nn.Sequential(
            nn.LayerNorm(condition_dim + step_embedding_dim + 1),
            nn.Linear(
                condition_dim + step_embedding_dim + 1, hidden_dim
            ),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, condition, current_openess, traj_len=1):
        if condition.ndim != 2:
            raise ValueError(
                "condition must have shape (B, D); got "
                f"{tuple(condition.shape)}."
            )
        if current_openess.ndim != 3 or current_openess.shape[-1] != 1:
            raise ValueError(
                "current_openess must have shape (B, nhand, 1); got "
                f"{tuple(current_openess.shape)}."
            )
        if not 1 <= traj_len <= self.max_prediction_len:
            raise ValueError(
                f"traj_len must be in [1, {self.max_prediction_len}]; "
                f"got {traj_len}."
            )
        nhand = current_openess.shape[1]
        batch_size = condition.shape[0]
        condition = condition[:, None, None].expand(
            -1, traj_len, nhand, -1
        )
        current_openess = current_openess[:, None].expand(
            -1, traj_len, -1, -1
        )
        step_ids = torch.arange(traj_len, device=condition.device)
        step_features = self.step_embedding(step_ids)[None, :, None].expand(
            batch_size, -1, nhand, -1
        )
        residual = self.network(
            torch.cat((condition, step_features, current_openess), dim=-1)
        )
        current_is_open = current_openess >= 0.5
        prior = torch.where(
            current_is_open,
            self.hold_prior_logit.to(dtype=residual.dtype),
            -self.hold_prior_logit.to(dtype=residual.dtype),
        )
        return prior + residual
