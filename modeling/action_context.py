"""Fixed observations: batch slicing for JVP, with attached prediction gradients."""

from dataclasses import dataclass, fields
from typing import Optional

import torch


@dataclass(frozen=True)
class TokenGroup:
    """Named, aligned views; xyz is in world meters, True mask means padding."""
    tokens: torch.Tensor
    xyz: Optional[torch.Tensor] = None
    padding_mask: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class ActionContext:
    # Every tensor has a leading batch axis, including expanded bounds.
    global_condition: torch.Tensor
    scene_tokens: torch.Tensor
    scene_xyz: torch.Tensor
    language_tokens: torch.Tensor
    language_padding_mask: Optional[torch.Tensor]
    proprio_tokens: torch.Tensor
    workspace_bounds: torch.Tensor
    # Scene storage stays concatenated once, with zero-copy source views. This
    # preserves step-5 attention weights (including duplicated sampled points).
    dense_scene_count: Optional[int] = None
    proprio_xyz: Optional[torch.Tensor] = None
    # History-major, then hand-major: token index = history_index*nhand+hand.
    proprio_layout: Optional[tuple] = None

    @property
    def dense_scene(self):
        count = self.scene_tokens.shape[1] if self.dense_scene_count is None else self.dense_scene_count
        return TokenGroup(self.scene_tokens[:, :count], self.scene_xyz[:, :count])

    @property
    def sparse_scene(self):
        count = self.scene_tokens.shape[1] if self.dense_scene_count is None else self.dense_scene_count
        return TokenGroup(self.scene_tokens[:, count:], self.scene_xyz[:, count:])

    @property
    def language(self):
        return TokenGroup(self.language_tokens, padding_mask=self.language_padding_mask)

    @property
    def proprio(self):
        return TokenGroup(self.proprio_tokens, self.proprio_xyz)

    def validate(self):
        """Validate once when observations are prepared, never inside JVP."""
        if self.global_condition.ndim != 2:
            raise ValueError("global_condition must be [B, D].")
        batch = self.global_condition.shape[0]
        token_dim = self.scene_tokens.shape[-1]
        for name in ("dense_scene", "sparse_scene", "language", "proprio"):
            group = getattr(self, name)
            if group.tokens.ndim != 3 or group.tokens.shape[0] != batch or group.tokens.shape[-1] != token_dim:
                raise ValueError(f"{name} tokens must be [B, N, C] with common batch/width.")
            if group.xyz is not None and group.xyz.shape != (*group.tokens.shape[:2], 3):
                raise ValueError(f"{name} xyz must align with its tokens as [B, N, 3].")
            if group.padding_mask is not None and (
                group.padding_mask.shape != group.tokens.shape[:2] or group.padding_mask.dtype != torch.bool
            ):
                raise ValueError(f"{name} padding mask must be bool [B, N].")
        if self.dense_scene_count is not None and not 0 <= self.dense_scene_count <= self.scene_tokens.shape[1]:
            raise ValueError("dense_scene_count is outside the concatenated scene.")
        if self.workspace_bounds.shape != (batch, 2, 3):
            raise ValueError("workspace_bounds must be batch-expanded [B, 2, 3].")
        if self.proprio_layout is not None:
            history, hands = self.proprio_layout
            if history < 1 or hands < 1 or history * hands != self.proprio_tokens.shape[1] or self.proprio_xyz is None:
                raise ValueError("proprio layout/xyz must match history-major, hand-major tokens.")
        return self

    def _map(self, operation):
        return type(self)(**{
            field.name: operation(value) if torch.is_tensor(value := getattr(self, field.name)) else value
            for field in fields(self)
        })

    def __getitem__(self, batch_index):
        if isinstance(batch_index, int):
            raise TypeError("ActionContext requires a batch slice or mask, not an integer.")
        return self._map(lambda tensor: tensor[batch_index])

    def detach(self):
        return self._map(lambda tensor: tensor.detach())

    def float(self):
        # Padding masks must stay bool, including under outer BF16 autocast.
        return self._map(lambda tensor: tensor.float() if tensor.is_floating_point() else tensor)
