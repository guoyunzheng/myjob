"""Deterministic token-query action decoder, compatible with exact forward AD.

Only action queries self-attend. Observations are encoded outside the JVP and
read through cross-attention; no scene-sized self-attention is constructed here.
Both 'auto' and 'math' resolve to the SAME explicit matmul/softmax path for
training, JVP and inference. No fused SDPA, dropout, or mutable feature cache.
Train/eval entry points also disable CUDA matmul TF32 at startup via
configure_action_precision; standalone CUDA callers should do the same.
"""

import torch
from torch import nn

from .action_context import ActionContext
from .flow_objectives import interval_coordinates
from .utils.position_encodings import SinusoidalPosEmb


class MathAttention(nn.Module):
    def __init__(self, hidden_dim, context_dim, num_heads, geometry=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(context_dim, hidden_dim)
        self.value = nn.Linear(context_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Smooth relative geometry. The encoder's detached/no_grad positional
        # encoding must NOT be reused for action-dependent positions.
        self.relative_bias = (
            nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, num_heads))
            if geometry else None
        )

    def forward(self, query, context, padding_mask=None, relative_xyz=None):
        batch, queries, hidden = query.shape
        if context.shape[1] == 0:
            return torch.zeros_like(query)
        if padding_mask is not None:
            context = context.masked_fill(padding_mask[..., None], 0.)

        def split(tensor):
            return tensor.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.query(query)), split(self.key(context)), split(self.value(context))
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if self.relative_bias is not None:
            scores = scores + self.relative_bias(relative_xyz).permute(0, 3, 1, 2)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        if padding_mask is not None:
            # All-padded language yields zero attention, not softmax(-inf) NaNs.
            weights = weights.masked_fill(padding_mask[:, None, None, :], 0.)
        result = (weights @ v).transpose(1, 2).reshape(batch, queries, hidden)
        return self.output(result)


class ActionTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, token_dim, num_heads):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(5)])
        self.modulation = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.self_attention = MathAttention(hidden_dim, hidden_dim, num_heads)
        self.language_attention = MathAttention(hidden_dim, token_dim, num_heads)
        self.proprio_attention = MathAttention(hidden_dim, token_dim, num_heads)
        self.scene_attention = MathAttention(hidden_dim, token_dim, num_heads, geometry=True)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 4 * hidden_dim), nn.SiLU(),
                                 nn.Linear(4 * hidden_dim, hidden_dim))

    def forward(self, query, modulation, context, relative_xyz):
        scale, shift = self.modulation(modulation).chunk(2, dim=-1)
        normalized = self.norms[0](query) * (1 + scale[:, None]) + shift[:, None]
        query = query + self.self_attention(normalized, normalized)
        query = query + self.language_attention(
            self.norms[1](query), context.language.tokens, context.language.padding_mask)
        query = query + self.proprio_attention(self.norms[2](query), context.proprio.tokens)
        query = query + self.scene_attention(
            self.norms[3](query), context.scene_tokens, relative_xyz=relative_xyz)
        return query + self.ffn(self.norms[4](query))


class TransformerActionHead(nn.Module):
    def __init__(self, token_dim, hidden_dim=256, condition_dim=256,
                 num_heads=8, num_blocks=6, nhand=1, attention_backend="auto"):
        super().__init__()
        if hidden_dim < 4 or hidden_dim % 2 or num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("Transformer action_hidden_dim must be even, >=4, and divisible by num_attn_heads.")
        if num_blocks < 1 or nhand < 1:
            raise ValueError("Transformer action_num_blocks and nhand must be positive.")
        if attention_backend not in ("auto", "math"):
            raise ValueError("Transformer action head supports only auto/math (both use math attention).")
        self.attention_backend = "math"
        self.nhand = nhand
        self.input_projection = nn.Linear(9, hidden_dim)
        self.step_embedding = SinusoidalPosEmb(hidden_dim)
        self.hand_embedding = nn.Embedding(nhand, hidden_dim)
        self.time_embedding = SinusoidalPosEmb(hidden_dim)
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(condition_dim + 2 * hidden_dim),
            nn.Linear(condition_dim + 2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([
            ActionTransformerBlock(hidden_dim, token_dim, num_heads) for _ in range(num_blocks)
        ])
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 10))

    def forward(self, trajectory, r, t, context):
        if not isinstance(context, ActionContext):
            raise TypeError("TransformerActionHead requires token-valued ActionContext.")
        if trajectory.ndim != 4 or trajectory.shape[2:] != (self.nhand, 9):
            raise ValueError("Expected normalized pose trajectory [batch, steps, nhand, 9].")
        # Enforce the same FP32 field even when called without the actor wrapper.
        with torch.autocast(device_type=trajectory.device.type, enabled=False):
            return self._forward_fp32(trajectory.float(), r.float(), t.float(), context.float())

    def _forward_fp32(self, trajectory, r, t, context):
        batch, steps, hands, _ = trajectory.shape
        end_time, interval = interval_coordinates(r.reshape(batch), t.reshape(batch))
        modulation = self.condition_projection(torch.cat((
            context.global_condition, self.time_embedding(end_time), self.time_embedding(interval)
        ), dim=-1))
        query = self.input_projection(trajectory)
        query = query + self.step_embedding(torch.arange(steps, device=query.device).float())[None, :, None]
        query = query + self.hand_embedding(torch.arange(hands, device=query.device))[None, None]
        query = query.flatten(1, 2)

        lower, upper = context.workspace_bounds[:, 0], context.workspace_bounds[:, 1]
        query_xyz = (trajectory[..., :3].flatten(1, 2) + 1.) * .5
        query_xyz = query_xyz * (upper - lower)[:, None] + lower[:, None]
        # World-coordinate deltas in meters. No clipping/detach: D_z u must
        # include this positional route, even outside the workspace during flow.
        relative_xyz = query_xyz[:, :, None] - context.scene_xyz[:, None]
        for block in self.blocks:
            query = block(query, modulation, context, relative_xyz)
        # Final channel is a compatibility gripper logit, NOT integrated by flow.
        return [self.output(query).reshape(batch, steps, hands, 10)]
