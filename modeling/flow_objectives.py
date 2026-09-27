"""Pose-flow objectives independent of the observation encoder and action head.

Convention: z_t=(1-t)*data+t*noise, v=noise-data, and inference runs 1 -> 0.
Fields have signature field(z, r, t, condition) and return ONLY pose velocity.
The condition is fixed along the characteristic; its regular training forward
must remain attached to the encoder. Targets themselves are stop-gradient.
"""

import torch
from torch.func import jvp
from torch.nn import functional as F


def interval_coordinates(r, t):
    """Convert the public (r,t) interval to end-time t and length h=t-r.

    Along the MeanFlow characteristic (dr,dt)=(0,1), hence (dt,dh)=(1,1).
    This is NOT a partial time derivative at fixed z or fixed h.
    """
    return t, t - r


class FlowMatchingObjective:
    name = "fm"

    @staticmethod
    def prediction_times(r, t):
        # The solver still integrates from t to r. Only the network input is
        # diagonal: FM learns instantaneous rather than interval-mean velocity.
        return t, t

    @staticmethod
    def target(field, z, r, t, velocity, condition, microbatch_size=0):
        # No target forward, no JVP, and no gradient through the target.
        return velocity.float().detach()

    @staticmethod
    def ivc_loss(field, z, r, t, velocity, condition):
        # FM's primary loss already supervises instantaneous velocity.
        return z.new_zeros((), dtype=torch.float32)


class MeanFlowObjective:
    name = "meanflow"

    @staticmethod
    def prediction_times(r, t):
        return r, t

    @staticmethod
    def target(field, z, r, t, velocity, condition, microbatch_size=0):
        """v - (t-r) * (D_z u @ v + partial_t u), with fixed r/condition.

        Exact forward-mode AD in FP32; only this detached target is chunked.
        Fields must be deterministic and independent across batch examples
        (no dropout or batch-statistic coupling) for chunk equivalence.
        """
        batch_size = z.shape[0]
        if batch_size < 1:
            raise ValueError("MeanFlow target requires a non-empty batch.")
        chunk_size = microbatch_size
        if chunk_size is None or chunk_size <= 0:
            chunk_size = batch_size
        derivatives = []
        with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                z_chunk = z[start:end].float()
                r_chunk = r[start:end].float()
                t_chunk = t[start:end].float()
                velocity_chunk = velocity[start:end].float()
                condition_chunk = condition[start:end].detach().float()

                def action_field(z_in, r_in, t_in):
                    return field(z_in, r_in, t_in, condition_chunk)

                _, derivative = jvp(
                    action_field,
                    (z_chunk, r_chunk, t_chunk),
                    (velocity_chunk, torch.zeros_like(r_chunk), torch.ones_like(t_chunk)),
                )
                derivatives.append(derivative)
        total_derivative = torch.cat(derivatives, dim=0).float()
        delta = (r - t).view([t.size(0)] + [1] * (total_derivative.dim() - 1)).float()
        return (velocity.float() + delta * total_derivative).detach()

    @staticmethod
    def ivc_loss(field, z, r, t, velocity, condition):
        # Do not replace interval-mean predictions with instantaneous targets.
        # The auxiliary prediction is evaluated separately on the diagonal.
        mask = r != t
        if not mask.any():
            return z.new_zeros((), dtype=torch.float32)
        diagonal_t = t[mask]
        prediction = field(z[mask], diagonal_t, diagonal_t, condition[mask]).float()
        return F.mse_loss(prediction, velocity[mask].float())


class ImprovedMeanFlowObjective(MeanFlowObjective):
    """Boundary-reuse iMF, arXiv:2512.02012, Algorithm 1 (without CFG).

    V = u(z,r,t,c) + sg((t-r) * JVP(u; u(z,t,t,c), 0, 1)).
    Regress V against the fixed conditional velocity noise-data. The ordinary
    u forward stays attached to the encoder; neither the tangent nor the JVP
    correction has reverse-mode gradients. Inference still integrates u, not V.
    No auxiliary velocity head, adaptive loss weighting or guidance distillation.
    """

    name = "imf"
    target = staticmethod(FlowMatchingObjective.target)

    @staticmethod
    def prediction_correction(field, z, r, t, condition, microbatch_size=0):
        """Detached FP32 correction; never accepts the ground-truth velocity.

        Evaluate only off-diagonal samples: r=t has exactly zero correction and
        needs neither a boundary forward nor a JVP. Chunking changes memory,
        not the sampled batch or the trainable prediction forward.
        """
        if z.shape[0] < 1:
            raise ValueError("iMF correction requires a non-empty batch.")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("iMF training correction cannot run inside inference_mode (forward AD is disabled); use no_grad instead.")
        correction = torch.zeros_like(z, dtype=torch.float32)
        indices = (r != t).nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return correction
        chunk_size = microbatch_size
        if chunk_size is None or chunk_size <= 0:
            chunk_size = len(indices)
        with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
            for start in range(0, len(indices), chunk_size):
                selected = indices[start:start + chunk_size]
                z_chunk, r_chunk, t_chunk = z[selected].float(), r[selected].float(), t[selected].float()
                fixed_condition = condition[selected].detach().float()

                def action_field(z_in, r_in, t_in):
                    return field(z_in, r_in, t_in, fixed_condition)

                tangent = action_field(z_chunk, t_chunk, t_chunk).float().detach()
                _, derivative = jvp(
                    action_field, (z_chunk, r_chunk, t_chunk),
                    (tangent, torch.zeros_like(r_chunk), torch.ones_like(t_chunk)),
                )
                interval = (t_chunk - r_chunk).view([-1] + [1] * (z.dim() - 1))
                correction[selected] = interval * derivative.float()
        return correction.detach()


def fetch_flow_objective(name):
    if name == "fm":
        return FlowMatchingObjective()
    if name == "meanflow":
        return MeanFlowObjective()
    if name == "imf":
        return ImprovedMeanFlowObjective()
    raise ValueError(f"Flow objective {name!r} is not implemented.")
