import torch
from torch import nn
from torch.nn import functional as F
import einops
from torch.func import jvp
from ..noise_scheduler import fetch_schedulers
from ..utils.layers import AttentionModule
from ..utils.position_encodings import SinusoidalPosEmb
from .gripper_head import GripperStateHead
from ..utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix
)

class DenoiseActor(nn.Module):
    def __init__(self,
                 # Encoder and decoder arguments
                 embedding_dim=128,
                 num_attn_heads=8,
                 nhist=3,
                 nhand=1,
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 relative=False,
                 rotation_format='quat_xyzw',
                 # Denoising arguments
                 denoise_timesteps=2,
                 denoise_model="meanflow",  
                 # Training arguments
                 lv2_batch_size=4,
                 action_hidden_dim=256,
                 action_num_blocks=6,
                 jvp_microbatch_size=8,
                 guidance_scale=1.0,
                 endpoint_loss_weight=0.25,
                 ivc_loss_weight=0.0,
                 condition_dropout_prob=0.0,
                 gripper_transition_weight=2.0,
                 gripper_closed_hold_weight=2.0,
                 gripper_prediction_mode="legacy_denoise",
                 gripper_hold_prior_logit=2.0):
        super().__init__()
        if not 0.0 <= condition_dropout_prob <= 1.0:
            raise ValueError("condition_dropout_prob must be in [0, 1].")
        if gripper_transition_weight < 0.0:
            raise ValueError("gripper_transition_weight must be non-negative.")
        if gripper_closed_hold_weight < 0.0:
            raise ValueError("gripper_closed_hold_weight must be non-negative.")
        if gripper_prediction_mode not in {"direct", "legacy_denoise"}:
            raise ValueError(
                "gripper_prediction_mode must be 'direct' or "
                "'legacy_denoise'."
            )
        if gripper_hold_prior_logit < 0.0:
            raise ValueError("gripper_hold_prior_logit must be non-negative.")
        # Arguments to be accessed by the main class
        self._rotation_format = rotation_format
        self._relative = relative
        self._lv2_batch_size = lv2_batch_size
        self._jvp_microbatch_size = jvp_microbatch_size
        self._guidance_scale = guidance_scale
        self._endpoint_loss_weight = endpoint_loss_weight
        self._ivc_loss_weight = ivc_loss_weight
        self._gripper_transition_weight = gripper_transition_weight
        self._gripper_closed_hold_weight = gripper_closed_hold_weight
        self._gripper_prediction_mode = gripper_prediction_mode

        # Vision-language encoder, runs only once
        self.encoder = None  # Implement this!
        # The encoded visual tokens are already language-conditioned, so
        # clearing only the pooled language tokens does not form a valid CFG
        # unconditional branch. Keep it off when inference uses scale 1.
        self.cond_mask_prob = condition_dropout_prob
        # The observation encoder may keep using attention/Flash Attention.
        # Only this compact, deterministic action decoder is traversed by JVP.
        action_dim = 6 if rotation_format == 'euler' else 9
        self.condition_pooler = ConditionPooler(
            embedding_dim=embedding_dim,
            condition_dim=action_hidden_dim,
        )
        self.prediction_head = FiLMTemporalConvHead(
            action_dim=action_dim,
            hidden_dim=action_hidden_dim,
            condition_dim=action_hidden_dim,
            num_blocks=action_num_blocks,
            nhand=nhand,
            rot_dim=3 if rotation_format == 'euler' else 6,
        )
        self.gripper_state_head = (
            GripperStateHead(
                condition_dim=action_hidden_dim,
                hidden_dim=action_hidden_dim,
                hold_prior_logit=gripper_hold_prior_logit,
            )
            if gripper_prediction_mode == "direct"
            else None
        )

        # Noise/denoise schedulers and hyperparameters
        self.position_scheduler, self.rotation_scheduler = fetch_schedulers(
            denoise_model, denoise_timesteps
        )
        self.n_steps = denoise_timesteps

        # Normalization for the 3D space, will be loaded in the main process
        if rotation_format == 'euler':  # normalize pos+rot
            self.workspace_normalizer = nn.Parameter(
                torch.Tensor([[0., 0, 0, 0, 0, 0], [1., 1, 1, 1, 1, 1]]),
                requires_grad=False
            )
        else:
            self.workspace_normalizer = nn.Parameter(
                torch.Tensor([[0., 0., 0.], [1., 1., 1.]]),
                requires_grad=False
            )
        self.nrm_dim = int(self.workspace_normalizer.size(-1))

    def encode_inputs(self, rgb3d, rgb2d, pcd, instruction, proprio):
        fixed_inputs = self.encoder(
            rgb3d, rgb2d, pcd, instruction,
            proprio.flatten(1, 2)
        )
        # Query trajectory (for relative trajectory prediction)
        query_trajectory = proprio[:, -1:]
        return (query_trajectory,) + fixed_inputs

    def encode_condition(self, fixed_inputs):
        """Pool observation tokens once, outside the action-head JVP path."""
        (
            query_trajectory,
            rgb3d_feats, pcd,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos
        ) = fixed_inputs

        return self.condition_pooler(
            rgb3d_feats=rgb3d_feats,
            rgb3d_pos=pcd,
            rgb2d_feats=rgb2d_feats,
            rgb2d_pos=rgb2d_pos,
            instr_feats=instr_feats,
            proprio_feats=proprio_feats,
            fps_scene_feats=fps_scene_feats,
            fps_scene_pos=fps_scene_pos,
        )

    def policy_forward_pass(self, trajectory, timestep1, timestep2, condition):
        # Accept the old fixed-input tuple as a convenience for external callers.
        if not torch.is_tensor(condition):
            condition = self.encode_condition(condition)
        # The Transformer/Flash encoder remains under BF16 autocast, but the
        # compact FiLM-TCN directly emits normalized metric actions. Keeping
        # this small head in FP32 prevents millimeter-scale corrections from
        # being quantized before either the loss or the MeanFlow integration.
        with torch.autocast(
            device_type=trajectory.device.type,
            enabled=False,
        ):
            return self.prediction_head(
                trajectory.float(),
                timestep1.float(),
                timestep2.float(),
                condition.float(),
            )

    def predict_gripper_logits(self, condition, current_openess, traj_len):
        """Predict target gripper state without diffusion-time dependence."""
        if self.gripper_state_head is None:
            raise RuntimeError(
                "Direct gripper prediction requested in legacy_denoise mode."
            )
        # current_openess is (B, nhand, 1). The target data currently contains
        # one keypose; expanding keeps the public trajectory shape general.
        with torch.autocast(
            device_type=condition.device.type,
            enabled=False,
        ):
            logits = self.gripper_state_head(
                condition.float(), current_openess.float()
            )
        return logits.unsqueeze(1).expand(-1, traj_len, -1, -1)

    def compute_gripper_loss(self, logits, target_openess, current_openess):
        """Cost-sensitive BCE that protects a closed gripper during transport."""
        elementwise_loss = F.binary_cross_entropy_with_logits(
            logits.float(), target_openess.float(), reduction='none'
        )
        target_is_open = target_openess.float() >= 0.5
        current_is_open = current_openess.float() >= 0.5
        transition = (target_is_open != current_is_open).to(
            dtype=elementwise_loss.dtype
        )
        closed_hold = (~target_is_open & ~current_is_open).to(
            dtype=elementwise_loss.dtype
        )
        weights = (
            1.0
            + self._gripper_transition_weight * transition
            + self._gripper_closed_hold_weight * closed_hold
        )
        return (elementwise_loss * weights).sum() / weights.sum().clamp_min(1.0)

    def conditional_sample(self, trajectory, device, fixed_inputs, guidance_scale=1.0, uncond_inputs=None):
        self.position_scheduler.set_timesteps(self.n_steps, device=device)
        self.rotation_scheduler.set_timesteps(self.n_steps, device=device)

        timesteps = self.position_scheduler.timesteps
        prev_timesteps = self.position_scheduler.prev_timesteps
        condition = self.encode_condition(fixed_inputs)

        uncond_condition = None
        if guidance_scale != 1.0:
            if uncond_inputs is None:
                uncond_fixed_inputs = list(fixed_inputs)
                if len(uncond_fixed_inputs) > 5 and uncond_fixed_inputs[5] is not None:
                    uncond_fixed_inputs[5] = torch.zeros_like(uncond_fixed_inputs[5])
                uncond_fixed_inputs = tuple(uncond_fixed_inputs)
            else:
                uncond_fixed_inputs = uncond_inputs
            uncond_condition = (
                uncond_fixed_inputs
                if torch.is_tensor(uncond_fixed_inputs)
                else self.encode_condition(uncond_fixed_inputs)
            )

        for idx, t in enumerate(timesteps):
            # 条件分支
            r = prev_timesteps[idx]
            out_cond = self.policy_forward_pass(
                trajectory,
                r * torch.ones(len(trajectory), device=device),
                t * torch.ones(len(trajectory), device=device),
                condition,
            )
            out_cond = out_cond[-1]

            if guidance_scale == 1.0:
                out = out_cond
            else:
                out_uncond = self.policy_forward_pass(
                    trajectory,
                    r * torch.ones(len(trajectory), device=device),
                    t * torch.ones(len(trajectory), device=device),
                    uncond_condition,
                )
                out_uncond = out_uncond[-1]
                out = out_uncond + guidance_scale * (out_cond - out_uncond)

            pos = self.position_scheduler.step(
                out[..., :3],
                t, r, trajectory[..., :3]
            ).prev_sample
            rot = self.rotation_scheduler.step(
                out[..., 3:-1],
                t, r, trajectory[..., 3:]
            ).prev_sample
            trajectory = torch.cat((pos, rot), -1)

        if self._gripper_prediction_mode == "direct":
            current_openess = fixed_inputs[0][:, -1, :, -1:]
            gripper_logits = self.predict_gripper_logits(
                condition, current_openess, trajectory.shape[1]
            )
        else:
            gripper_logits = out[..., -1:]
        return torch.cat((trajectory, gripper_logits), -1)

    def compute_trajectory(self, trajectory_mask, rgb3d, rgb2d, pcd, instruction, proprio, guidance_scale=None, uncond_inputs=None):
        if guidance_scale is None:
            guidance_scale = self._guidance_scale
        fixed_inputs = self.encode_inputs(rgb3d, rgb2d, pcd, instruction, proprio)
        out_dim = 6 if self._rotation_format == 'euler' else 9
        trajectory = torch.randn(
            size=tuple(trajectory_mask.shape) + (out_dim,),
            device=trajectory_mask.device
        )
        trajectory = self.conditional_sample(
            trajectory,
            device=trajectory_mask.device,
            fixed_inputs=fixed_inputs,
            guidance_scale=guidance_scale,
            uncond_inputs=uncond_inputs
        )
        # The normalizer bounds already include a workspace safety buffer.
        # Prevent denoising overshoot from producing unreachable world poses.
        trajectory[..., :3] = trajectory[..., :3].clamp(-1.0, 1.0)
        _, traj_len, nhand, _ = trajectory.shape
        trajectory = self.unconvert_rot(
            trajectory.flatten(1, 2)
        ).unflatten(1, (traj_len, nhand))
        trajectory = self.unnormalize_pos(trajectory)
        trajectory[..., -1] = trajectory[..., -1].sigmoid()
        return trajectory

    def compute_loss(self, gt_trajectory, rgb3d, rgb2d, pcd, instruction, proprio):
        fixed_inputs = self.encode_inputs(
            rgb3d, rgb2d, pcd, instruction, proprio
        )

        # Classifier-free condition masking is sampled once, before both the
        # target and prediction forwards, so all MeanFlow evaluations see the
        # same condition.
        if self.cond_mask_prob > 0:
            fixed_inputs_list = list(fixed_inputs)
            if torch.rand((), device=gt_trajectory.device) < self.cond_mask_prob:
                fixed_inputs_list[5] = torch.zeros_like(fixed_inputs_list[5])
            fixed_inputs = tuple(fixed_inputs_list)

        # Pool observation features once. Only the compact action head below
        # participates in JVP; the attention-based encoder stays outside it.
        condition = self.encode_condition(fixed_inputs)

        gt_openess = gt_trajectory[..., -1:]
        gt_position_world = gt_trajectory[..., :3].float()
        current_openess = proprio[:, -1:, :, -1:].float()
        gt_trajectory = self.normalize_pos(gt_trajectory[..., :-1])
        _, traj_len, nhand, _ = gt_trajectory.shape
        gt_trajectory = self.convert_rot(
            gt_trajectory.flatten(1, 2)
        ).unflatten(1, (traj_len, nhand))

        direct_gripper_loss = None
        if self._gripper_prediction_mode == "direct":
            direct_gripper_logits = self.predict_gripper_logits(
                condition,
                current_openess[:, -1],
                traj_len,
            )
            direct_gripper_loss = self.compute_gripper_loss(
                direct_gripper_logits, gt_openess, current_openess
            )

        total_loss = 0
        for _ in range(self._lv2_batch_size):
            noise = torch.randn_like(gt_trajectory)
            t, r = self.position_scheduler.sample_noise_step(
                num_noise=len(noise), device=noise.device
            )

            pos = self.position_scheduler.add_noise(
                gt_trajectory[..., :3], noise[..., :3], t
            )
            rot = self.rotation_scheduler.add_noise(
                gt_trajectory[..., 3:], noise[..., 3:], t
            )
            noisy_trajectory = torch.cat((pos, rot), dim=-1)
            velocity = noise - gt_trajectory

            u_target = self.compute_meanflow_target(
                noisy_trajectory,
                r,
                t,
                velocity,
                condition.detach(),
            )

            prediction = self.policy_forward_pass(
                noisy_trajectory, r, t, condition
            )
            ivc_mask = torch.isclose(r, t)

            for layer_prediction in prediction:
                u_prediction = layer_prediction[..., :-1].float()
                loss_position = 30 * F.l1_loss(
                    u_prediction[..., :3],
                    u_target[..., :3],
                    reduction='mean',
                )
                loss_rotation = 10 * F.l1_loss(
                    u_prediction[..., 3:],
                    u_target[..., 3:],
                    reduction='mean',
                )
                if self._gripper_prediction_mode == "legacy_denoise":
                    loss_openess = self.compute_gripper_loss(
                        layer_prediction[..., -1:],
                        gt_openess,
                        current_openess,
                    )
                else:
                    loss_openess = layer_prediction.new_zeros(())
                ivc_per_sample = (
                    u_prediction - velocity.float()
                ).square().flatten(1).mean(1)
                ivc_weights = ivc_mask.to(dtype=ivc_per_sample.dtype)
                loss_ivc = (
                    ivc_per_sample * ivc_weights
                ).sum() / ivc_weights.sum().clamp_min(1)

                # Supervise the exact multi-step solver used at evaluation.
                # The previous one-step endpoint objective did not match a
                # two- or five-step rollout and could improve its own loss
                # while the executed endpoint remained imprecise.
                self.position_scheduler.set_timesteps(
                    self.n_steps, device=noise.device
                )
                endpoint_reconstruction = noise.float()
                for endpoint_t, endpoint_r in zip(
                    self.position_scheduler.timesteps,
                    self.position_scheduler.prev_timesteps,
                ):
                    endpoint_prediction = self.policy_forward_pass(
                        endpoint_reconstruction,
                        endpoint_r.expand(len(noise)),
                        endpoint_t.expand(len(noise)),
                        condition,
                    )[-1][..., :-1].float()
                    endpoint_pos = self.position_scheduler.step(
                        endpoint_prediction[..., :3],
                        endpoint_t,
                        endpoint_r,
                        endpoint_reconstruction[..., :3],
                    ).prev_sample
                    endpoint_rot = self.rotation_scheduler.step(
                        endpoint_prediction[..., 3:],
                        endpoint_t,
                        endpoint_r,
                        endpoint_reconstruction[..., 3:],
                    ).prev_sample
                    endpoint_reconstruction = torch.cat(
                        (endpoint_pos, endpoint_rot), dim=-1
                    )
                endpoint_position_world = self.unnormalize_pos(
                    endpoint_reconstruction
                )[..., :3]
                loss_endpoint_position = 30 * F.smooth_l1_loss(
                    endpoint_position_world,
                    gt_position_world,
                    beta=0.005,
                )
                pred_rotation = compute_rotation_matrix_from_ortho6d(
                    endpoint_reconstruction[..., 3:].reshape(-1, 6)
                )
                gt_rotation = compute_rotation_matrix_from_ortho6d(
                    gt_trajectory[..., 3:].float().reshape(-1, 6)
                )
                rotation_cosine = (
                    (pred_rotation * gt_rotation).sum(dim=(-1, -2)) - 1.0
                ) * 0.5
                # 1-cos(theta) is a stable SO(3) endpoint objective whose scale
                # reflects physical angular error rather than raw 6D distance.
                loss_endpoint_rotation = 10 * (
                    1.0 - rotation_cosine.clamp(-1.0, 1.0)
                ).mean()
                total_loss = (
                    total_loss
                    + loss_position
                    + loss_rotation
                    + loss_openess
                    + self._ivc_loss_weight * loss_ivc
                    + self._endpoint_loss_weight * (
                        loss_endpoint_position + loss_endpoint_rotation
                    )
                )

        total_loss = total_loss / self._lv2_batch_size
        if direct_gripper_loss is not None:
            total_loss = total_loss + direct_gripper_loss
        return total_loss

    def compute_meanflow_target(self, z, r, t, velocity, condition):
        """Construct the stop-gradient target using the exact MeanFlow JVP.

        The characteristic direction is ``(dz, dr, dt) = (velocity, 0, 1)``.
        Chunking applies only to the target computation; the trainable forward
        above still uses the complete optimizer batch.
        """
        batch_size = z.shape[0]
        chunk_size = self._jvp_microbatch_size
        if chunk_size is None or chunk_size <= 0:
            chunk_size = batch_size

        directional_derivatives = []
        # ``torch.func.jvp`` is exact AD, but it still inherits the outer BF16
        # autocast. Run the compact FiLM-TCN target pass in FP32 so small action
        # differences are not quantized before the derivative is formed.
        device_type = z.device.type
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                z_chunk = z[start:end].float()
                r_chunk = r[start:end].float()
                t_chunk = t[start:end].float()
                velocity_chunk = velocity[start:end].float()
                condition_chunk = condition[start:end].float()

                def action_field(z_in, r_in, t_in):
                    return self.policy_forward_pass(
                        z_in, r_in, t_in, condition_chunk
                    )[-1][..., :-1]

                _, derivative = jvp(
                    action_field,
                    (z_chunk, r_chunk, t_chunk),
                    (
                        velocity_chunk,
                        torch.zeros_like(r_chunk),
                        torch.ones_like(t_chunk),
                    ),
                )
                directional_derivatives.append(derivative)

        total_derivative = torch.cat(
            directional_derivatives, dim=0
        ).float()
        delta = (r - t).view(
            [t.size(0)] + [1] * (total_derivative.dim() - 1)
        ).float()
        return (
            velocity.float() + delta * total_derivative
        ).detach()

    def normalize_pos(self, signal):
        _min = self.workspace_normalizer[0]
        _max = self.workspace_normalizer[1]
        diff = _max - _min

        out = signal.clone()
        out[..., :self.nrm_dim] = (
            (signal[..., :self.nrm_dim] - _min) / diff * 2.0
            - 1.0
        )
        return out

    def unnormalize_pos(self, signal):
        _min = self.workspace_normalizer[0]
        _max = self.workspace_normalizer[1]
        diff = _max - _min

        out = signal.clone()
        out[..., :self.nrm_dim] = (
            (signal[..., :self.nrm_dim] + 1.0) / 2.0 * diff
            + _min
        )
        return out

    def convert_rot(self, signal):
        # If Euler then no conversion
        if self._rotation_format == 'euler':
            return signal
        # Else assume quaternion
        rot = normalise_quat(signal[..., 3:7])
        res = signal[..., 7:] if signal.size(-1) > 7 else None
        # The following code expects wxyz quaternion format!
        if self._rotation_format == 'quat_xyzw':
            rot = rot[..., (3, 0, 1, 2)]
        # Convert to rotation matrix
        rot = quaternion_to_matrix(rot)
        # Convert to 6D
        if len(rot.shape) == 4:
            B, L, D1, D2 = rot.shape
            rot = rot.reshape(B * L, D1, D2)
            rot = get_ortho6d_from_rotation_matrix(rot)
            rot = rot.reshape(B, L, 6)
        else:
            rot = get_ortho6d_from_rotation_matrix(rot)
        # Concatenate pos, rot, other state info
        signal = torch.cat([signal[..., :3], rot], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        return signal

    def unconvert_rot(self, signal):
        # If Euler then no conversion
        if self._rotation_format == 'euler':
            return signal
        # Else assume quaternion
        res = signal[..., 9:] if signal.size(-1) > 9 else None
        if len(signal.shape) == 3:
            B, L, _ = signal.shape
            rot = signal[..., 3:9].reshape(B * L, 6)
            mat = compute_rotation_matrix_from_ortho6d(rot)
            quat = matrix_to_quaternion(mat)
            quat = quat.reshape(B, L, 4)
        else:
            rot = signal[..., 3:9]
            mat = compute_rotation_matrix_from_ortho6d(rot)
            quat = matrix_to_quaternion(mat)
        # The above code handled wxyz quaternion format!
        if self._rotation_format == 'quat_xyzw':
            quat = quat[..., (1, 2, 3, 0)]
        signal = torch.cat([signal[..., :3], quat], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        return signal

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb3d,
        rgb2d,
        pcd,
        instruction,
        proprio,
        run_inference=False
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, nhand, 3+4+X)
            trajectory_mask: (B, trajectory_length, nhand)
            rgb3d: (B, num_3d_cameras, 3, H, W) in [0, 1]
            rgb2d: (B, num_2d_cameras, 3, H, W) in [0, 1]
            pcd: (B, num_3d_cameras, 3, H, W) in world coordinates
            instruction: tokenized text instruction
            proprio: (B, nhist, nhand, 3+4+X)

        Note:
            The input rotation is expressed either as:
                a) quaternion (4D), then the model converts it to 6D internally.
                b) Euler angles (3D).

        Returns:
            - loss: scalar, if run_inference is False
            - trajectory: (B, trajectory_length, nhand, 3+rot+1), at inference
        """
        # Inference, don't use gt_trajectory
        if run_inference:
            return self.compute_trajectory(
                trajectory_mask,
                rgb3d, rgb2d, pcd, instruction, proprio
            )

        # Training, use gt_trajectory to compute loss
        return self.compute_loss(
            gt_trajectory,
            rgb3d, rgb2d, pcd, instruction, proprio
        )


class ConditionPooler(nn.Module):
    """Query-free relevance pooling from encoder tokens to one condition vector.

    The FiLM-TCN action head remains attention-free. This pooler runs once,
    outside the JVP path, and preserves target-related spatial information that
    plain global mean pooling would otherwise erase.
    """

    def __init__(self, embedding_dim=128, condition_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.position_projection = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.token_norm = nn.LayerNorm(embedding_dim)
        self.relevance_score = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, 1),
        )
        # A single softmax centroid tends to land between the grasped object
        # and its matching receptacle. Two additional learned slots preserve
        # separate object/target anchors while keeping the action head's fixed
        # condition-vector interface and exact-JVP memory profile unchanged.
        self.num_secondary_slots = 2
        self.secondary_relevance_score = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, self.num_secondary_slots),
        )
        secondary_summary_dim = self.num_secondary_slots * (
            embedding_dim + 6
        )
        self.secondary_slot_projection = nn.Sequential(
            nn.LayerNorm(secondary_summary_dim),
            nn.Linear(secondary_summary_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.spatial_moment_projection = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, embedding_dim),
        )
        # Zero logits give uniform weights, exactly reproducing mean pooling
        # when upgrading an existing 8.10 checkpoint.
        nn.init.zeros_(self.relevance_score[-1].weight)
        nn.init.zeros_(self.relevance_score[-1].bias)
        # Keep old-checkpoint inference behavior unchanged. During new training
        # this path learns to expose the metric centroid and spatial spread of
        # relevant tokens instead of asking a global feature vector to recover
        # millimeter-scale geometry implicitly.
        nn.init.zeros_(self.spatial_moment_projection[-1].weight)
        nn.init.zeros_(self.spatial_moment_projection[-1].bias)
        # This zero residual makes old checkpoints behavior preserving even
        # though their newly initialized slot scorers are intentionally
        # asymmetric. New training quickly learns the residual projection.
        nn.init.zeros_(self.secondary_slot_projection[-1].weight)
        nn.init.zeros_(self.secondary_slot_projection[-1].bias)

        # mean + max pooling for each of: dense scene, sampled scene, 2D
        # scene, language and proprioception.
        pooled_dim = 10 * embedding_dim
        self.output_projection = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, 2 * condition_dim),
            nn.SiLU(),
            nn.Linear(2 * condition_dim, condition_dim),
            nn.LayerNorm(condition_dim),
        )

    def _pool_tokens(self, tokens, positions, reference):
        if tokens is None or tokens.shape[1] == 0:
            return reference.new_zeros(
                reference.shape[0], 2 * self.embedding_dim
            )

        if positions is not None:
            position_features = self.position_projection(positions)
            tokens = tokens + position_features.to(dtype=tokens.dtype)
        tokens = self.token_norm(tokens)
        scores = self.relevance_score(tokens).squeeze(-1)
        weights_float = scores.float().softmax(dim=1)
        weights = weights_float.to(dtype=tokens.dtype)
        relevant = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        if positions is not None:
            positions_float = positions.float()
            centroid = (
                positions_float * weights_float.unsqueeze(-1)
            ).sum(dim=1)
            centered = positions_float - centroid.unsqueeze(1)
            spread = torch.sqrt(
                (
                    centered.square() * weights_float.unsqueeze(-1)
                ).sum(dim=1).clamp_min(1e-8)
            )
            spatial_moments = torch.cat((centroid, spread), dim=-1)
            relevant = relevant + self.spatial_moment_projection(
                spatial_moments
            ).to(dtype=relevant.dtype)
        secondary_scores = self.secondary_relevance_score(tokens)
        secondary_weights_float = secondary_scores.float().softmax(dim=1)
        secondary_weights = secondary_weights_float.to(dtype=tokens.dtype)
        secondary_features = torch.einsum(
            "bns,bne->bse", secondary_weights, tokens
        )
        if positions is not None:
            positions_float = positions.float()
            secondary_centroids = torch.einsum(
                "bns,bnd->bsd", secondary_weights_float, positions_float
            )
            secondary_centered = (
                positions_float.unsqueeze(2)
                - secondary_centroids.unsqueeze(1)
            )
            secondary_spreads = torch.sqrt(
                torch.einsum(
                    "bns,bnsd->bsd",
                    secondary_weights_float,
                    secondary_centered.square(),
                ).clamp_min(1e-8)
            )
            secondary_moments = torch.cat(
                (secondary_centroids, secondary_spreads), dim=-1
            ).to(dtype=secondary_features.dtype)
        else:
            secondary_moments = secondary_features.new_zeros(
                secondary_features.shape[0],
                self.num_secondary_slots,
                6,
            )
        secondary_summary = torch.cat(
            (secondary_features, secondary_moments), dim=-1
        ).flatten(1)
        relevant = relevant + self.secondary_slot_projection(
            secondary_summary.float()
        ).to(dtype=relevant.dtype)
        return torch.cat(
            (relevant, tokens.amax(dim=1)), dim=-1
        )

    def forward(
        self,
        rgb3d_feats,
        rgb3d_pos,
        rgb2d_feats,
        rgb2d_pos,
        instr_feats,
        proprio_feats,
        fps_scene_feats,
        fps_scene_pos,
    ):
        reference = rgb3d_feats
        pooled_features = [
            self._pool_tokens(rgb3d_feats, rgb3d_pos, reference),
            self._pool_tokens(
                fps_scene_feats, fps_scene_pos, reference
            ),
            self._pool_tokens(rgb2d_feats, rgb2d_pos, reference),
            self._pool_tokens(instr_feats, None, reference),
            self._pool_tokens(proprio_feats, None, reference),
        ]
        return self.output_projection(torch.cat(pooled_features, dim=-1))


class FiLMResidualConvBlock(nn.Module):
    """Deterministic Conv1D block with per-sample FiLM modulation."""

    def __init__(self, hidden_dim, condition_dim, dilation=1):
        super().__init__()
        groups = min(8, hidden_dim)
        while hidden_dim % groups != 0:
            groups -= 1

        self.norm1 = nn.GroupNorm(groups, hidden_dim)
        self.norm2 = nn.GroupNorm(groups, hidden_dim)
        self.conv1 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.film = nn.Linear(condition_dim, 4 * hidden_dim)

        # Start each residual branch close to identity.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    @staticmethod
    def _modulate(features, scale, shift):
        return features * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

    def forward(self, features, condition):
        scale1, shift1, scale2, shift2 = self.film(condition).chunk(4, dim=-1)

        residual = features
        features = self._modulate(self.norm1(features), scale1, shift1)
        features = self.conv1(F.silu(features))
        features = self._modulate(self.norm2(features), scale2, shift2)
        features = self.conv2(F.silu(features))
        return residual + features


class FiLMTemporalConvHead(nn.Module):
    """JVP-friendly temporal action head without attention or dropout."""

    def __init__(
        self,
        action_dim=9,
        hidden_dim=256,
        condition_dim=256,
        num_blocks=6,
        nhand=1,
        rot_dim=6,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.output_dim = 3 + rot_dim + 1
        self.input_projection = nn.Linear(action_dim, hidden_dim)
        self.sequence_position_embedding = SinusoidalPosEmb(hidden_dim)
        self.hand_embedding = nn.Embedding(nhand, hidden_dim)

        self.time_embedding = SinusoidalPosEmb(hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(condition_dim + hidden_dim),
            nn.Linear(condition_dim + hidden_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        dilation_pattern = (1, 2, 4, 8)
        self.blocks = nn.ModuleList([
            FiLMResidualConvBlock(
                hidden_dim=hidden_dim,
                condition_dim=condition_dim,
                dilation=dilation_pattern[index % len(dilation_pattern)],
            )
            for index in range(num_blocks)
        ])
        groups = min(8, hidden_dim)
        while hidden_dim % groups != 0:
            groups -= 1
        self.output_norm = nn.GroupNorm(groups, hidden_dim)
        self.output_projection = nn.Conv1d(
            hidden_dim, self.output_dim, kernel_size=1
        )

    def _encode_condition(self, r, t, condition):
        r = r.reshape(condition.shape[0])
        t = t.reshape(condition.shape[0])
        time_features = torch.cat(
            (
                self.time_embedding(r),
                self.time_embedding(t),
                self.time_embedding(t - r),
            ),
            dim=-1,
        )
        time_features = self.time_projection(time_features)
        return self.condition_projection(
            torch.cat((condition, time_features), dim=-1)
        )

    def forward(self, trajectory, r, t, condition):
        batch_size, traj_len, nhand, action_dim = trajectory.shape
        if action_dim != self.action_dim:
            raise ValueError(
                f"Expected trajectory dimension {self.action_dim}, "
                f"got {action_dim}."
            )
        if nhand != self.hand_embedding.num_embeddings:
            raise ValueError(
                f"Expected {self.hand_embedding.num_embeddings} hands, "
                f"got {nhand}."
            )

        features = self.input_projection(trajectory)
        step_ids = torch.arange(
            traj_len,
            device=trajectory.device,
            dtype=trajectory.dtype,
        )
        step_features = self.sequence_position_embedding(step_ids).to(
            dtype=features.dtype
        )
        step_features = step_features.view(1, traj_len, 1, -1)
        hand_ids = torch.arange(nhand, device=trajectory.device)
        hand_features = self.hand_embedding(hand_ids).to(dtype=features.dtype)
        hand_features = hand_features.view(1, 1, nhand, -1)
        features = features + step_features + hand_features
        features = einops.rearrange(
            features, 'b l h c -> b c (l h)'
        )

        film_condition = self._encode_condition(r, t, condition)
        for block in self.blocks:
            features = block(features, film_condition)

        output = self.output_projection(F.silu(self.output_norm(features)))
        output = einops.rearrange(
            output,
            'b c (l h) -> b l h c',
            l=traj_len,
            h=nhand,
        )
        return [output]


class TransformerHead(nn.Module):

    def __init__(self,
                 embedding_dim=128,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 nhist=3,
                 rotary_pe=True,
                 rot_dim=6):
        super().__init__()

        # Different embeddings
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)
        self.hand_embed = nn.Embedding(2, embedding_dim)

        # Attention from trajectory queries to language


        self.traj_lang_attention = AttentionModule(
            num_layers=1,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=False,
            use_adaln=False,
            is_self=False
        )

        self.cross_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=False
        )

        # Shared attention layers
        self.self_attn = AttentionModule(
            num_layers=num_shared_attn_layers,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=True
        )

        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        self.rotation_self_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=True
        )
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rot_dim)
        )

        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        self.position_self_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=True
        )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )

        # 3. Openess
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, traj_feats, trajectory, timesteps1,timesteps2,
                rgb3d_feats, rgb3d_pos, rgb2d_feats, rgb2d_pos,
                instr_feats, instr_pos, proprio_feats,
                fps_scene_feats, fps_scene_pos):
        """
        Arguments:
            traj_feats: (B, trajectory_length, nhand, F)
            trajectory: (B, trajectory_length, nhand, 3+6+X)
            timesteps: (B, 1)
            rgb3d_feats: (B, N, F)
            rgb3d_pos: (B, N, 3)
            rgb2d_feats: (B, N2d, F)
            rgb2d_pos: (B, N2d, 3)
            instr_feats: (B, L, F)
            instr_pos: (B, L, 3)
            proprio_feats: (B, nhist*nhand, F)
            fps_scene_feats: (B, M, F), M < N
            fps_scene_pos: (B, M, 3)

        Returns:
            list of (B, trajectory_length, nhand, 3+6+X)
        """
        _, traj_len, nhand, _ = trajectory.shape

        # Trajectory features
        if nhand > 1:
            traj_feats = traj_feats + self.hand_embed.weight[None, None]
        traj_feats = einops.rearrange(traj_feats, 'b l h c -> b (l h) c')
        trajectory = einops.rearrange(trajectory, 'b l h c -> b (l h) c')

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_len, device=traj_feats.device)
        )[None, None].repeat(len(traj_feats), 1, nhand, 1)
        traj_time_pos = einops.rearrange(traj_time_pos, 'b l h c -> b (l h) c')

        traj_feats = self.traj_lang_attention(
            seq1=traj_feats,
            seq2=instr_feats,
            seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
        )[-1]

        traj_feats = traj_feats + traj_time_pos
        traj_xyz = trajectory[..., :3]

        # Denoising timesteps' embeddings
        time_embs = self.encode_denoising_timestep(
            timesteps1,timesteps2, proprio_feats
        )

        # Positional embeddings
        rel_traj_pos, rel_scene_pos, rel_pos = self.get_positional_embeddings(
            traj_xyz, traj_feats,
            rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
            timesteps1,timesteps2, proprio_feats,
            fps_scene_feats, fps_scene_pos,
            instr_feats, instr_pos
        )

        # Cross attention from gripper to full context
        traj_feats = self.cross_attn(
            seq1=traj_feats,
            seq2=rgb3d_feats,
            seq1_pos=rel_traj_pos,
            seq2_pos=rel_scene_pos,
            ada_sgnl=time_embs
        )[-1]

        # Self attention among gripper and sampled context
        features = self.get_sa_feature_sequence(
            traj_feats, fps_scene_feats,
            rgb3d_feats, rgb2d_feats, instr_feats
        )
        features = self.self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=rel_pos,
            seq2_pos=rel_pos,
            ada_sgnl=time_embs
        )[-1]

        # Rotation head
        rotation = self.predict_rot(
            features, rel_pos, time_embs, traj_feats.shape[1]
        )

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, traj_feats.shape[1]
        )

        # Openess head from position head
        openess = self.openess_predictor(position_features)

        return [
            torch.cat((position, rotation, openess), -1)
                 .unflatten(1, (traj_len, nhand))
        ]

    def encode_denoising_timestep(self, timestep1,timestep2 ,proprio_feats):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep1)
        time_feats2 = self.time_emb(timestep2)
        proprio_feats = proprio_feats.flatten(1)
        curr_gripper_feats = self.curr_gripper_emb(proprio_feats)
        return 0.5*time_feats + curr_gripper_feats+ 0.5*time_feats2
    

    def get_positional_embeddings(
        self,traj_xyz, traj_feats,
        rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
        timesteps1,timesteps2, proprio_feats,
        fps_scene_feats, fps_scene_pos,
        instr_feats, instr_pos
    ):
        return None, None, None

    def get_sa_feature_sequence(
        self,
        traj_feats, fps_scene_feats,
        rgb3d_feats, rgb2d_feats, instr_feats
    ):
        return torch.cat([traj_feats, fps_scene_feats], 1)

    def predict_pos(self, features, pos, time_embs, traj_len):
        position_features = self.position_self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=pos,
            seq2_pos=pos,
            ada_sgnl=time_embs
        )[-1]
        position_features = position_features[:, :traj_len]
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, pos, time_embs, traj_len):
        rotation_features = self.rotation_self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=pos,
            seq2_pos=pos,
            ada_sgnl=time_embs
        )[-1]
        rotation_features = rotation_features[:, :traj_len]
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation
