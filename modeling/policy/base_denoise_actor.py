import torch
from torch import nn
from torch.nn import functional as F
import einops
from torch.func import jvp
from ..noise_scheduler import fetch_schedulers
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
        self.collect_training_diagnostics = False
        self.loss_diagnostics = {}

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
        instr_padding_mask = self.encoder.instruction_padding_mask(instruction)
        fixed_inputs = self.encoder(
            rgb3d, rgb2d, pcd, instruction,
            proprio.flatten(1, 2)
        )
        # Query trajectory (for relative trajectory prediction)
        query_trajectory = proprio[:, -1:]
        return (query_trajectory,) + fixed_inputs + (instr_padding_mask,)

    def encode_condition(self, fixed_inputs):
        """Pool observation tokens once, outside the action-head JVP path."""
        (
            query_trajectory,
            rgb3d_feats, pcd,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos,
            instr_padding_mask,
        ) = fixed_inputs

        return self.condition_pooler(
            rgb3d_feats=rgb3d_feats,
            rgb3d_pos=pcd,
            rgb2d_feats=rgb2d_feats,
            rgb2d_pos=rgb2d_pos,
            instr_feats=instr_feats,
            instr_padding_mask=instr_padding_mask,
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
        with torch.autocast(
            device_type=condition.device.type,
            enabled=False,
        ):
            logits = self.gripper_state_head(
                condition.float(), current_openess.float(), traj_len=traj_len
            )
        return logits

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

    def denoise_trajectory(
        self, trajectory, condition, guidance_scale=1.0, uncond_condition=None
    ):
        """Shared differentiable solver for inference and endpoint supervision."""
        self.position_scheduler.set_timesteps(self.n_steps, device=trajectory.device)
        self.rotation_scheduler.set_timesteps(self.n_steps, device=trajectory.device)
        for t, r in zip(
            self.position_scheduler.timesteps,
            self.position_scheduler.prev_timesteps,
        ):
            batch_r, batch_t = r.expand(len(trajectory)), t.expand(len(trajectory))
            out = self.policy_forward_pass(
                trajectory, batch_r, batch_t, condition
            )[-1]
            if guidance_scale != 1.0:
                out_uncond = self.policy_forward_pass(
                    trajectory, batch_r, batch_t, uncond_condition
                )[-1]
                out = out_uncond + guidance_scale * (out - out_uncond)
            pos = self.position_scheduler.step(
                out[..., :3], t, r, trajectory[..., :3]
            ).prev_sample
            rot = self.rotation_scheduler.step(
                out[..., 3:-1], t, r, trajectory[..., 3:]
            ).prev_sample
            trajectory = torch.cat((pos, rot), dim=-1)
        return trajectory, out

    def conditional_sample(self, trajectory, device, fixed_inputs, guidance_scale=1.0, uncond_inputs=None):
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

        trajectory, out = self.denoise_trajectory(
            trajectory, condition, guidance_scale, uncond_condition
        )

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
        trajectory = self.unconvert_rot(trajectory)
        trajectory = self.unnormalize_pos(trajectory)
        trajectory[..., -1] = trajectory[..., -1].sigmoid()
        return trajectory

    def compute_loss(self, gt_trajectory, rgb3d, rgb2d, pcd, instruction, proprio):
        diagnostics = {} if self.collect_training_diagnostics else None

        def record(name, value):
            if diagnostics is not None:
                diagnostics[name] = diagnostics.get(name, 0) + value.detach()

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
        traj_len = gt_trajectory.shape[1]
        gt_trajectory = self.convert_rot(gt_trajectory)

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
            loss_ivc = self.compute_ivc_loss(
                noisy_trajectory, r, t, velocity, condition
            )
            total_loss = total_loss + self._ivc_loss_weight * loss_ivc
            record('ivc', self._ivc_loss_weight * loss_ivc)

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

                total_loss = (
                    total_loss
                    + loss_position
                    + loss_rotation
                    + loss_openess
                )
                record('velocity_position', loss_position)
                record('velocity_rotation', loss_rotation)
                record('gripper', loss_openess)

            if self._endpoint_loss_weight != 0:
                endpoint_loss = self.compute_endpoint_loss(
                    noise, condition, gt_position_world, gt_trajectory
                )
                total_loss = total_loss + (
                    len(prediction) * self._endpoint_loss_weight * endpoint_loss
                )
                if diagnostics is not None:
                    for name, value in self.endpoint_loss_diagnostics.items():
                        record(
                            name,
                            len(prediction) * self._endpoint_loss_weight * value,
                        )

        total_loss = total_loss / self._lv2_batch_size
        if direct_gripper_loss is not None:
            total_loss = total_loss + direct_gripper_loss
        if diagnostics is not None:
            self.loss_diagnostics = {
                name: value / self._lv2_batch_size for name, value in diagnostics.items()
            }
            if direct_gripper_loss is not None:
                self.loss_diagnostics['gripper'] = direct_gripper_loss.detach()
        return total_loss

    def compute_endpoint_loss(self, noise, condition, gt_position_world, gt_trajectory):
        """Supervise the same multi-step rollout executed at evaluation."""
        reconstruction, _ = self.denoise_trajectory(noise.float(), condition)
        position_loss = 30 * F.smooth_l1_loss(
            self.unnormalize_pos(reconstruction)[..., :3],
            gt_position_world,
            beta=0.005,
        )
        pred_rotation = compute_rotation_matrix_from_ortho6d(
            reconstruction[..., 3:].reshape(-1, 6)
        )
        gt_rotation = compute_rotation_matrix_from_ortho6d(
            gt_trajectory[..., 3:].float().reshape(-1, 6)
        )
        rotation_cosine = (
            (pred_rotation * gt_rotation).sum(dim=(-1, -2)) - 1.0
        ) * 0.5
        rotation_loss = 10 * (1.0 - rotation_cosine.clamp(-1.0, 1.0)).mean()
        if self.collect_training_diagnostics:
            self.endpoint_loss_diagnostics = {
                'endpoint_position': position_loss.detach(),
                'endpoint_rotation': rotation_loss.detach(),
            }
        return position_loss + rotation_loss

    def compute_ivc_loss(self, z, r, t, velocity, condition):
        """Supervise u(z_t, t, t) only for samples drawn with r != t.

        Diagonal samples already receive velocity supervision from the main
        MeanFlow loss (with L1 rather than this auxiliary MSE). For off-diagonal
        samples, evaluate a separate diagonal prediction: u(z_t, r, t) is an
        average velocity and must retain its MeanFlow target. Keep gradients
        through the selected condition features as well as the action head.
        """
        if self._ivc_loss_weight == 0:
            return z.new_zeros((), dtype=torch.float32)
        # The sampler sets r=t exactly. Preserve even very short nonzero
        # intervals instead of discarding them using an isclose tolerance.
        mask = r != t
        if not mask.any():
            return z.new_zeros((), dtype=torch.float32)
        diagonal_t = t[mask]
        diagonal_prediction = self.policy_forward_pass(
            z[mask], diagonal_t, diagonal_t, condition[mask]
        )[-1][..., :-1].float()
        return F.mse_loss(diagonal_prediction, velocity[mask].float())

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
        # The following code expects wxyz quaternion format!
        if self._rotation_format == 'quat_xyzw':
            rot = rot[..., (3, 0, 1, 2)]
        matrix = quaternion_to_matrix(rot)
        rot = get_ortho6d_from_rotation_matrix(matrix.reshape(-1, 3, 3))
        rot = rot.reshape(*signal.shape[:-1], 6)
        return torch.cat((signal[..., :3], rot, signal[..., 7:]), dim=-1)

    def unconvert_rot(self, signal):
        if self._rotation_format == 'euler':
            return signal
        matrix = compute_rotation_matrix_from_ortho6d(
            signal[..., 3:9].reshape(-1, 6)
        )
        quat = matrix_to_quaternion(matrix).reshape(*signal.shape[:-1], 4)
        if self._rotation_format == 'quat_xyzw':
            quat = quat[..., (1, 2, 3, 0)]
        return torch.cat((signal[..., :3], quat, signal[..., 9:]), dim=-1)

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

    @staticmethod
    def _spatial_moments(positions, weights):
        """Centroid and spread for relevance weights shaped (B, N, S)."""
        positions = positions.float()
        # Preserve the primary slot's FP32 reductions under encoder autocast.
        # Secondary slots retain their existing batched einsum precision.
        if weights.shape[-1] == 1:
            centroid = (positions * weights).sum(dim=1).unsqueeze(1)
        else:
            centroid = torch.einsum("bns,bnd->bsd", weights, positions)
        centered = positions.unsqueeze(2) - centroid.unsqueeze(1)
        if weights.shape[-1] == 1:
            variance = (centered.square() * weights.unsqueeze(-1)).sum(dim=1)
        else:
            variance = torch.einsum("bns,bnsd->bsd", weights, centered.square())
        return torch.cat((centroid, variance.clamp_min(1e-8).sqrt()), dim=-1)

    def _pool_tokens(
        self, tokens, positions, reference, padding_mask=None
    ):
        if tokens is None or tokens.shape[1] == 0:
            return reference.new_zeros(
                reference.shape[0], 2 * self.embedding_dim
            )

        if padding_mask is not None:
            if padding_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    "padding_mask must match the first two token dimensions; "
                    f"got {tuple(padding_mask.shape)} for "
                    f"{tuple(tokens.shape)}."
                )
            padding_mask = padding_mask.to(
                device=tokens.device, dtype=torch.bool
            )
            # CLIP always starts with a valid BOS token. Keeping it unmasked
            # also guarantees finite softmax/max output for malformed input.
            padding_mask = padding_mask.clone()
            padding_mask[:, 0] = False

        if positions is not None:
            position_features = self.position_projection(positions)
            tokens = tokens + position_features.to(dtype=tokens.dtype)
        tokens = self.token_norm(tokens)
        scores = self.relevance_score(tokens).squeeze(-1)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, -torch.inf)
        weights_float = scores.float().softmax(dim=1)
        weights = weights_float.to(dtype=tokens.dtype)
        relevant = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        if positions is not None:
            spatial_moments = self._spatial_moments(
                positions, weights_float.unsqueeze(-1)
            ).squeeze(1)
            relevant = relevant + self.spatial_moment_projection(
                spatial_moments
            ).to(dtype=relevant.dtype)
        secondary_scores = self.secondary_relevance_score(tokens)
        if padding_mask is not None:
            secondary_scores = secondary_scores.masked_fill(
                padding_mask.unsqueeze(-1), -torch.inf
            )
        secondary_weights_float = secondary_scores.float().softmax(dim=1)
        secondary_weights = secondary_weights_float.to(dtype=tokens.dtype)
        secondary_features = torch.einsum(
            "bns,bne->bse", secondary_weights, tokens
        )
        if positions is not None:
            secondary_moments = self._spatial_moments(
                positions, secondary_weights_float
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
        if padding_mask is not None:
            max_tokens = tokens.masked_fill(
                padding_mask.unsqueeze(-1),
                torch.finfo(tokens.dtype).min,
            ).amax(dim=1)
        else:
            max_tokens = tokens.amax(dim=1)
        return torch.cat((relevant, max_tokens), dim=-1)

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
        instr_padding_mask=None,
    ):
        reference = rgb3d_feats
        pooled_features = [
            self._pool_tokens(rgb3d_feats, rgb3d_pos, reference),
            self._pool_tokens(
                fps_scene_feats, fps_scene_pos, reference
            ),
            self._pool_tokens(rgb2d_feats, rgb2d_pos, reference),
            self._pool_tokens(
                instr_feats,
                None,
                reference,
                padding_mask=instr_padding_mask,
            ),
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
