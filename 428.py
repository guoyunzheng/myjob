import torch
from torch import nn
from torch.nn import functional as F
import einops
import torch.backends
from torch.func import jvp
from ..noise_scheduler import fetch_schedulers
from ..utils.layers import AttentionModule
from ..utils.position_encodings import SinusoidalPosEmb
from ..utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix
)
from torch.cuda.amp import autocast

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
                 denoise_timesteps=100,
                 denoise_model="meanflow",  
                 # Training arguments
                 lv2_batch_size=4):
        super().__init__()
        # Arguments to be accessed by the main class
        self._rotation_format = rotation_format
        self._relative = relative
        self._lv2_batch_size = lv2_batch_size

        # Vision-language encoder, runs only once
        self.encoder = None  # Implement this!
        self.cond_mask_prob = 0.1
        # Action decoder, runs at every denoising timestep
        self.traj_encoder = nn.Linear(
            6 if rotation_format == 'euler' else 9,  # XYZ + Euler or 6D
            embedding_dim
        )
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format == 'euler' else 6
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

    def policy_forward_pass(self, trajectory, timestep1, timestep2, fixed_inputs):
        # Parse inputs
        (
            query_trajectory,
            rgb3d_feats, pcd,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos
        ) = fixed_inputs

        # Get features from normalized (relative) trajectory
        trajectory_feats = self.traj_encoder(trajectory)

        # But use positions from unnormalized absolute trajectory
        traj_xyz = self.unnormalize_pos(trajectory)[..., :3]
        if self._relative:  # relative to absolute
            traj_xyz = (
                query_trajectory[..., :3]
                + torch.cumsum(traj_xyz, dim=1)
            )

        return self.prediction_head(
            trajectory_feats,
            traj_xyz,
            timestep1,
            timestep2,
            rgb3d_feats=rgb3d_feats,
            rgb3d_pos=pcd,
            rgb2d_feats=rgb2d_feats,
            rgb2d_pos=rgb2d_pos,
            instr_feats=instr_feats,
            instr_pos=instr_pos,
            proprio_feats=proprio_feats,
            fps_scene_feats=fps_scene_feats,
            fps_scene_pos=fps_scene_pos
        )

    def conditional_sample(self, trajectory, device, fixed_inputs, guidance_scale=2.0, uncond_inputs=None):
        self.position_scheduler.set_timesteps(self.n_steps, device=device)
        self.rotation_scheduler.set_timesteps(self.n_steps, device=device)

        timesteps = self.position_scheduler.timesteps
        for t_ind, t in enumerate(timesteps):
            # 条件分支
            out_cond = self.policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory)).to(device).long(),
                t * torch.ones(len(trajectory)).to(device).long(),
                fixed_inputs
            )
            out_cond = out_cond[-1]

            if guidance_scale == 1.0:
                out = out_cond
            else:
                if uncond_inputs is None:
                    uncond_fixed_inputs = list(fixed_inputs)
                    if len(uncond_fixed_inputs) > 5 and uncond_fixed_inputs[5] is not None:
                        uncond_fixed_inputs[5] = torch.zeros_like(uncond_fixed_inputs[5])
                    if len(uncond_fixed_inputs) > 6 and uncond_fixed_inputs[6] is not None:
                        uncond_fixed_inputs[6] = torch.zeros_like(uncond_fixed_inputs[6])
                    uncond_fixed_inputs = tuple(uncond_fixed_inputs)
                else:
                    uncond_fixed_inputs = uncond_inputs

                out_uncond = self.policy_forward_pass(
                    trajectory,
                    t * torch.ones(len(trajectory)).to(device).long(),
                    t * torch.ones(len(trajectory)).to(device).long(),
                    uncond_fixed_inputs
                )
                out_uncond = out_uncond[-1]
                out = out_uncond + guidance_scale * (out_cond - out_uncond)

            pos = self.position_scheduler.step(
                out[..., :3],
                t_ind, trajectory[..., :3]
            ).prev_sample
            rot = self.rotation_scheduler.step(
                out[..., 3:-1],
                t_ind, trajectory[..., 3:]
            ).prev_sample
            trajectory = torch.cat((pos, rot), -1)

        return torch.cat((trajectory, out[..., -1:]), -1)

    def compute_trajectory(self, trajectory_mask, rgb3d, rgb2d, pcd, instruction, proprio, guidance_scale=2.0, uncond_inputs=None):
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
        _, traj_len, nhand, _ = trajectory.shape
        trajectory = self.unconvert_rot(
            trajectory.flatten(1, 2)
        ).unflatten(1, (traj_len, nhand))
        trajectory = self.unnormalize_pos(trajectory)
        trajectory[..., -1] = trajectory[..., -1].sigmoid()
        return trajectory

    def compute_loss(self, gt_trajectory, rgb3d, rgb2d, pcd, instruction, proprio):
    
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):#此处可能需要关掉
            fixed_inputs = self.encode_inputs(rgb3d, rgb2d, pcd, instruction, proprio)

            if self.cond_mask_prob > 0:
                fixed_inputs_list = list(fixed_inputs)
                instr_feats = fixed_inputs_list[5]
                instr_pos = fixed_inputs_list[6]
                if torch.rand(1) < self.cond_mask_prob:
                    instr_feats = torch.zeros_like(instr_feats)
                    instr_pos = torch.zeros_like(instr_pos)
                    fixed_inputs_list[5] = instr_feats 
                    fixed_inputs_list[6] = instr_pos 
                fixed_inputs = tuple(fixed_inputs_list)

            gt_openess = gt_trajectory[..., -1:]
            gt_trajectory = gt_trajectory[..., :-1]
            gt_trajectory = self.normalize_pos(gt_trajectory)
            _, traj_len, nhand, _ = gt_trajectory.shape
            gt_trajectory = self.convert_rot(
                gt_trajectory.flatten(1, 2)
            ).unflatten(1, (traj_len, nhand))
            
            total_loss = 0
        for _ in range(self._lv2_batch_size):
            # torch.cuda.empty_cache()
            noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)
            t, r = self.position_scheduler.sample_noise_step(
                num_noise=len(noise), device=noise.device
            )
            eps = 1e-4
            t_next = (t + eps).clamp(max=1.0)
            r_next = (r + eps).clamp(max=1.0)
            pos = self.position_scheduler.add_noise(gt_trajectory[..., :3], noise[..., :3], t)
            rot = self.rotation_scheduler.add_noise(gt_trajectory[..., 3:], noise[..., 3:], t)
            noisy_trajectory = torch.cat((pos, rot), -1) # 并不对应at
            v = noise - gt_trajectory  # velocity target
            z = noisy_trajectory.clone().requires_grad_(True)
            r_var = r.clone().detach().requires_grad_(False)
            t_var = t.clone().detach().requires_grad_(True)
            # pos_next = self.position_scheduler.add_noise(gt_trajectory[..., :3], noise[..., :3], t_next)
            # rot_next = self.rotation_scheduler.add_noise(gt_trajectory[..., 3:], noise[..., 3:], t_next) # 修正旋转
            # z_next = torch.cat((pos_next, rot_next), -1)
            z_next=z+v*eps
            pred= self.policy_forward_pass(z, r, t, fixed_inputs)
            u_t = pred[-1][..., :9]
            z_last=z-v*eps
            t_last = (t - eps).clamp(min=0.0)
            r_last = (r - eps).clamp(min=0.0)
            
          
          
            # def u_fn(z_in, r_in, t_in):
            #     out = self.policy_forward_pass(z_in, r_in, t_in, fixed_inputs)
            #     out = out[-1]
            #     return out[..., :9].float()

            # zeros_r = torch.zeros_like(r_var)
            # ones_t = torch.ones_like(t_var)
            rlast=self.policy_forward_pass(z, r_last, t, fixed_inputs)[-1][..., :9]
            rnext=self.policy_forward_pass(z, r_next, t, fixed_inputs)[-1][..., :9]
            u_next_z = self.policy_forward_pass(z, r, t_next, fixed_inputs)[-1][..., :9]
            u_last=self.policy_forward_pass(z, r, t_last, fixed_inputs)[-1][..., :9]
            # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
            #     _, dudt = jvp(u_fn, (z, r, t), (v, zeros_r, ones_t))
            #     # _, dudr = jvp(u_fn, (z_f32, r_f32, t_f32), (torch.zeros_like(z_f32), torch.ones_like(r_f32), torch.zeros_like(t_f32)))
            #尝试计算完一个再算另一个
            # u_next_t = self.policy_forward_pass(z, r, t_next, fixed_inputs)[-1][..., :9]
            # dudt_z=(u_next_z-u_t)/eps
            # dudt_t = (u_next_t - u_t) / eps
            dudt_approx=(u_next_z-u_last)/(2*eps)
            delta = (r - t).view([t.size(0)] + [1] * (dudt_approx.dim() - 1))
            u_tgt = (v + delta * dudt_approx).detach()
            # u_r=(v+delta*(dudt+dudr)).detach()

            # pred_ivc_list = self.policy_forward_pass(noisy_trajectory, t, t, fixed_inputs)#换成t试一下
            # u_ivc_pred = pred_ivc_list[-1][..., :9]
            # loss_ivc =F.mse_loss(u_ivc_pred, v) 
            dudr=(rnext-rlast)/(2*eps)
            delta2 = (r - t).view([t.size(0)] + [1] * (dudr.dim() - 1))
            v_r=(v + delta2 * dudt_approx+ delta * dudt_approx).detach()
            r_trajectory=(noisy_trajectory+(r-t)*u_t).detach()
            pred_r= self.policy_forward_pass(r_trajectory, r,r, fixed_inputs)
            u_oldpred2=pred_r[-1][...,: 9].detach()
            loss_ivc2=F.mse_loss(u_oldpred2,v_r)
        
            # pred_list=self.policy_forward_pass(noisy_trajectory,t,r,fixed_inputs)
            # u_oldpred=pred_list[-1][...,: 9].detach()
            # a_r=(noisy_trajectory+(r-t)*u_oldpred).detach()
            # pred_list2=(self.policy_forward_pass(a_r.detach(),r,r,fixed_inputs)).detach()
            # u_oldpred2=pred_list2[-1][...,: 9].detach()
            # loss_ivc2=F.mse_loss(u_oldpred2,u_r)
        
            for layer_pred in pred:
                u_pred = layer_pred[..., :9]
                loss_u = 30 * F.l1_loss(u_pred[..., :3], u_tgt[..., :3], reduction='mean')
                loss_rot = 10 * F.l1_loss(u_pred[..., 3:9], u_tgt[..., 3:], reduction='mean')
                
                openess = layer_pred[..., -1:]
                loss = loss_u + loss_rot + F.binary_cross_entropy_with_logits(openess, gt_openess)
                total_loss = total_loss + loss
            total_loss=total_loss+loss_ivc2##0.5*loss_ivc##
        return total_loss

    def normalize_pos(self, signal):
        _min = self.workspace_normalizer[0]
        _max = self.workspace_normalizer[1]
        diff = _max - _min
        out = signal.clone()
        out[..., :self.nrm_dim] = (signal[..., :self.nrm_dim] - _min) / diff * 2.0 - 1.0
        return out

    def unnormalize_pos(self, signal):
        _min = self.workspace_normalizer[0]
        _max = self.workspace_normalizer[1]
        diff = _max - _min
        out = signal.clone()
        out[..., :self.nrm_dim] = (signal[..., :self.nrm_dim] + 1.0) / 2.0 * diff + _min
        return out

    def convert_rot(self, signal):
        if self._rotation_format == 'euler':
            return signal
        rot = normalise_quat(signal[..., 3:7])
        res = signal[..., 7:] if signal.size(-1) > 7 else None
        if self._rotation_format == 'quat_xyzw':
            rot = rot[..., (3, 0, 1, 2)]
        rot = quaternion_to_matrix(rot)
        if len(rot.shape) == 4:
            B, L, D1, D2 = rot.shape
            rot = rot.reshape(B * L, D1, D2)
            rot = get_ortho6d_from_rotation_matrix(rot)
            rot = rot.reshape(B, L, 6)
        else:
            rot = get_ortho6d_from_rotation_matrix(rot)
        signal = torch.cat([signal[..., :3], rot], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        return signal

    def unconvert_rot(self, signal):
        if self._rotation_format == 'euler':
            return signal
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
        if self._rotation_format == 'quat_xyzw':
            quat = quat[..., (1, 2, 3, 0)]
        signal = torch.cat([signal[..., :3], quat], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        return signal

    def forward(self, gt_trajectory, trajectory_mask, rgb3d, rgb2d, pcd, instruction, proprio, run_inference=False):
        if run_inference:
            return self.compute_trajectory(trajectory_mask, rgb3d, rgb2d, pcd, instruction, proprio)
        return self.compute_loss(gt_trajectory, rgb3d, rgb2d, pcd, instruction, proprio)


class TransformerHead(nn.Module):
    def __init__(self,
                 embedding_dim=128,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 nhist=3,
                 rotary_pe=True,
                 rot_dim=6):
        super().__init__()

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

        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, traj_feats, trajectory, timesteps1, timesteps2,
                rgb3d_feats, rgb3d_pos, rgb2d_feats, rgb2d_pos,
                instr_feats, instr_pos, proprio_feats,
                fps_scene_feats, fps_scene_pos):
        _, traj_len, nhand, _ = trajectory.shape

        if nhand > 1:
            traj_feats = traj_feats + self.hand_embed.weight[None, None]
        traj_feats = einops.rearrange(traj_feats, 'b l h c -> b (l h) c')
        trajectory = einops.rearrange(trajectory, 'b l h c -> b (l h) c')

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

        time_embs = self.encode_denoising_timestep(timesteps1, timesteps2, proprio_feats)

        rel_traj_pos, rel_scene_pos, rel_pos = self.get_positional_embeddings(
            traj_xyz, traj_feats, rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
            timesteps1, timesteps2, proprio_feats, fps_scene_feats, fps_scene_pos,
            instr_feats, instr_pos
        )

        traj_feats = self.cross_attn(
            seq1=traj_feats, seq2=rgb3d_feats,
            seq1_pos=rel_traj_pos, seq2_pos=rel_scene_pos,
            ada_sgnl=time_embs
        )[-1]

        features = self.get_sa_feature_sequence(traj_feats, fps_scene_feats, rgb3d_feats, rgb2d_feats, instr_feats)
        features = self.self_attn(
            seq1=features, seq2=features,
            seq1_pos=rel_pos, seq2_pos=rel_pos,
            ada_sgnl=time_embs
        )[-1]

        rotation = self.predict_rot(features, rel_pos, time_embs, traj_feats.shape[1])
        position, position_features = self.predict_pos(features, rel_pos, time_embs, traj_feats.shape[1])
        openess = self.openess_predictor(position_features)

        return [torch.cat((position, rotation, openess), -1).unflatten(1, (traj_len, nhand))]

    def encode_denoising_timestep(self, timestep1, timestep2, proprio_feats):
        time_feats = self.time_emb(timestep1)
        time_feats2 = self.time_emb(timestep2)
        proprio_feats = proprio_feats.flatten(1)
        curr_gripper_feats = self.curr_gripper_emb(proprio_feats)
        return 0.5 * time_feats + curr_gripper_feats + 0.5 * time_feats2

    def get_positional_embeddings(self, *args, **kwargs):
        return None, None, None

    def get_sa_feature_sequence(self, traj_feats, fps_scene_feats, *args):
        return torch.cat([traj_feats, fps_scene_feats], 1)

    def predict_pos(self, features, pos, time_embs, traj_len):
        position_features = self.position_self_attn(
            seq1=features, seq2=features,
            seq1_pos=pos, seq2_pos=pos,
            ada_sgnl=time_embs
        )[-1]
        position_features = position_features[:, :traj_len]
        position_features = self.position_proj(position_features)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, pos, time_embs, traj_len):
        rotation_features = self.rotation_self_attn(
            seq1=features, seq2=features,
            seq1_pos=pos, seq2_pos=pos,
            ada_sgnl=time_embs
        )[-1]
        rotation_features = rotation_features[:, :traj_len]
        rotation_features = self.rotation_proj(rotation_features)
        rotation = self.rotation_predictor(rotation_features)
        return rotation