from ..encoder.multimodal.encoder3d import Encoder

from .base_denoise_actor import DenoiseActor as BaseDenoiseActor


class DenoiseActor(BaseDenoiseActor):

    def __init__(self,
                 # Encoder arguments
                 backbone="clip",
                 finetune_backbone=False,
                 finetune_text_encoder=False,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 # Encoder and decoder arguments
                 embedding_dim=60,
                 num_attn_heads=9,
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
                 lv2_batch_size=1,
                 action_hidden_dim=256,
                 action_num_blocks=6,
                 jvp_microbatch_size=8,
                 guidance_scale=1.0,
                 endpoint_loss_weight=0.25,
                 ivc_loss_weight=0.0,
                 condition_dropout_prob=0.0,
                 gripper_transition_weight=2.0,
                 gripper_closed_hold_weight=2.0,
                 gripper_prediction_mode="direct",
                 gripper_hold_prior_logit=2.0):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            nhist=nhist,
            nhand=nhand,
            num_shared_attn_layers=num_shared_attn_layers,
            relative=relative,
            rotation_format=rotation_format,
            denoise_timesteps=denoise_timesteps,
            denoise_model=denoise_model,
            lv2_batch_size=lv2_batch_size,
            action_hidden_dim=action_hidden_dim,
            action_num_blocks=action_num_blocks,
            jvp_microbatch_size=jvp_microbatch_size,
            guidance_scale=guidance_scale,
            endpoint_loss_weight=endpoint_loss_weight,
            ivc_loss_weight=ivc_loss_weight,
            condition_dropout_prob=condition_dropout_prob,
            gripper_transition_weight=gripper_transition_weight,
            gripper_closed_hold_weight=gripper_closed_hold_weight,
            gripper_prediction_mode=gripper_prediction_mode,
            gripper_hold_prior_logit=gripper_hold_prior_logit,
        )

        # Vision-language encoder, runs only once
        self.encoder = Encoder(
            backbone=backbone,
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder
        )
