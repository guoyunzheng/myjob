from .ddim import DDIMScheduler
from .ddpm import DDPMScheduler
from .rectified_flow import RFScheduler
from .meanflow import MFScheduler


def fetch_schedulers(denoise_model, denoise_timesteps):
    """
    根据 `denoise_model` 返回位置与旋转的调度器实例。

    注意：函数只负责返回实例；具体的 `set_timesteps` 调用以及在训练/推理
    中如何使用调度器的方法由上层模块（如 `DenoiseActor`）负责。
    """
    if denoise_model == "ddpm":
        position_noise_scheduler = DDPMScheduler(
            num_train_timesteps=denoise_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        rotation_noise_scheduler = DDPMScheduler(
            num_train_timesteps=denoise_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )
    elif denoise_model == "ddim":
        position_noise_scheduler = DDIMScheduler(
            num_train_timesteps=denoise_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        rotation_noise_scheduler = DDIMScheduler(
            num_train_timesteps=denoise_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )

    elif denoise_model in ("rectified_flow", "unit", "pi0", "flow_uniform"):
        # 这里为 rectified_flow 系列调度器提供统一的构造逻辑
        # noise_sampler_config 可以控制 logit_normal 的 mean/std 等
        noise_sampler_config = {"mean": 0, "std": 1.5}
        if denoise_model == "unit":
            # unit 使用较小的 std（示例）
            noise_sampler_config = {"mean": 0, "std": 1.0}
        samplers = {
            "rectified_flow": "logit_normal",
            "unit": "logit_normal",
            "pi0": "pi0",
            "flow_uniform": "uniform"
        }
        position_noise_scheduler = RFScheduler(
            noise_sampler=samplers[denoise_model],
            noise_sampler_config=noise_sampler_config
        )
        rotation_noise_scheduler = RFScheduler(
            noise_sampler=samplers[denoise_model],
            noise_sampler_config=noise_sampler_config
        )
    elif denoise_model == "meanflow":
        # 返回 MeanFlowScheduler：默认启用 meanflow 模式
        noise_sampler_config = {"mean": 0, "std": 1.5}
        position_noise_scheduler = MFScheduler(
            noise_sampler="logit_normal",
            noise_sampler_config=noise_sampler_config
        )
        rotation_noise_scheduler = MFScheduler(
            noise_sampler="logit_normal",
            noise_sampler_config=noise_sampler_config
        )
     
    else:
        raise ValueError(f"Unknown denoise_model: {denoise_model}")

    return position_noise_scheduler, rotation_noise_scheduler




