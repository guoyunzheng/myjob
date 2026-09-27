from .meanflow import MFScheduler
from .flow_matching import FMScheduler


def fetch_schedulers(denoise_model, denoise_timesteps, *, flow_config=None):
    """
    根据 `denoise_model` 返回位置与旋转的调度器实例。

    注意：函数只负责返回实例；具体的 `set_timesteps` 调用以及在训练/推理
    中如何使用调度器的方法由上层模块（如 `DenoiseActor`）负责。
    """
    if flow_config is not None:
        flow_config.validate()
        if denoise_model != flow_config.flow_objective:
            raise ValueError("Scheduler must match the explicit flow_objective.")
    if denoise_model == "ddpm":
        from .ddpm import DDPMScheduler
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
        from .ddim import DDIMScheduler
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
        from .rectified_flow import RFScheduler
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
    elif denoise_model in ("meanflow", "fm", "imf"):
        # Sampling is separate from the training objective, owned by the actor.
        noise_sampler_config = {"mean": 0, "std": 1.5}
        noise_sampler = "logit_normal"
        offdiag_ratio = 0.25
        if flow_config is not None:
            noise_sampler = flow_config.time_sampler
            noise_sampler_config = {
                "mean": flow_config.time_sampler_mean,
                "std": flow_config.time_sampler_std,
            }
            offdiag_ratio = flow_config.meanflow_offdiag_ratio
        scheduler_class = FMScheduler if denoise_model == "fm" else MFScheduler
        kwargs = {"noise_sampler": noise_sampler, "noise_sampler_config": noise_sampler_config}
        if denoise_model in ("meanflow", "imf"):
            kwargs["meanflow_r_ne_t_ratio"] = offdiag_ratio
        position_noise_scheduler = scheduler_class(**kwargs)
        rotation_noise_scheduler = scheduler_class(**kwargs)
     
    else:
        raise ValueError(f"Unknown denoise_model: {denoise_model}")

    return position_noise_scheduler, rotation_noise_scheduler
