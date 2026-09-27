import math

import torch


class MFScheduler:
    """MeanFlow time sampling and integration from noise (t=1) to data (t=0).

    The actor constructs the exact JVP target; this scheduler supplies
    configurable (t, r) pairs, linear interpolation and solver steps.
    """

    def __init__(self, noise_sampler="logit_normal", noise_sampler_config=None,
                 meanflow_r_ne_t_ratio=0.25):
        if noise_sampler not in ("logit_normal", "uniform"):
            raise ValueError("noise_sampler must be logit_normal or uniform.")
        self.noise_sampler = noise_sampler
        self.noise_sampler_config = (
            {} if noise_sampler_config is None else dict(noise_sampler_config)
        )
        mean = self.noise_sampler_config.get('mean', 0.0)
        std = self.noise_sampler_config.get('std', 1.5)
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError("Sampler mean must be finite and std finite and positive.")
        if not 0.0 <= meanflow_r_ne_t_ratio <= 1.0:
            raise ValueError("meanflow_r_ne_t_ratio must be in [0, 1].")
        self.meanflow_r_ne_t_ratio = meanflow_r_ne_t_ratio

    def set_timesteps(self, num_inference_steps, device='cpu'):
        """Build descending t and r values; the final r is zero."""
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive.")
        self.timesteps = torch.linspace(
            1.0, 1.0 / num_inference_steps, num_inference_steps,
            device=device, dtype=torch.float32,
        )
        self.prev_timesteps = torch.cat((
            self.timesteps[1:], self.timesteps.new_zeros(1)
        ))

    def sample_noise_step(self, num_noise, device):
        """Return (t, r), each of shape (B,), with a chosen fraction of r < t."""
        mean = self.noise_sampler_config.get('mean', 0.0)
        std = self.noise_sampler_config.get('std', 1.5)
        r_samples = torch.empty(num_noise, device=device, dtype=torch.float32)
        t_samples = torch.empty_like(r_samples)
        if self.noise_sampler == "logit_normal":
            r_samples = r_samples.normal_(mean=mean, std=std).sigmoid()
            t_samples = t_samples.normal_(mean=mean, std=std).sigmoid()
        else:
            r_samples.uniform_()
            t_samples.uniform_()
        t = torch.maximum(r_samples, t_samples)
        r = torch.minimum(r_samples, t_samples)

        ratio = self.meanflow_r_ne_t_ratio
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("meanflow_r_ne_t_ratio must be in [0, 1].")
        if ratio == 0:
            r = t
        elif ratio < 1:
            mask = torch.rand(num_noise, device=device) < ratio
            r = torch.where(mask, r, t)
        return t, r

    def add_noise(self, original_samples, noise, timesteps):
        """Interpolate z_t = (1 - t) * data + t * noise per batch item."""
        t = timesteps.view(-1, *([1] * (original_samples.ndim - 1)))
        return ((1 - t) * original_samples + t * noise).to(original_samples.dtype)

    def step(self, model_output, timestep, prev_timestep, sample):
        """Integrate z_r = z_t - (t - r) * u(z_t, r, t)."""
        if torch.any(prev_timestep > timestep):
            raise ValueError(
                "MeanFlow inference must run from noise to data (1 -> 0): "
                "prev_timestep cannot be greater than timestep."
            )
        return DummyClass(sample - (timestep - prev_timestep) * model_output)

    def prepare_target(self, noise, gt):
        """Instantaneous velocity; the actor adds the MeanFlow JVP correction."""
        return noise - gt


class DummyClass:
    def __init__(self, prev_sample):
        self.prev_sample = prev_sample
