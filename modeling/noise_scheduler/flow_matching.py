"""Single-time FM sampling with the shared noise-to-data Euler solver."""

import torch

from .meanflow import MFScheduler


class FMScheduler(MFScheduler):
    def __init__(self, noise_sampler="logit_normal", noise_sampler_config=None):
        # Reuse interpolation, descending grid and z_r=z_t-(t-r)*velocity.
        # Do NOT reuse MeanFlow's pair sorting: that changes the marginal of t.
        super().__init__(noise_sampler, noise_sampler_config, meanflow_r_ne_t_ratio=0.)

    def sample_noise_step(self, num_noise, device):
        t = torch.empty(num_noise, device=device, dtype=torch.float32)
        if self.noise_sampler == "uniform":
            t.uniform_()
        else:
            t = t.normal_(mean=self.noise_sampler_config.get("mean", 0.),
                          std=self.noise_sampler_config.get("std", 1.5)).sigmoid()
        return t, t

    def prepare_target(self, noise, gt):
        """FM's instantaneous velocity target; no MeanFlow JVP correction."""
        return noise - gt
