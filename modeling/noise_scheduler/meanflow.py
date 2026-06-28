import torch
import numpy as np


class MFScheduler:
    """
    RFScheduler: 一个兼容层，采用与 MeanFlow 思路一致的采样步长计算。

    功能概要：
    - 生成/设置时间步序列（`set_timesteps`）
    - 根据指定分布采样随机时间点（`sample_noise_step`）
    - 根据时间 t 将干净样本和噪声插值得到 z_t（`add_noise`）
    - 给定模型输出（期望为平均速度 u），通过一步求解更新样本（`step`）
    - 为训练提供兼容的 denoising 目标（`prepare_target`，当前为简单差值）

    备注（MeanFlow 关联）:
    - MeanFlow 的采样求解器通常为 z_{t-dt} = z_t - dt * u，其中 u 是模型
      预测的 "平均速度"。在此实现中，`step` 按上述形式计算。
    - 若需训练时复现实验论文中的目标，需要额外计算 du/dt （通过 JVP），
      这超出了本兼容实现的范围。
    """

    def __init__(self, noise_sampler="logit_normal", noise_sampler_config={}):
        # noise_sampler: 支持 "uniform", "logit_normal", "pi0"
        # noise_sampler_config: 对于 logit_normal 可包含 mean/std
        # 支持 MeanFlow 模式：当 meanflow_enabled=True 时，sample_noise_step
        # 会返回一对 (t, r)（并保证 t >= r），并且训练环节会使用 JVP
        # 计算 MeanFlow 目标。默认关闭以保留向后兼容。
        self.noise_sampler = noise_sampler
        self.noise_sampler_config = noise_sampler_config
        
        # 在 MeanFlow 模式下，控制 r != t 的占比（默认 0.25），即有多少比例
        # 的样本使用 r < t，而其余使用 r == t（等价于 Flow Matching）
        self.meanflow_r_ne_t_ratio = 0.25
        
#控制 r != t 的占比（默认 0.25），即有多少比例参数，在config中修改





    # def set_timesteps(self, num_inference_steps, device='cpu'):
    #     """
    #     构造时间步序列并预计算前一步的时间值。

    #     参数:
    #     - num_inference_steps: 总的推理步数（整数）
    #     - device: torch 设备

    #     结果:
    #     - self.timesteps: 降序的时间张量，范围 (0, 1], 长度等于 num_inference_steps
    #     - self.prev_timesteps: 对应的上一个时间点（最后一个元素为 0）
    #     """
    #     self.timesteps = torch.from_numpy(
    #         np.arange(num_inference_steps + 1)[1:][::-1].astype(np.float32)
    #         / num_inference_steps
    #     ).to(device)
    #     # 预计算 prev_timesteps：用于计算 dt = t - prev_t
    #     self.prev_timesteps = torch.cat((
    #         self.timesteps[1:],
    #         torch.zeros(1, device=device, dtype=self.timesteps.dtype)
    #     ))
    def set_timesteps(self, num_inference_steps, device='cpu'):
        """
            构造时间步序列并预计算下一步的时间值（升序排列）。

            参数:
            - num_inference_steps: 总的推理步数（整数）
            - device: torch 设备

            结果:
            - self.timesteps: 升序的时间张量，范围 [0, 1), 长度等于 num_inference_steps
            - self.prev_timesteps: 对应的下一个时间点（最后一个元素为 1.0）
            """
            # 1. 生成从 0 开始递增的序列
        self.timesteps = torch.from_numpy(
            np.arange(num_inference_steps).astype(np.float32)
            / num_inference_steps
        ).to(device)
        
        # 2. 预计算 prev_timesteps（在升序中，它代表前向的下一步）：用于计算 dt = prev_t - t
        self.prev_timesteps = torch.cat((
            self.timesteps[1:],
            torch.ones(1, device=device, dtype=self.timesteps.dtype)
        ))
    def sample_noise_step(self, num_noise, device):
        """
        从定义的噪声时间分布中采样 t。

        支持三种策略：
        - "uniform": 直接从 [0,1) 均匀采样
        - "logit_normal": 先对标准正态变换 x ~ N(mean, std)，再取 sigmoid(x)
          （等同于 logit-normal 分布），适用于论文中常用的 t 分布。
        - "pi0": 使用 Beta(alpha=1.5, beta=1.0) 采样并截断到 [0, 0.999]

        返回 shape 为 (num_noise,) 的张量，值在 (0, 1)
        """
                # 如果启用了 MeanFlow 模式，则我们需要返回 (t, r) 对
   
            # 先独立采样 r 和 t，然后保证 t >= r（交换排序）
            # 为了保持与论文一致的行为，这里先独立采样两个同分布变量
            # 然后按元素比较，保证 t >= r
            

        mean = self.noise_sampler_config.get('mean', 0.0)
        std = self.noise_sampler_config.get('std', 1.5)
        r_samples = torch.full((num_noise,), 0, dtype=torch.float32,
                                 device=device) 
        r_samples=r_samples.normal_(mean=mean, std=std)
        r_samples = torch.sigmoid(r_samples)
        t_samples = torch.full((num_noise,), 0, dtype=torch.float32,
                                 device=device)
        t_samples=t_samples.normal_(mean=mean, std=std)
        t_samples = torch.sigmoid(t_samples)
            

        t = torch.max(r_samples, t_samples)
        r = torch.min(r_samples, t_samples)

            # 强制按照 meanflow_r_ne_t_ratio 的比例保留 r < t，其余置为 r == t
        if 0.0 < self.meanflow_r_ne_t_ratio < 1.0:
            mask = torch.rand((num_noise,), device=device) < self.meanflow_r_ne_t_ratio
            r = torch.where(mask, r, t)

        return t, r
       




    def add_noise(self, original_samples, noise, timesteps):
        """
        根据给定的时间 t 将干净样本 x 与噪声 e 插值得到 z_t：
            z_t = (1 - t) * x + t * e

        参数说明：
        - original_samples: 干净样本 x，形状通常为 (B, ...)
        - noise: 噪声样本 e，形状与 x 相同
        - timesteps: t 张量，形状为 (B,) 或可广播到 x 的 batch 维

        返回：插值后的 z_t，类型与 x 一致
        """
        x = original_samples
        z1 = noise
        t = timesteps
        b = x.size(0)
        
        # 将 t 扩展到 x 的形状以便按元素计算
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        zt = (1 - texp) * x + texp * z1
        return zt.to(x.dtype)

    def step(self, model_output, t,r, sample):
        """
        单步求解（采样阶段使用）：

        - 输入：
            model_output: 模型输出，期望为平均速度 u（与论文 MeanFlow 对齐）
            timestep_ind: 当前时间索引（整数），用于从 self.timesteps 中获取 t
            sample: 当前的 z_t（即上面的 zt）

        - 计算：
            dt = t - prev_t
            z_{t-dt} = z_t - dt * u

        - 返回：一个包含 prev_sample 的简单对象（兼容原代码习惯）

        注意：这里假设模型输出就是平均速度 u；若模型输出表示其他量
        （例如噪声残差 epsilon），需要在模型或调用方做对应转换。
        """
        zt = sample
        vc = model_output


        dt = r - t
        pred_prev_sample = zt - dt * vc  # z_t'（前一步的估计）

        return DummyClass(prev_sample=pred_prev_sample)

    def prepare_target(self, noise, gt):
        """
        为训练提供 denoising 目标。当前实现保持与原始代码库一致：
            target = noise - gt

        说明：严格的 MeanFlow 训练目标依赖于 u 的导数 du/dt（通过 JVP
        计算），并且需要在训练过程中访问模型函数来计算 JVP。如果要
        实现该目标，需要修改训练循环以传入 model 并在此处或训练里
        计算 du/dt，再组合出论文定义的 u_tgt。这里为了兼容保留了
        简单实现；若需要我可以帮你把训练流改为支持 JVP 计算。
        """
        return noise - gt


class DummyClass:

    def __init__(self, prev_sample):
        # 仅包装 prev_sample，方便与原接口兼容
        self.prev_sample = prev_sample

