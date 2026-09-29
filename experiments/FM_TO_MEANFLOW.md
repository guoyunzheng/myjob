# FM 预训练与 MeanFlow 微调

以下命令在项目根目录、已安装训练依赖的 Linux 环境执行。脚本不会自动串联两阶段。

## 1. 一次 FM 训练，保留 24 万和 30 万步

```bash
SEED=0 RUN_LOG_DIR=fm_pretrain_300k bash train_flow_stages.sh fm
```

该入口训练 300000 步：全部训练时间对 `r=t`，位姿目标是 `noise-data`，不计算 JVP。
位姿 endpoint/IVC 辅助损失均为 0；夹爪损失和观察编码器仍正常训练。
主位姿回归指标沿用当前默认 L1；如需 L2，在两个阶段都显式设置 `FLOW_LOSS_TYPE=l2`。

在 240000 和 300000 步强制运行离线验证，并分别保存完整的：

- `train_logs/Peract/fm_pretrain_300k/step240000.pth`
- `train_logs/Peract/fm_pretrain_300k/step300000.pth`
- 同目录 `validation_step240000.json`、`validation_step300000.json`

每份检查点包含原始权重、EMA、优化器等训练状态。后续 `last.pth` 和 `best.pth` 的更新不会覆盖这两份快照。
常规验证默认仍每 4000 步执行；即使修改 `VAL_FREQ`，这两个指定步数仍会验证和保存。
FM 验证使用 `u(z,t,t)` 做 Euler 积分。默认验证步数为 2，可通过 `DENOISE_TIMESTEPS` 调整。
验证报告记录目标类型、采样步数、EMA/raw、每任务指标及检查点选择分数；这些是离线动作指标，RLBench 成功率需运行下方仿真评估。

## 2. 每个新实验从固定 FM 24 万步权重开始

```bash
INIT_FROM=train_logs/Peract/fm_pretrain_300k/step240000.pth \
INIT_WEIGHTS=raw SEED=0 RUN_LOG_DIR=mf_from_fm240k_off075 \
bash train_flow_stages.sh meanflow
```

这个新实验训练 60000 步，默认 `MEANFLOW_OFFDIAG_RATIO=0.75`：约 75% 样本使用 `r<t` 的完整 MeanFlow JVP 目标，25% 使用 `r=t`。
默认 endpoint 权重 0，IVC 权重 0.5，额外维护瞬时速度边界；需要只比较主流目标时可显式设 `IVC_LOSS_WEIGHT=0`。
保存 `train_logs/Peract/mf_from_fm240k_off075/step60000.pth` 及 `validation_step60000.json`。
此处检查点步数是新实验的 60000，训练来源在 `run_metadata` 中记录为 FM 240000。

`INIT_FROM` 仅加载所选权重并保留归一化参数；优化器、EMA 累计状态和学习率计划重新开始。
默认使用原始权重，也可统一改用 `INIT_WEIGHTS=ema`。各对照实验应使用同一份源检查点和同一权重选择。
第一阶段 cosine 的总长度为 300000，第二阶段为独立的 60000；可通过 `LEARNING_RATE` 设置第二阶段初始学习率。
修改代码后开始新的 MF 实验时，请换一个 `RUN_LOG_DIR`，继续使用同一个 `INIT_FROM`。

只有继续同一配方、同一代码的中断训练才用 `RESUME`，例如：

```bash
RESUME=train_logs/Peract/fm_pretrain_300k/last.pth \
RUN_LOG_DIR=fm_pretrain_300k bash train_flow_stages.sh fm
```

MF 同理使用 `RESUME=.../last.pth` 和原来的日志目录，清除 `INIT_FROM`。
模型结构或输入表示改变时，当前严格权重加载可能拒绝复用，需要另行处理参数迁移。

## 3. 独立复测 FM 24 万、FM 30 万、MF 24+6 万

离线复测（使用 EMA；示例都用 2 次网络调用便于比较）：

```bash
EVAL_ONLY=true FLOW_OBJECTIVE=fm DENOISE_TIMESTEPS=2 \
CHECKPOINT=train_logs/Peract/fm_pretrain_300k/step240000.pth \
RUN_LOG_DIR=eval_fm240k_n2 bash train.sh

EVAL_ONLY=true FLOW_OBJECTIVE=fm DENOISE_TIMESTEPS=2 \
CHECKPOINT=train_logs/Peract/fm_pretrain_300k/step300000.pth \
RUN_LOG_DIR=eval_fm300k_n2 bash train.sh

EVAL_ONLY=true FLOW_OBJECTIVE=meanflow DENOISE_TIMESTEPS=2 \
CHECKPOINT=train_logs/Peract/mf_from_fm240k_off075/step60000.pth \
RUN_LOG_DIR=eval_mf240plus60k_n2 bash train.sh
```

每次复测在对应 `train_logs/Peract/<RUN_LOG_DIR>/evaluation.json` 写出结果。
要测试一步生成，将 `DENOISE_TIMESTEPS=1`，并更换报告目录；建议同时比较 FM/MF 的 1、2、5 步。
FM 始终查询 `(t,t)`；MeanFlow 查询 `(r,t)`，一步时为 `(0,1)`。

RLBench 仿真推理入口（`my3d.sh` 中保留了本机原有 Conda/CoppeliaSim 路径，需在相应机器运行）：

```bash
CHECKPOINT=train_logs/Peract/fm_pretrain_300k/step240000.pth \
DENOISE_MODEL=fm DENOISE_TIMESTEPS=2 CHECKPOINT_ALIAS=fm240k_n2 bash my3d.sh

CHECKPOINT=train_logs/Peract/fm_pretrain_300k/step300000.pth \
DENOISE_MODEL=fm DENOISE_TIMESTEPS=2 CHECKPOINT_ALIAS=fm300k_n2 bash my3d.sh

CHECKPOINT=train_logs/Peract/mf_from_fm240k_off075/step60000.pth \
DENOISE_MODEL=meanflow DENOISE_TIMESTEPS=2 CHECKPOINT_ALIAS=mf240plus60k_n2 bash my3d.sh
```

检查点目标类型与推理目标不匹配时会报错，避免误用 FM/MF 路径。
更换动作头或夹爪配置的实验，复测时也需要传入相同的结构参数；以上命令对应脚本默认 FiLM-TCN 配置。

## 4. 只查看启动参数

```bash
PYTHON_BIN=echo bash train_flow_stages.sh fm
PYTHON_BIN=echo INIT_FROM=/path/to/step240000.pth bash train_flow_stages.sh meanflow
```

这两条仅打印命令，不启动训练。若使用 `train_systemd_limited.sh`，需显式传入上述模式的 `FLOW_OBJECTIVE`、`TRAIN_ITERS`、`MILESTONE_CKPT_STEPS`、损失权重和 `INIT_FROM`；两阶段入口默认直接调用 `train.sh`。
