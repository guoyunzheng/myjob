# 第九步：可选的 boundary-reuse iMF

实现依据：[Improved Mean Flows，§4.1 / Algorithm 1](https://arxiv.org/html/2512.02012v1)。
这里只实现其无额外速度头的条件目标；不是整套图像生成配方的复现，也没有机器人成功率结论。

## 计算与梯度边界

延续当前约定：`z_t=(1-t)*data+t*noise`，`v_c=noise-data`，采样从 1 到 0。

```text
v_boundary = u(z_t, t, t, fixed_condition)
dudt       = JVP(u, (z_t,r,t), (v_boundary,0,1))
V          = u(z_t,r,t,condition) + stopgrad((t-r)*dudt)
loss_pose  = 30*metric(V_xyz, v_c_xyz) + 10*metric(V_rot6d, v_c_rot6d)
```

- iMF 的 JVP 方向不接收 `noise-data`，也不遍历观察编码器；固定条件在普通预测中仍有梯度。
- 修正项用 FP32 精确 JVP、no-grad 和独立小批次计算；只处理 `r!=t` 的样本。对角样本修正严格为零，没有额外前向。
- 沿 `(dr,dt)=(0,1)`，动作头的 `(t,h=t-r)` 方向为 `(dt,dh)=(1,1)`；不是固定姿态的偏时间导数。
- 推理和 endpoint 辅助损失仍积分 **u**，不能积分训练复合量 **V**。guidance=1 时，N 步仍是 N 次动作头调用。
- TCN / Transformer、direct / legacy_denoise 夹爪均可用；无新增模型参数、无新夹爪先验、无执行门控修改。
- `training/flow/imf_correction_rms` 是每个重复噪声批次的全批次 RMS 的平均；`imf_offdiag_fraction` 是实际非对角比例。它们不是损失项。

## 配置边界

| 配置 | 原 FM / MeanFlow | 可选 iMF |
| --- | --- | --- |
| `flow_objective` | `fm` / `meanflow` | `imf` |
| 默认 `flow_loss_type` | `l1`，保持不变 | `l2`，逐元素平方误差的均值 |
| 默认 offdiag 比例 | FM=0，MeanFlow=0.25 | 0.25，沿用 MeanFlow 时间采样 |
| `train.sh` 默认 endpoint / IVC | FM=0.25/0，MF=0.25/0.5 | 0/0，用于隔离目标 |
| 原始 Python CLI / Actor 的 endpoint / IVC 默认 | 0.25/0 | 仍为 0.25/0，需显式设为 0/0 做纯目标对照 |

`l1` / `l2` 都可显式用于三个目标；这不会改变 gripper BCE、endpoint 或 IVC 的定义。
iMF 使用 L1 时属于控制消融：L1 的回归最优量是条件中位数，不能直接等同于论文 L2 的边际均值论证。

限制：iMF 只允许 `guidance_scale=1`、`condition_dropout_prob=0`、`use_compile=false`。
本步不包含辅助 v-head、自适应损失重加权、CFG 蒸馏、guidance token 或新的 Transformer 结构。
训练修正不能在 `torch.inference_mode()` 下计算（该模式关闭前向 AD）；`torch.no_grad()` 可用。
完整推理不调用此修正，因此可以正常使用 inference_mode。

不要直接把 offdiag 比例设成 1 并关闭 IVC：这会失去显式的对角边界监督。
设成 0 又无法充分监督长区间平均速度。0.25 只是继承的起点，不是验证过的最优值。

## 启用与比较

以下只打印参数，不训练、不创建日志目录；夹爪配置沿用脚本配置，不能据此认定它最优：

```bash
PYTHON_BIN=echo ACTION_HEAD=transformer FLOW_OBJECTIVE=imf \
  SEED=0 MATMUL_PRECISION=ieee RUN_LOG_DIR=step9_imf_preview \
  bash train.sh
```

正式运行前明确选择并锁定同一夹爪 profile（9.5 两次修改导致成功率下降的反馈不变）：

- `compat`：`GRIPPER_PREDICTION_MODE=direct GRIPPER_LOSS_TYPE=weighted_bce GRIPPER_TRANSITION_WEIGHT=2 GRIPPER_CLOSED_HOLD_WEIGHT=2 GRIPPER_HOLD_PRIOR_LOGIT=2`。
- `plain`：`GRIPPER_PREDICTION_MODE=direct GRIPPER_LOSS_TYPE=bce GRIPPER_TRANSITION_WEIGHT=0 GRIPPER_CLOSED_HOLD_WEIGHT=0 GRIPPER_HOLD_PRIOR_LOGIT=0`。

确认数据、profile、资源预算和独立输出目录后，去掉 `PYTHON_BIN=echo` 才会真正训练；本次修改未执行此操作。
systemd 包装脚本把未设定的 loss / endpoint 交给 `train.sh` 按目标解析，不会把 iMF 偷改回 L1 + endpoint=0.25。

与第八步结果比较有两种不同问题，必须分开标记：

1. **只比较 JVP 方向**：沿用第八步所有设置，iMF 显式 `FLOW_LOSS_TYPE=l1 ENDPOINT_LOSS_WEIGHT=0 IVC_LOSS_WEIGHT=0`，与同架构 MeanFlow-L1 比较。
2. **测试 L2-iMF**：同架构重跑 FM-L2 / MeanFlow-L2 / iMF-L2，各组显式 `FLOW_LOSS_TYPE=l2 ENDPOINT_LOSS_WEIGHT=0 IVC_LOSS_WEIGHT=0`；不能把旧 L1 结果当作“只改目标”的对照。

保持数据快照、种子、训练预算、最终 EMA、NFE=1/2/5、任务 / variation / episode 覆盖与执行设置一致。
MeanFlow 与 iMF 共用同一时间采样；FM 的 t 边缘分布仍不同。相同更新数也不等于相同训练耗时，iMF 非对角样本会多一次边界前向。
旧第八步四组生成器和汇总器仍保持四组，**不会自动纳入 iMF**；不要手改旧 manifest 塞入新结果。
本次源文件变更后，旧 preview manifest 的 source hash 会失效，需要重新 plan；不要绕过该检查。

从同架构 FM/MF 权重开始只能使用 `--init_from ...`（或 `INIT_FROM=...`），这是新实验；目标 / metric 改变不能用 strict resume，也不能直接把旧 checkpoint 当成 iMF 评估。
从 FM 转入区间目标时需要重新训练非零区间条件，不能认为同形状权重天然兼容。

本机仅验证 CPU 的解析公式、实际动作头、梯度和接口。尚未完成全尺寸 CUDA / CLIP / RLBench 验证，不能推断训练收敛或任务成功率。
