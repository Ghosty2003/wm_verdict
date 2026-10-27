# WM-Verdict Benchmark
## Unified B-Family Image Degradation Protocol (v1.0)

**适用范围：** LIBERO-Long + Franka Panda，适用于所有参与比较的 Policy / World Model。

**目标：** 所有模型使用相同的图像劣化源码、强度、相机、处理分辨率、时间窗口和随机种子规则，保证跨模型实验条件可复现、可比较。

### 1. 统一代码版本

所有模型统一采用 LIBERO-plus 的以下版本，不允许自行改变图像劣化函数及参数。

- Repository: https://github.com/sylvestf/LIBERO-plus
- Commit: `4976dc30028e805ff8094b55501d532c48fec182`
- Source: `libero/libero/envs/env_wrapper.py`
- Expected SHA256: `e91d7b7b35cc3ad2b073606c99860def6b3ac43b66eba00ef0ddb6bfd8f39c3c`

运行前核验实际安装文件的 SHA256。若实现需要适配新接口，应保持原始扰动算法不变，独立记录适配代码版本。

### 2. 统一扰动类型与强度

| 类型 | Severity 2 | Severity 5 | Severity 9 |
|---|---|---|---|
| Gaussian Blur | σ = 2 | σ = 5 | σ = 9 |
| Fog | (1.0, 2.8) | (2.5, 2.0) | (4.5, 1.4) |
| Glass Blur | (0.7, 1, 3) | (1.1, 3, 2) | (2.2, 5, 1) |

对应 LIBERO-plus 原始 `noise` 参数：

| 类型 | Level 2 | Level 5 | Level 9 |
|---|---|---|---|
| Gaussian Blur | 12 | 15 | 19 |
| Fog | 32 | 35 | 39 |
| Glass Blur | 42 | 45 | 49 |

`noise=0` 为无扰动 Baseline。

这三个等级是 WM-Verdict 的统一测试档位，不是 LIBERO-plus 强制规定的档位组合。

### 3. 统一相机和分辨率

| 参数 | 统一设置 |
|---|---|
| Camera | `agentview_image` |
| Render resolution | 128 × 128 |
| Wrist camera | 保持无 B 扰动 |
| Corruption stage | LIBERO 原始相机观察输出后、模型专用预处理之前 |
| 图像数据 | 使用原版函数要求的 RGB / uint8 输入 |

不同模型允许使用各自 checkpoint 所需的图像缩放、归一化和其他标准预处理，但这些操作应发生在统一扰动之后。

禁止分别在 128×128、224×224、512×512 等不同模型输入图像上直接施加相同 severity 后，将结果视为同一个共享实验条件。

### 4. 统一时间窗口

采用 LIBERO-plus 原始持续扰动方式。

- Start: `t = 0`，包括 reset 后的初始观察。
- End: Episode termination。
- Frequency: 每次获得新的相机观察时施加。
- Duration: 整个 Episode 持续有效。
- Time unit: LIBERO environment step，而不是 Policy 推理次数或 Action Chunk 次数。

本版本不采用 t=50 才开始的 B 扰动。

如果后续需要研究中途相机故障，可另外建立 `B_delayed_onset` 实验组，不能与本协议结果混合统计。

### 5. 统一随机数与可复现性

Gaussian Blur 在固定输入和参数下通常是确定性的。

Fog 和 Glass Blur 使用随机数，因此在 WM-Verdict 中补充以下复现要求：

- 每条 Episode 固定并记录初始种子。
- 图像劣化使用独立的随机数流，不得改变仿真环境及 Policy 的随机数流。
- 随机数由固定的 Episode ID、环境步、相机和扰动配置确定。
- 不同模型对同一实验单元使用相同的随机生成规则。
- 保存随机种子、随机数算法版本和必要的随机状态或劣化图像。

不得使用依赖 Python 进程随机化的内置 `hash()` 生成跨运行种子。可以使用 SHA256 等稳定方法派生种子。

若为此需要给原函数增加 RNG 适配层，应记录适配代码，并验证其没有改变扰动算法。

### 6. 与 World Model 评测的关系

**Real Rollout：**

真实 LIBERO 仿真按照相同任务、相同初始状态运行。每次产生相机观察后应用 B，再将观察传给 Policy。保存未劣化与已劣化图像、动作、真实状态及任务结果。

**Imagined Rollout：**

采用相同 B 配置。对于需要将预测图像重新输入 Policy 的模型，在规定的观察接口实施相同劣化规则，避免遗漏或重复劣化，并记录 World Model 实际接收的信息。

固定动作回放与 Policy-in-the-Loop 想象必须分开报告。固定动作回放中，后续图像劣化不会改变预先锁定的真实执行动作，因此不能直接代表持续输入故障下的闭环 Policy 鲁棒性。

### 7. 每次实验必须保存的记录

至少记录：

`model_id`, `checkpoint`, `task_id`, `episode_id`, `initial_state_id`, `seed`, `libero_plus_commit`, `source_sha256`, `corruption_type`, `severity`, `noise_code`, `camera`, `render_resolution`, `start_step`, `end_step`, `rng_seed`, `policy_preprocessing`, `success`, `completion_step`。

同时保留：

- 原始观察与扰动观察的样本。
- Real Rollout 视频及状态、动作日志。
- Imagined Rollout 视频及其观察、动作条件（如果模型支持）。
- Verdict 结果及其任务判定方法。
- 环境、依赖版本和代码修改记录。

### 8. 运行前检查

正式运行前必须验证：同一原始图像、相同 severity 和相同随机种子能够产生一致的 B 输出；Baseline 不改变原始图像；B 只作用于指定相机；扰动在 t=0 及后续预定观察时刻生效。

若某模型不能满足其中一项，必须单独标记为协议偏离，不应直接纳入共享 B 族结果。

---

**Protocol Status:** WM-Verdict proposed common specification v1.0.

**Source-derived settings:** LIBERO-plus 的函数、参数映射、agentview 默认处理方式、128×128 默认分辨率及从 reset 开始的持续劣化。

**WM-Verdict additions:** Severity 2/5/9、独立随机种子管理、跨模型处理顺序、Imagined/Real 对齐和日志要求。

所有参与模型应使用此协议，不得单独调整 B 的强度、处理时机、分辨率或相机范围而不登记协议变更。

---

> **世界模型组注（2026-10-27）**：以上为 v1.0 原文，未改动。体检后的裁决：世界模型组首批将毛玻璃替换为缩放模糊（Zoom Blur，第 2/5/9 档 = noise 22/25/29），原因是毛玻璃参数表存在剂量缺陷；雾因不兼容 NumPy 2，在依赖固定的子进程中运行。执行口径见 `protocol_B_v1.0_execution_notes.md`，体检见 `healthcheck/report.md`。
