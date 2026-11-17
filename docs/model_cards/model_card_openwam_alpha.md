# 模型卡：OpenWAM-α（`OpenWAM/OpenWAM-Alpha-Sim-LIBERO`）— LIBERO 世界模型评估用

> **口径**：本阶段只在虚拟环境内评测，"真实"指 LIBERO 模拟器真值分支。物理机器人不在范围内。
> **状态**：预检已关闭，进入正式接入（2026-10-23 裁决）。本卡记录事实；没有证据的写 `unverified`。

### 出处缩写

| 缩写 | 指向 |
|---|---|
| `[R]/x:N` | 本地仓库副本 `libero/openwam/third_party/OpenWAM`（`OpenWAM-Official/OpenWAM` @ `1e103e1`，即筛查时核对的提交），文件 x 第 N 行 |
| `[CK]` | 检查点目录 `[R]/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Sim-LIBERO/`（`config.yaml`、`README.md`、`checkpoint_step_10690.safetensors` 24.8 GB、`normalization_stats.npy`） |
| `[P]` | 论文 arXiv 2609.07398 |
| `[S]` | 本项目筛查记录 `libero/candidate_screening.md` §2.2 |

## A. 结构与骨干

| 项 | 内容 | 出处 |
|---|---|---|
| 类型 | 联合世界-动作模型（WAM）：视频与动作在一个去噪过程中联合生成；不是外部动作驱动的前向动力学模型 | `[S]`；`[CK]` README |
| 视频骨干 | Wan2.2-TI2V-5B（DiT 30 层，宽 3072，24 头），Wan-VAE（`WanVideoVAE38`）编解码，umT5-XXL 文本编码 | `[CK]` config.yaml `model.video_backbone` |
| 动作骨干 | 宽 1024，FFN 4096 | `[CK]` config.yaml `model.action_backbone` |
| 架构 | `dual_system / joint_self_attn`；注意力掩码 `mutual`（动作看全部视频，视频首帧之外看动作）；视频内部 `first_frame_causal` | `[CK]` config.yaml `architecture`；`[R]/openwam/model/architectures/utils/mask_modes.py:76-86` |
| 条件输入 | 单帧首帧（一张 320 × 384 的 L 形画布：agentview 320 × 256 在上，腕部 160 × 128 在左下，右下黑）＋ 任务指令（`task.language`，无模板）＋ 本体状态 | `[R]/openwam/deploy/obs_preprocess.py:189-209` |
| 本体状态 | 10 维原始 EEF：`[eef_pos(3), rot6d(eef_quat)(6), 夹爪开度(+1 张开)]`；min-max 归一化后经一个线性层编码为**一个上下文 token**。**只作输入，模型不预测状态**（无状态预测头；输出只有视频与动作） | `[R]/benchmarks/utils/action_conversion.py:492-518`；`[R]/openwam/model/architectures/base.py:506-545` |
| 输出（每次调用） | 32 个动作 + 9 帧视频（视频步长 4：第 k 帧 = 执行动作 0..4k−1 之后；第 0 帧是输入帧的 VAE 重建） | `[R]/openwam/model/architectures/base.py:1488-1604`；`lerobot_v3_reader.py:337` |
| 动作 | 80 维统一空间，只有前 10 维有训练信号：`[delta_xyz, rot6d(Exp(delta_axis_angle)), 夹爪开度指令]`；xyz 与旋转即 LIBERO 的归一化 OSC 指令。转成 LIBERO 7 维时夹爪取反 | `[R]/openwam/dataloader/libero.py:1-9`；`[R]/benchmarks/libero/openwam2libero_interface.py:37-56` |
| 采样 | flow-matching，部署默认 10 步去噪；视频噪声来自 CPU 生成器、动作噪声来自 CUDA 生成器，均由每次调用的 `seed` 决定 | `[R]/configs/deploy.yaml`；`[R]/openwam/model/video_backbone/wan/conditioning.py:38-51` |
| 精度 | bf16 | `[CK]` config.yaml `training.mixed_precision` |

### A 补充：可重置生成、联合一致性、帧可解码（三条结构闸门）

- **可重置生成：PASS**，`generate(..., first_frame_image=...)` 从任意真实画面出发（`[S]`）。
- **联合一致性：PASS（代码）＋ 数值抽查 PASS（2026-10-23）**。在同一起点、同一种子下，把动作固定（动作噪声级置 0，初始动作替换为指定值，**仅为测试补丁**），喂"模型自己的动作"和"xyz 取反的动作"，生成画面第 1–8 帧的 PSNR 为 24.9 → 19.8 dB，逐像素差 3.8–9.9；同一动作两遍逐位一致。画面确实依赖动作。
- **帧可解码：PASS**，部署配置默认 `decode_video: false`，打开即可（`[R]/configs/deploy.yaml:26`）。

### A 补充：本项目的使用口径（预检）

- **确定性**：关闭 DiT 缓存（相似度跳步近似）与 torch.compile；注意力用 SDPA；每块种子 = 基础种子（20261035）+ 块序号。同种子两遍逐位一致（见上）。
- **长程想象（训练外外推）**：代码不提供多块续滚。本项目按 17 次调用链式拼接到 H = 520（17 × 32 = 544）：下一块从上一块最后一帧出发，每块 32 个动作全部使用。官方闭环评测每块只执行 10 个动作即用真实观测重新规划。
- **块间本体状态（拼接口径）：标定控制器（2026-10-23 裁决）。** 模型不预测状态，纯想象中块间状态必须由我们提供。三口径对比（3 个锚）真值成功：标定控制器 3/3、简单积分 0/3、沿用初始状态 0/3（`libero/openwam/precheck_report.md` §2）。
  - **控制器模型**：位置与世界系姿态增量各为一阶 ARX，夹爪一阶跟随；拟合数据只来自真实重放（lingbot_base 100 条，70 条拟合 / 30 条留出，种子 20261036）。
  - **留出验证误差**（从块边界的真实状态出发，用真实指令滚 32 步）：位置中位数 5.0 cm（90 分位 9.9 cm），姿态中位数 3.2°（90 分位 8.4°）；对照"简单积分"52.8 cm / 34.4°，"沿用初始状态"6.3 cm / 9.1°。
  - **登记**：参数与拟合数据的 SHA 写入 protocol.yaml `libero.openwam_alpha.chaining`。属于**我方注入部件**，表格脚注声明，与 WorldVLA 的约束重试同等对待。

## B. 训练任务与数据

| 项 | 内容 | 出处 |
|---|---|---|
| 预训练 | 5.185 亿帧（约 6,400 小时）：机器人数据 70%（真实 40%、仿真 30%），人类第一视角数据 30% | `[CK]` README |
| LIBERO 微调 | 数据集 `OpenWAM/LIBERO`：1693 条回合、40 个任务，即 Spatial、Object、Goal、10 四个 suite 合训，由 `lerobot/libero` 转换 | `[S]` |
| LIBERO-10 覆盖 | 全部 10 个任务在微调数据中，含本项目的 t5、t3 | `[S]` |
| 训练窗口 | 33 步（32 个动作）、视频步长 4、窗口步长 1 | `[CK]` config.yaml `dataloader` |
| 官方评测设置 | 相机 256 × 256，旋转 180°；settle 30 步；`libero_10` 时域 700 步；同步执行，每块执行 10 个动作；10 步去噪 | `[R]/benchmarks/libero/policy_config.yml`；`scheduler.py:803-829, 1140-1143` |
| 与本项目口径的差异 | 本项目锚库 settle 10 步、H = 520；扰动族 A/P 的变化范围在其微调数据中的覆盖 `unverified`（未做定向核实） | — |

## 未核实项（unverified）

- 官方对照成功率（论文报告值与本机复现）：预检未跑官方闭环评测。
- 族 A（物体移位）与族 P（初始位姿）在其预训练与微调数据中的覆盖。
- 长程链式拼接下的行为：属于训练外外推，由预检实测。
