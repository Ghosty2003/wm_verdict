# Model Card：LaWAM（LeRobot 集成，LIBERO SFT checkpoint）

> **口径（2026-10-14）**：本阶段只在虚拟环境内评测。文中的“真实”“现实”“real”一律指 **LIBERO 模拟器真值分支**，“迁移”指想象结论到模拟器执行的迁移。物理机器人不在范围内。本档案引用论文中的“真实世界 / real-world”实验时，指的是论文作者的物理机器人实验，与本项目的真值分支无关。

> 用途：LIBERO 上 world-model verdict 评测的第一步，基于证据整理的模型卡。
> 编写日期：2026-10-14。所有条目都标注出处；找不到证据的写 `unverified`；来源互相矛盾时两边都列出。

**出处缩写**

| 缩写 | 指代 |
|---|---|
| `L/…` | 本地代码 `$LEROBOT_ROOT/src/lerobot/policies/lawam/…`（file:line） |
| `ENV/…` | 本地代码 `$LEROBOT_ROOT/src/lerobot/envs/…` 或 `…/processor/…` |
| `DOC-mdx` | `$LEROBOT_ROOT/docs/source/lawam.mdx` |
| `DOC-README` | `$LEROBOT_ROOT/docs/source/policy_lawam_README.md` |
| `HF-SFT` | `jialei02/lawam-libero-sft-lerobot`：README / `config.json` / `conversion_manifest.json` / `policy_{pre,post}processor.json`，以及 `/api/models`（safetensors 参数统计） |
| `HF-PRE` | `jialei02/lawam-pretrain-lerobot`：同上 |
| `HF-COLL` | collection `jialei02/lawam-checkpoints`（包含 `lawam_lam`、`lawam_pretrain`、`lawam_libero_sft_release`、`lawam_robotwin_sft_release`）及这些原始仓库里的小文本文件：`lawam_lam/dino_large_vae.yaml`、`lawam_pretrain/config.yaml`、`lawam_pretrain/dataset_statistics.json`、`lawam_libero_sft_release/{README.md,config.yaml,dataset_statistics.json}` |
| `HF-DS` | 数据集 `jialei02/libero_merged_no_noops_20hz`：README、`meta/info.json`、`/api/datasets` |
| `PAPER` | Chen et al., *LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies*, **arXiv:2606.15768**（HTML 版，§ 为论文章节号） |
| `GH/…` | 上游仓库 `github.com/RLinf/LaWAM`（main 分支）：README.md、`starVLA/config/training/train_libero.yaml`、`starVLA/dataloader/gr00t_lerobot/mixtures.py`、`examples/LIBERO/eval_files/{libero_eval_core.py,libero_benchmark_adapters.py}` |

**名称说明**：LaWAM = **La**tent **W**orld **A**ction **M**odel（GH/README.md 第 1 段）。“La”指 latent（潜空间）。模型内部依次预测三样东西：(1) 一个连续的 *latent action* ẑ；(2) 由 ẑ 解码出的 *未来帧 DINOv3 特征图* û_T，即“latent visual subgoal”；(3) 机器人动作块。其中 (2) 这部分叫 LaWM（Latent World Model）（PAPER §3.1 式(2)(3)、§3.2）。

---

## A. 结构与骨干

| 字段 | 内容 | 出处 |
|---|---|---|
| **骨干来源** | **不是从零训练，是在预训练模型上微调**，共三层预训练：<br>① **VLM 骨干**：`Qwen/Qwen3-VL-2B-Instruct`（固定 revision `89644892…`），只保留 LLM 的**前 16 层**（原模型 28 层），删掉 `lm_head`。SFT 时视觉塔、LLM 都**可训练**（`freeze_vision_backbone=false`、`freeze_llm_backbone=false`），只冻结词嵌入（`freeze_embedding=true`）。<br>② **视觉编码器**：DINOv3 ViT-B/16（`facebook/dinov3-vitb16-pretrain-lvd1689m`），**始终冻结**。<br>③ **LAM/LaWM**（`jialei02/lawam_lam`，即 `dino_large_vae`）：先单独预训练，见 B 节。到策略阶段，LAM 的 encoder（逆动力学）、VAE 瓶颈、DINOv3 全部 `requires_grad=False`；**只有 LAM decoder（=LaWM）解冻、参与训练**（`unfreeze_lam_decoder=true`）。<br>④ 训练链：LaWM 预训练 → LaWAM 策略预训练（`jialei02/lawam_pretrain` → `lawam-pretrain-lerobot`）→ LIBERO SFT（`lawam_libero_sft_release` → `lawam-libero-sft-lerobot`）。 | HF-SFT `config.json`：`base_vlm`、`keep_llm_first_n_layers: 16`、`freeze_*`、`unfreeze_lam_decoder`；HF-SFT `conversion_manifest.json`：`source_revisions`；Qwen `config.json`：`text_config.num_hidden_layers: 28`；L/vlas/qwen3vl.py:70-73（删 lm_head）、83-93（截断层数）、109-136（冻结逻辑）；L/latent_world/runtime/freeze_policy.py:88-94；L/lam_core/core/lam_model.py:160-163（LAM 全部冻结）；L/lam_core/core/vjepa_encoder.py:23-42（DINOv3 冻结）；GH/README.md “Model Preparation” 表；PAPER §C.2（“first 16 transformer layers of Qwen3-VL”）、§C.1（“distilled DINOv3 ViT-B/16”）；GH/train_libero.yaml：`pretrained_checkpoint: …/lawam_pretrain/…` |
| **生成机制** | 三段，**都没有自回归 token，也没有离散 token**：<br>• **Latent action ẑ**：8 个 `<ACT_PH>` query token 注入 Qwen 输入序列，取对应 hidden state，经 1 层 QFormer（`VLMToLAMQFormer`）回归成 1×32 的连续向量。属于**确定性回归**，用 MSE 蒸馏 LAM encoder 给出的 teacher z。<br>• **LaWM（未来特征预测）**：`LAMDecoderV2` 是 12 层 AdaLN-Zero Transformer，输入当前帧 DINOv3 patch 特征 u（256×768）和条件向量 ẑ，**单次前向、确定性地**输出未来帧特征 û_T（256×768）。**不是扩散，也不是 flow matching。**<br>• **动作**：连续 **flow matching**（rectified-flow 线性路径 x_t=(1−t)z+tx，网络预测速度），用 DiT/AlternateVLDiT 实现；推理时 Euler 积分 10 步，CFG scale=1.0（即不做 CFG）。<br>• LAM 瓶颈是**连续 VAE**（`VAEQuantizer`，名字里有 Quantizer 但不做量化），推理时取 μ。 | L/vlas/lawam.py:83-140（QFormer）、362-432（注入 query、得到 pred_latent）、494-511（decoder 单次前向）、593-598；L/lam_core/core/utils/lam_decoder.py:27-84（AdaLNBlock）、87-169；L/lam_core/core/vq.py:20、54、60-63；L/vlas/flowmatching_expert.py:120-127（说明）、386-388（路径/速度目标）、527（MSE）、646-752（Euler 采样）；HF-SFT `config.json`：`flow_num_inference_steps: 10`、`flow_cfg_guidance_scale: 1.0`；PAPER §3.1 式(3)（“deterministically decodes”）、§3.3、§1（“single forward pass”） |
| **参数规模** | **总量（HF safetensors 统计）：2,555,179,360**，其中 BF16 1,523,500,032、F32 1,031,679,328。按模块拆分如下：<br>• **VLM（Qwen3-VL-2B 截断到 16 层）≈1,523.5M**。推算：2,127,532,032（Qwen 全量）− 12 层×50,336,000 = 1,523,500,032，与 BF16 计数完全一致。<br>• **LAM 全体 725,095,008**（按本地代码在 meta device 上实例化统计）：DINOv3 ViT-B 85.66M（冻结）、LAM encoder 309.43M（冻结，只用作 teacher）、**LAM decoder = LaWM 228.29M**、state_decoder 101.71M、VAE 2.2K。<br>• **Flow 动作头 ≈306.4M**：F32 总量 − LAM − QFormer(0.146M) − query(0.033M) 推算得出。<br>• **论文给的数字不同**：总量 2.3B、LaWM 230M（230M 与 decoder 228.29M 吻合，总量与 HF 的 2.56B **不一致**）。<br>• 视频部分：**没有视频生成器或 VAE tokenizer**。“视频部分”就是 LaWM decoder 228M + 冻结的 DINOv3 86M。 | HF-SFT/HF-PRE `/api/models` → `safetensors.parameters`；Qwen `/api/models/Qwen/Qwen3-VL-2B-Instruct`（total 2,127,532,032）+ `config.json`（hidden 2048、inter 6144、28 层、kv heads 8、head_dim 128）；LAM 计数来自 L/lam_core/core/lam_model.py:29-123 按 HF-SFT `config.json` 的 `lam_*`/`dinov3_*` 参数实例化；PAPER §1、§4.1 Table 1（“2.3B”）、Fig.1 |
| **表示与分辨率** | • 输入帧统一 resize 到 **256×256**（两路相机都如此），LAM 输入另做 ImageNet 归一化。<br>• latent 形状：DINOv3 **倒数第 2 层** patch token，去掉 CLS 和 4 个 register token，经无仿射 LayerNorm 后得到 **16×16=256 个 token × 768 维**。LaWM 输出形状相同（[B,1,256,768]）。<br>• latent action：1 个 query × **32 维**（`lam_code_dim`）。<br>• **每个预测 latent 帧覆盖的环境步数**：SFT 的物理时距 τ=0.4 s，20 Hz 下等于 **8 步**（上游 `sec_chunk: 0.4`、manifest `flow_horizon_sec: 0.4`）。但 LeRobot 适配器训练 teacher 时取的帧下标是 `[0, 7]`（`round(i·(action_horizon−1)/(num_video_frames−1))`），即 **7 步 = 0.35 s**。**两处来源不一致，都列出。** 预训练阶段 τ=1.2 s（24 步@20Hz）。<br>• 推理时只用当前一帧：LAM 只取 `primary_video[:,0]`，Qwen 每路相机也只取第 0 帧。 | L/processor_lawam.py:131-147、161-202、285-292、447-453、475；L/latent_world/batch_utils.py:21-48（ImageNet 归一化）；L/lam_core/core/lam_model.py:64-72、146-151；L/lam_core/core/vjepa_encoder.py:83（encode，去掉 prefix token）；L/configuration_lawam.py:104、267-276；HF-SFT `config.json`：`lam_image_hw [256,256]`、`lam_latent_layer_to_use -2`、`lam_norm_latents_type ln`、`num_video_frames 2`；HF-SFT `conversion_manifest.json`：`flow_horizon_sec 0.4`、`action_hz 20.0`；GH/train_libero.yaml：`sec_chunk: 0.4`、`image_resolution: 256`；PAPER §C.2（“images resized to 256×256”）、§C.3 |
| **动作接入方式** | • **LaWM（世界模型）不接收机器人动作**，它的条件只有 latent action ẑ（32 维，经 `action_in_proj` 投到 1024 维后用 AdaLN 调制）。<br>• 机器人动作只出现在 flow 动作头：作为**输出/去噪目标**（`action_encoder(x_t)`）。<br>• 动作维度：LIBERO 7 维（xyz、axis-angle 3 维、gripper），**补零到 32 维**（`flow_action_dim=32`）并配 mask。<br>• 归一化：ACTION 用 MIN_MAX 归一到 [-1,1]；LIBERO 配方在归一化后把 gripper（第 6 维）snap 成 0/1。<br>• 物理时间编码：动作 token i 的时间为 t=i/hz（hz=数据集 fps=20），做 sinusoidal 编码。<br>• **不输入本体状态**（`flow_use_state=false`，state 被从 processor 中移除）。 | L/lam_core/core/utils/lam_decoder.py:128-139；L/vlas/flowmatching_expert.py:45-61、94-106、394-404；L/processor_lawam.py:419-441（补零/mask）、402-417、571-579（不用 state）、593-599（pre-snap）；L/configuration_lawam.py:79-85；HF-DS `meta/info.json`（action shape [7] 及字段名）；HF-SFT `config.json`：`flow_action_dim 32`、`flow_use_state false`、`gripper_dim 6`；DOC-mdx 第 61、65 行；PAPER §C.2（“we do not provide proprioceptive state inputs”；动作统一转为 EEF 表示）、§C.3 |
| **上下文与预测窗口** | • 历史长度：**1 帧**（`n_obs_steps=1`，观测和 state 的 delta 都是 `[0]`）。<br>• 每次前向：预测 **1 个**未来 latent 帧（τ 时刻），外加 1 个动作块。动作块按 chunk_size=50 padding，有效长度 `action_horizon=8`。<br>• 长时 rollout 拼接：**LeRobot 代码里没有**多步想象或 latent 回灌的接口。论文 §4.4、App. D.4/Fig.10 提到在 500 条 LIBERO 轨迹上做 “extended open-loop rollouts”，但具体机制（是否把 û_T 当下一步的 u 回灌、latent action 从哪来）未公开 → `unverified`。<br>• **能否当作动作条件模拟器（输入外部动作 → 输出未来帧或 latent）**：**不能直接用外部机器人动作**。理由：`predict_action` 只返回动作（L/vlas/lawam.py:732-768），û_T（`h_t1_pred`）只存在于内部 `PolicyEncodingState`；LaWM 的条件是 32 维 latent action，代码里**没有“机器人动作 → latent action”的映射**。<br>• 可能的替代路径（代码层面存在，但 LeRobot 没有公开 API，只能调私有方法）：(a) 直接调 `lam.decoder(h_t, z)`，传入**外部 latent action z**（L/lam_core/core/utils/lam_decoder.py:136；L/vlas/lawam.py:506）；(b) z 可由 `lam.get_latent_action(videos=[当前帧, 目标帧])` 从**一对真实帧**反推（L/lam_core/core/lam_model.py:125-144）。所以它只能算“latent-action 条件”的单步特征预测器，不是 action-conditioned simulator。论文 §4.4/Fig.5 正是用“从源视频抽出的 latent action”做 open-loop 迁移 rollout。 | L/configuration_lawam.py:73-77、262-286；L/modeling_lawam.py:165-168、248-280；L/vlas/lawam.py:552-609、732-768；HF-SFT `config.json`：`n_obs_steps 1`、`chunk_size 50`、`action_horizon 8`；PAPER §4.4、App. D.4 |
| **自带头** | • **Policy 头**：flow-matching 动作专家（AlternateVLDiT，16 层，hidden 1024，16 头；VLM 文本流和动力学流 (u, û_T) 交替做 cross-attention）。<br>• **latent-action 头**：`act_query` + `VLMToLAMQFormer`，用 MSE 蒸馏 teacher z。<br>• **world 头**：LaWM decoder（由 LAM 预训练初始化，SFT 继续训练）。<br>• LAM 内含 `StatePredictor`（辅助状态预测），只在 LaWM 预训练阶段使用（论文称训练后丢弃；本地代码仍实例化但不调用）。<br>• **reward 头 / value 头：无**（代码中不存在；`reward_delta_indices` 返回 None）。 | L/vlas/flowmatching_expert.py:169-237；L/vlas/cross_attention_dit.py:215、342-349、404；L/vlas/lawam.py:283-295、320-323；L/lam_core/core/lam_model.py:99-106；L/configuration_lawam.py:288-291；PAPER §3.2（“After pretraining, we discard this auxiliary head”）、§3.3、§C.2 |
| **训练目标** | **SFT/策略阶段（LeRobot）**：`loss_total = loss_flow + 0.1·loss_perceptual + 0.1·loss_distill`<br>• `loss_flow`：flow-matching 速度 MSE（带 mask）；每个样本重复 2 次噪声采样（`repeated_diffusion_steps=2`）。<br>• `loss_perceptual`：MSE(û_T, u_T)，监督 LaWM 预测与真实未来 DINOv3 特征一致。<br>• `loss_distill`：MSE(ẑ, z_teacher)，z_teacher 由冻结的 LAM encoder 用 μ 计算。<br>• Knowledge Insulation：flow 头使用的 û_T 被 `detach`（`detach_future_feature=true`），动作梯度不回传到 LaWM。<br>**LaWM 预训练阶段**（论文式 4）：L_wm（特征 L2）+ L_aux（状态预测 L2）+ β·KL，β=1e-5。上游 yaml 写的 `loss_type: smooth_l1`、`state_loss_type: l2`，与论文写法不完全一致，两者都列出。 | L/vlas/lawam.py:464-492、639-730（尤其 668-671、717-721）、527-534；L/vlas/flowmatching_expert.py:386-388、525-533；HF-SFT `config.json`：`perceptual_weight 0.1`、`lam_encoder_distill_weight 0.1`、`repeated_diffusion_steps 2`、`detach_future_feature true`；PAPER §3.2 式(4)、§3.3 式(5)、§C.5；HF-COLL `lawam_lam/dino_large_vae.yaml`：`loss_type: smooth_l1`、`state_loss_type: l2` |
| **能否解码出画面** | **不能。** 发布的 LAM 把 `train_in_latent=True` 写死，decoder 只输出 DINOv3 特征（[B,1,256,768]）。`to_pixel`（ConvTranspose2d）只在 `train_in_latent=False` 时才会构建，而发布配置走不到这个分支，checkpoint 里也没有像素解码权重。整个模型中没有 VAE 或像素 decoder。论文的可视化方式是计算 DINO 特征余弦相似度热图或 PCA，**不重建像素**。 | L/lam_core/core/lam_model.py:85-98（`train_in_latent=True`）；L/lam_core/core/utils/lam_decoder.py:133-134、157-169；PAPER §1（“instead of reconstructed future video”）、App. B（“rather than reconstructing future pixels”） |

---

## B. 训练任务与数据

### B.1 SFT checkpoint 使用的 LIBERO 子集
- **LIBERO-Spatial、Object、Goal、Long(10)，共 40 个任务**；**没有 LIBERO-90**。
  - 证据：PAPER §D.1（“train on four suites: LIBERO-Spatial, -Object, -Goal, -Long … 40 tasks”）；HF-DS `meta/info.json`：`total_tasks: 40`；GH/mixtures.py 中 `"libero"` mixture 只含 `libero_merged_no_noops_20hz`，`libero_90_no_noops_lerobot` 一行被注释掉；HF-COLL `lawam_libero_sft_release/config.yaml`：`data_mix: libero`。
  - 补充：上游还定义了 `"libero_all"` mixture（含 `libero_plus_lerobot`），但发布的 SFT 配置**没有用它**（GH/mixtures.py、GH/train_libero.yaml `data_mix: libero`）。

### B.2 训练数据来源与规模（SFT）
- 数据集 `jialei02/libero_merged_no_noops_20hz`：**1,693 条 episode，273,465 帧，fps=20**，折合约 **3.80 小时**（273,465/20/3600，本卡推算）。robot_type=franka，LeRobot v3.0 格式（HF-DS `meta/info.json`）。
- 来源：由 `IPEC-COMMUNITY/libero-benchmark-dataset` 派生，把四个子集合并后转为 LeRobot 3.0（GH/README.md “LIBERO SFT”；HF-COLL `lawam_libero_sft_release/README.md`）。来源是**官方 LIBERO 人类示教的再处理版本**；IPEC 版本本身的生成方式 `unverified`。
- no-noops：数据集名称中有 “no_noops”；PAPER §D.1 写的是“按 OpenVLA 设置**去掉失败示教**”。no-op 动作的过滤规则、轨迹是否在仿真中重新生成 → `unverified`（任一来源都没有明确说明）。
- 20 Hz：PAPER §C.4 称 LIBERO 轨迹本身是“native 20 Hz”，即**没有重采样**；info.json `fps: 20`。上游另有 5/10 Hz 降采样版本，只用于 §C.4 的对照实验（GH/mixtures.py `libero_mix_hz`），发布的 SFT 没有使用。
- Counterfactual 或额外合成数据：**无证据** → `unverified`（倾向于没有，但不作推断）。
- SFT 训练量：PAPER §D.1 写 **25k steps，global batch 256**；GH/train_libero.yaml 为 `max_train_steps: 25000`、8 GPU×32。LeRobot 的 HF-SFT `config.json` 里 `scheduler_decay_steps: 30000`（HF-PRE 为 25000）。这是调度参数，不等于实际训练步数，与论文的 25k **存在差异，两者都列出**。

### B.3 LIBERO 之外的预训练数据
- **LaWM（LAM）预训练**：约 **3,000 h 机器人视频 + 1,500 h 第一人称人类视频**（PAPER §1、§4）。论文 §4 引用的数据源：EgoDex[15]、Ego4D[14]、LEGO[22]（人类）；AgiBot World[1]、RoboMIND[38]、RoboCOIN[39]、Open X-Embodiment[9]、DROID[18]（机器人）。上游 yaml 为 `data_mix: lam_plus_human`（HF-COLL `dino_large_vae.yaml`）。训练配置：16×H100，100k steps，batch 1024，lr 3e-4（PAPER §C.5）。上游 yaml 的 optimizer lr 是 1e-4，max_epochs 40，与论文不一致，两者都列出。
- **LaWAM 策略预训练**（`lawam_pretrain`）：**只用带语言指令的机器人轨迹**，人类视频只通过 LaWM 间接起作用（PAPER §4、§C.5）。训练配置：200k steps，64×H100，batch 1024（PAPER §C.5）。`data_mix: lam`、`sec_chunk: 1.2`（HF-COLL `lawam_pretrain/config.yaml`）。归一化统计里的 embodiment 分组及轨迹数（HF-COLL `lawam_pretrain/dataset_statistics.json` `num_trajectories`）：oxe_bridge 50,368；oxe_rt1 87,125；franka 95,489；agibot_genie1 32,700；agilex 46,429；ur 24,977；franka_fr3_dual 1,773。
- `"lam"` mixture 的具体组成在公开的 GH/mixtures.py 中**没有定义** → 具体数据集清单 `unverified`。
- **预训练是否包含 LIBERO（或 LIBERO-90）**：`unverified`。`franka` 分组的来源不明。

### B.4 观测配置、控制频率、序列长度
- 相机（SFT）：两路 RGB。`observation.images.image`（agentview，主视角，同时供 LAM 使用）和 `observation.images.image2`（wrist；数据集中原名 `wrist_image`，训练时通过 `rename_map` 改名）。分辨率 256×256（HF-DS info.json；HF-SFT `config.json` `primary_image_features`/`wrist_image_features`/`lam_image_feature`；DOC-mdx 第 52 行）。仅 RGB（PAPER §C.2）。
- 不使用本体状态（`flow_use_state=false`，PAPER §C.2）。数据集中有 8 维 state，但不进入模型（HF-DS info.json）。
- 控制频率：20 Hz（HF-DS `fps: 20`；manifest `action_hz: 20.0`）。
- 训练样本长度：1 帧当前观测 + 1 帧未来观测（只用于 LAM teacher 与 û_T 监督，下标 `[0,7]`）+ **8 步动作**（0.4 s）（L/configuration_lawam.py:267-286；GH/train_libero.yaml `num_frames: 2`、`sec_chunk: 0.4`）。预训练阶段为 24 步 / 1.2 s（HF-PRE `config.json` `action_horizon: 24`；HF-COLL `lawam_pretrain/config.yaml` `sec_chunk: 1.2`）。
- LaWM 预训练的帧间隔有冲突：PAPER §C.5 写机器人 **1.2 s**、人类 0.4 s；HF-COLL `dino_large_vae.yaml` 写 `frame_dt_sec: 1.6`、`human_frame_dt_sec: 0.4`。**两者都列出。**
- 增广：LeRobot / 上游的 SFT 与策略预训练都**不做随机图像增广**（DOC-mdx 第 63 行；HF-COLL configs `enable_primary_random_resized_crop: false`）。LaWM 预训练有随机裁剪和颜色增广（PAPER §C.5；`dino_large_vae.yaml` `image_aug: true`、`dual_view_aug: true`）。

### B.5 变化覆盖（供后续 known/unknown 标签使用）

| 变化类型 | (a) LIBERO 微调数据 | (b) 预训练数据 | 出处 / 备注 |
|---|---|---|---|
| 相机视角 | 只有固定命名的 agentview + wrist 两路，**无视角扰动的证据**，也没有随机裁剪；`unverified`（不作推断） | 跨数据集的相机配置天然多样（OXE/DROID/AgiBot/RoboMIND 等）；LaWM 阶段有随机裁剪；系统性视角扰动 `unverified`。论文 §5 指出模型适用于“相对稳定的相机视角”，大幅视角变化是已知局限 | HF-DS info.json；DOC-mdx 第 63 行；PAPER §4、§5、§C.5 |
| 光照 | `unverified` | LaWM 阶段有颜色增广（photometric），属于间接覆盖；策略预训练阶段无增广；数据集本身的光照多样性 `unverified` | PAPER §C.5；HF-COLL `lawam_pretrain/config.yaml` |
| 背景 / 纹理 | `unverified` | 跨数据集多场景（来源列表见 B.3）；没有针对性的随机化证据 → `unverified` | PAPER §4 |
| 干扰物体 | `unverified` | `unverified` | — |
| 物体布局 / 超出 LIBERO 原生 init 分布的初始位置 | **没有额外布局的证据**（数据集是官方示教的再处理版本；未使用 `libero_plus_lerobot`）；`unverified` | 不适用于 LIBERO 场景；`unverified` | GH/mixtures.py；GH/README.md |
| 机器人初始位姿 | `unverified` | `unverified` | — |
| 语言改写（paraphrase） | 无证据；info.json `total_tasks=40`（与 40 个任务一一对应）；`unverified` | `unverified` | HF-DS info.json |
| 传感器噪声 | `unverified` | `unverified` | — |
| 动力学（摩擦 / 质量 / 增益） | `unverified` | 多 embodiment（Franka、UR、AgileX、AgiBot、Google RT-1、WidowX/Bridge）带来机器人本体差异，但不是动力学随机化；`unverified` | HF-COLL `lawam_pretrain/dataset_statistics.json` |

---

## C. 与评测相关的接口事实

### C.1 动作块与后处理
- `action_horizon=8`，`n_action_steps=8`：每次推理生成 8 步并全部执行完才重新规划。`chunk_size=50` 只是 flow 序列的 padding 长度。预训练 checkpoint 中两者都是 24（HF-SFT/HF-PRE `config.json`；DOC-mdx 第 61 行；L/modeling_lawam.py:215-217、271-280；L/vlas/flowmatching_expert.py:578-580，`time_valid` 只保留前 8 步）。
- 后处理流水线（HF-SFT `policy_postprocessor.json`；L/processor_lawam.py:602-626）：
  1. `lawam_clip_actions`：把归一化动作 clamp 到 [-1, 1]；
  2. `lawam_pre_snap_gripper`：在归一化空间中，第 6 维 ≥0.5 记为 1，否则记为 0；
  3. MIN_MAX 反归一化；
  4. `lawam_binarize_gripper`：第 6 维 >0.5 输出 +1，否则输出 −1（对应 LIBERO 的 ±1 约定）。
- 推理超参：flow 采样 10 步，CFG scale 1.0（HF-SFT `config.json`）；`embodiment_id=25`；`action_hz=20`（HF-SFT `policy_preprocessor.json`）。

### C.2 仿真版本与最大步数
- **MuJoCo 必须是 3.3.2**。换成更新版本会显著降低成功率；例如 spatial task 5 在 3.8.1 下为 18/50，在 3.3.2 下为 47/50（DOC-mdx 第 25-31、98 行；DOC-README；HF-SFT README；GH/README.md）。
- 参考成功率（LeRobot 复现，50 ep/task）：spatial 98.4、object 99.6、goal 98.0、libero_10 98.0，overall 98.5%（DOC-mdx 表；HF-SFT README）。**论文数字不同**：Long 97.0、Goal 98.4、Object 99.6、Spatial 99.4，Avg 98.6%（PAPER §4.1 Table 1）。两者都列出。
- **每个 suite 的最大步数，两套来源不一致**：
  - LeRobot env：spatial 280 / object 280 / goal 300 / libero_10 520 / libero_90 400（ENV/libero.py:98-104；`episode_length=None` 时使用）。
  - 上游评测：spatial 250 / object 300 / goal 320 / libero_10 550 / libero_90 420，另加 `num_steps_wait=10` 步空等（GH/libero_benchmark_adapters.py `_get_canonical_max_steps`；GH/libero_eval_core.py:45、456）。
- 渲染分辨率：LeRobot env 默认 360×360（ENV/configs.py:334-335），由策略 processor resize 到 256。上游评测直接渲染 256（GH/libero_eval_core.py:35）。两边都对图像做 180° 翻转（ENV/…/processor/env_processor.py:45-59；GH/libero_eval_core.py:369-372）。
- LeRobot env 默认 `init_states=True`，使用 LIBERO 官方初始状态（ENV/configs.py:331）。

### C.3 想象、外部动作与画面解码
- **从给定的 LIBERO 初始状态做“想象”**：可以，但仅限**单步、策略自己驱动**。具体流程：当前 agentview 帧 → u；VLM 预测 ẑ → LaWM 输出 û_T（τ≈0.35–0.4 s 之后的 256×768 DINOv3 特征）。û_T 不在公开输出中，需要调用私有的 `policy.model.policy_backend._run_shared_encoding_infer(...)`，从返回的 `PolicyEncodingState.h_t1_pred` 取得；ground-truth 对照可用同一 DINOv3 编码真实未来帧（L/vlas/lawam.py:238-246、552-609、625-637）。由于 Qwen 是 causal attention，ẑ 只取决于主视角图像和指令（act query 位于 wrist 图像之前，见 L/latent_world/vlm_adapter.py:32-92；PAPER §C.2）；是否完全不受 wrist 图像影响 → `unverified`（依赖 Qwen chat template 的 token 顺序）。
- **输入外部机器人动作**：**不支持**。没有“机器人动作 → ẑ / û_T”的路径；动作只是 flow 头的输出（见 A 节“上下文与预测窗口”）。可行的替代只有：输入外部 **latent action z**（32 维），或用 LAM encoder 从一对真实帧反推 z 后再送入 LaWM。
- **多步 / 长时想象**：LeRobot 中没有实现；论文 Fig.10 的 open-loop rollout 机制 `unverified`。
- **解码画面**：**不能**，只能得到 DINOv3 特征空间的结果（见 A 节）；做可视化只能用特征相似度热图或 PCA（PAPER App. B）。
- 其他：推理时 LAM 的 DINOv3 与 LaWM 都在 bf16 autocast 下运行（L/vlas/lawam.py:560-561、580）；flow 采样从随机噪声开始（L/vlas/flowmatching_expert.py:585-590），**结果不确定**，需要固定 seed。

---

## 未核实项（unverified）与来源冲突

**未核实**
1. 论文 Fig.10 / §D.4 “extended open-loop rollouts” 的具体机制：是否把 û_T 回灌作为下一步的 u、latent action 来自哪里。
2. LIBERO SFT 数据中 no-op 的过滤规则；是否在仿真中重新生成；IPEC 源数据的处理细节。
3. 预训练 `lam` mixture 的完整数据集清单；预训练是否包含 LIBERO 或 LIBERO-90。
4. B.5 中几乎所有变化类型的覆盖情况：视角、光照、背景、干扰物、布局、初始位姿、语言改写、传感器噪声、动力学。
5. ẑ 是否完全不受 wrist 图像影响（取决于 chat template 中的 token 顺序）。
6. 论文总参数量 2.3B 与 HF 统计 2.56B 的差异来源；flow 头约 306.4M 是推算值，没有逐层核对。

**来源冲突（两边都列出）**
- 总参数：PAPER 2.3B ↔ HF safetensors 2,555,179,360。
- LaWM 层数：PAPER §C.1 写 encoder 与 decoder 都是 24 层 ↔ 代码/config 为 `lam_enc_layers=24`、`lam_dec_layers=12`。
- LaWM 帧间隔：PAPER §C.5 写 1.2 s ↔ `dino_large_vae.yaml` `frame_dt_sec: 1.6`。LaWM lr：PAPER 3e-4 ↔ yaml 1e-4。
- SFT 未来帧偏移：上游 `sec_chunk 0.4`（8 步）↔ LeRobot teacher 下标 7（0.35 s）。
- SFT 步数：PAPER / 上游 25k ↔ LeRobot `scheduler_decay_steps: 30000`。
- LIBERO 成功率：PAPER 98.6%（Spatial 99.4 / Goal 98.4 / Long 97.0）↔ LeRobot 98.5%（Spatial 98.4 / Goal 98.0 / 10 98.0）。
- 每个 suite 的最大步数：LeRobot 280/280/300/520 ↔ 上游 250/300/320/550（+10 步 wait）。
- state key 数：HF README 写 “1,912 policy state keys” ↔ `conversion_manifest.json` `state_key_count: 1419`。
- HF README 提到可用 `base_vlm_path` 覆盖 ↔ 本地 `LaWAMConfig` 中没有该字段（L/configuration_lawam.py:73-174）。
