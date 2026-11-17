# 模型卡：VLA-JEPA（LeRobot 集成，`lerobot/VLA-JEPA-LIBERO`）

> **口径（2026-10-14）**：本阶段只在虚拟环境内评测。文中的“真实”“现实”“real”一律指 **LIBERO 模拟器真值分支**，“迁移”指想象结论到模拟器执行的迁移。物理机器人不在范围内。本档案引用论文中的“真实世界 / real-world”实验时，指的是论文作者的物理机器人实验，与本项目的真值分支无关。

> 用途：LIBERO 上世界模型评判（world-model verdict）的第一步，基于证据整理的模型事实。
> 编写日期：2026-10-14。每条都注明出处；找不到证据的标 `unverified`；来源冲突时两边都列出。

**出处缩写**

| 缩写 | 含义 |
|---|---|
| `CODE/x.py:N` | `$LEROBOT_ROOT/src/lerobot/policies/vla_jepa/x.py` 第 N 行（本地 LeRobot 代码） |
| `MDX:N` | `$LEROBOT_ROOT/docs/source/vla_jepa.mdx` 第 N 行 |
| `README_LR:N` | `$LEROBOT_ROOT/docs/source/policy_vla_jepa_README.md` 第 N 行 |
| `ENV/x.py:N` | `$LEROBOT_ROOT/src/lerobot/envs/x.py` 第 N 行 |
| `HFcfg:N` | `https://huggingface.co/lerobot/VLA-JEPA-LIBERO/raw/main/config.json` 第 N 行（抓取于 2026-10-14） |
| `HFcard` | `https://huggingface.co/lerobot/VLA-JEPA-LIBERO/raw/main/README.md`（自动生成的 LeRobot 模型卡） |
| `HFpost` / `HFpre` | 同一仓库的 `policy_postprocessor.json` / `policy_preprocessor.json` |
| `HFapi` | `https://huggingface.co/api/models/<repo>?expand[]=safetensors` |
| `UP-HF:<file>` | 上游 HF 仓库 `ginwind/VLA-JEPA` 的文件，例如 `LIBERO/config.yaml`、`LIBERO/dataset_statistics.json`、`Pretrain/config.yaml` |
| `UP-GH:<file>:N` | 上游 GitHub `ginwind/VLA-JEPA` 的 commit `ec907778b4af46a45742baf5ebfb2e055560346e` 中的文件 |
| `Paper §x` | Sun et al., *VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model*，arXiv:2602.10098v2（2026-02-14），按章节 / 表 / 附录引用 |
| `DS-info` | `https://huggingface.co/datasets/HuggingFaceVLA/libero/raw/main/meta/info.json` |

---

## A. 结构与骨干

| 字段 | 内容 | 出处 |
|---|---|---|
| 骨干来源 | **语言/VLM 骨干**：`Qwen/Qwen3-VL-2B-Instruct`。论文称其视觉编码器为 SigLIP-2；**全量微调**，`freeze_qwen=false`。**视频编码器（world-state encoder）**：`facebook/vjepa2-vitl-fpc64-256`，即 V-JEPA 2 ViT-L，64 帧预训练，256 分辨率，`tubelet_size=2`，`hidden_size=1024`，24 层。它**冻结**：代码在 `torch.no_grad()` 下编码；论文也写明预训练时"除 world state encoder 外的所有参数都训练"，并且它是 stop-gradient 的目标编码器。**世界模型预测器**（`ActionConditionedVideoPredictor`）在预训练开始时**随机初始化**。**动作头**是 DiT-B，从头训练。LIBERO 检查点从 Pretrain 检查点继续训练而来。LeRobot 版本是由上游 `ginwind/VLA-JEPA` **转换**的，不是在 LeRobot 中重新训练的。 | `HFcfg:52-55`；`CODE/configuration_vla_jepa.py:45-47`；`CODE/modeling_vla_jepa.py:89-93,251-253`；V-JEPA2 config `https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/raw/main/config.json`（hidden_size=1024, num_hidden_layers=24, image_size=256, patch_size=16, tubelet_size=2）；Paper §3.1, §3.2 Eq.4, §4.1（"train all parameters except for the world state encoder"）, App. A.1（"randomly initialized predictor"）；`UP-HF:LIBERO/config.yaml`（`trainer.pretrained_checkpoint: .../VLA-JEPA-pretrain.pt`, `freeze_modules: ''`）；`MDX:60` |
| 生成机制 | 预测的是 **JEPA 潜空间**：下一个时间位置的 V-JEPA2 patch 特征。**不生成像素。** 预测器是一个 12 层的 transformer，带 3D-RoPE 和 block-causal 注意力。它做**确定性回归**：一次前向、teacher forcing、shift-by-one 切分（输入 = 时间位置 0..T-2 的**真实**潜表示，目标 = 位置 1..T-1），损失为 L1。它**不是**扩散或 flow 模型，也不做采样。论文称之为 "auto-regressive transformer"，但训练和代码都是 teacher forcing 下的单步并行预测；**代码里没有把预测结果回灌的自回归 rollout 函数**。（另：动作头才是 flow-matching，见"自带头"一行。） | `CODE/world_model.py:322-432`（forward 389-432）；`CODE/modeling_vla_jepa.py:261-287`；Paper §3.2 Eq.3, Eq.5（"teacher-forcing objective"）, App. A.1；上游一致：`UP-GH:starVLA/model/framework/VLA_JEPA.py:205-219` |
| 参数规模 | **总计 3,081,498,247**（BF16，HF safetensors 元数据）。按模块：**① Qwen3-VL-2B**：2,127,532,032（HF API，输入/输出嵌入绑定计一次）。LeRobot 检查点里似乎还多存了一份未绑定的 `lm_head`（151936×2048 = 311,164,928）；这一点是由总数差额**推算**出来的。**② V-JEPA2 ViT-L**：325,971,328。这个数字是 HF `VJEPA2Model` 的全部参数，包含它自带但 VLA-JEPA 不使用的 mask-predictor（`pred_*`，hidden 384，12 层）；仅编码器约 3.0 亿，为估算，`unverified`。**③ 世界模型预测器**：157,452,288，我用 meta 设备实例化 `world_model.py` 精确计数得到（12 个 block 共 151.2M，`predictor_embed` 2.10M，`action_encoder` 2.10M，`predictor_proj` 2.10M）。检查点里另有约 4,195,328 个遗留键（`extrinsics_encoder` + `state_encoder`），加载时作为 unexpected key 忽略，同样是推算。**④ 动作头（DiT-B + 编解码 MLP）**：155,182,343，按 `action_head.py` 解析计算，其中 DiT 本体 150,740,992。①+lm_head+②+③+遗留键+④ = 3,081,498,247，与 HF 总数**完全吻合**。 | `HFapi`（lerobot/VLA-JEPA-LIBERO：total 3,081,498,247；Qwen/Qwen3-VL-2B-Instruct：2,127,532,032；facebook/vjepa2-vitl-fpc64-256：325,971,328）；Qwen config `tie_word_embeddings: true, vocab_size: 151936`；`CODE/world_model.py:344-378`（含 346-351 关于遗留键的注释）；`CODE/action_head.py:209-262`；`CODE/modeling_vla_jepa.py:100-113` |
| 表示与分辨率 | **世界模型潜表示**：每个时间位置有 16×16 = **256 个 token**，每个 token 维度 `embed_dim = num_views × hidden = 2 × 1024 = 2048`（两个视角的特征按通道拼接）。预测器内部宽度为 1024。**世界模型输入帧**：V-JEPA2 的视频处理器先把短边缩放到 292，再中心裁剪到 **256×256**。上游配置为 `video_resolution_size: 256`；论文 A.2 写"world-state encoder 的视频裁剪 resize 到 256×256"。**VLM 输入**：**224×224**（`resize_images_to=[224,224]`）。**Tubelet**：`tubelet_size=2`，即 1 个潜时间位置 = 2 个连续数据帧。每个数据帧对应一个环境控制步（LIBERO 数据集每帧 1 个动作），因此 **1 个潜帧约等于 2 个 env step**；"1 帧 = 1 步"这一点是推断，见 B 节的频率冲突。每个样本加载 `num_video_frames=8` 帧，对应 4 个潜时间位置：3 个作为上下文，3 个作为目标。 | `CODE/modeling_vla_jepa.py:94-105,240-263`；`CODE/world_model.py:357-358`（grid = img/16）；V-JEPA2 `video_preprocessor_config.json`（`size.shortest_edge: 292`, `crop_size: 256`）；`HFcfg:78,84,86-89`；`MDX:54,217`；Paper App. A.1 Table 5（image token 维度 2048、每步 256 个 image token、视角数 2、future video horizon 8）, App. A.2；`UP-HF:LIBERO/config.yaml`（`resolution_size: 224`, `video_resolution_size: 256`） |
| 动作接入方式 | **真实机器人动作不进入世界模型预测器。** 预测器的"动作"条件是 Qwen3-VL 最后一层 decoder 在特殊 token `<|action_i|>` 位置的隐状态，即"潜动作"，维度 2048。数量为 (8/2−1) = 3 个时间步 × 每步重复 8 次 = 24 个 token。这些 token 经 `action_encoder: Linear(2048→1024)` 投影后，与对应时间步的帧 token 拼接。Qwen 只看**当前帧**（t=0 的两个视角）和语言指令，因此潜动作是 VLM 根据当前观测和指令推出的"意图/动态摘要"，**不是外部可控的动作输入**。**机器人动作**（7 维：Δxyz、Δ轴角 3 维、夹爪）只进入 DiT 动作头。归一化方面：LeRobot 对 7 维统一用 **MIN_MAX**；上游 6 个连续维用 min-max，**夹爪维不归一化**（mask=false，原始值 ∈[0,1]）。这一点**两侧不同**，见 C 节。状态为 8 维（eef_pos 3 + 轴角 3 + gripper_qpos 2），MEAN_STD 归一化，只进入动作头。 | `CODE/modeling_vla_jepa.py:106,125-137,167-194,275-282,432-433`；`CODE/world_model.py:345,400-412`；`CODE/qwen_interface.py:22-55`；`CODE/action_head.py:279-311`；`HFcfg:47-51,61-63`；`UP-HF:LIBERO/dataset_statistics.json`（action `mask: [T,T,T,T,T,T,F]`，gripper min 0 / max 1）；`UP-GH:starVLA/dataloader/gr00t_lerobot/data_config.py`（`Libero4in1DataConfig` 只对 x..yaw 做 min_max）；Paper §3.2 Eq.2-3, App. A.1（"whose action inputs are the latent action tokens produced by the VLM"） |
| 上下文与预测窗口 | **训练窗口**：8 帧，对应数据集偏移 0..7（`num_video_frames ≥ chunk_size` 时取 `range(8)`），即 4 个潜时间位置。上下文为位置 0..2 的真实潜表示，**每个位置预测下一位置**，即一步 = 2 个 env step；一次前向同时并行输出位置 1..3 的预测（teacher forcing）。**Attention**：block-causal。同一时间步内，动作 token 与帧 token 之间是双向全注意力；跨时间步是严格因果的，窗口 = 全部历史（`local_window_time = num_frames`）。**长 rollout 拼接**：代码中**没有**实现，也没有自回归回灌的 API。**泄漏注意**：默认 `causal_world_model_context=False`，此时上下文潜表示取自对整个 8 帧片段做的一次**双向** V-JEPA2 编码，所以上下文位置已经"看到"了未来帧（LeRobot issue #4153，上游实现也是这样）。设为 True 才会对每个前缀单独做因果编码。**能否作为外部动作驱动的模拟器：不能。** 预测器没有任何接收机器人动作的接口，条件只有 Qwen 产出的潜动作 token；若硬注入任意 2048 维向量，那也不是训练过的"动作→未来"映射。**推理时是否使用：不使用。** `predict_action` 只走 Qwen 加动作头；推理时 `_prepare_model_inputs(training=False)` 不组装视频，因此 V-JEPA2 和预测器都不运行。世界模型**只是训练期的辅助损失**。 | `CODE/configuration_vla_jepa.py:94,108-110,235-247`；`CODE/world_model.py:24-39,414-427`；`CODE/modeling_vla_jepa.py:196-211,255-282,352-370,450-451,505-513`；`MDX:15,26-27,83`；`README_LR:15,17`；Paper §3.2（time-causal attention 描述）；上游：`UP-GH:starVLA/model/framework/VLA_JEPA.py:191-213`（单次全片段编码），`:246-305`（`predict_action` 不调用 `vj_predictor`） |
| 自带头 | **策略/动作头**：DiT-B flow-matching。内部宽度 768（12 头 × 64），16 层，偶数层对 Qwen 的 32 个 `<|embodied_action|>` 隐状态做 cross-attention，奇数层做 self-attention；输出宽度 1024，再经 MLP 解码为 7 维。DiT 的输入序列是 [state token, 32 个可学习 future token, 带噪动作]。训练时时间 t 按 Beta(1.5, 1.0) 采样，用 MSE 回归速度 (a−ε)；每个样本重复 8 次噪声采样（`repeated_diffusion_steps=8`）。推理从高斯噪声出发，用 **4 步 Euler** 积分，一次输出 7 步动作块。**奖励/价值头**：**无**（`reward_delta_indices` 返回 None，代码中没有任何 reward/value/success 头）。 | `CODE/action_head.py:142-187,202-354`；`CODE/modeling_vla_jepa.py:289-315,367-370`；`CODE/configuration_vla_jepa.py:253-255`；`HFcfg:63-77,85`；Paper §3.3 Eq.6-8, App. A.1 Table 6 |
| 训练目标 | 总损失 = **动作 flow-matching 损失**（速度 MSE，按 `action_is_pad` 掩码）+ **0.1 × 世界模型 JEPA 损失**（预测潜表示与 V-JEPA2 目标潜表示之间的 **L1**）。论文把 JEPA 损失解释为 ELBO，并说明它在确定性编码器下退化为"潜空间重建损失"（Eq.4-5），λ 为可调超参（Eq.9）。预训练阶段：人类视频（SSv2）只用 JEPA 损失，机器人数据（DROID）用两项联合损失。LIBERO 阶段：两项联合损失。 | `CODE/modeling_vla_jepa.py:283-287,297-315,336-348`；`CODE/modeling_vla_jepa.py:489-500`；`HFcfg:83`；`UP-GH:starVLA/model/framework/VLA_JEPA.py:215-244`（`wm_loss * 0.1`）；`UP-HF:LIBERO/config.yaml`（`loss_scale: vla 1.0, vlm 0.1`）；Paper §3.2 Eq.4-5, §3.3 Eq.8-9, Fig.2, §4.1 |
| 能否解码出画面 | **不能。** 代码中没有任何从 V-JEPA2 潜表示到像素的解码器。`action_decoder` 只把 DiT 输出解码为 7 维动作；V-JEPA2 只调用 `get_vision_features`，它自带的 mask-predictor 也不是像素解码器。论文明确说"without pixel reconstruction"。因此**无法在"想象出的输出"上读取帧级成功判定（success predicate）**。可能的潜空间替代方案（均需自行实现，原仓库都没有提供）：(a) 把真实成功或终态帧编码为 V-JEPA2 潜表示，与预测器输出的潜表示比较距离；(b) 在 V-JEPA2 特征上训练一个成功/谓词探针（linear probe），再作用于预测出的潜表示。但要注意：每次只预测 3 个位置（约 6 个 env step）；默认上下文存在未来泄漏；预测器的条件是 VLM 潜动作而非外部动作。这三点都会限制此类判定的可解释性。 | `CODE/action_head.py:232-240`；`CODE/modeling_vla_jepa.py:252,280-287`；`CODE/world_model.py:429-432`（输出为 `predictor_proj` 投影回 2048 维潜空间）；Paper Abstract、§1 contributions（"without pixel reconstruction"） |

---

## B. 训练任务与数据

### B.1 LIBERO 套件与任务

- **来源冲突**：
  - LeRobot 文档的检查点表把 `lerobot/VLA-JEPA-LIBERO` 标为 **"LIBERO-10"**（`MDX:64`）。
  - 上游 LIBERO 训练配置是 `data_mix: libero_all`（`UP-HF:LIBERO/config.yaml`）。上游 `mixtures.py` 中的 `libero_all` 定义为 **4 个套件**：`libero_object_no_noops_1.0.0`、`libero_goal_no_noops_1.0.0`、`libero_spatial_no_noops_1.0.0`、`libero_10_no_noops_1.0.0`；其中 `libero_90` 被注释掉（`UP-GH:starVLA/dataloader/gr00t_lerobot/mixtures.py:15-21`）。
  - 上游 `LIBERO/dataset_statistics.json` 记录了 `num_trajectories: 1693` 和 `num_transitions: 272104`，这是 4 个套件合计的量级。
  - 论文 §4.1 说 LIBERO 阶段使用"约 2K 条专家演示的 LIBERO 数据集"；表 1 中**同一模型**在 Spatial/Object/Goal/LIBERO-10 四个套件上评测。
  - LeRobot 文档里的评测命令同样在四个套件上评测这个检查点（`MDX:186`）。
  - **结论**：从证据看，训练集应是 **LIBERO-Spatial + Object + Goal + 10 四个套件（不含 LIBERO-90）**。"LIBERO-10" 标注疑似笔误，但 LeRobot 侧的转换记录无法直接核实 → 冲突保留。
- 任务数为 40（`DS-info: total_tasks: 40`，HuggingFaceVLA/libero，即 LeRobot 文档复现时使用的数据集，见 `MDX:176`）。上游 IPEC 数据集的任务数为 `unverified`，但 4 个套件 × 10 任务应为 40。

### B.2 训练数据来源与规模

- **LIBERO 微调数据（上游，即被转换的检查点所用的数据）**：IPEC-COMMUNITY 的 LIBERO no-noops LeRobot 数据集，包括上文 4 个套件，共 **1693 条轨迹、272,104 个 transition**。来源：`UP-HF:LIBERO/dataset_statistics.json`、`UP-GH:README.md:93`（`IPEC-COMMUNITY/libero-benchmark-dataset`）和 `mixtures.py:15-21`。不包含 LIBERO-Plus 的增强数据（Paper §4.1："without incorporating the augmented dataset from LIBERO-Plus"）。
- **LeRobot 文档给出的复现数据集**：`HuggingFaceVLA/libero`，1693 个 episode，**273,465 帧**，40 个任务，标注 fps=10（`DS-info`；`MDX:172-178`）。它与上游的 272,104 transitions 有约 1.4K 的差异 → 冲突保留，原因 `unverified`。
- **训练步数与 batch**：从 Pretrain 检查点继续训练 **30K 步**，8 GPU × 32 = 全局 batch 256（Paper App. A.2；`UP-HF:LIBERO/config.yaml`：`max_train_steps: 30000`, `per_device_batch_size: 32`；`MDX:169-170`）。
- **学习率（冲突）**：
  - 论文 A.2：VLM 与世界模型 1e-5，动作头 1e-4。
  - 上游 config：base 3e-5、qwen 1e-5、action_model 1e-4，warmup 5000，cosine，最低 1e-6。
  - LeRobot `HFcfg:94-104`：单一 lr 1e-4，warmup 5000，decay 30000 到 1e-6，grad clip 1.0。
  - LeRobot 的 `get_optim_params` 把所有参数放在同一组（`CODE/modeling_vla_jepa.py:502-503`），所以如果在 LeRobot 中重训，各模块会使用同一个 lr。
- **动作类型**：上游 config 写 `action_type: delta_qpos`，但 `Libero4in1DataConfig` 的动作键是 x/y/z/roll/pitch/yaw/gripper；论文 A.2 说 VLA-JEPA 使用"end-effector delta position + delta axis-angle"。字段命名与论文描述不一致；**按论文与动作键，实际应是 EEF delta** → 命名冲突保留。

### B.3 LIBERO 之外的预训练数据

| 组件 | 数据 | 出处 |
|---|---|---|
| VLA-JEPA 预训练（整体模型） | **Something-Something-v2**：220K 人类视频，无动作标签，只用 JEPA 损失。**DROID**：76K 条机器人轨迹，使用联合损失。两者**联合训练 50K 步**。 | Paper §4.1, App. A.2；`MDX:113`；`UP-HF:Pretrain/config.yaml`（`vla_data.data_mix: droid`, `video_data.video_dir: .../ssv2/...`, `max_train_steps: 50000`）；`UP-GH:README.md:91-92` |
| LeRobot 的 `VLA-JEPA-Pretrain` 检查点 | 文档表只写 "DROID 1.0.1"，没有提 SSv2，与上一行冲突。不过上游 Pretrain config 包含 SSv2。LIBERO 检查点的 `pretrained_checkpoint` 指向 `VLA-JEPA-pretrain.pt`。由此**推断**这次 LIBERO 微调所基于的预训练包含人类视频。论文表 1 另有一个 "w/o human videos" 消融变体；发布的检查点是否为该变体：`unverified`（按配置推断不是）。 | `MDX:65`；`UP-HF:Pretrain/config.yaml`；`UP-HF:LIBERO/config.yaml`；Paper Table 1 |
| Qwen3-VL-2B 的 VLM 预训练 | 具体数据 `unverified`。HF 模型卡只有 "Broader, higher-quality pretraining" 之类的笼统描述。 | `https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/main/README.md` 第 33 行 |
| V-JEPA 2 ViT-L 的视频预训练 | 具体数据 `unverified`。HF 模型卡没有列出数据，只引用了 V-JEPA 2 论文，本卡未核对该论文。 | `https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/raw/main/README.md` |
| 其它机器人预训练（OXE 等） | LIBERO 线路中没有证据显示使用过。OXE Bridge/Fractal 只用于 SimplerEnv 检查点。 | Paper §4.1；`MDX:66` |

### B.4 观测配置、控制频率、序列长度

- **观测**：2 个视角，`observation.images.image`（agentview）和 `observation.images.image2`（wrist）；VLM 输入为 3×224×224；另有 `observation.state` 8 维。来源：`HFcfg:4-27`、`HFcard` Inputs 表、`MDX:64`。数据集原始图像为 256×256（`DS-info`）。
- **图像朝向**：训练数据和评测都把图像旋转 180°（H、W 同时翻转）。来源：`UP-GH:examples/LIBERO/eval_libero.py`（"rotate 180 degrees to match train preprocessing"）；LeRobot 侧见 `$LEROBOT_ROOT/src/lerobot/processor/env_processor.py:59`。
- **控制频率（冲突）**：LeRobot `LiberoEnv.fps = 20`，注释为 "Must match robosuite's default control_freq (20 Hz)"（`ENV/configs.py:326`）；`HuggingFaceVLA/libero` 的元数据标注 `fps: 10.0`（`DS-info`）。模型训练时按**帧索引**取样（`observation_delta_indices` / `action_delta_indices`），所以真正相关的是"每帧 = 1 个控制步"，而不是挂钟时间 → 秒级换算 `unverified`。
- **训练序列长度**：
  - 动作块 `chunk_size=7`，即目标为偏移 0..6 的动作（`CODE/configuration_vla_jepa.py:249-251`；`HFcfg:45`；上游 `action_horizon: 7`）。
  - 世界模型视频 8 帧，偏移 0..7（`CODE/configuration_vla_jepa.py:236-247`；`HFcfg:78`）。
  - VLM 只看 t=0 的单帧（`n_obs_steps=1`，`HFcfg:3`；`CODE/modeling_vla_jepa.py:432-433`）。
  - 论文表 4 消融了 future video horizon（T = 4/8/16），T=8 最好。

### B.5 变化覆盖（为后续 known/unknown 标注准备）

总前提：LIBERO 微调**没有**使用 LIBERO-Plus 的增强数据（Paper §4.1）。上游 LIBERO 数据变换 `Libero4in1DataConfig.transform()` **只含动作变换，没有任何图像增强**（没有 crop/color jitter；`UP-GH:starVLA/dataloader/gr00t_lerobot/data_config.py`，`Libero4in1DataConfig`）。LeRobot 的 `VLAJEPAConfig` 中也没有图像增强字段（`CODE/configuration_vla_jepa.py` 全文）。论文在 LIBERO-Plus 上对这 7 个扰动维度的评测（表 3）对本模型而言都是**分布外**。

| 变化类型 | (a) LIBERO 微调数据 | (b) 预训练数据（DROID + SSv2 + VLM/V-JEPA2 预训练） |
|---|---|---|
| 相机视角 | 无额外视角变化，只有 LIBERO 原生的 agentview + wrist 两个固定视角。证据：没有 LIBERO-Plus 数据，也没有图像增强（Paper §4.1；上游 data_config）。"LIBERO 原生演示中相机位姿固定"这一点本身在本卡来源里没有直接核实 → 原生相机是否完全固定：`unverified` | DROID / SSv2 是否覆盖多视角：`unverified`（本卡未取得其数据卡证据）。论文只有动机层面的描述，说潜空间预测"robust to camera motion"（Abstract），这不是数据覆盖的证据 |
| 光照 | 无额外光照变化（同上证据）；原生场景内部的光照差异 `unverified` | `unverified` |
| 背景/纹理 | 无额外背景或纹理变化（同上）；LIBERO 原生不同场景之间的桌面/背景差异 `unverified` | `unverified` |
| 干扰物 | 只有 LIBERO 原生场景中的非目标物体，没有额外干扰物（同上） | `unverified` |
| 物体布局 / 初始位置（超出 LIBERO 原生初始化分布） | **无**：1693 条演示全部来自 LIBERO 原生演示，未使用 LIBERO-Plus 的 Layout 增强（Paper §4.1；`UP-HF:LIBERO/dataset_statistics.json`） | `unverified`（DROID 场景不同，但与 LIBERO 布局分布的关系无从核实） |
| 机器人初始姿态 | 只有原生演示中的初始姿态，没有额外扰动（同上） | `unverified` |
| 语言改写 | **无**：每个任务一条指令，共 40 个任务（`DS-info total_tasks: 40`），代码与配置中都没有改写或增强。prompt 模板固定（`HFcfg:58`） | `unverified`（SSv2 自带人工标注文本，`UP-HF:Pretrain/config.yaml` 中有 `text_file`，但它是否构成对 LIBERO 指令的改写覆盖：`unverified`） |
| 传感器噪声 | **无**：没有噪声增强（上游 data_config；LeRobot config） | `unverified` |
| 动力学（质量、摩擦等） | **无**：只有 LIBERO 原生仿真参数，没有动力学随机化的证据 | `unverified`（DROID 为真实世界 Franka；SSv2 为人手。与 LIBERO 动力学的对应关系 `unverified`） |

---

## C. 与评测相关的接口事实

- **动作块与执行长度**：`chunk_size=7`、`n_action_steps=7`，即每 7 步重新调用一次策略，中间不做时间集成（`HFcfg:45-46`；`CODE/modeling_vla_jepa.py:515-523` 的动作队列）。上游评测也是每 `step % chunk == 0` 推理一次，然后依次执行 7 步（`UP-GH:examples/LIBERO/model2libero_interface.py:116-125`）。Flow 采样用 4 步 Euler，**初始噪声是随机的**，因此策略具有随机性（`CODE/action_head.py:331-354`）。
- **夹爪处理**：
  - 发布的 config 中 `pre_snap_gripper_action=true` 且 `binarize_gripper_action=true`（`HFcfg:90-92`）。
  - 后处理顺序为 `clip[-1,1]` → `pre_snap`（归一化空间中 ≥0.5 记 1，否则记 0）→ `unnormalize`（MIN_MAX）→ `binarize`（物理值 >0.5 记 −1，否则记 +1）→ cpu（`HFpost`；`CODE/processor_vla_jepa.py:126-183,268-306`）。
  - `HFpost` 中 snap/binarize 的 config 都是 `{}`，因此加载时取类默认值 `gripper_dim=6`、`threshold=0.5`（`CODE/processor_vla_jepa.py:137,150-153`）。
  - 上游的做法是：clip → 夹爪维 <0.5 记 0、否则记 1 → 夹爪维**不做**反归一化（mask=false）→ `1-2*(v>0.5)`，其中 1 = open（`UP-GH:examples/LIBERO/model2libero_interface.py:130,136-147`；`UP-GH:examples/LIBERO/eval_libero.py:25-30`）。
  - 两条路径是否完全等价，取决于 LeRobot 检查点中夹爪维的 min/max 统计量（存放在二进制 `policy_postprocessor_step_2_unnormalizer_processor.safetensors` 中，本卡未下载）→ `unverified`。`MDX:96-107` 也说明了这两个开关只适用于 LIBERO 的约定。
- **评测最大步数（冲突）**：

  | 套件 | LeRobot `ENV/libero.py:98-104` | 上游 `UP-GH:examples/LIBERO/eval_libero.py:94-103` |
  |---|---|---|
  | libero_spatial | **280** | **250** |
  | libero_object | 280 | 280 |
  | libero_goal | 300 | 300 |
  | libero_10 | 520 | 520 |
  | libero_90 | 400 | 400 |

  两边都在开头先执行 `num_steps_wait=10` 步空动作（`ENV/libero.py:127,351`；上游 eval 第 48 行和第 150-156 行）。每任务评测次数也不同：上游与论文为 50 次/任务（Paper Table 1；上游 eval 第 49 行），LeRobot 文档示例为 10 次/任务（`MDX:187`）。
- **已报告的 LIBERO 结果（冲突 / 文档内部不一致）**：
  - 论文表 1：Spatial 96.2 / Object 99.6 / Goal 97.2 / LIBERO-10 95.8 / Avg 97.2（每任务 50 次）。
  - LeRobot 文档：95.0 / 100.0 / 98.0 / 93.0 / 96.5（每套件 100 次）。其中 Spatial 一行写 "93 successes" 却是 95.0%，LIBERO-10 一行写 "96 successes" 却是 93.0%，合计 387/400 = 96.75% 与 96.5% 也不符（`MDX:205-211`）。这是文档内部的算术矛盾。
- **观测预处理**：
  - LeRobot 的 LIBERO env 默认渲染 360×360（`ENV/configs.py:334-335`），再由模型内部的 `resize_images_to=[224,224]` 用 area 插值缩放（`CODE/modeling_vla_jepa.py:360-365,428-436`）。
  - 发布的 `HFpre` 中没有 `vla_jepa_image_prep` 这一步；缩放依赖模型内部的保护性 resize。代码注释说明了这一设计（`CODE/processor_vla_jepa.py:47-49`）。
  - 上游 env 以 256 渲染，再缩放到 224（上游 eval 第 24 行与 `resize_size`）。
  - 状态为 eef_pos(3) + quat→轴角(3) + gripper_qpos(2)（上游 eval 第 168-174 行）。
- **从给定的 LIBERO 初始状态做"想象"**：原仓库**不支持**，需要自己拼装。可行的最小路径是：
  1. 用 `ENV/libero.py:344-345` 设置 init_state，渲染两视角帧；
  2. V-JEPA2 至少需要 `tubelet_size=2` 帧才能得到第一个潜位置，单帧必须复制；
  3. Qwen 根据 t=0 的图像和指令产出 24 个潜动作 token；
  4. 调用 `video_predictor(context_latents, action_tokens)`。

  限制：只训练过 teacher forcing、3 个上下文位置的情形（`CODE/modeling_vla_jepa.py:261-282`）；自回归回灌预测、超过 3 步（约 6 个 env step）的外推都没有训练过，也没有实现；`num_action_prompt_steps` 固定为 3（`CODE/modeling_vla_jepa.py:130-134`）。
- **输入外部动作**：**不可能直接实现**。预测器的条件维度是 Qwen 隐层的 2048 维（`CODE/modeling_vla_jepa.py:106`；`CODE/world_model.py:345`），不存在从机器人动作到潜动作 token 的映射；机器人动作只进入 DiT 动作头。
- **解码画面**：**不可能实现**，没有像素解码器，见 A 节最后一行。
- **推理时的世界模型**：推理时**不运行**（`MDX:26-27`；`CODE/modeling_vla_jepa.py:352-370,450-451`）。想把世界模型用于评测，需要自己写调用代码，并且需要 `enable_world_model=True`，这样加载时才会构建 V-JEPA2 和预测器（`CODE/modeling_vla_jepa.py:88-117`；`HFcfg:55`）。

---

## 未核实项

1. `lerobot/VLA-JEPA-LIBERO` 是否真的只用了 LIBERO-10。文档写 LIBERO-10；上游配置、统计和论文都指向 4 套件共 1693 条轨迹。转换记录未能直接核实。
2. LeRobot 检查点中夹爪维的 MIN_MAX 统计量，以及 LeRobot 与上游的夹爪后处理是否逐值等价（统计量在二进制 safetensors 中，未下载）。
3. LeRobot 检查点是否多存了一份未绑定的 `lm_head`，以及是否带有遗留的 `extrinsics_encoder`/`state_encoder` 键。这是由参数总数精确吻合推算的，未逐键核对。V-JEPA2 仅编码器部分的参数量（约 3.0 亿）为估算。
4. 数据集 fps 标注为 10，仿真控制频率为 20 Hz：一个潜帧（2 帧）对应的真实时间 `unverified`；"每数据帧 = 1 个 env step"为推断。
5. `HuggingFaceVLA/libero` 有 273,465 帧，上游统计为 272,104 transitions，差异原因不明。
6. 上游配置中 `action_type: delta_qpos` 与论文"EEF delta + 轴角"的命名不一致。
7. 发布的 Pretrain 检查点是否包含 SSv2 人类视频：按上游配置推断包含；LeRobot 文档只写 DROID。
8. Qwen3-VL-2B 与 V-JEPA 2 自身预训练数据的组成。
9. 全部"变化覆盖"条目中，预训练数据（DROID / SSv2 / VLM / V-JEPA2）是否覆盖相机、光照、背景、干扰物、布局、初始姿态、语言、噪声、动力学等变化。
10. LIBERO 原生演示内部是否存在相机、光照、纹理差异（本卡来源中没有直接证据）。
11. 论文 Table 5 写 "Number of action tokens per time step = 3"，代码与配置为每时间步重复 8 次、共 3 个时间步、合计 24 个 token；论文 Q3 又说"潜动作数 = 帧数 − 1"，代码中实际为 T/tubelet − 1 = 3。按代码理解，但论文表述存在歧义或冲突。
