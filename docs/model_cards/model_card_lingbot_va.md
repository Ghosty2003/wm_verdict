# 模型卡：LingBot-VA（LeRobot 集成）— LIBERO 世界模型评估用

> **口径（2026-10-14）**：本阶段只在虚拟环境内评测。文中的“真实”“现实”“real”一律指 **LIBERO 模拟器真值分支**，“迁移”指想象结论到模拟器执行的迁移。物理机器人不在范围内。本档案引用论文中的“真实世界 / real-world”实验时，指的是论文作者的物理机器人实验，与本项目的真值分支无关。

> 用途：LIBERO 上世界模型判定评估的第一步，只记录事实。每一条都注明出处；没有证据的写 `unverified`；来源互相矛盾时把双方都列出。
> 编写日期：2026-10-14。

### 出处缩写

| 缩写 | 指向 |
|---|---|
| `[CODE]/x.py:N` | 本地 `$LEROBOT_ROOT/src/lerobot/policies/lingbot_va/x.py` 第 N 行（本地 `README.md` 只有一行相对路径 `../../../../docs/source/lingbot_va.mdx`，指向下面的 DOC） |
| `[DOC]:N` | 本地 `$LEROBOT_ROOT/docs/source/lingbot_va.mdx` 第 N 行 |
| `[LR]/x:N` | 本地 `$LEROBOT_ROOT/src/lerobot/x` 第 N 行（LIBERO env / eval 脚本） |
| `[HF-LR]` | `lerobot/lingbot_va_libero_long` 的 `README.md` / `config.json` / `policy_*processor.json`，以及 HF API 给出的文件大小 |
| `[HF-RB]` | `robbyant/lingbot-va-posttrain-libero-long` 的 `README.md`、`transformer/config.json`、`vae/config.json`、`text_encoder/config.json`、`transformer/diffusion_pytorch_model.safetensors.index.json`，以及 HF API 给出的文件大小 |
| `[HF-DS]` | 数据集 `robbyant/libero-long-lerobot` 的 `README.md`、`meta/info.json`、`meta/tasks.jsonl`、`meta/episodes.jsonl`，以及 HF API 文件列表 |
| `[PAPER §x]` | Li et al., *Causal World Modeling for Robot Control*，arXiv:2601.21998v2（2026-03-22 修订） |
| `[GH]/path:N` | 上游 GitHub `Robbyant/lingbot-va`，commit `7c6ffa9b`（2026-07-09） |

RoboTwin checkpoint：`lerobot/lingbot_va_robotwin` / `robbyant/lingbot-va-posttrain-robotwin` 是存在的（[DOC]:53；[HF-RB] README），但不在本卡范围内。另有 `lerobot/lingbot_va_base` ← `robbyant/lingbot-va-base`，即预训练基座（[DOC]:54；HF API cardData.base_model）。

---

## A. 结构与骨干

| 字段 | 内容 | 出处 |
|---|---|---|
| 骨干来源 | **视频流**：从 **Wan2.2-5B** 初始化，Wan2.2-5B 是一个预训练视频扩散模型（d_v=3072，30 层）。具体是 Wan2.2 的哪个 5B 变体（例如 TI2V-5B），本卡所用来源中都没有写：`unverified`。**动作流**：论文写的初始化方式是“按动作维度插值预训练视频权重，再乘以 α=√(d_v/d_a)”。**训练链路**：Wan2.2-5B → 在约 16K 小时机器人数据上预训练 1.4T tokens，得到 `lingbot-va-base` → 在 LIBERO 上后训练，得到 `lingbot-va-posttrain-libero-long` → 转换为 LeRobot 格式的 `lerobot/lingbot_va_libero_long`。**冻结部分**：Wan2.2 VAE（`AutoencoderKLWan`，z_dim=48）、UMT5-XXL 文本编码器、tokenizer 都冻结，不存进 LeRobot checkpoint，加载时从 `wan_pretrained_path` 拉取。**可训练部分**：只有 DiT transformer。 | [PAPER §3.3 Architecture/Action Network Initialization, §4.2 Implementation/Pre-Training/Post-Training]；[DOC]:16-20, 56-58；[CODE]/configuration_lingbot_va.py:78-81；[CODE]/modeling_lingbot_va.py:77-100, 148-151；[HF-LR] README front-matter `base_model: robbyant/lingbot-va-posttrain-libero-long`；[HF-LR] config.json `wan_pretrained_path="robbyant/lingbot-va-posttrain-libero-long"`；[HF-RB] text_encoder/config.json `_name_or_path: google/umt5-xxl` |
| 骨干来源（冲突：MoT 是否独立参数） | **论文**：双流 MoT。动作流是一套独立 transformer，深度相同、宽度更小（d_a=768），QKV 投影各自独立，在联合 self-attention 中融合。**已发布权重与 LeRobot 代码**：上游 README 和 HF 卡写明发布的是 “**w/ shared backbone**” 版本，并说 “Please stay tuned for our separated version”。权重索引中只有一套 `blocks.N.*`（共 841 个 key），没有任何动作专属的 block 权重。LeRobot 的 `WanTransformer3DModel` 中，视频 token 和动作 token 都经过**同一组 30 个 `WanTransformerBlock`**；动作专属的只有 `action_embedder`、`condition_embedder_action`（timestep/text 嵌入的一份深拷贝）和 `action_proj_out`。DOC 把它称作 “dual-stream mixture-of-transformers … share the same 30 transformer blocks”。 | [PAPER §3.3 Mixture-of-Transformer Block, §4.2]；[GH]/README.md:57, 71-73；[HF-RB] README 模型表；[HF-RB] transformer index.json；[CODE]/utils.py:841-866, 1082-1090；[DOC]:11-14 |
| 生成机制 | **连续潜空间上的 flow matching（rectified-flow 式线性插值）**，块级（chunk）**自回归**：块内双向，块间因果，并用 KV cache 跨块保存历史。它不是离散 token 自回归，也不是离散扩散。推理时每个 chunk 内**先**用视频流把视频潜变量去噪（20 步，CFG=5），**再**用动作流去噪动作（50 步，CFG=1）。两个流各有一个 `FlowMatchScheduler`（shift 分别为 5.0 和 0.05）。采样器是 Euler 式更新：`prev = sample + v·(σ_next−σ)`。论文的表述是“先预测未来视觉状态，再用逆动力学解码动作”。 | [PAPER §1, §3.2 Eq.7-9, §3.3, Fig.2]；[CODE]/modeling_lingbot_va.py:176-182, 750-834；[CODE]/utils.py:187-282；[CODE]/configuration_lingbot_va.py:103-109；[DOC]:22-25 |
| 参数规模 | **可训练 DiT**：LeRobot `model.safetensors` 为 10,177,841,732 字节；按 bf16 算约 **5.09B**。按 config 逐项推算为 5,088,872,670。分模块看（同样由 config 推算，非官方数字）：共享的 30 个 block ≈ 4.910B；视频专属部分（`patch_embedding_mlp` + `condition_embedder` + `proj_out`）≈ 0.090B；动作专属部分（`action_embedder` + `condition_embedder_action` + `action_proj_out`）≈ 0.089B。上游 transformer 分片合计 10,179,017,396 字节，比 LeRobot 多约 1.18MB。这与上游多出、LeRobot 代码不使用的 `patch_embedding.*` 卷积（≈0.59M 参数）一致；“LeRobot 转换时丢弃了它”只是推断。**VAE**：`vae/diffusion_pytorch_model.safetensors` 为 2,818,777,808 字节；参数量和 dtype 均 `unverified`。**UMT5-XXL 编码器**：3 个分片合计约 11.36GB。DOC 写“frozen VAE+UMT5 ≈ 20 GB”。**冲突**：论文写视频流 Wan2.2-5B + 动作流约 350M = **5.3B**；DOC 写 “~5B”；已发布的共享骨干权重约 5.09B，没有独立的 350M 动作流。 | [HF-LR] HF API siblings size；[HF-RB] index.json `metadata.total_size` 及 API size；[CODE]/configuration_lingbot_va.py:62-74；[CODE]/utils.py:836-866；[PAPER §4.2]；[DOC]:18, 56-60, 181-182 |
| 表示与分辨率 | **VAE**：Wan2.2 因果 VAE，压缩比 4×16×16（时间×高×宽），48 通道，`patch_size=2`、`scale_factor_temporal=4`、`scale_factor_spatial=16`。DiT 再做 (1,2,2) patchify。**LIBERO 设置**：每个相机先 resize 到 **128×128**，像素缩放到 [-1,1]，每相机潜变量为 48×8×8；两路相机在**宽度维拼接**，得到 **48×F×8×16**。patchify 后每个潜帧 **32 个视频 token**。**时间**：每个潜帧对应 4 个环境帧。`action_per_frame=4`，即**每个潜帧 = 4 个环境步**。keyframe stride = 4//4 = 1，也就是每个环境步的观测都送入 VAE。**论文冲突**：论文写每帧 N=192 个空间 token（对应其预训练分辨率；上游 README 也建议约 256×256），LIBERO 配置则为 128×128。 | [HF-RB] vae/config.json；[PAPER §3.3 Video Sparsification, §4.2]；[CODE]/configuration_lingbot_va.py:62, 98-101；[CODE]/modeling_lingbot_va.py:164-167, 530-576, 604-611；[GH]/wan_va/configs/va_libero_cfg.py:16-22；[GH]/README.md（Video Format: “~256 × 256 pixels … 5-15 fps”） |
| 动作接入方式 | **形式**：动作是连续向量，每个环境步一个 token，通过 `action_embedder`（`nn.Linear(30→3072)`）嵌入。它们与视频 token 用 RoPE 网格 id 交错排布：视频的 t 维为 0，动作为 1，动作的帧 id 带小数偏移。两类 token 在**同一组 block 的同一个 self-attention** 中交互，即共享 KV cache。**论文冲突**：论文写的是单层 MLP，隐藏层 256。**动作维度**：统一的 30 维（左/右 EEF 各 7、左/右关节各 7、左/右夹爪各 1）。LIBERO 只用通道 **0–6**：xyz、roll/pitch/yaw 的 EEF delta，加上夹爪。注意，这里夹爪放在通道 6，而不是 schema 中的通道 28；其余通道在推理中清零，训练中被 mask。**归一化**：分位数归一化 `(a−q01)/(q99−q01)·2−1`，训练时再 clip 到 ±1.5。LIBERO 的 q01/q99 写在上游配置里，夹爪通道为 [-1,1]。LeRobot 侧的统计量存放在 `policy_postprocessor` 的 QUANTILES unnormalizer 中。 | [CODE]/utils.py:82-96, 844, 902-919；[CODE]/modeling_lingbot_va.py:355-367, 698-705, 833；[CODE]/processor_lingbot_va.py:252-258；[CODE]/configuration_lingbot_va.py:112-114；[DOC]:128-147；[PAPER §3.2 Video-Action State Encoding, §4.1 Unified Action Representation, §4.2]；[GH]/wan_va/configs/va_libero_cfg.py:33-60；[GH]/wan_va/dataset/lerobot_latent_dataset.py `_action_post_process`；[HF-DS] info.json action names；[HF-LR] policy_postprocessor.json |
| 上下文与预测窗口 | **每次前向**：一个 chunk = `frame_chunk_size=4` 个潜帧（16 个环境帧）和 4×4 = **16 个动作**。首个 chunk 的第 0 帧是条件帧，丢弃它对应的动作后只返回 12 个。**历史**：`attn_window=30`，KV cache 容量为 (30//2)=15 个 chunk 的视频与动作 token（15×128 + 15×16 = 2160 个 token），满了之后按 id 淘汰最旧的。**训练**：`window_size∈[4,64]` 与 `chunk_size∈[1,4]` 随机采样。**长时滚动**：闭环拼接。每个 chunk 执行完后，先 `clear_pred_cache` 去掉预测的 KV，再把**真实观测**（VAE 编码）和**实际执行的动作**写回 KV cache，然后预测下一个 chunk。 | [CODE]/configuration_lingbot_va.py:100-102, 142-150；[CODE]/modeling_lingbot_va.py:402-476, 613-628, 717-746；[CODE]/utils.py:585-605, 877-899；[CODE]/modeling_lingbot_va.py:317-318；[PAPER §3.3 Variable Chunk Size Training, §3.4 Alg.1]；[DOC]:22-25 |
| 上下文与预测窗口（能否当动作条件模拟器） | **结论：现有代码不能当作“输入外部动作 → 输出这些动作导致的未来画面”的动作条件模拟器。它只能联合预测视频和动作（先出视频，再用逆动力学出动作）。** 证据：①训练 mask 中，第 c 个 chunk 的视频帧 id 为 2c，动作帧 id 为 2c+1；带噪 token 只能看帧 id **严格更小**的干净 token。因此视频 chunk c 只以 chunk < c 的动作为条件，看不到“导致它”的同 chunk 动作。②推理 `_infer` 先完整去噪视频，此时输入里根本没有本 chunk 的动作；之后动作流才去 attend 已缓存的预测视频。③外部动作只能作为**历史**，经 `_compute_kv_cache(obs_buffer, executed_actions)` 写入，而且这一步同时需要真实观测帧：`obs_buffer` 为空时直接 return。这些历史动作只影响**下一个** chunk 的视频预测。论文 Eq.8 也写作 z_{t+1:t+K} ~ p(·\|z≤t, a<t)。④论文 §3.4 和 Eq.13 / Alg.2 描述了一个 **FDM（forward dynamics）** 推理步与 L_fdm 后训练损失，它以正在执行的 a_t 为条件“想象” z_t，用于异步部署。但 LeRobot 代码和上游公开的 `wan_va/train.py` / `wan_va_server.py` 中都没有 FDM 损失或 FDM 推理路径（grep 无匹配）。已发布的 LIBERO checkpoint 是否用 L_fdm 训练过：`unverified`。⑤上游有一个 `i2va` 模式：从初始图像开环滚动 N 个 chunk，联合生成视频和动作，没有真实反馈，最后解码成 mp4。LeRobot 没有对应入口。在 LeRobot 中不喂反馈、连续调用 `predict_action_chunk` 时，`_compute_kv_cache` 会提前返回，`_frame_st_id` 不会前进。 | [CODE]/utils.py:386-394, 444-471；[CODE]/modeling_lingbot_va.py:445-457, 717-746, 768-834；[PAPER §3.2 Eq.8-9, §3.4 Eq.13, Alg.2]；[GH]/wan_va/wan_va_server.py:572-605, 641-675；[GH]/wan_va/configs/va_libero_i2va.py:8-11 |
| 自带头 | 只有两个输出头：`proj_out`（视频潜变量的速度场）和 `action_proj_out`（动作的速度场）。**策略头**就是动作流本身，即逆动力学解码器，与视频联合用 flow matching 训练。**没有 reward 头，也没有 value 头。** 代码里没有，论文也没有提到。成功判定完全来自环境的 `check_success`。 | [CODE]/utils.py:864-865, 1099-1105；[CODE]/modeling_lingbot_va.py:260-323；[PAPER §3.3]；[LR]/envs/libero.py:376 |
| 训练目标 | **flow matching 的速度 MSE**。`loss = latent_loss + action_loss`，即 λ=1。每帧的 timestep 权重为钟形（`training_weight`）。动作损失按 action mask 只在使用的通道上计算。代码中的目标为 `noise − sample`，论文写的是 ż = z − ε；两者符号约定相反，σ 的方向也相反。**Noisy History Augmentation**：视频历史以 p=0.5 加噪，s_aug∈[0.5,1]（代码中 `noisy_cond_prob=0.5`，cond timestep 取 [0.5,1]）；动作流不加。**文本 CFG**：上游训练以 0.1 的概率把文本换成空嵌入。LeRobot 的 `forward` 中没有看到文本 dropout。**论文另述**：后训练还额外加入 L_fdm（Eq.13），公开代码中没有（见上一行）。 | [CODE]/modeling_lingbot_va.py:208-323；[CODE]/utils.py:280-289；[PAPER §3.3 Training Objective Eq.10-12, §3.4 Eq.13, §4.2]；[GH]/wan_va/configs/va_libero_train_cfg.py:15；[GH]/wan_va/dataset/lerobot_latent_dataset.py `_cat_video_latents`（cfg_prob） |
| 能否解码出画面 | **能。** `decode_predicted_latents` / `_decode_predicted_video` 先反归一化潜变量，再调用冻结的 Wan2.2 VAE `decode`，输出 `[T,H,W,3]` uint8。打开 `save_predicted_video=True` 后，每个 chunk 的预测潜变量存进 `last_predicted_latents`；`lerobot-eval` 在一次 rollout 结束后把各 chunk 拼接、一次性解码，写成 `pred_episode_*.mp4`。**注意**：①首个 chunk 的第 0 帧被强制替换为真实初始帧的潜变量；②LeRobot 默认 `video_exec_step=-1` 和 20 步完整去噪。论文部署时只积分到 s≈0.5–0.6（3 步），即部分去噪。两者设置不同。 | [CODE]/modeling_lingbot_va.py:462-466, 801-802, 836-853；[CODE]/utils.py:135-139；[LR]/scripts/lerobot_eval.py:522-531, 635-660；[DOC]:95-99；[PAPER §3.3 Noisy History Augmentation, §4.2] |

---

## B. 训练任务与数据

### B.1 `lingbot_va_libero_long` 覆盖的 LIBERO suite 和任务
- **LIBERO-Long = LIBERO-10。** 依据：LeRobot 文档的评测命令是 `--env.task=libero_10`（[DOC]:68）；上游 LIBERO 评测脚本是 `--libero-benchmark libero_10`（[GH]/evaluation/libero/launch_client.sh:5）；配套数据集 `robbyant/libero-long-lerobot` 的 `tasks.jsonl` 正好是 10 条 LIBERO-10 指令，例如 “put both moka pots on the stove”、“turn on the stove and put the moka pot on it” 等（[HF-DS] meta/tasks.jsonl）。
- **后训练有没有用到其他 suite：存在冲突，结论 `unverified`。**
  - 论文 §4.3.2 写 “we train our model on four LIBERO suites: Spatial, Object, Goal, Long … finetuned for 4K steps”。这句话没有说明是 4 个 suite 联合训练一个模型，还是每个 suite 各训一个。
  - 另一方面，已发布的 checkpoint 名为 “posttrain-libero-long”。上游 README 列出的 LIBERO 后训练数据集只有 `libero-long-lerobot`（“LIBERO-Long dataset in LeRobot format for post-training”）（[GH]/README.md:54, 73, 80）。
  - 发布的这个 checkpoint 实际用了哪些数据训练，HF 卡和 config 都没有写。

### B.2 训练数据来源与规模
- **配套数据集 `robbyant/libero-long-lerobot`**（[HF-DS] info.json / episodes.jsonl）：
  - 500 条 episode，10 个任务各 50 条，共 138,090 帧，单条长度 150–517 帧。
  - `robot_type: Franka`，两路相机 `agentview_rgb` 和 `eye_in_hand_rgb`，**128×128**。
  - 预先提取好了 Wan2.2 潜变量，放在 `latents/` 下。
  - `info.json` 写 `fps: 60`，但 LIBERO 的控制频率是 20Hz（[LR]/envs/configs.py:326）。两者矛盾，以哪个为准 `unverified`。
  - 许可为 CC BY-NC-SA 4.0。
- **它是否就是发布 checkpoint 的训练数据：`unverified`。** README 只说它是 “for post-training”。
- **是官方示范还是重新生成的：`unverified`。** 推断（不是证据）：500 = 10×50 条，128×128，最长 517 步。最长长度比 LeRobot 注释中 no-noops 版本的最长 505 步还长（[LR]/envs/libero.py:102），所以它更像是 LIBERO 原始官方 HDF5 示范，而不是 OpenVLA 那种重新生成、去掉 no-op 的版本。但论文 §4.3.2 又写了 “Following OpenVLA, we filter unsuccessful demonstrations”。两者之间的冲突无法核实。
- **反事实或额外数据**：所有来源都没有提到：`unverified`，倾向于没有。
- **后训练超参（存在冲突）**：
  - 论文：4K 步，lr 1e-5，序列长度 1×10^5 token（[PAPER §4.3.2]）。
  - 上游 `va_libero_train_cfg.py`：`num_steps=5000`，lr 1e-5，wd 0.1，warmup 10，batch 1 × grad-accum 10（[GH]/wan_va/configs/va_libero_train_cfg.py:18-25）。
  - LeRobot 预设：AdamW lr 1e-5，wd 1e-4，warmup 1000（[CODE]/configuration_lingbot_va.py:129-135）。
  - 发布 checkpoint 实际用的是哪套：`unverified`。

### B.3 LIBERO 之外的预训练数据
- **LingBot-VA 预训练**（[PAPER §4.1, §4.2]）：
  - 约 **16K 小时**的机器人操作数据，来自 6 个来源：Agibot、RoboMind、InternData-A1（仿真）、OXE（只用 OpenVLA 子集）、UMI Data（不含 DexUMI）、RoboCOIN，另加内部采集的示范。
  - 共 **1.4T tokens**，lr 1e-4，cosine 调度，各数据源均匀采样。
  - Fig.1 的图注写 “pretrained on diverse in-the-wild videos and robot action data”，但 §4.1 只列了机器人数据。是否还有纯视频（无动作）数据：`unverified`。
- **LIBERO 是否出现在预训练数据中**：§4.1 的列表里没有 LIBERO。OXE 的 OpenVLA 子集是否含 LIBERO，论文没有说明：`unverified`。
- **Wan2.2-5B 自身的预训练数据**：论文只引用了 [79] Wan, arXiv:2503.20314。具体数据构成本卡没有核实：`unverified`。

### B.4 观测配置、控制频率、训练序列长度
- **相机**：`observation.images.image`（agentview，第三人称）和 `observation.images.image2`（eye-in-hand 腕部），顺序固定，按宽度拼接（[DOC]:152-160；[CODE]/configuration_lingbot_va.py:87-89）。
- **分辨率**：模型内部 resize 到 128×128（[CODE]/configuration_lingbot_va.py:98-99；[CODE]/modeling_lingbot_va.py:539-541）。
  - 不同来源给的输入分辨率不同：checkpoint `config.json` 的 input_features 写 `(3,256,256)`（[HF-LR] config.json）；文档评测命令用 `--env.observation_height=128`（[DOC]:69）；LeRobot LIBERO env 的默认值是 360（[LR]/envs/configs.py:334）；上游客户端直接以 128×128 渲染（[GH]/evaluation/libero/client.py:85-86）。
- **控制频率**：20Hz（[LR]/envs/configs.py:326）。每个潜帧对应 4 个环境步，约 0.2s/潜帧（由 `action_per_frame=4` 推得）。
- **训练序列长度**：
  - 论文：LIBERO 后训练 1×10^5 token（[PAPER §4.3.2]）；预训练把多条 episode 打包成最长 10K token（[PAPER §4.2]）。
  - LeRobot：每个样本是 16 个环境帧（`observation_delta_indices=range(0,16)`）和 16 个动作（[CODE]/configuration_lingbot_va.py:181-189）；`window_size∈[4,64]`、`chunk_size∈[1,4]` 随机采样（[CODE]/modeling_lingbot_va.py:317-318）。

### B.5 变化覆盖（用于之后的 known/unknown 标注）

| 变化类型 | (a) LIBERO 后训练数据 | (b) 预训练数据 |
|---|---|---|
| 相机视角 | 只有 LIBERO 固定的 agentview + 腕部相机，没有视角扰动。依据：[HF-DS] 只有这两路相机；数据加载器（[GH]/wan_va/dataset/lerobot_latent_dataset.py）只读预提取的潜变量，没有几何增强。有无视角扰动的直接声明：`unverified` | 多本体、多数据集，间接说明视角多样（“diverse embodiments, environments”，[PAPER §4.1]）。具体视角覆盖：`unverified` |
| 光照 | 没有看到光照增强（同上，潜变量预先提取、没有增强代码）。数据集本身是否有光照变化：`unverified` | 间接证据：“in-the-wild”、“diverse environments”（[PAPER Fig.1, §4.1]）。具体：`unverified` |
| 背景/纹理 | LIBERO-10 各任务场景固定，没有额外增强。是否有纹理随机化：`unverified` | 同上，间接证据。具体：`unverified` |
| 干扰物 | 只有 LIBERO-10 场景里原本就有的非目标物体。额外干扰物：`unverified` | `unverified` |
| 物体布局/初始位置（超出 LIBERO 原生 init 分布） | 没有证据表明超出 LIBERO 原生的示范分布：`unverified` | 论文 §4.5.3 只在**真实世界**做了 OOD 位置泛化**测试**，那不是训练数据。预训练数据中：`unverified` |
| 机器人初始位姿 | 跟随 LIBERO 示范和 init state。有无额外扰动：`unverified` | `unverified` |
| 语言改写 | **没有**。每个任务只有一条固定指令，`action_config.action_text` 等于任务名（[HF-DS] tasks.jsonl / episodes.jsonl） | `unverified`。论文只说指令由冻结的 T5 编码，文本 CFG 以 0.1 的概率丢弃（[PAPER §4.2]） |
| 传感器噪声 | 没有图像噪声增强。只有**潜空间**的 noisy-history 增强（p=0.5，s∈[0.5,1]），它是训练技巧，不是传感器噪声（[CODE]/modeling_lingbot_va.py:236-242；[PAPER Eq.10]） | 同样有 noisy-history 增强（[PAPER §4.2]）。真实传感器噪声随真实数据而来，具体：`unverified` |
| 动力学 | 只有 LIBERO 单一仿真器（robosuite / MuJoCo），没有动力学随机化的证据：`unverified` | 多本体加上仿真数据（InternData-A1）（[PAPER §4.1]）。具体动力学变化：`unverified` |

---

## C. 与评测相关的接口事实

- **动作分块**：
  - `chunk_size = n_action_steps = frame_chunk_size × action_per_frame = 16`，整块执行完再重新规划（[CODE]/configuration_lingbot_va.py:142-150）。
  - 第一个 chunk 只返回 12 个动作：第 0 帧的动作被丢弃，与上游 `start_idx=1` 一致（[CODE]/modeling_lingbot_va.py:468-475；[GH]/evaluation/libero/client.py:105）。
  - 只支持单环境（`--eval.batch_size=1`）（[DOC]:74-75；[CODE]/modeling_lingbot_va.py:31）。
- **夹爪后处理**：
  - LingBot-VA 路径里没有二值化。postprocessor 只做 QUANTILES 反归一化，再转到 CPU（[CODE]/processor_lingbot_va.py:252-260）。
  - LIBERO 夹爪通道的 q01/q99 = −1/1，所以反归一化后基本不变（[GH]/wan_va/configs/va_libero_cfg.py:49, 58）。
  - `LiberoEnv.step` 把动作直接传给 `self._env.step(action)`（[LR]/envs/libero.py:374）。
  - 上游客户端也是直接执行，不做二值化（[GH]/evaluation/libero/client.py:107-109）。
- **评测步数上限（存在冲突）**：
  - LeRobot：`libero_10` 为 520 步（spatial/object 280，goal 300），开始前有 10 步 no-op 等待（[LR]/envs/libero.py:98-103, 127, 351）。
  - 上游客户端：`cur_env.env.timestep < 800`，开始前 5 步 no-op（[GH]/evaluation/libero/client.py:64-69, 98）。
- **图像朝向**：
  - 上游只做竖直翻转 `[::-1]`（[GH]/evaluation/libero/client.py:59-60）。
  - LeRobot 的 LIBERO env processor 同时翻转 H 和 W（[LR]/processor/env_processor.py:59）。
  - checkpoint 中 `image_hflip: true` 用来撤销多出的水平翻转（[HF-LR] config.json；[CODE]/modeling_lingbot_va.py:537-538）。代码默认值是 `False`（[CODE]/configuration_lingbot_va.py:91），与 checkpoint 不同。
- **从给定 LIBERO 初始状态做想象**：
  - 第一次 `select_action` 时，把当前观测编码为 `init_latent`，作为第 0 帧条件，然后生成一个 chunk 的视频和动作（[CODE]/modeling_lingbot_va.py:415-421, 445-452）。
  - 初始状态来自 LIBERO 的 `init_states`，按 `episode_index` 固定绑定，经过 `num_steps_wait` 次 no-op 后才是第一帧观测（[LR]/envs/libero.py:168-175, 344-351）。
  - 想象的最小单位是一个 chunk（4 个潜帧 ≈ 16 个环境步），以文本指令为条件，视频 CFG=5，空字符串作为负提示（[CODE]/modeling_lingbot_va.py:517-523, 794-797）。
  - 多 chunk 开环想象只在上游 `i2va` 模式中有（[GH]/wan_va/wan_va_server.py:648-675）。
- **输入外部动作**：现有接口不支持“以外部动作为条件预测这些动作的后果画面”，详见 A 表“能否当动作条件模拟器”一行。外部动作只能和真实观测一起，作为历史写入 KV cache（[CODE]/modeling_lingbot_va.py:717-746）。
- **解码画面**：可以，见 A 表“能否解码出画面”一行。解码出的预测视频按 env 的 `render_fps` 写成 mp4（[LR]/scripts/lerobot_eval.py:655-659；LIBERO `render_fps=80`，[LR]/envs/libero.py:108）。
- **推理超参（LeRobot/上游默认值 vs 论文）**：
  - LeRobot/上游默认：视频 20 步，动作 50 步；CFG 5 / 1；shift 5.0 / 0.05（[CODE]/configuration_lingbot_va.py:103-109；[GH]/wan_va/configs/va_libero_cfg.py:23-31）。
  - 论文：视频 3 步积分到 s=0.6，动作 10 步（[PAPER §4.2]）。两者冲突。
- **报告的 LIBERO 成绩**：LIBERO-Long 98.5±0.5，四个 suite 平均 98.5，取 3 个 seed × 500 次试验（[PAPER §4.3.2 Table 2]；[GH]/README.md）。LeRobot HF 卡写的是 “No evaluation results have been provided”（[HF-LR] README）。
- **显存与性能**：推理约 18–24GB VRAM；UMT5 默认放在 CPU（[DOC]:56-60, 181-182）。

---

## 未核实项（unverified）

1. Wan2.2-5B 的具体变体名（例如 TI2V-5B），以及 Wan2.2 的预训练数据构成。
2. 发布的 `lingbot-va-posttrain-libero-long` 是只用 LIBERO-10 训练，还是联合了 4 个 suite（论文 §4.3.2 与 checkpoint 命名存在冲突）。另外实际训练步数是 4K（论文）还是 5000（上游配置）。
3. 训练数据是否就是 `robbyant/libero-long-lerobot`（500 条，未过滤？），是官方原始示范还是重新生成的；以及 `info.json` 中 fps=60 与 LIBERO 20Hz 的矛盾。
4. 发布的 checkpoint 是否用论文 Eq.13 的 L_fdm（forward dynamics）后训练过。公开代码中没有 FDM 的训练或推理路径。
5. 预训练数据（16K 小时）是否含 LIBERO 或其他 LIBERO 风格的仿真数据；Fig.1 提到的 “in-the-wild videos” 具体指什么。
6. 变化覆盖：预训练数据中的相机视角、光照、背景、干扰物、布局、机器人初始位姿、语言改写、动力学变化，都只有“diverse”这类间接表述，没有具体证据。LIBERO 后训练数据中是否有光照或纹理变化、机器人初始位姿扰动，也没有直接声明。
7. VAE 的参数量和 dtype（只知道文件大小约 2.82GB）。
8. 论文的 5.3B（独立 350M 动作流）与发布权重约 5.09B（共享骨干）之间的差异，按论文所述是“separated version”尚未发布；这一点只来自 README，没有更多证据。
