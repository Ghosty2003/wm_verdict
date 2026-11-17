# FastWAM 模型卡（LeRobot 集成，LIBERO 检查点 `ZibinDong/fastwam_libero_uncond_2cam224`）

> **口径（2026-10-14）**：本阶段只在虚拟环境内评测。文中的“真实”“现实”“real”一律指 **LIBERO 模拟器真值分支**，“迁移”指想象结论到模拟器执行的迁移。物理机器人不在范围内。本档案引用论文中的“真实世界 / real-world”实验时，指的是论文作者的物理机器人实验，与本项目的真值分支无关。

> 用途：LIBERO 上 world-model verdict 评测的第一步，所有条目都附出处；查不到证据的写 `unverified`，来源互相冲突的两边都列出。
> 编写日期：2026-10-14。

**出处缩写**

| 缩写 | 指向 |
|---|---|
| `CFG` | `$LEROBOT_ROOT/src/lerobot/policies/fastwam/configuration_fastwam.py` |
| `POL` | `$LEROBOT_ROOT/src/lerobot/policies/fastwam/modeling_fastwam.py` |
| `PROC` | `$LEROBOT_ROOT/src/lerobot/policies/fastwam/processor_fastwam.py` |
| `MOD` | `$LEROBOT_ROOT/src/lerobot/policies/fastwam/wan/modular.py` |
| `VDIT` | `$LEROBOT_ROOT/src/lerobot/policies/fastwam/wan/video_dit.py` |
| `COMP` / `ADP` | `.../fastwam/wan/components.py` / `.../fastwam/wan/adapters.py` |
| `WANRM` | `.../fastwam/wan/README.md`（从 Wan2.2 vendored 的部分，upstream commit `42bf4cf`） |
| `DOC` | `$LEROBOT_ROOT/docs/source/fastwam.mdx` |
| `DOC2` | `$LEROBOT_ROOT/docs/source/policy_fastwam_README.md` |
| `ENV` | `$LEROBOT_ROOT/src/lerobot/envs/libero.py`，`.../envs/configs.py`，`.../processor/env_processor.py` |
| `HF-CARD` / `HF-CFG` | `ZibinDong/fastwam_libero_uncond_2cam224` 的 `README.md` / `config.json`（2026-10-14 获取；lastModified 2026-06-10） |
| `HF-PRE` / `HF-POST` | 同一仓库的 `policy_preprocessor.json`（含其 normalizer `.safetensors` 统计量，1.2 KB）/ `policy_postprocessor.json` |
| `HF-STATS` | 同一仓库的 `libero_uncond_2cam224_dataset_stats.json`（sha256 与 `yuanty/fastwam` 中的同名文件完全一致） |
| `HF-HDR` | 同一仓库 `model.safetensors` 的 header（只用 HTTP range 读取 header 拿张量名/形状/dtype，**没有下载权重**） |
| `UP-HF` | 上游检查点仓库 `yuanty/fastwam` 的 README |
| `DS` | 上游数据集 `yuanty/LIBERO-fastwam` 的 README 和 `lerobot_v30/*/meta/info.json` |
| `PAPER` | Yuan et al., *Fast-WAM: Do World Action Models Need Test-time Future Imagination?*，arXiv:2603.16666v2 |
| `GH` | 上游代码 `github.com/yuantianyuan01/FastWAM`，main 分支 commit `7faa711`（2026-08-20） |
| `WAN` | `Wan-AI/Wan2.2-TI2V-5B` 的 HF README |

---

## A. 结构与骨干

| 字段 | 内容 | 出处 |
|---|---|---|
| 骨干来源 | **在预训练模型上微调，不是从零训练。** 基座是 **Wan2.2-TI2V-5B**（`Wan-AI/Wan2.2-TI2V-5B`），沿用它的 video DiT、T5 文本编码器（UMT5-XXL）和 video VAE（Wan2.2-VAE）。**冻结的部分**：VAE 和 UMT5 文本编码器（`requires_grad_(False)`，不注册为子模块，也不进 `model.safetensors`）。**训练的部分**：整个 MoT DiT（video expert 加 action expert）以及 proprio encoder；LeRobot 里可选 `freeze_video_expert`，默认 `False`。上游训练器同样只训练 `dit` 和 `proprio_encoder`。**Action expert 的初始化**：上游先把 Wan2.2 DiT 权重沿隐藏维做线性插值（再乘 alpha=sqrt(dv/da)）缩到 1024 维，得到 `ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` 作为初值，所以 action expert 也不是随机初始化。论文明确说**没有 embodied（机器人）预训练**。 | PAPER §3.2 "Architecture"、§4.1；DOC L9、L40-44；CFG L30-33、L182-186、L219；MOD L868-878；POL L80-90、L142-152、L247-256；GH `README.md`「pre-generate the ActionDiT backbone (interpolated from Wan22 DiT)」、`scripts/preprocess_action_dit_backbone.py` L148-216、`configs/model/fastwam.yaml` L10、`src/fastwam/trainer.py` L82-90、L304-318；PAPER Abstract、Table 2（"Embodied PT. ✗"） |
| 生成机制 | **连续潜空间上的 flow matching**（rectified-flow 式线性插值 `y_t=(1-t)y+tε`，目标速度 `ε−y`），video latent 和 action chunk 用同一种形式。没有自回归 token，也没有离散 token 扩散。推理时用 Euler 步 `x ← x + v·Δσ`，默认 10 步，CFG=1.0。**采样分布有冲突**：论文说训练和推理都用 "logit-normal distribution over t"；LeRobot 代码则是 `u~U(0,1)` 再做 shift 变换 `σ=s·u/(1+(s−1)u)`（s=5），训练损失再乘一个高斯形的时间步权重。 | PAPER §3.2 "Training objective" Eq.(5)-(9)、§4.1；VDIT L105-195（`WanContinuousFlowMatchScheduler`：`sample_training_t` L132-138、`training_weight` L140-148、`add_noise` L150-159、`training_target` L161-164、`step` L189-195）；CFG L209、L212、L221-226 |
| 参数规模 | **总计 6,020,710,599（约 6.02B），bf16**（HF-HDR 统计）：video expert（Wan2.2 DiT，30 层，hidden 3072）**4,999,787,712**；action expert（30 层，hidden 1024）**1,020,886,023**；proprio encoder（Linear 8→4096）**36,864**。论文称 "1B action expert … total 6B"。**VAE 和 UMT5-XXL 不在内**（不在 `model.safetensors` 里）：HF 仓库里 `Wan2.2_VAE.safetensors` 有 2,818,778,392 B，`models_t5_umt5-xxl-enc-bf16.safetensors` 有 11,361,845,464 B。这两个模块的参数量 **unverified**（没有读它们的 header）。 | HF-HDR；PAPER §4.1；MOD L868-878；HF API 文件大小 |
| 表示与分辨率 | **输入帧**：两路相机（agentview `observation.images.image` 和腕部 `observation.images.image2`），每路 224×224，**横向拼成 224×448** 后送进 VAE。**VAE**：Wan2.2-VAE，空间 16×、时间 4×压缩，48 个 latent 通道（latent 用 mean/std 标准化）。**DiT patch**：`[1,2,2]`，总压缩 4×32×32。按代码推算：每个 latent 帧 14×28 → patchify 后 **7×14=98 个 video token**。**时间**：一个训练窗口取 33 个原始 env 帧（`num_video_frames=33`），每 4 帧取 1 帧（`action_video_freq_ratio=4`），得到 **9 个视频帧**（对应 env 步 0,4,…,32）；VAE 时间压缩 4× 后是 **3 个 latent 帧**（第 1 个是条件帧，后 2 个是未来帧）。所以**每个未来 latent 帧≈4 个模型帧≈16 个 env 步**（推算）；1 个模型帧=4 个 env 步。HF-CFG 里没有 `action_video_freq_ratio` 这个键，取的是代码默认值 4（上游 data config 也是 4）。 | HF-CFG `input_features`、`image_size=[224,448]`、`num_video_frames=33`、`video_dit_config.patch_size`；CFG L194-196、L339-356、L387-400；POL L408-426；ADP L25-38；MOD L1063-1075、L1666-1676；WAN README L196；GH `configs/data/libero_2cam.yaml` L8-37 |
| 动作接入方式 | **MoT（Mixture-of-Transformers）共享注意力**：每层把 video expert 和 action expert 的 Q/K/V 拼起来做一次带掩码的 SDPA，FFN 和 cross-attn 各用各的参数。动作是 action expert 里的 **32 个 token**（`Linear(7→1024)` 编码，1D RoPE）。**注意力掩码**：action→action 双向；**action→只看第一帧（干净条件帧）的 video token**；video→video 是 `first_frame_causal`（首帧只看自己，未来帧看全部 video token）；**video→action 全部屏蔽**。所以在这个检查点里，**video 分支看不到任何动作，action 分支也看不到未来 video**。代码里另有一条 `action_conditioned=True` 的路径（动作作为 video expert 的 cross-attn context，按组对角掩码），但**这个检查点 `action_conditioned=false`**，`model.safetensors` 里也没有 `action_embedding` 权重。**动作维度** 7（6 维 EEF delta 加 1 维夹爪）。**归一化**：检查点 processor 用 **MIN_MAX**（state 和 action）；LeRobot 代码默认是 MEAN_STD，二者冲突，以检查点里保存的 processor 为准。上游 data config 也是 `norm_default_mode: min/max`，前 6 维是 delta，夹爪不是 delta。训练数据里夹爪动作 ∈[0,1]（HF-STATS action min/max 最后一维为 0/1）。 | MOD L244-326、L454-525（MoTLayer）、L1216-1237（`_build_mot_attention_mask`）；MOD L70-126（ActionDiT）；VDIT L583-617、L679-722、L500-503；HF-CFG `video_dit_config.action_conditioned=false`、`normalization_mapping`；HF-HDR（无 `video.action_embedding.*`）；HF-PRE；CFG L230-236；HF-STATS `action.global_min/max`；GH `configs/data/libero_2cam.yaml` L42-53；PAPER §3.2 "Architecture" |
| 上下文与预测窗口 | **历史条件**：只有**当前 1 帧**（两路相机拼图）、当前 1 个 proprio（8 维，作为额外的文本 context token），以及语言指令（`n_obs_steps=1`）。训练时 proprio 只用第 0 帧。**单次前向的预测量**：动作 32 步；video（仅训练时，或调用 `infer_joint` 时）预测 9 帧中后 8 帧对应的 2 个未来 latent 帧，覆盖约 32 个 env 步。**拼接长 rollout 的方式**：LeRobot `select_action` 用动作队列执行 `n_action_steps` 步（检查点 config 是 32，文档评测命令用 10），然后用**真实环境的新观测**重新规划，是闭环 receding-horizon。**没有**视频自回归或滑窗拼接的实现。**能否当动作条件模拟器（喂外部动作、得到未来帧）：这个检查点不能。** 理由有四条：(1) `video_dit_config.action_conditioned=false`；(2) `infer_joint(action=...)` 的 `action` 只传到 `video_expert.pre_dit`，在 `action_conditioned=False` 时会被直接忽略；(3) MoT 掩码禁止 video→action 注意力；(4) 权重里没有动作嵌入层。`infer_joint` 能生成未来帧，但这些帧**只以首帧加语言（加 proprio）为条件**，相当于"模型自己的隐式策略"想象出的未来，和同时采出来的动作 chunk 之间也没有信息交互。LeRobot 的 `FastWAMPolicy` 只暴露动作接口，`predict_action_chunk` 走 `infer_action`，只做首帧 prefill 加动作去噪，不生成视频。 | CFG L189-194；POL L208-245、L310-326；MOD L1109-1213（proprio 取 `[:,0]` 在 L1191）、L1041-1060、L1679-1786（`infer_joint`，L1685-1686 注释 "gt action for conditioning videos"、L1767 `gt_action=action`）、L1789-1904（`infer_action`）；VDIT L679-722；HF-CFG；HF-HDR；DOC L77-93；PAPER §3.2（"At inference time, Fast-WAM removes the future video branch entirely"） |
| 自带头 | **Policy head**：action expert 的 `head = Linear(1024→7)`，输出动作速度场，用 flow-matching MSE 训练。**Video head**：Wan DiT 的 head，输出 48 通道 latent 速度场，用 flow-matching MSE 训练。**Reward head / value head：没有**（代码里没有，`reward_delta_indices` 返回 `None`）。 | MOD L56-67、L126、L205-206；VDIT L742-744；CFG L406-408；MOD L1364-1421 |
| 训练目标 | `L = λ_action·L_act + λ_video·L_vid`，两项都是**带时间步权重的 flow-matching 速度 MSE**。video loss 不算首帧（首帧 latent 换成干净 latent，timestep 记 0），按 `image_is_pad` 掩掉 padding；action loss 按 `action_is_pad` 掩掉。检查点里 `lambda_video=1`、`lambda_action=1`。论文写法是 `L_act + λ·L_vid`，没给 λ 的数值。 | MOD L1239-1421；VDIT L656-661；HF-CFG `loss`；PAPER Eq.(9) |
| 能否解码出画面 | **能，但只在 `infer_joint` 路径上**：`FastWAM._decode_latents` 用冻结的 Wan2.2-VAE 把 latent 解码成 uint8 PIL 帧列表（9 帧，224×448，左边 agentview、右边腕部，按 key 排序后拼接）。LeRobot 的 `select_action` / `predict_action_chunk` **不会**生成或解码视频，要直接调 `policy.model.infer_joint(...)`。上游 LIBERO 评测脚本有 `visualize_future_video` 选项，会用 `infer_joint` 解码未来帧，并和 GT 计算 PSNR（要求 `action_conditioned=false`）。可以在想象帧上读成功谓词，但这些帧**不受外部动作控制**。 | MOD L1097-1107、L1783-1786；ADP L91-108；POL L408-426；GH `experiments/libero/eval_libero_single.py` L282-310、L406-436 |

---

## B. 训练任务与数据

**B.1 LIBERO 套件和任务（本检查点）**
- 用 **LIBERO-Spatial、Object、Goal、Long(10)** 四个套件**联合训练一个多任务模型**（每套 10 个任务，共 40 个）。**没有用 LIBERO-90**（训练 `dataset_dirs` 里没有它）。出处：PAPER §4.2 "LIBERO"（"train our models on the four LIBERO suites"）；GH `configs/data/libero_2cam.yaml` L3-7；DS `info.json`（每套 `total_tasks=10`）。
- 本检查点和上游 `yuanty/fastwam` 的 `libero_uncond_2cam224.pt`（对应任务 `libero_uncond_2cam224_1e-4`）**很可能是同一模型的格式转换版**。证据：dataset_stats 文件 sha256 完全相同；MOD L681-684 注释称它是 "the released `ZibinDong/fastwam_libero_uncond_2cam224`"，用的是旧版 key 布局；`.pt` 和 `model.safetensors` 大小相近（12.04 GB）。但没有官方明确声明，严格说是 `unverified`。

**B.2 训练数据来源与规模**
- **数据是重新生成/处理过的 "no-noops" 版本，不是 LIBERO 官方原始 HDF5**：`yuanty/LIBERO-fastwam`，"preprocessed LIBERO data … not an official upstream LIBERO data dump"，用 **MuJoCo 3.3.2** 生成，目录名为 `libero_*_no_noops_lerobot`。出处：DS README；GH README "Dataset Download"。
- 规模：**1,712 条轨迹，277,713 帧**（Spatial 434 / 53,229；Object 457 / 67,309；Goal 433 / 52,895；10 为 388 / 104,280），fps=20。按 20 Hz 推算约 **3.86 小时**。出处：DS README 和各套件 `meta/info.json`；HF-STATS `num_episodes=1712`、`num_transition=277713`。
- **冲突**：PAPER §4.2 写 "Each suite contains 500 demonstrations"（4 套共 2000 条）；实际发布的数据和检查点统计是 1,712 条（过滤 no-op 或失败轨迹之后的数字，具体过滤规则 `unverified`）。
- 反事实数据、失败轨迹、额外合成数据：FastWAM 来源都没有提到，**unverified**。
- 训练步数：PAPER §4.2 写 "All models are trained for 20k steps"；GH `configs/task/libero_uncond_2cam224_1e-4.yaml` 写 `num_epochs: 10`、`batch_size: 16`（每卡），README 写 8 卡单节点。两者口径不同，有没有冲突 `unverified`。

**B.3 LIBERO 以外的预训练数据**
- **FastWAM 本身没有机器人（embodied）或跨本体预训练**。出处：PAPER Abstract、§1、Table 2。
- **基座 Wan2.2-TI2V-5B 的预训练数据**：网络规模的图像和视频，"compared to Wan2.1 … +65.6% more images and +83.2% more videos"，并且有带光照、构图、对比度、色调等标签的美学数据。具体数据集、规模、是否含机器人视频：**unverified**（Wan2.2 README 没有列明）。出处：WAN README L28-30。
- UMT5-XXL 文本编码器的预训练数据：**unverified**。
- `lerobot/fastwam_base`（LeRobot 默认 `base_model_id`，CFG L32、L201）和本检查点的关系：**unverified**。它创建于 2026-06-18，晚于本检查点。

**B.4 观测配置、控制频率、序列长度**
- 相机：`agentview`（第三人称）和 `robot0_eye_in_hand`（腕部），各 224×224，横向拼成 224×448。出处：HF-CFG `input_features`；GH `configs/data/libero_2cam.yaml` L9-17、L31、L36。
- 分辨率有**冲突**：数据集里存的原始帧是 **512×512**（DS `info.json` 中 `observation.images.image [512,512,3]`；GH data config `raw_shape: [3,512,512]`）；上游评测注释写 `LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data`（GH `experiments/libero/libero_utils.py` L16）。
- 训练图像变换只有 `ToTensor` 加 `Resize(224,224)`，**没有颜色、裁剪等数据增强**。出处：GH `configs/data/libero_2cam.yaml` L58-65。
- Proprio 8 维：EEF 位置 3 维、轴角 3 维、夹爪 qpos 2 维。出处：GH data config L43；GH `eval_libero_single.py` L244-256；ENV `processor/env_processor.py` L28-80。
- 控制频率：**20 Hz**（DS `info.json` `fps=20`；ENV `envs/configs.py` `LiberoEnv.fps=20`）。
- 训练序列长度：每个样本是 33 个原始帧窗口，含 32 个动作和 9 个视频帧（stride 4）；`global_sample_stride=1`。出处：GH data config L28-30；CFG L189-196、L393-404。
- 文本模板：`"A video recorded from a robot's point of view executing the following instruction: {task}"`，`tokenizer_max_len=128`。出处：CFG L206-208；GH `src/fastwam/datasets/lerobot/robot_video_dataset.py` L23；HF-CFG。

**B.5 变化覆盖（供后续 known/unknown 标注）**

| 变化类型 | (a) LIBERO 微调数据 | (b) 预训练数据（Wan2.2） |
|---|---|---|
| 相机视角变化 | FastWAM 来源没有说明有视角扰动，训练也没有几何增强（libero_2cam.yaml L58-65）；固定的两路相机键（agentview + wrist）。数据本身是否含视角变化：**unverified** | **unverified**（Wan2.2 README 只说是更大规模的网络视频） |
| 光照变化 | 没有颜色或光照增强（同上）。数据本身是否含光照变化：**unverified** | 有带 "lighting" 标签的美学数据（WAN README L28），说明数据中有光照多样性；具体分布 **unverified** |
| 背景/纹理变化 | 没有增强。数据本身：**unverified** | **unverified** |
| 干扰物 | **unverified**（FastWAM 来源没有提到） | **unverified** |
| 超出 LIBERO 原生初始分布的物体布局/初始位置变化 | FastWAM 来源没有提到任何超出原生分布的扰动，数据是从 LIBERO 预处理来的（DS README）。是否有超出：**unverified** | 不适用 / **unverified** |
| 机器人初始位姿变化 | **unverified** | **unverified** |
| 语言改写（paraphrase） | **没有**。每个套件 `total_tasks=10`（DS `info.json`），每个任务一条指令；上游 processor 默认 `drop_high_level_prob=1.0`，只用 low-level 指令原文，没有改写增强（GH `fastwam_processor.py` L38-39、L122-150）；文本模板固定（CFG L206-208） | **unverified** |
| 传感器噪声 | 没有图像或 state 噪声增强（`action_state_transforms: null`，图像只 Resize；GH data config L48、L58-65）。数据本身是否带噪声：**unverified** | **unverified** |
| 动力学变化（摩擦/质量/增益） | **unverified**（只知道数据用 MuJoCo 3.3.2 生成，且"对 MuJoCo 版本敏感"，DS README） | 不适用 / **unverified** |

---

## C. 与评测相关的接口事实

**C.1 动作 chunk、执行步长、夹爪后处理、最大步数**
- 动作 chunk：`action_horizon=32`（HF-CFG；PAPER §4.1 "h=32"）。执行步长有冲突：HF-CFG 里 `n_action_steps=32`；DOC 和 DOC2 的评测命令都显式传 `--policy.n_action_steps=10`；上游 `configs/sim_libero.yaml` L25 是 `replan_steps: 10`。复现论文和文档结果应该用 **10**。
- 推理设置：`num_inference_steps=10`，`text_cfg_scale=1.0`，`inference_seed=42`（CFG L209-212）；`sigma_shift` 在 LeRobot 中为 `None`，即用 config 的 infer_shift=5（HF-CFG `action_scheduler.infer_shift=5`）。上游 README 要求评测原始发布检查点时设 `EVALUATION.sigma_shift=5.0`；上游新版 model config 的 action shift 已改为 1.0（GH `configs/model/fastwam.yaml` L53-56），**和本检查点不同**。
- 夹爪后处理：`toggle_action_dimensions=[-1]`，对最后一维做 `sign(-(2a-1))`，把训练时 [0,1] 的约定（1=开）映射成 LIBERO 的 −1=开、+1=关，并二值化。出处：HF-CFG；HF-POST；PROC L38-57、L111-113；DOC L139-147。上游等价实现：`a*2-1` 再取反再 `np.sign`（GH `eval_libero_single.py` L440-448；`libero_utils.py` L182-189）。
- 最大步数（三处来源**互相冲突**）：
  - LeRobot env 默认 `TASK_SUITE_MAX_STEPS`：spatial 280 / object 280 / goal 300 / 10 为 520 / 90 为 400（ENV `envs/libero.py` L98-104）。
  - LeRobot FastWAM 文档评测命令：`--env.episode_length=600`（libero_10），spatial/object/goal 用 300（DOC L77-96；DOC2 L41-56）。
  - 上游评测：spatial/object/goal 为 400，libero_10 和 libero_90 为 700，另加 `num_steps_wait=30` 步空动作（GH `eval_libero_single.py` L451-462、L478-502；`configs/sim_libero.yaml` L24）。LeRobot env 的 `num_steps_wait` 默认是 10（ENV `envs/libero.py` L127）。
- 环境观测分辨率有**冲突**：DOC L87-88 用 `observation_height/width=224`；DOC L161 和 DOC2 L41、L52 用 256；LeRobot `LiberoEnv` config 默认是 360（`envs/configs.py`）；上游评测渲染 256，然后对每路相机**中心裁剪加缩放**到 224（GH `eval_libero_single.py` L155-224）。LeRobot 路径是对每路相机做 bilinear（antialias）缩放到 224×224，**不裁剪**（POL L391-426）。
- LeRobot 的 LIBERO env processor 会把图像旋转 180°（同时翻转 H 和 W）；上游评测同样用 `[::-1, ::-1]`（ENV `processor/env_processor.py` L45-59；GH `libero_utils.py` L47-51）。
- 报告成功率有冲突：PAPER Table 2 和 HF-CARD 是 **97.6%**（Spatial 98.2 / Object 100.0 / Goal 97.0 / Long 95.2）；LeRobot 复现（DOC L149-159）是 **96.4%**（97.6 / 99.0 / 95.0 / 94.0，每套 500 episode）；UP-HF 中 Optional-IDM 检查点的 first-frame 模式是 97.75%（那是另一个检查点）。

**C.2 想象、外部动作、解码相关的接口**
- **从给定的 LIBERO 初始状态起想象**：可以。用 `env.set_init_state(init_state)` 渲染出首帧（两路相机，旋转 180°，缩放并拼成 [1,3,224,448]，取值 [0,1]），加上 8 维 proprio 和带模板的 prompt，调用 `policy.model.infer_joint(prompt, input_image, num_video_frames=9, action_horizon=32, proprio=..., num_inference_steps=10, seed=...)`，返回 `{"video": 9 张 PIL 帧, "action": [32,7]（归一化空间）}`。注意事项：
  - `test_action_with_infer_action` 默认为 True，此时必须提供 `seed`（MOD L1700-1702）；
  - `infer_joint` 的 `num_inference_steps` 默认是 20，而 policy 默认 10（MOD L1692 对比 CFG L209）；
  - 返回的动作需要经过 HF-POST（unnormalize 加 toggle）才能执行；
  - proprio 需要按检查点的 MIN_MAX 统计归一化（HF-PRE）。
  - 出处：MOD L1580-1621、L1679-1786；ENV `envs/libero.py` L344-346。
- **喂入外部动作**：这个检查点**不支持**（见 A 表"上下文与预测窗口"）。`infer_joint` 的 `action` 参数会被静默忽略，想象视频只取决于首帧、语言和 proprio。如果要做动作条件模拟器，需要用 `action_conditioned=True` 重新训练（代码路径在 VDIT L500-503、L679-713），或者使用上游的 IDM / Optional-IDM 变体。这些变体的视频分支是否以动作为条件：**unverified**（UP-HF 只说 IDM 是"先想象视频再预测动作"）。
- **解码画面**：可以（MOD L1097-1107），输出 9 帧，对应 env 步 0,4,…,32（首帧是输入帧的重建）；每帧左半是 agentview，右半是腕部相机。只有 1 次前向（覆盖 32 步），**没有内置的长时程视频 rollout**。要更长的想象，只能把最后一帧当作新首帧自回归调用；这种做法没有被训练或验证过，**unverified**。
- **图像值域可能被双重归一化（需要实测核实）**：本地代码的设计是 VISUAL 走 IDENTITY，模型在 VAE 编码边界做一次 `x*2-1`（PROC L85-89；MOD L1063-1068、L1087-1088）。但检查点保存的 preprocessor 对两路图像用 **MEAN_STD，mean=std=0.5**（HF-PRE normalizer 统计量；HF-CFG `normalization_mapping.VISUAL=MEAN_STD`），而 `lerobot-eval` 会加载这个保存的 pipeline（`policies/factory.py` L182-209）。如果两处都生效，送进 VAE 的图像会落在 [−3,1]。走 `select_action` 时是否真的发生、对 96.4% 的复现结果有什么影响：**unverified**。手动调用 `infer_joint` 时应该直接传 [0,1] 图像。上游评测脚本自己先把图像变成 [−1,1]（GH `eval_libero_single.py` L236-237），上游模型内部是否再做一次映射：**unverified**。
- 推理延迟（只是参考）：论文报告 190 ms（RTX 5090D，只预测动作），Fast-WAM-IDM 是 810 ms（PAPER §4.3.3）；上游 README 报告新版 H20 上是 210 ms。`infer_joint`（完整视频去噪加解码）的延迟：**unverified**。

---

## 未核实项

1. VAE（Wan2.2-VAE）和 UMT5-XXL 编码器的具体参数量（只知道文件大小）。
2. 训练时间步采样分布：论文写 logit-normal，代码是 uniform 加 shift 变换再加高斯权重，以哪个为准没有确认。
3. 发布的 1,712 条轨迹（论文写每套 500 条，共 2000）是怎么过滤的（no-op 或失败轨迹），以及是否含反事实或失败数据。
4. 论文写训练 20k steps，上游 config 是 10 epochs × bs16 × 8 卡，两者是否一致。
5. 本检查点和 `yuanty/fastwam` 的 `libero_uncond_2cam224.pt` 是否严格是同一组权重（stats 完全一致，但没有官方声明）。
6. LIBERO 微调数据中相机视角、光照、背景/纹理、干扰物、超出原生分布的布局、机器人初始位姿、传感器噪声、动力学的变化覆盖（FastWAM 来源都没有说明；能确认的只有"训练没有图像增强"和"没有语言改写"）。
7. Wan2.2 预训练数据的具体构成（数据集名、规模、是否含机器人或第一视角操作视频、视角和背景分布）。
8. 训练数据原始渲染分辨率：512（数据集 info.json）还是 256（上游评测注释）。
9. 检查点 preprocessor 的 VISUAL MEAN_STD(0.5,0.5) 加上模型内部的 `x*2-1`，是否在 LeRobot 评测中造成图像双重归一化，以及对结果的影响。
10. 上游 IDM / Optional-IDM 变体的视频分支是否以动作为条件（能否当动作条件模拟器）。
11. 用首帧自回归拼接做长时程想象的质量（没有训练或验证过）。
12. `infer_joint` 的延迟；`lerobot/fastwam_base` 和本检查点的关系。
