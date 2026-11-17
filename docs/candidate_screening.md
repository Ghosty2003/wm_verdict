# LIBERO 批次：补充候选筛查（WorldVLA / OpenWAM-FDM / dWorldEval）

2026-10-14。闸门口径同 `eval_design.md` 第 8 节（条 1 可重置生成、条 2 联合一致性、条 3 帧可解码），H 口径同第 9 节（真值分支 H = 520）。

**方法**：只看代码、配置和论文，没有下载任何权重。小文本（`config.json`、`args.json`、`config.yaml`、README）通过 HF raw/API 读取。GitHub 仓库只读浅克隆到 scratchpad，没有安装、也没有运行。下文的 `file:line` 都指以下快照：

| 仓库 | 分支 / commit | 日期 |
|---|---|---|
| `alibaba-damo-academy/WorldVLA`（现主分支为 RynnVLA-002） | `main` @ `e548ccc` | 2025-12-02 |
| 同上，原 WorldVLA 代码 | `worldvla` @ `ebdc36d` | — |
| `OpenWAM/OpenWAM`（Stanford） | `main` @ `030da76` | 2026-10-10 |
| `OpenWAM-Official/OpenWAM`（同名、另一项目，见 §2.2） | `main` @ `1e103e1` | 2026-10-14 |
| `dworldeval/dworldeval.github.io`（只有项目主页） | `main` @ `7dee38b` | 2026-05-06 |

"PASS（仅论文）"表示论文正文支持该结论，但没有代码可以核对。`unverified` 表示现有材料无法判定。

---

## 0. 汇总表

| 候选 | LIBERO 检查点公开？ | 条 1 可重置生成 | 条 2 联合一致性 | 条 3 帧可解码 | rollout 长度 | 判定 |
|---|---|---|---|---|---|---|
| **WorldVLA / RynnVLA-002** | 公开：4 个 suite 各一份，含 `libero_10`；每份约 14 GB，7B | PASS（代码） | PASS（画面以动作为条件；动作可由同一权重的 VLA 模式提供，交替循环需自写） | PASS（Chameleon VQGAN 解码） | 每次调用生成 1 步 1 帧，只看上一帧（Markov）。官方评测把整条验证轨迹自回归滚完；LIBERO-10 演示最长 505 步。滚到 520 步需 520 次调用，代码路径支持，但该长度下的质量未验证 | **可进入实测** |
| **OpenWAM FDM**（Stanford，arXiv 2610.07922） | 否。只公开纯视频预训练权重；LIBERO 策略检查点"待发布"；FDM 检查点不在发布路线图上 | PASS（代码） | PASS（FDM：干净的外部动作作为条件生成视频；动作需由外部策略提供） | PASS（Wan VAE 解码） | 每块 4 个潜帧对应 16 个动作；诊断脚本默认 16 个潜帧（约 64 步）。分块因果续滚到 520 步未验证 | **待定**（需自训 FDM，或等作者发布） |
| （附）**OpenWAM-α LIBERO**（同名的另一项目，arXiv 2609.07398） | 公开：`OpenWAM/OpenWAM-Alpha-Sim-LIBERO`，24.8 GB，单个检查点覆盖 4 个 suite（40 个任务） | PASS（代码，`first_frame_image`） | PASS（`mutual` 掩码：视频与动作双向可见，联合去噪） | PASS（`decode_video=True` 经 VAE 解码；部署默认关闭，需手动打开） | 每次调用输出 32 个动作和 9 帧（视频步长 4）。到 520 步需约 17 次链式调用，属于训练外外推，代码未提供 | **可进入实测**（注意：它不是 FDM，而是联合 WAM） |
| **dWorldEval**（arXiv 2604.22152） | 否。没有代码，也没有权重 | PASS（仅论文 §3.1） | PASS（仅论文 §3.2.1） | unverified（论文未明说） | unverified（论文未给最大长度） | **关闭**（不可获得） |

---

## 1. WorldVLA（及其升级版 RynnVLA-002）

- **论文**：WorldVLA，arXiv **2506.21539**（阿里达摩院）。升级版 RynnVLA-002，arXiv **2511.17502**。
- **代码**：`github.com/alibaba-damo-academy/WorldVLA`，主分支已换成 RynnVLA-002（README 新闻：2025-11-10 升级）；原 WorldVLA 代码在 `worldvla` 分支。`worldvla` 分支 README 第 233 行仍有未完成项"Release the code of world model on LIBERO dataset"。世界模型的 LIBERO 评测代码只在 RynnVLA-002 主分支里。
- **生成机制**：自回归离散 token，基于 Chameleon 7B。图像用 VQGAN（压缩比 16，码本 8192；256² 对应 256 个 token，512² 对应 1024 个 token）；动作每维离散成 256 个 bin（论文 §3.2）。

### 闸门

1. **可重置生成：PASS。** 世界模型 rollout 以外部提供的首帧图像为种子：`rynnvla-002/eval_solver_libero_g_video_512_third_wrist.py:206`（`his_img.append(Image.open(img_files[0]))`，wrist 视角在第 221 行）。输入是任意 PIL 图像，所以可以喂入 LIBERO 指定初始状态渲染出的首帧。预处理要和训练一致：先旋转 180°（`libero_util/regenerate_libero_dataset_save_img_action_state_wrist.py:141,149`），再缩放到 512²。种子是图像而不是模拟器状态，从状态到图像的渲染由我们完成。
2. **联合一致性：PASS（画面以动作为条件）。**
   - 世界模型模式的序列是"当前帧（第三人称 + wrist）→ 动作 → 生成下一帧"，动作 token 在被生成的帧 token 之前：`libero_util/Chameleon_utils.py:759`（prompt `"<|image|><|image|><|action|>"`）、`:766-767`（只取上一帧和上一个动作）、`:787`（`model.generate_img`）。论文 §3.3 的序列为 `[BOI]{image}[EOI][BOA]{action}[EOA]` 后接 `[BOI]{image}[EOI]`。
   - **注意**：动作和画面**不是在同一次生成里联合产出的**。论文 §3.3 写明"We mix the action model data and world model data"：同一权重，两种对话格式，分开调用。官方评测里动作来自数据集文件（`eval_solver_...third_wrist.py:232`），也就是外部动作。
   - verdict 轨道用的自有策略：同一检查点的 VLA 模式（输入"图像历史 + 状态"，输出动作块）可以提供动作。把想象帧作为 VLA 的历史输入，就形成"动作看画面、画面看动作"的交替闭环。这个循环仓库里没有，需要自写胶水代码。
3. **帧可解码：PASS。** `Chameleon_utils.py:829-830` 中的 `item_processor.decode_image(...)` 把 VQ token 解码成 PIL 图像。

### 其他记录

- **LIBERO 检查点**（全部公开、未 gated）：
  - `Alibaba-DAMO-Academy/WorldVLA`：`model_256/{spatial,object,goal,10}` 和 `model_512/{...}`，每份约 14.0 GB，card 许可 apache-2.0。这些是动作模型与世界模型混合训练的结果（`1a2i` 配置）。世界模型模式的评测代码在这个分支里没有发布。
  - `Alibaba-DAMO-Academy/RynnVLA-002`：`VLA_model_256/*`、`World_model_512/*`、`Action_World_model_512/*`，每类 4 个 suite 齐全，每份约 14.1 GB。合计 12 份，约 169 GB。
  - **LIBERO-10 覆盖**：两个仓库都有 `libero_10` 目录，按 suite 训练，覆盖该 suite 的全部 10 个任务。验证轨迹列表：`rynnvla-002/exps_libero_world_model/10_val_ind_trajectory_paths.json`。README 中 LIBERO-Long 的 Action World Model 指标为 FVD 427.86 / PSNR 19.36。
- **模型规模**：7B（`config.json`：hidden 4096，32 层，vocab 65536，`max_position_embeddings` 8196）。
- **许可**：代码 Apache-2.0（README "License" 节；`worldvla` 分支附有 LICENSE）。RynnVLA-002 的 HF card 没有 license 字段。权重派生自 Chameleon / Lumina-mGPT，Chameleon 自身许可是否传递：unverified。
- **到 520 步**：
  - 每次调用生成 1 步 1 帧，只用上一帧和上一个动作（Markov，没有长程记忆）。
  - 官方评测按整条验证轨迹逐步滚动，并把生成帧回灌作为下一步输入（`eval_solver_...third_wrist.py:229,245`）。也就是说，训练内的用法本来就是全轨迹自回归。
  - LIBERO-10 演示最长 505 步（`eval_solver_libero_continous_w_state.py:231`：`max_steps = 520  # longest training demo has 505 steps`）。
  - 滚到 520 步等于 520 次调用，每次贪心解码约 2×1024 个图像 token，代价高，但没有上下文长度障碍。
  - 训练数据过滤了 no-op 动作（README "VLA Model Training"），真值分支重放时要注意步数对齐。
- **未决细节**：`Action_World_model_512/libero_10/args.json` 写的是 `resolution: 256`、`data_config_train: ..._1a2i.yaml`，与目录名 512 不一致。哪一份检查点同时适合在 512² 下做 VLA 和世界模型：unverified。

**小结**：检查点公开，LIBERO-10 齐全，三闸门代码证据都成立，生成帧以动作为条件，可以从任意首帧开始，并能整条轨迹自回归滚动。主要成本有两项：自写"VLA 出动作 ↔ 世界模型出帧"的交替循环；520 次 7B 自回归调用的推理开销。**可进入实测。**

---

## 2. OpenWAM

"OpenWAM"这个名字对应**两个不同项目**，arXiv 编号和代码仓库都不同。用户点名的 FDM 属于 Stanford 项目（§2.1）。另一个同名项目有公开的 LIBERO 检查点，一并记录在 §2.2。

### 2.1 OpenWAM 前向动力学模型（Stanford，用户点名的候选）

- **论文**：《OpenWAM: An Open Framework for Composable World-Action Models》，arXiv **2610.07922**（2026-10-06，Yu、Yuan、Zhang …… Fei-Fei、Wu、Adeli）。
- **代码**：`github.com/OpenWAM/OpenWAM`。PyPI 包为 `openwam`。许可 **AGPL-3.0-only**（`LICENSE`、README "Contributing and license" 节）。
- **生成机制**：潜空间 flow matching（项目页："All programs use latent flow matching"）。双专家结构：5B 视频专家（Wan2.2-5B / LingBot-VA 初始化）加 2B 动作专家。FDM 配置：`configs/experiments/dual_expert_libero_conditional_dynamics.yaml:108`（`lingbot_va_base`）、`:122`（`preset: fastwam_joint`）、`:123`（`program: forward_dynamics`）。

#### 闸门

1. **可重置生成：PASS。** `docs/architecture.md:110`："Conditional samples begin with a clean t0 video frame in a singleton chunk"。仿真反事实脚本先用 `env.set_init_state(init_state)` 复位到指定的 LIBERO 状态，再取 t0 观测：`scripts/research_dynamics/counterfactual.py:406-407`。
2. **联合一致性：PASS（纯 FDM，画面以动作为条件）。** `docs/policy_architectures.md:44`："`forward_dynamics`: clean actions condition video prediction; only video loss"。配置路由为 `action_conditioned_video`（`dual_expert_libero_conditional_dynamics.yaml:95-97`）。
   - 动作必须外部提供。`docs/running_experiments.md:581-583` 写明 FDM 推理只能离线进行，LIBERO 实时仿真 runner 会拒绝这个 program。
   - **动作来源**：可以是 OpenWAM 自己的联合策略（LIBERO 检查点未发布），也可以是 LeRobot 中已有的 LIBERO 策略（例如 pi0 / pi05 的 LIBERO 检查点）。具体选哪个，待定。
3. **帧可解码：PASS。** `scripts/research_dynamics/visualization.py:11-37`（`decode_latent_video` → `vae.decode`）；`counterfactual.py:580-587` 写明"FDM eval requires a VAE to decode predicted latents"。

#### 其他记录

- **LIBERO 检查点：没有。**
  - README 第 71 行："Policy checkpoints and OpenWAM datasets are being prepared for release"。
  - 路线图中 `[ ] LIBERO policy checkpoints`（第 79 行）、`[ ] Inverse dynamics (IDM) checkpoints`（第 81 行）都未完成。**FDM 检查点不在路线图上。**
  - `configs/artifacts.sample.yaml` 里 LIBERO 条目为 `download_url: null`。
  - HF 组织 `OpenWAM-Stanford` 下只有 `OpenWAM-Pretraining` 一个模型（`model_state.pt` 20.4 GB，纯视频的块因果预训练，没有动作专家，card 没有 license 字段）。
- **自训代价**：要先在仿真中生成反事实数据（`scripts/build_libero_fdm_counterfactual_demo_dataset.py`，训练集 10k 行、验证集 1k 行），再编码潜变量，然后训练（配置 `num_steps: 20000`，参考配置使用 4×48 GB GPU）。
- **分辨率与块**：每个视角 128²，双视角横拼成 128×256（配置第 32-33 行）。`action_horizon: 16`，`frame_chunk_size: 4`，即每个潜帧对应 4 个环境步。
- **到 520 步**：诊断脚本默认 `--horizon-frames 16`（`counterfactual.py:1022`），即 16 个潜帧，约 64 步。训练契约是分块因果，每块只看最近一个干净边界帧（`docs/running_experiments.md:576-578`），结构上可以分块续滚。滚到 520 步（约 130 个潜帧）没有现成脚本，也没有验证：unverified，属于训练外外推。

**小结**：三闸门在代码层面都成立，FDM 设计也最贴近"画面是动作的后果"。但 LIBERO FDM 检查点不存在，也不在发布计划上。要实测只能自训，而且需要另配一个动作来源策略。**待定**：作者发布 FDM 检查点，或我们决定投入自训时，再重开。

### 2.2（附）OpenWAM-α（OpenWAM-Official，同名的另一项目）

- **论文**：《OpenWAM: An Open, Modular Exploration Towards Systematic World-Action Model Pretraining》，arXiv **2609.07398**（Wang、Huang …… Zhao）。
- **代码**：`github.com/OpenWAM-Official/OpenWAM`，Apache-2.0（`LICENSE`）。HF 组织为 `OpenWAM`。
- **生成机制**：Wan2.2-TI2V-5B 视频 DiT 加独立的 ActionDiT（dim 1024），按层做联合自注意力（`dual_system / joint_self_attn`），潜空间 flow matching 联合去噪。

#### 闸门

1. **可重置生成：PASS。** `openwam/model/architectures/base.py:1367` 的 `generate(..., first_frame_image=...)`；部署端把观测填入 `conditions["first_frame_image"]`（`openwam/deploy/policy.py:97`）。LIBERO 客户端会把图像旋转 180°（`benchmarks/libero/openwam2libero_interface.py:181`）。
2. **联合一致性：PASS（双向）。** 检查点 `config.yaml:65` 设置 `attention_mask_mode: mutual`。掩码实现见 `openwam/model/architectures/utils/mask_modes.py:76-86`：动作看全部视频，视频（首帧除外）看动作。返回值同时包含 `video` 和 `actions`（`base.py:1595-1605`）。
3. **帧可解码：PASS。** `base.py:1593-1595` 中 `decode_video=True` 时调用 `vb.decode_video(...)`，返回 PIL 帧列表。部署配置默认 `decode_video: false`（`configs/deploy.yaml:26`），只返回动作，实测时要打开。

#### 其他记录

- **LIBERO 检查点**：`OpenWAM/OpenWAM-Alpha-Sim-LIBERO`，公开、未 gated，card 许可 apache-2.0。`checkpoint_step_10690.safetensors` 为 24.8 GB。另需 Wan2.2-TI2V-5B 资产（VAE 与 T5 文本编码器；下载脚本标注约 34.2 GB，见 `scripts/download_assets/download_video_backbone.py:87`）。
- **LIBERO-10 覆盖**：单个检查点在 `OpenWAM/LIBERO` 上微调。该数据集 card 写明 "1693 episodes … 40 tasks"，由 `lerobot/libero` 转换而来，即 Spatial、Object、Goal、10 四个 suite 合训，覆盖 LIBERO-10 全部任务。评测协议中 `libero_10` 的 `max_steps` 为 700（`benchmarks/libero/policy_config.yml:15`）。
- **模型规模**：视频部分 Wan2.2-TI2V-5B（`config.yaml:20-27`：dim 3072、30 层），动作 DiT dim 1024。总参数量：unverified。
- **单次调用视野**：`num_frames: 33`，`video_stride: 4`（`configs/dataloader/libero.yaml`），即每次输出 32 个动作和 9 帧（(33−1)/4+1），覆盖 32 个环境步。画布为 384×320，三槽 L 形拼图（第三人称 + wrist）。
- **到 520 步**：需要约 17 次调用，并把上一段的末帧作为下一段的 `first_frame_image`。这种链式想象代码里没有提供：仿真评测在每个块之后都使用真实观测。所以这属于训练外外推，按 `eval_design.md` §9 记为"训练外外推"，并记录逐块漂移。

**小结**：这不是 FDM，而是视频与动作双向注意力的联合 WAM。三闸门都有代码证据，单个公开检查点覆盖 LIBERO-10。**可进入实测**；单次调用只覆盖 32 步，所以按"前 k 步想象"或链式外推的口径记录。

---

## 3. dWorldEval

- **论文**：《dWorldEval: Scalable Robotic Policy Evaluation via Discrete Diffusion World Model》，arXiv **2604.22152**（2026-04-24，只有 v1；Li、Zhou、Chen、Xue、Zhu；Current Robotics / 多伦多大学）。ICML 2026 列表中的题名为"Scaling Real-World Robot Policy Evaluation via Discrete Diffusion World Model"。arXiv 页面的许可为 CC BY-NC-ND 4.0（针对论文本身）。
- **代码与权重：都没有。**
  - arXiv 页面没有代码链接。
  - 项目主页仓库 `dworldeval/dworldeval.github.io` 只链接 arXiv（`index.html:378,387`），没有 code、checkpoint 或"coming soon"字样。
  - GitHub 搜索 "dWorldEval" 只返回这个主页仓库；HF 上搜索 dWorldEval，模型和数据集都为 0。
- **生成机制**（仅论文）：离散扩散（掩码式并行解码，16 步，§4.1）。视觉用 MAGVIT-v2 token，动作用 FAST token，语言用 LLaDA tokenizer，全部串成一个序列（§3.2.1）。从 MMaDA-VLA-8B 初始化，双向 Transformer，32 层，hidden 4096（附录 A.2）。引言又说"trained from scratch on robotic data"，与附录说法矛盾。

### 闸门（全部只有论文证据）

1. **可重置生成：PASS（仅论文）。** §3.1：模型以"current observation o_t, a history h_t, and a sequence of future actions"加语言指令为输入。
2. **联合一致性：PASS（仅论文）。** §3.2.1："each visual token directly attends to action tokens"。动作由被评测的策略提供（LIBERO 实验用的是 π0，见 §4.1 / 图 6），属于 FDM 式条件化。
3. **帧可解码：unverified。** 论文没有明确描述像素解码。输出是 256² 多视角（§4.1），MAGVIT-v2 本身带解码器，图中也有画面，但没有代码可以核对。

### 其他记录

- **LIBERO**：§4.1 写的是"LIBERO-Object, LIBERO-Spatial, LIBERO-Goal, and LIBERO-100 suites"，使用第三人称和 wrist 同步视角。LIBERO-100 名义上包含 LIBERO-10，但训练和评测是否具体覆盖 LIBERO-10 的任务：unverified。
- **到 520 步**：
  - 论文没有给出最大 rollout 长度，也没有说明如何拼接。
  - 每帧对应的动作块长度 Δ 在训练时从 [2,8] 中采样（§4.1、附录 A.2），推理时的取值：unverified。
  - 附录 A.2 给出单条轨迹 30–90 秒、1.5 秒/帧（H800），即约 20–60 帧。
  - 能否覆盖 520 步：unverified。

**小结**：从论文描述看，三闸门大体会通过，"动作可控性"也正是它的卖点。但没有代码、没有权重，无从实测。**关闭**（原因：不可获得）。如果作者发布，再重开。

---

## 4. 为什么最初没进入四个候选

**事实原因**：最初的候选取自 LeRobot 仓库内的"World Models"列表。`lerobot/README.md:109` 列出的是 VLA-JEPA、LingBot-VA、FastWAM、LaWAM、FLUX 3 Action，前四个有模型档案（`model_cards/`）。上面三个候选都**没有集成进 LeRobot**：本地 `lerobot/src`、`lerobot/docs` 中检索 `worldvla|rynnvla|openwam|dworldeval` 没有结果，`lerobot/src/lerobot/policies/` 下也没有对应目录。所以它们不在最初的取样范围内。

逐个说明是"不可获得"还是"只是没看"：

| 候选 | 当时为什么没进入 | 归类 |
|---|---|---|
| WorldVLA / RynnVLA-002 | 不在 LeRobot 列表中。检查点 2025-06（WorldVLA）和 2025-11（RynnVLA-002）就已公开，四个 suite 含 LIBERO-10 都有 | **只是没看** |
| OpenWAM FDM（Stanford） | 不在 LeRobot 列表中。代码已公开（arXiv 2026-10-06），但 LIBERO FDM 检查点不存在、也不在路线图上 | **不可获得**（无公开 LIBERO 检查点，代码可用） |
| OpenWAM-α（同名项目，附） | 不在 LeRobot 列表中。LIBERO 检查点 2026-09-06 已在 HF 公开 | **只是没看** |
| dWorldEval | 不在 LeRobot 列表中。没有代码、没有权重 | **不可获得**（无代码、无检查点） |

另外值得一提：Stanford OpenWAM 的 FDM 配置直接以 LingBot-VA 底座初始化，并沿用 FastWAM 的联合预设（`dual_expert_libero_conditional_dynamics.yaml:108,122`）。它和已在实测中的 LingBot-VA、已关闭的 FastWAM 同属一个技术谱系。
