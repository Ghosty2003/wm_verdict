# 模型卡：WorldVLA（`Alibaba-DAMO-Academy/WorldVLA`，`model_256/libero_10`）— LIBERO 世界模型评估用

> **口径（2026-10-14）**：本阶段只在虚拟环境内评测。文中的“真实”“现实”“real”一律指 **LIBERO 模拟器真值分支**，“迁移”指想象结论到模拟器执行的迁移。物理机器人不在范围内。论文中的“真实世界 / real-world”实验，指的是论文作者的物理机器人实验，与本项目的真值分支无关。补充一点：本卡核对的论文 v1 正文里**没有**物理机器人实验；仓库 README 把 “Release the code of real-world expriment” 列为未完成项（[REPO]/README.md:234）。

> 用途：LIBERO 上世界模型判定评估的第一步，只记录事实。每一条都注明出处；没有证据的写 `unverified`；来源互相矛盾时把双方都列出。编写过程中没有加载任何模型权重，也没有使用 GPU。参数量只读取了 safetensors 文件头（JSON header），没有读取张量数据。
> 编写日期：2026-10-15。

### 出处缩写

| 缩写 | 指向 |
|---|---|
| `[WV]/x.py:N` | 本地仓库副本 `$WMV_ROOT/libero/worldvla/third_party/WorldVLA_worldvla_branch/worldvla/x.py` 第 N 行（`alibaba-damo-academy/WorldVLA` 的 `worldvla` 分支 @ `ebdc36d`，即与 HF 权重配套的原 WorldVLA 代码；`main` 分支已换成 RynnVLA-002） |
| `[REPO]/x:N` | 同一副本的仓库根目录（`README.md`、`xllmx/…`） |
| `[HF]/x` | HF 快照 `models--Alibaba-DAMO-Academy--WorldVLA/snapshots/5952d61278d5…/x`：`README.md`、`model_256/libero_10/{args.json, config.json, generation_config.json, model.safetensors.index.json}`、三个 safetensors 分片的文件头、`chameleon/starting_point/config.json`、`chameleon/tokenizer/vqgan.yaml` |
| `[PAPER §x]` | Cen et al., *WorldVLA: Towards Autoregressive Action World Model*，arXiv:2506.21539**v1**（2025-06-26；2026-10-15 取得的 arXiv 页面上只有 v1） |
| `[LIBERO]/x:N` | conda 环境 `worldvla` 中安装的 hf-libero 0.1.4：`site-packages/libero/libero/x` |
| `[UNIT]:N` | 本项目 `$WMV_ROOT/libero/worldvla/worldvla_unit.py` 第 N 行 |
| `[PRE §x]` / `[ENV]` / `[SCREEN]` | 本项目笔记：`worldvla/precheck_plan.md`、`worldvla/env_setup.md`、`candidate_screening.md` |

范围：只覆盖 `model_256/libero_10`。同一 HF 仓库中的 `model_256/{libero_spatial, libero_object, libero_goal}` 和 `model_512/*`（[HF]/README.md:86-101）不在本卡范围内。`[SCREEN]` 中提到的 “`_512` 文件夹里 `resolution: 256`” 是 RynnVLA-002 的问题，与本 checkpoint 无关（[PRE §5 第 5 条]）。

---

## A. 结构与骨干

| 字段 | 内容 | 出处 |
|---|---|---|
| 骨干来源 | **Chameleon-7B**，统一的离散 token 自回归多模态 LLM，经 Lumina-mGPT 的格式转换。**训练链路**：`Alpha-VLLM/Chameleon_7B_mGPT` → 制作 starting point（代码把 `lm_head` 中图像 token 那几行清零）→ 在 LIBERO-10 上用“动作模型数据 + 世界模型数据”混合微调，得到 `model_256/libero_10`。**没有大规模机器人数据预训练**：论文 Table 2 中 WorldVLA 的 “Pretraining” 一栏是 ✗。HF 卡 front-matter 写 `base_model: facebook/chameleon-7b`。**冻结与可训练部分**：VQGAN 图像 tokenizer 冻结，训练前离线预先 token 化，训练时 `del model.model.vqmodel`；transformer 全部参数可训练（代码中没有 LoRA 或冻结层的设置）。 | [PAPER §3.2]；[PAPER Table 2]；[HF]/README.md:3-4；[HF]/chameleon/starting_point/config.json `_name_or_path: "Alpha-VLLM/Chameleon_7B_mGPT"`；[HF]/model_256/libero_10/config.json `_name_or_path: "lumina_mgpt/output/starting_point"`；[WV]/pretrain_solver_awm_w_ck.py:81, 88-108；[REPO]/xllmx/solvers/pretrain/pretrain_ck.py:263-286；[REPO]/README.md:107-108, 269 |
| 生成机制 | **离散 token 的 next-token 自回归**（因果 Transformer，交叉熵训练）。它不是扩散，也不是 flow matching。文本、图像（VQGAN code）、动作（256 个分箱）共用一个 65,536 大小的词表，共用同一套权重。**由提示词决定工作模式**：动作模型模式是 “What action should the robot take to {task}?” + 图像 → 动作 token；世界模型模式是 “Generate the next … ” + 图像 + 动作 → 下一帧的图像 token。**解码**：官方动作评测和仓库中的世界模型辅助函数都用 **greedy 解码**（`do_sample=False`，`top_k=None`，`temperature=1`，eos=8710）。仓库里只有一段被注释掉的采样配置（top_k=50，`do_sample=True`）。**因此，在官方用法下模型是确定性的**：同一个输入总得到同一个输出，只差 GPU 数值上的不确定性。**本项目的协议决定**：主判定使用 greedy 解码，每个锚点只想象一次；`--sample_wm` 只作为附加选项。 | [PAPER §1, §3.2, Fig.2]；[HF]/model_256/libero_10/config.json `vocab_size: 65536`；[WV]/libero_util/Chameleon_utils.py:548-554（VLA，greedy）、:615-621（WM 辅助函数，greedy）、:622-628（被注释的采样配置）；[UNIT]:231-233, 651；[PRE §5 第 4 条] |
| **假设声明** | **官方世界模型评测代码未发布（README 第 233 行附近）；本项目世界模型模式的提示词格式是按训练数据格式推断的，属于假设；官方策略评测对照跑通后，这一假设的风险有界。** 依据如下。①README:233 TODO：“[ ] Release the code of world model on LIBERO dataset”。②**训练数据格式**：human 回合是 `"Generate the next image based on the provided sequence of historical images and corresponding actions." + "<|image|><|action|>"`，gpt 回合是 `"<|image|>"`；两个回合都以分隔符 `<reserved08706>`（id 8710）结尾。③**仓库里的辅助函数措辞不同**：`get_action_Chameleon_dis_awm_g_video(his_type="1a2i")` 用的是 `"Generate the image based on the current image and the action."`。④**论文给出的是第三种措辞**：“Generate the next frame based on the current image and the action.”。本项目默认用②（`--wm_prompt train`），`onestep` 步骤会对②和③都打分。“官方策略评测跑通”指的是 `official-eval`，它验证的是同一份权重、同一套 tokenizer 和环境链路，而不是 WM 提示词本身。 | [REPO]/README.md:156-176, 188-193, 233；[WV]/data/world_model_conv_generation.py:118-128；[WV]/data/convertsation.py:5-20；[WV]/libero_util/Chameleon_utils.py:598-612；[PAPER §3.3 World Model Data, Fig.2]；[UNIT]:66-67, 650；[PRE §5 第 1 条] |
| 参数规模 | **Transformer 合计 7,013,158,912 个参数**，按分片头逐项相加；全部为 BF16，与 index 中 `total_size` 14,026,317,824 B ÷ 2 一致。**分模块**：`embed_tokens` 268,435,456（65536×4096）；32 个 decoder 层合计 6,476,283,904，每层 202,383,872，其中 attention q/k/v/o 为 67,108,864，SwiGLU MLP（4096→11008）为 135,266,304，RMSNorm 和 q_norm/k_norm 约 8.7K；`lm_head` 268,435,456（不与 embedding 共享权重，`tie_word_embeddings: false`）；最终 norm 4,096。**没有独立的动作头或世界模型头**：两种模式共用同一套参数。**VQGAN 图像 tokenizer**：`vqgan.ckpt` 文件大小 281,270,377 B，参数量 `unverified`（没有加载）。**starting point** 的 `total_size` 为 14,085,165,056 B，比微调后的权重多约 58.8MB；这与训练时删除 `vqmodel` 相符，但这只是推断。 | [HF]/model_256/libero_10/model.safetensors.index.json `metadata.total_size`；三个分片文件头的 shape/dtype；[HF]/model_256/libero_10/config.json `hidden_size 4096, intermediate_size 11008, num_hidden_layers 32, num_attention_heads 32, tie_word_embeddings false`；[HF]/chameleon/starting_point/model.safetensors.index.json；[ENV]（文件大小）；[WV]/pretrain_solver_awm_w_ck.py:81 |
| 表示与分辨率 | **图像**：Chameleon VQGAN（taming `VQModel`），codebook 8192，embed_dim 256，`ch_mult` 有 5 级，即下采样 16 倍。256×256 输入得到 16×16=256 个 code。**每帧 token 布局**：`<racm3:break>`(8197) + 高度格数 token + 宽度格数 token（都是 8812，即 8 格×32px）+ 16 行×(16 个 code + 换行 8803) + `<eoss>`(8196) = **276 token/帧**（CPU 上实测）。`var_center_crop` 对 256² 输入不起作用。**这个 checkpoint 的分辨率**：args.json 里没有分辨率字段；256 是从目录名、`eval_solver_libero_256.py` 的 `ItemProcessor(target_size=256)` 以及 8×8 格的 header token 推断出来的。VQGAN 本身按 512 训练（yaml `resolution: 512`）；论文说 Chameleon 的 tokenizer 和 LLM 都“inherently optimized at 512”。**VQ 往返损失**：真实锚点帧经 VQGAN 编码再解码，PSNR 为 23.97 dB（本项目在 CPU 上实测）。 | [PAPER §3.2, §4.2 Benchmark Results]；[HF]/chameleon/tokenizer/vqgan.yaml:5-10, 14-19；[WV]/data/item_processor.py:200-205, 227-228, 252-300, 368-391；[WV]/eval_solver_libero_256.py:140；[ENV] CPU-only checks；[PRE §1 image tokens, §5 第 5 条] |
| 动作接入方式 | **动作是输入 token 序列的一部分**，不经过单独的嵌入网络。7 维动作的处理：先做 min-max 归一化 `2(a−min)/(max−min+1e-8)−1` 并 clip 到 [−1,1]，再用 `np.digitize` 落入 256 个均匀分箱，token id = 分箱号 + 10005；前后加 `<reserved10000>`(10004) 和 `<reserved15000>`(15004)，所以每个动作 9 个 token。min/max 是写死的常数，所有 suite 共用：xyz 为 ±0.9375，roll 为 [−0.32571429, 0.375]，pitch/yaw 为 ±0.375，夹爪为 [−1,1]。**冲突**：论文说 “bin widths determined by the range of the training data”，代码则是固定常数。语义上是 3 个相对位移、3 个相对角度和 1 个绝对夹爪状态（论文）。**WM 模式输入的是原始环境单位的动作 a_t**，即 `.npy` 中的原始演示动作，由 `process_action` 编码；不是分箱中心值。VLA 输出的是分箱中心值，经 `unnorm_min_max` 反归一化后，再编码回去得到完全相同的 9 个 token；往返误差 ≤0.004（环境单位）。**注意力掩码**：`generate_att_mask_3` 只对“最后一个图像块之后的动作块”屏蔽先前的动作块，这是论文 Fig.3(b) 的动作块掩码；WM 序列中动作在目标图像之前，所以退化为普通因果掩码，与论文 Fig.3(c) 一致。 | [PAPER §3.2, §3.3 Attention Mask, Fig.3]；[WV]/data/item_processor.py:207-208, 246-250, 302-334；[WV]/model/modeling_xllmx_chameleon_ck.py:216-299, 403-408；[WV]/eval_solver_libero_256.py:125-133；[WV]/data/world_model_conv_generation.py:114, 134；[ENV] Action codec；[PRE §1 action norm, WM mode] |
| 上下文与预测窗口 | **WM 模式是 Markov 的**：输入**一帧上一时刻的图像 + 一个原始环境单位动作**，输出**下一帧**。训练数据由 `world_model_conv_generation.py --his 1` 生成：`img_c_h = img_list[j:j+1]`，`action_c = action_list[j:j+1]`，目标 `img_list[j+1]`。论文的 Training Setting 写明 “the world model operates with a single round N = 1”。式 (2) 写成一般形式 o_t = f(o_{t−h:t−1}, a_{t−h:t−1})，但实际配置是 h=1。WM 提示词约 304 token，生成 276+1 个 token。**VLA 模式**：输入 M=2 帧（上一帧和当前帧，`2h_1a_img_only`）加任务文本，输出 K=10 个动作（LIBERO-Long 的设置）。训练时 `max_seq_len` 为 4096，评测脚本传入 8192。**能否当动作条件模拟器**：**能，但要自己写循环。** 每次调用生成 1 步 1 帧；把生成的帧喂回去，就可以滚动任意步数（520 步需要 520 次调用）。仓库里没有官方的多步 WM 滚动代码，见“假设声明”。RynnVLA-002 `main` 分支的参考实现把生成的**像素**重新编码后再喂回。**Markov 的后果**：a_t 只能通过 o_{t+1} 影响 o_{t+2} 及之后的帧；模型对被遮挡的物体或超过一步的历史没有记忆。 | [WV]/data/world_model_conv_generation.py:109-134；[REPO]/README.md:188-193；[PAPER §3.1 Eq.2-3, §4.1 Training Setting]；[WV]/data/action_model_conv_generation.py:108-133；[WV]/libero_util/Chameleon_utils.py:499-509；[HF]/model_256/libero_10/args.json:35 `max_seq_len 4096`；[WV] exps_256_all/eval_libero_7B_2_action_all_epochs_img_only_ck_5_1a2i_10.sh:45；[ENV] Prompt lengths；[PRE §1 WM eval loop, §5 第 7 条] |
| 自带头 | 只有一个 `lm_head`（4096→65536），**文本、图像、动作 token 共用**。**策略**：VLA 提示词下由同一个 `lm_head` 输出动作 token。**没有 reward 头，也没有 value 头**：代码和论文都没有。成功判定完全来自环境的 `done`。 | [HF] 分片文件头 `lm_head.weight [65536,4096]`；[WV]/eval_solver_libero_256.py:282-290；[PAPER §3] |
| 训练目标 | **加权交叉熵（next-token）**。论文写作 L = L_action + α·L_world，α=0.04；只对动作模型数据中的动作 token 和世界模型数据中的生成图像 token 计算损失（prompt 部分的 label 为 −100）。**代码实现**：`CrossEntropyLoss(weight=loss_weights)`，其中 `loss_weights[3:8195]=0.04`（图像 code 的 token id 区间），其余 token 权重为 1；此外加 z-loss，权重 1e-5。这是“按 token 类别加权的单一加权平均 CE”，与论文分开写的两项损失在归一化上不完全相同。**训练数据混合**：动作模型对话与世界模型对话直接拼接（`concate_action_world_model_data.py`）。**超参（args.json）**：bf16、FSDP、`batch_size 8` × `world_size 16` × `accum_iter 1`，按代码计算有效 batch 为 128；`epochs 40`；`lr 1e-6`（`min_lr` 也是 1e-6，即恒定学习率）；`wd 0.1`；`warmup_epochs 0.01`；`clip_grad 4`；`dropout 0.05`；`mask_image_logits false`。**冲突**：仓库中公开的训练脚本是 `lr=5e-6`、单机 4 卡，`exp_name` 含 `lr5e6`；args.json 的 `output_dir` 含 `lr1e6`。所以发布的权重不是用公开脚本原样训练出来的。发布的是第几个 epoch：`unverified`。 | [PAPER §3.3 Eq.4, §4.1 Training Setting]；[REPO]/xllmx/solvers/pretrain/pretrain_ck.py:442, 913-933, 960；[WV]/model/chameleon/modeling_chameleon.py:1585-1600；[WV]/model/modeling_xllmx_chameleon_ck.py:72-81；[WV]/data/item_processor.py:340-356；[HF]/model_256/libero_10/args.json:2-9, 20, 28-38；[WV]/exps_256_all/7B_ts_his_2_img_only_10_ck_10_1a2i_all.sh:3, 12, 17 |
| 能否解码出画面 | **能。** 生成的 token 去掉末尾的 8710 后，交给 `item_processor.decode_image`：去掉起止 token，读出格数，把 BPE id 映射回 VQ code，再用 `pil_from_img_toks` 走 VQGAN 解码，得到 256×256 RGB 的 PIL 图像。画面方向是**训练视角，即原始 agentview 旋转 180°**。**风险**：greedy 解码对网格结构没有约束；token 数不对时，`decode_image` 的 `assert` 会失败。本项目在这种情况下用带结构约束的 greedy 重试一次（仓库中没有这一步，已记录为偏差，计数为 `n_struct_fallback`）。 | [WV]/libero_util/Chameleon_utils.py:631-634；[WV]/data/item_processor.py:368-391；[WV]/libero_util/libero_utils.py:66；[UNIT]:302, 372；[PRE §5 第 3 条] |

**输入（两种模式共同的事实）**：
- **只用 agentview 一路相机**，256×256，**旋转 180°**（`img[::-1, ::-1]`）。训练数据和评测时都这样处理。
- **没有腕部相机，也没有本体感觉（proprio）输入。**
  - 训练图像只读取 `agentview_rgb`。过滤脚本虽然把 `eye_in_hand_rgb` 存进了 HDF5，但后续步骤没有读取。
  - 评测循环中构造了 `state`，但没有传给模型。注释原文是 “OpenVLA does not take proprio state as input”。
  - 出处：[WV]/libero_util/libero_utils.py:18-25, 60-68；[WV]/libero_util/regenerate_libero_dataset_save_img_action.py:94, 107；[WV]/libero_util/regenerate_libero_dataset_filter_no_op.py:168-169, 189-190；[WV]/eval_solver_libero_256.py:188, 234-248；[WV]/libero_util/Chameleon_utils.py:499-509；[WV]/data/item_processor.py:259-300。
- **WM 模式不带任务文本**。论文原话是 “It does not need the task instruction since the action itself could totally determine the next state”（[PAPER §3.3 World Model Data]；[WV]/data/world_model_conv_generation.py:123）。

### A 补充：提示词选择与场景前缀（2026-10-15 规则与实测）

**世界模型提示词：按实测选择，不靠默认值**（eval_design §16）
- 三种写法全部参加 onestep 抽查：
  - `train` = “Generate the next image based on the provided sequence of historical images and corresponding actions.”
  - `helper` = “Generate the image based on the current image and the action.”
  - `paper` = “Generate the next frame based on the current image and the action.”

  三者共用同一排布：文本，接 `<|image|><|action|>`，再接分隔符。论文写法可以复现：§3.3 给出了完整句子和与训练数据相同的 token 排布，见 [PAPER §3.3 World Model Data] 中 :282-:300 的公式。
- **帧来源**：官方策略评测（任务 #5）闭环时逐步记录的真实转移。这批帧不经过任何世界模型提示词，对三种写法中立。固定种子 20261015 抽 60 对；三种写法用同一批帧对；指标为一步预测 PSNR。
- **选择结果（2026-10-15 实测）：`helper`**，已冻结到 `protocol.yaml` 的 `libero.worldvla.wm_prompt`。
- **三种写法的数值**：60 对一步预测，均值 PSNR（dB）。

  | 写法 | 均值 PSNR | 结构回退 | 相对 helper 的配对差（95% 区间）| helper 胜 / 负 |
  |---|---|---|---|---|
  | `helper` | 24.266 | 0/60 | — | — |
  | `paper` | 24.246 | 0/60 | −0.020（−0.064 至 +0.017）| 16 / 10，其余 34 对完全相同 |
  | `train` | 24.154 | **56/60** | −0.112（−0.199 至 −0.033）| 32 / 27 |
  | 参照：复制上一帧 | 31.31 | — | — | — |
  | 参照：VQ 往返上限 | 24.80 | — | — | — |

  原始记录：`out/worldvla/prompt_select/onestep_official_task5.json`（sha256 6eee7655…）。
- **选择理由**：
  - `helper` 的均值最高。
  - `helper` 与 `paper` 实际上打平：配对差的区间含 0，多数帧对生成的 token 完全相同。
  - `train` 在 56/60 次调用中没有正常收尾：图像末行之后不输出 `<eoss>`(8196)，而是接着生成动作 token。仓库的掩码构造函数因此报错，只能靠约束重试得到画面。这说明发布的权重很可能不是用该措辞训练的。
- **读数注意**：一步预测的 PSNR 受 VQ 往返上限（24.8 dB）约束，所以低于"复制上一帧"（31.3 dB，未经 VQ）是 tokenizer 造成的，不等于动力学无效。
- 选定写法冻结在 `protocol.yaml` 的 `libero.worldvla.wm_prompt`，之后所有 WorldVLA 想象只用这一种写法。

**策略指令的场景前缀**（见 B.5 “语言改写”行）
- 状态为 unverified：训练数据的指令可能带场景前缀（取自 `task.name`），评测时不带（取自 `task.language`）。
- **应对**：
  - 官方对照的成功率若与论文值差距大，判据是 10 回合 Wilson 95% 区间不含 0.59，就先跑带前缀的变体（`official-eval --instr name`），不直接判定我们的循环有错。
  - 两个变体都报告。
  - 只有两个变体都对不上论文值，才算对照失败，此时停下报告。
- **官方对照实测（2026-10-15，任务 #5，各 10 回合，0 个回合因异常中止）**：
  - 不带前缀（`task.language`，官方评测写法）：10/10，Wilson 95% 区间 [0.722, 1.000]。
  - 带前缀（`task.name`）：9/10，区间 [0.596, 0.982]。
  - 两个变体的区间都不含 0.59，但都在 0.59 **之上**。0.59 是 10 个任务的均值，单任务值没有公布。按事先写定的双侧判据，两个变体都"对不上"；按"循环有错"的本意（成功率偏低），两个都不构成证据。此事已报告，待裁决。
- **十任务官方对照（2026-10-15，裁决 3：每任务 10 回合，判读用单侧判据加偏离带）**：
  - 不带前缀（`task.language`）：套件均值 0.52（52/100，Wilson 95% 区间 [0.42, 0.62]）。各任务依次为 0.1 / 0.5 / 0.4 / 0.5 / 0.5 / 1.0 / 0.7 / 0.5 / 0.3 / 0.7。
  - 带前缀（`task.name`）：套件均值 0.53（53/100，区间 [0.43, 0.63]）。各任务依次为 0.2 / 0.7 / 0.6 / 0.5 / 0.4 / 0.9 / 0.7 / 0.6 / 0.2 / 0.5。
  - 两个变体都没有明显低于论文值 0.59，对照通过。按规则，以更接近论文值的**带前缀变体**作为官方对照。两者只差 1 个回合，并不能说明训练数据确实带了前缀，所以"带前缀"一项仍标 unverified。
  - 两个变体在所有回合中都没有出现因异常而中止的情况。
- **掩码重置等价性（eval_design §15）：通过。** 从 `libero10_t5_a0` 的首帧起跑 5 步，循环内（每次调用前重置）的输出与官方单次调用路径（`helper` 写法，新模型状态）逐 token 完全一致，每帧 277 个 token。补充发现：官方 helper 连续调用、不做重置时也完全一致。原因是 WM 序列中动作位于目标图像之前，仓库的动作块掩码不生效，掩码退化为普通因果掩码。因此重置对 WM 调用无害但并非必需；它只可能影响 VLA 调用。
- **策略模式畸形率对照（裁决 4）**：
  - 真实帧：同一批 10 个锚的 init state，按官方闭环逐步执行，共 381 次 VLA 调用，**畸形 0 次**。
  - 想象帧：目前只有 30 步冒烟测试的数据，3 次调用中畸形 1 次（第 20 步，在一个完整动作块之后接连输出结束 token）。正式数字待 10 个 WorldVLA unit 跑完后填入。
  - 与 `train` 写法 56/60 畸形属于同一类证据：模型在自己生成的画面或非训练格式的输入上，输出结构会退化。
- **受约束重试口径（表格脚注必须写明）**：VLA 与 WM 的输出畸形时，在官方提示词构造不变的前提下，用受约束的贪心解码重试：VLA 用 `ActionStructure`，WM 用 `ImageStructure`。同一步连续 5 次重试仍畸形，该 unit 标为 generation-invalid。每个 unit 记录调用次数、畸形次数、重试次数和首次畸形步数。重试属于对模型的干预。
- **对想象分支的影响**：世界模型的画面生成不受影响，因为世界模型提示词不含任务文本。交替循环里的动作由策略模式给出，策略模式带任务文本；若最终选用带前缀的指令，策略调用同步改用，并一起冻结。

### A 补充：E4 结论（2026-10-15 裁决）

- **WorldVLA 退出主 verdict 表**，记为排除类别 **E4：自有策略在自身想象帧上长程崩溃**（行为性排除，见 `model_exclusions.md`）。
- **证据**：
  - 官方十任务对照通过（0.53，带前缀）；
  - 真实帧上 VLA 畸形 0/381；
  - 想象帧上 VLA 畸形 47/52，从第 50 步起每次都畸形（首个基线 unit）；
  - 10 个基线 unit 全部跑完（2026-10-21）：t5 的 5 个全部崩溃；t3 的 5 个中 2 个崩溃，3 个（a0、a3、a4）全程无畸形，作为"干净子集"单列。排除理由按 2026-10-21 裁决改写为：10 个 unit 中 7 个的 verdict 由约束重试主导，主表要求 verdict 是模型的自由输出。明细与按步归一的畸形占比见 `model_exclusions.md` E4 节；
  - 世界模型调用畸形 46/520（9%）；
  - `train` 措辞下世界模型 56/60 无法正常收尾，属于同一类证据。
- **在受约束重试口径下**，从第 50 步起几乎所有动作块都来自重试，表里测的是"模型加我们的修补"。这一口径下的 verdict 只进附录，每个 unit 标注自由输出占比，表头写明口径。
- **族 A 批次全部取消。** 10 个基线 unit 的数据用于：畸形起始与占比曲线、想象帧画质随步数的退化（与畸形起始对齐）、附录 verdict。
- **WorldVLA 视角判据**：重渲染标定若在 4 h GPU 以内就做，使附录 verdict 可解释；否则附录 verdict 列留 pending 并注明原因。

---

## B. 训练任务与数据

### B.1 本 checkpoint 覆盖的 LIBERO suite 和任务
- **只有 LIBERO-10（LIBERO-Long）**，共 10 个任务。依据：
  - 目录名为 `model_256/libero_10`（[HF]/README.md:93）。
  - args.json 的数据配置名为 `his_2_train_10_img_only_ck_10_1a2i_all.yaml`（[HF]/model_256/libero_10/args.json:12）。
  - 官方评测脚本使用 `--task_suite_name libero_10`（[WV]/exps_256_all/eval_libero_7B_2_action_all_epochs_img_only_ck_5_1a2i_10.sh:24）。
  - 任务列表见 [LIBERO]/benchmark/libero_suite_task_map.py:38-49。
- 每个 suite 单独训练一个模型：4 个 suite 各有一个目录（[HF]/README.md:88-93）。
- 是否混入了其他 suite 或 LIBERO-90：**没有证据表明混入了**；数据流程按单个 suite 处理（[REPO]/README.md:112-200）。严格地说仍是 `unverified`。

### B.2 训练数据来源与规模
- **来源**：LIBERO 官方 HDF5 演示数据（[REPO]/README.md:114）。处理流程是按 OpenVLA 的做法在环境中**重放**演示，并做以下处理：
  - 以 **256px** 重新渲染（`camera_heights/widths=256`）；
  - 每条演示先用 dummy 动作 settle 10 步；
  - **去掉 no-op 动作**（除夹爪外的各维范数 < 1e-4，且夹爪与上一步相同）；
  - **只保留重放成功的轨迹**。
  - 出处：[WV]/libero_util/regenerate_libero_dataset_filter_no_op.py:1-10, 39-61, 116-120, 133-172, 174-197；[PAPER §4.1 Datasets]。
- **规模**：原始数据最多 10 任务 × 50 条 = 500 条（[WV]/libero_util/regenerate_libero_dataset_save_img_action.py:87）。过滤后实际保留多少条：`unverified`。
  - 仓库里附带的 `libero_10_metainfo.json` 只有 7 个任务键，共 336 条，其中标记成功的 265 条。它显然是一次不完整运行的产物，不能用来推算。
- **训练用的是全部数据还是 90% 训练划分：结论倾向“全部”，但 `unverified`。**
  - 论文 §4.1：默认是 90% 训练 / 10% 验证，“with the exception of Table 2, where all available data are utilized”。
  - README Model Zoo 中 LIBERO-Long (256) 的 59.0 与 Table 2 相同（[REPO]/README.md:94；[PAPER Table 2]）。
  - 公开脚本使用 `libero_256_all/his_2_all_…` 配置，“all” 是 train+val_ind+val_ood 的拼接（[WV]/exps_256_all/7B_ts_his_2_img_only_10_ck_10_1a2i_all.sh:8；[WV]/data/concate_action_world_model_data.py:84-87）。
  - 但 args.json 中的路径是 `configs/libero_ck_all/his_2_train_10_…_all.yaml`，同时带有 “train” 和 “all”，而且这个文件不在仓库里。
- **划分规则**（代码）：按排序后的任务列表，前 ceil(0.9×10)=9 个任务的前 90% 轨迹为 train，后 10% 为 val_ind；第 10 个任务整体为 val_ood（[WV]/data/world_model_conv_generation.py:60-61, 72, 137-144）。
- **样本构成**：
  - **动作模型样本**：每个时间步一条，输入 2 帧，输出 10 个动作；尾部不足 10 个动作的时间步丢弃（[WV]/data/action_model_conv_generation.py:108-133）。
  - **世界模型样本**：每个相邻帧对一条，即 (o_j, a_j) → o_{j+1}。因为 no-op 步是被跳过而**没有执行**的，保留下来的相邻帧确实是一步因果后继（[WV]/data/world_model_conv_generation.py:109-134；filter 脚本 :133-172）。
- **仓库流程与 libero_10 配置不一致**：`pretoken.py` 和 `concate_action_world_model_data.py` 把 `ck_5` 写死了，但 libero_10 的配置引用的是 `ck_10` 文件（[WV]/data/pretoken.py:14-15；[WV]/data/concate_action_world_model_data.py:45-49；[WV]/configs/libero_256_all/his_2_all_10_img_only_ck_10_1a2i.yaml:2）。因此用公开代码无法原样复现 libero_10 的数据。
- **反事实或额外数据**：所有来源都没有提到：`unverified`，倾向于没有。

### B.3 LIBERO 之外的预训练数据（含 Chameleon 基座）
- **没有机器人数据预训练**：Table 2 中 Pretraining 一栏为 ✗（[PAPER Table 2]）。
- **Chameleon-7B 基座**：由 Meta 在大规模图文混合数据上预训练（引用 [PAPER §3.2] 中的 “Team, 2024”）。具体的数据构成和规模本卡没有核实：`unverified`。
  - `Alpha-VLLM/Chameleon_7B_mGPT` 相对于 Meta 原始权重，除格式转换外有没有额外训练：`unverified`。
  - 代码在制作 starting point 时把 `lm_head` 的图像 token 行清零（[WV]/pretrain_solver_awm_w_ck.py:105-106）。基座原本是否具备图像生成能力：`unverified`。
- **VQGAN**：yaml 元数据中有 `dataset: coco`、`image_size: 512`，以及含 “shutterstock_laion” 的 checkpoint 路径字符串（[HF]/chameleon/tokenizer/vqgan.yaml:39, 47）。这些只是配置残留，不能当作训练数据的声明；它的实际训练数据：`unverified`。
- **LIBERO 是否出现在基座预训练数据中**：`unverified`，可能性低。

### B.4 观测配置、控制频率、训练序列长度（与本项目 H = 520 对比）
- **观测**：只有 agentview，256×256，旋转 180°；没有腕部相机，没有 proprio。见 A 节末的“输入”。
- **控制频率**：LIBERO 环境 `control_freq=20`，即 20Hz（[LIBERO]/envs/bddl_base_domain.py:64）。论文和仓库都没有单独说明。
- **动作块大小**：K=10（LIBERO-Long），M=2，N=1（[PAPER §4.1 Training Setting]；[HF]/model_256/libero_10/args.json:12 `ck_10`）。
- **训练序列长度**：
  - **WM** 只在**单步、teacher-forcing** 的转移上训练：一个样本约 581 token，其中 277 个计入损失（[ENV] Prompt lengths）。**模型从来没有在“喂入自己生成的帧”的条件下训练过。**
  - **VLA** 一个样本是 2 帧加 10 个动作。
  - `max_seq_len` 为 4096（[HF]/model_256/libero_10/args.json:35）。
- **与 H = 520 对比**：
  - **按单次调用的格式看，是内插**：每次调用都是 1 帧 + 1 个动作 → 1 帧，与训练格式一致。
  - **按滚动长度看，是外推，而且远超训练**：520 次自回归调用 vs 训练时的 1 次。论文评测过的最长 WM 生成是 50 帧（[PAPER Table 4]），而且那是 90% 划分训练出的模型，不是发布的这份权重（[PAPER §4.1 Datasets]）。
  - **按 episode 时间轴看，基本是内插，末尾有轻微外推**：官方评测的上限为 520 步，代码注释称 “longest training demo has 505 steps”（[WV]/eval_solver_libero_256.py:212-213；该注释沿用自 OpenVLA，本卡没有独立核实）。因此第 506–520 步超出了最长的训练演示。

### B.5 变化覆盖（用于之后的 known/unknown 标注）

| 变化类型 | (a) LIBERO-10 微调数据 | (b) 预训练数据（Chameleon-7B / VQGAN） |
|---|---|---|
| 相机视角 | **没有变化**。只有 LIBERO 固定的 agentview；渲染参数里只设置了分辨率；没有几何增强。`var_center_crop` 对 256² 输入不起作用；训练预 token 化和评测走同一个 `process_image`，调用时不传 `random_top_k`，默认 1，所以裁剪是确定性的（[WV]/libero_util/libero_utils.py:18-25；[WV]/data/item_processor.py:34-40, 272） | 通用网络图像，视角多样，但不是机器人视角。具体：`unverified` |
| 光照 | 没有光照增强；数据是 LIBERO 默认渲染的重放（同上）。场景本身是否有光照变化：`unverified`，倾向于没有 | `unverified` |
| 背景/纹理 | LIBERO-10 各任务场景固定，没有纹理随机化的代码。直接声明：`unverified` | `unverified` |
| 干扰物 | 只有 LIBERO-10 场景里原本就有的非目标物体，没有额外干扰物（数据流程中没有相关代码） | `unverified` |
| 物体布局（超出原生 init 分布） | **没有**。只来自原始演示的初始状态 `orig_states[0]`，重放前 `env.seed(0)`，settle 10 步（[WV]/libero_util/regenerate_libero_dataset_filter_no_op.py:116-120；[WV]/libero_util/libero_utils.py:24）。评测用的 LIBERO `init_states` 与演示初始状态是否重合：`unverified` | 不适用（非机器人数据）/ `unverified` |
| 机器人初始位姿 | 跟随演示的初始状态，没有额外扰动的代码。直接声明：`unverified` | 不适用 / `unverified` |
| 语言改写 | **没有**。每个任务一条固定指令，来自数据目录名，`_` 换成空格。**注意**：按公开代码，libero_10 的目录名是 `task.name`，带场景前缀，例如 `LIVING_ROOM_SCENE2_put_both_…`，而评测时用的是 `task.language`，不带前缀。所以训练和评测的 VLA 指令文本可能不同。实际训练数据是否如此：`unverified`。这只影响 VLA 模式；WM 模式不带文本 | Chameleon 的文本预训练有大量自然语言，但没有针对机器人指令的改写。具体：`unverified` |
| 传感器噪声 | **没有**图像噪声增强，也没有动作噪声增强（数据流程中没有相关代码）。只有 VQ 量化损失（往返 PSNR 约 24 dB），以及动作 256 分箱的量化 | `unverified` |
| 动力学 | 只有单一的 LIBERO 仿真器（robosuite / MuJoCo），没有动力学随机化的代码。作者使用的 MuJoCo 版本：`unverified`（仓库固定 3.2.7；[ENV]） | 不适用（图文数据，没有动力学） |

出处补充：语言改写一行见 [WV]/libero_util/regenerate_libero_dataset_save_img_action.py:84；[WV]/data/action_model_conv_generation.py:58, 124；[LIBERO]/benchmark/__init__.py:44-53, 69-72；[LIBERO]/benchmark/libero_suite_task_map.py:38-49；[WV]/eval_solver_libero_256.py:188, 254-258。

---

## C. 与评测相关的接口事实

- **动作空间**：
  - 7 维：Δxyz、Δ旋转、夹爪；夹爪 −1 为张开、+1 为闭合，dummy 动作是 `[0,0,0,0,0,0,−1]`（[WV]/libero_util/libero_utils.py:28-30）。
  - 反归一化后直接送入 `env.step`。**夹爪不做反转，也不做二值化**，相关代码被注释掉了（[WV]/eval_solver_libero_256.py:268-282）。
- **动作分块（VLA 模式）**：
  - 一次生成约 10 个动作，开环执行完再重新查询。`--action_steps` 默认 25，所以 `max_new_tokens=300`（[WV]/eval_solver_libero_256.py:76, 251-268；[WV]/libero_util/Chameleon_utils.py:548）。
  - VLA 的历史只取上一帧，即 `his_img[-1:]`。
- **评测步数**：开始前 10 步 dummy，之后最多 520 步（libero_10）。结束条件是 `done`，或者异常退出（`try/except` → break）（[WV]/eval_solver_libero_256.py:206-231, 287-295）。官方每个任务 50 个 episode，用 LIBERO 的 init states 0..49，并设置 `env.seed(0)`（:194-201；[WV]/libero_util/libero_utils.py:24）。
- **no-op 与 settle 步**：训练数据去掉了 no-op 动作，所以 WM 从来没见过静止动作。开头 10 步 settle 也不在训练数据里（重放时 settle 之后才开始记录）（[WV]/libero_util/regenerate_libero_dataset_filter_no_op.py:119-139）。
- **从给定 LIBERO 状态做想象**：
  - 拿锚点帧（256px、旋转 180°；LeRobot `LiberoEnv` 原始的 `pixels.image` 没有翻转，需要自己翻转）经 VQGAN 编码，作为 WM 的第一帧；
  - 每一步：WM(帧_t, a_t) → 帧_{t+1}，解码后再编码喂回。
  - 本项目的循环是把官方策略循环里的 `env.step` 换成 WM（[PRE §1 Unit loop]；[UNIT]）。
- **`generate_img` 不重置 `model.init_input_ids`**（[WV]/model/modeling_xllmx_chameleon_ck.py:27-46, 317-327；对照 :306, :330）。VLA 和 WM 交替调用时，注意力掩码会建在上一次调用的序列上。本项目每次调用前都会重置（[UNIT]:235；[PRE §5 第 2 条]）。
- **每步成本（估计，等实测替换）**：
  - 每个环境步约 286 次串行 7B 解码；一条 520 步的滚动约 1.0–1.7 小时。
  - 显存峰值估计约 16–17 GB。
  - 出处：[PRE §3, §4]（都是估计，没有实测）。
- **不能并行或批处理**：官方代码是 batch=1 的 HF `generate`，每生成一个 token 都重建 L×L 掩码（[WV]/model/modeling_xllmx_chameleon_ck.py:31-39, 216-299）。
- **报告的成绩**：
  - LIBERO-Long (256) 成功率 **59.0%**，即 10 任务 × 50 episode 的平均；没有逐任务的数字（[REPO]/README.md:94；[PAPER Table 2]）。
  - WM 的指标（FVD/PSNR/SSIM/LPIPS，10 帧和 50 帧）只在 90% 划分训练出的模型上、在验证集上报告过，没有说明是哪个 suite（[PAPER Table 4, §4.1]）。
- **环境偏差**（本项目）：
  - torch 用 2.7.1+cu128，替代仓库固定的 2.5.0+cu124，因为后者不支持 sm_120；
  - 没有安装 flash-attn，官方路径会自动选 sdpa；
  - 环境侧（LeRobot）用 MuJoCo 3.8.1，官方评测侧用 3.2.7。
  - 出处：[ENV]；[PRE §5 第 8、10 条]。
- **许可**：代码与 HF 卡都是 Apache-2.0（[REPO]/README.md:236-238；[HF]/README.md:2）。Chameleon 自身的许可是否传递下来：`unverified`（[SCREEN]）。

---

## 未核实项（unverified）

1. **世界模型模式的正确提示词。** 官方 WM 评测代码没有发布（README:233）。现有三种措辞：训练数据格式、仓库辅助函数、论文正文，三者不一致。本项目默认用训练数据格式，这是**假设**，要等 `onestep` 的打分结果确认。
2. **发布的 `model_256/libero_10` 用了哪些数据训练。** 是全部数据还是 90% 划分（args.json 的路径同时含 “train” 和 “all”，而且该配置文件不在仓库里）；过滤后实际保留了多少条轨迹；发布的是哪个 epoch。另外，公开训练脚本（lr 5e-6，4 卡）与 args.json（lr 1e-6，world_size 16）不一致。
3. **libero_10 的 VLA 训练指令是否带场景前缀**（例如 “LIVING ROOM SCENE2 …”）。按公开代码推断会带，与评测时的 `task.language` 不同。只影响 VLA 模式。
4. **评测上限 520 步与最长训练演示 505 步的关系。** “505” 只来自沿用 OpenVLA 的代码注释，没有独立核实。
5. **Chameleon-7B 的预训练数据构成；** `Alpha-VLLM/Chameleon_7B_mGPT` 相对 Meta 原始权重是否另有训练；VQGAN 的训练数据和参数量。
6. **变化覆盖**：LIBERO-10 场景本身是否有光照、纹理或机器人初始位姿上的变化（代码里没有增强，但也没有直接的书面声明）；评测用的 init states 与演示初始状态是否重合；作者使用的 MuJoCo/robosuite 版本。
7. **论文中 WM 指标（Table 4）的 suite 归属，以及 50 帧生成是否为自回归喂回。** 这些指标来自 90% 划分训练出的模型，不是本 checkpoint。
8. **运行时数字**（每步耗时、显存峰值、`n_struct_fallback` 的频率）目前都是估计，要等 GPU 空闲后跑 smoke 测试再替换。
