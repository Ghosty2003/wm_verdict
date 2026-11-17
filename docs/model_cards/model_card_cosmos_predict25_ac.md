# 模型卡：Cosmos-Predict2.5-2B `robot/action-cond`（纯动作条件视频世界模型，候选）

> **口径（2026-11-08）**：本阶段只在虚拟环境内评测。文中的"真实""现实""real"一律指 **LIBERO 模拟器真值分支**，"迁移"指想象结论到模拟器执行的迁移。物理机器人不在范围内。
> 用途：为 WM-Verdict 补"**纯视频生成世界模型 + 外部策略**"这一架构格（与现有一体式 LingBot-VA/OpenWAM-α、三头式 Cosmos Policy 均不重复）。所有条目附出处；查不到的写 `NOT FOUND` 或 `unverified`；来源冲突两边都列。
> 编写日期：2026-11-08。页面内容为 2026-11-08 抓取。

**出处缩写**

| 缩写 | 指向 |
|---|---|
| `GH` | https://github.com/nvidia-cosmos/cosmos-predict2.5 README（main 分支，仓库已进入仅维护状态，后继 Cosmos 3） |
| `HF-CARD` | https://huggingface.co/nvidia/Cosmos-Predict2.5-2B 模型卡 |
| `HF-TREE` | 同仓库 `robot/action-cond/` 文件夹列表 |
| `INF` | GH `docs/inference_robot_action_cond.md`（2025-10-14 发布，见 GH 新闻） |
| `PT` | GH `docs/post-training_video2world_action.md` |
| `RSRCH` | https://research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5/ |
| `PAPER` | arXiv 2511.00062（HF-CARD/GH 引用；**未读全文**） |

---

## A. 结构与骨干

| 字段 | 内容 | 出处 |
|---|---|---|
| 形态 | **纯动作条件视频生成世界模型**：输入首帧图像 + 机器人动作序列，输出未来视频。无策略头、无价值头（与 Cosmos Policy 的三头结构是两条产品线）。 | HF-CARD "Model Versions"（"Given image as the first frame and a robot action sequence… Produces 256p video with 4FPS"） |
| 底座 | 扩散 Transformer（潜空间去噪，self-attn + cross-attn + FFN，AdaLN 时间嵌入），**flow based（整流流）**；基于 Cosmos-Predict2-2B；文本编码器为 Cosmos-Reason1 VLM。 | HF-CARD "Model Architecture"；GH（"flow based model"）；PT（实验名含 `rectified_flow`） |
| 参数量 | 2,059,174,912（HF 卡给的是 2B 模型族数字；action-cond 变体单独参数量 `unverified`）。 | HF-CARD |
| tokenizer/VAE | NOT FOUND（三个页面都未写明）。 | — |
| 动作接入 | 专门的 action conditioner（"we create new `conditioner` to support action"）；Bridge 配方中动作为 **7 维**（夹爪坐标系位移 x,y,z,roll,pitch,yaw + 1 维二值夹爪）。⚠️ 文档内部矛盾：`continuous_gripper_state` 写 "0=open"，action 第 7 维写 "1 open, 0 close"。 | PT §1.1 |
| 输出规格 | 256p、4 FPS；精确宽高、单次生成帧数 NOT FOUND（后训练配置名含 `bridge_13frame_256x320`，仅训练配置，不能当发布检查点规格）。 | HF-CARD；PT |
| 分块生成 | 推理输出 `0_chunk.mp4 / 1_chunk.mp4 / 3_chunk.mp4`，按 chunk 生成；示例 `chunk_size: 12`。README（2025-11-07 新闻）提到通用的 "autoregressive sliding window generation mode"，是否适用于 action-cond `unverified`。 | INF；GH |
| ⚠️ 勿混淆 | HF 卡上 1280×704 / 720p / "requires 32.54 GB" 等数字属于 Video2World 等**其他变体**，不适用于 action-cond。 | HF-CARD |

## B. 训练数据（**本项目最大的坑在这里**）

- **`robot/action-cond` 的后训练与 LIBERO 的关系：NOT FOUND。** 官方 action-cond 后训练文档只用 **Bridge 数据集**（IRASim 划分）作配方示例；发布检查点是否就是 Bridge 训练的也 NOT FOUND。出处：PT §1.1。
- **README/HF 卡里 "post-trained on Libero and RoboCasa" 说的是 `robot/policy` 变体**（2026-02-23 新闻），**不是 action-cond**。此前调研（2026-11-07 本会话）误读为 action-cond 有 LIBERO 版，**本卡更正**。出处：GH 模型表与新闻。
- 轨迹数、视频预训练总量：NOT FOUND。
- 质量指标（评测集名 NOT FOUND）：PSNR 24.95 / SSIM 0.85 / Latent L2 0.28 / FVD 146（对比 Predict1-7B 21.14/0.82/0.32/190）。出处：RSRCH "Robotics"。
- RSRCH 与 GH 的机器人模型列表不一致（RSRCH 含 `robot/multiview-basic`、`robot/gr00tdream-gr1`；GH 含 `robot/multiview-agibot`、`robot/policy`）。

## C. 检查点与推理接口

- 位置：HF `nvidia/Cosmos-Predict2.5-2B` 仓库 `robot/action-cond/` 文件夹：`38c6c645-…_ema_bf16.pt`（4.25 GB）+ `cr1_empty_string_text_embeddings.pt`（103 MB）。**gated**，需在 HF 同意条款。出处：HF-TREE。
- 推理：`python examples/action_conditioned.py -i …/inference_params.json`；关键参数 `chunk_size: 12`、`guidance: 7`、`camera_id: "base"`；**不支持多 GPU**（`context_parallel_size=1`）；setup 默认模型是 `robot/multiview`，须显式指定 `robot/action-cond`。显存需求 NOT FOUND（实测后补）。出处：INF。
- 自产后训练权重转换：`scripts/convert_distcp_to_pt.py`，推荐 `model_ema_bf16.pt`；Bridge 复现实验名 `ac_reason_embeddings_rectified_flow_2b_256_320`。另有 DMD2 蒸馏配方（4 步学生）。出处：PT §3–4。

## D. 许可证

- 模型：**NVIDIA Open Model License**（2025-09-23 版）：可商用、允许衍生模型、NVIDIA 不主张输出所有权；发起版权/专利诉讼或绕过安全护栏则终止授权；分发需附 "Licensed by NVIDIA … " 与 "Built on NVIDIA Cosmos" 声明。代码 Apache-2.0。⚠️ HF 卡 "Additional Information" 链到 Qwen3Guard 的 Apache LICENSE，疑为错链。出处：HF-CARD；GH。

## E. 与现有模型的区分（不重复性）

| 对比对象 | 区别 |
|---|---|
| LingBot-VA / OpenWAM-α | 它们是视频-动作**联合**生成（自己出动作）；本模型**只**做动作→视频，动作必须来自外部策略 |
| Cosmos Policy（Yuhang） | 同属 Cosmos 家族但是**不同产品线**：Cosmos Policy = Predict2 微调出的策略+世界模型+价值三头、以动作为主输出；本模型 = Predict2.5 的纯世界模型变体，无策略/价值头。同底座不同形态，正好构成内部对照 |
| FastWAM | FastWAM 的发布检查点 `action_conditioned=false`，**不能**当动作条件模拟器（见 FastWAM 卡 A"上下文与预测窗口"）；本模型的卖点恰是动作条件 |
| VLA-JEPA | 那是潜空间非生成类；本模型是像素级生成 |

## F. 本项目适配评估（结论）

1. **必须自己做 LIBERO 后训练**：官方没有 LIBERO 版 action-cond 检查点。路径：按 PT 配方，把我们手里的 LIBERO 演示（LeRobot 格式，四 suite）转成其动作条件格式后训练；Foresight（arXiv 2606.23085）已证明该检查点可以在 LIBERO-Long 上微调成功（先例，非代码）。工作量 = 数据转换 + 一次 2B 后训练（单节点可行，DMD2 蒸馏可降推理成本）。
2. **闭环协议**：外部策略拟用 openpi π0.5-LIBERO（与 WorldFailBench 一致，便于对照）；策略读生成帧选动作 → 模型按 chunk 生成下一段 → 循环。4 FPS/256p 与我们 20 Hz 环境的时间对齐方式**待定**（需在预检里标定，类似 OpenWAM 的块间状态口径问题）。
3. **verdict 口径**：模型无自评头，verdict = 判定管线打在生成帧上（与 LingBot 同口径）；真值 = 动作重放。扰动族全套适用（O/B 打在模型读到的真实帧，沿用 2026-10-26 裁决）。
4. **风险**：仓库仅维护状态（后继 Cosmos 3）；gated 下载需账号同意；若后训练成本超预算，备选方案是改用 RynnVLA-002 的 world-model 变体（见注册表空行）或等 Cosmos 3。

**状态：候选（待裁决）。** 裁决点：是否接受"自行 LIBERO 后训练"这一步。若否，此格改由其他模型填。
