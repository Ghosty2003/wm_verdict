# LIBERO 扰动清单与 known/unknown 标注（第 2 步交付，停点）

> **口径（2026-10-14）**：本阶段只在虚拟环境内评测。文中的“真实”“现实”“real”一律指 **LIBERO 模拟器真值分支**，“迁移”指想象结论到模拟器执行的迁移。物理机器人不在范围内。

2026-10-14。依据：四份模型档案 `model_cards/model_card_{fastwam,lawam,vla_jepa,lingbot_va}.md`（每条标注的原始出处在档案 B.5 节"变化覆盖"）、LIBERO 原生钩子清单 `libero_native_hooks.md`（LIBERO commit `8f1084e`、robosuite v1.4.0，表中"钩子 #n"即该文件第 4 节的编号）。本文件只做汇总，不新增事实；凡是档案里没有证据的都保留 `unverified`。

---

## 一、候选模型（跨模型表的行头四字段）

| 模型 | 生成机制 | 骨干来源 | 参数规模 | 动作接入方式 |
|---|---|---|---|---|
| FastWAM | 连续潜变量 flow matching；视频与动作在共享注意力的 MoT 中联合去噪（10 步 Euler） | 从 Wan2.2-TI2V-5B 微调；VAE 与 UMT5-XXL 冻结 | 6.02B（视频专家 5.00B + 动作专家 1.02B；VAE、文本编码器另计） | 动作 token 与视频 token 在 MoT 中共享注意力，但注意力掩码禁止视频 token 看动作 token；检查点 `action_conditioned=false` |
| LingBot-VA | 连续潜变量 flow matching；块内双向、块间因果的自回归（KV 缓存） | 从 Wan2.2-5B 微调；VAE 与 UMT5-XXL 冻结 | 约 5.09B（已发布的共享骨干版本；论文写 5.3B，为未发布的分离版本） | 动作 token 与视频 token 共享 30 层；当前块的视频只能看到之前块的动作 |
| LaWAM | 潜空间确定性回归：由 32 维潜动作预测下一帧的 DINOv3 特征；机器人动作由另一个 flow-matching 头生成 | Qwen3-VL-2B（截取前 16 层）+ 冻结的 DINOv3 ViT-B/16 + 预训练的潜动作模型 | 2.56B（HF 元数据；论文写 2.3B） | 机器人动作不进世界模型，世界模型只吃潜动作 |
| VLA-JEPA | 潜空间确定性回归（V-JEPA 2 特征，L1 损失），只作为训练辅助损失，推理时不运行 | Qwen3-VL-2B（全量微调）+ 冻结的 V-JEPA 2 ViT-L | 3.08B | 机器人动作不进世界模型，世界模型只吃 24 个潜动作 token |

FLUX 3 Action 未列入：只有 SO-101 检查点，没有 LIBERO 检查点，而本任务不训练模型。

**训练用的 LIBERO 套件**（决定选哪个任务）：

| 模型 | 套件 | 证据状态 |
|---|---|---|
| FastWAM | Spatial、Object、Goal、10（无 LIBERO-90），1,712 条演示 | 数据集卡；论文写每个套件 500 条，二者不一致 |
| LaWAM | Spatial、Object、Goal、Long(10)，40 个任务，1,693 条 | 数据集卡 |
| VLA-JEPA | 上游配置为四个套件（1,693 条）；LeRobot 文档写 LIBERO-10 | **冲突**，转换记录不可得 |
| LingBot-VA | LIBERO-10（500 条，10 任务 × 50）；是否用了其余套件 | `unverified`（论文 §4.3.2 与检查点名称不一致） |

四个模型都训练过的只有 **LIBERO-10**。

---

## 二、标注规则

- **known**：LIBERO 微调数据内含这类变化（有证据）。
- **known-via-pretraining**：微调数据外，但预训练数据覆盖（有证据）。
- **unknown**：两者都有证据表明不含。
- **unverified**：缺少上述任一判定所需的证据。为了不丢信息，细分两种：
  - **unverified-PT**：微调数据有证据表明**不含**，但预训练数据是否含**未核实**；
  - **unverified**：微调数据和预训练数据都未核实。
- 绝不用模型表现反推标注。
- 带"（弱）"的 known-via-pretraining 表示证据只是一般性陈述（例如"预训练数据含带光照标签的美学数据"），并不针对机器人场景。是否承认这类证据，请你定。

---

## 三、扰动清单

通道：观测 / 动作 / 动力学 / 初始状态。"首帧可见"指以首帧为条件的世界模型能否从首帧看到这个变化（钩子清单第 4 节）。

| 钩子 | 改了什么 | 通道 | 首帧可见 | FastWAM | LingBot-VA | LaWAM | VLA-JEPA |
|---|---|---|---|---|---|---|---|
| #1 init_state 索引（每任务 50 选 1） | 在原生初始分布内换一个起点 | 初始状态 | 是 | known | known | known | known |
| #7 seed 引起的 fixture 位姿（约 ±1 cm） | 原生 reset 时随机采样的柜子、炉子等位姿 | 初始状态 | 弱 | known | known | known | known |
| #9 关节物体初值（原生 `:init` 范围内） | 抽屉开合、炉子开关等 | 初始状态 | 是 | known | known | known | known |
| #2 / #4 物体位姿或摆放区域超出原生分布 | 编辑 init 向量或 BDDL region | 初始状态 | 是 | unverified | **unverified-PT**（2026-10-14 定向核实） | unverified | unverified-PT |
| #3 机器人初始关节 | 改初始位姿 | 初始状态 | 是 | unverified | **unverified-PT**（同上；超出约 0.06 rad/关节才算微调外） | unverified | unverified-PT |
| #5 增删物体 / 干扰物 | BDDL `:objects`（原 init 文件失效） | 初始状态（+观测） | 是 | unverified | unverified | unverified | unverified-PT |
| #6 fixture 种类或较大位移 | BDDL `:fixtures` / `:regions` | 初始状态（+动力学） | 是（小位移弱） | unverified | unverified | unverified | unverified-PT |
| #10 相机位姿 / FOV | `sim.model.cam_*`、`CameraModder` | 观测 | 是 | unverified | unverified | unverified | unverified-PT |
| #11 相机选择 / 分辨率 | `camera_names`、分辨率 | 观测 | 是 | unverified | unverified | unverified | unverified |
| #12 光照 | 场景 XML `<light>`、`LightingModder` | 观测 | 是 | known-via-pretraining（弱） | unverified | known-via-pretraining（弱，颜色增广） | unverified-PT |
| #13 墙 / 地纹理、场景风格 | `scene_xml`、`TextureModder` | 观测 | 是 | unverified | unverified | unverified | unverified-PT |
| #14 物体颜色 / 材质 | `TextureModder`、`geom_rgba` | 观测 | 是 | unverified | unverified | unverified | unverified-PT |
| #15 观测噪声 | robosuite 观测损坏器 | 观测 | 是 | unverified | unverified | unverified | unverified-PT |
| #16 语言改写 | 改 `Task.language` | 观测（语言条件） | 只有以语言为条件的模型可见 | unverified-PT | unverified-PT | unverified | unverified-PT |
| #17–#21 控制器类型 / 增益 / 增量绝对模式 / 控制频率 / 夹爪速度 | 改动作语义 | 动作 | 否 | unverified | unverified | unverified | unverified-PT |
| #22 动作噪声 / 丢弃 / 缩放 / 旋转 | 需 wrapper，无原生入口 | 动作 | 否 | unverified | unverified | unverified | unverified-PT |
| #23–#26 物体质量 / 摩擦、桌面摩擦、关节阻尼、仿真步长 | `sim.model.*`、`DynamicsModder` | 动力学 | 否 | unverified | unverified | unverified | unverified-PT |

**各标注的证据摘要**（全文见各档案 B.5）：
- **原生初始分布（#1、#7、#9）为 known**：四个模型的 LIBERO 微调数据都是从原生初始分布采集的官方演示（各档案 B.1–B.2），fixture 位姿在原生 reset 时随机采样（钩子清单第 5 节）。
- **VLA-JEPA 的 unverified-PT**：论文 §4.1 明确未使用 LIBERO-Plus 增强数据，上游 LIBERO 数据变换只有动作变换、没有任何图像增强。因此微调数据不含这些变化有证据，预训练（DROID + Something-Something-v2）是否覆盖则未核实。
- **FastWAM 与 LingBot-VA 的语言改写 unverified-PT**：两者每个任务都只有一条固定指令，没有改写增强。
- **FastWAM 与 LingBot-VA 的观测噪声保持 unverified**：档案只确认没有做噪声增强（FastWAM `action_state_transforms: null`；LingBot-VA 只有潜空间的 noisy-history 增强，这是训练技巧，不算传感器噪声），数据本身是否带噪声未核实。
- **LaWAM 的语言改写保持 unverified**：数据集 `total_tasks=40` 与任务数一一对应，但档案本身标为 unverified，这里不越过档案。
- **FastWAM 光照 known-via-pretraining（弱）**：Wan2.2 的说明提到带"lighting"标签的美学数据，是通用视频，不针对机器人场景。
- **LaWAM 光照 known-via-pretraining（弱）**：潜动作模型预训练阶段有颜色增广（论文 §C.5），属于光度扰动，不是真实光照变化。
- **LaWAM 相机视角仍为 unverified**：预训练跨数据集，相机配置多样，也有随机裁剪；但论文 §5 自述"大幅视角变化是已知局限"，没有覆盖的直接证据。

不属于四个通道的两个旋钮单列，不作扰动：#27 goal 谓词（只改成功标签）、#28 最大步数（预算）。

**LIBERO-Plus**：本环境未安装。它的 7 个维度（物体布局、相机、机器人初始状态、语言、光照、背景纹理、传感器噪声）都落在上表已有的初始状态和观测通道里，没有动作或动力学扰动。安装它会替换 `hf-libero`。

---

## 四、需要你决定的事

**1. 预注册假设现在无法按原设计检验。** 证据标准下，表中没有任何一格能标 unknown：微调数据"不含"有时有证据，预训练数据"不含"一次都没能核实。可选做法：
- (a) 把 **unverified-PT**（微调数据确实不含、预训练未核实）作为"微调外"组，与 known（原生分布内）对比，假设改写为"微调外扰动下想象与真实的差距大于微调内"，并在结论里注明预训练覆盖未排除。
- (b) 继续去核实预训练数据集（DROID、Something-Something-v2、OXE 等）的数据卡，尝试把部分格子升级为 unknown 或 known-via-pretraining。工作量大，多数可能仍然查不到。
- (c) 只在 VLA-JEPA 上做 known 与 unverified-PT 的对比，因为它的微调侧证据最完整。

**2. 闸门预判（基于档案，第 3 步实测前不作结论）**：四个模型都没有"外部机器人动作驱动想象"的通路。

| 模型 | 外部动作进入想象 | 解码出画面 | 想象长度 |
|---|---|---|---|
| FastWAM | 否：视频只取决于首帧与语言 | 能 | 每次 32 步，无长 rollout（把末帧当新首帧的路径未训练过） |
| LingBot-VA | 只能作为历史（连同真实画面）进入 KV 缓存；当前块的视频看不到本块动作 | 能 | 块间自回归，每块 16 步 |
| LaWAM | 否：只吃潜动作，且只能前推一步 | 否：DINOv3 特征 | 一步 |
| VLA-JEPA | 否：推理时不运行世界模型 | 否：V-JEPA 2 特征 | 训练时 3 个潜步 |

第 5 步对有策略头的模型走 own-policy 闭环（两侧同策略同噪声种子），其中关键在于：想象出来的画面是否是"这组动作的后果"。FastWAM 的画面与动作互不影响；LingBot-VA 只有块间依赖。照你的规则"任一闸门 FAIL 如实关闭该模型、不绕过"，预计 LaWAM 和 VLA-JEPA 会在"帧能否解码"一条关闭；FastWAM 和 LingBot-VA 在"动作条件化真伪"一条的判法需要你先定：
- 对 own-policy 模型，"动作条件化"是要求画面依赖于本 rollout 实际执行的动作（严格），还是只要求模型联合给出画面和动作（宽松）？
- 严格口径下，FastWAM 关闭，LingBot-VA 需要实测"改动历史动作是否改变后续块画面"。

**3. 任务**：公共任务集建议见第五节（LIBERO-10，四个模型训练覆盖的交集）。最大步数各处说法不一（libero_10：LeRobot 520，FastWAM 文档 600，FastWAM 上游 700，LaWAM 上游 550，LingBot 上游 800），需要定一个统一的 H，并按 DINO-WM 阶段的口径标明对每个模型是内插还是外推。

**4. 扰动与档位**：上表任选。如果选观测类，按对齐清单第 1 条它会进入模型首帧，属于 fidelity 轨道，需要分开列表；动作类和动力学类首帧不可见，属于 verdict 轨道。

**5. 环境**：LIBERO 与 robosuite 未安装（需 `hf-libero`，固定 robosuite 1.4.0）；LaWAM 要求 MuJoCo 3.3.2，当前 lerobot 环境是 3.14.0。另外钩子清单第 5 节指出，扁平状态不足以逐位重放：fixture 位姿、求解器热启动、夹爪积分器、控制器状态、全局 RNG 都不在其中。最稳的做法是用同一 seed、同一 init 索引硬重置，再从头重放动作；第 5 步的锚恢复要按这个方案先过逐位重放验证。

---

## 五、公共任务集建议（四个模型训练覆盖的交集）

| 套件 | FastWAM | LingBot-VA | LaWAM | VLA-JEPA |
|---|---|---|---|---|
| LIBERO-10（Long） | 训过 | 训过 | 训过 | 训过（LeRobot 文档与上游配置都包含） |
| LIBERO-Spatial / Object / Goal | 训过 | **unverified**（论文 §4.3.2 与检查点名称不一致） | 训过 | **冲突**（上游配置包含，LeRobot 文档只写 LIBERO-10） |
| LIBERO-90 | 未训过 | 未训过（无证据） | 未训过 | 未训过 |

**建议的公共任务集 = LIBERO-10 的 10 个任务**（任务名取自 LIBERO `libero_suite_task_map.py`，commit `8f1084e`）：

| # | 任务 |
|---|---|
| 0 | LIVING_ROOM_SCENE2：把字母汤和番茄酱都放进篮子 |
| 1 | LIVING_ROOM_SCENE2：把奶油奶酪盒和黄油都放进篮子 |
| 2 | KITCHEN_SCENE3：打开炉子，把摩卡壶放上去 |
| 3 | KITCHEN_SCENE4：把黑碗放进柜子的底层抽屉并关上 |
| 4 | LIVING_ROOM_SCENE5：白杯放左盘、黄白杯放右盘 |
| 5 | STUDY_SCENE1：拿起书放进收纳盒的后格 |
| 6 | LIVING_ROOM_SCENE6：白杯放到盘子上，巧克力布丁放在盘子右边 |
| 7 | LIVING_ROOM_SCENE1：把字母汤和奶油奶酪盒都放进篮子 |
| 8 | KITCHEN_SCENE8：把两个摩卡壶都放到炉子上 |
| 9 | KITCHEN_SCENE6：把黄白杯放进微波炉并关上 |

**公共集之外的任务**：Spatial / Object / Goal 对 FastWAM 和 LaWAM 是训过的；对 LingBot-VA 和 VLA-JEPA 是否训过不能确认。按追加指令，"某模型没训过的任务"标**任务级 unknown**，不进主对比表、单独列。这两个模型在这三个套件上的训练覆盖既没有证据表明训过，也没有证据表明没训过，按证据标准应标**任务级 unverified**，同样不进主表、单独列。LIBERO-90 四个模型都没训过（各档案都没有它出现在训练数据中的证据，FastWAM、LaWAM 档案明确写了不含），作为任务级 unknown 的候选。

公共任务集、锚库、扰动设置和结果表的约定见 `eval_design.md`。

---

## 六、族 A 定向核实（2026-10-14，LingBot-VA）

详见 `familyA_evidence.md`。结论：LingBot-VA 的**物体布局超出原生分布（#2/#4）**与**机器人初始位姿（#3）**两格由 unverified 升级为 **unverified-PT**，附以下缺口：
- 训练集 `robbyant/libero-long-lerobot` 与官方 LIBERO-10 演示指纹一致（500 条，每任务 50 条；128×128；键名与官方 hdf5 相同；两个独立转换版本的逐任务回合长度集合完全一致），但数据卡没有出处声明，属于指纹匹配，不是直接声明。
- 检查点与该数据集的对应关系没有官方说明：上游 README 把它列为唯一的 LIBERO 后训练数据集，训练配置里的 `dataset_path` 是占位符；论文 §4.3.2 说用了四个套件并过滤了失败演示，与 500 条未过滤相矛盾。即使用了四个套件，其他套件的场景与 LIBERO-10 不同（例外：LIBERO-Object 在地面场景里有若干同名物体，位置不同）。
- 官方演示的布局来自 `env.reset()`，即在 BDDL region 内均匀采样；数据加载器只读预提取的潜变量，没有增强。
- 机器人初始位姿 = 统一的 `init_qpos` 加每关节 0.02 rad 的高斯噪声（eval init 文件与训练集首帧的统计都与此吻合）；扰动需明显大于约 0.06 rad（3σ）才算微调外。
- 补缺途径：用 HTTP range 只读官方 hdf5 的 `states` 数组（约 100–150 MB）逐条核对首帧物体位置都在 region 内；向作者确认检查点的实际训练集（本地无法关闭的唯一缺口）。
