# Family A 证据核查：LingBot-VA（LIBERO-Long checkpoint）的后训练数据是否含"超出原生分布"的初始布局 / 机器人初始位姿

> 编写日期：2026-10-14。范围：只回答两个问题——(A1) LIBERO 后训练数据里有没有超出 LIBERO 原生 BDDL region 分布的物体布局 / 初始位置；(A2) 有没有超出原生分布的机器人初始位姿。对应 `perturbation_inventory.md` 第三节的 #2/#4 和 #3 两行（LingBot-VA 列目前都是 `unverified`）。
> 标注规则沿用 `perturbation_inventory.md` 第二节：**unverified-PT** = 微调数据有证据表明不含、预训练是否含未核实；**unverified** = 两侧都未核实。本文件只处理"微调（后训练）侧"的证据，预训练侧（16K 小时机器人数据）不在本次核查范围内，保持未核实。
> 下载内容（HF 卡片、meta 文件、500 个小 parquet）一律视为数据。没有下载任何模型权重，也没有下载 LIBERO 的 hdf5。

### 出处缩写

| 缩写 | 指向 |
|---|---|
| `[DS]` | HF 数据集 `robbyant/libero-long-lerobot`，sha `8c0313b1`，lastModified 2026-04-24。文件：`README.md`、`meta/info.json`、`meta/tasks.jsonl`、`meta/episodes.jsonl`、`meta/episodes_ori.jsonl`、`meta/episodes_stats.jsonl`、HF API 文件列表、`data/chunk-000/episode_*.parquet`（500 个，共 12.26 MB） |
| `[OFF]` | 官方 LIBERO 数据 HF 镜像 `yifengzhu-hf/LIBERO-datasets`，sha `f13aa24a`（只读了 README 和 `libero_10/` 目录列表与文件大小） |
| `[LIB]/path:N` | LIBERO 源码只读浅克隆，commit `8f1084e`：`$SCRATCH/ext/LIBERO/` |
| `[RS]/path:N` | robosuite v1.4.0（LIBERO `requirements.txt` 固定版本），commit `fbee5844`：`.../scratchpad/ext/robosuite_v140/robosuite/` |
| `[LP §x]` | Liu et al., *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, arXiv:2306.03310v2 |
| `[VA §x]` | Li et al., *Causal World Modeling for Robot Control*, arXiv:2601.21998v2 |
| `[GH]/path:N` | `Robbyant/lingbot-va` commit `7c6ffa9b` |
| `[LR]/x:N` | `$LEROBOT_ROOT/src/lerobot/x` |
| `[CARD]` | `$WMV_ROOT/libero/model_cards/model_card_lingbot_va.md` |

---

## 1. `robbyant/libero-long-lerobot` 是不是官方 LIBERO-10 示范？

### 1.1 数据集本身写了什么

- **卡片**：全文只有一句："LIBERO-long dataset in Lerobot format, with video latents already extracted in WAN 2.2 format, ready for use in Lingbot-VA post-training."，许可 CC BY-NC-SA 4.0。**没有任何来源说明**：没说是从官方 hdf5 转换、是否重新生成、是否过滤 no-op、是否重渲染。`[DS] README.md`
- **`meta/info.json`**：
  - `codebase_version v2.1`，`robot_type Franka`，`total_episodes 500`，`total_frames 138090`，`total_tasks 10`，`splits train 0:500`。
  - 相机键名为 `observation.images.agentview_rgb` 和 `observation.images.eye_in_hand_rgb`，都是 **128×128**，h264。
  - `observation.state` 为 8 维（名字写作 x,y,z,roll,pitch,yaw,gripper,gripper）；`action` 为 7 维。
  - `fps: 60`。
  - 数据集里没有物体位姿、没有 MuJoCo state，也没有 init_state 字段。parquet 只有 `episode_index, index, frame_index, task_index, timestamp, action, observation.state` 这几列。
- **`meta/tasks.jsonl`**：10 条指令，正好是 LIBERO-10 的 10 个任务。
- **`meta/episodes.jsonl`**：500 行，**每个任务正好 50 条**。单条长度 150–517，合计 138,090。`episodes_ori.jsonl` 与它的 episode/长度一致，只少了 `action_config` 字段。
- **文件列表**：`data/` 500 个，`videos/` 1000 个，`latents/` 1001 个（其中 1 个是 `.DS_Store`）。

### 1.2 与官方发布的对照（指纹比对）

| 项目 | 官方 LIBERO-10 hdf5 | `[DS]` | 判断 |
|---|---|---|---|
| 示范来源 | 每任务 50 条人类遥操作示范（3Dconnexion SpaceMouse）`[LP §4.4 末段]`；采集脚本默认 `--num-demonstration 50` `[LIB]/scripts/collect_demonstration.py:249-253` | 10×50 = 500 | 一致 |
| 图像分辨率 | `create_dataset.py` 以 `camera_heights=128, camera_widths=128` 回放渲染 `[LIB]/scripts/create_dataset.py:97-98` | 128×128 | 一致（不是 OpenVLA 那种 256×256 重新生成版） |
| 图像键名 | hdf5 中为 `obs/agentview_rgb`、`obs/eye_in_hand_rgb` `[LIB]/scripts/create_dataset.py:248-251` | `agentview_rgb`、`eye_in_hand_rgb` | 一致（OpenVLA/openpi 系列转换用的是 `image`/`wrist_image`） |
| 本体状态 | `ee_states`=eef 位置+轴角（6 维），`gripper_states`（2 维）`[LIB]/scripts/create_dataset.py:199-210, 240-246` | 8 维；首帧第 4 维约 3.14、夹爪两维为 ±0.036，即轴角+两指 qpos | 一致 |
| 控制频率 | `control_freq=20` `[LIB]/scripts/create_dataset.py:96` | `fps: 60`，`timestamp` 步长 1/60 s | **不一致**，应是元数据错误，与布局问题无关 |
| 官方文件 | `[OFF] libero_10/` 下 10 个 `*_demo.hdf5`，合计约 13.73 GB；下载脚本 `HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"` `[LIB]/libero/libero/utils/download_utils.py:108` | — | — |

**与其他独立转换的逐任务长度比对**（我这次做的核查）：

- HF 上 LIBERO-10 的 LeRobot 转换里，凡是由 OpenVLA no-noops / openpi 重新生成的版本，都是 **379 条 / 101,469 帧**（例如 `lerobot/libero_10`、`IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot`、`fracapuano/libero_10`）。另有 391 条的变体（`mimimimi2002/libero_10`）。
- 另外两个与 robbyant 无关的转换 `ZibinDong/libero_10`（224×224，fps 20）和 `sean1295/libero_10_various_action_spaces`（256×256，15 维 state）都是 **500 条 / 138,090 帧**。
- 把每个任务的 50 条长度当作多重集合比较：`[DS]` 与这两个数据集在 **10 个任务上全部完全相同**。三者的任务内顺序不同，说明各自独立排序。
- `XinY0201/libero-long-lerobot` 的 info.json 与 `[DS]` 完全相同（500 / 138,090 / fps 60 / 128×128），看起来是 `[DS]` 的镜像。

**结论（1.2）**：`[DS]` 几乎可以确定就是官方 LIBERO-10 hdf5 的直接格式转换，具体来说：

- 没有过滤（500/500），保留了 no-op；
- 没有重新生成、没有重渲染，128×128 就是官方 hdf5 原生分辨率；
- 首帧对应官方 hdf5 的第 0 帧，也就是原始采集的第 5 步，见 2.3。

这是**指纹推断**：独立转换的逐条长度一致，键名和分辨率也一致。数据集卡片本身没有任何来源声明。要把推断变成直接证据，需要 4.1 中的 ① 号核查。

**旁证**：LIBERO-10 的最长示范 517 步，大于 LeRobot 注释里 no-noops 版的最长 505 步 `[LR]/envs/libero.py:102`。去掉 no-op 只会把轨迹变短，这与"含 no-op 的原始版本"一致。

### 1.3 LingBot-VA 这一侧怎么说

- **上游 README**：
  - 2026-04-24 发布 LIBERO-LONG 后训练权重 `[GH]/README.md:54`；
  - checkpoint 名为 `lingbot-va-posttrain-libero-long` `[GH]/README.md:73`；
  - "Post-Training Dataset" 表中 LIBERO 只列了 `libero-long-lerobot`，描述为 "LIBERO-Long dataset in LeRobot format for post-training" `[GH]/README.md:75-80`；
  - `[DS]` 的 lastModified 也是 2026-04-24。
- **上游训练配置**：`va_libero_train_cfg.dataset_path = '/path/to/your/dataset'`，是个占位符 `[GH]/wan_va/configs/va_libero_train_cfg.py:9`，没有指定是哪个数据集。
- **数据加载器**：只读取预先提取的 VAE 潜变量 `.pth`，没有任何几何、状态或初始化相关的增强 `[GH]/wan_va/dataset/lerobot_latent_dataset.py:209-222`；`[CARD]` B.5 也是这个结论。因此后训练不可能在数据侧"生成"新的初始布局。新布局只能来自额外仿真采集的数据，而 README 没有提到任何额外数据。
- **论文 §4.3.2**（与上面有冲突）：原文是 "we train our model on four LIBERO suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long. Each suite contains 10 tasks with 50 demonstrations per task (500 total). Following OpenVLA, we filter unsuccessful demonstrations before training."
  - 如果真的按 OpenVLA 过滤，LIBERO-10 应剩约 379 条；而 `[DS]` 是 500 条、未过滤。
  - 所以，**"发布的 checkpoint 恰好用 `[DS]` 训练"仍然没有直接声明**。HF 模型卡和 config 都没写训练数据 `[CARD]` B.2。
- **冲突对本问题的影响有限**：即使 checkpoint 实际上联合训练了 4 个 suite，其他三个 suite 也都是官方人类示范，场景与 LIBERO-10 完全不重叠：
  - LIBERO-10：Kitchen / Living_Room / Study Tabletop，region target 为 `kitchen_table` / `living_room_table` / `study_table`；
  - Spatial / Goal：`LIBERO_Tabletop_Manipulation`；
  - Object：`LIBERO_Floor_Manipulation`。

  依据 `[LIB]/libero/libero/bddl_files/{libero_10,libero_spatial,libero_object,libero_goal}/*.bddl` 的 `(define (problem …))` 行。因此不会出现"LIBERO-10 场景里物体越出 region"的样本。**需要注意**：LIBERO-Object 中有与 LIBERO-10 相同的物体（alphabet soup、cream cheese、tomato sauce、butter、basket 等），它们以不同位置出现在地面场景中。如果 4-suite 联训属实，模型在微调阶段确实见过"同一物体、不同位置、不同场景"。这削弱的是"未知性"，但不构成"LIBERO-10 任务的超分布布局"。

---

## 2. 官方 LIBERO 示范的初始布局是否都在原生 BDDL region 内采样？

### 2.1 采集流程（代码证据）

1. **建环境**：`collect_demonstration.py` 直接用 `TASK_MAPPING[problem_name](bddl_file_name=…, robots, controller_configs, …, control_freq=20)` 建环境，**没有传 `initialization_noise`，也没有传任何 placement 覆盖** `[LIB]/scripts/collect_demonstration.py:281-291`。之后套上 robosuite 的 `DataCollectionWrapper` `[LIB]/scripts/collect_demonstration.py:306`。
2. **每条示范开始先 `env.reset()`**，失败（例如 `RandomizationError`）就重试 `[LIB]/scripts/collect_demonstration.py:36-42`。一共循环到采满 50 条成功示范 `[LIB]/scripts/collect_demonstration.py:346-356`。操作员中途按 reset 的那条直接丢弃，不保存 `[LIB]/scripts/collect_demonstration.py:73-77, 98-99`。成功条件是 `_check_success()` 连续成立 10 步 `[LIB]/scripts/collect_demonstration.py:84-94`。
3. **reset 时的物体放置**：robosuite 基类默认 `deterministic_reset = False` `[RS]/environments/base.py:128`。于是 `BDDLBaseDomain._reset_internal` 走随机分支 `[LIB]/libero/libero/envs/bddl_base_domain.py:738`：
   - 关节物体（抽屉、炉子等）按 `:init` 中的 open/close/turnon/turnoff，在 `default_*_ranges` 内采样 `[…/bddl_base_domain.py:741-754]`；
   - 然后依次调用 `placement_initializer.sample()` → `conditional_placement_initializer.sample()` → `conditional_placement_on_objects_initializer.sample()`，再把结果写入 `set_joint_qpos` 或 fixture 的 `body_pos` `[…/bddl_base_domain.py:757-777]`。
4. **采样器从 BDDL 的 `:regions` / `:init` 构造** `[…/bddl_base_domain.py:562-729]`：
   - `(On obj region)` 中 region 的 target 是桌面时：可动物体用 `get_region_samplers(problem, "living_room_table"…)`，即 `Libero100TableRegionSampler`；fixture 用 `MultiRegionRandomSampler`。`x_ranges/y_ranges` 由 BDDL `:ranges` 的矩形转换而来 `[…/bddl_base_domain.py:590-630]`；`rectangle2xyrange` 见 `[LIB]/libero/libero/envs/utils.py:283-289`；`REGION_SAMPLERS` 见 `[LIB]/libero/libero/envs/regions/__init__.py`。
   - `(On obj site)` 中 region 的 target 是物体或 fixture 时，用 `SiteRegionRandomSampler`，在该 site 的 xy 尺寸内采样 `[…/bddl_base_domain.py:674-692]`。
   - `(On obj other_obj)` 用 `ObjectBasedSampler`，偏移为 0 `[…/bddl_base_domain.py:693-709]`。
   - `(In obj region)` 用 `InSiteRegionRandomSampler` `[…/bddl_base_domain.py:710-728]`。
5. **采样方式**：物体中心在 region 范围内**均匀采样**，`np.random.uniform(low=min, high=max)` `[LIB]/libero/libero/envs/regions/base_region_sampler.py:68-94`。不允许重叠，最多重试 5000 次，失败就抛 `RandomizationError` `[…/base_region_sampler.py:179-215]`。偏航角在 BDDL 的 `:yaw_rotation` 或物体自带的 `rotation` 范围内采样 `[…/base_region_sampler.py:96-125]`。
   - 示例：`LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl` 中，alphabet_soup 的 region 是 `(-0.125 -0.175 -0.075 -0.125)`，即 5 cm × 5 cm `[LIB]/libero/libero/bddl_files/libero_10/…tomato_sauce_in_the_basket.bddl:60-70]`。
6. **保存的是 reset 当时的状态**：`DataCollectionWrapper.reset()` 在 `env.reset()` 之后记录 `_current_task_instance_state` 和 xml `[RS]/wrappers/data_collection_wrapper.py:56-78, 130-139]`，第一次 `step` 时把它作为 `states[0]` 存盘 `[…:80-107, 141-170]`。原始 `demo.hdf5` 中每条示范的 `states[0]` 和 `model_file` 就是这个采样出来的初始布局 `[LIB]/scripts/collect_demonstration.py:104-179]`。
7. **论文表述**：µ0 由 PDDL/BDDL 描述"物体类别及其摆放（Fig.2-A）和初始状态（Fig.2-B）" `[LP §4.1 Initial State Distribution]`；示范为 "50 trajectories of high-quality demonstrations for every single task … collected by human experts through teleoperation with 3Dconnexion Spacemouse" `[LP §4.4 末段]`。论文**没有**单独写"示范初始状态从 BDDL region 采样"这句话，这一点是由上面的代码链得出的。

### 2.2 示范与 `init_states` 文件的关系

- 评测用的 `init_files/libero_10/*.pruned_init` 每个任务 50 个，`*.init` 每个任务 100 个，内容都是扁平化的 MuJoCo state。由 `benchmark.get_task_init_states` 读取 `[LIB]/libero/libero/benchmark/__init__.py:76, 158-165]`；评测时 `env.reset()` 之后再 `set_init_state` 覆盖 `[LIB]/libero/libero/envs/env_wrapper.py:136-145]`。LingBot 上游客户端 `[GH]/evaluation/libero/client.py:64-66, 88-91]` 和 LeRobot `[LR]/envs/libero.py:67-86]` 都是这样用的。
- **仓库里没有生成 init 文件的脚本**（grep `pruned_init` / `init_states` 只找到读取代码）。所以"init 文件是不是取自示范首帧"在仓库层面无法确认：`unverified`。不过有间接证据表明两者来自**同一个 reset 采样器**，见第 3 节的关节噪声统计。
- 因此"原生分布"有两种口径：(i) BDDL region 采样器诱导的分布；(ii) 50 个固定 init state。两者都由同一个 `_reset_internal` 产生。示范属于 (i) 的独立样本，与 (ii) 的 50 个点**不一定逐点相同**。只要扰动定义为"越出 BDDL region"，这两种口径对本问题没有区别。

### 2.3 一个需要知道的细节：已发布 hdf5 的第 0 帧是原始第 5 步

`create_dataset.py` 的处理顺序：

1. 先从原始 `states[0]` 回放 `[LIB]/scripts/create_dataset.py:150-154]`；
2. 跳过前 `cap_index=5` 步（注释写"force sensor is not stable in the beginning"）`[…:123, 190-195]`；
3. 再执行 `states = states[valid_index]` `[…:226]`。

结果是已发布 hdf5 中的 `states[0]` 以及 `attrs["init_state"] = states[init_idx]` `[…:267]` 实际上是**原始第 5 步**（约 0.25 s）的状态，`[DS]` 的第 0 帧也一样。这段时间里物体会从 z_offset 落下并稳定，但还不会被机器人推动（原因：操作员刚开始动作）。所以在核查"首帧物体位置是否在 region 内"时，需要留出约 1 cm 的容差。

### 2.4 如何用实证核查"全部示范的物体初始位置都在 BDDL region 内"（只描述，没有执行）

- **数据**：`[OFF] libero_10/*.hdf5`，共 10 个文件约 **13.73 GB**。每条示范有 `data/demo_i/states`（N×D 扁平 state，D 在 45–123 之间）、`attrs["init_state"]`、`attrs["model_file"]`；group 属性有 `env_args`、`problem_info`、`bddl_file_content` `[LIB]/scripts/create_dataset.py:79-118, 260-267]`。
- **步骤**：
  1. 先比对 `bddl_file_content` 与仓库 BDDL，确认 region 定义没有改过。
  2. 用 `env_args` 建环境，对每条示范执行 `reset_from_xml_string(model_file)`、`set_state_from_flattened(states[0])`、`forward()`。
  3. 读取每个可动物体 free joint 的 xy。对每个 sampler，按 `x_ranges/y_ranges + reference_pos(workspace_offset) + 目标 fixture 的位置` 换算出世界坐标下的矩形，检查物体中心是否落在矩形内，容差约 1 cm。
  4. site 类物体检查是否在 site 尺寸内。
  5. 关节物体检查关节值是否在 `default_*_ranges` 内。
  6. fixture 从 `model_file` 中读取 body_pos，检查是否在 region 内。
  7. **顺带核实来源**：把每条示范的 `num_samples` 和 `obs/ee_states[0]` 与 `[DS]` 对应 episode 的长度和 `observation.state[0]` 逐一比较。完全一致就把 1.2 的指纹推断变成直接证据。
- **代价**：
  - 全量下载约 13.7 GB；之后 500 次 `set_state` 在单 CPU 上只需几分钟。
  - 更省的做法：用 h5py 的 ros3 驱动或 fsspec 对 HF 的 resolve URL 做 HTTP Range 读取，**只读 `states` 数据集和属性**，不读图像。每条约 N≈280 × D≈120 × 8 B ≈ 0.3 MB，总共约 100–150 MB，外加元数据。
  - 两种做法都需要运行 LIBERO/MuJoCo，本次按要求没有执行。

---

## 3. 机器人初始关节配置

### 3.1 代码证据

- **初始关节**：LIBERO 的 `MountedPanda.init_qpos = [0, -0.161, 0, -2.4446, 0, 2.2268, π/4]` `[LIB]/libero/libero/envs/robots/mounted_panda.py:34-38]`。`OnTheGroundPanda` 完全相同 `[LIB]/libero/libero/envs/robots/on_the_ground_panda.py:34-38]`，所以 4 个 suite 的关节初值都是同一个。
- **初始化噪声**：
  - `BDDLBaseDomain` 的默认值是 `initialization_noise="default"` `[LIB]/libero/libero/envs/bddl_base_domain.py:49, 141]`；
  - robosuite 把 `"default"` 解释为 `{"magnitude": 0.02, "type": "gaussian"}` `[RS]/robots/robot.py:65-68]`；
  - reset 时，只要不是 deterministic，就执行 `init_qpos += randn(7) * 0.02` `[RS]/robots/robot.py:129-141]`，调用点见 `[RS]/environments/robot_env.py:520]`；
  - 采集脚本没有覆盖这个参数（见 2.1 第 1 条），所以**示范的机器人初始关节 = init_qpos + 每个关节独立的 N(0, 0.02 rad)**。
  - 对比：评测封装 `ControlEnv` 的默认值是 `initialization_noise=None`，即无噪声 `[LIB]/libero/libero/envs/env_wrapper.py:19, 61]`。但评测会用 init 文件覆盖，所以评测时实际起点是 init 文件里的关节值。
- **夹爪**：初始张开。`[DS]` 首帧的两指 qpos 平均 ±0.0362，见下文。

### 3.2 我做的实证核查

**(a) 评测 init 文件中的关节值。** 只读解码，用受限 unpickler，只放行 numpy 重建函数。脚本在 `.../scratchpad/fa_scripts/read_init.py`。

- libero_10 的 10 个 `.pruned_init`（50×D）和 10 个 `.init`（100×D）：取扁平 state 的 `[1:8]`，与 `init_qpos` 相减。
  - 每个关节的标准差在 **0.015–0.025 rad** 之间；
  - 平均 |dq| 为 **0.015–0.017**，正好等于 0.02·√(2/π) ≈ 0.016；
  - 最大 |dq| 为 0.05–0.08，约 3–4σ。
- libero_spatial / object / goal 的 init 文件结果相同。

这说明评测初始状态就是 `init_qpos + N(0, 0.02)` 这个 reset 采样器生成的，**与采集示范用的是同一个关节噪声分布**。

**(b) 训练数据首帧的 EEF。** 读取 `[DS]` 全部 500 个 parquet 的第 0 帧 `observation.state`（12 MB，存放在 `.../scratchpad/fa_ds/parquet/`；脚本 `.../scratchpad/fa_scripts/frame0.py`）。

- 每个任务内首帧 EEF 的 xyz 标准差为 **0.6–1.5 cm**，极差 **2.5–6.8 cm**；轴角各分量的标准差在 0.013–0.047 之间。
- 任务之间的均值差异只来自场景对应的机座偏移：kitchen/study 场景 z≈1.17，living room 场景 z≈0.68（见 `mounted_panda.py:41-48` 的 `base_xpos_offset`）。
- 7 个关节各带 0.02 rad 噪声，传到 EEF 大约是厘米级。再加上首帧已经是第 5 步（约 0.25 s 的遥操作），这个离散度与"原生噪声分布"吻合，**看不到任何系统性的机器人初始位姿扰动**。
- 这项核查的局限：它是 EEF 空间的统计，而且不是严格的 reset 时刻；没有用 FK 把 init 文件的关节值换算成 EEF 后做分布比较。

---

## 4. 结论

### A1：物体布局 / 初始位置（#2 / #4）——**升级为 unverified-PT**

**证据链：**

1. 唯一公开的 LIBERO 后训练数据 `robbyant/libero-long-lerobot` 由上游 README 列为 LIBERO 后训练数据集，与 checkpoint 同日发布 `[GH]/README.md:54, 73, 80]`。
2. 它是官方 LIBERO-10 人类示范的直接格式转换：500 = 10×50、未过滤、128×128 原生分辨率、键名与官方 hdf5 一致，逐任务长度多重集合与两个独立的 500 条转换完全相同（见 1.2）。
3. 官方示范的初始布局由 `env.reset()` → BDDL `:regions`/`:init` 构造的 region 采样器均匀采样得到 `[LIB]/scripts/collect_demonstration.py:36-42, 281-291]`、`[LIB]/…/bddl_base_domain.py:562-777]`、`[LIB]/…/base_region_sampler.py:68-94, 179-215]`、`[LP §4.1, §4.4]`。
4. 训练管线只读取预提取的潜变量，没有任何能改变初始布局的增强 `[GH]/wan_va/dataset/lerobot_latent_dataset.py:209-222]`。
5. 即便按论文所说联合训练了 4 个 suite，其余 suite 的场景与 LIBERO-10 不重叠（见 1.3）。

**残留缺口（不影响升级，但应在结论里注明）：**

- (i) 没有直接声明"发布的 checkpoint 只用了这个数据集"，而且论文的"4 suites + filter"与 500 条未过滤的数据存在冲突；
- (ii) "官方 hdf5 的首帧都在 region 内"是代码推导，没有逐条实证；
- (iii) 如果 4-suite 联训属实，LIBERO-Object 中的同名物体以其他位置出现在地面场景，这会削弱"微调外"的纯度；
- (iv) 预训练侧仍未核实（这正是 unverified-PT 中"PT"的含义）。

### A2：机器人初始位姿（#3）——**升级为 unverified-PT**

**证据链：**

1. 所有 LIBERO 示范和评测 init 文件的关节初值都是 `init_qpos + N(0, 0.02 rad)` `[LIB]/…/mounted_panda.py:34-38]`、`[RS]/robots/robot.py:65-68, 129-141]`。
2. 实证：libero_10 init 文件中关节偏差的统计与 σ=0.02 高斯完全吻合（3.2 a）；`[DS]` 500 条首帧 EEF 的离散度在厘米级，没有系统性扰动（3.2 b）。
3. 两个机器人类的 init_qpos 相同，所以无论 checkpoint 用了 1 个还是 4 个 suite，关节初值分布都不变，这一项不受 1.3 冲突的影响。

**残留缺口：**

- 与 A1 的 (i) 相同（checkpoint 与数据集的对应关系只是强推断）；
- 首帧是第 5 步，而不是 reset 时刻；
- 预训练侧未核实。
- **需要特别说明（2026-10-15 改为实测依据）**：known 的经验边界取 **±0.06 rad / 关节**。依据是实测，而不是从噪声参数推断：
  - LIBERO 原生评测起点（`pruned_init`，每个任务 50 个）中，机器人 7 个关节相对 `init_qpos` 的每状态最大偏差：
    - t5：中位 0.034 rad，最大 0.059；
    - t3：中位 0.035，最大 0.052。
  - 全部关节都在 ±0.02 以内的起点只有 1/50（t5）和 3/50（t3）；全部在 ±0.06 以内的，两个任务都是 50/50。
  - `init_states=False` 时，原生 `initialization_noise` 默认关闭，20 次重置的偏差都是 0。可见原生起点的离散度来自 init 文件本身。
  - 测量见 `eval_design.md` §19。
  - **对族 A 位姿档的含义**：P1（0.02）与 P2（0.06）都标 **known**，其中 P2 是 known 的边界；P3（0.12）与 P4（0.24）标**微调外**（预训练覆盖未排除）。
  - 这个结论与原先"约 3σ ≈ 0.06"的参数推断恰好一致，但依据已经从参数推断升级为实测。

### 要把残留缺口补上，还需要什么

1. **（补 A1-ii，并把 1.2 的推断变成直接证据）**：按 2.4 用 HTTP Range 只读官方 hdf5 的 `states`、`num_samples`、`obs/ee_states[0]`（约 100–150 MB），在 LIBERO 环境中检查 500 条示范首帧的物体是否都在 region 内（容差约 1 cm），同时把长度和首帧 EEF 与 `[DS]` 逐条比对。代价：小时级人工，加几分钟 CPU。
2. **（补 A2 的 EEF 对比）**：用 MuJoCo FK 把 libero_10 init 文件的关节值换算成 EEF，与 `[DS]` 首帧 EEF 的分布做比较。更严格的做法是直接用第 1 步得到的 hdf5 `states[0][1:8]`，在关节空间做比较。
3. **（补 A1-i / A2 共同缺口）**：需要作者明确说明 `lingbot-va-posttrain-libero-long` 的实际训练集，包括是否只用了 `libero-long-lerobot`、是否联合了 4 个 suite、是否做过成功过滤。办法是在 `Robbyant/lingbot-va` 提 issue，或查找 HF 模型卡后续更新。这是唯一无法靠本地核查解决的缺口。
4. **（预训练侧，超出本次范围）**：要把 unverified-PT 进一步变成 known-via-pretraining 或 unknown，需要核实 16K 小时预训练数据（尤其是 InternData-A1 仿真数据和 OXE 子集）中有没有 LIBERO 或 LIBERO 风格的场景。

## 局限（2026-10-14 补记）

- **检查点训练集没有官方声明**：`lerobot/lingbot_va_libero_long` 是否只用 `robbyant/libero-long-lerobot` 训练，没有任何官方说明。训练配置中的 `dataset_path` 是占位符；论文 §4.3.2 说用了四个套件并过滤了失败演示，与该数据集 500 条未过滤相矛盾。本文件"升级为 unverified-PT"的结论建立在数据集指纹匹配与 LIBERO 采集代码之上，这一缺口如实保留。
- 后续可以给作者发邮件确认检查点的实际训练集，但不阻塞本批次。
