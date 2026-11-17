# LIBERO 原生扰动 / 变化钩子清单（面向 world-model verdict benchmark）

> **口径（2026-10-14）**：本阶段只在虚拟环境内评测。文中的“真实”“现实”“real”一律指 **LIBERO 模拟器真值分支**，“迁移”指想象结论到模拟器执行的迁移。物理机器人不在范围内。

> 调研日期：2026-10-14。本地 **未安装** LIBERO / robosuite / mujoco（`pip show` 与 `importlib.util.find_spec` 均为空）。
> 以下结论全部来自**只读**源码：
>
> - 上游 LIBERO：`github.com/Lifelong-Robot-Learning/LIBERO`，浅克隆 commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`（2025-03-15）。下文 `LIBERO:` 前缀的路径相对仓库根目录。
> - robosuite：LIBERO / hf-libero 都钉死 `robosuite==1.4.0`，因此读的是 tag `v1.4.0`。下文 `robosuite:` 前缀的路径相对 `robosuite/` 包目录。
> - LeRobot 封装：`$LEROBOT_ROOT/src/lerobot/envs/libero.py`（下文记为 `lerobot:libero.py`）与 `src/lerobot/envs/configs.py`。
> - hf-libero 0.1.4 wheel（PyPI），仅解包后做 diff，未安装、未执行。
>
> 通道分类沿用我们的 taxonomy：**观测 / 动作 / 动力学 / 初始状态**。
> “WM 首帧可见”指：一个以第一帧观测为条件的 world model，能否从初始帧（或初始帧 + 语言）看出这个扰动。

---

## 0. hf-libero 与上游的差异（结论：判定相关逻辑无差别）

- PyPI `hf-libero` 0.1.4（2026-06-10 发布），主页 `github.com/huggingface/LIBERO`，在 GitHub 上是 `Lifelong-Robot-Learning/LIBERO` 的 fork。依赖钉死 `robosuite==1.4.0`、`bddl==1.0.1`、`mujoco>=3.0,<3.9`、`gymnasium>=0.29`。LeRobot 的 extra 为 `hf-libero>=0.1.4,<0.2.0`（`lerobot/pyproject.toml:289`）。
- 对 `libero/libero/` 做 `diff -rq`（wheel 不含 `assets/`）：
  - `init_files/`、`bddl_files/`、`envs/env_wrapper.py` **完全相同**。
  - `benchmark/__init__.py`：唯一改动是 `torch.load(..., weights_only=False)`（适配 PyTorch ≥2.6）。
  - `envs/bddl_base_domain.py`、`envs/objects/*.py`、`__init__.py`、`utils/download_utils.py`：资产路径改为 `get_assets_path()`。本地没有资产时，会从 HF Hub 的 `lerobot/libero-assets` 下载。
  - `envs/venv.py`：`gym` 改为 `gymnasium`。
- 因此下面所有 LIBERO 行号对 hf-libero 同样适用，只有 `bddl_base_domain.py` 因多一行 import 而整体后移约 1–2 行。

---

## 1. 初始状态机制

### 1.1 固定 init_states 文件

- **位置**：`LIBERO: libero/libero/init_files/<suite>/<task>.pruned_init`。各 suite 另有 `<task>.init`（libero_goal 没有 `.init`）。
- **使用哪一种**：`benchmark/__init__.py:76` 写死 `init_states_file=f"{task}.pruned_init"`。LIBERO 原生评测、OpenVLA 和 LeRobot 都用 `.pruned_init`。
- **格式**：`torch.save` 保存的 zip 包，里面的 `data.pkl` 是一个 numpy `float64` ndarray。我用 `pickletools` 静态反汇编确认，没有反序列化执行。
  - `.pruned_init` 的形状为 **(50, D)**，所有 130 个任务都是 **每个任务 50 个**。
  - `.init` 的形状为 (100, D)。
- **向量内容**：`MjSim.get_state().flatten()`，即 **`[time, qpos..., qvel...]`**（`robosuite: utils/binding_utils.py:240-241`）。
  - D = 1 + nq + nv，例如 libero_spatial 为 92，libero_object 为 110，libero_goal 为 79，libero_10/90 在 45–123 之间不等。
  - 手工核对：spatial 任务 = Panda（7 臂 + 2 指）+ 5 个自由物体（7 qpos / 6 qvel）+ 柜子 3 抽屉 + 炉子 1 旋钮，得到 qpos 48、qvel 43，加上 1 位 time 正好 92；KITCHEN_SCENE3 = 9 + 2×7 + 1 → 24 / 22，加 1 得 47。两者都吻合。
  - 第 0 列 time 为 0。
  - **不含 act、ctrl、qacc_warmstart、mocap、model 参数**（见第 5 节）。
- **`set_init_state`**（`LIBERO: libero/libero/envs/env_wrapper.py:136-145`）调用 `regenerate_obs_from_state`，依次执行：
  1. `sim.set_state_from_flattened`：只写 `data.time / qpos / qvel`，见 `robosuite: utils/binding_utils.py:1157-1169`；
  2. `sim.forward()`；
  3. `_check_success()`；
  4. `_post_process()`：更新炉火等可视化 site；
  5. `_update_observables(force=True)`。

  它**不重置**控制器、夹爪积分器、`timestep` 计数器或 RNG。
- **其它入口**：`get_sim_state()` / `set_state()` 位于 `env_wrapper.py:118-128`；`benchmark.get_task_init_states(i)` 位于 `benchmark/__init__.py:158-165`。

### 1.2 原生随机摆放（不使用 init_states 时）

`env.reset()` 每次都会通过 placement sampler 重新采样，`deterministic_reset=False`（`LIBERO: envs/bddl_base_domain.py:731-777`）。

- **采样器构建**：`_add_placement_initializer`（`bddl_base_domain.py:562-729`）按 BDDL 的 `:init` 谓词构建：
  - `(On obj <table>_<region>)` 且 obj 是可动物体：使用 `get_region_samplers(problem, workspace)`，即 `TableRegionSampler` 或 `Libero100TableRegionSampler`（`envs/regions/__init__.py:14-26`）。在 region 的 `:ranges` 矩形内做均匀 xy 采样；旋转使用物体类自带的 `rotation / rotation_axis`，例如 `google_scanned_objects.py:33-34`。
  - `(On fixture region)`：使用 `MultiRegionRandomSampler`，yaw 取 BDDL 的 `:yaw_rotation`，`ensure_valid_placement=False`（`bddl_base_domain.py:602-616`）。
  - `(On obj <fixture>_<site_region>)` 或 `(In obj site_region)`：使用 `SiteRegionRandomSampler` 或 `InSiteRegionRandomSampler`，相对 fixture 的 site 采样（`675-692`、`711-729`）。
  - `(On objA objB)`：使用 `ObjectBasedSampler`，叠放（`694-709`）。
  - `(Open|Close|Turnon|Turnoff obj)`：使用 `OpenCloseSampler` / `TurnOnOffSampler`，在 `default_*_ranges` 内均匀采样关节角（`631-672`，`envs/regions/object_property_sampler.py:84-117`）。
- **RNG**：所有采样都用 **numpy 全局 RNG** `np.random.*`（`envs/regions/base_region_sampler.py:80,94,105,180,...`），没有使用 robosuite 的 `random_state`。`env.seed(s)` 实际就是 `np.random.seed(s)`（`bddl_base_domain.py:162-163`）。
- **采样失败**：抛出 `RandomizationError`，由 `ControlEnv.reset` 无限重试（`env_wrapper.py:90-101`）。
- **关键坑（影响 bit-exact）**：fixture（柜子、炉子、微波炉等）没有 freejoint。它们的采样结果直接写进 **`sim.model.body_pos / body_quat`**（`bddl_base_domain.py:773-777`），**不在 qpos 中**，所以 `set_init_state` 恢复不了。
  - fixture 的 region 一般是约 2 cm 的方块，个别还有约 0.06 rad 的 yaw 范围，例如 spatial 任务的 `cabinet_region`。
  - 因此即使使用固定 init_state，fixture 位姿（以及挂在 fixture 上的目标 site）仍然取决于 reset 前的 `np.random` 状态。
  - OpenVLA 的 `libero_utils.py:24` 也注释了这一现象：“seed seems to affect object positions even when using fixed initial state”。

### 1.3 LeRobot 封装怎么挑 init state

- `get_task_init_states`（`lerobot:libero.py:67-87`）用 `torch.load(weights_only=False)` 读取 `init_states/<suite>/<task>.pruned_init`。如果是 LIBERO-plus，会去掉文件名中的扰动后缀，或者改读 `libero_newobj/`。
- 每个子环境的 `init_state_id = episode_index`（第 k 个并行 env 从第 k 个开始），每次 reset 后加 `n_envs`，取模 50（`libero.py:168-175, 344-346`）。
- `reset()` 流程（`libero.py:339-364`）：
  1. `self._env.seed(seed)`，即 `np.random.seed`。`seed=None` 时随机，因此 fixture 位姿不可复现。`lerobot-eval` 在传入 `--seed` 时，按 `start_seed + batch_ix*n_envs + i` 给每个子环境设种子（`scripts/lerobot_eval.py:546-559`）。
  2. `self._env.reset()`：默认 `hard_reset=True`，重建 MJCF、model 和渲染器，并重新采样。
  3. `set_init_state(init_states[id % 50])`。
  4. 执行 **`num_steps_wait=10`** 步空动作 `[0,0,0,0,0,0,-1]`（`libero.py:90-92, 351-352`），让物体落稳，同时把夹爪积分器推到“全开”。
  5. 设置 `controller.use_delta`（`control_mode`：relative/absolute）。

  对比：LIBERO 原生评测执行 5 步全零动作（`libero/lifelong/metric.py:117-122`）。
- 配置项 `init_states: bool = True`（`configs.py:332`）设为 False 时会跳过第 3 步，直接使用原生随机摆放。

---

## 2. BDDL 场景定义：哪些可编辑、在哪里

BDDL 文件位于 `LIBERO: libero/libero/bddl_files/<suite>/<task>.bddl`，由 `envs/bddl_utils.py:87-171`（`robosuite_parse_problem`）解析。示例见 `libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl`。

| BDDL 段 | 含义 / 可改什么 | 解析 / 使用位置 | 备注 |
|---|---|---|---|
| `(problem LIBERO_<Scene>_Manipulation)` | 选择 problem 类：arena 类型、桌子尺寸、相机、墙地纹理默认值 | `env_wrapper.py:49-56` → `TASK_MAPPING`；`envs/problems/*.py` | 共 6 类：tabletop / kitchen / living_room / study / coffee_table / floor |
| `:language` | 任务语言 | `bddl_utils.py:147-149`；`bddl_base_domain.py:840-842` | **LeRobot 与 LIBERO benchmark 并不读它**：`task.language` 由文件名生成（`benchmark/__init__.py:44-53, 69`；`lerobot:libero.py:180`）。改语言要改 benchmark/Task，或在 wrapper 层替换 `task_description` |
| `:regions` | 每个 region 的 `:target`（桌面或 fixture）、`:ranges`（xmin ymin xmax ymax，可写多个矩形）、`:yaw_rotation`、`:rgba` | `bddl_utils.py:9-44` | `:yaw_rotation` 的值会被 `eval()`（`bddl_utils.py:36`），只能用可信文件 |
| `:fixtures` | 不可移动物体（桌子、柜子、炉子、微波炉…），类别 → 实例 | `bddl_utils.py:127-142`；`problems/*._load_fixtures_in_arena` | 类别必须在 `OBJECTS_DICT` 中注册（`envs/base_object.py`） |
| `:objects` | 可动物体（类别 → 实例），**增删即增减干扰物** | `bddl_utils.py:113-124`；`_load_objects_in_arena` | 会改变 nq/nv，**原 init_states 立即失效**，需要重新生成 |
| `:obj_of_interest` | 只用于分割和可视化（`SegmentationRenderEnv`） | `bddl_utils.py:125-127` | 与判定无关 |
| `:init` | 初始谓词：`On / In / Open / Close / Turnon / Turnoff`，决定采样器 | `bddl_base_domain.py:562-729` | 只在**不使用 init_states** 时生效；全套 BDDL 中 `:init` 有 On×647、Open×22、Close×3、Turnon×3、In×1 |
| `:goal` | 目标谓词（只支持合取） | `bddl_utils.py:154-155`；`problems/*._check_success` | 全套中 goal 有 In×63、On×61、Close×11、Turnon×8、Open×7、Turnoff×1 |
| `:scene_properties` | 本意是指定 floor / wall 风格 | `bddl_utils.py:47-60` | **实际不可用**：`get_scenes` 引用了未定义变量 `region`（第 52 行），解析结果也没有传给 env；现有 BDDL 都没有使用它。墙地纹理需要通过构造参数修改（见第 3 节） |

可用谓词注册表见 `envs/predicates/__init__.py:4-18`（true/false/in/on/up/open/close/turnon/turnoff/printjointstate）。扩展接口为 `update_predicate_fn_dict`。

---

## 3. 其它环境级旋钮（无需重训即可施加）

`OffScreenRenderEnv` / `ControlEnv` 的构造参数见 `env_wrapper.py:12-81`。其中 `**kwargs` 会透传给 problem 类，所以 `scene_xml`、`scene_properties`、`initialization_noise` 等都可以传入。LeRobot 只透传了 `camera_heights/widths`、`control_freq`、`hard_reset`（`lerobot:libero.py:267-275`），其余参数要改 wrapper 才能用。

### 3.1 观测侧

- **相机名**：默认 `camera_names=["agentview","robot0_eye_in_hand"]`（`env_wrapper.py:31-34`）。LeRobot 使用 `agentview_image,robot0_eye_in_hand_image`，并映射为 `image / image2`（`lerobot:libero.py:116, 156-160`）。
  - 场景中另有 `frontview`、`galleryview`、`paperview`，以及 tabletop 的 `canonical_agentview`（`problems/libero_kitchen_tabletop_manipulation.py:188-217`、`bddl_base_domain.py:274-295`）。
- **相机位姿**：
  - 写死在各 problem 类的 `_setup_camera()`（`mujoco_arena.set_camera(name,pos,quat)`）。例如 agentview 的 pos 为 `[0.6586, 0, 1.6104]`，见 `libero_tabletop_manipulation.py:187-...` 和 kitchen 版 `188-198`。
  - 运行时可以直接改 `sim.model.cam_pos / cam_quat / cam_fovy`，或用 `robosuite: utils/camera_utils.py:246` 的 `CameraMover.set_camera_pose`，或用 `utils/mjmod.py:518` 的 `CameraModder`。
  - 腕部相机定义在 robosuite 的 Panda 夹爪 XML 中。
- **分辨率**：`camera_heights / camera_widths`。LIBERO 默认 128，LeRobot 配置默认 360（`configs.py:334-335`），wrapper 默认 256，OpenVLA 用 256。
- **其它渲染输出**：`camera_depths`、`camera_segmentations`（`SegmentationRenderEnv`）。
- **图像方向**：图像为 OpenGL 方向（上下颠倒，`robosuite: macros.py:28`），LeRobot 在 `processor/env_processor.py:~59` 做 180° 翻转。
- **光照**：写在场景 XML 的 `<light>` 中，例如 `assets/scenes/libero_tabletop_base_style.xml:49-50`、`libero_kitchen_tabletop_base_style.xml:51,71-72`。原生**没有参数入口**，可以换 `scene_xml`、运行时改 `sim.model.light_*`，或用 `robosuite: utils/mjmod.py:62` 的 `LightingModder`。
- **墙地纹理**：
  - 构造参数 `scene_properties={"floor_style":..., "wall_style":...}`（各 problem 类，如 `libero_kitchen_tabletop_manipulation.py:28-40`）。可选值在 `envs/arenas/style.py:1-25`：地板 8 种，墙面 9 种。
  - 构造参数 `scene_xml` 可选 `assets/scenes/` 下的 `*_blue_style.xml` 和 `*_warm_style.xml`（tabletop / floor / coffee_table 有这些变体）。
  - 运行时也可以用 `robosuite: utils/mjmod.py:782` 的 `TextureModder`，或改 `sim.model.geom_rgba / mat_*`。
- **物体外观**：物体网格和纹理在 `assets/stable_scanned_objects` 等目录，原生不随机。可以使用 `TextureModder`；也可以设置 `robosuite.macros.USING_INSTANCE_RANDOMIZATION`（`macros.py:16`），这需要在建模型之前设置。
- **传感器 / 观测噪声**：LIBERO 本身**没有**。robosuite 的 `Observable` 支持 `corrupter / delayer`（`robosuite: utils/observables.py:43-125`，有 gaussian / uniform / deterministic 等），通过 `env.env.modify_observable(name, "corrupter", fn)` 设置（`environments/base.py:631`）。图像 observable 也能挂。注意 hard reset 会重建 observables（`base.py:262-266` 只恢复 sensor），需要在每次 reset 后重新挂载。

### 3.2 动作侧

- **控制器类型**：`ControlEnv(controller="OSC_POSE")`（`env_wrapper.py:17, 47`），也可以换成 robosuite 1.4 的 `OSC_POSITION / JOINT_POSITION / JOINT_VELOCITY / IK_POSE` 等。LIBERO 的 `step` 对 4 维动作做了特殊处理（`bddl_base_domain.py:800-804`）。换类型会改变动作语义，策略必然失配。
- **控制器参数**：`robosuite: controllers/config/osc_pose.json`：
  - `kp=150`、`damping_ratio=1`、`impedance_mode="fixed"`；
  - `output_max=[0.05]*3 + [0.5]*3`（每步最大 5 cm / 0.5 rad）；
  - `control_delta=true`、`interpolation=null`、`uncouple_pos_ori=true`。

  修改方式：load 之后改 config dict 再构造 env；或运行时修改 `env.robots[0].controller.kp/kd/output_max/output_min`。
- **增量 / 绝对**：LeRobot 的 `control_mode`（`libero.py:354-361`）会改写 `controller.use_delta`。
- **控制频率**：`control_freq=20`（`env_wrapper.py:27`，LeRobot 的 `fps=20`，见 `configs.py:326`）。每个控制步包含 `int((1/control_freq)/0.002)` 个 0.002 s 物理子步，20 Hz 时为 25 步（`robosuite: environments/base.py:178-184, 388-394`）。
- **夹爪**：`PandaGripper.format_action` 是**有状态积分器**，每个物理子步 `current_action += ±0.01·sign(a)`，钳位在 [-1, 1]（`robosuite: models/grippers/panda_gripper.py:43-62`）。`speed=0.01` 是 property，改它需要 monkeypatch。
- **动作噪声 / 延迟**：原生不提供，只能在 wrapper 层实现。

### 3.3 动力学侧

- **物体物性**：写在资产 XML 中，例如 `assets/stable_scanned_objects/*/…xml` 的 `density="100" friction="0.95 0.3 0.1" solref="0.001 1" solimp="0.998 0.998 0.001"`。
- **桌面摩擦**：`table_friction=(0.6, 0.005, 0.0001)`，写死在 `bddl_base_domain.py:312`。
- **机器人关节阻尼**：`MountedPanda` 的 `damping=(0.1×5, 0.01×2)`（`envs/robots/mounted_panda.py:17-20`）。
- **以上物性都没有构造参数**。运行时可以改 `sim.model.geom_friction / body_mass / body_inertia / dof_damping / geom_solref ...`，或使用 `robosuite: utils/mjmod.py:1405` 的 `DynamicsModder`（支持 mass / inertia / friction / solref / solimp / stiffness / frictionloss / damping / armature）。
- **仿真步长**：`robosuite.macros.SIMULATION_TIMESTEP=0.002`（`macros.py:11`）。
- **robosuite 的 `DomainRandomizationWrapper`**（`wrappers/domain_randomization_wrapper.py:119-131`）可以一次性随机化 color / camera / lighting / dynamics，需要包装 `ControlEnv.env`。
- **与 hard reset 的交互**：所有对 `sim.model` 的运行时修改都会在 **hard reset 时丢失**（model 会重建），必须在 `reset()` 之后、`set_init_state` 之前或之后重新施加。

### 3.4 初始状态侧

- **机器人初始关节**：
  - `MountedPanda.init_qpos=[0,-0.161,0,-2.445,0,2.227,π/4]`（`mounted_panda.py:34-38`）；
  - 夹爪 `init_qpos=[0.020833,-0.020833]`（`panda_gripper.py:25-26`）；
  - `initialization_noise` 在 `ControlEnv` 中默认 **None，即无噪声**（`env_wrapper.py:19`；`robosuite: robots/robot.py:64-68, 129-141`），可以传 `{"type":"gaussian","magnitude":σ}`。

  **但是**，使用 init_states 时，机器人 qpos 会被 init 向量覆盖，因此要扰动机器人初始位姿，必须直接改 init 向量中前 9 个 qpos。注意 `controller.initial_joint`（OSC 零空间目标，`robosuite: controllers/osc.py:351`）是在 reset 时按 `init_qpos` 设置的，`set_init_state` 不会更新它。
- **物体布局**：可以选不同 init_state 索引（0–49），可以直接编辑 init 向量中物体 freejoint 的 qpos，也可以改 BDDL 的 region 后用原生采样（`init_states=False`）。
- **关节物体初值**：抽屉、微波炉门、炉子旋钮的 qpos 在 init 向量中；原生采样范围在各物体的 `default_*_ranges`，例如 `objects/articulated_objects.py:238-239`。
- **seed**：通过 `np.random` 决定 fixture 位姿，以及不用 init_states 时的整个布局。
- **settle 步数**：LeRobot 的 `num_steps_wait`（默认 10）会改变首帧观测。

---

## 4. 旋钮 → 通道分类与“WM 首帧可见性”

| # | 旋钮 | 原生入口 | 通道 | WM 首帧可见？ | 备注 |
|---|---|---|---|---|---|
| 1 | init_state 索引（50 选 1） | `.pruned_init` + `set_init_state` | 初始状态 | **是** | 物体位置和机器人位姿都体现在首帧 |
| 2 | 编辑 init 向量（物体 / 关节物体 qpos） | `set_state` | 初始状态 | **是** | 需保证无穿插；之后应 settle |
| 3 | 机器人初始关节（init 向量或 `init_qpos` / `initialization_noise`） | 同上 / `robot.py:129-141` | 初始状态 | **是**（agentview 与腕部相机） | 用 init_states 时 `initialization_noise` 无效 |
| 4 | BDDL region ranges / yaw | `.bddl :regions` | 初始状态 | **是** | 需 `init_states=False` 或重新生成 init 文件 |
| 5 | 增删物体 / 干扰物 | `.bddl :objects` | 初始状态（+观测） | **是** | 改变 nq/nv，原 init 失效 |
| 6 | fixture 种类 / 位置 | `.bddl :fixtures / :regions` | 初始状态（+动力学） | **是**（小幅位移只有微弱可见性） | fixture 位姿存在 model 中，不在 state 中 |
| 7 | seed（`np.random`） | `env.seed` | 初始状态 | 弱 | 用 init_states 时只影响 fixture 约 ±1 cm 的位姿 |
| 8 | settle 步数 | LeRobot `num_steps_wait` | 初始状态 | 弱 | |
| 9 | 关节物体初值（开 / 关 / 点火） | `.bddl :init` / init 向量 | 初始状态 | **是**（炉火 site 也可见） | |
| 10 | 相机位姿 / FOV | `_setup_camera`、`sim.model.cam_*`、`CameraModder` | 观测 | **是**（首帧直接改变） | 不影响物理 |
| 11 | 相机选择 / 分辨率 | `camera_names`、`camera_heights/widths` | 观测 | **是** | |
| 12 | 光照 | 场景 XML `<light>`、`LightingModder` | 观测 | **是** | |
| 13 | 墙 / 地纹理、场景风格 | `scene_properties`、`scene_xml`、`TextureModder` | 观测 | **是** | |
| 14 | 物体颜色 / 材质 | `TextureModder`、`geom_rgba` | 观测 | **是** | |
| 15 | 观测噪声 / 延迟 | robosuite `Observable` corrupter / delayer | 观测 | **是**（作用于首帧时）；延迟在首帧不可见 | hard reset 后需重新挂载 |
| 16 | 语言改写 | `Task.language` / wrapper | 观测（语言条件） | 帧中不可见；只有 WM 以语言为条件时可见 | BDDL `:language` 对 LeRobot 无效 |
| 17 | 控制器类型 | `ControlEnv(controller=...)` | 动作 | **否** | 改变动作语义 |
| 18 | 控制器增益 / 输出限幅 | `osc_pose.json`、`controller.kp/kd/output_max` | 动作 | **否** | 只在 rollout 中显现 |
| 19 | 增量 / 绝对模式 | LeRobot `control_mode` | 动作 | **否** | |
| 20 | 控制频率 | `control_freq` | 动作（时间尺度） | **否** | 每步子步数改变 |
| 21 | 夹爪速度 / 积分器 | `PandaGripper.speed` | 动作 | **否** | |
| 22 | 动作噪声 / 延迟 / 缩放 | 无原生入口（需 wrapper） | 动作 | **否** | |
| 23 | 物体质量 / 摩擦 / solref | XML、`sim.model.*`、`DynamicsModder` | 动力学 | **否** | |
| 24 | 桌面摩擦 | `bddl_base_domain.py:312` / `geom_friction` | 动力学 | **否** | |
| 25 | 关节阻尼 / armature | `mounted_panda.py:17-20` / `dof_*` | 动力学 | **否** | |
| 26 | 仿真步长 | `macros.SIMULATION_TIMESTEP` | 动力学 | **否** | 同时改变控制子步数 |
| 27 | goal 谓词 | `.bddl :goal` | 不属于四个通道（判定标签） | **否** | 只改变成功标签，不改变世界 |
| 28 | 最大步数 | LeRobot `episode_length` | 不属于四个通道（预算） | **否** | |

---

## 5. 精确保存 / 恢复（bit-exact replay）

**已有 API**：

- `ControlEnv.get_sim_state()` 返回 `sim.get_state().flatten()`，即 `[time, qpos, qvel]`（`env_wrapper.py:118-119`）。
- `set_state(x)` 等价于 `sim.set_state_from_flattened(x)`（`env_wrapper.py:127-128`）。
- `set_init_state(x)` = `set_state` + `forward` + `_post_process` + `_update_observables`（`env_wrapper.py:136-145`）。
- 底层 `sim.get_state()` 返回 `MjSimState(time, qpos, qvel)`（`robosuite: utils/binding_utils.py:213-241, 1140-1169`）。
- `env.env.sim` 暴露了原生 `mujoco` 的 `model` 和 `data`（`_model` / `_data`），因此也可以直接调用 `mujoco.mj_getState(model, data, mjSTATE_*)`。LIBERO 和 robosuite 1.4.0 本身都没有封装这个接口。
- `reset_from_xml_string(xml)` 可以从 XML 确定性重建（`env_wrapper.py:130-131`；`robosuite: environments/base.py:539-560`，会置 `deterministic_reset=True`，跳过采样）。

**flattened state 中缺失的部分**（逐项都会破坏 bit-exact 重放）：

1. **MuJoCo 求解器 warmstart（`data.qacc_warmstart`）**：不在 state 中。恢复后第一步的约束求解初值不同，接触多时会产生 ~1e-10 量级起步、随后放大的偏差。需要用 `mj_getState(..., mjSTATE_INTEGRATION)`，或者把 `qacc_warmstart` 一并存取。
2. **`data.act / ctrl / qfrc_applied / xfrc_applied / mocap_*`**：不在 state 中。
   - `ctrl` 在每个子步 `_pre_action` 中会被重写（`base.py:388-391`），影响较小；
   - Panda 的执行器在常规配置下没有激活状态（na=0，从 D = 1+nq+nv 的吻合可以印证），所以 `act` 也是空的；
   - 其余三项正常运行时都是 0。
3. **model 级参数**：
   - **fixture 的 `body_pos / body_quat`**：reset 时由 `np.random` 采样（`bddl_base_domain.py:773-777`），挂在 fixture 上的目标 site 位置也随之变化，因此会影响 `_check_success`；
   - 任何 Modder 或手动修改的摩擦、质量、光照、相机、纹理；
   - 炉火等 `site_rgba`：这一项会由 `_post_process` 根据 qpos 重算（`problems/*:set_visualization`），所以可以恢复。
4. **夹爪积分器 `PandaGripper.current_action`**：Python 侧状态（`panda_gripper.py:53-58`），每个子步累加 ±0.01。只恢复 qpos 时，指令夹爪开度会跳变。
   - hard reset 时它会随 gripper model 重建而归零；
   - `hard_reset=False` 时它会跨 episode 残留；
   - LeRobot 的 10 步 `-1` settle 会把它推到饱和的“全开”。
5. **OSC 控制器内部状态**：
   - `goal_pos / goal_ori` 在每个 policy step 的 `set_goal` 中由当前 eef 位姿重新计算（`osc.py:202-276`，`interpolation=null`），基本无记忆；
   - 但 `initial_joint`（零空间目标，`osc.py:351`，`base_controller.py:91`）、`use_delta`、`kp/kd` 是 Python 状态，不在 state 中。
6. **环境计数器**：`timestep / cur_time / done`（`base.py:302-305, 378, 396, 429`）。`horizon=1000` 到达后，`step` 会抛出 “executing action in terminated episode”。
7. **RNG**：`np.random` 全局状态（LIBERO 采样器、robosuite 初始化噪声、Modder 默认的 `random_state` 都用它，见 `mjmod.py:35-39`）不在 state 中，需要用 `np.random.get_state()` 另存。
8. **observable 缓冲**：`_obs_cache`，以及挂了 delayer / corrupter 时的内部缓冲和随机数。
9. **渲染**：给定 model + data 时渲染是确定的，但 EGL 与 OSMesa 之间、不同 GPU 驱动之间可能有像素级差异。

**建议的“完整快照”**：

- `mj_getState(mjSTATE_INTEGRATION)`（含 qacc_warmstart、act、ctrl 等）；
- 加上 fixture 的 `model.body_pos/quat`，或者固定 seed 后重新 hard reset；
- 加上 `gripper.current_action`；
- 加上 `controller.initial_joint / use_delta`；
- 加上 `env.timestep`；
- 加上 `np.random.get_state()`；
- 加上所有 model 修改的记录。

最稳妥的做法是：同一 seed 加同一 init 索引做 hard reset，再从头重放动作序列，而不是在中途恢复。

---

## 6. 成功判定与最大步数

- **判定**：`ControlEnv.check_success()` → `env._check_success()`（`env_wrapper.py:103-104`）。各 problem 类的实现相同（例如 `problems/libero_kitchen_tabletop_manipulation.py:136-163`）：对 BDDL `:goal` 中**所有谓词取合取**，通过 `eval_predicate_fn` 查 `VALIDATE_PREDICATE_FN_DICT`（`predicates/__init__.py:4-27`）。
  - `On(a,b)`：`b.check_ontop(a)`。物体对物体时，要求 a 的 z ≤ b 的 z、两者有接触、xy 距离 < 3 cm（`object_states/base_object_states.py:78-94`）。目标是 site 区域时，使用 site 的 `under()` 与父物体接触判断（`:178-200`）。
  - `In(a,b)`：`b.check_contact(a) and b.check_contain(a)`。对 site 区域，site 的 contact 恒为 True，再加 `in_box` 的点在盒内检查（`:159-176`；`objects/site_object.py:35-...`）。
  - `Open / Close / TurnOn / TurnOff`：关节 qpos 与各物体 `default_*_ranges` 阈值比较（`objects/articulated_objects.py`，例如 `65-73`、`241-266`）。
  - `Up`：z ≥ 1.0（`predicates/base_predicates.py:79-81`）。
- **step / done / reward 语义**：
  - LIBERO 的 `step` 把 `done` 改写为 `_check_success()`（`bddl_base_domain.py:800-809`）；
  - reward 为稀疏奖励，成功为 1（`165-189`）；
  - LeRobot 的 `terminated = done or is_success`，`info["is_success"]` 来自 `check_success()`（`lerobot:libero.py:374-385`）。
  - 判定**只看当前帧状态**（瞬时），不要求保持。
  - 依赖 fixture 的 site 位姿，因此受第 5 节第 3 条影响。
- **最大步数**（不含 settle 步）：

| suite | LeRobot `TASK_SUITE_MAX_STEPS`（`lerobot:libero.py:98-104`） | OpenVLA `run_libero_eval.py:174-182` | LIBERO 原生 `configs/eval/default.yaml` |
|---|---|---|---|
| libero_spatial | 280 | 220 | 600（统一） |
| libero_object | 280 | 280 | 600 |
| libero_goal | 300 | 300 | 600 |
| libero_10 | 520 | 520 | 600 |
| libero_90 | 400 | 400 | 600 |

  - LeRobot 中，未知 suite 默认 500，可以用 `env.episode_length` 覆盖（`libero.py:188-193`）。
  - OpenVLA 的循环上限是 `max_steps + num_steps_wait(10)`。
  - LeRobot 的 10 步 settle 放在 `reset()` 内，不计入步数。
  - LIBERO 原生评测使用 5 步全零动作 settle，`n_eval=20`。
  - 底层 robosuite 的 `horizon=1000`（`env_wrapper.py:28`）是硬上限。

---

## 7. LIBERO-plus（未安装，仅列出供参考）

- 论文：*LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models*，arXiv 2510.13626。仓库：`github.com/sylvestf/LIBERO-plus`。资产在 HF `Sylvest/LIBERO-plus`。
- 共 10,030 个任务变体，覆盖与 LIBERO 相同的 5 个 suite。变体到维度和难度等级的映射位于 `libero/libero/benchmark/task_classification.json`。
- **7 个扰动维度**，括号内为我们的 taxonomy 映射：
  1. 物体布局：干扰物、目标物体位移（初始状态）
  2. 相机视角：位置、朝向、FOV（观测）
  3. 机器人初始状态（初始状态）
  4. 语言指令：LLM 改写（观测 / 语言）
  5. 光照：强度、方向、颜色、阴影（观测）
  6. 背景纹理（观测）
  7. 传感器噪声：光度畸变、图像退化（观测）

  它**没有动作通道和动力学通道的扰动**。
- LeRobot 已有 `is_libero_plus` 开关（`configs.py:336`；`lerobot:libero.py:60-87`）。变体编码在 init 文件名后缀中（`_language_*`、`_view_*`、`_light_*`、`_table_N`、`_tb_N`；新增物体的变体用 `_add_`、`_level`，文件位于 `libero_newobj/`）。
- 安装 LIBERO-plus 会**替换** hf-libero，两者不能共存。LeRobot 文档建议搭配 `robosuite==1.4.1`（`docs/source/libero_plus.mdx`）。

---

## 8. 实测发现：同一环境对象反复硬重置会让 fixture 位姿漂移（2026-10-14）

- 现象：KITCHEN_SCENE4 同一 seed、同一 init 索引反复 `reset`，`qpos` 每次相同，但 `white_cabinet_1_main`、`wine_rack_1_main` 的 `model.body_pos` 每次不同（约 0.5–1.8 cm）。STUDY_SCENE1 不受影响。
- 根因：`bddl_base_domain.py` 第 653/674 行在每次硬重置（`_load_model`）时把 OpenClose / TurnOnOff 采样器追加进 `self.object_property_initializers`，从不清空。于是每多重置一次，摆放之前就多一次 `np.random.uniform`，全局 RNG 状态随之漂移，fixture 位姿随之改变。fixture 位姿在 model 里，不在 init state 向量中，`set_init_state` 拉不回来。
- 受影响的任务：BDDL `:init` 中含 Open/Close/Turnon/Turnoff 谓词的任务。
- 处理：锚恢复一律新建 `LiberoEnv` 再 `reset(seed)`（`anchor_bank.py`），列表长度固定，逐位可复现。LeRobot 的向量化评测会在同一环境对象上反复重置，因此官方评测在这类任务上，各回合的 fixture 位姿实际上并非由 seed 唯一决定。


## 9. 实测发现：`initialization_noise` 的类型会改变物体摆位（2026-10-15）

- robosuite 每次 reset 都用全局 `np.random` 抽取机器人初始噪声（`robosuite: robots/robot.py:63-71, 129-141`）。`initialization_noise=None` 被解释为高斯、幅度 0，照样抽取。这一步发生在物体摆位采样之前。
- 把类型换成 `uniform`，全局 RNG 的抽取次序随之改变（`np.random.uniform` 与 `np.random.randn` 消耗随机数的方式不同）。所以即使幅度为 0，同一种子下的物体摆位也与原生路径不同。
  - 实测：t5 单一种子 2.1 cm，5 个种子 1.2–4.4 cm；t3 单一种子 1.2 cm。
  - 采样器本身仍是原生的，摆位仍落在原生 BDDL 区域内。
  - 所有 `uniform` 档彼此之间的摆位相同。
- **用法上的含义**：用 `initialization_noise` 做位姿扰动时，零扰动档必须用同一噪声类型、幅度 0（族 P 的 P0），不能拿 `None` 当零点。sham 的语义改为同构自一致：同一种子生成两遍，逐位一致。
