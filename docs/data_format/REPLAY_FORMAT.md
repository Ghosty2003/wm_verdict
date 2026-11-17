# replay_frames 格式说明（集群侧，2026-11-14）

- 生成脚本：`replay_frames.py`，即本目录这一份，是集群 `libero/rerun/replay_frames.py` 的原样拷贝。SHA256 为 `5c0f5cdf2f0c80a51365ccb5693b1307ca031f49747699f556e9bea9d4151953`。
- 依赖的仓库内模块（不在本目录）：`libero_env_adapter._make_env`、`phys/phys_perturb.py`（`spec_of`、`apply_physics`、`maybe_push`）。
- 用法：`python replay_frames.py list | <shard> <nshards>`，lerobot_libero 解释器，`MUJOCO_GL=egl`。

## 做什么
- 输入是已有闭环动作、但当时没存真实画面的单元。在模拟器里按原样重放第一遍闭环动作，补出每一步的真实画面。
- 不调用模型。
- 环境与原 runner 一致，都是 LeRobot LiberoEnv，经 `_make_env`，每个单元新建一个 env。
- 扰动的处理：
  - 物理扰动（Em/Ef/G）在重放前施加；
  - K 推动在同一步施加；
  - 动作取已执行的动作（`executed`），所以 N 族不再重复施加动作流。

## 文件命名
| 内容 | 路径 |
|---|---|
| 逐单元画面 | `libero/out/runs/<run>/replay_frames/<unit_id>.npz` |
| 逐单元比对记录 | `libero/out/runs/<run>/replay_frames.<shard>.jsonl`，每单元一行，追加写 |
| 写入方式 | 先写 `<unit_id>.tmp.npz` 再原子改名；已有 `<unit_id>.npz` 的单元直接跳过 |

`<run>` 的取值如下（与原记录目录同名）：
- `lingbot_{base,famA,famP,famO,famB,phys}`
- `lingbot_t{2,4,6}/famB`
- `openwam_{base,famA,famP,famO,famB,phys}`
- `openwam_t{2,4,6}`
- `openwam_t{2,4,6}/famB`
- `openwam_t{4,6}/phys`
- `openwam_t{2,4,6}/rerun_closed`

分片：`zlib.crc32(f"{run}/{unit_id}") % nshards == shard`。卡片单元优先，见 `card_first`，种子 20261112，每格 5 个。

## npz 键（`np.savez_compressed`）
设 n = n_actions。

| 键 | 形状 / 类型 | 含义 |
|---|---|---|
| `agentview` | [n, R, R, 3] uint8 | 主视角相机 `agentview_image`（LeRobot 映射名 `pixels/image`），**每一步执行后**的画面：第 i 帧对应第 i+1 步之后。**不含第 0 步的初始画面** |
| `wrist` | [n, R, R, 3] uint8 | 腕部相机 `robot0_eye_in_hand_image`（`pixels/image2`），时间对齐同上 |
| `success` | [n] bool | 每步执行后 `env._env.check_success()` 的值 |
| `states` | [n+1, D] float64 | MuJoCo `mjSTATE_FULLPHYSICS` 全物理状态；`states[0]` 是初始状态，`states[i]` 是第 i 步之后 |
| `actions` | [n, 7] float32 | 实际重放的动作，即原记录里的 `executed` 或 `closed_loop` 动作 |

- **分辨率 R**：LingBot 128，OpenWAM 256，B 族（两个模型都是）128。与原 runner 的相机分辨率一致。jsonl 的 `res` 字段记录了 R。
- **翻转**：存的是 LIBERO 原始渲染，**没有翻转**（画面上下颠倒）。要得到正视画面，只需上下翻转：`frames[:, ::-1]`。不要转 180°，否则左右会镜像。
- **是否每一步都存**：是。每个执行步都存一帧主视角、一帧腕部和一个状态。

## n_actions：全程还是截到首次成功
n_actions 等于原闭环动作文件里的动作数，脚本不做截断，所以长度取决于各模型 runner 自身的停机规则：

| 模型 | 成功单元 | 失败单元 |
|---|---|---|
| LingBot-VA | 截到首次成功（runner 成功即停）：n = 首次成功步 | 跑满时长：新任务（T2–T4）及 phys/famB 为 400；T0/T1 的 base/famA/famO/famP/famB 老批次是按 H=520 跑的，n > 400（论文口径按 §28 事后截到 400，回放不截） |
| OpenWAM-α | **不截断**，成功后继续跑满时长（n = 400，个别批次为原时长） | 跑满时长（400） |

集群侧已有的 21635 行回放记录（LingBot 7550，OpenWAM 14085）的实测分布与上表一致。

## jsonl 每行字段
- `unit_id`, `run`, `model`, `task`（t2…）, `level`, `card_unit`
- `n_actions`, `res`
- `replay{success, first_success_step}`
- `recorded{success, first_success_step}`（原记录）
- `matches_recorded_success`, `matches_recorded_step`
- `frames_file`（相对仓库根目录）
- `seconds`, `machine`

`first_success_step` 等于 success 首个 True 的下标加 1。

**已知不一致**：重放与原记录不符的只出现在 OpenWAM 几批里。

| 批次 | 不符数 |
|---|---|
| t2/famB | 5 / 500 |
| t4/famB | 6 / 500 |
| t6/famB | 19 / 500 |
| t4/phys | 9 / 1724 |
| t6/phys | 6 / 1834 |

LingBot 全部 7550 行都相符。

## 回放范围裁决（2026-11-14）
- **LingBot 动作文件共 6300 个**：
  - 已回放 3850 个：T0/T1 的 base/famA/famO/famP/famB 共 2500 个，加 T2–T4 famB 1350 个。用的都是 `closed_loop` 闭环动作。
  - 其余 2450 个处理如下。
- **不回放：T2–T4 的 base/famA/famO/famP 共 2400 个。**
  - 这些动作文件只有 `env` 一个键，存的是开环想象出来的动作，回放出来是开环真值画面，不是闭环画面。
  - 口径与开环零校准补测的 D/E/F 相同：附录用不到，不补。
  - 这几条线的闭环动作和画面，由集群上的 LingBot 全量重跑（`libero/rerun/lb_rerun.py`，protocol.yaml 中的 `runs.lingbot_tx_closed_full_rerun`）产出，不靠回放。
- **4090 只补 T2 famB_bridge 的 50 个闭环单元。**
  - 动作文件：`lingbot_t2/famB_bridge/actions/<uid>.npz`，键 `closed_loop`（另有第二遍 `closed_loop2`）。回放第一遍 `closed_loop`。
  - 输出按上文格式写到 `lingbot_t2/famB_bridge/replay_frames/`。
  - 补完转入待命。
- 逐单元清单见 `cluster_lingbot_action_units.tsv`，`decision` 列分别标 `replayed` / `skip_open_loop_only (D/E/F)` / `4090_replay_famB_bridge`。
- famB_bridge 不在 `replay_frames.py` 的 `jobs()` 枚举里，需要按同一格式补一个条目：
  - `run = "lingbot_t2/famB_bridge"`；
  - `bank = libero/out/bank/libero_anchors_t2.json`，`anchor_id` 取 records 行的 `anchor_id`；
  - `res = 128`；
  - `act = (lingbot_t2/famB_bridge/actions/<uid>.npz, "closed_loop")`；
  - `setting = None`；
  - `rec = records 行的 closed_loop`（5090 第一遍，50 个单元中 49 个成功）。

  逐单元比对时，`matches_recorded_*` 应当 50/50 一致。
