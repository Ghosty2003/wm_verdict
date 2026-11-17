# F 族（单次全帧置黑）健康检查报告 · 2026-11-12

结论：LingBot-VA 和 OpenWAM-α 两个模型的检查**全部通过**（`lingbot/hc_lingbot.json`、`openwam/hc_openwam.json` 中 `PASS = true`）。正式跑尚未提交，等用户确认。

## 定义（用户裁决，规则 (a)）
- 在名义时刻 t 之后的第一次规划调用，把模型读入的主视角观测整帧置为全 0（RGB 0）。置黑发生在模型预处理之前；腕部相机和想象帧都不动。
- 档位名用名义时刻 F60 / F160 / F260。实际置黑步按模型不同：
  - LingBot：60 / 172 / 268。规划步实测为 0, 12, 28, 44, 60, …（之后每 16 步一次）。
  - OpenWAM：64 / 160 / 288。规划步为 32c，置黑落在第 2 / 5 / 9 次调用。
- 跨模型比较在名义档位层面进行。
- F0（sham）走同一挂点，只是不置黑。

## 检查单元（T0 = t5）
- LingBot：`lingbot_libero10_t5_a0__k1__none`。基线闭环失败、跑满 400 步，三个置黑点都能到达。作业 1775，节点 $NODE_C。
- OpenWAM：`openwam_libero10_t5_a0__k1`，同一锚点、同一种子序号。作业 1776，节点 $NODE_E。

## 结果

| 检查项 | LingBot-VA | OpenWAM-α |
|---|---|---|
| F0 与同进程未挂钩的原版闭环 | 逐位相同 | 逐位相同 |
| F0 与基线存档闭环 | 400 步逐位相同 | 逐位相同 |
| F0 的模型输入与原始观测 | 400 步全部相同 | 13 次调用的画布全部等于 `canvas_from_raw(原始)` |
| 实际置黑点 | 60 / 172 / 268，都是规划调用步 | 64 / 160 / 288（第 2 / 5 / 9 次调用） |
| (a) 模型输入和原始不同的地方 | 每档恰好 1 步 | 每档恰好 1 次调用 |
| (a) 置黑点之前的动作 | 与 F0 逐位相同；第一个不同的动作恰好在置黑步 | 与 F0 逐位相同；第一个不同的动作恰好在置黑步 |
| (a) 目标规划之前的模型读入 | 之前各块逐位相同；目标块 16 帧里只有最后一帧主视角不同 | 目标调用之前的画布逐位相同 |
| (b) 置黑在预处理之前 | 原始帧均值约 73–77；送进 `obs_to_batch` 的数组全 0；预处理后模型侧张量最大值 0.0（sham 同一帧约 0.29） | 原始主视角均值约 74–77；经过旋转、缩放、拼画布后，画布主视角区最大值为 0 |
| (c) 腕部未被碰 | 原对象原样传入；模型侧腕部张量与 F0 相同 | 画布腕部区与原始腕部、与 F0 都逐位相同 |
| (c) 想象未被碰 | 目标块之前的想象 latent 全部与 F0 逐位相同 | 目标调用之前的想象帧全部与 F0 逐位相同 |
| F60 跑两遍 | 动作逐位相同，结论一致 | 动作逐位相同，结论一致 |

## 截图
- `lingbot/sheet_F60.png`、`sheet_F160.png`、`sheet_F260.png`。每张的行依次是：原始主视角 → 喂给模型的帧（置黑步全黑）→ 目标块模型侧张量（F 与 sham 对照）→ 腕部 → 目标块想象解码（F 与 sham 对照）。
- `openwam/sheet_F60.png`、`sheet_F160.png`、`sheet_F260.png`。每张的行依次是：原始主视角 → F 和 sham 的模型画布 → 腕部 → 目标调用想象帧（F 与 sham 对照）。
- 证据数组：`lingbot/evidence_*.npz`、`openwam/evidence_*.npz`。

## 观察（不是检查项）
- OpenWAM 读到黑帧的那次调用，想象出来的主视角整段是黑的；下一次调用读到真实帧后恢复。
- LingBot 目标块想象的第 0 帧发暗，几帧之后恢复。

## 实现位置
- 共用定义：`libero/ffam/ffam_def.py`（档位、规划步、实际置黑步、置黑函数）。F 没有放进 `phys_perturb.FAMILIES`，物理族 runner 不会误跑。
- LingBot：`libero/ffam/lingbot_famF.py`。和 O 族同一挂点，即 LeRobot 原始观测进 `obs_to_batch` 之前；在"本步触发规划调用且步数 ≥ t"的第一步替换主视角。
- OpenWAM：`libero/ffam/ow_famF.py`。和 `ow_famO.closed_occ` 同一挂点，即原始 256 px 主视角进 `canvas_from_raw` 之前。
- 作业脚本：`libero/ffam/famF_job.sh`。健康检查脚本：`libero/ffam/hc_lingbot.py`、`hc_openwam.py`。
- 注册表 `libero/perturbation_registry.md` 升到 v2，新增 F 行和变更记录，改前备份在 `out/task2/backups/perturbation_registry_before_F_20261112.md`。
- `protocol.yaml` 的 `runs_new_tasks` 新增 `lingbot_famF` / `openwam_famF`，状态为 `pending_user_confirmation`，改前备份在 `libero/out/protocol_backup_20261112.yaml`。
