# 两个 conda 环境的非 PyPI 来源与本地补丁（给 4090 新机器用）

生成日期 2026-11-13。所有信息都是在计算节点上用脚本实测得到的（作业 1925–1932，只用 CPU）：
- `pip` 的 `direct_url.json`：哪些包是本地或 git 安装的；
- 运行时 `import` 的实际路径；
- `git status/diff`：本地仓库的改动；
- 按 pip `RECORD` 哈希逐文件核对 site-packages：装好之后有没有文件被改过。

原始输出在 NAS 的 `wm_verdict/env_setup/raw/<节点>/`。

## 0. yml 从哪来，三个节点是否一致
- `lerobot_libero.yml` 和 `openwam.yml` 是在 $NODE_C 上用 `conda env export -n <env> --no-builds` 导出的。只有这台节点装了 conda 本体；585 和 627 的环境是 10-27 / 11-01 从 622 打包解压过去的（`node_sync/envs.tar`），只有 envs 目录，没有 conda。
- 三个节点的 `pip freeze` 对比：openwam 完全一致；lerobot_libero 只有 setuptools 不同（627 是 78.1.0，585 和 620 是 84.0.0），不影响运行。
- 环境基本情况：

  | 环境 | Python | torch | 其他 |
  |---|---|---|---|
  | lerobot_libero | 3.12.14 | 2.11.0+cu128 | mujoco 3.8.1、robosuite 1.4.0 |
  | openwam | 3.10.22 | 2.7.1+cu128 | diffusers 0.41.0、transformers 5.20.2 |

  cu128 的 wheel 要求驱动 ≥ 570，并且同时支持 4090（sm_89）和 5090。

## 1. 非 PyPI 安装的包：只有两个，都是 editable 本地仓库

| 环境 | 包 | 本地路径（NAS） | 仓库 | commit | 本地改动 |
|---|---|---|---|---|---|
| lerobot_libero | lerobot 0.6.2（editable，`.pth` 指向 `src/`） | `$LEROBOT_ROOT` | https://github.com/huggingface/lerobot | `e624f3f7f8411ec3a02635d06e79373341e5ef35`（main，`v0.6.1-103-ge624f3f7`，2026-09-24），和 origin/main 一致，没有本地提交 | 有未提交改动，但**不在我们用到的代码里**，见下 |
| openwam | openwam 0.1.0（editable） | `$WMV_ROOT/libero/openwam/third_party/OpenWAM` | https://github.com/OpenWAM-Official/OpenWAM | `1e103e1c61905c230dab6b0f99980370cf2f9e3c`（detached HEAD，2026-10-14，"Merge pull request #50"），没有本地提交 | 改了一行配置，见下 |

**lerobot 的本地改动**（`git diff HEAD`：4 个文件，+359 / −106；另有 6 个未跟踪文件）：
- 全部在 `src/lerobot/async_inference/` 里：改了 configs.py、helpers.py、policy_server.py、robot_client.py；新增了 external_planner_server.py、hardcoded_action_server.py、rl_client.py、robot_server.py、teleop_with_planner.py、test_communication.py。
- wm_verdict 的代码没有引用 async_inference（已 grep 确认），所以这些改动属于别的项目，不影响 LingBot-VA 的评测。
- LingBot-VA 的策略代码 `lerobot/policies/lingbot_va/` 就是上游这个 commit 的原样，没有改动。
- 补丁存档：`raw/$NODE_C/git_lerobot.patch`。

**OpenWAM 的本地改动**（只有 1 行）：
- `configs/model/video_backbone/wan22_ti2v_5b.yaml` 里的 `model_path`，从占位符 `/path/to/Wan2.2-TI2V-5B` 改成了本机权重路径 `.../OpenWAM/assets/video_backbone_ckpt/Wan2.2-TI2V-5B`。
- 新机器上改成那台机器的权重目录即可。
- 子模块 `third_party/cosmos-predict2.5`（`441b89740d91922737008a61e7f71407d47944e7`）**没有初始化**，目录是空的，运行不需要它。
- 补丁存档：`raw/$NODE_C/git_OpenWAM.patch`。

## 2. LIBERO、robosuite 等都是 PyPI 原包
运行时实测的导入路径都在 lerobot_libero 的 site-packages 里，没有 PYTHONPATH，也没有其他 `.pth` 注入。

| 模块 | 来源（PyPI） |
|---|---|
| `libero` / `libero.libero` | `hf-libero==0.1.4`（HuggingFace 打包的 LIBERO，不是原仓库源码安装） |
| `robosuite` | `robosuite==1.4.0` |
| `bddl` | `bddl==1.0.1` |
| `mujoco` | `mujoco==3.8.1` |
| `egl_probe` | `egl-probe==1.0.2` 和 `hf-egl-probe==1.0.2` 两个都装了，写的是同一个目录 |

openwam 环境里没有 libero 和 robosuite，仿真部分由 lerobot_libero 环境负责。

## 3. site-packages 里装好后被改过的文件（RECORD 哈希核对）
**真正的手动补丁只有 1 处：robosuite 日志路径。**
- 文件：`robosuite/utils/log_utils.py`，`DefaultLogger.__init__` 里的这一行：
  ```
  - fh = logging.FileHandler("/tmp/robosuite.log")
  + fh = logging.FileHandler(__import__("os").path.expanduser("~/robosuite.log"))
  ```
- 原因：robosuite 把日志写死在 `/tmp/robosuite.log`。在共享节点上，如果这个文件已经被别的用户创建，建环境时会因为没有权限直接崩溃。
- 打补丁的方式：作业脚本启动时自动幂等打上，见 `wm_verdict/libero/bfam/gate_traj_wrap.sh` 第 7–15 行，`out/task2/*_job.sh` 里也有同样的块。
- 现状：585 和 627 上已经打过；620 还是原版，因为 620 上从来没遇到这个冲突。
- 新机器：如果只有自己一个用户，可以不打；多人共用的话，照抄那段脚本，或者手动改这一行。
- 这个补丁只改日志位置，不影响仿真结果。

其余被哈希核对标出的文件都不是手动补丁：
- `bin/*` 入口脚本：解压环境时 shebang 被改写成了新前缀。
- `opencv-python-headless` 和 `opencv-python==5.0.0.93` 同时装了，后装的覆盖了 `cv2/`。
- `packaging` 和 `setuptools` 由 conda 和 pip 各装了一份，旧版 RECORD 对不上。
- `egl_probe/CMakeLists.txt` 和 `egl_probe/build/*`：两个 egl-probe 包互相覆盖，加上本地编译的产物。差别只在 `cmake_minimum_required` 那一行，只影响编译。

## 4. 新机器安装要点
1. 先 `conda env create -f lerobot_libero.yml`（openwam 同理）。在这之前要把 yml 的 `pip:` 段里 **`lerobot==0.6.2` 和 `openwam==0.1.0` 两行删掉**，它们不在 PyPI 上。
2. torch 的 `+cu128` 版本需要额外的索引：
   ```
   export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128
   ```
3. 装两个 editable 仓库：
   ```
   git clone https://github.com/huggingface/lerobot && cd lerobot && git checkout e624f3f7f8411ec3a02635d06e79373341e5ef35 && <lerobot_libero 的 python> -m pip install --no-deps -e .
   git clone https://github.com/OpenWAM-Official/OpenWAM && cd OpenWAM && git checkout 1e103e1c61905c230dab6b0f99980370cf2f9e3c && <openwam 的 python> -m pip install --no-deps -e .
   ```
   然后把 OpenWAM 的 `wan22_ti2v_5b.yaml` 里的 `model_path` 改成新机器的路径。
4. yml 最后一行的 `prefix:` 是旧机器的路径，`conda env create` 会忽略它，也可以直接删掉。
5. 除了环境本身，运行还需要以下文件（都不在 yml 里）：
   - `~/.libero/config.yaml`
   - `~/.cache/libero`
   - HF 缓存里的 `lerobot/libero-assets`、`lerobot/libero_10_image`、`lerobot/lingbot_va_libero_long`、`robbyant/lingbot-va-base`、`robbyant/lingbot-va-posttrain-libero-long`
   - OpenWAM 的 `assets/`（权重）

   原来的打包清单见 `node_sync/pack_on4.sh`，`node_sync/caches.tar` 可以直接用。
6. 跨机器结果复现（GPU 架构从 5090 换成 4090）需要先做 sham 逐位比对，再跑正式实验（eval_design §26）。
