# WM-Verdict B 族图像劣化：统一协议、共享劣化代码与体检

这个仓库给参与 WM-Verdict 比较的各组共用。内容是：B 族（图像劣化）统一协议 v1.0；一份按协议原样调用 LIBERO-plus 劣化函数的劣化代码，带种子规则和随机数隔离；跨组等价自检；世界模型组的函数体检报告。

## 目录

| 路径 | 内容 |
|---|---|
| `protocol/protocol_B_v1.0.md` | 统一协议 v1.0 原文，未改动；文末附世界模型组注 |
| `protocol/execution_notes_wm_group.md` | 世界模型组的执行口径：首批类型、第 6 节注记、协议偏离备注 |
| `corruption/bcorrupt_server.py` | 劣化子进程：从固定 commit 的 `env_wrapper.py` 原样取出函数，按上游调用点的方式调用 |
| `corruption/bcorrupt.py` | 调用方：设置表、种子规则、`Corruptor` 客户端 |
| `corruption/requirements_pinned.txt` | 子进程的固定依赖：numpy 1.22.4、scikit-image 0.21.0 等，Python 3.10 |
| `corruption/fetch_upstream.sh` | 下载 LIBERO-plus 4976dc3 的 `env_wrapper.py` 并核验 sha256 |
| `selftest/` | 跨组等价自检，附 6 张测试帧和世界模型组存档的参考输出 |
| `healthcheck/` | 函数体检报告、联系表、剂量曲线、原始数字和脚本（脚本里的路径是我方机器的，仅作存档） |
| `wm_group_runners/` | 世界模型组两个模型的 B 族接入脚本，仅供参考接入位置，依赖我方基础设施，不能直接运行 |

## 快速开始

```bash
bash corruption/fetch_upstream.sh                      # 打印 "upstream/env_wrapper.py: OK" 才算核验通过
python3.10 -m venv .venv_pinned && .venv_pinned/bin/pip install -r corruption/requirements_pinned.txt
B_PINNED_PY=$PWD/.venv_pinned/bin/python python selftest/selftest.py   # 调用方的 Python 和 NumPy 版本随意
```

自检打印 `SELFTEST PASS`、并且 `2_matches_wm_group_reference` 为 `72/72`，说明你那边跑的劣化代码和世界模型组完全一致：同一张图、同一档、同一个种子，输出逐字节相同。请把自检输出的 JSON 回传给我们存档。

## 接入方式（协议第 3–6 节）

```python
from bcorrupt import Corruptor          # 需要 B_PINNED_PY 和 LIBERO_PLUS_ENV_WRAPPER 两个环境变量
corr = Corruptor()                      # 启动劣化子进程；启动时核验源文件 sha256
img_c = corr(raw_agentview_128, "Bf5", unit_id, step)   # 雾第 5 档；返回 uint8 [128, 128, 3]
```

- **输入**：LIBERO 原始的 `agentview_image`（128×128，uint8，不做任何翻转），在各模型自己的预处理之前调用。腕部相机不劣化。
- **设置代号**：`Bg2/Bg5/Bg9`（高斯模糊，noise 12/15/19）、`Bf2/Bf5/Bf9`（雾，32/35/39）、`Bz2/Bz5/Bz9`（缩放模糊，22/25/29）。毛玻璃也能调用：`corr.raw(img, "glass_blur", sev, seed)`。
- **时间**：环境步序号 `step` 从 0 开始，0 是 reset 后的观察；从 reset 到回合结束，每个新观察都劣化。
- **种子**：`int(sha256("<unit_id>|<step>|agentview|<函数名>|<档位>").hexdigest()[:8], 16)`，在子进程里用 `np.random.seed` 设定。劣化的随机数只存在于子进程中，不会动到仿真和策略的随机数流。高斯模糊和缩放模糊不用随机数。
- **运动模糊**：需要 Wand/ImageMagick，本仓库没有接入。

## 体检发现（详见 `healthcheck/report.md`）

1. **毛玻璃剂量不单调**：1→2、3→4、5→6、8→9 四步，PSNR 和 SSIM 都不降反升；第 5 档与第 9 档只差 0.07–0.4 dB。原因是参数表里迭代次数随档位递减，而最后一次模糊会抹平打乱的痕迹。世界模型组首批改用缩放模糊；毛玻璃的修正版需两组商定。
2. **雾在 NumPy ≥ 2 下无法运行**：`plasma_fractal` 里用了 `np.float_`。所以劣化放在依赖固定的子进程里运行，代码和 SHA 都不改。**请各组确认自己运行时的 NumPy 版本**：如果是 2.x，你们的雾函数必然被改过，SHA 会对不上。
3. **轻微项**（照跑，记入协议偏离备注）：uint8 截断偏置 −0.5 个灰度级；边缘劣化偏弱；雾和缩放模糊的高档持平。这三条都不在第 2/5/9 档之间。

## 来源

- LIBERO-plus：https://github.com/sylvestf/LIBERO-plus ，commit `4976dc30028e805ff8094b55501d532c48fec182`，`libero/libero/envs/env_wrapper.py`，sha256 `e91d7b7b35cc3ad2b073606c99860def6b3ac43b66eba00ef0ddb6bfd8f39c3c`。本仓库不包含上游代码，用 `fetch_upstream.sh` 下载。
- 测试帧：LIBERO-Long 任务 5 和任务 3 的真实 agentview 帧，各取开头、中段、结尾，共 6 张，128×128。
