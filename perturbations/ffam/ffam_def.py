"""扰动族 F（单次全帧遮挡，信息 × 一次性；注册表 v2，2026-11-12 用户裁决）——两个模型共用的定义。

定义：名义时刻 t ∈ {60, 160, 260}（与 K 族同网格）。在 t 之后（含 t）的【第一次规划调用】读入的那一帧主视角观测，
在进入模型预处理之前整帧置黑（RGB 全 0）；腕部相机不动；想象帧不碰；其余所有步的观测原样。
档位名保持名义时刻 F60 / F160 / F260；F0 = 零剂量对照（sham，走同一条代码路径，不置黑）。
每个模型实际置黑的环境步（= 规划调用发生的步，由各自的重规划节奏决定）：
  LingBot-VA：首块 12 个动作，之后每块 16 个 → 规划调用在第 0, 12, 28, 44, 60, …, 12+16m 步 → 60 / 172 / 268
  OpenWAM-α ：每 32 步一次规划调用（第 32c 步）                                   → 64 / 160 / 288
跨模型比较在名义档位层面进行。F 只在闭环里有意义（与 K 一样 closed_loop_only），不放进 phys_perturb.FAMILIES，
避免物理族 runner（按 FAMILIES 全量循环）误跑 F。
"""
import numpy as np

LEVELS = {"F0": None, "F60": 60, "F160": 160, "F260": 260}
SETTINGS = ["F60", "F160", "F260"]                      # 正式档位（不含 sham）
H = 400


def lingbot_plan_steps(H=H):
    s, out = 0, []
    while s < H:
        out.append(s); s += 12 if s == 0 else 16
    return out


def openwam_plan_steps(H=H, block=32):
    return list(range(0, H, block))


def expected_blackout_step(model, t_nom):
    if t_nom is None:
        return None
    steps = lingbot_plan_steps() if model == "lingbot" else openwam_plan_steps()
    return next(s for s in steps if s >= t_nom)


EXPECTED = {m: {k: expected_blackout_step(m, v) for k, v in LEVELS.items()} for m in ("lingbot", "openwam")}
assert [EXPECTED["lingbot"][k] for k in SETTINGS] == [60, 172, 268], EXPECTED
assert [EXPECTED["openwam"][k] for k in SETTINGS] == [64, 160, 288], EXPECTED


def blackout(img):
    """整帧置黑（新数组，原数组不动）。"""
    return np.zeros_like(np.asarray(img))


def spec(model, setting, applied_step):
    t = LEVELS[setting]
    return dict(family="F", kind="blackout", setting=setting, sham=t is None, nominal_t=t,
                expected_blackout_step=EXPECTED[model][setting], actual_blackout_step=applied_step,
                camera="agentview", wrist="untouched", imagined="untouched",
                rule="first planning call at or after nominal t; agentview obs read by that call set to RGB 0 before model preprocessing",
                registry="perturbation_registry v2 (2026-11-12)")
