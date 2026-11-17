"""终局想象乐观率 · 第 3 步："想象成功 vs 现实失败"对比案例（只读现有数据；LingBot 潜变量在 GPU 上解码，不补采任何帧）。
  python cases.py prep     （lerobot_libero，GPU）对每个现实失败单元抽出：最后一次规划调用的想象帧 + 同名义步的真实帧 → cache/<uid>.npz；
                            再按 模型×任务 拼候选总览 sheets/<model>_<T>.png（每格左 = 现实第 400 步，右 = 想象末帧），供人工挑选
  python cases.py figure <uid> [<uid> ...]   → figures/<uid>.png（上排想象帧条、下排真实帧条，最右一列为判定帧）
步与帧的对应：
  LingBot：闭环各块潜变量沿时间拼接后整段解码（与上游一致），第 f 帧 ≈ 第 f 步；最后一块 = 解码序列最后 16 帧（名义第 397–412 步）。
  OpenWAM：最后一次（第 13 次）调用 9 帧，第 j 帧 = 第 384+4j 步（j=0 为读入的真实条件帧）。
  真实帧：步号 > 400 的一律取第 400 步（闭环只执行到 400）。方向统一为正视（与 round2 卡片、Cosmos/UVA 视频同向）：
  原始渲染只做上下翻转；OpenWAM 画布顶部 = 原始转 180°，再左右翻转回正视；LingBot 解码帧本身即正视。
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import json
import os
import sys
import zlib

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
RUNS = f"{ROOT}/libero/out/runs"
H = 400
OW_SAMPLE = 60              # OpenWAM T3/T4 失败单元数以百计：候选总览每任务按固定种子抽 60 个（其余任务全收）


def real_source(u):
    """(npz, 第 f 步 → 下标 的偏移)。rerun npz 第 0 帧 = 初始观测；replay npz 第 i 帧 = 第 i+1 步后。"""
    if u["source"] == "lb_rerun":
        return u["frames"], 0
    tid = {"T0": 5, "T1": 3, "T2": 2, "T3": 4, "T4": 6}[u["T"]]
    run = {"ow_backfill": None, "ow_phys": f"openwam_t{tid}/phys", "ow_famB": f"openwam_t{tid}/famB",
           "ow_rerun": f"openwam_t{tid}/rerun_closed", "lb_famB_t2": "lingbot_t2/famB"}[u["source"]]
    if run is None:
        run = os.path.basename(os.path.dirname(os.path.dirname(u["frames"])))      # openwam_{base,famA,famP}
    return f"{RUNS}/{run}/replay_frames/{u['unit_id']}.npz", -1


def upright_real(x):
    return np.ascontiguousarray(np.asarray(x)[::-1])


def pick_units():
    U = [u for u in json.load(open(f"{HERE}/units.json")) if not u["success"]]
    out = []
    for key in sorted({(u["model"], u["T"]) for u in U}):
        us = sorted([u for u in U if (u["model"], u["T"]) == key], key=lambda u: u["unit_id"])
        if key[0] == "openwam" and len(us) > OW_SAMPLE:
            rng = np.random.RandomState(zlib.crc32(f"20261113/{key}".encode()))
            us = [us[i] for i in sorted(rng.choice(len(us), OW_SAMPLE, replace=False))]
        out += us
    return out


def prep():
    import torch
    _wmvp.bootstrap()
    os.makedirs(f"{HERE}/cache", exist_ok=True)
    U = pick_units()
    rig = None
    done = 0
    for u in U:
        dst = f"{HERE}/cache/{u['unit_id']}.npz"
        if os.path.exists(dst):
            continue
        rp, off = real_source(u)
        if not os.path.exists(rp):
            print("NO_REAL", u["unit_id"], rp, flush=True); continue
        z = np.load(u["frames"])
        if u["model"] == "openwam":
            im, cl = z["imagined"], z["imagined_call_lens"]
            last = im[len(im) - int(cl[-1]):]
            imag = np.ascontiguousarray(last[:, :256, ::-1])            # 画布顶部 = 原始转 180° → 左右翻回正视
            steps = [384 + 4 * j for j in range(len(last))]
        else:
            if rig is None:
                from lingbot_unit import LingBotRig
                rig = LingBotRig(2)
                rig.policy._ensure_frozen_modules()                   # VAE 懒加载；不跑模型时需手动初始化
            p = "imag_latents" if u["key"] == "imag_latents" else "latents"
            n = int(z[f"imag_n_chunks" if p == "imag_latents" else "n_chunks"])
            lat = torch.cat([torch.from_numpy(z[f"{p}_{i:02d}"].astype(np.float32)) for i in range(n)], dim=2)
            vid = rig.decode(lat).numpy()                                # [T, 128, 256, 3] = 主视角 | 腕部，正视
            imag = vid[-16:, :, :128]
            steps = list(range(len(vid) - 16, len(vid)))
        R = np.load(rp)["agentview"]
        real = np.stack([upright_real(R[min(max(min(s, H) + off, 0), len(R) - 1)]) for s in steps])
        np.savez_compressed(dst, imagined=imag, real=real, steps=np.array(steps), unit=u["unit_id"])
        done += 1
        if done % 25 == 0:
            print("prep", done, flush=True)
    print("PREP_DONE", done, "of", len(U), flush=True)
    sheets(U)


def tile(a, h=112):
    im = Image.fromarray(a)
    return im.resize((int(im.width * h / im.height), h))


def sheets(U):
    os.makedirs(f"{HERE}/sheets", exist_ok=True)
    idx = {}
    for key in sorted({(u["model"], u["T"]) for u in U}):
        us = [u for u in U if (u["model"], u["T"]) == key and os.path.exists(f"{HERE}/cache/{u['unit_id']}.npz")]
        cols, cw, chh = 6, 2 * 112 + 12, 112 + 18
        S = Image.new("RGB", (cols * cw, ((len(us) + cols - 1) // cols) * chh), (255, 255, 255)); d = ImageDraw.Draw(S)
        for i, u in enumerate(us):
            c = np.load(f"{HERE}/cache/{u['unit_id']}.npz")
            x, y = (i % cols) * cw, (i // cols) * chh
            S.paste(tile(c["real"][-1]), (x, y + 16)); S.paste(tile(c["imagined"][-1]), (x + 112 + 4, y + 16))
            d.text((x + 2, y + 2), f"#{i} {u['level']}", fill=(0, 0, 0))
            idx[f"{key[0]}_{key[1]}#{i}"] = u["unit_id"]
        S.save(f"{HERE}/sheets/{key[0]}_{key[1]}.png")
    json.dump(idx, open(f"{HERE}/sheets/index.json", "w"), indent=0)
    print("SHEETS", sorted({f"{m}_{T}" for m, T in {(u['model'], u['T']) for u in U}}))


def figure(uids):
    os.makedirs(f"{HERE}/figures", exist_ok=True)
    meta = {u["unit_id"]: u for u in json.load(open(f"{HERE}/units.json"))}
    for uid in uids:
        c = np.load(f"{HERE}/cache/{uid}.npz"); u = meta[uid]
        sel = np.unique(np.linspace(0, len(c["steps"]) - 1, min(8, len(c["steps"]))).round().astype(int))
        sq = lambda a: Image.fromarray(a).resize((128, 128))     # OpenWAM 画布主视角区是 256→320 横向拉伸存的，还原成正方形与真实帧同比例
        top = [sq(c["imagined"][i]) for i in sel]; bot = [sq(c["real"][i]) for i in sel]
        W = sum(t.width for t in top) + 4 * len(top) + 130; Ht = 2 * 128 + 70
        F = Image.new("RGB", (W, Ht), (255, 255, 255)); d = ImageDraw.Draw(F)
        d.text((6, 4), f"{u['model']} {u['T']} {u['level']}  {uid}   real closed loop: FAIL (H=400)", fill=(0, 0, 0))
        x = 130
        for j, i in enumerate(sel):
            F.paste(top[j], (x, 22)); F.paste(bot[j], (x, 22 + 128 + 22))
            s = int(c["steps"][i]); d.text((x + 2, 22 + 128 + 4), f"step {s}" + (" (>400: real@400)" if s > H else ""), fill=(80, 80, 80))
            x += top[j].width + 4
        d.text((6, 22 + 56), "imagined\n(last plan)", fill=(0, 0, 0)); d.text((6, 22 + 128 + 22 + 56), "real", fill=(0, 0, 0))
        xl = x - top[-1].width - 4
        d.rectangle([xl - 2, 20, xl + top[-1].width + 1, 22 + 128 + 22 + 128 + 1], outline=(255, 196, 0), width=3)
        F.save(f"{HERE}/figures/{uid}.png")
        print("FIG", uid)


if __name__ == "__main__":
    {"prep": lambda: prep(), "figure": lambda: figure(sys.argv[2:]), "sheets": lambda: sheets(pick_units())}[sys.argv[1]]()
