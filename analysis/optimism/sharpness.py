"""想象帧清晰度 vs 模型读入画面清晰度（B 族图像腐蚀；纯 CPU；不判成败、不需要真实帧比对）。
清晰度 = 灰度图拉普拉斯方差（scipy.ndimage.laplace，float64，0–255 灰度）。数值越大越清楚。
OpenWAM-α（T3 = openwam_t4/famB，T4 = openwam_t6/famB，B0 + Bg/Bf/Bz × 2/5/9）：
  想象末帧 = imagined[-1] 的主视角槽（画布上 256 行）；同时记最后一次调用 9 帧的中位数。
  读入画面 = 重放干净 128×128 主视角（replay_frames/<uid>.npz 的 agentview[s]，环境原始朝向，s = 0, 32, …, 384）
            → LIBERO-plus 原版腐蚀函数（B0 不加；种子串用强度数字）→ ow_view.canvas_from_raw（内含 180° 旋转）→ 主视角槽；
            顺序与种子串经 verify2.py 在 LingBot 存档读入画面上核对：先腐蚀后旋转、强度数字种子，三种腐蚀 94–98% 像素逐位相同；记 13 个读入时刻的中位数和最后一帧（s=384）。
  干净对照 = 同一时刻不加腐蚀的读入画面。
LingBot-VA（T2 = lingbot_t2/famB）：读入画面 = 存档 real_XX__observation_images_image（模型实际读入）；
  想象末帧 = 已有解码帧（r3 卡片，按 128×128 重采样，分辨率说明见表注）；其余待 4090 解码。
验证：用 LingBot 重放帧 agentview[0] 按同法重建第 0 步读入画面，与存档 real_00 逐像素比对，确认重建方法（含雾的种子字符串）。
  python sharpness.py <out_dir>     （腐蚀函数用 LIBERO-plus 固定版本环境：numpy 1.22.4 / scipy 1.10.1 / skimage 0.21.0）"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import ast
import glob
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import laplace

LIB = _wmvp.P("$WMV_ROOT/libero")
RUNS = f"{LIB}/out/runs"
HC = f"{LIB}/out/b_family/healthcheck"
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
_wmvp.bootstrap()
from ow_view import canvas_from_raw  # noqa: E402  (纯 numpy / PIL)

FN = {"Bg": "gaussian_blur", "Bf": "fog", "Bz": "zoom_blur"}
EXPECTED = "e91d7b7b35cc3ad2b073606c99860def6b3ac43b66eba00ef0ddb6bfd8f39c3c"


def load_functions():
    src = open(f"{HC}/src/env_wrapper.py", "rb").read()
    assert hashlib.sha256(src).hexdigest() == EXPECTED
    tree = ast.parse(src.decode())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("gaussian_blur", "fog", "zoom_blur", "plasma_fractal", "clipped_zoom")]
    from skimage.filters import gaussian
    from scipy.ndimage import zoom as scizoom
    ns = dict(np=np, gaussian=gaussian, scizoom=scizoom)
    exec(compile(ast.Module(body=keep, type_ignores=[]), "env_wrapper.py", "exec"), ns)
    return ns


NS = load_functions()


def seed_of(uid, step, fn, level, fmt):
    key = {"level": f"{uid}|{step}|agentview|{fn}|{level}", "sev": f"{uid}|{step}|agentview|{fn}|{level[2:]}"}[fmt]
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def corrupt(img, uid, step, level, fmt="sev"):   # 种子串用强度数字（2/5/9），verify2 证实
    if level == "B0":
        return img
    fn = FN[level[:2]]
    np.random.seed(seed_of(uid, step, fn, level, fmt))
    return np.asarray(NS[fn](Image.fromarray(img), severity=int(level[2:]))).astype(np.uint8)


def lapvar(rgb):
    g = np.asarray(rgb, dtype=np.float64) @ np.array([0.299, 0.587, 0.114])
    return float(laplace(g).var())


def to_u8(x):
    """存档 real_XX：(n,1,3,128,128) float16，自动判断 [-1,1] / [0,1] / [0,255]。"""
    a = np.asarray(x, dtype=np.float32)[..., :, :, :]
    a = a.reshape(-1, 3, a.shape[-2], a.shape[-1]).transpose(0, 2, 3, 1)
    lo, hi = float(a.min()), float(a.max())
    if hi <= 1.01 and lo < -0.01:
        a = (a + 1) / 2 * 255
    elif hi <= 1.01:
        a = a * 255
    return np.clip(np.round(a), 0, 255).astype(np.uint8)


def verify_lingbot(n_per_level=3):
    """重建 vs 存档：LingBot T2 famB，每档取 n 个单元的第 0 步。返回各档逐像素相同比例与最大差。"""
    res = {}
    recs = [json.loads(l) for f in sorted(glob.glob(f"{RUNS}/lingbot_t2/famB/records.*.jsonl")) for l in open(f) if l.strip()]
    by = {}
    for r in recs:
        lv = (r.get("setting_id") or "").split("__")[-1]
        by.setdefault(lv, []).append(r["unit_id"])
    for lv in sorted(by):
        if not lv.startswith("B"):
            continue
        rows = []
        for uid in by[lv][:n_per_level]:
            rp = f"{RUNS}/lingbot_t2/famB/replay_frames/{uid}.npz"; cf = f"{RUNS}/lingbot_t2/famB/closed_frames/{uid}.npz"
            if not (os.path.exists(rp) and os.path.exists(cf)):
                continue
            clean0 = np.load(rp)["agentview"][0]
            stored = to_u8(np.load(cf)["real_00__observation_images_image"])[0]
            best = None
            for fmt in ("level", "sev"):
                for orient in ("raw", "flip_v", "rot180"):
                    c = {"raw": clean0, "flip_v": clean0[::-1], "rot180": clean0[::-1, ::-1]}[orient]
                    rec = corrupt(np.ascontiguousarray(c), uid, 0, lv, fmt)
                    d = np.abs(rec.astype(int) - stored.astype(int))
                    cand = dict(fmt=fmt, orient=orient, exact=float((d == 0).mean()), maxdiff=int(d.max()))
                    if best is None or cand["exact"] > best["exact"]:
                        best = cand
            rows.append(dict(uid=uid, **best))
        res[lv] = rows
    return res


def openwam_units():
    out = []
    for run, T in (("openwam_t4", "T3"), ("openwam_t6", "T4")):
        for f in sorted(glob.glob(f"{RUNS}/{run}/famB/closed_imagined/*.npz")):
            uid = os.path.basename(f)[:-4]
            out.append(dict(model="openwam", T=T, uid=uid, level=uid.split("__")[-1], imag=f, replay=f"{RUNS}/{run}/famB/replay_frames/{uid}.npz"))
    return out


def do_openwam(u, orient):
    z = np.load(u["imag"]); im, cl = z["imagined"], z["imagined_call_lens"]
    last = im[-1][:256]; lastcall = im[len(im) - int(cl[-1]):]
    to128 = lambda c: np.asarray(Image.fromarray(np.ascontiguousarray(c)).resize((128, 128), Image.BILINEAR))   # 同 ow_view.pred_view
    r = dict(model=u["model"], task=u["T"], unit_id=u["uid"], level=u["level"], imag_last=lapvar(last), imag_last_128=lapvar(to128(last)),
             imag_lastcall_median=float(np.median([lapvar(f[:256]) for f in lastcall])))
    if os.path.exists(u["replay"]):
        av = np.load(u["replay"])["agentview"]
        steps = [s for s in range(0, 400, 32) if s < len(av)]
        blank = np.zeros_like(av[0])
        inp, cln, inp128, cln128 = [], [], [], []
        for s in steps:
            raw = np.ascontiguousarray(av[s])          # 环境原始朝向：先腐蚀，再由 canvas_from_raw 旋转 180°（verify2 证实的顺序）
            inp.append(lapvar(canvas_from_raw(corrupt(raw, u["uid"], s, u["level"]), blank)[:256]))
            cln.append(lapvar(canvas_from_raw(raw, blank)[:256]))
            cr = corrupt(raw, u["uid"], s, u["level"]); inp128.append(lapvar(cr)); cln128.append(lapvar(raw))   # 128 原生分辨率
        r.update(input_median=float(np.median(inp)), input_last=inp[-1], clean_median=float(np.median(cln)), n_reads=len(steps),
                 input_median_128=float(np.median(inp128)), clean_median_128=float(np.median(cln128)))
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    import numpy, scipy, skimage
    env = dict(numpy=numpy.__version__, scipy=scipy.__version__, skimage=skimage.__version__, python=sys.version.split()[0])
    ver = verify_lingbot() if os.environ.get("SKIP_VERIFY") != "1" else json.load(open(f"{OUT}/verify.json"))["verify_lingbot_step0"]
    json.dump(dict(env=env, verify_lingbot_step0=ver), open(f"{OUT}/verify.json", "w"), indent=1)
    fmts = [r["fmt"] for v in ver.values() for r in v]; orients = [r["orient"] for v in ver.values() for r in v]
    orient = max(set(orients), key=orients.count) if orients else "raw"
    print("env", env, "verify orient", orient, {k: [round(r["exact"], 4) for r in v] for k, v in ver.items()}, flush=True)
    with open(f"{OUT}/openwam_sharpness.jsonl", "w") as f:
        for i, u in enumerate(openwam_units()):
            f.write(json.dumps(do_openwam(u, orient)) + "\n")
            if i % 100 == 0:
                print("openwam", i, flush=True)
    # LingBot：读入画面（存档）+ 已解码想象末帧（r3 卡片）
    r3 = {c["unit_id"]: c for c in json.load(open(f"{LIB}/out/annotation/round3_imagined/manifest_r3.json"))["cards"]}
    with open(f"{OUT}/lingbot_sharpness.jsonl", "w") as f:
        for cf in sorted(glob.glob(f"{RUNS}/lingbot_t2/famB/closed_frames/*.npz")):
            uid = os.path.basename(cf)[:-4]; lv = uid.split("__")[-1]
            z = np.load(cf); keys = sorted(k for k in z.files if k.startswith("real_") and k.endswith("_image"))
            firsts = [to_u8(z[k])[0] for k in keys]; lastf = to_u8(z[keys[-1]])[-1]
            r = dict(model="lingbot", task="T2", unit_id=uid, level=lv, input_median=float(np.median([lapvar(x) for x in firsts])), input_last=lapvar(lastf))
            if uid in r3:
                card = Image.open(r3[uid]["png"]).convert("RGB")
                card = card.crop((4, 4, card.width - 4, card.height - 4)).resize((128, 128), Image.BICUBIC)   # 去掉黄框，回到 128
                r["imag_last_r3card"] = lapvar(card)
            f.write(json.dumps(r) + "\n")
    print("done")


if __name__ == "__main__":
    main()
