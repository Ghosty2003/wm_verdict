"""同卷对比标注 · 第 4 步：为 pages.json 出缺的卡片图 + 网站页面表 + 入库清单。
REAL（round 2）：13 帧等距帧条，最右 = 最后一帧（与 round 2 的 make_card 完全相同）；库里已有同 unit_id 的 round 2 卡就沿用，不重画。
IMAGINED（round 3）：想象末帧正方形卡（与 round 3 的 card() 相同）；已有 round 3 卡就沿用。
  我们：真实帧 = replay / lb_rerun / F 族闭环存档 npz（第 0 帧是否为初始观测按 帧数 vs success 长度自动判断）；
        OpenWAM 想象 = 闭环存档 imagined[-1]（同 r3）；LingBot 想象 = 已缓存解码帧，没有缓存的记 need_decode（GPU 解码另跑）。
  Cosmos：真实 = 闭环视频左半主视角；想象 = 终局帧左半主视角（T0 = 11-13 全量新重跑同一次执行；T1 = 原执行末窗补算；T2–T4 = 原执行末窗补算）。
  UVA：真实 = native-closed.mp4；想象未存。
子命令：
  probe            每种（模型 × 来源）各出一张样卡到 probe/，核对方向与帧归属
  render <i> <n>   分片出图（已存在的跳过）
  site             → 网站仓库 data/samecase_pages.json（页面表）；manifest_samecase.json（新卡入库清单）；need_decode.json
v4（2026-11-16）：页面按“组”（第 1–5 组）组织，不再按 seed；pages.json 由 v4 select_pages.py 生成。
  新增来源：H100 运行（OpenWAM x86_rerun 同格式；LingBot 存了 imag_decoded，直接取最后一帧左半）、
  LingBot 集群 rerun_closed 同一 npz 的 imag_latents（CPU 解码，decode <shard> <n> 分片）、Cosmos F 族 closed_loop1、Cosmos T1 11-13 全量新重跑、UVA T0 K。
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import collections
import glob
import json
import os
import re
import sys
import zlib

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
OPT = f"{ROOT}/libero/out/optimism"
WEB = _wmvp.P("$WMV_WEB_ROOT")
_wmvp.bootstrap()
from real_cards import make_card, H                      # noqa: E402
from r3_cards import card as imag_card                   # noqa: E402
from build_manifest import GOAL, FAM as FAM0              # noqa: E402

TT = {"T0": "t5", "T1": "t3", "T2": "t2", "T3": "t4", "T4": "t6"}
LABELS = ["looks_success", "borderline", "not_success", "uncertain"]
HOW_REAL = ("Closed-loop deployment, REAL camera view (no imagination): 13 evenly spaced frames from the start of the episode "
            "to its final frame (yellow box, rightmost). Judge the boxed frame: ")
HOW_IMAG = ("Closed-loop deployment, IMAGINED view: this is the world model's own predicted camera frame at its LAST planning call "
            "(the final imagined frame, yellow box). It is NOT a real camera frame. Judge only what the imagined image shows: ")
FAM = lambda lv: "F" if lv.startswith("F") else FAM0(lv)
UNITS = {u["unit_id"]: u for u in json.load(open(f"{OPT}/units.json"))}


def pages():
    return json.load(open(f"{HERE}/pages.json"))["pages"]


def left_half(a):
    return a[:, : a.shape[1] // 2] if a.shape[1] >= 2 * a.shape[0] - 4 else a


# ---------------- REAL
def real_frames(model, real):
    """→ (frames 正向主视角, last_idx, success, first_success_step)。"""
    p = real["path"]
    if model in ("lingbot", "openwam"):
        if real["src"] == "replay":                     # 回放完成后帧在 runs/<run>/replay_frames/<uid>.npz
            p = real["replayed_path"]
        z = np.load(p)
        av = z["agentview" if "agentview" in z.files else "agentview_steps"][:, ::-1]   # 原始渲染上下颠倒：只做上下翻转（同 round 2）；x86 重跑存为 agentview_steps（第 i 帧 = 第 i+1 步后）
        succ = np.asarray(z["success"]).astype(bool).reshape(-1)
        fs = int(np.argmax(succ)) + 1 if succ.any() else None
        if len(av) == len(succ):                        # 第 i 帧 = 第 i+1 步后
            li = (fs - 1) if fs else min(len(av), H) - 1
        else:                                           # 第 0 帧 = 初始观测
            li = fs if fs else min(len(av) - 1, H)
        return av, max(0, min(li, len(av) - 1)), bool(fs), fs
    import imageio.v3 as iio
    fr = iio.imread(p)
    if model == "cosmos":
        fr = np.stack([left_half(f) for f in fr]) if fr.shape[2] >= 2 * fr.shape[1] - 4 else fr
    ok, fs = real["success"], real["first_success_step"]
    off = 1 if (model == "uva" or "rerun1113" in real["unit_id"]) else 0   # UVA / Cosmos 新重跑：第 0 帧 = 初始观测（probe 核对）；Cosmos F 族视频帧数 = 步数（probe_v4：176 帧、首次成功第 176 步），不加偏移
    li = ((fs if ok and fs else H) - 1 + off)
    return fr, max(0, min(li, len(fr) - 1)), ok, fs


# ---------------- IMAGINED
def imag_frame(model, imag):
    if model == "openwam":
        c = f"{OPT}/cache/{imag['unit_id']}.npz"
        fresh = "/x86_rerun/" in imag["path"] or "/lastcall/" in imag["path"] or "/h100/" in imag["path"]   # 重跑 / 末次重算 / H100：直接取该文件，不用旧缓存
        if os.path.exists(c) and not fresh:
            return np.load(c)["imagined"][-1]
        z = np.load(imag["path"])
        return np.ascontiguousarray(z["imagined"][-1][:256, ::-1])
    if model == "lingbot":
        if imag.get("key") == "imag_decoded":           # H100 运行时已解码（与 lb_decode 同一流程），取最后一帧主视角左半
            return np.ascontiguousarray(np.load(imag["path"])["imag_decoded"][-1][:, :128])
        c = f"{OPT}/cache/{imag['unit_id']}.npz"
        if os.path.exists(c):
            return np.load(c)["imagined"][-1]
        return None                                     # 需解码（decode 子命令，CPU）
    if model == "cosmos":
        if "#" in imag["path"]:                   # 规划预测 npz：取该次规划预测的最后一帧主视角
            f, k = imag["path"].split("#"); a = np.asarray(np.load(f)[k])
            a = a.reshape((-1,) + a.shape[-3:])[-1]
            if a.dtype != np.uint8:
                a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8) if a.max() <= 1.0 else np.clip(a, 0, 255).astype(np.uint8)
            return left_half(a)
        return left_half(np.asarray(Image.open(imag["path"]).convert("RGB")))
    return None


def out_real(uid):
    return f"{WEB}/cards/r2/{uid}.webp"


def out_imag(uid):
    return f"{WEB}/cards/r3/{uid}.webp"


def jobs():
    """去重后的出图任务：(kind, model, entry)。已有卡（库里 round 2 / 3 已有同 unit_id）不出。"""
    J, seen = [], set()
    for p in pages():
        for r in p["rows"]:
            if r["real"] and not r["real"]["r2_card"] and ("r", r["real"]["unit_id"]) not in seen:
                seen.add(("r", r["real"]["unit_id"])); J.append(("real", r["model"], r["real"]))
            if r["imag"] and not r["imag"]["r3_card"] and ("i", r["imag"]["unit_id"]) not in seen:
                seen.add(("i", r["imag"]["unit_id"])); J.append(("imag", r["model"], r["imag"]))
    return J


def replayed(e):
    if e["src"] != "replay":
        return e
    m = re.match(r"(lingbot|openwam)_", e["unit_id"])
    hits = glob.glob(f"{ROOT}/libero/out/runs/*/replay_frames/{e['unit_id']}.npz") + glob.glob(f"{ROOT}/libero/out/runs/*/*/replay_frames/{e['unit_id']}.npz")
    return dict(e, replayed_path=hits[0] if hits else None) if m else e


def save_webp(im, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path + ".tmp", "WEBP", quality=85, method=6); os.replace(path + ".tmp", path)


def render_one(kind, model, e):
    if kind == "real":
        e = replayed(e)
        if e["src"] == "replay" and not e.get("replayed_path"):
            return "wait_replay"
        fr, li, ok, fs = real_frames(model, e)
        im = make_card(fr, li)
        im.save(f"{HERE}/cards/real/{e['unit_id']}.png", optimize=True); save_webp(im, out_real(e["unit_id"]))
        return dict(success=ok, first_success_step=fs, last_idx=int(li), n_frames=int(len(fr)), path=e.get("replayed_path") or e["path"], run=e.get("run"))
    f = imag_frame(model, e)
    if f is None:
        return "need_decode"
    im = imag_card(f)
    im.save(f"{HERE}/cards/imag/{e['unit_id']}.png", optimize=True); save_webp(im, out_imag(e["unit_id"]))
    used = f"{OPT}/cache/{e['unit_id']}.npz" if model in ("openwam", "lingbot") and e.get("key") != "imag_decoded" and os.path.exists(f"{OPT}/cache/{e['unit_id']}.npz") \
        and not (model == "openwam" and any(x in e["path"] for x in ("/x86_rerun/", "/lastcall/", "/h100/"))) else e["path"]
    return dict(shape=list(f.shape), path=used, src_path=e["path"], run=e.get("run"))


def render(shard, nsh):
    os.makedirs(f"{HERE}/cards/real", exist_ok=True); os.makedirs(f"{HERE}/cards/imag", exist_ok=True)
    log = open(f"{HERE}/render.{shard}.jsonl", "a"); c = collections.Counter()
    for kind, model, e in jobs():
        if zlib.crc32(f"{kind}/{e['unit_id']}".encode()) % nsh != shard:
            continue
        if os.path.exists((out_real if kind == "real" else out_imag)(e["unit_id"])):
            c["exists"] += 1; continue
        try:
            res = render_one(kind, model, e)
        except Exception as ex:  # noqa: BLE001
            res = f"FAIL {ex!r}"
        c[res if isinstance(res, str) else "ok"] += 1
        log.write(json.dumps(dict(kind=kind, model=model, unit_id=e["unit_id"], result=res), ensure_ascii=False) + "\n"); log.flush()
    print("RENDER", shard, dict(c), flush=True)


def probe():
    os.makedirs(f"{HERE}/probe_v4", exist_ok=True)
    got = {}
    for kind, model, e in jobs():
        k = (kind, model, e["src"], e.get("execution", "")[:20])
        if k in got or (kind == "real" and e["src"] == "replay"):
            continue
        try:
            if kind == "real":
                fr, li, ok, fs = real_frames(model, replayed(e)); im = make_card(fr, li); info = f"ok={ok} fs={fs} li={li} n={len(fr)}"
            else:
                f = imag_frame(model, e)
                if f is None:
                    continue
                im = imag_card(f); info = str(f.shape)
        except Exception as ex:  # noqa: BLE001
            print("PROBE FAIL", k, e["unit_id"], repr(ex)); continue
        fn = f"{HERE}/probe_v4/{kind}_{model}_{e['src']}_{len(got):02d}.png"
        im.save(fn); got[k] = (e["unit_id"], info, fn)
        print("PROBE", k, e["unit_id"], info, fn, flush=True)


def site():
    """页面表（网站读）+ 入库清单（页上所有卡；入库脚本跳过库里已有的 unit_id）。v4：按组，不带 seed / anchor / 代用说明。"""
    P = pages(); man, need_dec, missing = [], [], collections.Counter()
    seen = set()
    ok = lambda kind, e: bool(e) and bool(e["r2_card" if kind == "real" else "r3_card"] or os.path.exists((out_real if kind == "real" else out_imag)(e["unit_id"])))
    for p in P:
        t = TT[p["T"]]
        for r in p["rows"]:
            for kind in ("real", "imag"):
                e = r[kind]
                if not e:
                    continue
                if not ok(kind, e):
                    missing[(kind, r["model"])] += 1
                    if kind == "imag" and r["model"] == "lingbot":
                        need_dec.append(e["unit_id"])
                    continue
                if (kind, e["unit_id"]) not in seen:
                    seen.add((kind, e["unit_id"]))
                    man.append(dict(round_id=2 if kind == "real" else 3, model=r["model"], task=t, family=FAM(p["level"]), level=p["level"],
                                    unit_id=e["unit_id"], question=(HOW_REAL if kind == "real" else HOW_IMAG + "does the imagined scene show the task COMPLETED — ") + GOAL[t],
                                    labels=LABELS, ref=f"cards/ref/{t}.png", source=e["src"], execution=e.get("execution"), machine=e.get("machine")))
    json.dump(dict(cards=man), open(f"{HERE}/manifest_samecase.json", "w"), ensure_ascii=False, indent=0)
    SP = []
    for p in P:
        rows = []
        for r in p["rows"]:
            re_, im_ = (r["real"]["unit_id"] if ok("real", r["real"]) else None), (r["imag"]["unit_id"] if ok("imag", r["imag"]) else None)
            row = dict(model=r["model"], in_cell=r["in_cell"], real=re_, imag=im_ if re_ else None)   # 想象卡只与同一单元的真实卡一起出现
            if r.get("note"):
                row["note"] = r["note"]                                                               # UVA 待重跑留白
            rows.append(row)
        SP.append(dict(id=p["page_id"], T=p["T"], task=TT[p["T"]], level=p["level"], group=p["group"], rows=rows))
    os.makedirs(f"{WEB}/data", exist_ok=True)
    json.dump(dict(generated="2026-11-16", rule="v4 groups: per task x level up to 5 groups; each model picks its own units (complete = real + imagined from the same run first); "
                   "see wm_verdict/libero/out/annotation/samecase_20261115/select_pages.py", pages=SP),
              open(f"{WEB}/data/samecase_pages.json", "w"), ensure_ascii=False, separators=(",", ":"))
    nq = sum((r["real"] is not None) + (r["imag"] is not None) for p in SP for r in p["rows"])
    print("site pages", len(SP), "questions", nq)
    json.dump(sorted(set(need_dec)), open(f"{HERE}/need_decode.json", "w"))
    print("cards on pages", collections.Counter((m["round_id"], m["model"]) for m in man), "missing images", dict(missing), "need_decode", len(set(need_dec)))


def decode(shard=0, nsh=1):
    """LingBot 想象末帧：只加载 VAE 解码已存 latent（与 r3 完全相同的 lb_decode，取最后一帧主视角左半），出 round 3 卡。
    v4：latent 取页面表里该单元自己的文件（imag path / key），不再只限 optimism/units.json；DECODE_DEVICE=cpu 时在 CPU 上解码；按 crc32 分片。"""
    from r3_cards import lb_decode
    if os.environ.get("DECODE_DEVICE") == "cpu":      # GPU 都在跑重跑时改在 CPU 上解码（float32；与 lb_decode 同一流程，只换 device/dtype）
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.lingbot_va.utils import denormalize_latents, load_vae
        from r3_cards import lb_latents
        _vae = load_vae(PreTrainedConfig.from_pretrained("lerobot/lingbot_va_libero_long").wan_pretrained_path, torch_dtype=torch.float32,
                        torch_device="cpu", subfolder="vae").eval()
        def lb_decode(u, box):   # noqa: F811
            with torch.no_grad():
                lat = denormalize_latents(lb_latents(u).float(), _vae.config.latents_mean, _vae.config.latents_std, _vae.config.z_dim)
                v = (_vae.decode(lat, return_dict=False)[0].float().clamp(-1, 1) + 1.0) / 2.0
                return (v[0].permute(1, 2, 3, 0) * 255.0).round().to(torch.uint8).numpy()
    todo = {e["unit_id"]: e for kind, model, e in jobs() if kind == "imag" and model == "lingbot" and e["src"] == "latent"
            and not os.path.exists(f"{OPT}/cache/{e['unit_id']}.npz") and zlib.crc32(f"dec/{e['unit_id']}".encode()) % nsh == shard}
    box, n = [], 0
    print("decode todo", len(todo), flush=True)
    os.makedirs(f"{HERE}/cards/imag", exist_ok=True)
    # 每个分片写自己的日志（2026-11-16：11 个分片并发追加同一个 NAS 文件时行被交错 / 丢失，见 render.decode.jsonl.bak_20261116_interleaved）
    log = open(f"{HERE}/render.decode.jsonl" if nsh == 1 else f"{HERE}/render.decode.s{shard:02d}of{nsh}.jsonl", "a")
    for uid in sorted(todo):
        if os.path.exists(out_imag(uid)):
            continue
        u = dict(frames=todo[uid]["path"], key=todo[uid].get("key") or "imag_latents")
        f = lb_decode(u, box)[-1][:, :128]
        im = imag_card(f); im.save(f"{HERE}/cards/imag/{uid}.png", optimize=True); save_webp(im, out_imag(uid))
        log.write(json.dumps(dict(kind="imag", model="lingbot", unit_id=uid, result=dict(shape=list(f.shape), decoded_from=u["frames"], device=os.environ.get("DECODE_DEVICE", "cuda")))) + "\n"); log.flush()
        n += 1
    print("DECODE_DONE", n, "of", len(todo), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "decode":
        decode(*map(int, sys.argv[2:4])) if len(sys.argv) > 3 else decode()
    elif cmd == "render":
        render(int(sys.argv[2]), int(sys.argv[3]))
    else:
        {"probe": probe, "site": site}[cmd]()
