"""Round2（替换版，2026-11-12）：闭环真实画面卡片。
每格（模型 × 任务 × 档位，主表 20261112 口径）按种子抽 5 个单元；卡片 = 第一遍闭环主视角真实画面横条：
从第 0 步到最后一帧等间隔取 13 帧（每张卡帧数相同，横条长度不泄露成败），最右是最后一帧（成功单元 = 首次成功那一步，失败 = 第 400 步，统一口径）。
本脚本处理 Cosmos / UVA（队友已存真实闭环视频）；LingBot / OpenWAM 由重放脚本产出帧后用同一个 make_card。
  python real_cards.py list            -> units_cosmos_uva.json（抽样清单，不渲染）
  python real_cards.py render <shard> <nshard>
  python real_cards.py probe
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import csv
import glob
import json
import os
import sys
import zlib

import numpy as np
from PIL import Image, ImageDraw

_wmvp.bootstrap()
import ext_models as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/cards"
SEED, K, ROWH, NFRAMES, H = 20261112, 5, 168, 13, 400
COSMOS_ROOT = _wmvp.P("$COSMOS_DATA_ROOT/cosmos_policy")
TID = {"T0": "t5", "T1": "t3", "T2": "t2", "T3": "t4", "T4": "t6"}


def make_card(frames, last_idx):
    """frames: (T, h, w, 3) uint8 主视角真实画面（已正向）；last_idx: 最后一帧下标。"""
    idx = sorted(set(np.linspace(0, last_idx, NFRAMES).round().astype(int).tolist()))   # 固定帧数，横条长度不泄露成败
    ims = []
    for i in idx:
        im = Image.fromarray(np.ascontiguousarray(frames[i]))
        ims.append(im.resize((int(im.width * ROWH / im.height), ROWH)))
    gap = 6
    W = sum(i.width for i in ims) + gap
    cat = Image.new("RGB", (W, ROWH), (255, 255, 255))
    x = 0
    for j, im in enumerate(ims):
        if j == len(ims) - 1:
            x += gap                                  # 最后一帧前留白，便于辨认
        cat.paste(im, (x, 0)); x += im.width
    d = ImageDraw.Draw(cat)
    d.rectangle([W - ims[-1].width, 0, W - 1, ROWH - 1], outline=(255, 196, 0), width=3)
    return cat


def pick(units, key):
    rng = np.random.RandomState(zlib.crc32(f"{SEED}/{key}".encode()))
    units = sorted(units, key=lambda u: u["unit_id"])
    return [units[i] for i in sorted(rng.choice(len(units), size=min(K, len(units)), replace=False).tolist())]


def cosmos_units():
    cells = {}
    for r in csv.DictReader(open(f"{COSMOS_ROOT}/analysis/closed_loop_audit_20261111/unit_outcomes.csv", encoding="utf-8-sig")):
        lv = E.cosmos_level(r["condition"])
        if lv is None:
            continue
        d = os.path.dirname(f"{COSMOS_ROOT}/{r['source']}")
        vid = f"{d}/closed_loop1.mp4" if os.path.exists(f"{d}/closed_loop1.mp4") else f"{d}/closed_loop.mp4"
        ok = r["primary_success"] == "True"
        fs = int(float(r["primary_first_success_step"])) if ok and r["primary_first_success_step"] else None
        cells.setdefault((r["task"], lv), []).append(dict(
            model="cosmos", T=r["task"], level=lv, video=vid, success=ok, first_success_step=fs,
            unit_id=f"cosmos_libero10_{TID[r['task']]}_{os.path.basename(d)}__{lv}"))
    return cells


def uva_units():
    cells = {}
    for c in json.load(open(_wmvp.P("$WMV_ROOT/libero/out/task2/uva_index.json"))):
        lv = E.uva_level(c["family"], c["level"])
        if lv is None:
            continue
        for f in sorted(glob.glob(f"{c['closed_dir']}/*/repeat0/result.json")):
            r = json.load(open(f)); u = os.path.basename(os.path.dirname(os.path.dirname(f)))
            cells.setdefault((c["task"], lv), []).append(dict(
                model="uva", T=c["task"], level=lv, video=f"{os.path.dirname(f)}/native-closed.mp4",
                success=bool(r.get("native_success")), first_success_step=r.get("first_native_success_step"),
                unit_id=f"uva_libero10_{TID[c['task']]}_{u}__{lv}"))
    return cells


def frames_of(u):
    import imageio.v3 as iio
    fr = iio.imread(u["video"])
    if u["model"] == "cosmos":
        fr = fr[:, :, : fr.shape[2] // 2]             # 左半 = 主视角（右半腕部）
    return fr


def last_index(u, n_frames):
    # 视频第 0 帧 = 初始观测（UVA 401 帧）；Cosmos 视频第 i 帧 = 第 i+1 步后的画面（400 帧，成功即止）
    off = 0 if u["model"] == "cosmos" else 1
    tgt = (u["first_success_step"] if u["success"] and u["first_success_step"] else H) - 1 + off
    return max(0, min(tgt, n_frames - 1))


def main():
    cmd = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    if cmd == "list":
        sel = []
        for cells in (cosmos_units(), uva_units()):
            for (T, lv), us in sorted(cells.items()):
                sel += pick(us, f"{us[0]['model']}/{T}/{lv}")
        json.dump(sel, open(f"{HERE}/units_cosmos_uva.json", "w"), ensure_ascii=False, indent=0)
        print("selected", len(sel))
        return
    sel = json.load(open(f"{HERE}/units_cosmos_uva.json"))
    if cmd == "probe":
        sel = [next(u for u in sel if u["model"] == m and u["success"]) for m in ("cosmos", "uva")]
        for u in sel:
            fr = frames_of(u); li = last_index(u, len(fr))
            print(u["unit_id"], fr.shape, "last", li, u["first_success_step"])
            make_card(fr, li).save(f"{HERE}/probe_{u['model']}.png")
        return
    shard, nsh = int(sys.argv[2]), int(sys.argv[3])
    done = 0
    for u in sel:
        if zlib.crc32(u["unit_id"].encode()) % nsh != shard:
            continue
        png = f"{OUT}/{u['unit_id']}.png"
        if os.path.exists(png):
            continue
        try:
            fr = frames_of(u)
            make_card(fr, last_index(u, len(fr))).save(png, optimize=True)
            done += 1
        except Exception as e:  # noqa: BLE001
            print("SKIP", u["unit_id"], e, flush=True)
    print("RENDER_DONE shard", shard, done)


if __name__ == "__main__":
    main()
