"""预筛乐观率 · 准备（纯 CPU）。
1) items.json：要判的每一帧（OpenWAM 现实失败 + OpenWAM T1 判定器已判的现实成功单元作校准 + LingBot 现实失败中已有 r3 卡片帧的）。
   每任务内按固定种子打乱，看图板只标板内编号，不显示现实成败、扰动档或判定器结果（盲判）。
2) frames/<item>.png（单帧存档）+ sheets/<T>_<k>.png（每板 12 格，4 列 × 3 行）+ sheets.json（板内编号 → item）。
   OpenWAM 帧 = imagined[-1][:256, ::-1]（主视角槽，与 r3 卡片同朝向）；LingBot 帧 = r3 卡片图（解码后末帧主视角）。
3) decode_pkg/：LingBot 其余现实失败单元（没有 r3 卡片的）的想象 latent + 清单 + 解码脚本，发 4090 解码末帧。
  python prep.py   （lerobot_libero，CPU）"""
import hashlib
import json
import os
import random
import shutil
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
R3 = os.path.join(OPT, "..", "annotation", "round3_imagined")
U = json.load(open(f"{OPT}/units.json"))
r3 = {c["unit_id"]: c for c in json.load(open(f"{R3}/manifest_r3.json"))["cards"]}
judged = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{OPT}/judged_openwam_T1.jsonl")}
SEED = 20261115
FW, FH, COLS, ROWS = 320, 256, 4, 3


def build_items():
    items = []
    for u in U:
        if u["model"] == "openwam" and not u["success"]:
            items.append(dict(set="ow_fail", **u))
        elif u["model"] == "openwam" and u["T"] == "T1" and u["unit_id"] in judged and u["success"]:
            items.append(dict(set="ow_t1_cal", **u))
        elif u["model"] == "lingbot" and not u["success"] and u["unit_id"] in r3:
            items.append(dict(set="lb_fail_r3", **u, r3_png=r3[u["unit_id"]]["png"]))
    by = {}
    for it in items:
        by.setdefault((it["model"], it["T"]), []).append(it)
    out = []
    for key in sorted(by):
        lst = sorted(by[key], key=lambda x: x["unit_id"])
        random.Random(f"{SEED}/{key}").shuffle(lst)
        out += lst
    for i, it in enumerate(out):
        it["item"] = f"p{i:05d}"
    return out


def load_frame(it):
    p = f"{HERE}/frames/{it['item']}.png"
    if not os.path.exists(p):
        if it["model"] == "openwam":
            im = np.load(it["frames"])["imagined"][-1]
            fr = Image.fromarray(np.ascontiguousarray(im[:256, ::-1]))
        else:
            fr = Image.open(it["r3_png"]).convert("RGB")
        fr.save(p)
    return p


def sheet(args):
    name, chunk = args
    W = COLS * FW + (COLS + 1) * 6; H = ROWS * (FH + 22) + (ROWS + 1) * 6
    S = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(S)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for k, it in enumerate(chunk):
        r, c = divmod(k, COLS)
        x = 6 + c * (FW + 6); y = 6 + r * (FH + 22 + 6)
        d.text((x + 2, y), f"#{k + 1}", fill=(0, 0, 0), font=font)
        fr = Image.open(load_frame(it)).convert("RGB")
        fr.thumbnail((FW, FH))
        S.paste(fr, (x + (FW - fr.width) // 2, y + 22))
    S.save(f"{HERE}/sheets/{name}.png")
    return name, [it["item"] for it in chunk]


def decode_pkg():
    need = [u for u in U if u["model"] == "lingbot" and not u["success"] and u["unit_id"] not in r3]
    D = f"{HERE}/decode_pkg"; os.makedirs(f"{D}/latents", exist_ok=True)
    man = []
    for u in need:
        z = np.load(u["frames"])
        pre = "imag_latents" if u["key"] == "imag_latents" else "latents"
        n = int(z["imag_n_chunks" if pre == "imag_latents" else "n_chunks"])
        arrs = {f"latents_{i:02d}": z[f"{pre}_{i:02d}"] for i in range(n)}
        out = f"{D}/latents/{u['unit_id']}.npz"
        np.savez_compressed(out, n_chunks=np.int64(n), **arrs)
        man.append(dict(unit_id=u["unit_id"], task=u["T"], source=u["source"], n_chunks=n, file=f"latents/{u['unit_id']}.npz",
                        sha256=hashlib.sha256(open(out, "rb").read()).hexdigest()))
    json.dump(dict(created="2026-11-14", purpose="LingBot-VA 现实失败单元想象末帧解码（预筛乐观率）", n=len(man), units=man),
              open(f"{D}/manifest.json", "w"), ensure_ascii=False, indent=1)
    return len(man)


def main():
    os.makedirs(f"{HERE}/frames", exist_ok=True); os.makedirs(f"{HERE}/sheets", exist_ok=True)
    items = build_items()
    json.dump(items, open(f"{HERE}/items.json", "w"), ensure_ascii=False, indent=0)
    jobs = []
    for key in sorted({(it["model"], it["T"]) for it in items}):
        lst = [it for it in items if (it["model"], it["T"]) == key]
        for s in range(0, len(lst), COLS * ROWS):
            jobs.append((f"{key[0]}_{key[1]}_{s // (COLS * ROWS) + 1:03d}", lst[s:s + COLS * ROWS]))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as p:
        res = p.map(sheet, jobs)
    json.dump({n: ids for n, ids in res}, open(f"{HERE}/sheets.json", "w"), indent=0)
    print("items", len(items), "sheets", len(res))
    print("decode_pkg units", decode_pkg())


if __name__ == "__main__":
    main()
