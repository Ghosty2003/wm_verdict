"""Cosmos / VPP2 并表准备（预筛乐观率，2026-11-14 批准方案）。
来源：$COLLAB_COSMOS 的终局想象初审 terminal_imagination_agent_review_20261114.zip（Dropbox CosmosPolicy/.../Completed_20261114/）里的 human_review.csv，
     每行一个唯一终局帧：unit_id、conditions（共用帧以 " | " 分隔）、actual_success、agent_label（success/failure/uncertain）、terminal_png（NAS 路径）。
条件映射到主表档位（与 ext_models.cosmos_level 相同）；只属于 NLw 或零剂量对照（O0/Em0/Ef0/G0/K0/N_sham）的帧剔除；共用帧只要含一个主表档位即保留，按一个单元计。
1) cosmos_t4：Cosmos T4 现实失败帧，按我们的 r3 口径全部重判。
2) cosmos_spot：Cosmos T2、T3 现实失败帧各抽 30 个（numpy RandomState(crc32("20261115/cosmos_spot/<T>"))，无放回），按我们的口径重判，与他们的标签比一致率。
3) 其余（Cosmos T2/T3 全部、VPP2 无扰动）保留他们的标签，并表时按映射使用。
终局帧 448×224 = 左主视角 + 右腕部；与 OpenWAM / LingBot 一致，只裁左半主视角给代判看。看图板盲判：不显示他们的标签、现实成败或档位。
输出：cosmos_rows.json（全部映射后行）、cosmos_items.json、cosmos_sheets.json、cosmos_frames/、sheets/cosmos_*.png
  python3 cosmos_prep.py"""
import csv
import io
import json
import os
import random
import zipfile
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP = f"{HERE}/cosmos_src/review/terminal_imagination_agent_review_20261114.zip"
NR = {0.05: 1, 0.1: 2, 0.2: 3, 0.3: 4, 0.4: 5}
FW, FH, COLS, ROWS = 224, 224, 4, 3


def level_of(c):
    c = c.strip()
    if c == "baseline":
        return "none"
    if c == "N1_step50":
        return "N1_1"
    if c.startswith("NR_"):
        return f"NR{NR[round(float(c[3:]), 2)]}"
    if c.startswith("NL_"):
        i = int(c[3:]); return f"NL{i}" if i else None
    if c.startswith("NLw") or c in ("O0", "Em0", "Ef0", "G0", "K0", "N_sham"):
        return None
    return c


def fam(lv):
    if lv == "none":
        return "none"
    for p in ("NLw", "NL", "NR", "N1", "Em", "Ef"):
        if lv.startswith(p):
            return p
    return "B" if lv[0] == "B" else lv[0]


def main():
    z = zipfile.ZipFile(ZIP)
    H = list(csv.DictReader(io.StringIO(z.read("human_review.csv").decode("utf-8-sig"))))
    rows = []
    for r in H:
        if r["cohort"] != "historical_export_snapshot" or (r["model"] == "Cosmos" and r["task"] in ("T0", "T1")):
            continue                                       # Cosmos T0 历史批次 = NLw（不用）；T0/T1 新重跑另行导出
        lvls = [level_of(c) for c in r["conditions"].split("|")]
        lvls = [l for l in lvls if l]
        if not lvls:
            continue
        rows.append(dict(annotation_id=r["annotation_id"], model=r["model"], T=r["task"], level=lvls[0], family=fam(lvls[0]), all_levels=lvls,
                         unit_id=r["unit_id"], real_success=r["actual_success"] == "True", their_label=r["agent_label"],
                         their_confidence=r["agent_confidence"], their_reason=r["agent_reason"], png=r["terminal_png"]))
    json.dump(rows, open(f"{HERE}/cosmos_rows.json", "w"), ensure_ascii=False, indent=0)
    items = []
    t4 = sorted([r for r in rows if r["model"] == "Cosmos" and r["T"] == "T4" and not r["real_success"]], key=lambda x: x["annotation_id"])
    random.Random("20261115/cosmos_t4").shuffle(t4)
    items += [dict(set="cosmos_t4", **r) for r in t4]
    spot_seed = {}
    for T in ("T2", "T3"):
        pool = sorted([r for r in rows if r["model"] == "Cosmos" and r["T"] == T and not r["real_success"]], key=lambda x: x["annotation_id"])
        seed = zlib.crc32(f"20261115/cosmos_spot/{T}".encode()); spot_seed[T] = dict(seed=seed, pool=len(pool))
        idx = np.random.RandomState(seed).choice(len(pool), 30, replace=False)
        items += [dict(set="cosmos_spot", **pool[i]) for i in sorted(idx)]
    for i, it in enumerate(items):
        it["item"] = f"c{i:05d}"
    json.dump(dict(spot_rule='numpy RandomState(crc32("20261115/cosmos_spot/<T>")).choice(len(pool), 30, replace=False)，pool = 按 annotation_id 排序的现实失败帧',
                   spot_seed=spot_seed, items=items), open(f"{HERE}/cosmos_items.json", "w"), ensure_ascii=False, indent=0)
    os.makedirs(f"{HERE}/cosmos_frames", exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    sheets = {}
    groups = [("cosmos_T4", [it for it in items if it["set"] == "cosmos_t4"]),
              ("cosmos_T2spot", [it for it in items if it["set"] == "cosmos_spot" and it["T"] == "T2"]),
              ("cosmos_T3spot", [it for it in items if it["set"] == "cosmos_spot" and it["T"] == "T3"])]
    for name, lst in groups:
        for s in range(0, len(lst), COLS * ROWS):
            chunk = lst[s:s + COLS * ROWS]; sname = f"{name}_{s // (COLS * ROWS) + 1:03d}"
            W = COLS * FW + (COLS + 1) * 6; Hh = ROWS * (FH + 22) + (ROWS + 1) * 6
            S = Image.new("RGB", (W, Hh), "white"); d = ImageDraw.Draw(S)
            for k, it in enumerate(chunk):
                fp = f"{HERE}/cosmos_frames/{it['item']}.png"
                if not os.path.exists(fp):
                    im = Image.open(it["png"]).convert("RGB"); im.crop((0, 0, im.width // 2, im.height)).save(fp)
                r_, c_ = divmod(k, COLS); x = 6 + c_ * (FW + 6); y = 6 + r_ * (FH + 22 + 6)
                d.text((x + 2, y), f"#{k + 1}", fill=(0, 0, 0), font=font)
                fr = Image.open(fp); fr.thumbnail((FW, FH)); S.paste(fr, (x, y + 22))
            S.save(f"{HERE}/sheets/{sname}.png"); sheets[sname] = [it["item"] for it in chunk]
    json.dump(sheets, open(f"{HERE}/cosmos_sheets.json", "w"), indent=0)
    from collections import Counter
    print("rows", len(rows), Counter((r["model"], r["T"], r["real_success"]) for r in rows))
    print("items", Counter((it["set"], it["T"]) for it in items), "sheets", len(sheets), Counter(k.rsplit("_", 1)[0] for k in sheets))


if __name__ == "__main__":
    main()
