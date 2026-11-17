"""预筛乐观率 · 第二批 Cosmos 并表准备（2026-11-15 用户：“这些先分析掉，加到我们的最终表格里面去”）。
三个新来源，只取现实失败、主表档位（F 族为新族，单列）：
  cz_T0：T0 全量新重跑 cosmos_closed_rerun_20261113（同一次执行同时有真实结果与终局帧；非 A/P 族用官方 init0..4 × 种子 195..204）；
  cz_T1：T1 末窗恢复 t1_terminal_restore_20261114（原执行 = 主表那一遍，只重算最后一个规划窗口）；
  cz_F ：F 族 f_family_20261114（primary_for_statistics 那一遍）。
终局帧若为主视角+腕部并排，裁左半主视角（与第一批相同）。看图板盲判：不显示来源、档位、现实成败。
12 格一板（4×3），格间距 24 px（第一批 6 px 曾导致相邻格看串）；板内顺序按固定种子打乱。
输出：cosmos2_items.json、cosmos2_sheets.json、cosmos2_frames/、sheets/cz_*.png
  python cosmos2_prep.py     （需要 PIL；在节点上用 lerobot_libero 跑）"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import csv
import glob
import json
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SA = _wmvp.P("$COSMOS_DATA_ROOT")
_wmvp.bootstrap()
import ext_models as E  # noqa: E402

MT = {(r["model"], r["task"], r["level"]) for r in csv.DictReader(open(_wmvp.P("$WMV_ROOT/libero/out/closed_main/主表_长表_20261114.tsv")), delimiter="\t")}
FW = FH = 224; COLS, ROWS, GAP, LAB = 4, 3, 24, 26


def fam(lv):
    if lv == "none":
        return "none"
    for p in ("NLw", "NL", "NR", "N1", "Em", "Ef"):
        if lv.startswith(p):
            return p
    return "B" if lv[0] == "B" else lv[0]


def main():
    items = []
    for res in sorted(glob.glob(f"{SA}/cosmos_closed_rerun_20261113/T0/*/units/*/closed_loop_result.json")):
        d = os.path.dirname(res); cond = d.split("/")[-3]; lv = E.cosmos_level(cond)
        if lv is None or ("Cosmos Policy", "T0", lv) not in MT:
            continue
        r = json.load(open(res)); png = glob.glob(f"{d}/*_terminal.png")
        if r.get("success") or not png:
            continue
        u = json.load(open(f"{d}/unit.json"))
        items.append(dict(set="cz_T0", T="T0", level=lv, family=fam(lv), condition=cond, unit=f"T0/{cond}/{os.path.basename(d)}",
                          anchor_id=u.get("anchor_id"), policy_seed=u.get("policy_seed"), png=png[0], real_success=False,
                          first_success_step=r.get("first_success_step"), source="cosmos_closed_rerun_20261113"))
    for e in json.load(open(f"{SA}/t1_terminal_restore_20261114/terminal_manifest.json")):
        lv = E.cosmos_level(e["condition"])
        if lv is None or ("Cosmos Policy", "T1", lv) not in MT or e["original_success"] or not e.get("terminal_png"):
            continue
        items.append(dict(set="cz_T1", T="T1", level=lv, family=fam(lv), condition=e["condition"], unit=e["unit_id"],
                          anchor_id=f"anchor{e['anchor_index']}", policy_seed=e["policy_seed"], png=e["terminal_png"], real_success=False,
                          first_success_step=e.get("original_first_success_step"), source="t1_terminal_restore_20261114 (original episode, last window recomputed)"))
    for e in json.load(open(f"{SA}/f_family_20261114/terminal_manifest.json")):
        if not e.get("primary_for_statistics") or e["actual_success"] or not e.get("terminal_png"):
            continue
        T = e["unit_id"].split("/")[1]
        items.append(dict(set="cz_F", T=T, level=e["condition"], family="F", condition=e["condition"], unit=e["unit_id"],
                          anchor_id=e["anchor"].get("anchor_id"), policy_seed=e.get("policy_seed"), png=e["terminal_png"], real_success=False,
                          first_success_step=e.get("first_success_step"), source="f_family_20261114 (primary pass)"))
    os.makedirs(f"{HERE}/cosmos2_frames", exist_ok=True); os.makedirs(f"{HERE}/sheets", exist_ok=True)
    for i, it in enumerate(items):
        it["item"] = f"z{i:05d}"
        fp = f"{HERE}/cosmos2_frames/{it['item']}.png"
        im = Image.open(it["png"]).convert("RGB"); it["png_size"] = list(im.size)
        if im.width >= 2 * im.height - 4:
            im = im.crop((0, 0, im.width // 2, im.height)); it["cropped_left_half"] = True
        im.save(fp)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    sheets = {}
    groups = [("cz_T0", lambda it: it["set"] == "cz_T0"), ("cz_T1", lambda it: it["set"] == "cz_T1")] + \
             [(f"cz_F_{T}", (lambda T: lambda it: it["set"] == "cz_F" and it["T"] == T)(T)) for T in ("T0", "T1", "T2", "T3", "T4")]   # F 族按任务分板（判定标准随任务）
    for name, sel in groups:
        lst = [it for it in items if sel(it)]
        random.Random(f"20261115/{name}").shuffle(lst)
        for s in range(0, len(lst), COLS * ROWS):
            chunk = lst[s:s + COLS * ROWS]; sname = f"{name}_{s // (COLS * ROWS) + 1:03d}"
            W = COLS * FW + (COLS + 1) * GAP; H = ROWS * (FH + LAB) + (ROWS + 1) * GAP
            S = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(S)
            for k, it in enumerate(chunk):
                r_, c_ = divmod(k, COLS); x = GAP + c_ * (FW + GAP); y = GAP + r_ * (FH + LAB + GAP)
                d.text((x, y), f"#{k + 1}", fill=(0, 0, 0), font=font)
                fr = Image.open(f"{HERE}/cosmos2_frames/{it['item']}.png"); fr = fr.resize((FW, FH), Image.BICUBIC)
                S.paste(fr, (x, y + LAB)); d.rectangle([x - 1, y + LAB - 1, x + FW, y + LAB + FH], outline=(120, 120, 120))
            S.save(f"{HERE}/sheets/{sname}.png"); sheets[sname] = [it["item"] for it in chunk]
    json.dump(dict(rule=__doc__, items=items), open(f"{HERE}/cosmos2_items.json", "w"), ensure_ascii=False, indent=0)
    json.dump(sheets, open(f"{HERE}/cosmos2_sheets.json", "w"), indent=0)
    from collections import Counter
    print("items", Counter((it["set"], it["T"]) for it in items), "sheets", Counter(k.rsplit("_", 1)[0] for k in sheets),
          "sizes", Counter(tuple(it["png_size"]) for it in items))


if __name__ == "__main__":
    main()
