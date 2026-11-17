"""同卷对比标注 · 第 5 步（v4，2026-11-16，按组）：
A. 全表（所有页、所有卡）：
   1) 每张卡的图片文件存在（库里旧卡 / 本次出图）；页面表（网站）与 pages.json 一致；页上不出现 seed / anchor / 代用字段。
   2) 同一运行（新增）：每行同时有 REAL 与 IMAGINED 卡时，二者是同一个 unit_id，且
      a) inventory 的 real_run == imag_run（运行标识规则见 inventory.py 文件头）；
      b) file:<路径> 运行：真实卡与想象卡实际所用的文件（新出图取 render.*.jsonl 记录的 path；旧卡取盘点路径）都在同一个文件 / 目录里；
      c) 机器一致：真实卡与想象卡的来源路径要么都在 out/h100/ 下，要么都不在；
      d) rec:<uid> 运行：两边都不是 H100，且想象来源路径（或 LingBot 解码来源）里含该单元的 anchor 写法与种子序号 / 名称。
      任何一处不符都记 MISMATCH，退出码 1。
   3) 档位：每张卡 unit_id 解析出的档位后缀与页的档位一致；模型与行一致。
   4) UVA 待重跑格（O、Bg/Bf/Bz、F、T1 K9 / T2 K9 / T3 K8 / T3 K9 / T4 K9）：UVA 行必须留白（只有 note，无卡）。
B. 抽 10 页（numpy RandomState(crc32("20261116/groups/check")).choice，T0/T1 与 T2–T4 各 5 页）拼核对图 check/page_XX.png 供人工目视。
输出 check/check_report.json。  python check_pages.py
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
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = _wmvp.P("$WMV_WEB_ROOT")
H100 = _wmvp.P("$WMV_ROOT/libero/out/h100/")


def level_of(model, uid):
    uid = re.sub(r"__(x86rerun|h100)$", "", uid)
    if model in ("lingbot", "openwam"):
        return uid.rsplit("__", 1)[1] if uid.count("__") >= 2 else "none"     # T0/T1 物理档：<base>__none__Em1 → 取最后一段
    return uid.rsplit("__", 1)[1]


def img_of(kind, e):
    rnd = 2 if kind == "real" else 3
    old = e["r2_card" if kind == "real" else "r3_card"]
    return old or next((x for x in (f"{WEB}/cards/r{rnd}/{e['unit_id']}.webp",) if os.path.exists(x)), None)


def ident(uid):
    """单元自己的 anchor 写法与种子名（用于 rec 运行的路径核对）。"""
    m = re.search(r"libero10_t\d_([a-z]+\d+)__(k\d+)", uid)
    if m:
        return [m.group(1), m.group(2)]
    m = re.search(r"anchor\d+_(seed\d+)", uid)
    if m:
        return [m.group(1)]
    m = re.search(r"([a-z]+\d+_k\d+)", uid)
    return [m.group(1)] if m else [uid]


def main():
    pages = json.load(open(f"{HERE}/pages.json"))["pages"]
    site = {p["id"]: p for p in json.load(open(f"{WEB}/data/samecase_pages.json"))["pages"]}
    inv = {}
    for l in open(f"{HERE}/inventory.jsonl"):
        u = json.loads(l); inv[u["unit_id"]] = u
    rlog = {}
    for f in sorted(glob.glob(f"{HERE}/render.*.jsonl")):
        for l in open(f):
            x = json.loads(l)
            if isinstance(x.get("result"), dict):
                rlog[(x["kind"], x["unit_id"])] = x["result"]
    bad, cnt = [], collections.Counter()
    forbidden = {"seed_no", "seed_value", "anchor", "k", "proxy", "slot"}
    for p in pages:
        sp = site.get(p["page_id"])
        if sp is None:
            bad.append(dict(page=p["page_id"], why="page missing on site")); continue
        if forbidden & set(sp) or any(forbidden & set(r) for r in sp["rows"]) or "seed" in sp["id"].lower():
            bad.append(dict(page=p["page_id"], why="seed/anchor/proxy field on site page"))
        for r, srow in zip(p["rows"], sp["rows"]):
            assert r["model"] == srow["model"]
            if r.get("note") or srow.get("note"):                       # UVA 待重跑留白：不得有任何卡
                cnt["uva_blank_rows"] += 1
                if r["model"] != "uva" or srow.get("real") or srow.get("imag") or r.get("real") or r.get("imag"):
                    bad.append(dict(page=p["page_id"], model=r["model"], why="blank row has a card or non-UVA blank"))
                continue
            if r["model"] == "uva" and (p["level"][:1] in ("O", "F") or p["level"][:2] in ("Bg", "Bf", "Bz")
                                        or (p["T"], p["level"]) in {("T1", "K9"), ("T2", "K9"), ("T3", "K8"), ("T3", "K9"), ("T4", "K9")}):
                bad.append(dict(page=p["page_id"], model="uva", why="UVA cell should be blank (pending rerun) but is not"))
            for kind in ("real", "imag"):
                e = r[kind]
                if not e or not srow[kind]:
                    continue
                cnt[f"cards_{kind}"] += 1
                if srow[kind] != e["unit_id"]:
                    bad.append(dict(page=p["page_id"], model=r["model"], kind=kind, why="site unit != pages.json unit"))
                if not img_of(kind, e):
                    bad.append(dict(page=p["page_id"], model=r["model"], kind=kind, why="image missing"))
                if level_of(r["model"], e["unit_id"]) != p["level"] or not e["unit_id"].startswith(("cosmos" if r["model"] == "cosmos" else r["model"])):
                    bad.append(dict(page=p["page_id"], model=r["model"], kind=kind, unit=e["unit_id"], why="level/model suffix mismatch"))
            if not (srow["real"] and srow["imag"]):
                continue
            # ---- 同一运行
            cnt["pairs"] += 1
            re_, im_ = r["real"], r["imag"]
            why = []
            if re_["unit_id"] != im_["unit_id"]:
                why.append("real/imag unit_id differ")
            u = inv.get(re_["unit_id"])
            if u is None:
                why.append("unit not in inventory")
            else:
                if not (u["real_run"] and u["real_run"] == u["imag_run"]):
                    why.append(f"run ids differ: {u['real_run']} vs {u['imag_run']}")
                rp = (rlog.get(("real", u["unit_id"])) or {}).get("path") or u["real_path"]
                ip = (rlog.get(("imag", u["unit_id"])) or {}).get("src_path") or u["imag_path"]
                if ("/lingbot" in str(ip) or u["model"] == "lingbot") and str(ip).startswith(f"{WEB}/cards/"):
                    ip = u["imag_run"][5:] if u["imag_run"].startswith("file:") else ip    # 旧 round 3 卡：来源 = 盘点记下的解码文件
                ip = str(ip).split("#")[0]
                if (H100 in str(rp)) != (H100 in ip) or ((u["machine"] == "H100") != (H100 in ip)):
                    why.append("machine differs (H100 vs cluster)")
                run = u["real_run"] or ""
                if run.startswith("file:"):
                    tgt = os.path.normpath(run[5:])
                    for nm, pth in (("real", rp), ("imag", ip)):
                        pth = os.path.normpath(str(pth))
                        if not (pth == tgt or pth.startswith(tgt + "/") or os.path.dirname(pth) == tgt
                                or (nm == "real" and "/replay_frames/" in pth and os.path.basename(pth) == os.path.basename(tgt))):
                            why.append(f"{nm} file {pth} not in run {tgt}")
                    cnt["pairs_file_run"] += 1
                elif run.startswith("rec:"):
                    if u["machine"] == "H100":
                        why.append("rec run on H100")
                    if ip.startswith(f"{WEB}/cards/"):
                        cnt["pairs_rec_old_card"] += 1                 # 库里旧想象卡：来源按出卡时的原执行文件（盘点规则），不再按路径核
                    elif not all(re.search(re.escape(t) + r"(?!\d)", ip) for t in ident(u["unit_id"])):
                        why.append(f"imag source path does not name the unit: {ip}")
                    cnt["pairs_rec_run"] += 1
            if why:
                bad.append(dict(page=p["page_id"], model=r["model"], unit=re_["unit_id"], why="; ".join(why)))
                cnt["pairs_MISMATCH"] += 1
    # ---- 抽 10 页拼核对图
    rng = np.random.RandomState(zlib.crc32(b"20261116/groups/check"))
    i01 = [i for i, p in enumerate(pages) if p["T"] in ("T0", "T1")]; i24 = [i for i, p in enumerate(pages) if p["T"] not in ("T0", "T1")]
    pick = sorted(rng.choice(i01, 5, replace=False).tolist() + rng.choice(i24, 5, replace=False).tolist())
    os.makedirs(f"{HERE}/check", exist_ok=True)
    for f in glob.glob(f"{HERE}/check/page_*.png"):
        os.remove(f)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    sheets = []
    for n, i in enumerate(pick):
        p = pages[i]; sp = site[p["page_id"]]
        sheet = Image.new("RGB", (2300, 60 + 250 * len(p["rows"])), "white"); d = ImageDraw.Draw(sheet)
        d.text((10, 10), f"{p['page_id']}  (group {p['group']})", fill=(0, 0, 0), font=font)
        y = 50
        for r, srow in zip(p["rows"], sp["rows"]):
            ri = img_of("real", r["real"]) if r["real"] and srow["real"] else None
            ii = img_of("imag", r["imag"]) if r["imag"] and srow["imag"] else None
            u = inv.get(srow["real"] or "", {})
            d.text((10, y), f"{r['model']}  REAL {srow['real'] or '无'}  [{u.get('machine', '')}, success={u.get('success')}]", fill=(0, 100, 90), font=font)
            d.text((1700, y), f"IMAG {'有' if ii else '无'}", fill=(110, 50, 200), font=font)
            if ri:
                im = Image.open(ri).convert("RGB"); im = im.resize((int(im.width * 150 / im.height), 150))
                sheet.paste(im.crop((0, 0, min(im.width, 1650), 150)), (10, y + 22))
            if ii:
                im = Image.open(ii).convert("RGB"); sheet.paste(im.resize((int(im.width * 220 / im.height), 220)), (1700, y + 22))
            y += 250
        fn = f"{HERE}/check/page_{n:02d}.png"; sheet.save(fn); sheets.append(dict(page=p["page_id"], sheet=fn))
    rep = dict(counts=dict(cnt), n_bad=len(bad), bad=bad, sampled=sheets)
    json.dump(rep, open(f"{HERE}/check/check_report.json", "w"), ensure_ascii=False, indent=1)
    print("CHECK", json.dumps(dict(cnt), ensure_ascii=False))
    print("BAD", len(bad), json.dumps(bad[:20], ensure_ascii=False))
    print("sampled", [s["page"] for s in sheets])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
