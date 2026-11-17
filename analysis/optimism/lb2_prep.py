"""预筛乐观率 · LingBot 待解码 146 单元并表准备（2026-11-15）。帧 = decode_pkg/out_cpu/<uid>.png（CPU 解码的想象末帧主视角 128×128）。
元信息取 optimism/units.json；按任务分板（判定标准随任务），12 格一板、格间距 24 px，板内按固定种子打乱，盲判。
输出：lb2_items.json、lb2_sheets.json、sheets/lb2_<T>_NNN.png
  python lb2_prep.py   （需要 PIL）"""
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
FW = FH = 224; COLS, ROWS, GAP, LAB = 4, 3, 24, 26


def main():
    U = {u["unit_id"]: u for u in json.load(open(f"{OPT}/units.json"))}
    man = json.load(open(f"{HERE}/decode_pkg/manifest.json"))["units"]
    items, miss = [], []
    for i, m in enumerate(man):
        png = f"{HERE}/decode_pkg/out_cpu/{m['unit_id']}.png"
        if not os.path.exists(png):
            miss.append(m["unit_id"]); continue
        u = U[m["unit_id"]]
        items.append(dict(item=f"l{i:05d}", set="lb2", model="lingbot", T=u["T"], level=u["level"], family=u["family"], unit_id=m["unit_id"],
                          source=u["source"], real_success=bool(u["success"]), png=png, decode="cpu float32 (decode_last_frame_cpu.py)"))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    sheets = {}
    for T in ("T0", "T1", "T2", "T3", "T4"):
        lst = [it for it in items if it["T"] == T]
        random.Random(f"20261115/lb2/{T}").shuffle(lst)
        for s in range(0, len(lst), COLS * ROWS):
            chunk = lst[s:s + COLS * ROWS]; sname = f"lb2_{T}_{s // (COLS * ROWS) + 1:03d}"
            W = COLS * FW + (COLS + 1) * GAP; H = ROWS * (FH + LAB) + (ROWS + 1) * GAP
            S = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(S)
            for k, it in enumerate(chunk):
                r_, c_ = divmod(k, COLS); x = GAP + c_ * (FW + GAP); y = GAP + r_ * (FH + LAB + GAP)
                d.text((x, y), f"#{k + 1}", fill=(0, 0, 0), font=font)
                S.paste(Image.open(it["png"]).convert("RGB").resize((FW, FH), Image.BICUBIC), (x, y + LAB))
                d.rectangle([x - 1, y + LAB - 1, x + FW, y + LAB + FH], outline=(120, 120, 120))
            S.save(f"{HERE}/sheets/{sname}.png"); sheets[sname] = [it["item"] for it in chunk]
    json.dump(dict(rule=__doc__, items=items, missing=miss), open(f"{HERE}/lb2_items.json", "w"), ensure_ascii=False, indent=0)
    json.dump(sheets, open(f"{HERE}/lb2_sheets.json", "w"), indent=0)
    from collections import Counter
    print("items", len(items), "missing", len(miss), Counter((it["T"], it["real_success"]) for it in items), "sheets", sorted(sheets))


if __name__ == "__main__":
    main()
