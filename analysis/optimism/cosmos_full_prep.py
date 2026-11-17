"""Cosmos T2/T3 全部重判的看图板（抽检主口径一致 51/60 = 85% < 90%，按批准方案全部重判）。
已抽检的 60 帧沿用其盲判标签；其余现实失败帧补判。固定种子打乱，盲判，只裁主视角。追加到 cosmos_items.json / cosmos_sheets.json。
  python3 cosmos_full_prep.py"""
import json, os, random
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
FW, FH, COLS, ROWS = 224, 224, 4, 3
CI = json.load(open(f"{HERE}/cosmos_items.json")); items = CI["items"]; sheets = json.load(open(f"{HERE}/cosmos_sheets.json"))
rows = json.load(open(f"{HERE}/cosmos_rows.json"))
done = {it["annotation_id"] for it in items if it["set"] == "cosmos_spot"}
assert not any(it["set"] == "cosmos_full" for it in items), "already prepared"
n0 = len(items); new = []
for T in ("T2", "T3"):
    pool = sorted([r for r in rows if r["model"] == "Cosmos" and r["T"] == T and not r["real_success"] and r["annotation_id"] not in done], key=lambda x: x["annotation_id"])
    random.Random(f"20261115/cosmos_full/{T}").shuffle(pool)
    new += [dict(set="cosmos_full", **r) for r in pool]
for i, it in enumerate(new):
    it["item"] = f"c{n0 + i:05d}"
font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18) if os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") else ImageFont.load_default()
for T in ("T2", "T3"):
    lst = [it for it in new if it["T"] == T]
    for s in range(0, len(lst), COLS * ROWS):
        chunk = lst[s:s + COLS * ROWS]; sname = f"cosmos_{T}full_{s // (COLS * ROWS) + 1:03d}"
        S = Image.new("RGB", (COLS * FW + (COLS + 1) * 6, ROWS * (FH + 22) + (ROWS + 1) * 6), "white"); d = ImageDraw.Draw(S)
        for k, it in enumerate(chunk):
            fp = f"{HERE}/cosmos_frames/{it['item']}.png"
            if not os.path.exists(fp):
                im = Image.open(it["png"]).convert("RGB"); im.crop((0, 0, im.width // 2, im.height)).save(fp)
            r_, c_ = divmod(k, COLS); x = 6 + c_ * (FW + 6); y = 6 + r_ * (FH + 22 + 6)
            d.text((x + 2, y), f"#{k + 1}", fill=(0, 0, 0), font=font)
            fr = Image.open(fp); fr.thumbnail((FW, FH)); S.paste(fr, (x, y + 22))
        S.save(f"{HERE}/sheets/{sname}.png"); sheets[sname] = [it["item"] for it in chunk]
CI["items"] = items + new
CI["full_rule"] = 'Cosmos T2/T3：抽检主口径一致 51/60 = 85% < 90%，全部重判；已抽检 60 帧沿用盲判标签，其余按 random.Random("20261115/cosmos_full/<T>") 打乱成板'
json.dump(CI, open(f"{HERE}/cosmos_items.json", "w"), ensure_ascii=False, indent=0)
json.dump(sheets, open(f"{HERE}/cosmos_sheets.json", "w"), indent=0)
from collections import Counter
print("new items", Counter(it["T"] for it in new), "new sheets", Counter(k.rsplit("_", 1)[0] for k in sheets if "full" in k))
