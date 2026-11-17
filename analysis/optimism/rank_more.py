"""LingBot 候选第二批：排序第 36–83 名（B 族除外）放大图。"""
import json, os
import numpy as np
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
R = [r for r in json.load(open(f"{HERE}/rank_diff.json")) if r["model"] == "lingbot" and r["family"] != "B"][36:84]
for page in range(0, len(R), 6):
    chunk = R[page:page + 6]; S = Image.new("RGB", (2 * (2 * 240 + 30), 3 * 270), (255, 255, 255)); dr = ImageDraw.Draw(S)
    for i, r in enumerate(chunk):
        c = np.load(f"{HERE}/cache/{r['unit_id']}.npz"); x, y = (i % 2) * (2 * 240 + 30), (i // 2) * 270
        S.paste(Image.fromarray(c["real"][-1]).resize((240, 240)), (x, y + 24)); S.paste(Image.fromarray(c["imagined"][-1]).resize((240, 240)), (x + 244, y + 24))
        dr.text((x + 2, y + 4), f"[L{36 + page + i}] {r['T']} {r['level']} diff={r['diff']}  left=REAL@400  right=IMAGINED last", fill=(0, 0, 0))
    S.save(f"{HERE}/review/lingbot_more_{page // 6}.png")
print(len(R))
