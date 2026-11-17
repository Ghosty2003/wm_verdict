"""候选初筛：按 想象末帧 与 现实第 400 步帧 的像素差排序（差得大 = 想象偏离现实，是"想象完成而现实失败"的必要条件），
每个模型取前 N 个放大成大图供人工挑选。只是排序辅助，最终挑选由人看图决定。
  python rank_view.py [N]
"""
import glob, json, os, sys
import numpy as np
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
meta = {u["unit_id"]: u for u in json.load(open(f"{HERE}/units.json"))}
rows = []
for f in sorted(glob.glob(f"{HERE}/cache/*.npz")):
    c = np.load(f); uid = str(c["unit"]); u = meta[uid]
    im = Image.fromarray(c["imagined"][-1]).resize((128, 128)); re = Image.fromarray(c["real"][-1]).resize((128, 128))
    d = float(np.abs(np.asarray(im, np.float32) - np.asarray(re, np.float32)).mean())
    rows.append(dict(unit_id=uid, model=u["model"], T=u["T"], level=u["level"], family=u["family"], diff=round(d, 2)))
json.dump(sorted(rows, key=lambda r: -r["diff"]), open(f"{HERE}/rank_diff.json", "w"), indent=0)
os.makedirs(f"{HERE}/review", exist_ok=True)
for m in ("lingbot", "openwam"):
    top = sorted([r for r in rows if r["model"] == m and r["family"] != "B"], key=lambda r: -r["diff"])[:N]   # B 族：模型读入腐蚀画面，想象本就是腐蚀样，排除
    for page in range(0, len(top), 6):
        chunk = top[page:page + 6]; S = Image.new("RGB", (2 * (2 * 240 + 30), 3 * 270), (255, 255, 255)); dr = ImageDraw.Draw(S)
        for i, r in enumerate(chunk):
            c = np.load(f"{HERE}/cache/{r['unit_id']}.npz"); x, y = (i % 2) * (2 * 240 + 30), (i // 2) * 270
            S.paste(Image.fromarray(c["real"][-1]).resize((240, 240)), (x, y + 24)); S.paste(Image.fromarray(c["imagined"][-1]).resize((240, 240)), (x + 244, y + 24))
            dr.text((x + 2, y + 4), f"[{m[0].upper()}{page + i}] {r['T']} {r['level']} diff={r['diff']}  left=REAL@400  right=IMAGINED last", fill=(0, 0, 0))
        S.save(f"{HERE}/review/{m}_top_{page // 6}.png")
json.dump({m: [r for r in sorted([r for r in rows if r["model"] == m and r["family"] != "B"], key=lambda r: -r["diff"])[:N]] for m in ("lingbot", "openwam")},
          open(f"{HERE}/review/index.json", "w"), indent=0)
print("ranked", len(rows))
