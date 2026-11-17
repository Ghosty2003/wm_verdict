"""清晰度汇总表 v2（拉普拉斯方差，128×128 原生分辨率，越大越清楚）。读 openwam_sharpness.jsonl / lingbot_sharpness.jsonl。
OpenWAM-α：想象末帧缩到 128（BILINEAR，同 ow_view.pred_view）；读入画面 = 128 原生腐蚀帧（重建，见 sharpness.py）；干净画面 = 同时刻不加腐蚀。
  读入保留率 = 读入 ÷ 干净（逐单元，13 个读入时刻中位数之比）；
  想象保留率 = 想象末帧 ÷ 同模型同任务 B0 想象末帧的中位数；
  修复指数 = 想象保留率中位数 ÷ 读入保留率中位数（>1：想象比读入相对更清楚；≈1：想象沿用读入的模糊程度）。
画布尺度（320×256）的旧版数值保存在 openwam_sharpness_canvas_scale.jsonl：读入画面经 128→320 放大后接近下限，比值不稳定，弃用。
LingBot-VA：读入 = 存档实际读入帧（128）；想象末帧只有 r3 卡片里已解码的部分（卡片放大后重采样回 128，有平滑），且 LingBot 没跑 B0，暂不算保留率，只列数值。
不判成败，成功与失败单元都计入。  python3 sharp_report.py"""
import collections
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["B0", "Bg2", "Bg5", "Bg9", "Bf2", "Bf5", "Bf9", "Bz2", "Bz5", "Bz9"]
NAME = {"B0": "B0 对照（不加腐蚀）", "Bg": "高斯模糊", "Bf": "雾", "Bz": "缩放模糊"}


def q(xs):
    xs = sorted(xs)
    if not xs:
        return None
    n = len(xs)
    def p(f):
        k = (n - 1) * f; i = int(k); return xs[i] + (xs[min(i + 1, n - 1)] - xs[i]) * (k - i)
    return p(0.25), p(0.5), p(0.75)


def fq(xs, nd=0, pct=False):
    r = q(xs)
    if r is None:
        return "—"
    f = (lambda v: f"{100 * v:.{nd}f}%") if pct else (lambda v: f"{v:.{nd}f}")
    return f"{f(r[1])} [{f(r[0])}, {f(r[2])}]"


rows = {m: [json.loads(l) for l in open(f"{HERE}/{m}_sharpness.jsonl")] for m in ("openwam", "lingbot") if os.path.exists(f"{HERE}/{m}_sharpness.jsonl")}
out, L = [], []
for t in ("T3", "T4"):
    R = [r for r in rows.get("openwam", []) if r["task"] == t]
    b0 = st.median([r["imag_last_128"] for r in R if r["level"] == "B0"])
    L += ["", f"### OpenWAM-α · {t}（B0 想象末帧中位数 {b0:.0f}）", "",
          "| 档位 | 腐蚀 | n | 想象末帧 | 读入画面 | 干净画面 | 读入保留率 | 想象保留率 | 修复指数 |", "|---|---|---|---|---|---|---|---|---|"]
    for lv in ORDER:
        rs = [r for r in R if r["level"] == lv and r.get("input_median_128")]
        if not rs:
            continue
        im = [r["imag_last_128"] for r in rs]; inp = [r["input_median_128"] for r in rs]; cln = [r["clean_median_128"] for r in rs]
        ret_in = [r["input_median_128"] / r["clean_median_128"] for r in rs]; ret_im = [r["imag_last_128"] / b0 for r in rs]
        rep = st.median(ret_im) / st.median(ret_in)
        out.append(dict(model="openwam", task=t, level=lv, n=len(rs), imag=q(im), input=q(inp), clean=q(cln), ret_input=q(ret_in), ret_imag=q(ret_im), repair_index=rep))
        L.append(f"| {lv} | {NAME.get(lv, NAME.get(lv[:2]))} | {len(rs)} | {fq(im)} | {fq(inp)} | {fq(cln)} | {fq(ret_in, pct=True)} | {fq(ret_im, pct=True)} | {rep:.2f} |")
LB = rows.get("lingbot", [])
if LB:
    L += ["", "### LingBot-VA · T2（初步：想象末帧只有 r3 卡片已解码的子集，且无 B0 对照）", "",
          "| 档位 | 腐蚀 | n（有想象帧） | 读入画面 | 想象末帧（r3 卡片子集） |", "|---|---|---|---|---|"]
    for lv in ORDER[1:]:
        rs = [r for r in LB if r["level"] == lv]
        if not rs:
            continue
        im = [r["imag_last_r3card"] for r in rs if r.get("imag_last_r3card") is not None]
        out.append(dict(model="lingbot", task="T2", level=lv, n=len(rs), n_imag=len(im), input=q([r["input_median"] for r in rs]), imag=q(im)))
        L.append(f"| {lv} | {NAME[lv[:2]]} | {len(rs)}（{len(im)}） | {fq([r['input_median'] for r in rs])} | {fq(im)} |")
head = ["# 想象帧清晰度 vs 模型读入画面清晰度（B 族图像腐蚀）", "",
        "指标：灰度拉普拉斯方差，128×128 原生分辨率（腐蚀实际发生的分辨率），中位数 [四分位]。只看画面本身，不判成败，成功与失败单元都计入。",
        "OpenWAM 读入画面由重放干净帧按原规则重建（先腐蚀、后 180° 旋转、强度数字种子），在 LingBot 存档的实际读入帧上核对：三种腐蚀 94–98% 像素逐位相同（verify2.json）。",
        "读入保留率 = 读入 ÷ 同时刻干净画面；想象保留率 = 想象末帧 ÷ B0 想象末帧中位数；修复指数 = 想象保留率 ÷ 读入保留率（>1 表示想象比读入相对更清楚）。"]
open(f"{HERE}/sharpness_table.md", "w").write("\n".join(head + L) + "\n")
json.dump(out, open(f"{HERE}/sharpness_numbers.json", "w"), ensure_ascii=False, indent=1)
print("\n".join(head + L))
