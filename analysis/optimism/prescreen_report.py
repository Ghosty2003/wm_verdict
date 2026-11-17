"""预筛乐观率汇总（agent 代判、未校准、仅内部参考）。数字全部读文件。
输入：items.json、sheets.json、judgments/*.jsonl（record.py 写入；同一 sheet+slot 取最后一条）、../judged_openwam_T1.jsonl（判定器）。
输出：prescreen_table.md、prescreen_numbers.json、prescreen_labels.jsonl（逐单元：单元、真实成败、代判标签）。
口径：只计现实失败单元；主口径 k = looks_success，n = 已判且非 uncertain；宽口径 k = looks_success + borderline；Wilson 95%。
F 族（F60/F160/F260）单独成行，不与其他族混合；OpenWAM T1 上与判定器（OpenWAM t3 v4 联合，末帧主口径）逐单元比对一致率。
  python3 prescreen_report.py"""
import collections
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
items = {it["item"]: it for it in json.load(open(f"{HERE}/items.json"))}
sheets = json.load(open(f"{HERE}/sheets.json"))
judge = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{OPT}/judged_openwam_T1.jsonl")}
MN = {"lingbot": "LingBot-VA", "openwam": "OpenWAM-α"}

lab = {}
for f in sorted(glob.glob(f"{HERE}/judgments/*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        assert sheets[r["sheet"]][r["slot"] - 1] == r["item"], r
        lab[r["item"]] = r                                   # 后写覆盖先写
done_sheets = {r["sheet"] for r in lab.values()}


def wilson(k, n):
    if not n:
        return None
    p = k / n; z = 1.96; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - s) / d), min(1.0, (c + s) / d)


def fmt(k, n):
    if not n:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.0f}%（{100 * lo:.0f}–{100 * hi:.0f}%）"


rows = []
with open(f"{HERE}/prescreen_labels.jsonl", "w") as fo:
    for it in items.values():
        r = lab.get(it["item"])
        fo.write(json.dumps(dict(item=it["item"], unit_id=it["unit_id"], model=it["model"], task=it["T"], level=it["level"], family=it["family"],
                                 source=it["source"], set=it["set"], real_success=it["success"], agent_label=r and r["label"],
                                 agent_note=r and r["note"], judge=r and r["judge"]), ensure_ascii=False) + "\n")

grp = collections.defaultdict(lambda: collections.Counter())
total = collections.Counter()
for it in items.values():
    if it["set"] not in ("ow_fail", "lb_fail_r3"):
        continue
    part = "F 族" if it["family"] == "F" else "B 族（图像腐蚀）" if it["family"] == "B" else "其余各族"
    key = (it["model"], it["T"], part)
    total[key] += 1
    r = lab.get(it["item"])
    if r:
        grp[key][r["label"]] += 1
T = []
PORDER = {"其余各族": 0, "F 族": 1, "B 族（图像腐蚀）": 2}
for key in sorted(total, key=lambda k: (k[0], k[1], PORDER[k[2]])):
    c = grp[key]; n_lab = sum(c.values()); valid = n_lab - c["uncertain"]
    T.append(dict(model=MN[key[0]], task=key[1], part=key[2], n_fail_units=total[key], n_labeled=n_lab, uncertain=c["uncertain"],
                  looks_success=c["looks_success"], borderline=c["borderline"], not_success=c["not_success"],
                  main=fmt(c["looks_success"], valid), wide=fmt(c["looks_success"] + c["borderline"], valid),
                  main_k=c["looks_success"], wide_k=c["looks_success"] + c["borderline"], n_valid=valid))

# OpenWAM T1 与判定器一致率（非 F 族、判定器已判的单元）
cmp_rows = []
for it in items.values():
    if it["model"] == "openwam" and it["T"] == "T1" and it["unit_id"] in judge and it["item"] in lab:
        a = lab[it["item"]]["label"]
        cmp_rows.append((judge[it["unit_id"]]["real_success"], judge[it["unit_id"]]["imag_success"], a))


def agree(rows_, wide):
    ok = [(j, a in (("looks_success", "borderline") if wide else ("looks_success",))) for _, j, a in rows_ if a != "uncertain"]
    k = sum(j == a for j, a in ok); n = len(ok)
    m = collections.Counter((j, a) for j, a in ok)
    return dict(k=k, n=n, text=fmt(k, n), j1a1=m[(True, True)], j1a0=m[(True, False)], j0a1=m[(False, True)], j0a0=m[(False, False)])


agr = dict(all_main=agree(cmp_rows, False), all_wide=agree(cmp_rows, True),
           fail_main=agree([r for r in cmp_rows if not r[0]], False), n_uncertain=sum(1 for r in cmp_rows if r[2] == "uncertain"))
pending_lb = [u for u in json.load(open(f"{OPT}/units.json")) if u["model"] == "lingbot" and not u["success"]
              and not any(it["unit_id"] == u["unit_id"] for it in items.values())]
json.dump(dict(table=T, agreement_openwam_T1=agr, sheets_done=len(done_sheets), sheets_total=len(sheets), labeled_items=len(lab),
               pending_lingbot_decode=len(pending_lb)), open(f"{HERE}/prescreen_numbers.json", "w"), ensure_ascii=False, indent=1)

L = ["# 预筛乐观率（agent 代判、未校准、仅内部参考）", "",
     f"进度：看图板 {len(done_sheets)}/{len(sheets)} 张已判，{len(lab)} 帧有标签。LingBot 另有 {len(pending_lb)} 个现实失败单元等 4090 解码后补判。", "",
     "口径：对象是每个现实失败单元的想象末帧（最后一次规划的最后一帧；OpenWAM 为 imagined[-1]，LingBot 为解码后末帧），只看这一帧判“想象里任务完成了没有”，r3 四选项。"
     "现实成败直接读记录里的 success 字段。主口径 = looks_success ÷（已判 − uncertain）；宽口径把 borderline 也算判成功。Wilson 95% CI。"
     "看图时盲判：板上不显示现实成败、扰动档或判定器结果。F 族单独成行；B 族单独成行——模型读入腐蚀画面后想象帧沿用同样的腐蚀外观（Bg 高斯模糊最明显，约一半帧糊到看不出物体、记为 uncertain 不进分母；Bf 雾和 Bz 缩放模糊大多仍可判），不与其他族混排。", "",
     "| 模型 | 任务 | 范围 | 现实失败单元 | 已判 | uncertain | 预筛乐观率（主口径） | 宽口径（含 borderline） |", "|---|---|---|---|---|---|---|---|"]
L += [f"| {r['model']} | {r['task']} | {r['part']} | {r['n_fail_units']} | {r['n_labeled']} | {r['uncertain']} | {r['main']} | {r['wide']} |" for r in T]
a = agr["all_main"]
L += ["", "## OpenWAM-α T1：代判与判定器（OpenWAM t3 v4 联合，末帧主口径）逐单元一致率", "",
      f"全部已判单元（含现实成功的校准单元）：主口径一致 {a['text']}；宽口径一致 {agr['all_wide']['text']}；代判为 uncertain 的 {agr['n_uncertain']} 个不计。",
      f"主口径 2×2（判定器 × 代判）：判定器成功且代判成功 {a['j1a1']}，判定器成功代判否 {a['j1a0']}，判定器否代判成功 {a['j0a1']}，两者都否 {a['j0a0']}。",
      f"只看现实失败单元：{agr['fail_main']['text']}（样本极小）。", "",
      "表注：agent 代判、未校准、仅内部参考，不进论文正文；人工标注（r3）与判定器口径优先。"]
open(f"{HERE}/prescreen_table.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
