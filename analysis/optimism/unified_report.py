"""四模型统一口径预筛乐观率总表（2026-11-14 批准方案）。数字全部读文件，标签按映射规则使用，不手改。
输入：prescreen_labels.jsonl（OpenWAM / LingBot 我们代判）；cosmos_rows.json（$COLLAB_COSMOS 初审标签）；cosmos_items.json + cosmos_sheets.json + judgments/c*.jsonl（Cosmos 我们代判）；
     ../confusion_openwam_T1.json（判定器）。
输出：unified_table.md、unified_numbers.json。
  python3 unified_report.py"""
import collections
import glob
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
THRESH = 0.90


def wilson(k, n):
    p = k / n; z = 1.96; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - s) / d), min(1.0, (c + s) / d)


def fmt(k, n):
    if not n:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.0f}%（{100 * lo:.0f}–{100 * hi:.0f}%）" + ("　⚠样本极小" if n < 10 else "")


# ---------- 映射规则（他们三选项 → 我们四选项；我们四选项 → 三类比较用）
THEIRS_TO_OURS = {"success": "looks_success", "failure": "not_success", "uncertain": "uncertain"}
OURS_TO3_MAIN = {"looks_success": "success", "borderline": "failure", "not_success": "failure", "uncertain": "uncertain"}
OURS_TO3_WIDE = {"looks_success": "success", "borderline": "success", "not_success": "failure", "uncertain": "uncertain"}

# ---------- Cosmos 我们代判标签
csheets = json.load(open(f"{HERE}/cosmos_sheets.json"))
citems = {it["item"]: it for it in json.load(open(f"{HERE}/cosmos_items.json"))["items"]}
cl = {}
for f in sorted(glob.glob(f"{HERE}/judgments/c*_cosmos_*.jsonl")):
    for l in open(f):
        r = json.loads(l); assert csheets[r["sheet"]][r["slot"] - 1] == r["item"]; cl[r["item"]] = r["label"]
# 抽检一致率
# 抽检一致率只用原始盲抽检（c3_cosmos_spot）的标签决定门槛；T3 抽检帧后来按全部重判的同一套规则再判过，那批标签只用于填表
spot_lab = {}
for l in open(f"{HERE}/judgments/c3_cosmos_spot.jsonl"):
    r = json.loads(l); spot_lab[r["item"]] = r["label"]
spot = [it for it in citems.values() if it["set"] == "cosmos_spot" and it["item"] in spot_lab]
agree_main = sum(OURS_TO3_MAIN[spot_lab[it["item"]]] == it["their_label"] for it in spot)
agree_wide = sum(OURS_TO3_WIDE[spot_lab[it["item"]]] == it["their_label"] for it in spot)
n_spot = len(spot); rate_main = agree_main / n_spot if n_spot else None
spot_ok = rate_main is not None and n_spot == 60 and rate_main >= THRESH
spot_cm = collections.Counter((it["their_label"], spot_lab[it["item"]]) for it in spot)

# ---------- 汇总单元
U = []   # dict(model, task, part, label, source)
PART = lambda fam: "F 族" if fam == "F" else "B 族（图像腐蚀）" if fam == "B" else "其余各族"
MN = {"lingbot": "LingBot-VA", "openwam": "OpenWAM-α"}
for l in open(f"{HERE}/prescreen_labels.jsonl"):
    r = json.loads(l)
    if r["set"] in ("ow_fail", "lb_fail_r3"):
        U.append(dict(model=MN[r["model"]], task=r["task"], level=r["level"], part=PART(r["family"]), label=r["agent_label"], source="我们代判"))
rows = json.load(open(f"{HERE}/cosmos_rows.json"))
t4_ours = {it["annotation_id"]: cl.get(it["item"]) for it in citems.values() if it["set"] == "cosmos_t4"}
t23_ours = {it["annotation_id"]: cl.get(it["item"]) for it in citems.values() if it["set"] in ("cosmos_spot", "cosmos_full")}   # 后写的判定覆盖先写的
for r in rows:
    if r["real_success"]:
        continue
    model = "Cosmos Policy" if r["model"] == "Cosmos" else "VPP2"
    lvl = "none" if r["level"] == "nominal" else r["level"]
    if model == "Cosmos Policy" and r["T"] == "T4":
        U.append(dict(model=model, task="T4", level=lvl, part=PART(r["family"]), label=t4_ours.get(r["annotation_id"]), source="我们代判"))
    elif model == "Cosmos Policy":
        if spot_ok:
            U.append(dict(model=model, task=r["T"], level=lvl, part=PART(r["family"]), label=THEIRS_TO_OURS[r["their_label"]],
                          source=f"他们代判（经 60 帧抽检，一致率 {100 * rate_main:.0f}%）"))
        else:
            U.append(dict(model=model, task=r["T"], level=lvl, part=PART(r["family"]), label=t23_ours.get(r["annotation_id"]),
                          source=f"我们代判（抽检一致 {agree_main}/{n_spot} = {100 * rate_main:.0f}% < 90%，全部重判）"))
    else:
        U.append(dict(model=model, task=r["T"], level=lvl, part=PART(r["family"]), label=THEIRS_TO_OURS[r["their_label"]], source="他们代判（未抽检）"))

# ---------- 第二批（2026-11-15）：Cosmos T0 新重跑 / T1 末窗恢复 / F 族（cosmos2_*），LingBot 待解码 146（lb2_*，CPU 解码）
def load_batch(items_file, sheets_file, judge_glob):
    if not os.path.exists(f"{HERE}/{items_file}"):
        return {}, {}
    sh = json.load(open(f"{HERE}/{sheets_file}"))
    its = {it["item"]: it for it in json.load(open(f"{HERE}/{items_file}"))["items"]}
    lab = {}
    for f in sorted(glob.glob(f"{HERE}/judgments/{judge_glob}")):
        for l in open(f):
            r = json.loads(l); assert sh[r["sheet"]][r["slot"] - 1] == r["item"]; lab[r["item"]] = r["label"]   # 同板重判以最后一条为准
    return its, lab
SRC2 = {"cz_T0": "我们代判（T0：11-13 全量新重跑，真实结果与终局帧同一次执行）", "cz_T1": "我们代判（T1：原执行末窗恢复）", "cz_F": "我们代判（F 族主遍）"}
c2_items, c2_lab = load_batch("cosmos2_items.json", "cosmos2_sheets.json", "c[789]_cz_*.jsonl")
for it in c2_items.values():
    U.append(dict(model="Cosmos Policy", task=it["T"], level=it["level"], part=PART(it["family"]), label=c2_lab.get(it["item"]), source=SRC2[it["set"]]))
l2_items, l2_lab = load_batch("lb2_items.json", "lb2_sheets.json", "l*_lb2_*.jsonl")
for it in l2_items.values():
    if not it["real_success"]:
        U.append(dict(model="LingBot-VA", task=it["T"], level=it["level"], part=PART(it["family"]), label=l2_lab.get(it["item"]), source="我们代判（CPU 解码）"))
batch2 = dict(cosmos2_items=len(c2_items), cosmos2_labeled=len(c2_lab), lb2_items=len(l2_items), lb2_labeled=len(l2_lab))

U = [u for u in U if not str(u.get("level") or "").startswith("NLw")]   # 2026-11-15 用户：去掉所有 NLw
G = collections.defaultdict(list)
for u in U:
    G[(u["model"], u["task"], u["part"])].append(u)
MORDER = {"LingBot-VA": 0, "OpenWAM-α": 1, "Cosmos Policy": 2, "VPP2": 3}
PORDER = {"其余各族": 0, "F 族": 1, "B 族（图像腐蚀）": 2}
T = []
for key in sorted(G, key=lambda k: (MORDER[k[0]], k[1], PORDER[k[2]])):
    us = G[key]; lab = [u["label"] for u in us if u["label"]]
    c = collections.Counter(lab); valid = len(lab) - c["uncertain"]
    T.append(dict(model=key[0], task=key[1], part=key[2], n_fail=len(us), n_labeled=len(lab), uncertain=c["uncertain"],
                  main=fmt(c["looks_success"], valid), wide=fmt(c["looks_success"] + c["borderline"], valid),
                  k_main=c["looks_success"], k_wide=c["looks_success"] + c["borderline"], n_valid=valid,
                  source=" / ".join(sorted({u["source"] for u in us}))))

# ---------- 按档位细分（2026-11-15）：模型 × 任务 × 档位；同时给主表现实失败数，看覆盖
import csv
MT = list(csv.DictReader(open(f"{OPT}/../closed_main/主表_长表_20261114.tsv"), delimiter="\t"))
MTF = {(r["model"], r["task"], r["level"]): int(r["n"]) - int(r["n_success"]) for r in MT}
LORDER = {}
for r in MT:
    LORDER.setdefault((r["model"], r["task"]), []).append(r["level"])
GL = collections.defaultdict(list)
for u in U:
    GL[(u["model"], u["task"], u["level"])].append(u)
BL = []
for (m, t, lv), us in GL.items():
    lab = [u["label"] for u in us if u["label"]]; c = collections.Counter(lab); valid = len(lab) - c["uncertain"]
    BL.append(dict(model=m, task=t, level=lv, part=us[0]["part"], n_fail_frames=len(us), n_fail_main=MTF.get((m, t, lv)), uncertain=c["uncertain"],
                   k=c["looks_success"], k_wide=c["looks_success"] + c["borderline"], n_valid=valid,
                   rate=c["looks_success"] / valid if valid else None,
                   lo=wilson(c["looks_success"], valid)[0] if valid else None, hi=wilson(c["looks_success"], valid)[1] if valid else None))
BL.sort(key=lambda r: (r["task"], MORDER[r["model"]], LORDER.get((r["model"], r["task"]), []).index(r["level"]) if r["level"] in LORDER.get((r["model"], r["task"]), []) else 999))
json.dump(dict(by_level=BL, level_order={f"{m}|{t}": v for (m, t), v in LORDER.items()},
               fail_main={f"{m}|{t}|{lv}": v for (m, t, lv), v in MTF.items()}), open(f"{HERE}/unified_by_level.json", "w"), ensure_ascii=False, indent=0)
cm = json.load(open(f"{OPT}/confusion_openwam_T1.json"))
jm, js = cm["末帧(主口径)"], cm["第400步帧(敏感性)"]
n_lb_pending = sum(1 for u in json.load(open(f"{OPT}/units.json")) if u["model"] == "lingbot" and not u["success"]) - \
    sum(1 for u in U if u["model"] == "LingBot-VA")

json.dump(dict(table=T, spot=dict(n=n_spot, agree_main=agree_main, agree_wide=agree_wide, rate_main=rate_main, threshold=THRESH, adopted=spot_ok,
                                  confusion={f"{a}|{b}": v for (a, b), v in spot_cm.items()}),
               judge_openwam_T1=dict(main=[jm["real_fail__imag_success"], jm["n_real_fail"]], step400=[js["real_fail__imag_success"], js["n_real_fail"]]),
               lingbot_pending_decode=n_lb_pending, batch2=batch2), open(f"{HERE}/unified_numbers.json", "w"), ensure_ascii=False, indent=1)

L = ["# 预筛乐观率 · 四模型统一口径总表（agent 代判、未校准、仅内部参考）", "",
     "对象：每个现实失败单元的想象末帧（最后一次规划的最后一帧，主视角），只看这一帧判“想象里任务完成了没有”；现实成败读记录的 success 字段。"
     "主口径 = 判成功 ÷（已判 − 判不了）；宽口径把 borderline 也算判成功。Wilson 95% CI。只计主表档位：NLw 与零剂量对照不计。", "",
     "| 模型 | 任务 | 范围 | 现实失败单元 | 已判 | 判不了 | 预筛乐观率（主口径） | 宽口径 | 判定来源 |", "|---|---|---|---|---|---|---|---|---|"]
L += [f"| {r['model']} | {r['task']} | {r['part']} | {r['n_fail']} | {r['n_labeled']} | {r['uncertain']} | {r['main']} | {r['wide']} | {r['source']} |" for r in T]
L += ["", "### 判定器与人工两条腿（同一指标的其他判定来源）", "",
      f"- OpenWAM-α · T1 · 判定器（OpenWAM t3 v4 联合）：主口径（末帧）{fmt(jm['real_fail__imag_success'], jm['n_real_fail'])}；"
      f"第 400 步帧 {fmt(js['real_fail__imag_success'], js['n_real_fail'])}。",
      "- 人工标注（r3，900 张）：标注现状待导出，出数后单列，口径优先于代判。",
      f"- LingBot-VA 另有 {n_lb_pending} 个现实失败单元等 4090 解码后补判；F 族 450 个“有判定器且有帧”单元的判定器判定等集群空 GPU，分列不混。", "",
      "### 口径映射（并表脚本按此执行，不手动改标签）", "",
      "- Cosmos 初审（$COLLAB_COSMOS）三选项 → 我们四选项：success → looks_success；failure → not_success；uncertain → uncertain（不进分母）。他们没有 borderline，因此他们来源的行宽口径 = 主口径。",
      "- 我们四选项 → 抽检比较用三类：主口径 borderline 记 failure；宽口径 borderline 记 success；uncertain 对 uncertain。",
      f"- Cosmos T2/T3 抽检：现实失败帧 T2、T3 各抽 30（numpy RandomState(crc32(\"20261115/cosmos_spot/<T>\")) 无放回，种子记档于 cosmos_items.json），"
      f"按我们的口径盲判后与他们的标签比：主口径一致 {agree_main}/{n_spot}" + (f" = {100 * agree_main / n_spot:.0f}%" if n_spot else "") +
      f"，宽口径一致 {agree_wide}/{n_spot}；门槛 90%，" + ("达到，沿用他们的标签。" if spot_ok else "未达到，按我们的口径全部重判（抽检过的 T3 30 帧按重判的同一套 T3 规则再判一次，以保持尺度一致；T2 抽检帧沿用）。不一致主要在：T3 两杯在正确盘上但被画面边缘切掉一部分（他们 success、抽检判 borderline），T2 他们 uncertain 而我们能判。"),
      "- Cosmos T4 按我们的 r3 口径（画面上布丁在盘子右边即算）对 745 个现实失败帧全部重判；VPP2 无扰动（现实失败 T2 1、T3 2、T4 4 个）沿用他们的标签。",
      "- 所有模型均只看主视角：Cosmos 终局帧为主视角 + 腕部并排，裁左半主视角判。",
      f"- 第二批（2026-11-15，同一套规则盲判，图板格间距加宽到 24 px）：Cosmos T0 = 11-13 全量新重跑的现实失败（真实结果与终局帧同一次执行；非 A/P 族的初始条件是官方 init0..4 × 种子 195..204，与我们不同），"
      f"Cosmos T1 = 原执行（主表那一遍）的末窗恢复，只覆盖主表 T1 现实失败的一部分，Cosmos F 族 = 主遍；LingBot 待解码单元改在 CPU 上解码（float32，与 GPU bf16 解码逐位不同、内容一致）。"
      f"已判 Cosmos {batch2['cosmos2_labeled']}/{batch2['cosmos2_items']}、LingBot {batch2['lb2_labeled']}/{batch2['lb2_items']}。", "",
      "B 族行的宽口径不可用：B 族里判 borderline 的帧大多是“腐蚀后关键物体看不清”（如雾中杯在盘上但看不到布丁、模糊色块在盘子附近），实质接近 uncertain，计入宽口径会虚高（如 Cosmos T4 B 族宽口径 72/72）；B 族以主口径为准。",
      "表注：想象判定是单帧视觉口径，是 BDDL 真值的近似；所有模型用同一标准，因此跨模型可比。agent 代判、未校准、仅内部参考，不进论文正文；人工标注（r3）与判定器口径优先。",
      "脚注：Cosmos 初审曾按 BDDL 目标区域严判 T4（主比例 0%，模糊界 0–67%，大量判为不确定），口径与本表不同，不进表。"]
open(f"{HERE}/unified_table.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
