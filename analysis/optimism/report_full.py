"""终局想象乐观率 · 完整结果包（脚本生成，数字全部读文件，不手抄）。
输入：coverage.tsv、units.json（opt_coverage.py）；confusion_openwam_T1.json（judge_ow_t1.py）；
      ../annotation/round3_imagined/manifest_r3.json 与 给Jasmine_r3说明.md（抽样规则原句）；
      r3_annotations_export.json（可选；wm_verdict_web/tools/export_round_annotations.py 由 Jasmine 本机导出）；
      case_selection.json（案例图选定状态，Jasmine 改）。
输出：report_full.md、report_full.html（网页，图片引用 figures/*.png）、report_full_numbers.json（全部数字）。
  python3 report_full.py
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import collections
import csv
import html
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R3 = os.path.join(HERE, "..", "annotation", "round3_imagined")
TODAY = "2026-11-15"
TN = {"t5": "T0", "t3": "T1", "t2": "T2", "t4": "T3", "t6": "T4"}
MNAME = {"lingbot": "LingBot-VA", "openwam": "OpenWAM-α"}
JUDGE = {("lingbot", "T0"): "LingBot t5 v3", ("lingbot", "T1"): "LingBot t3 v4 联合", ("openwam", "T1"): "OpenWAM t3 v4 联合"}
MIN_N_HUMAN = 10          # 人工口径：某格已标现实失败卡 >= 10 张才出初步数（本报告暂定）


def wilson(k, n):
    if not n:
        return None
    p = k / n; z = 1.96; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d, (c + s) / d


def pct(x):
    return f"{100 * x:.0f}%"


def rate(k, n):
    """'k/n = xx%（95% CI a–b%）'"""
    if not n:
        return "—（n = 0）"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {pct(k / n)}（95% CI {pct(lo)}–{pct(hi)}）"


# ---------------- 读数据
cov = [r for r in csv.DictReader(open(f"{HERE}/coverage.tsv"), delimiter="\t") if r["family"] != "NLw"]
U = [u for u in json.load(open(f"{HERE}/units.json")) if not u["level"].startswith("NLw")]   # 2026-11-15 用户：去掉所有 NLw
cm = json.load(open(f"{HERE}/confusion_openwam_T1.json"))
judged_ids = {json.loads(l)["unit_id"] for l in open(f"{HERE}/judged_openwam_T1.jsonl")}
M3 = json.load(open(f"{R3}/manifest_r3.json"))
rule_line = next(l for l in open(f"{R3}/给Jasmine_r3说明.md", encoding="utf-8") if l.startswith("抽样规则"))
rule_sentence = rule_line.split("：", 1)[1].strip()
EXP = f"{HERE}/r3_annotations_export.json"
exp = json.load(open(EXP)) if os.path.exists(EXP) else None
cases = json.load(open(f"{HERE}/case_selection.json"))
cases["candidates"] = [c for c in cases["candidates"] if "__NLw" not in c["unit_id"]]   # 2026-11-15 去掉 NLw 案例（文件保留）

# 覆盖汇总
tot = collections.Counter(); fails = collections.Counter()
for r in cov:
    tot[(r["model"], r["task"])] += int(r["n_units"]); fails[(r["model"], r["task"])] += int(r["n_real_fail"])
srcs = collections.Counter(u["source"] for u in U)

# 自动判定：有判定器且有帧
inter = [u for u in U if u["judge"]]
judged = [u for u in inter if u["model"] == "openwam" and u["T"] == "T1" and u["unit_id"] in judged_ids]
pending = [u for u in inter if not (u["model"] == "openwam" and u["T"] == "T1" and u["unit_id"] in judged_ids)]
pend = collections.defaultdict(lambda: [0, 0, set()])
for u in pending:
    p = pend[(u["model"], u["T"], u["judge"])]; p[0] += 1; p[1] += (not u["success"]); p[2].add(u["source"])
mT = cm["末帧(主口径)"]; s4 = cm["第400步帧(敏感性)"]
assert mT["n_units"] == len(judged), (mT["n_units"], len(judged))

# 人工标注（r3）
cards = M3["cards"]
r3c = collections.Counter((c["model"], TN[c["task"]], c["group"]) for c in cards)
r3_mt = sorted({(c["model"], TN[c["task"]]) for c in cards}, key=lambda x: (x[0], x[1]))
human = {}
if exp:
    assert exp["n_cards"] == len(cards), ("导出卡片数与 manifest 不符", exp["n_cards"], len(cards))
    byid = {c["id"]: c for c in exp["cards"]}
    man = {c["unit_id"]: c for c in cards}
    lab = collections.defaultdict(list)
    for a in exp["annotations"]:
        lab[byid[a["card_id"]]["unit_id"]].append(a["label"])

    def card_label(ls):
        c = collections.Counter(ls).most_common()
        if len(c) > 1 and c[0][1] == c[1][1]:
            return "borderline"                       # 平票按 borderline 计（暂定）
        return c[0][0]
    for mt in r3_mt:
        fl = [u for u, c in man.items() if (c["model"], TN[c["task"]]) == mt and c["group"] == "fail_all"]
        done = [card_label(lab[u]) for u in fl if lab.get(u)]
        valid = [x for x in done if x != "uncertain"]
        human[mt] = dict(n_fail_cards=len(fl), n_annotated=len(done), n_valid=len(valid),
                         k_main=sum(x == "looks_success" for x in valid), k_wide=sum(x in ("looks_success", "borderline") for x in valid),
                         n_uncertain=len(done) - len(valid))
    n_done_cards = len({a["card_id"] for a in exp["annotations"]})

numbers = dict(generated=TODAY, coverage_total=sum(tot.values()), coverage_by_model_task={f"{m}|{t}": [tot[(m, t)], fails[(m, t)]] for m, t in tot},
               openwam_T1_auto=dict(main=mT, step400=s4), pending_auto={f"{m}|{t}": dict(judge=j, n=v[0], n_fail=v[1], sources=sorted(v[2])) for (m, t, j), v in pend.items()},
               r3=dict(n_cards=len(cards), export=bool(exp), exported_at=exp and exp["exported_at"], n_annotated_cards=exp and n_done_cards,
                       per_model_task={f"{m}|{t}": v for (m, t), v in human.items()}), cases=cases["candidates"])
json.dump(numbers, open(f"{HERE}/report_full_numbers.json", "w"), ensure_ascii=False, indent=1, default=list)

# ---------------- 文本块（md 与 html 共用）
DEF = [
    ("指标", "终局想象乐观率 = 现实失败的闭环单元中，想象末帧被判为“任务已完成”的比例。这是主报告数字；现实成败 × 想象判定 2×2 混淆矩阵的其余格（含现实成功单元）放附录。"),
    ("单元范围", "所有存了闭环想象帧的单元，第一遍闭环，H = 400。现实成败取想象帧所属那一遍闭环的模拟器判定（同一遍，不混用两遍）。只读现有数据：不为这个指标补采想象帧，也不新建判定器（11-13 裁决）。"),
    ("想象末帧", "最后一次规划调用预测的最后一帧，主视角。OpenWAM-α 双口径：第 13 次调用覆盖第 384–416 步，闭环只执行到第 400 步；主口径取 imagined[-1]（名义第 416 步），敏感性对照取第 400 步那一帧（最后一次调用的第 4 帧，帧距 4 步）。LingBot-VA 的想象帧以 latent 存档，用 LingBot 的 VAE 解码后取最后一帧（解码与 LingBotVAPolicy._decode_predicted_video 逐行相同，已与 rig.decode 的缓存结果逐位比对）；LingBot 每 16 步重读一次真实画面。"),
    ("判定腿一：自动", "凡“有部署判定器且有帧”的单元全部判（包括现实成功单元），判定器单帧作用在想象末帧上，出 2×2 混淆矩阵。部署判定器：OpenWAM t3 = OpenWAM 视角 v4 联合（p_close ≥ 阈值且 p_bowl_out < 阈值）；LingBot t5 = v3；LingBot t3 = v4 联合；OpenWAM t5 没有（三次未过门槛）；t2 / t4 / t6 两个模型都没有。"),
    ("判定腿二：人工", "第三轮标注 r3_imagined：每张卡只放想象末帧（不放真实画面），问“想象画面是否显示任务已完成”，选项 looks_success / borderline / not_success / uncertain。人工乐观率 = 现实失败卡片中判 looks_success 的比例，表注写“人工标注口径”。汇总规则（本报告暂定，待确认）：同一张卡多人标注取多数，平票按 borderline 计；uncertain 不进分母；把 borderline 也算作判成功的宽口径作敏感性对照；某格已标现实失败卡不少于 " + str(MIN_N_HUMAN) + " 张才出初步数，否则列“待标注”。"),
    ("置信区间", "全部比例给 k/n 和 Wilson 95% 置信区间（z = 1.96）。"),
    ("覆盖限制", "自动腿只有判定器覆盖的 模型×任务 能出数：LingBot T0/T1 和 OpenWAM T1。T2–T4 两个模型都没有判定器，OpenWAM T0 的判定器没过门槛，这些格只能走人工腿。LingBot T0/T1 原来没有闭环想象帧，F 族按 11-13 存档规矩跑、想象帧齐全，是它们第一次进入“有判定器且有帧”的交集。"),
    ("抽样规则（原句）", rule_sentence),
]

cov_rows = [[MNAME[r["model"]], r["task"], r["family"], r["n_units"], r["n_real_fail"], r["judge"], r["sources"]] for r in cov]
old_total = sum(int(r["n_units"]) for r in csv.DictReader(open(f"{HERE}/backup_20261113/coverage.tsv"), delimiter="\t"))
cov_note = (f"合计 {sum(tot.values())} 个单元（11-13 版为 {old_total} 个）。新增来源：lb_rerun（LingBot T2–T4 全量重跑，2026-11-14 应要求暂停，暂停时完成 {srcs['lb_rerun']} 个单元）、"
            f"lb_famF / ow_famF（F 族，两模型各 {srcs['lb_famF']} / {srcs['ow_famF']} 个单元）。各来源单元数："
            + "、".join(f"{k} {v}" for k, v in sorted(srcs.items())) + "。")
mt_rows = [[MNAME[m], t, tot[(m, t)], fails[(m, t)], JUDGE.get((m, t), "无")] for m, t in sorted(tot, key=lambda x: (x[0], x[1]))]


def matrix(c):
    return [["现实成功", c["real_success__imag_success"], c["real_success__imag_fail"]],
            ["现实失败", c["real_fail__imag_success"], c["real_fail__imag_fail"]]]


auto_main = rate(mT["real_fail__imag_success"], mT["n_real_fail"])
auto_400 = rate(s4["real_fail__imag_success"], s4["n_real_fail"])
succ_main = rate(mT["real_success__imag_success"], mT["n_real_success"])
succ_400 = rate(s4["real_success__imag_success"], s4["n_real_success"])
N4 = (f"显著性提醒：现实失败只有 n = {mT['n_real_fail']} 个单元，95% CI 很宽。这一格只能说明“在极少数失败单元上观察到想象判成功”，不能当作强结论引用。")
pend_rows = [[MNAME[m], t, j, v[0], v[1], " ".join(sorted(v[2]))] for (m, t, j), v in sorted(pend.items())]
pend_note = ("这些单元已经有帧也有判定器，但还没判：LingBot 要先用 GPU 解码 latent，判定器也写死在 GPU 上跑。"
             "2026-11-14 GPU 让给队友，这部分暂缓，判完后 LingBot T0/T1 出 2×2 和乐观率，OpenWAM T1 的 F 族单元并入同一张矩阵。"
             f"现实失败数很少（LingBot T0 {pend[('lingbot', 'T0', JUDGE[('lingbot', 'T0')])][1]} 个、T1 {pend[('lingbot', 'T1', JUDGE[('lingbot', 'T1')])][1]} 个），出数后同样要带显著性提醒。")

if exp:
    hum_rows = []
    for mt in r3_mt:
        h = human[mt]
        if h["n_valid"] >= MIN_N_HUMAN:
            main_, wide_ = rate(h["k_main"], h["n_valid"]), rate(h["k_wide"], h["n_valid"])
        else:
            main_ = wide_ = f"待标注（有效 {h['n_valid']} 张，不足 {MIN_N_HUMAN}）"
        hum_rows.append([MNAME[mt[0]], mt[1], h["n_fail_cards"], h["n_annotated"], h["n_uncertain"], main_, wide_])
    hum_status = f"导出时间 {exp['exported_at']}：900 张卡片中已标 {n_done_cards} 张（{exp['n_annotations']} 条标注）。"
else:
    hum_rows = [[MNAME[m], t, r3c[(m, t, "fail_all")], "—", "—", "待标注（标注现状待导出）", "—"] for m, t in r3_mt]
    hum_status = ("标注现状待导出：标注存在 Supabase，读取需要服务密钥，只能由 Jasmine 本机导出。"
                  "导出后重跑 report_full.py，本节自动填数。")
EXPORT_CMD = (_wmvp.P("cd $WMV_WEB_ROOT && export SUPABASE_SERVICE_KEY=... && "
              "python3 tools/export_round_annotations.py --round-id 3 --out $WMV_ROOT/libero/out/optimism/r3_annotations_export.json"))
r3_rows = [[MNAME[m], t, r3c[(m, t, "fail_all")], r3c[(m, t, "success_control")], r3c[(m, t, "fail_all")] + r3c[(m, t, "success_control")]] for m, t in r3_mt]
case_rows = [[c["file"].split("/")[-1], MNAME.get(c["model"], c["model"]), c["task"], c["level"], "失败" if c["real_success"] is False else "成功", c["status"]] for c in cases["candidates"]]
n_sel = sum(c["status"] == "已选" for c in cases["candidates"])

# ---------------- markdown
def mdtable(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]


L = [f"# 终局想象乐观率 · 完整结果包（{TODAY}，脚本生成）", "",
     "由 `libero/out/optimism/report_full.py` 生成，数字全部读文件，不手抄。网页版：report_full.html。", "",
     "## 0. 定义", ""]
for k, v in DEF:
    L += [f"**{k}**：{v}", ""]
L += ["## 1. 覆盖面", "", cov_note, "", "### 1.1 按 模型 × 任务", ""] + mdtable(["模型", "任务", "存帧单元", "现实失败", "判定器"], mt_rows)
L += ["", "### 1.2 按 模型 × 任务 × 扰动族", ""] + mdtable(["模型", "任务", "族", "单元数", "现实失败", "判定器", "来源"], cov_rows)
L += ["", "## 2. 自动判定", "", f"### 2.1 OpenWAM-α · T1 · {mT['judge']} · n = {mT['n_units']}", "",
      f"**乐观率（主口径，末帧）：{auto_main}**", "", f"敏感性对照（第 400 步帧）：{auto_400}", "", f"**{N4}**", "",
      "2×2 混淆矩阵（主口径，末帧）：", ""] + mdtable(["", "想象判成功", "想象判失败"], matrix(mT))
L += ["", "2×2 混淆矩阵（敏感性，第 400 步帧）：", ""] + mdtable(["", "想象判成功", "想象判失败"], matrix(s4))
L += ["", f"附：现实成功中想象也判成功，主口径 {succ_main}；第 400 步帧 {succ_400}。", "",
      "### 2.2 新进入“有判定器且有帧”交集、待判定的单元", ""] + mdtable(["模型", "任务", "判定器", "单元数", "现实失败", "来源"], pend_rows)
L += ["", pend_note, "", "## 3. 人工标注（r3_imagined，人工标注口径，标注进行中）", "", hum_status, "", "r3 卡片构成：", ""]
L += mdtable(["模型", "任务", "现实失败卡", "成功对照卡", "合计"], r3_rows)
L += ["", "初步人工乐观率（按 模型 × 任务）：", ""] + mdtable(["模型", "任务", "现实失败卡", "已标", "uncertain", "乐观率（主口径）", "宽口径（含 borderline）"], hum_rows)
L += ["", f"表注：人工标注口径、标注进行中。OpenWAM T3/T4 的现实失败单元为抽样（各 60 个），其余格全收。LingBot T0/T1 不在 r3 里（出 r3 时还没有闭环想象帧）。", ""]
if not exp:
    L += ["导出命令（服务密钥只在本机）：", "", "```", EXPORT_CMD, "```", ""]

# ---------------- 预筛乐观率（四模型统一口径）与清晰度（2026-11-15 并入；读 prescreen/unified_numbers.json、sharpness/sharpness_numbers.json）
UN = json.load(open(f"{HERE}/prescreen/unified_numbers.json"))
SH = json.load(open(f"{HERE}/sharpness/sharpness_numbers.json"))
PRE_FN = "预筛口径（agent 代判、未校准），正式数字以 r3 人工标注为准。"
pre_head = ["模型", "任务", "范围", "现实失败", "判不了", "预筛乐观率（主口径）", "宽口径", "判定来源"]
pre_rows = [[r["model"], r["task"], r["part"], r["n_fail"], r["uncertain"], r["main"], "不可用" if r["part"].startswith("B") else r["wide"], r["source"]] for r in UN["table"]]
sp = UN["spot"]
pre_notes = [
    "对象：每个现实失败单元的想象末帧（主视角），只看这一帧判“想象里任务完成了没有”。主口径 = 判成功 ÷（已判 − 判不了）；宽口径把 borderline 也算判成功；Wilson 95% CI。只计主表档位。",
    f"Cosmos T2/T3 先抽 {sp['n']} 帧比对 $COLLAB_COSMOS 初审标签：主口径一致 {sp['agree_main']}/{sp['n']}，未达 {100 * sp['threshold']:.0f}% 门槛，按我们的口径全部重判；Cosmos T4 按 r3 口径全部重判；VPP2 沿用他们的标签。",
    "B 族宽口径不可用：B 族 borderline 多为“腐蚀后关键物体看不清”，实质接近判不了。Cosmos 初审曾按 BDDL 目标区域严判 T4，口径不同，不进表。",
    "读法（代判口径，待人工标注验证）：乐观是跨模型普遍现象，T3/T4 三个模型都明显非零，差异是梯度而非有无；LingBot-VA 各任务最低，T2 为零、T3/T4 非零，随任务难度上升（旧表述“LingBot 零整任务乐观”为早期样本结论，已修订）；F 族（致盲）失败局乐观率明显高于同模型其余族，记为“致盲抬高想象自评”。",
]
MT = list(csv.DictReader(open(f"{HERE}/../closed_main/主表_长表_20261114.tsv"), delimiter="\t"))
mt_fail = lambda m, t: sum(int(r["n"]) - int(r["n_success"]) for r in MT if r["model"] == m and r["task"] == t)
_NR = {"0.05": "NR1", "0.1": "NR2", "0.2": "NR3", "0.3": "NR4", "0.4": "NR5"}
def _lv(c):
    if c == "baseline": return "none"
    if c.startswith("NR_"): return _NR[c[3:]]
    if c.startswith("NL_"): return "NL" + c[3:]
    return c
_idx = json.load(open(f"{HERE}/prescreen/cosmos_src/indexes_20261115/Cosmos_T1_terminal_completed_part000.json"))
_t1u = {p.split("/")[2] for p in ((x if isinstance(x, str) else x.get("path") or x.get("name")) for x in _idx["entries"]) if p.endswith("_terminal.png")}
_t1lv = collections.Counter(_lv(u.split("__")[1]) for u in _t1u)
_t1fail = {r["level"]: int(r["n"]) - int(r["n_success"]) for r in MT if r["model"] == "Cosmos Policy" and r["task"] == "T1"}
t1_overlap_max = sum(min(v, _t1fail.get(k, 0)) for k, v in _t1lv.items())
have_r3 = os.path.exists(f"{HERE}/r3_annotations_export.json")
MISSING = [
    ("LingBot-VA 待解码", f"{UN['lingbot_pending_decode']} 个现实失败单元的想象帧只有 latent，解码包已上传 Dropbox（optimism_prescreen/decode_pkg_20261114），等 4090 解码回传后补判。"),
    ("Cosmos T0", f"主表现实失败 {mt_fail('Cosmos Policy', 'T0')} 个。终局图在 T0 全量重跑包里（每单元一张 terminal.png），但包合计上百 GB；需 $COLLAB_COSMOS 单独导出失败单元终局图的小包。"),
    ("Cosmos T1", f"主表现实失败 {mt_fail('Cosmos Policy', 'T1')} 个。已上传的 T1 终局恢复包只恢复了 {len(_t1u)} 个单元（" + "、".join(f"{k} {v}" for k, v in _t1lv.most_common()) + f"），与现实失败最多重叠 {t1_overlap_max} 个；其余要等继续恢复。"),
    ("F 族判定器", f"{len(pending)} 个“有判定器且有帧”的 F 族单元（" + "、".join(f"{MNAME[k.split(chr(124))[0]]} {k.split(chr(124))[1]} {v[0]}" for k, v in ((f"{m}|{t}", v) for (m, t, j), v in pend.items())) + "）等集群空 GPU 跑判定器，结果与代判分列。"),
    ("r3 人工标注", "已导出，见人工标注一节。" if have_r3 else "尚未导出（r3_annotations_export.json 不存在）。需在有 service key 的机器上跑 export_round_annotations.py；导出后所有乐观率数字以人工口径替换预筛口径。"),
    ("VPP2", "只有无扰动档，现实失败 " + "、".join(f"{r['task']} {r['n_fail']}" for r in UN["table"] if r["model"] == "VPP2") + " 个，数太小，只列不比。"),
]
sh_rows = [[r["task"], r["level"], r["n"], f"{100 * r['ret_input'][1]:.0f}%", f"{100 * r['ret_imag'][1]:.0f}%", f"{r['repair_index']:.2f}"] for r in SH if r["model"] == "openwam"]
sh_head = ["任务", "档位", "n", "读入保留率", "想象保留率", "修复指数"]
sh_note = ("OpenWAM-α，灰度拉普拉斯方差，128×128 原生分辨率，中位数。读入保留率 = 模型读入画面 ÷ 同时刻干净画面；想象保留率 = 想象末帧 ÷ 同任务 B0 想象末帧中位数；"
           "修复指数 = 想象保留率 ÷ 读入保留率（>1 表示想象比读入相对更清楚）。档位后缀为强度（g 高斯模糊、f 雾、z 缩放模糊）。成功与失败单元都计入，不判成败。"
           "读法：想象帧比读入画面相对清楚（修复指数 > 1，弱档尤其明显），但想象保留率随强度一路下降，最强档高斯模糊与雾的想象帧几乎和读入一样糊：腐蚀外观穿透进想象（腐蚀穿透）。这也是 B 族宽口径不可用的原因。")
L += ["## 3b. 预筛乐观率 · 四模型统一口径（agent 代判、未校准、仅内部参考）[^pre]", ""] + pre_notes[:1] + [""] + mdtable(pre_head, pre_rows) + [""] + [f"- {x}" for x in pre_notes[1:]]
L += ["", "### 3c. 待补数据", ""] + [f"- **{k}**：{v}" for k, v in MISSING]
L += ["", "### 3d. 想象帧清晰度（B 族图像腐蚀）", "", sh_note, ""] + mdtable(sh_head, sh_rows) + ["", f"[^pre]: {PRE_FN}", ""]

# ---------------- 按扰动档细分（读 prescreen/unified_by_level.json；主表现实失败数一起列，看想象帧覆盖）
BYL = json.load(open(f"{HERE}/prescreen/unified_by_level.json"))
BLD = {(r["model"], r["task"], r["level"]): r for r in BYL["by_level"]}
FM = {tuple(k.split("|")): v for k, v in BYL["fail_main"].items()}
LO = {tuple(k.split("|")): v for k, v in BYL["level_order"].items()}
BM = ["LingBot-VA", "OpenWAM-α", "Cosmos Policy", "VPP2"]
def byl_task(t):
    ms = [m for m in BM if any(k[0] == m and k[1] == t for k in BLD)] or BM[:3]
    order = []
    for m in ["OpenWAM-α", "LingBot-VA", "Cosmos Policy", "VPP2"]:   # 行序按档位最全的 OpenWAM 主表顺序
        for lv in LO.get((m, t), []):
            if lv not in order: order.append(lv)
    rows = []
    for lv in [x for x in order if not x.startswith("NLw")]:
        cells = []
        for m in ms:
            f = FM.get((m, t, lv)); r = BLD.get((m, t, lv))
            if r and r["n_valid"]:
                cells.append(dict(kind="rate", rate=r["rate"], k=r["k"], n=r["n_valid"], unc=r["uncertain"], frames=r["n_fail_frames"], fail=f))
            elif r:
                cells.append(dict(kind="unc", frames=r["n_fail_frames"], fail=f))
            elif f:
                cells.append(dict(kind="noframe", fail=f))
            else:
                cells.append(dict(kind="nofail" if f == 0 else "notrun"))
        if any(c["kind"] not in ("nofail", "notrun") for c in cells):
            rows.append((lv, cells))
    cov = {m: (sum(r["n_fail_frames"] for k, r in BLD.items() if k[0] == m and k[1] == t), sum(v for k, v in FM.items() if k[0] == m and k[1] == t)) for m in ms}
    return ms, rows, cov
def cell_txt(c):
    if c["kind"] == "rate":
        x = f"{100 * c['rate']:.0f}%（{c['k']}/{c['n']}）"
        if c["fail"] and c["frames"] < c["fail"]: x += f" 有帧 {c['frames']}/{c['fail']}"
        if c["unc"]: x += f" 判不了 {c['unc']}"
        return x
    return {"unc": lambda: f"全判不了（{c['frames']}）", "noframe": lambda: f"无帧（失败 {c['fail']}）", "nofail": lambda: "无失败", "notrun": lambda: "—"}[c["kind"]]()
BYL_TASKS = [t for t in ("T0", "T1", "T2", "T3", "T4")]
BYL_NOTE = ("每格 = 该档现实失败单元中想象末帧判成功的比例（主口径，k/n）；“有帧 a/b”= 有想象帧的失败单元 a 个 / 主表该档现实失败 b 个，只有 a 个进了分母；"
            "“无帧”= 主表该档有失败但没存想象帧；“无失败”= 该档全部成功。单档 n 普遍很小，单档数字只看方向，比较请用上方按族汇总的总表。" + PRE_FN)
L += ["", "## 3e. 按扰动档细分（预筛口径）", "", BYL_NOTE, ""]
for t in BYL_TASKS:
    ms, rows, cov = byl_task(t)
    L += [f"### {t}", "", "想象帧覆盖（有帧失败 / 主表失败）：" + "；".join(f"{m} {a}/{b}" for m, (a, b) in cov.items()), ""]
    L += mdtable(["档位"] + ms, [[lv] + [cell_txt(c) for c in cs] for lv, cs in rows]) + [""]
L += ["## 4. 对比案例候选", "", f"11-13 裁决出 6 张对比案例图；下列 {len(cases["candidates"])} 张为候选（NLw 案例 2026-11-15 起不展示），已选 {n_sel} 张。选定状态记在 case_selection.json，改完重跑本脚本。", ""]
L += mdtable(["文件", "模型", "任务", "档位", "现实", "状态"], case_rows)
L += ["", "## 5. 文件", "", "- 覆盖清点：opt_coverage.py → units.json、coverage.tsv",
      "- 自动判定：judge_ow_t1.py → judged_openwam_T1.jsonl、confusion_openwam_T1.json",
      "- 人工标注导出：wm_verdict_web/tools/export_round_annotations.py → r3_annotations_export.json",
      "- 本报告：report_full.py → report_full.md、report_full.html、report_full_numbers.json",
      "- 11-13 版本存档：backup_20261113/"]
open(f"{HERE}/report_full.md", "w").write("\n".join(L) + "\n")

# ---------------- html
e = html.escape


def htable(head, rows, cls=""):
    h = "".join(f"<th>{e(str(x))}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{e(str(x))}</td>" for x in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def hmatrix(c, caption):
    return (f'<figure class="mx"><figcaption>{e(caption)}</figcaption><div class="mxg">'
            f'<span></span><span class="mh">想象判成功</span><span class="mh">想象判失败</span>'
            f'<span class="mh row">现实成功</span><b>{c["real_success__imag_success"]}</b><b>{c["real_success__imag_fail"]}</b>'
            f'<span class="mh row">现实失败</span><b class="hot">{c["real_fail__imag_success"]}</b><b>{c["real_fail__imag_fail"]}</b></div></figure>')


lo, hi = wilson(mT["real_fail__imag_success"], mT["n_real_fail"])
stat = [("自动判定 · OpenWAM T1", f'{mT["real_fail__imag_success"]}/{mT["n_real_fail"]}', f"乐观率 {pct(mT['optimism_rate'])}，CI {pct(lo)}–{pct(hi)}"),
        ("存帧单元", f"{sum(tot.values())}", f"现实失败 {sum(fails.values())}"),
        ("待判定（有判定器有帧）", f"{len(pending)}", f"其中现实失败 {sum(1 for u in pending if not u['success'])}"),
        ("r3 人工标注", f"{n_done_cards}/{len(cards)}" if exp else f"?/{len(cards)}", "已标 / 卡片" if exp else "待导出")]
H = [f'<title>终局想象乐观率</title><meta name="claude:viewport-fit-sides" content="cover">',
     '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
     '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Noto+Sans+SC:wght@400;500;700&display=swap">',
     "<style>", open(f"{HERE}/report_full.css").read(), "</style>",
     '<main class="wrap">',
     f'<header><p class="eyebrow">WM-Verdict · LIBERO · 闭环</p><h1>终局想象乐观率</h1>'
     f'<p class="lede">现实失败的闭环单元里，世界模型最后一次规划时想象出的画面，有多少已经“把任务做完了”。完整结果包，{TODAY} 由 report_full.py 生成，数字全部读文件。</p></header>',
     '<section class="stats">' + "".join(f'<div class="stat"><span class="sl">{e(a)}</span><span class="sv">{e(b)}</span><span class="sn">{e(c)}</span></div>' for a, b, c in stat) + "</section>",
     '<section><h2>定义</h2><dl class="def">' + "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in DEF) + "</dl></section>",
     f'<section><h2>自动判定</h2><h3>OpenWAM-α · T1 · {e(mT["judge"])} · n = {mT["n_units"]}</h3>'
     f'<p class="big">乐观率（主口径，末帧）<strong>{e(auto_main)}</strong></p><p>敏感性对照（第 400 步帧）：{e(auto_400)}</p>'
     f'<p class="warn">{e(N4)}</p><div class="mxs">{hmatrix(mT, "主口径 · 末帧（名义第 416 步）")}{hmatrix(s4, "敏感性 · 第 400 步帧")}</div>'
     f'<p class="note">现实成功中想象也判成功：主口径 {e(succ_main)}；第 400 步帧 {e(succ_400)}。</p>'
     f'<h3>新进入交集、待判定</h3>{htable(["模型", "任务", "判定器", "单元数", "现实失败", "来源"], pend_rows)}<p class="note">{e(pend_note)}</p></section>',
     f'<section><h2>人工标注 <span class="tag">人工标注口径 · 标注进行中</span></h2><p>{e(hum_status)}</p>'
     + (f'<pre class="cmd">{e(EXPORT_CMD)}</pre>' if not exp else "")
     + htable(["模型", "任务", "现实失败卡", "已标", "uncertain", "乐观率（主口径）", "宽口径（含 borderline）"], hum_rows)
     + '<p class="note">OpenWAM T3/T4 的现实失败单元为抽样（各 60 个），其余格全收。LingBot T0/T1 不在 r3 里（出 r3 时还没有闭环想象帧）。</p>'
     + "<h3>r3 卡片构成</h3>" + htable(["模型", "任务", "现实失败卡", "成功对照卡", "合计"], r3_rows) + "</section>",
     f'<section><h2>预筛乐观率 · 四模型 <span class="tag">agent 代判 · 未校准 · 仅内部参考</span></h2><p>{e(pre_notes[0])}</p>{htable(pre_head, pre_rows, "dense")}'
     + "".join(f'<p class="note">{e(x)}</p>' for x in pre_notes[1:]) + f'<p class="warn">脚注：{e(PRE_FN)}</p></section>',
     '<section><h2>按扰动档细分 <span class="tag">预筛口径</span></h2>' + f'<p class="note">{e(BYL_NOTE)}</p>' + "".join(
         (lambda ms, rows, cov: f'<details{" open" if t in ("T3", "T4") else ""}><summary>{t} · 覆盖 ' + "；".join(f"{e(m)} {a}/{b}" for m, (a, b) in cov.items()) + "</summary>"
          + '<div class="tw"><table class="dense lv"><thead><tr><th>档位</th>' + "".join(f"<th>{e(m)}</th>" for m in ms) + "</tr></thead><tbody>"
          + "".join(f"<tr><td><code>{e(lv)}</code></td>" + "".join(
                (f'<td class="r{" sm" if c["n"] < 10 else ""}" style="background:color-mix(in srgb, var(--hot) {round(50 * c["rate"])}%, transparent)"><b>{100 * c["rate"]:.0f}%</b> <span>{c["k"]}/{c["n"]}</span>'
                 + (f'<small>有帧 {c["frames"]}/{c["fail"]}</small>' if c["fail"] and c["frames"] < c["fail"] else "")
                 + (f'<small>判不了 {c["unc"]}</small>' if c["unc"] else "") + "</td>") if c["kind"] == "rate"
                else f'<td class="na">{e(cell_txt(c))}</td>' for c in cs) + "</tr>" for lv, cs in rows)
          + "</tbody></table></div></details>")(*byl_task(t)) for t in BYL_TASKS) + "</section>",
     '<section><h2>待补数据</h2><dl class="def">' + "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in MISSING) + "</dl></section>",
     f'<section><h2>想象帧清晰度 · 腐蚀穿透</h2><p>{e(sh_note)}</p>{htable(sh_head, sh_rows, "dense")}</section>',
     f'<section><h2>覆盖面</h2><p>{e(cov_note)}</p><h3>按 模型 × 任务</h3>{htable(["模型", "任务", "存帧单元", "现实失败", "判定器"], mt_rows)}'
     f'<details><summary>按 模型 × 任务 × 扰动族展开（{len(cov_rows)} 行）</summary>{htable(["模型", "任务", "族", "单元数", "现实失败", "判定器", "来源"], cov_rows, "dense")}</details></section>',
     f'<section><h2>对比案例候选</h2><p>11-13 裁决出 6 张对比案例图；下列 {len(cases["candidates"])} 张为候选（NLw 案例不展示），已选 {n_sel} 张，选定状态记在 case_selection.json。每张上排为想象帧条，下排为真实帧条。</p><div class="cases">'
     + "".join(f'<figure class="case"><img src="{e(c["file"])}" alt="{e(c["unit_id"])} 想象帧条与真实帧条" loading="lazy">'
               f'<figcaption><span class="st st-{ {"已选": "on", "弃用": "off"}.get(c["status"], "todo") }">{e(c["status"])}</span>'
               f'{e(MNAME.get(c["model"], c["model"]))} · {e(c["task"])} · {e(c["level"])}<br><code>{e(c["unit_id"])}</code></figcaption></figure>' for c in cases["candidates"])
     + "</div></section>",
     '<footer><p>文件：opt_coverage.py → units.json / coverage.tsv；judge_ow_t1.py → confusion_openwam_T1.json；report_full.py → report_full.md / report_full.html / report_full_numbers.json；11-13 版存档在 backup_20261113/。</p></footer>',
     "</main>"]
open(f"{HERE}/report_full.html", "w").write("\n".join(H) + "\n")
print("ok", sum(tot.values()), "judged", len(judged), "pending", len(pending), "r3 export", bool(exp))
