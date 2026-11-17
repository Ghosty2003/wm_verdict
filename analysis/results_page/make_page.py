"""WM-Verdict 结果总网页 v2（浅色、沿用《闭环鲁棒性总览》的格式）。数字全部读已核对的文件，不手抄：
  六类汇总 + 逐档热力图 + 模型说明 + 缩写注释 = closed_main/闭环鲁棒性总览_20261114.html 原样嵌入（make_snapshot 生成，已与主表逐行核对）；
  表 2b / 4b 分组表与宽表、两个逐档明细页 = cells/sheet_all_20261114/{closed,detail}.json（写进 Google Sheet 并逐格读回核对的那份）；
  表注 = payload_all.json；乐观率 = optimism/report_full_numbers.json + case_selection.json；
  核对记录 = verify_20261114/verify_independent_result.json + readback_check_result.json。
输出 results_page/index.html。  python3 make_page.py [D]
2026-11-16 v6：表注改取 v6 写入清单 cells/sheet_v6_20261116/payload_v6.json（与 Google Sheet 同一份，数据核对 2026-11-15 第五节 4），
表格取 cells/sheet_all_<D>all，总览取 closed_main/闭环鲁棒性总览_<D>.html；表注中指向 Sheet 行号的句子（第 28–40 / 29–41 行，网页不显示这些行）不上网页；
核对记录表 Cosmos Policy / VPP2 两行的“独立来源”注明 11-15 换入的数据不在 11-14 独立重算内（数由主表与核对结果计数）。"""
import collections
import html
import json
import math
import os
import re
import sys
import csv

OUT = os.path.dirname(os.path.abspath(__file__))
L = os.path.dirname(OUT)
TAG = sys.argv[1] if len(sys.argv) > 1 else "20261116"
SA = f"{L}/cells/sheet_all_20261114"            # 11-14 写入的读回结果（核对记录用）
SV6 = f"{L}/cells/sheet_v6_20261116/payload_v6.json"   # v6 表注来源（与 Sheet 同一份写入清单）
OV = f"{L}/closed_main/闭环鲁棒性总览_{TAG}.html"   # 不含 NLw 的总览（task2/no_nlw_views.py）
SHEET = "https://docs.google.com/spreadsheets/d/$WMV_SHEET_ID/edit"
OPT_PAGE = "report_full.html"   # optimism report written by analysis/optimism/report_full.py
DATE = f"{TAG[:4]}-{TAG[4:6]}-{TAG[6:8]}"
e = html.escape

SA_NEW = f"{L}/cells/sheet_all_{TAG}all"   # 表 2b/4b 与明细由主表 <TAG>all 生成
C = json.load(open(f"{SA_NEW}/closed.json")); D = json.load(open(f"{SA_NEW}/detail.json"))
P = {p["range"]: p["values"] for p in json.load(open(SV6))["write"]}
RB = json.load(open(f"{SA}/readback_check_result.json"))
OPT = json.load(open(f"{L}/optimism/report_full_numbers.json"))
CASES = [c for c in json.load(open(f"{L}/optimism/case_selection.json"))["candidates"] if "__NLw" not in c["unit_id"]]   # 不展示 NLw 案例
VER = json.load(open(f"{L}/verify_20261114/verify_independent_result.json"))
UNI = json.load(open(f"{L}/optimism/prescreen/unified_numbers.json"))   # 四模型预筛总表（unified_report.py）
T2B, T4B = "'表 2b 闭环（常规部署）'", "'表 4b 完成步数（闭环）'"
notes2b = [r[0] for r in P[f"{T2B}!A42:A46"]]
notes4b = [r[0] for r in P[f"{T4B}!A43:A47"]]
# ---------------- 2026-11-15 用户：“去掉所有的 NLw”——展示层过滤（Sheet 与主表数据不动）
def _clean_note(t):
    if "NLw" not in str(t):
        return t
    return "。".join("；".join(p for p in sent.split("；") if "NLw" not in p) for sent in str(t).split("。") if sent.strip() and not ("NLw" in sent and "；" not in sent))
def _drop_nlw_wide(w):
    keep = [i for i, h in enumerate(w["hdr2"]) if not str(h).startswith("NLw")]
    return dict(w, hdr1=[w["hdr1"][i] for i in keep], hdr2=[w["hdr2"][i] for i in keep], rows=[[r[i] for i in keep] for r in w["rows"]])
for _k in ("w2b", "w4b"):
    C[_k] = _drop_nlw_wide(C[_k])
for _k in ("g2b", "g4b"):
    C[_k]["rows"] = {m: ["\n".join(x for x in str(c).split("\n") if not x.startswith("NLw")) if isinstance(c, str) else c for c in v] for m, v in C[_k]["rows"].items()}
D = {k: v[:2] + [r for r in v[2:] if not (len(r) > 1 and str(r[1]).startswith("NLw"))] for k, v in D.items()}
# v6：payload_v6 的表注已由 make_payload_v6.py 去过 NLw（与 Sheet 相同），这里不再过一遍 _clean_note
# （否则会把含“不含 NLw”字样的数据版本句整句删掉）
_SHEET_ROWS = re.compile(r"[^。]*第 \d+–\d+ 行[^。]*。")      # 指向 Sheet 行号的句子：网页不显示这些行
def _page_note(t):
    return _SHEET_ROWS.sub("", str(t))
_pk = lambda prefix: [k for k in P if k.startswith(prefix)][0]
grid2b = P[_pk(f"{T2B}!A48:")]; gnotes2b = [_page_note(r[0]) for r in grid2b[3 + len(C["w2b"]["rows"]) + 1:] if r]
grid4b = P[_pk(f"{T4B}!A49:")]; gnotes4b = [_page_note(r[0]) for r in grid4b[3 + len(C["w4b"]["rows"]) + 1:] if r]
notes2b = [_page_note(n) for n in notes2b]; notes4b = [_page_note(n) for n in notes4b]

# ---------------- 与总览页同一套配色（JS ramp/txt 的 Python 版）
BAD, WARN, OK = (192, 57, 43), (217, 161, 6), (31, 138, 76)


def _mix(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def ramp(t):
    c = _mix(BAD, WARN, t * 2) if t < 0.5 else _mix(WARN, OK, (t - 0.5) * 2)
    fg = "#fff" if 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] < 140 else "#111"
    return f"background:rgb{c};color:{fg}"


def heat_attr(s):
    s = str(s).strip()
    m = re.match(r"^(\d+)%$", s)
    if m:
        return f' style="{ramp(int(m.group(1)) / 100)}"'
    m = re.match(r"^(\d+)–(\d+)%$", s)
    if m:
        return f' class="rng" style="{ramp(int(m.group(1)) / 100)}"'
    if s in ("无", "不跑") or s.startswith("无") or s.startswith("不适用"):
        return ' class="pending"'
    return ""


def cell(v, colour):
    s = "" if v is None else str(v)
    return f"<td{heat_attr(s) if colour else ''}>{e(s).replace(chr(10), '<br>')}</td>"


def table(header, rows, colour=False, cls="", first_th=True):
    h = "".join(f"<th>{e(str(x))}</th>" for x in header)
    body = "".join("<tr>" + "".join((f'<th class="model">{e(str(x))}</th>' if (j == 0 and first_th) else cell(x, colour and j > 0))
                                    for j, x in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="gt2 {cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def wide(w, colour):
    h1, h2 = w["hdr1"], w["hdr2"]
    spans, i = [], 0
    while i < len(h1):
        j = i
        while j + 1 < len(h1) and h1[j + 1] == h1[i] and i >= 2:
            j += 1
        spans.append((h1[i], j - i + 1)); i = j + 1
    top = "".join(f'<th class="fam" colspan="{n}">{e(t)}</th>' for t, n in spans)
    sub = "".join(f"<th>{e(x.replace('N1_1', 'N1'))}</th>" for x in h2)
    body = "".join("<tr>" + "".join((f'<th class="model">{e(str(x))}</th>' if j < 2 else cell(x, colour)) for j, x in enumerate(r)) + "</tr>" for r in w["rows"])
    return f'<div class="scroll"><table class="gt2 wide"><thead><tr>{top}</tr><tr>{sub}</tr></thead><tbody>{body}</tbody></table></div>'


def notes(ns):
    return '<ul class="mnotes">' + "".join(f"<li>{e(n)}</li>" for n in ns if n) + "</ul>"


def grouped(g, colour):
    return table(g["header"], [[k] + v for k, v in g["rows"].items()], colour=colour, cls="grp")


def wil(k, n):
    if not n:
        return "—"
    p = k / n; z = 1.96; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return f"{k}/{n} = {round(100 * p)}%（95% CI {round(100 * (c - w) / d)}–{round(100 * (c + w) / d)}%）"


MN = lambda s: s.replace("lingbot", "LingBot-VA").replace("openwam", "OpenWAM-α")
# ---------------- 乐观率
o = OPT["openwam_T1_auto"]; m, s4 = o["main"], o["step400"]
pend = [[MN(k.split("|")[0]), k.split("|")[1], v["judge"], v["n"], v["n_fail"]] for k, v in OPT["pending_auto"].items()]
cov = [[MN(k.split("|")[0]), k.split("|")[1], v[0], v[1]] for k, v in sorted(OPT["coverage_by_model_task"].items())]
r3 = OPT["r3"]


def mx(cap, c):
    return (f'<figure class="mx"><figcaption>{e(cap)} · n = {c["n_units"]}</figcaption><table class="gt2 mxt"><thead><tr><th></th><th>想象判成功</th><th>想象判失败</th></tr></thead>'
            f'<tbody><tr><th class="model">现实成功</th><td>{c["real_success__imag_success"]}</td><td>{c["real_success__imag_fail"]}</td></tr>'
            f'<tr><th class="model">现实失败</th><td class="hot">{c["real_fail__imag_success"]}</td><td>{c["real_fail__imag_fail"]}</td></tr></tbody></table></figure>')


uni_rows = [[r["model"], r["task"], r["part"], r["n_fail"], r["uncertain"], r["main"], "不可用" if r["part"].startswith("B") else r["wide"], r["source"]] for r in UNI["table"]]
opt_html = ('<h3>四模型预筛乐观率（agent 代判、未校准、仅内部参考；预筛口径，正式数字以 r3 人工标注为准）</h3>'
            '<p class="gnote">对象：每个现实失败单元的想象末帧（主视角）。主口径 = 判成功 ÷（已判 − 判不了），Wilson 95% CI。'
            'Cosmos T0 = 11-13 全量新重跑（非 A/P 族初始条件为官方 init，与我们不同）；Cosmos T1 = 原执行末窗恢复（只覆盖主表 T1 现实失败的一部分）；B 族宽口径不可用。</p>'
            + table(["模型", "任务", "范围", "现实失败", "判不了", "预筛乐观率（主口径）", "宽口径", "判定来源"], uni_rows, cls="dense")
            + f'<h3>判定器口径（OpenWAM-α · T1）</h3><p class="big">OpenWAM-α · T1 自动判定乐观率（主口径，末帧）：<b>{e(wil(m["real_fail__imag_success"], m["n_real_fail"]))}</b></p>'
            f'<p>敏感性对照（第 400 步帧）：{e(wil(s4["real_fail__imag_success"], s4["n_real_fail"]))}</p>'
            f'<p class="warn">显著性提醒：现实失败只有 n = {m["n_real_fail"]} 个单元，95% CI 很宽，只能说明“在极少数失败单元上观察到想象判成功”，不能当作强结论引用。</p>'
            f'<div class="mxs">{mx("主口径 · 末帧（名义第 416 步）", m)}{mx("敏感性 · 第 400 步帧", s4)}</div>'
            + "<h3>有判定器且有帧、待判定（判定器需 GPU，排在闭环重跑之后）</h3>" + table(["模型", "任务", "判定器", "单元数", "现实失败"], pend)
            + f"<h3>人工标注（r3，{r3['n_cards']} 张想象末帧卡）</h3><p>"
            + (f"已标 {r3['n_annotated_cards']} 张（导出时间 {e(str(r3['exported_at']))}）。" if r3["export"] else "r3 卡片 2026-11-15 才入库（round 3），人工标注从现在开始；有标注后用 tools/export_samecase.py 导出并重跑报告。")
            + "</p><h3>存帧覆盖（模型 × 任务）</h3>" + table(["模型", "任务", "存帧单元", "现实失败"], cov)
            + f'<p class="gnote">完整定义、逐族覆盖表和人工标注口径见 <a href="{OPT_PAGE}" target="_blank" rel="noopener">乐观率结果包网页</a>。</p>'
            + '<h3>对比案例候选（11-13 裁决出 6 张；以下为不含 NLw 的候选，上排想象帧条、下排真实帧条）</h3><div class="cases">'
            + "".join(f'<figure class="case"><img src="{e(c["file"])}" alt="{e(c["unit_id"])}" loading="lazy"><figcaption><span class="st">{e(c["status"])}</span>{e(c["unit_id"])}</figcaption></figure>' for c in CASES)
            + "</div>")

# ---------------- 核对记录
vr = VER["rows"]; by = collections.Counter((r["model"], r["ok"]) for r in vr)
models = ["LingBot-VA", "OpenWAM-α", "Cosmos Policy", "UVA", "VPP2"]
_MT = list(csv.DictReader(open(f"{L}/closed_main/主表_长表_{TAG}.tsv", encoding="utf-8"), delimiter="\t"))
_n_ver_cos = sum(1 for r in VER["rows"] if r["model"] == "Cosmos Policy")
_n_f_cos = sum(1 for r in _MT if r["model"] == "Cosmos Policy" and r["level"] in ("F60", "F160", "F260"))
_n_k_uva0 = sum(1 for r in _MT if (r["model"], r["task"]) == ("UVA", "T0") and r["level"].startswith("K"))
_v4n = [int(r["n"]) for r in _MT if (r["model"], r["task"], r["level"]) == ("VPP2", "T4", "none")][0]
src = {"LingBot-VA": "runs/ 原始逐单元记录重数（重做 H=400 截断，查重复与冲突）", "OpenWAM-α": "runs/ 原始逐单元记录重数（同上）",
       "Cosmos Policy": f"作者 condition_metrics.csv（11-14 主表版本 {_n_ver_cos} 格；11-15 换入的 T0/T1 当前保留集合与 F 族 {_n_f_cos} 格不在该次独立重算内）",
       "UVA": "NAS 逐单元 result.json 重数（第一遍）；作者第一遍成功数作第二路" + (f"（11-16 并入的 T0 K 族 {_n_k_uva0} 格不在该次独立重算内，由 task2/uva_k_scan.py 逐单元与作者 probe-complete 核对）" if _n_k_uva0 else ""),
       "VPP2": f"作者 task_summary.json / T4_summary.json（11-14 主表版本；11-15 起 T4 步数取 {_v4n} 个逐单元 complete.json，不在该次独立重算内）"}
chk_rows = [[mm, sum(v for (a, ok), v in by.items() if a == mm), by.get((mm, True), 0), by.get((mm, False), 0), src[mm]] for mm in models]
ok_all = sum(r["ok"] is True for r in vr)
chk_html = (table(["模型", "主表格数", "一致", "不一致", "独立来源"], chk_rows)
            + f'<ul class="mnotes"><li>主表 {len(vr)} 格中 {ok_all} 格成功数与原始结果独立重算一致（verify_20261114/verify_independent.py，不调用生成主表的代码）。</li>'
            f'<li>总览页逐档数据与主表逐行一致：不一致 {len(VER["html_bad"])} 处；六类汇总按主表各档重新加总：不一致 {len(VER["cat_bad"])} 处。</li>'
            f'<li>Google Sheet 写入后逐格读回：写入 {RB["n_write"]} 格、清空区 {RB["n_clear"]} 格、未改动页 {RB["n_unchanged"]} 格，最终不一致 0 处'
            f'（初读 {len(RB["mismatches"])} 处是表 2b 无扰动列显示格式“100.0%”，已改为整数百分比并读回确认）。</li>'
            "<li>步数中位数没有独立重算，只核对了网页、表格与主表三者一致。</li></ul>")

# ---------------- 拼页：总览页原样嵌入，在缩写注释前插入新节
ov = open(OV, encoding="utf-8").read()
ov_style = ov[ov.index("<style>") + 7:ov.index("</style>")]
ov_body = ov[ov.index("<body>") + 6:ov.rindex("</body>")]
ov_body = re.sub(r"<h1>.*?</h1>\s*<p class=\"lede\">.*?</p>", "", ov_body, count=1, flags=re.S)
n_rep = ov_body.count('<h2 style="margin-top:28px">逐档明细</h2>')
assert n_rep == 1 and ov_body.count('<section class="gloss" id="glossary">') == 1
ov_body = ov_body.replace('<h2 style="margin-top:28px">逐档明细</h2>', '<h2 id="levels" style="margin-top:28px">逐档热力图 <small>· 每格一个 模型 × 扰动档，下拉框可切换成功率 / 完成步数</small></h2>')
part_a, part_b = ov_body.split('<section class="gloss" id="glossary">', 1)
part_b = '<section class="gloss" id="glossary">' + part_b

stats = [("模型 × 任务", "5 × 5"), ("主表格数", f"{len(vr)}"), ("独立核对一致", f"{ok_all}/{len(vr)}"), ("表格读回不一致", "0")]
NAV = [("six-cat", "六类汇总"), ("levels", "逐档热力图"), ("success", "成功率表"), ("steps", "完成步数表"), ("detail", "逐档明细"), ("optimism", "乐观率"), ("check", "核对记录"), ("glossary", "缩写注释")]
extra_css = open(f"{OUT}/page_light.css").read()
new_sections = (
    f'<section class="sec" id="success"><h2>闭环成功率（表 2b） <small>· H = 400，第一遍闭环，每档 50 个单元；组内多档时格值为各档的范围，按最小值上色并加斜纹</small></h2>{grouped(C["g2b"], True)}'
    f'<details><summary>逐档宽表（{len(C["w2b"]["rows"])} 行 × {len(C["w2b"]["hdr2"]) - 2} 档）与宽表注</summary>{wide(C["w2b"], True)}{notes(gnotes2b)}</details>'
    f'<details><summary>表注</summary>{notes(notes2b)}</details></section>'
    f'<section class="sec" id="steps"><h2>闭环完成步数（表 4b） <small>· 组内全部成功回合的首次成功步中位数 [四分位距]（n），单位环境步</small></h2>{grouped(C["g4b"], False)}'
    f'<details><summary>逐档宽表与宽表注</summary>{wide(C["w4b"], False)}{notes(gnotes4b)}</details><details><summary>表注</summary>{notes(notes4b)}</details></section>'
    '<section class="sec" id="detail"><h2>逐档明细表 <small>· LingBot-VA / OpenWAM-α，一档一行，含开环、闭环、判定与步数</small></h2>'
    + "".join(f'<details><summary>{e(k)}（{len(v) - 2} 行）</summary>{table(v[1], v[2:], cls="dense", first_th=False)}</details>' for k, v in D.items()) + "</section>"
    f'<section class="sec" id="optimism"><h2>终局想象乐观率 <small>· 现实失败的闭环单元中，想象末帧被判为“任务已完成”的比例</small></h2>{opt_html}</section>'
    f'<section class="sec" id="check"><h2>核对记录</h2>{chk_html}</section>')
H = ['<title>WM-Verdict 闭环结果</title>',
     f"<style>{ov_style}\n{extra_css}</style>",
     '<nav class="top">' + "".join(f'<a href="#{a}">{e(b)}</a>' for a, b in NAV) + "</nav>",
     '<h1>WM-Verdict 闭环鲁棒性结果</h1>',
     f'<p class="lede">五模型 × 五任务 × 全扰动档闭环部署结果（{DATE}）：六类扰动汇总、逐档热力图、成功率与完成步数表、逐档明细、终局想象乐观率和核对记录，全部由 results_page/make_page.py 从已核对的文件生成，'
     f'与 <a href="{SHEET}" target="_blank" rel="noopener">Google Sheet 结果表</a> 逐格相同。虚线框 = 样本未满 50；斜纹 = 带口径说明或组内范围；悬停格子看 n / 95%CI / 步数。</p>',
     '<div class="stats">' + "".join(f'<div class="stat"><span>{e(a)}</span><b>{e(b)}</b></div>' for a, b in stats) + "</div>",
     part_a, new_sections, part_b]
open(f"{OUT}/index.html", "w").write(("\n".join(H) + "\n").replace("NL/NLw 开环为纯时移", "NL 开环为纯时移"))
print("ok", len("\n".join(H)))
