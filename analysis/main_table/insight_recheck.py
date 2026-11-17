"""insight 重核（2026-11-10 决定 3）：所有对比数字由同一脚本分别从旧快照(1109)与新快照(1110)推导，输出旧值 vs 新值。
不含任何手抄数字：旧值来自 主表_长表_20261109.tsv（当时 insight 的数据源）或从 eval_design.md 原文用正则抽取。"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import csv, re, collections
CM = _wmvp.P("$WMV_ROOT/libero/out/closed_main")
OLD, NEW = "20261109", "20261110"
def load(d):
    R = list(csv.DictReader(open(f"{CM}/主表_长表_{d}.tsv"), delimiter="\t"))
    R = [r for r in R if not r.get("exclude_cross")]      # 裁决排除格（UVA T0/T1 Ef 等）不进任何跨任务/跨模型汇总
    for r in R:
        r["rate"] = float(r["success_rate"]); r["n"] = int(r["n"]); r["k"] = int(r["n_success"])
        r["med"] = int(r["median_first_success_step"]) if r["median_first_success_step"] else None
    return R
SNAP = {OLD: load(OLD), NEW: load(NEW)}
def cell(d, m, t, l):
    x = [r for r in SNAP[d] if r["model"].startswith(m) and r["task"] == t and r["level"] == l]
    return x[0] if x else None
def fmt(r): return "—（无此格）" if r is None else f"{r['rate']:.0%} ({r['k']}/{r['n']})"
out = [f"# insight 重核：旧值（{OLD} 快照）vs 新值（{NEW} 快照）", "",
       "全部数字由 out/task2/insight_recheck.py 从两份 TSV（及 eval_design.md 原文）推导，未手抄。变了的行标 ⚠。", ""]
# ---- 1. 摘要对比格 ----
out += ["## 1. 摘要对比格（T0 · LingBot vs OpenWAM）", "", "| 格 | 模型 | 旧值 | 新值 | 变化 |", "|---|---|---|---|---|"]
for l in ("NL5", "G4", "Bg2"):
    for m in ("LingBot", "OpenWAM"):
        a, b = cell(OLD, m, "T0", l), cell(NEW, m, "T0", l)
        ch = "" if (a and b and (a["k"], a["n"]) == (b["k"], b["n"])) else "⚠"
        out.append(f"| T0 {l} | {m} | {fmt(a)} | {fmt(b)} | {ch} |")
# ---- 2. 每份快照独立重跑 insight 挖掘 ----
def mine(d):
    full = [r for r in SNAP[d] if r["complete"] == "y"]
    idx = {(r["model"], r["task"], r["level"]): r for r in full}
    by = collections.defaultdict(dict)
    for r in full: by[(r["task"], r["level"])][r["model"]] = r
    div = sorted(((abs(v["LingBot-VA"]["rate"] - v["OpenWAM-α"]["rate"]), k) for k, v in by.items() if len(v) == 2), reverse=True)[:10]
    ORD = {"NL": [f"NL{i}" for i in range(1, 6)], "NLw": [f"NLw{i}" for i in range(1, 6)], "NR": [f"NR{i}" for i in range(1, 6)],
           "G": [f"G{i}" for i in range(1, 5)], "Bg": ["Bg2", "Bg5", "Bg9"], "Bf": ["Bf2", "Bf5", "Bf9"], "Bz": ["Bz2", "Bz5", "Bz9"],
           "A": [f"A{i}" for i in range(5)], "P": [f"P{i}" for i in range(5)], "O": [f"O{i}" for i in range(1, 6)]}
    cliffs = []
    for (m, t) in {(r["model"], r["task"]) for r in full}:
        for seq in ORD.values():
            for a, b in zip(seq, seq[1:]):
                if (m, t, a) in idx and (m, t, b) in idx:
                    cliffs.append((idx[(m, t, a)]["rate"] - idx[(m, t, b)]["rate"], m, t, f"{a}→{b}"))
    cliffs = sorted(cliffs, reverse=True)[:8]
    base = {(r["model"], r["task"]): r["med"] for r in full if r["level"] in ("none", "B0") and r["med"]}
    slow = [(r["model"], r["task"], r["level"]) for r in full
            if r["rate"] >= 0.8 and r["med"] and base.get((r["model"], r["task"])) and r["level"] not in ("none", "B0")
            and r["med"] > base[(r["model"], r["task"])] * 1.15]
    low = collections.Counter(r["model"] for r in full if r["rate"] < 0.5)
    return dict(div=div, cliffs=cliffs, slow=slow, low=low, idx=idx)
M = {d: mine(d) for d in SNAP}
out += ["", "## 2. 摘要首句的低于 50% 格数（完整格）", "", "| 模型 | 旧值 | 新值 | 变化 |", "|---|---|---|---|"]
for m in ("LingBot-VA", "OpenWAM-α"):
    a, b = M[OLD]["low"][m], M[NEW]["low"][m]
    out.append(f"| {m} | {a} | {b} | {'⚠' if a != b else ''} |")
out += ["", "## 3. 模型分歧最大的 10 格（|LingBot − OpenWAM|）", "", "| 名次 | 旧：格 (LB / OW) | 新：格 (LB / OW) | 变化 |", "|---|---|---|---|"]
def dv(d, i):
    if i >= len(M[d]["div"]): return "—"
    g, (t, l) = M[d]["div"][i]; idx = M[d]["idx"]
    return f"{t} {l} ({idx[('LingBot-VA', t, l)]['rate']:.0%} / {idx[('OpenWAM-α', t, l)]['rate']:.0%})"
for i in range(10):
    a, b = dv(OLD, i), dv(NEW, i); out.append(f"| {i+1} | {a} | {b} | {'⚠' if a != b else ''} |")
out += ["", "## 4. 最陡悬崖（相邻档成功率骤降）前 8", "", "| 名次 | 旧 | 新 | 变化 |", "|---|---|---|---|"]
def cf(d, i):
    if i >= len(M[d]["cliffs"]): return "—"
    g, m, t, s = M[d]["cliffs"][i]; a, b = s.split("→"); idx = M[d]["idx"]
    return f"{m.split('-')[0]} {t} {s} ({idx[(m, t, a)]['rate']:.0%}→{idx[(m, t, b)]['rate']:.0%})"
for i in range(8):
    a, b = cf(OLD, i), cf(NEW, i); out.append(f"| {i+1} | {a} | {b} | {'⚠' if a != b else ''} |")
so, sn = set(M[OLD]["slow"]), set(M[NEW]["slow"])
out += ["", "## 5. \"没失败但变慢\"（成功率 ≥80% 且步数中位 > 基线 ×1.15）", "",
        f"- 格数：旧 {len(so)} → 新 {len(sn)} {'⚠' if len(so) != len(sn) else ''}",
        f"- 新增：{', '.join(f'{m[:7]} {t} {l}' for m, t, l in sorted(sn - so)) or '无'}",
        f"- 消失：{', '.join(f'{m[:7]} {t} {l}' for m, t, l in sorted(so - sn)) or '无'}"]
out += ["", "## 6. 同档跨任务：K1（LingBot）", "", "| 任务 | 旧 | 新 | 变化 |", "|---|---|---|---|"]
for t in ("T0", "T1", "T2", "T3", "T4"):
    a, b = cell(OLD, "LingBot", t, "K1"), cell(NEW, "LingBot", t, "K1")
    out.append(f"| {t} | {fmt(a)} | {fmt(b)} | {'' if (a and b and (a['k'], a['n']) == (b['k'], b['n'])) else '⚠'} |")
# ---- 7. eval_design.md 原文数字 ----
ED = open(_wmvp.P("$WMV_ROOT/libero/eval_design.md")).read()
line = [l for l in ED.splitlines() if "最强档闭环仍有" in l][0]
out += ["", "## 7. eval_design.md §28「背景文字」原文数字（LingBot，H=400 口径）", "", f"> {line.strip()}", "",
        "| 原文 | 对应格 | 原文值 | 新快照值 | 变化 |", "|---|---|---|---|---|"]
TMAP = {"t5": "T0", "t3": "T1"}
for tk, lv, k in re.findall(r"(t[35]) (A4|P4) (\d+)/50", line.split("H = 520")[0]):
    b = cell(NEW, "LingBot", TMAP[tk], lv)
    out.append(f"| {tk} {lv} {k}/50 | LingBot {TMAP[tk]} {lv} | {k}/50 | {fmt(b)} | {'' if b and b['k'] == int(k) and b['n'] == 50 else '⚠'} |")
m = re.search(r"两任务 P4 均为 (\d+)/50", line)
if m:
    for tk in ("t5", "t3"):
        b = cell(NEW, "LingBot", TMAP[tk], "P4")
        out.append(f"| 两任务 P4 均为 {m.group(1)}/50 | LingBot {TMAP[tk]} P4 | {m.group(1)}/50 | {fmt(b)} | {'' if b and b['k'] == int(m.group(1)) else '⚠'} |")
out += ["", "注：NAS 上未找到摘要草稿文件；第 1–6 节的对比数字此前只出现在对话回复里，此表即其更正依据。"]
open(f"{CM}/insight_recheck_{NEW}.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
