"""网页解析 + 由 verified.json 生成“网页应显示的值”。compare_page.py / apply_fixes.py / readback.py 共用。只用标准库。"""
import html
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ("Cosmos Policy", "VPP2")
TASKS = ("T0", "T1", "T2", "T3", "T4")
RESTORED = "original_episode_final_window_restored"


# ------------------------------------------------------------------ 解析
def get_json(s, name):
    i = s.index(f"const {name}=") + len(f"const {name}=")
    v, end = json.JSONDecoder().raw_decode(s, i)
    return v, i, end


def _cells(tr):
    out = []
    for m in re.finditer(r"<(td|th)([^>]*)>(.*?)</\1>", tr, re.S):
        out.append(html.unescape(re.sub(r"<br>", "\n", m.group(3))))
    return out


def section(s, sid):
    a = s.index(f'id="{sid}"'); a = s.rfind("<section", 0, a)
    b = s.index("</section>", a)
    return a, b


def tables_in(s, sid):
    a, b = section(s, sid)
    res = []
    for m in re.finditer(r"<table[^>]*>(.*?)</table>", s[a:b], re.S):
        body = m.group(1)
        thead = re.search(r"<thead>(.*?)</thead>", body, re.S).group(1)
        tbody = re.search(r"<tbody>(.*?)</tbody>", body, re.S).group(1)
        hrows = [_cells(tr) for tr in re.findall(r"<tr>(.*?)</tr>", thead, re.S)]
        brows = [_cells(tr) for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tbody, re.S)]
        res.append(dict(head=hrows, rows=brows, start=a + m.start()))
    return res


def notes_in(s, sid):
    a, b = section(s, sid)
    return [html.unescape(x) for x in re.findall(r"<li>(.*?)</li>", s[a:b], re.S)]


def parse(s):
    D, _, _ = get_json(s, "DATA"); C, _, _ = get_json(s, "CAT"); MN, _, _ = get_json(s, "MNOTES")
    succ = tables_in(s, "success"); steps = tables_in(s, "steps"); chk = tables_in(s, "check")
    return dict(DATA=D, CAT=C, MNOTES=MN, g2b=succ[0], w2b=succ[1], g4b=steps[0], w4b=steps[1], check=chk[0],
                notes_success=notes_in(s, "success"), notes_steps=notes_in(s, "steps"))


# ------------------------------------------------------------------ 期望值
def load_verified():
    return json.load(open(f"{HERE}/verified.json", encoding="utf-8"))


def cell(V, m, T, l):
    return V["cells"].get(f"{m}|{T}|{l}")


def fmt_med(x):
    x = float(x)
    return str(int(x)) if x.is_integer() else f"{x:.1f}"


def lin_q(xs, p):
    import math
    xs = sorted(xs); n = len(xs); h = (n - 1) * p; lo = math.floor(h); hi = min(lo + 1, n - 1)
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])


def med_text(steps):
    if not steps:
        return "无成功"
    return f"{fmt_med(lin_q(steps, .5))} [{lin_q(steps, .25):.0f}, {lin_q(steps, .75):.0f}]（n={len(steps)}）"


def pct(k, n):
    return f"{round(100 * k / n)}%"


def pct_range(vals):
    lo, hi = round(100 * min(vals)), round(100 * max(vals))
    return f"{lo}%" if lo == hi else f"{lo}–{hi}%"


GG = [("无扰动", ["none"]), ("A 训练内", ["A0", "A1"]), ("A 训练外", ["A2", "A3", "A4"]),
      ("P 训练内", ["P0", "P1", "P2"]), ("P 训练外", ["P3", "P4"]), ("K", [f"K{i}" for i in range(1, 10)]),
      ("E", [f"Em{i}" for i in range(1, 5)] + [f"Ef{i}" for i in range(1, 5)]), ("G", [f"G{i}" for i in range(1, 5)]),
      ("N 系", None), ("O", [f"O{i}" for i in range(1, 6)]),
      ("B", ["B0"] + [f"B{t}{s}" for t in "gfz" for s in (2, 5, 9)]), ("F", ["F60", "F160", "F260"])]
NSUB = (("N1", ["N1_1"]), ("NR", [f"NR{i}" for i in range(1, 6)]), ("NL", [f"NL{i}" for i in range(1, 6)]))


def group_text(V, kind, m, T, levels):
    use = [cell(V, m, T, l) for l in levels if cell(V, m, T, l)]
    if not use:
        return None
    if kind == "rate":
        return pct_range([c["k"] / c["n"] for c in use])
    return med_text([s for c in use for s in c["steps"]])


def expected_group_row(V, kind, m, T):
    vals = []
    for g, lv in GG:
        if g == "N 系":
            lines = [f"{n} {c}" for n, ls in NSUB for c in [group_text(V, kind, m, T, ls)] if c is not None]
            vals.append("\n".join(lines) if lines else "无")
        else:
            c = group_text(V, kind, m, T, lv)
            vals.append(c if c is not None else "无")
    return vals


def expected_wide_cell(V, kind, m, T, l):
    c = cell(V, m, T, l)
    if c is None:
        return "无"
    return pct(c["k"], c["n"]) if kind == "rate" else med_text(c["steps"])


def expected_note(V, m, T, l, page_note):
    """DATA.note 期望：T1 含原窗口恢复单元 → 恢复口径；F 有未触发置黑单元 → 如实注明；其余保持网页现值（不含 Cosmos/VPP2 口径错误的保持）。"""
    c = cell(V, m, T, l)
    if m == "Cosmos Policy" and T == "T1" and c.get("acq", {}).get(RESTORED):
        r = c["acq"][RESTORED]; n = c["n"]
        if r < n:
            if page_note == "T1 当前保留集合：部分单元为原闭环（只补想象帧），成败与步数取原闭环":
                return page_note
            return f"T1 当前保留集合：{r}/{n} 个单元为原闭环末窗恢复（只补想象帧，不增加闭环样本），成败与步数取原闭环"
        return f"T1 当前保留集合：{r}/{n} 个单元为原闭环末窗恢复（只补想象帧，不增加闭环样本），成败与步数取原闭环"
    if m == "Cosmos Policy" and l in ("F60", "F160", "F260") and c.get("early"):
        return (f"F 族 11-14 批次：{c['n']} 个独立单元，每单元两遍、取第一遍（不合计为 {2 * c['n']}）；"
                f"其中 {c['early']} 个在置黑前已成功结束、未触发置黑，实际触发 {c['trigger']} 个")
    return page_note


def expected_data_row(V, m, T, l, page_row):
    c = cell(V, m, T, l)
    r = dict(page_row)
    r.update(n=c["n"], success_rate=c["rate"], ci95_lo=c["ci_lo"], ci95_hi=c["ci_hi"], complete="y")
    med = c["median"]
    r["median_first_success_step"] = "" if med is None else (int(med) if float(med).is_integer() else med)
    r["note"] = expected_note(V, m, T, l, page_row.get("note", ""))
    return r


def same_num(a, b):
    if a == "" or b == "" or a is None or b is None:
        return (a in ("", None)) == (b in ("", None))
    return abs(float(a) - float(b)) < 1e-9


DATA_FIELDS = ("n", "success_rate", "ci95_lo", "ci95_hi", "median_first_success_step", "complete", "note")
CAT_FIELDS = ("n", "n_success", "success_rate", "ci95_lo", "ci95_hi", "levels_used", "noted")


def expected_cat_row(V, page_row, notes_by_level):
    key = f"{page_row['model']}|{page_row['task']}|{page_row['category']}"
    c = V["cats"][key]; r = dict(page_row)
    r.update(n=c["n"], n_success=c["k"], levels_used=c["levels_used"])
    if c["n"]:
        r.update(success_rate=c["rate"], ci95_lo=c["ci_lo"], ci95_hi=c["ci_hi"])
        used = [x.replace("N1", "N1_1") if x == "N1" else x for x in c["levels_used"].split()]
        r["noted"] = "y" if any(notes_by_level.get((page_row["model"], page_row["task"], l)) for l in used) else ""
    else:
        r.update(success_rate="", ci95_lo="", ci95_hi="", noted="")
    return r
