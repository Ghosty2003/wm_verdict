"""数据核对 2026-11-15 · 第 3 步：把已确认的差异写进 index_audited.html（从 index_before_audit.html 出发，逐处替换）。
只改：嵌入 DATA / CAT / MNOTES 中 Cosmos Policy、VPP2 的字段；表 2b/4b 中这两个模型的格（格的上色沿用 make_page.py 的 heat_attr 原代码）；
texts.py 列出的数据说明。每处改动记入 edits.json（位置、旧、新），diff_check.py 用它逆向还原验证“除此之外一字未改”。
  python3 apply_fixes.py"""
import html
import json
import re

import audit_lib as A
import texts

SRC = f"{A.HERE}/index_before_audit.html"; DST = f"{A.HERE}/index_audited.html"
s = open(SRC, encoding="utf-8").read()
P = A.parse(s); V = A.load_verified()
EDITS = []          # dict(kind, key, old, new)


# ---------------- make_page.py 的上色与格生成原代码（逐字照抄，保证新格与原生成方式一致）
BAD, WARN, OK = (192, 57, 43), (217, 161, 6), (31, 138, 76)
e = html.escape


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


def wide_row(r, colour):
    return "<tr>" + "".join((f'<th class="model">{e(str(x))}</th>' if j < 2 else cell(x, colour)) for j, x in enumerate(r)) + "</tr>"


def grp_row(r, colour):
    return "<tr>" + "".join((f'<th class="model">{e(str(x))}</th>' if j == 0 else cell(x, colour and j > 0)) for j, x in enumerate(r)) + "</tr>"


def replace_once(s, old, new, kind, key, lo=0, hi=None):
    hi = len(s) if hi is None else hi
    seg = s[lo:hi]; n = seg.count(old)
    assert n == 1, (kind, key, n, old[:120])
    i = lo + seg.index(old)
    EDITS.append(dict(kind=kind, key=key, pos=i, old=old, new=new))
    return s[:i] + new + s[i + len(old):]


# ---------------- 1. DATA
_, da, db = A.get_json(s, "DATA")
new_rows = {}
for r in P["DATA"]:
    k = (r["model"], r["task"], r["level"])
    if r["model"] not in A.MODELS or A.cell(V, *k) is None:
        continue
    ex = A.expected_data_row(V, *k, r)
    if json.dumps(ex, ensure_ascii=False) != json.dumps(r, ensure_ascii=False):
        new_rows[k] = (r, ex)
notes_by_level = {}
for r in P["DATA"]:
    k = (r["model"], r["task"], r["level"])
    if r["model"] in A.MODELS and A.cell(V, *k):
        notes_by_level[k] = A.expected_data_row(V, *k, r)["note"]
for k, (r, ex) in sorted(new_rows.items()):
    old = json.dumps(r, ensure_ascii=False); new = json.dumps(ex, ensure_ascii=False)
    _, da, db = A.get_json(s, "DATA")
    s = replace_once(s, old, new, "DATA", "|".join(k), da, db)

# ---------------- 2. CAT
for r in P["CAT"]["rows"]:
    if r["model"] not in A.MODELS:
        continue
    ex = A.expected_cat_row(V, r, notes_by_level)
    old = json.dumps(r, ensure_ascii=False); new = json.dumps(ex, ensure_ascii=False)
    if old != new:
        _, ca, cb = A.get_json(s, "CAT")
        s = replace_once(s, old, new, "CAT", f"{r['model']}|{r['task']}|{r['category']}", ca, cb)

# ---------------- 3. 静态表格（宽表 + 分组表）
for kind, key, colour in (("rate", "w2b", True), ("step", "w4b", False)):
    P2 = A.parse(s); t = P2[key]; hdr2 = t["head"][1]
    for row in t["rows"]:
        m, T = row[0], row[1]
        if m not in A.MODELS:
            continue
        exp = row[:2] + [A.expected_wide_cell(V, kind, m, T, "N1_1" if hdr2[j] == "N1" else hdr2[j]) for j in range(2, len(row))]
        if exp != row:
            old = wide_row(row, colour); new = wide_row(exp, colour)
            a = s.index(f'id="{"success" if key == "w2b" else "steps"}"'); b = s.index("</section>", a)
            s = replace_once(s, old, new, key, f"{m}|{T}", a, b)
for kind, key, colour in (("rate", "g2b", True), ("step", "g4b", False)):
    P2 = A.parse(s); t = P2[key]
    for row in t["rows"]:
        m, T = row[0].split(" · ")
        if m not in A.MODELS:
            continue
        exp = row[:1] + A.expected_group_row(V, kind, m, T)
        if exp != row:
            old = grp_row(row, colour); new = grp_row(exp, colour)
            a = s.index(f'id="{"success" if key == "g2b" else "steps"}"'); b = s.index("</section>", a)
            s = replace_once(s, old, new, key, f"{m}|{T}", a, b)

# ---------------- 4. 数据说明
for t in texts.edits(V, open(SRC, encoding="utf-8").read()):
    n = s.count(t["old"])
    assert n >= 1, ("旧文字未找到", t["where"], t["old"][:80])
    cur = 0
    while s.find(t["old"], cur) >= 0:
        i = s.find(t["old"], cur)
        EDITS.append(dict(kind="text", key=t["where"], pos=i, old=t["old"], new=t["new"]))
        s = s[:i] + t["new"] + s[i + len(t["old"]):]
        cur = i + len(t["new"])

open(DST, "w", encoding="utf-8").write(s)
json.dump(EDITS, open(f"{A.HERE}/edits.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
import collections
print("edits", len(EDITS), collections.Counter(x["kind"] for x in EDITS))
