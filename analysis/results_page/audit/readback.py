"""数据核对 2026-11-15 · 第 5 步：从 index_audited.html 读回 Cosmos Policy / VPP2 的全部数据字段，与核验值（verified.json，源文件逐单元重算）逐一对比。
检查项：
  A. 嵌入 DATA 每格 n / 成功率 / Wilson CI / 步数中位 / complete / 口径说明 = 核验值（无成功 → 空，不是 0）；格集合与源一致；
  B. 嵌入 CAT（六类汇总）每格 = 同批次逐档值按 P→环境·一次性、O→信息·一次性 重新加总；
  C. 表 2b / 4b 宽表与分组表每格 = 核验值；
  D. DATA 与静态表格同一个数互相一致（成功率百分比、步数中位）；
  E. texts.py 的新说明都在、旧说明都不在；
  F. 其他模型（LingBot-VA / OpenWAM-α / UVA）的 DATA、CAT、表格行与原副本完全相同。
输出 readback_result.json；不符项 > 0 时退出码 1。  python3 readback.py"""
import json
import sys

import audit_lib as A
import texts

X = open(f"{A.HERE}/index_audited.html", encoding="utf-8").read()
B = open(f"{A.HERE}/index_before_audit.html", encoding="utf-8").read()
P = A.parse(X); P0 = A.parse(B); V = A.load_verified()
bad = []; n = dict(A=0, B=0, C=0, D=0, E=0, F=0)

# A
page = {(r["model"], r["task"], r["level"]): r for r in P["DATA"] if r["model"] in A.MODELS}
keys = {tuple(k.split("|")) for k in V["cells"]}
if set(page) != keys:
    bad.append(("A", "格集合不同", sorted(set(page) ^ keys)))
for k in sorted(keys & set(page)):
    r = page[k]; c = A.cell(V, *k)
    chk = [("n", r["n"], c["n"]), ("success_rate", r["success_rate"], c["rate"]), ("ci95_lo", r["ci95_lo"], c["ci_lo"]),
           ("ci95_hi", r["ci95_hi"], c["ci_hi"]), ("median", r["median_first_success_step"], "" if c["median"] is None else c["median"]),
           ("complete", r["complete"], "y"), ("k", round(r["success_rate"] * r["n"]), c["k"])]
    for f, a, b in chk:
        n["A"] += 1
        if not (a == b if f == "complete" else A.same_num(a, b)):
            bad.append(("A", k, f, a, b))
    if c["k"] == 0 and r["median_first_success_step"] != "":
        bad.append(("A", k, "无成功单元的步数应为空", r["median_first_success_step"]))
    n["A"] += 1
    exp_note = A.expected_data_row(V, *k, P0["DATA"][[i for i, x in enumerate(P0["DATA"]) if (x["model"], x["task"], x["level"]) == k][0]])["note"]
    if r["note"] != exp_note:
        bad.append(("A", k, "note", r["note"], exp_note))
notes_by_level = {k: page[k]["note"] for k in page}
# B
for r in P["CAT"]["rows"]:
    if r["model"] not in A.MODELS:
        continue
    e = A.expected_cat_row(V, r, notes_by_level)
    for f in A.CAT_FIELDS:
        n["B"] += 1
        same = (r[f] == e[f]) if f in ("levels_used", "noted") else A.same_num(r[f], e[f])
        if not same:
            bad.append(("B", r["model"], r["task"], r["category"], f, r[f], e[f]))
# C
for kind, key in (("rate", "w2b"), ("step", "w4b")):
    t = P[key]; hdr2 = t["head"][1]
    for row in t["rows"]:
        if row[0] not in A.MODELS:
            continue
        for j in range(2, len(row)):
            l = "N1_1" if hdr2[j] == "N1" else hdr2[j]; n["C"] += 1
            ev = A.expected_wide_cell(V, kind, row[0], row[1], l)
            if row[j] != ev:
                bad.append(("C", key, row[0], row[1], l, row[j], ev))
            # D：与 DATA 一致
            r = page.get((row[0], row[1], l))
            if r is not None:
                n["D"] += 1
                if kind == "rate" and row[j] != f"{round(100 * r['success_rate'])}%":
                    bad.append(("D", key, row[0], row[1], l, row[j], r["success_rate"]))
                if kind == "step":
                    md = r["median_first_success_step"]
                    want = "无成功" if md == "" else A.fmt_med(md)
                    if row[j].split(" ")[0] != want:
                        bad.append(("D", key, row[0], row[1], l, row[j], md))
for kind, key in (("rate", "g2b"), ("step", "g4b")):
    for row in P[key]["rows"]:
        m, T = row[0].split(" · ")
        if m not in A.MODELS:
            continue
        ev = A.expected_group_row(V, kind, m, T)
        for j, x in enumerate(ev, 1):
            n["C"] += 1
            if row[j] != x:
                bad.append(("C", key, m, T, P[key]["head"][0][j], row[j], x))
# E
for t in texts.edits(V, B):
    n["E"] += 1
    if t["new"] and t["new"] not in X:
        bad.append(("E", t["where"], "新说明不在"))
    if t["old"] in X and t["old"] not in t["new"]:
        bad.append(("E", t["where"], "旧说明仍在"))
# F
for key in ("DATA",):
    o = [r for r in P0[key] if r["model"] not in A.MODELS]; x = [r for r in P[key] if r["model"] not in A.MODELS]
    n["F"] += 1
    if o != x:
        bad.append(("F", key))
o = [r for r in P0["CAT"]["rows"] if r["model"] not in A.MODELS]; x = [r for r in P["CAT"]["rows"] if r["model"] not in A.MODELS]
n["F"] += 1
if o != x or P0["CAT"]["cats"] != P["CAT"]["cats"]:
    bad.append(("F", "CAT"))
for key in ("w2b", "w4b"):
    n["F"] += 1
    if [r for r in P0[key]["rows"] if r[0] not in A.MODELS] != [r for r in P[key]["rows"] if r[0] not in A.MODELS] or P0[key]["head"] != P[key]["head"]:
        bad.append(("F", key))
for key in ("g2b", "g4b"):
    n["F"] += 1
    if [r for r in P0[key]["rows"] if r[0].split(" · ")[0] not in A.MODELS] != [r for r in P[key]["rows"] if r[0].split(" · ")[0] not in A.MODELS] or P0[key]["head"] != P[key]["head"]:
        bad.append(("F", key))
n["F"] += 1
if {k: v for k, v in P0["MNOTES"].items() if k not in A.MODELS} != {k: v for k, v in P["MNOTES"].items() if k not in A.MODELS}:
    bad.append(("F", "MNOTES"))

res = dict(checked=n, total_checked=sum(n.values()), mismatches=len(bad), detail=[list(map(str, b)) for b in bad],
           verified_source_checks=V["checks"])
json.dump(res, open(f"{A.HERE}/readback_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)[:3000])
print("READBACK", "OK：0 处不符" if not bad else f"FAIL：{len(bad)} 处不符")
sys.exit(1 if bad else 0)
