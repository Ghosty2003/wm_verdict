"""数据核对 2026-11-15 · 第 2 步：网页（index_before_audit.html）vs 核验值（verified.json）→ diff_list.csv。
数值：嵌入 DATA、CAT（六类汇总）、表 2b/4b 分组表与逐档宽表中 Cosmos Policy / VPP2 的每个格。
说明文字：texts.py 定义的需同步的数据说明（旧文字 → 新文字）。
  python3 compare_page.py [page.html]   （默认 index_before_audit.html）"""
import csv
import json
import sys

import audit_lib as A
import texts

PAGE = sys.argv[1] if len(sys.argv) > 1 else f"{A.HERE}/index_before_audit.html"
s = open(PAGE, encoding="utf-8").read()
P = A.parse(s); V = A.load_verified()
COLS = ["模型", "任务", "条件", "指标", "网页现值", "核验值", "来源文件", "批次", "分母", "原因", "位置"]
rows = []
summary = dict(data_rows=0, cat_rows=0, wide_cells=0, group_cells=0)


def src_of(m, T, l):
    c = A.cell(V, m, T, l)
    return (c["source"], c["batch"], c["n"]) if c else ("", "", "")


def reason_of(m, T, l, metric):
    if m == "Cosmos Policy" and T == "T1" and metric == "note":
        return "T1 该格含原闭环末窗恢复单元，网页未注明口径（主表 ext_models.py 的 acquisition 字符串与源文件不一致，只有混合格带注）；恢复不增加闭环样本"
    if m == "Cosmos Policy" and l.startswith("F") and metric == "note":
        return "F 族有单元在置黑前已成功结束、未触发置黑，网页未注明（规则 7）"
    if metric == "noted":
        return "该类所含档位的口径说明变化后，按 cat_summary.py 同一规则重算 noted 标记"
    return "网页值与源文件逐单元重算不一致"


# ---------------- DATA
page_rows = {(r["model"], r["task"], r["level"]): r for r in P["DATA"] if r["model"] in A.MODELS}
exp_keys = {tuple(k.split("|")) for k in V["cells"]}
for k in sorted(exp_keys - set(page_rows)):
    rows.append(dict(zip(COLS, [k[0], k[1], k[2], "DATA 行", "（缺）", "有核验数据", *src_of(*k), "网页 DATA 缺此格", "DATA"])))
for k in sorted(set(page_rows) - exp_keys):
    rows.append(dict(zip(COLS, [k[0], k[1], k[2], "DATA 行", "有", "（源中无）", "", "", "", "网页有、源中无", "DATA"])))
for k, r in sorted(page_rows.items()):
    if k not in exp_keys:
        continue
    summary["data_rows"] += 1
    e = A.expected_data_row(V, *k, r)
    for f in A.DATA_FIELDS:
        a, b = r.get(f), e.get(f)
        same = (a == b) if f in ("note", "complete") else A.same_num(a, b)
        if not same:
            rows.append(dict(zip(COLS, [k[0], k[1], k[2], f"DATA.{f}", a, b, *src_of(*k), reason_of(*k, f), "嵌入 DATA"])))
notes_by_level = {k: A.expected_data_row(V, *k, r)["note"] for k, r in page_rows.items() if k in exp_keys}

# ---------------- CAT
for r in P["CAT"]["rows"]:
    if r["model"] not in A.MODELS:
        continue
    summary["cat_rows"] += 1
    e = A.expected_cat_row(V, r, notes_by_level)
    for f in A.CAT_FIELDS:
        a, b = r.get(f), e.get(f)
        same = (a == b) if f in ("levels_used", "noted") else A.same_num(a, b)
        if not same:
            rows.append(dict(zip(COLS, [r["model"], r["task"], r["category"], f"CAT.{f}", a, b,
                                        "verified.json 逐档合计（P→环境·一次性，O→信息·一次性，不含 NLw）", "同一批次逐档值重算",
                                        e["n"], reason_of(r["model"], r["task"], r["category"], f), "嵌入 CAT（六类汇总）"])))

# ---------------- 宽表
for kind, key in (("rate", "w2b"), ("step", "w4b")):
    t = P[key]; hdr2 = t["head"][1]
    for row in t["rows"]:
        m, T = row[0], row[1]
        if m not in A.MODELS:
            continue
        for j in range(2, len(row)):
            l = "N1_1" if hdr2[j] == "N1" else hdr2[j]
            summary["wide_cells"] += 1
            e = A.expected_wide_cell(V, kind, m, T, l)
            if row[j] != e:
                rows.append(dict(zip(COLS, [m, T, l, f"{'表2b' if kind == 'rate' else '表4b'}宽表", row[j], e, *src_of(m, T, l),
                                            "静态宽表与源文件逐单元重算不一致", key])))
# ---------------- 分组表
for kind, key in (("rate", "g2b"), ("step", "g4b")):
    t = P[key]; hdr = t["head"][0]
    for row in t["rows"]:
        lab = row[0]
        m, T = lab.split(" · ")
        if m not in A.MODELS:
            continue
        e = A.expected_group_row(V, kind, m, T)
        for j, (g, ev) in enumerate(zip(hdr[1:], e), 1):
            summary["group_cells"] += 1
            if row[j] != ev:
                rows.append(dict(zip(COLS, [m, T, g, f"{'表2b' if kind == 'rate' else '表4b'}分组", row[j], ev, "verified.json 组内各档", "", "",
                                            "分组表与源文件重算不一致", key])))

# ---------------- 说明文字
for t in texts.edits(V, s):
    if t["old"] == t["new"]:
        continue
    present_new = s.count(t["new"]) if t["new"] else 0
    present_old = s.count(t["old"]) - (present_new if t["old"] in t["new"] else 0)
    if present_old:
        rows.append(dict(zip(COLS, [t["model"], t.get("task", "—"), t.get("cond", "—"), "说明文字", t["old"], t["new"], t["source"], t.get("batch", ""),
                                    t.get("denom", ""), t["reason"], f"{t['where']}（{present_old} 处）"])))
    elif t["new"] and not present_new:
        rows.append(dict(zip(COLS, [t["model"], "—", "—", "说明文字", "（未找到旧文字）", t["new"], t["source"], "", "", "旧文字定位失败", t["where"]])))

out = sys.argv[2] if len(sys.argv) > 2 else f"{A.HERE}/diff_list.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
print(json.dumps(summary, ensure_ascii=False), "差异", len(rows))
import collections
print(collections.Counter((r["模型"], r["任务"], r["指标"]) for r in rows))
