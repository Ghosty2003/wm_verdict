"""F（单次全帧置黑）vs K（中途推动）逐档对比，同一名义起始网格 t ∈ {60, 160, 260}（用户 2026-11-13 裁决：双口径）。
口径 1（主，与主表同）：全 50 单元闭环成功率。
口径 2（补充）："置黑 / 推动时尚未成功"单元的条件成功率。
  F：直接用记录字段——尚未成功 ⇔ blackout_applied 为真且 blackout_after_success 不为真
     （LingBot 成功即停，置黑前已成功的单元 blackout_applied=False；OpenWAM 照跑，置黑前已成功的标 blackout_after_success=True）。
  K：记录里没有对应字段，按记录推：尚未成功 ⇔ 不是"成功且首次成功步 ≤ 推动步"（推动步 = 名义 t，K 族推动就在第 t 步之后）。
晚起始档（尤其 260）条件子集样本极小、CI 宽，对比重心在早中起始档（60、160）。
F 的实际置黑步按模型不同（LingBot 60/172/268，OpenWAM 64/160/288），对比在名义档位层面进行。
输入：F = runs/{model}_famF/closed_t<tid>.*.jsonl（第一遍 closed_loop）；K = closed_main/<任务>/ 下已按 H=400 口径合并的闭环 jsonl。
输出：closed_main/F_K对比_<D>.tsv、.md
  python3 fk_compare.py 20261113
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import collections
import csv
import glob
import json
import math
import os
import sys

D = sys.argv[1]
OUT = _wmvp.P("$WMV_ROOT/libero/out")
TASKS = [("T0", "T0_t5", 5), ("T1", "T1_t3", 3), ("T2", "T2_t2", 2), ("T3", "T3_t4", 4), ("T4", "T4_t6", 6)]
MN = {"lingbot": "LingBot-VA", "openwam": "OpenWAM-α"}
F_ACT = {"lingbot": {60: 60, 160: 172, 260: 268}, "openwam": {60: 64, 160: 160, 260: 288}}
K_T = {1: 60, 2: 60, 3: 60, 4: 160, 5: 160, 6: 160, 7: 260, 8: 260, 9: 260}
K_D = {1: 2, 2: 5, 3: 10, 4: 2, 5: 5, 6: 10, 7: 2, 8: 5, 9: 10}


def wilson(k, n):
    if not n:
        return None, None
    p = k / n; z = 1.96; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - s) / d, (c + s) / d


def level(r):
    s = r.get("setting_id") or ""
    return s.split("__")[-1] if "__" in s else s


def main():
    acc = collections.defaultdict(lambda: [0, 0, 0, 0])      # n, k, n_pending, k_pending
    for m in ("lingbot", "openwam"):
        for T, Td, tid in TASKS:
            seen = set()
            for f in sorted(glob.glob(f"{OUT}/runs/{m}_famF/closed_t{tid}.*.jsonl")):
                for line in open(f):
                    r = json.loads(line); cl = r.get("closed_loop") or {}
                    if r["unit_id"] in seen or cl.get("success") is None or r["setting_id"] == "F0":
                        continue
                    seen.add(r["unit_id"])
                    t = int(r["setting_id"][1:]); a = acc[(m, T, "F", t, "")]
                    pend = bool(r.get("blackout_applied")) and not r.get("blackout_after_success")
                    a[0] += 1; a[1] += bool(cl["success"]); a[2] += pend; a[3] += pend and bool(cl["success"])
            for f in sorted(glob.glob(f"{OUT}/closed_main/{Td}/{m}_*.jsonl")):
                for line in open(f):
                    r = json.loads(line); lv = level(r); cl = r.get("closed_loop") or {}
                    if not (lv.startswith("K") and lv[1:].isdigit()) or cl.get("success") is None or r["unit_id"] in seen:
                        continue
                    seen.add(r["unit_id"])
                    i = int(lv[1:]); t = K_T[i]; ok = bool(cl["success"]); fs = cl.get("first_success_step")
                    pend = not (ok and fs is not None and fs <= t)
                    a = acc[(m, T, "K", t, f"{K_D[i]} cm")]
                    a[0] += 1; a[1] += ok; a[2] += pend; a[3] += pend and ok
    rows = []
    for (m, T, fam, t, d), (n, k, np_, kp) in sorted(acc.items(), key=lambda x: (x[0][0], x[0][1], x[0][3], x[0][2], x[0][4])):
        lo, hi = wilson(k, n); plo, phi = wilson(kp, np_)
        rows.append(dict(model=MN[m], task=T, nominal_t=t, family=fam, level=(f"F{t}" if fam == "F" else d),
                         actual_event_step=F_ACT[m][t] if fam == "F" else t, n=n, n_success=k, rate=round(k / n, 4) if n else "",
                         ci95=f"{lo:.2f}-{hi:.2f}" if n else "", n_pending=np_, n_pending_success=kp,
                         cond_rate=round(kp / np_, 4) if np_ else "", cond_ci95=f"{plo:.2f}-{phi:.2f}" if np_ else "",
                         complete="y" if n >= 50 else "partial"))
    with open(f"{OUT}/closed_main/F_K对比_{D}.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)
    pc = lambda x: "—" if x == "" else f"{100 * x:.0f}%"
    L = [f"# F（单次全帧置黑）vs K（中途推动）· 同一名义起始网格 · {D}（脚本生成）", "",
         "双口径：**全部单元**成功率（与主表同口径，k/n）｜**事件发生时尚未成功**单元的条件成功率（k/n）。"
         "F 的条件子集取自记录字段 blackout_applied / blackout_after_success；K 记录无对应字段，按"
         "\"成功且首次成功步 ≤ 推动步\"判为已成功。晚起始档（尤其 t=260）条件子集样本极小、CI 宽，对比重心在早中起始档（60、160）。"
         "F 的实际置黑步：LingBot 60/172/268，OpenWAM 64/160/288，对比在名义档位层面进行。", ""]
    for m in ("LingBot-VA", "OpenWAM-α"):
        L += [f"## {m}", "", "| 任务 | 名义 t | F 全部 | F 条件 | K 2 cm 全部 | K 2 cm 条件 | K 5 cm 全部 | K 5 cm 条件 | K 10 cm 全部 | K 10 cm 条件 |", "|---" * 10 + "|"]
        for T, _, _ in TASKS:
            for t in (60, 160, 260):
                cell = {(r["family"], r["level"]): r for r in rows if r["model"] == m and r["task"] == T and r["nominal_t"] == t}
                def two(key):
                    r = cell.get(key)
                    if not r:
                        return ["待出", "待出"]
                    a = f"{pc(r['rate'])} ({r['n_success']}/{r['n']})" + ("" if r["complete"] == "y" else " 未满")
                    b = f"{pc(r['cond_rate'])} ({r['n_pending_success']}/{r['n_pending']})"
                    return [a, b]
                L.append("| " + " | ".join([T, str(t)] + two(("F", f"F{t}")) + two(("K", "2 cm")) + two(("K", "5 cm")) + two(("K", "10 cm"))) + " |")
        L.append("")
    open(f"{OUT}/closed_main/F_K对比_{D}.md", "w").write("\n".join(L) + "\n")
    print("rows", len(rows), "F partial", sum(1 for r in rows if r["family"] == "F" and r["complete"] != "y"))


if __name__ == "__main__":
    main()
