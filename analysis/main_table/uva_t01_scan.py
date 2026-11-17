"""UVA T0/T1（作者 11-02~11-04 成对开/闭环批次，task5 = T0、task3 = T1）闭环逐单元扫描 → uva_t01_cells.json。
只读作者 NAS 目录，不改源文件。只取 closed/，每单元一个 result.json（老布局，无 repeat 子目录）。
来源：task-split/T{0,1}/{action,friction,initial,visual}-batch/<cond>/closed/、physical-batch/<cond>/closed/（两任务同目录）、
      baseline50-closed/（无扰动基线）。顶层 initial-batch 与 task-split/initial-batch 若都有，逐单元比对是否一致。
  python3 uva_t01_scan.py
"""
import collections
import glob
import json
import os
import re

import ext_models as E

NC = E.UVA_NAS
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uva_t01_cells.json")
TASK = {"T0": 5, "T1": 3}


def fam_lv(cond):
    m = re.match(r"(.+)_([\d.]+)$", cond)
    return m.group(1), float(m.group(2))


def read_units(d, tid):
    U = {}
    for f in sorted(glob.glob(f"{d}/task{tid}_*/result.json")):
        r = json.load(open(f))
        U[os.path.basename(os.path.dirname(f))] = r
    return U


def main():
    srcs = []
    for T, tid in TASK.items():
        for b in ("action-batch", "friction-batch", "initial-batch", "visual-batch"):
            for d in sorted(glob.glob(f"{NC}/task-split/{T}/{b}/*/closed")):
                srcs.append((T, tid, os.path.basename(os.path.dirname(d)), d))
        for d in sorted(glob.glob(f"{NC}/physical-batch/*/closed")):
            srcs.append((T, tid, os.path.basename(os.path.dirname(d)), d))
        srcs.append((T, tid, "baseline_0.0", f"{NC}/baseline50-closed"))
        have = {c for (TT, _, c, _) in srcs if TT == T}
        for d in sorted(glob.glob(f"{NC}/initial-batch/*/closed")):          # A0/A1 只在顶层 initial-batch（两任务同目录）
            c = os.path.basename(os.path.dirname(d))
            if c not in have:
                srcs.append((T, tid, c, d))
    cells, issues = [], []
    for T, tid, cond, d in srcs:
        U = read_units(d, tid)
        fam, lv = fam_lv(cond)
        level = E.uva_level(fam, lv) if fam not in ("A", "P") else None
        if fam in ("A", "P"):            # A/P 档位下标：A 用 0..4 原样；P 用噪声幅度 → P0..P4
            level = f"A{int(lv)}" if fam == "A" else {0.0: "P0", 0.02: "P1", 0.06: "P2", 0.12: "P3", 0.24: "P4"}[lv]
        bad = [u for u, r in U.items() if r.get("horizon") != 400 or r.get("diagnostic_only") or r.get("task_id") != tid]
        if bad:
            issues.append((T, cond, "horizon/diagnostic/task mismatch", bad[:3]))
        k = sum(bool(r.get("native_success")) for r in U.values())
        steps = [int(r["first_native_success_step"]) for r in U.values() if r.get("native_success") and r.get("first_native_success_step") is not None]
        pc = f"{d}/probe-complete.json"                  # 作者自己的逐单元汇总，与 result.json 逐单元核对
        if os.path.exists(pc):
            rows = [r for r in json.load(open(pc))["rows"] if r.get("task_id") == tid]
            kp = sum(bool(r.get("native_success")) for r in rows)
            if (len(rows), kp) != (len(U), k):
                issues.append((T, cond, "probe-complete mismatch", len(rows), kp, len(U), k))
        else:
            issues.append((T, cond, "no probe-complete.json"))
        cells.append(dict(task=T, cond=cond, family=fam, level_value=lv, level=level, n=len(U), k=k, steps=steps, closed_dir=d,
                          formal_results=sorted({str(r.get("formal_results")) for r in U.values()}),
                          replan=sorted({str(r.get("replan")) for r in U.values()})))
    # 顶层 initial-batch（两任务同目录）与 task-split 的一致性
    for d in sorted(glob.glob(f"{NC}/initial-batch/*/closed")):
        cond = os.path.basename(os.path.dirname(d))
        for T, tid in TASK.items():
            top = read_units(d, tid)
            ts = read_units(f"{NC}/task-split/{T}/initial-batch/{cond}/closed", tid)
            if top and ts:
                diff = [u for u in top if u in ts and (top[u].get("native_success"), top[u].get("first_native_success_step")) !=
                        (ts[u].get("native_success"), ts[u].get("first_native_success_step"))]
                if diff or set(top) != set(ts):
                    issues.append((T, cond, "top-level vs task-split initial-batch differ", len(diff), len(top), len(ts)))
    json.dump(dict(cells=cells, issues=issues), open(OUT, "w"), ensure_ascii=False, indent=0)
    c = collections.Counter((x["task"], x["n"]) for x in cells)
    print("cells", len(cells), "n-distribution", sorted(c.items()))
    for x in cells:
        print(f'{x["task"]} {x["cond"]:22s} {str(x["level"]):6s} n={x["n"]:3d} k={x["k"]:3d} formal={x["formal_results"]} replan={x["replan"]}')
    print("issues", issues)


if __name__ == "__main__":
    main()
