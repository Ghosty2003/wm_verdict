"""队友线索逐条对照：web_audit_20261115/cosmos_T0_T1_differences.csv（67 条）与 cosmos_F_available_results.csv（15 条）
在 index_before_audit.html / index_audited.html 中的现值，以及与核验值（verified.json）是否一致。
输出 teammate_clues_status.csv。  python3 crosscheck_teammate.py"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import collections
import csv

import audit_lib as A

SA = _wmvp.P("$COSMOS_DATA_ROOT/web_audit_20261115")
NR = {0.05: 1, 0.1: 2, 0.2: 3, 0.3: 4, 0.4: 5}


def level_of(c):
    if c == "baseline":
        return "none"
    if c == "N1_step50":
        return "N1_1"
    if c.startswith("NR_"):
        return f"NR{NR[round(float(c[3:]), 2)]}"
    if c.startswith("NL_"):
        return f"NL{int(c[3:])}"
    return c


V = A.load_verified()
pages = {nm: {(r["model"], r["task"], r["level"]): r for r in A.parse(open(f"{A.HERE}/{nm}", encoding="utf-8").read())["DATA"]}
         for nm in ("index_before_audit.html", "index_audited.html")}
out = []
for r in csv.DictReader(open(f"{SA}/cosmos_T0_T1_differences.csv", encoding="utf-8-sig")):
    k = ("Cosmos Policy", r["task"], level_of(r["condition"])); c = A.cell(V, *k)
    if r["metric"] == "median_first_success_step":
        cur = "" if r["current_value"] in ("", "None", "nan") else float(r["current_value"]); ver = "" if c["median"] is None else c["median"]
        pv = {nm: p[k]["median_first_success_step"] for nm, p in pages.items()}
    else:
        cur = float(r["current_value"]); ver = c["rate"]
        pv = {nm: p[k]["success_rate"] for nm, p in pages.items()}
    out.append(dict(source="cosmos_T0_T1_differences.csv", task=r["task"], condition=r["condition"], metric=r["metric"],
                    old_snapshot_value=r["page_value"], teammate_current=r["current_value"], verified=ver,
                    page_before=pv["index_before_audit.html"], page_audited=pv["index_audited.html"],
                    teammate_eq_verified=A.same_num(cur, ver), page_before_eq_verified=A.same_num(pv["index_before_audit.html"], ver)))
for r in csv.DictReader(open(f"{SA}/cosmos_F_available_results.csv", encoding="utf-8-sig")):
    k = ("Cosmos Policy", r["task"], r["condition"]); c = A.cell(V, *k)
    for metric, tv, vv, pf in (("primary_successes", r["primary_successes"], c["k"], lambda x: round(x["success_rate"] * x["n"])),
                               ("success_step_median", r["success_step_median"], c["median"], lambda x: x["median_first_success_step"])):
        pv = {nm: pf(p[k]) for nm, p in pages.items()}
        out.append(dict(source="cosmos_F_available_results.csv", task=r["task"], condition=r["condition"], metric=metric,
                        old_snapshot_value="（旧快照无 F）", teammate_current=tv, verified=vv,
                        page_before=pv["index_before_audit.html"], page_audited=pv["index_audited.html"],
                        teammate_eq_verified=A.same_num(tv, vv), page_before_eq_verified=A.same_num(pv["index_before_audit.html"], vv)))
with open(f"{A.HERE}/teammate_clues_status.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(len(out), collections.Counter((o["source"], o["metric"], o["teammate_eq_verified"], o["page_before_eq_verified"]) for o in out))
