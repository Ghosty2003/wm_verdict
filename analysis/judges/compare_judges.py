#!/usr/bin/env python3
"""All judges side by side, from existing label files only (no API). Binary mapping:
main = looks_success|success -> success; wide = + borderline. uncertain/unknown excluded from denominators (counted)."""
import csv, collections, math, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(os.environ.get("WMV_ROOT", "$WMV_ROOT"), "libero", "out")
PKG = f"{OUT}/judge_package_20261117"; VO = f"{PKG}/vlm_openrouter"; JB = f"{OUT}/judge_binary_20261124"
KEY = {r["frame_id"]: r for r in csv.DictReader(open(f"{PKG}/_key/key.csv"))}
def load(p): return {r["frame_id"]: r["label"] for r in csv.DictReader(open(p))}
J = [("qwen3-vl-235b (full)", f"{VO}/labels_qwen3vl235b.csv"), ("qwen4-max (full, main judge)", f"{VO}/labels_qwen4max.csv"),
     ("qwen3-vl-235b calib A", f"{VO}/labels_calibA.csv"), ("qwen3-vl-235b calib B (larger image input)", f"{VO}/labels_calibB.csv"),
     ("qwen3.5-397b calib", f"{VO}/labels_calib_qwen35_397b.csv"), ("qwen3.8-max calib", f"{VO}/labels_calib_qwen38max.csv"),
     ("qwen4-max calib", f"{VO}/labels_calib_qwen4max.csv"), ("qwen4-max BINARY pilot (2026-11-24)", f"{JB}/vlm_openrouter/labels_pilot_binary.csv")]
L = {n: load(p) for n, p in J}
L["Agent judge (11-14/15)"] = {f: r["agent_label"] for f, r in KEY.items() if r["agent_label"]}
L["CNN detector"] = {f: {"success": "looks_success", "failure": "not_success"}[r["cnn_verdict"]] for f, r in KEY.items() if r["cnn_verdict"]}
POS_MAIN = {"looks_success", "success"}; POS_WIDE = POS_MAIN | {"borderline"}; UNC = {"uncertain", "unknown", ""}
def wil(k, n, z=1.96):
    if not n: return float("nan"), float("nan")
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h, c+h
def pct(k, n): return f"{100*k/n:.1f}% ({k}/{n})" if n else "—"
def kappa(a, b):
    n = len(a); 
    if n == 0: return float("nan")
    po = sum(x == y for x, y in zip(a, b))/n; pa = sum(a)/n; pb = sum(b)/n; pe = pa*pb+(1-pa)*(1-pb)
    return (po-pe)/(1-pe) if pe < 1 else float("nan")
md = ["# VLM / judge comparison (all existing labels, 2026-11-24)", "",
      "Binary mapping: **main** = looks_success (or `success`) counts as success; **wide** = + borderline. uncertain excluded from denominators.",
      "TPR = judged success on REAL-success frames (sanity set); Optimism = judged success on REAL-failure frames. Same frames across judges only within the calibration table (section 3).", ""]
# 1. coverage + label mix
md += ["## 1. Coverage and label mix", "", "| judge | frames | looks_success/success | borderline | not_success | uncertain |", "|---|---|---|---|---|---|"]
for n, d in L.items():
    c = collections.Counter(d.values()); t = len(d)
    md.append(f"| {n} | {t} | {pct(c['looks_success']+c['success'], t)} | {pct(c['borderline'], t)} | {pct(c['not_success'], t)} | {pct(c['uncertain']+c['unknown'], t)} |")
# 2. vs simulator truth, per task (only frames with a key row)
md += ["", "## 2. Versus simulator truth, per task (main / wide)", ""]
for n, d in L.items():
    md += [f"### {n}", "", "| task | TPR main | TPR wide | Optimism main | Optimism wide |", "|---|---|---|---|---|"]
    for t in ("T0", "T1", "T2", "T3", "T4", "ALL"):
        cell = {}
        for truth in ("True", "False"):
            fs = [f for f in d if f in KEY and KEY[f]["real_success"] == truth and (t == "ALL" or KEY[f]["task"] == t) and d[f] not in UNC]
            cell[truth] = (sum(d[f] in POS_MAIN for f in fs), sum(d[f] in POS_WIDE for f in fs), len(fs))
        s, f_ = cell["True"], cell["False"]
        if s[2] + f_[2] == 0: continue
        md.append(f"| {t} | {pct(s[0], s[2])} | {pct(s[1], s[2])} | {pct(f_[0], f_[2])} | {pct(f_[1], f_[2])} |")
    md.append("")
# 3. calibration set: same frames for all judges
calib = [l.strip() for l in open(f"{VO}/calib_ids.txt") if l.strip()]
H = {"Jasmine (40-frame set)": {f: r["jasmine40_task"] for f, r in KEY.items() if r["jasmine40_task"]},
     "rater 3 (40-frame set)": {f: r["rater3_task"] for f, r in KEY.items() if r["rater3_task"]},
     "$COLLAB_COSMOS (Cosmos)": {f: r["collab_cosmos_label"] for f, r in KEY.items() if r["collab_cosmos_label"]}}
md += ["## 3. Agreement with humans (binary main; human success vs failure, unknown/uncertain dropped)", "",
       "| judge | " + " | ".join(f"vs {h}: agree, κ (n)" for h in H) + " |", "|---|" + "---|"*len(H)]
for n, d in L.items():
    cells = []
    for h, hd in H.items():
        fs = [f for f in d if f in hd and hd[f] in ("success", "failure") and d[f] not in UNC]
        a = [d[f] in POS_MAIN for f in fs]; b = [hd[f] == "success" for f in fs]
        cells.append(f"{100*sum(x==y for x,y in zip(a,b))/len(fs):.0f}%, κ={kappa(a,b):.2f} (n={len(fs)})" if fs else "—")
    md.append(f"| {n} | " + " | ".join(cells) + " |")
md += ["", f"## 4. Calibration set ({len(calib)} frames): every judge on the same frames", "",
       "| judge | labelled | TPR main | Optimism main | agree with the agent judge (binary main) |", "|---|---|---|---|---|"]
for n, d in L.items():
    fs = [f for f in calib if f in d]
    s = [f for f in fs if KEY[f]["real_success"] == "True" and d[f] not in UNC]; fl = [f for f in fs if KEY[f]["real_success"] == "False" and d[f] not in UNC]
    ag = [f for f in fs if f in L["Agent judge (11-14/15)"] and L["Agent judge (11-14/15)"][f] not in UNC and d[f] not in UNC]
    agree = sum((d[f] in POS_MAIN) == (L["Agent judge (11-14/15)"][f] in POS_MAIN) for f in ag)
    md.append(f"| {n} | {len(fs)} | {pct(sum(d[f] in POS_MAIN for f in s), len(s))} | {pct(sum(d[f] in POS_MAIN for f in fl), len(fl))} | {pct(agree, len(ag))} |")
open(f"{HERE}/vlm_compare.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
