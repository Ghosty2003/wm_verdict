#!/usr/bin/env python3
"""Score an agent judge on the merged 10,487-frame package against the key (same metrics as score_opus55.py),
and compare it with Opus 5.5, qwen4-max (4-way), CNN, humans (package columns) and the site human labels.
Usage: python3 score_agent.py <name> <labels.csv> <selfcheck.csv> <out_dir>
Main = looks_success counts as accepted; wide = + borderline; uncertain excluded from denominators."""
import csv, glob, collections, math, json, os, re, sys
NAME, LAB, SC, OD = sys.argv[1:5]
OUT = os.path.join(os.environ.get("WMV_ROOT", "$WMV_ROOT"), "libero", "out")
KEY = {r["new_id"]: r for r in csv.DictReader(open(f"{OUT}/judge_package_all_20261124/key/key_all.csv"))}
uva = {r["frame_id"]: r for r in csv.DictReader(open(f"{OUT}/judge_package_20261117/uva_ext/key_uva.csv"))}
pk = {r["frame_id"]: r for r in csv.DictReader(open(f"{OUT}/judge_package_20261117/_key/key.csv"))}
AG = {r["frame_id"]: r["label"] for r in csv.DictReader(open(LAB))}
OP = {}
for f in glob.glob(f"{OUT}/agent_judge_opus55_20261124/raw/labels_opus55_self_T*.csv"):
    for r in csv.DictReader(open(f)): OP[r["frame_id"]] = r["label"]
Q = {}
for f, src in ((f"{OUT}/judge_package_20261117/vlm_openrouter/labels_qwen4max.csv", "pkg20261117"),
               (f"{OUT}/judge_gap_20261124/vlm_openrouter/labels_gap_qwen4max.csv", "gap20261124"),
               (f"{OUT}/judge_package_20261117/vlm_openrouter/labels_uva_qwen4max.csv", "uva_ext")):
    for r in csv.DictReader(open(f)): Q[(src, r["frame_id"])] = r["label"]
def truth(k):
    if k["src_package"] == "uva_ext" and uva.get(k["src_frame_id"], {}).get("slot_status") != "have_real": return None
    return {"True": True, "False": False}.get(k["real_success"])
POS, WIDE, UNC = {"looks_success"}, {"looks_success", "borderline"}, {"uncertain"}
def wil(k, n, z=1.96):
    if not n: return float("nan"), float("nan")
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h, c+h
def pct(k, n):
    if not n: return "—"
    lo, hi = wil(k, n); return f"{100*k/n:.1f}% ({k}/{n}) [{100*lo:.0f}–{100*hi:.0f}]"
def kappa(a, b):
    n = len(a)
    if not n: return float("nan")
    po = sum(x == y for x, y in zip(a, b))/n; pa = sum(a)/n; pb = sum(b)/n; pe = pa*pb+(1-pa)*(1-pb)
    return (po-pe)/(1-pe) if pe < 1 else float("nan")
md = [f"# Agent judge `{NAME}` — scored against the key", "",
      f"Labeled frames: {len(AG)} of {len(KEY)}; unknown ids: {len([i for i in AG if i not in KEY])}; labels: {dict(collections.Counter(AG.values()))}", "",
      "## TPR (real success judged success) and optimism (real failure judged success); uncertain excluded", "",
      "| task | model | TPR main | TPR wide | optimism main | optimism wide | uncertain (fail frames) |", "|---|---|---|---|---|---|---|"]
rows = []
for t in ("T0", "T1", "T2", "T3", "T4", "ALL"):
    for m in ("lingbot", "openwam", "cosmos", "uva", "ALL"):
        S = [k for k in KEY.values() if (t == "ALL" or k["task"] == t) and (m == "ALL" or k["model"] == m) and k["new_id"] in AG and truth(k) is not None]
        if not S: continue
        s = [k for k in S if truth(k) and AG[k["new_id"]] not in UNC]; f = [k for k in S if not truth(k) and AG[k["new_id"]] not in UNC]
        unc = sum(1 for k in S if not truth(k) and AG[k["new_id"]] in UNC)
        c = lambda L, P: sum(AG[k["new_id"]] in P for k in L)
        md.append(f"| {t} | {m} | {pct(c(s, POS), len(s))} | {pct(c(s, WIDE), len(s))} | {pct(c(f, POS), len(f))} | {pct(c(f, WIDE), len(f))} | {unc} |")
        rows.append(dict(task=t, model=m, n_succ=len(s), tpr_main=c(s, POS), tpr_wide=c(s, WIDE), n_fail=len(f), opt_main=c(f, POS), opt_wide=c(f, WIDE), unc_fail=unc))
def compare(name, other):
    P = [(AG[i], other[i]) for i in AG if i in other]
    bm = [(a in POS, b in POS) for a, b in P if a not in UNC and b not in UNC]; bw = [(a in WIDE, b in WIDE) for a, b in P if a not in UNC and b not in UNC]
    return [f"| {name} | {len(P)} | {pct(sum(a == b for a, b in P), len(P))} | {kappa([x for x,_ in bm],[y for _,y in bm]):.2f} | {kappa([x for x,_ in bw],[y for _,y in bw]):.2f} |"]
qn = {k["new_id"]: Q[(k["src_package"], k["src_frame_id"])] for k in KEY.values() if (k["src_package"], k["src_frame_id"]) in Q}
md += ["", "## Agreement with the other automated judges (same frames)", "", "| judge | n | exact 4-way | kappa main | kappa wide |", "|---|---|---|---|---|"]
md += compare("Opus 5.5 agent", OP) + compare("qwen4-max (4-way)", qn)
H = {"Jasmine 40": "jasmine40_task", "rater 3 (40)": "rater3_task", "$COLLAB_COSMOS (Cosmos)": "collab_cosmos_label"}
md += ["", "## Agreement with package human labels / CNN (binary main)", "", "| reference | n | agree | kappa |", "|---|---|---|---|"]
pkg = [k for k in KEY.values() if k["src_package"] == "pkg20261117" and k["new_id"] in AG and AG[k["new_id"]] not in UNC]
for name, col in list(H.items()) + [("CNN detector", "cnn_verdict")]:
    L = [k for k in pkg if pk[k["src_frame_id"]][col] in ("success", "failure")]
    x = [AG[k["new_id"]] in POS for k in L]; y = [pk[k["src_frame_id"]][col] == "success" for k in L]
    md.append(f"| {name} | {len(L)} | {pct(sum(p == q for p, q in zip(x, y)), len(L))} | {kappa(x, y):.2f} |")
sc = list(csv.DictReader(open(SC)))
fl = "first_label" if "first_label" in sc[0] else list(sc[0].keys())[1]; sl = "second_label" if "second_label" in sc[0] else list(sc[0].keys())[2]
md += ["", f"## Self-check: {sum(r[fl] == r[sl] for r in sc)}/{len(sc)} re-labels agree", ""]
os.makedirs(OD, exist_ok=True)
open(f"{OD}/{NAME}_score.md", "w").write("\n".join(md) + "\n"); json.dump(rows, open(f"{OD}/{NAME}_score_rows.json", "w"), indent=1)
print("\n".join(md))
