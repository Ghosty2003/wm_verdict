#!/usr/bin/env python3
"""Human evaluation from the annotation-site export (newest export_2026* folder).

Cards come in pairs per run (same unit_id):
  REAL card     = the real final frame       -> rater accuracy against the simulator outcome
  IMAGINED card = the imagined final frame   -> human verdict; optimism = real failure judged looks_success
Main = looks_success counts as success; wide = + borderline; uncertain excluded from denominators.
Also: inter-rater agreement on doubly-labelled questions, and human vs qwen4-max / Opus 5.5 on the same run.
Writes human_eval.md and human_eval_tables.json next to this script."""
import json, csv, collections, math, glob, os, re
OUT = os.path.join(os.environ.get("WMV_ROOT", "$WMV_ROOT"), "libero", "out"); H = os.path.dirname(os.path.abspath(__file__))
EXP = sorted(glob.glob(f"{OUT}/annotation/export_2026*"))[-1]
d = json.load(open(f"{EXP}/samecase_annotations.json")); Q = d["samecase"]
inv = {}
for l in open(f"{OUT}/annotation/samecase_20261115/inventory.jsonl"):
    r = json.loads(l); inv.setdefault(r["unit_id"], []).append(r)
def truth(uid):
    vals = {r.get("success") for r in inv.get(uid, []) if r.get("success") is not None}
    return vals.pop() if len(vals) == 1 else None           # None if missing or conflicting records
POS, WIDE, UNC = {"looks_success"}, {"looks_success", "borderline"}, {"uncertain"}
def wil(k, n, z=1.96):
    if not n: return float("nan"), float("nan")
    p = k/n; dd = 1+z*z/n; c = (p+z*z/(2*n))/dd; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/dd; return c-h, c+h
def pct(k, n):
    if not n: return "—"
    lo, hi = wil(k, n); return f"{100*k/n:.1f}% ({k}/{n}) [{100*lo:.0f}–{100*hi:.0f}]"
def kappa(a, b):
    n = len(a)
    if not n: return float("nan")
    po = sum(x == y for x, y in zip(a, b))/n; pa = sum(a)/n; pb = sum(b)/n; pe = pa*pb+(1-pa)*(1-pb)
    return (po-pe)/(1-pe) if pe < 1 else float("nan")
rows = []                                                  # one row per (question, label)
for x in Q:
    for a in x["annotations"]:
        rows.append(dict(kind=x["kind"], model=x["model"], T=x["T"], level=x["level"], uid=x["unit_id"], card=x["card_id"],
                         rater=a.get("annotator") or "?", label=a["label"], truth=truth(x["unit_id"])))
rater_ids = {r: f"R{i+1}" for i, (r, _) in enumerate(collections.Counter(r["rater"] for r in rows).most_common())}
T = {}
md = [f"# Human evaluation (annotation site export {d['exported_at']})", "",
      f"- Source: `{os.path.relpath(EXP, OUT)}`; questions answered {sum(1 for x in Q if x['annotations'])} of {len(Q)}; "
      f"labels {len(rows)}; annotators {len(rater_ids)} (labels each: {sorted(collections.Counter(r['rater'] for r in rows).values(), reverse=True)})",
      f"- Label mix: {dict(collections.Counter(r['label'] for r in rows))}",
      f"- Rows without a unique simulator outcome (excluded from accuracy/optimism): {sum(r['truth'] is None for r in rows)}", ""]
def block(sel, title, key):
    out = [f"## {title}", "", "| group | TPR main (real success → success) | TPR wide | judged success on real failure (main) | wide | uncertain |", "|---|---|---|---|---|---|"]
    tab = []
    groups = [("ALL", lambda r: True)] + [(t, (lambda t: lambda r: r["T"] == t)(t)) for t in ("T0", "T1", "T2", "T3", "T4")] + \
             [(m, (lambda m: lambda r: r["model"] == m)(m)) for m in ("lingbot", "openwam", "cosmos", "uva")]
    for g, f in groups:
        R = [r for r in sel if f(r) and r["truth"] is not None]
        s = [r for r in R if r["truth"] and r["label"] not in UNC]; fl = [r for r in R if not r["truth"] and r["label"] not in UNC]
        u = sum(r["label"] in UNC for r in R)
        c = lambda L, P: sum(r["label"] in P for r in L)
        out.append(f"| {g} | {pct(c(s, POS), len(s))} | {pct(c(s, WIDE), len(s))} | {pct(c(fl, POS), len(fl))} | {pct(c(fl, WIDE), len(fl))} | {u} |")
        tab.append(dict(group=g, n_s=len(s), s_main=c(s, POS), s_wide=c(s, WIDE), n_f=len(fl), f_main=c(fl, POS), f_wide=c(fl, WIDE), unc=u))
    T[key] = tab; return out + [""]
md += block([r for r in rows if r["kind"] == "real"], "REAL final frames: rater accuracy against the simulator", "real")
md += block([r for r in rows if r["kind"] == "imag"], "IMAGINED final frames: human verdicts (the last column pair = human-judged optimism)", "imag")
# inter-rater
byq = collections.defaultdict(list)
for r in rows: byq[(r["card"])].append(r)
pairs = [(v[0], v[1]) for v in byq.values() if len({x["rater"] for x in v}) >= 2]
a4 = sum(p["label"] == q["label"] for p, q in pairs)
bm = [(p["label"] in POS, q["label"] in POS) for p, q in pairs if p["label"] not in UNC and q["label"] not in UNC]
bw = [(p["label"] in WIDE, q["label"] in WIDE) for p, q in pairs if p["label"] not in UNC and q["label"] not in UNC]
md += ["## Inter-rater agreement (questions labelled by two annotators)", "",
       f"- pairs: {len(pairs)} (real {sum(p['kind']=='real' for p,_ in pairs)}, imagined {sum(p['kind']=='imag' for p,_ in pairs)})",
       f"- exact 4-way agreement: {pct(a4, len(pairs))}",
       f"- binary main: agreement {pct(sum(x==y for x,y in bm), len(bm))}, kappa {kappa([x for x,_ in bm],[y for _,y in bm]):.2f}",
       f"- binary wide: agreement {pct(sum(x==y for x,y in bw), len(bw))}, kappa {kappa([x for x,_ in bw],[y for _,y in bw]):.2f}", ""]
T["interrater"] = dict(pairs=len(pairs), exact4=a4, bin_main=[sum(x==y for x,y in bm), len(bm)], bin_wide=[sum(x==y for x,y in bw), len(bw)])
# human vs judges on the imagined frame of the same run (match by unit_id, with and without the machine suffix)
jk = {}
for r in csv.DictReader(open(f"{OUT}/judge_package_20261117/_key/key.csv")): jk.setdefault(r["unit_id"], []).append(("pkg20261117", r["frame_id"]))
for r in csv.DictReader(open(f"{OUT}/judge_gap_20261124/key/key_gap.csv")): jk.setdefault(r["unit_id"], []).append(("gap20261124", r["frame_id"]))
for r in csv.DictReader(open(f"{OUT}/judge_package_20261117/uva_ext/key_uva.csv")): jk.setdefault(r["unit_id"], []).append(("uva_ext", r["frame_id"]))
qw = {}
for f, src in ((f"{OUT}/judge_package_20261117/vlm_openrouter/labels_qwen4max.csv", "pkg20261117"),
               (f"{OUT}/judge_gap_20261124/vlm_openrouter/labels_gap_qwen4max.csv", "gap20261124"),
               (f"{OUT}/judge_package_20261117/vlm_openrouter/labels_uva_qwen4max.csv", "uva_ext")):
    for r in csv.DictReader(open(f)): qw[(src, r["frame_id"])] = r["label"]
newid = {(r["src_package"], r["src_frame_id"]): r["new_id"] for r in csv.DictReader(open(f"{OUT}/judge_package_all_20261124/key/key_all.csv"))}
op = {}
for f in glob.glob(f"{OUT}/agent_judge_opus55_20261124/raw/labels_opus55_self_T*.csv"):
    for r in csv.DictReader(open(f)): op[r["frame_id"]] = r["label"]
def judge_frame(uid):
    c = jk.get(uid) or jk.get(re.sub(r"__(h100|x86|cl5090|cluster)$", "", uid))
    return c[0] if c and len({x for x in c}) == 1 else (c[0] if c else None)
md += ["## Human vs automated judges on the imagined frame of the same run", "",
       "Matched by unit_id. The site card and the judge-package frame are both the last imagined frame of that run.", "",
       "| judge | matched pairs | exact 4-way | binary main agree / kappa | binary wide agree / kappa |", "|---|---|---|---|---|"]
T["vs_judges"] = {}
GP = os.environ.get("WMV_SECOND_AGENT_LABELS", f"{OUT}/agent_judge_gpt61sol_20261124/labels.csv")   # second agent judge (labels CSV)
gp = {r["frame_id"]: r["label"] for r in csv.DictReader(open(GP))} if os.path.exists(GP) else {}
for name, get in (("qwen4-max (4-way)", lambda fr: qw.get(fr)), ("Opus 5.5 agent (4-way)", lambda fr: op.get(newid.get(fr))),
                  ("GPT 6.1 Sol agent (4-way)", lambda fr: gp.get(newid.get(fr)))):
    P = []
    for r in rows:
        if r["kind"] != "imag": continue
        fr = judge_frame(r["uid"])
        if not fr: continue
        j = get(fr)
        if j: P.append((r["label"], j))
    bm = [(h in POS, j in POS) for h, j in P if h not in UNC and j not in UNC]; bw = [(h in WIDE, j in WIDE) for h, j in P if h not in UNC and j not in UNC]
    md.append(f"| {name} | {len(P)} | {pct(sum(h==j for h,j in P), len(P))} | {pct(sum(x==y for x,y in bm), len(bm))} / {kappa([x for x,_ in bm],[y for _,y in bm]):.2f} | "
              f"{pct(sum(x==y for x,y in bw), len(bw))} / {kappa([x for x,_ in bw],[y for _,y in bw]):.2f} |")
    T["vs_judges"][name] = dict(n=len(P), exact=sum(h == j for h, j in P))
md.append("")
open(f"{H}/human_eval.md", "w").write("\n".join(md) + "\n"); json.dump(T, open(f"{H}/human_eval_tables.json", "w"), indent=1)
print("\n".join(md))
