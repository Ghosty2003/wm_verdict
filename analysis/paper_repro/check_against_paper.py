"""Compare every number produced by make_table3.py / make_fig3.py / make_fig4.py with the draft's values as transcribed
in paper_claims.md (libero/out/paper_audit_20261123/paper_claims.md, or $PAPER_CLAIMS). Prints MATCH / DIFF per cell and
writes out/check_log.txt (same text) and out/check_results.tsv. Exit code 0 even with DIFFs; the summary says which
variant matches.
"""
import csv
import json
import os
import re
import sys

from repro_common import LIBERO, out, track, write_manifest

CLAIMS = os.environ.get("PAPER_CLAIMS", os.path.join(LIBERO, "out", "paper_audit_20261123", "paper_claims.md"))
OUTS = ["check_log.txt", "check_results.tsv"]
COLS = ["Levels", "AR", "FRpl", "FSRpl", "MRpl", "FRdep", "FSRdep"]
LOG, RES = [], []


def say(s=""):
    print(s)
    LOG.append(s)


def rec(section, item, paper, ours, note=""):
    ok = str(paper) == str(ours)
    RES.append(dict(section=section, item=item, paper=paper, ours=ours, status="MATCH" if ok else "DIFF", note=note))
    say(f"{'MATCH' if ok else 'DIFF '}  {section:14s} {item:58s} paper={paper!s:12s} ours={ours!s:12s} {note}")
    return ok


def parse_claims():
    text = open(track(CLAIMS, OUTS), encoding="utf-8").read()
    claims = {}
    for line in text.splitlines():
        m = re.match(r"- (C\d\d) (.*)", line)
        if m:
            claims[m.group(1)] = m.group(2)
    t3, cur = [], None
    sec = text.split("## Table 3", 1)[1].split("\n## ", 1)[0]
    for line in sec.splitlines():
        if not line.startswith("|") or line.startswith("|---") or line.startswith("| Model"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        cur = c[0] or cur
        t3.append(dict(model_task=cur, condition=c[1], **dict(zip(COLS, c[2:]))))
    return claims, t3


def num(s):
    return float(s.replace(",", ""))


def main():
    claims, t3 = parse_claims()
    ours = list(csv.DictReader(open(out("table3.tsv"), encoding="utf-8"), delimiter="\t"))
    levels = list(csv.DictReader(open(out("table3_levels.tsv"), encoding="utf-8"), delimiter="\t"))
    f3 = json.load(open(out("fig3_stats.json"), encoding="utf-8"))
    f4 = json.load(open(out("fig4_counts.json"), encoding="utf-8"))
    say(f"paper values: {CLAIMS}")
    say(f"Table 3 cells in paper_claims.md: {len(t3)} rows x {len(COLS)} = {len(t3) * len(COLS)}")

    # ---------------------------------------------------------------- Table 3, both variants
    summary = {}
    for variant in ("113", "106"):
        say(f"\n== Table 3, variant {variant} levels ==")
        O = {(r["model_task"], r["condition"]): r for r in ours if r["variant"] == variant}
        n_ok = n = 0
        for p in t3:
            o = O[(p["model_task"], p["condition"])]
            for c in COLS:
                key = {"Levels": "levels"}.get(c, c)
                n += 1
                note = ""
                if variant == "106" and p["model_task"] == "All three pairs":
                    note = "(106 variant pools only the grouped levels; the draft's All row is the 113-level pool)"
                n_ok += rec(f"T3/{variant}", f"{p['model_task']} | {p['condition']} | {c}", p[c], o[key], note)
        summary[variant] = (n_ok, n)
        say(f"-- variant {variant}: {n_ok}/{n} MATCH")

    # ---------------------------------------------------------------- text claims derived from Table 3
    say("\n== Text claims built on Table 3 (113-level pool) ==")
    A = {(r["variant"], r["model_task"], r["condition"]): r for r in ours}
    all113 = A[("113", "All three pairs", "All levels")]
    m = re.search(r"across ([\d,]+) verdicts, ([\d.]+)% of runs are accepted\" \(= ([\d,]+) of", claims["C03"])
    rec("C03", "verdicts", int(num(m.group(1))), all113["N"])
    rec("C03", "acceptance %", m.group(2), all113["AR"])
    rec("C03", "accepted", int(num(m.group(3))), all113["accepted"])
    rec("C04", "accepted plans failing on replay %", re.search(r"\"([\d.]+)% of accepted", claims["C04"]).group(1), all113["FSRpl"])
    m = re.search(r"\"(\d+) verdicts per shift level over (\d+) levels\"", claims["C06"])
    per_level = sorted({int(r["n"]) for r in levels})
    rec("C06", "verdicts per level", m.group(1), ",".join(map(str, per_level)))
    rec("C06", "levels", m.group(2), all113["levels"])
    nom = lambda M, T: A[("113", f"{M} / {T}", "Nominal")]
    m = re.search(r"accepts (\d+)/(\d+), (\d+) of (\d+) plans fail on replay, FSRpl = ([\d.]+)%", claims["C16"])
    o = nom("OpenWAM-α", "T1")
    rec("C16", "OpenWAM-α/T1 nominal accepted", f"{m.group(1)}/{m.group(2)}", f"{o['accepted']}/{o['N']}")
    rec("C16", "OpenWAM-α/T1 nominal replay failures", m.group(3), o["replay_fail"])
    rec("C16", "OpenWAM-α/T1 nominal FSRpl", m.group(5), o["FSRpl"])
    m = re.search(r"accepts (\d+)/(\d+) on T0 and T1; (\d+) and (\d+) plans fail \(FSRpl ([\d.]+)%, ([\d.]+)%\)", claims["C17"])
    for T, fails, fsr in (("T0", m.group(3), m.group(5)), ("T1", m.group(4), m.group(6))):
        o = nom("LingBot-VA", T)
        rec("C17", f"LingBot-VA/{T} nominal accepted", f"{m.group(1)}/{m.group(2)}", f"{o['accepted']}/{o['N']}")
        rec("C17", f"LingBot-VA/{T} nominal replay failures", fails, o["replay_fail"])
        rec("C17", f"LingBot-VA/{T} nominal FSRpl", fsr, o["FSRpl"])
    G = [r for r in ours if r["variant"] == "106" and r["model_task"] != "All three pairs"]
    m = re.search(r"within ([\d.]+) pp of FRpl; MRpl ([\d.]+)–([\d.]+)%", claims["C18"])
    rec("C18", "max |FSRpl - FRpl| over groups (pp)", m.group(1), f"{max(abs(float(r['FSRpl']) - float(r['FRpl'])) for r in G):.1f}")
    rec("C18", "MRpl range over groups", f"{m.group(2)}–{m.group(3)}",
        f"{min(float(r['MRpl']) for r in G):.1f}–{max(float(r['MRpl']) for r in G):.1f}")
    rng = lambda cond, k: f"{min(float(r[k]) for r in G if r['condition'] == cond):.1f}–{max(float(r[k]) for r in G if r['condition'] == cond):.1f}"
    m = re.search(r"FSR from ([\d.]+)–([\d.]+)% nominal to ([\d.]+)–([\d.]+)% over hidden", claims["C19"])
    rec("C19", "nominal FSRpl range", f"{m.group(1)}–{m.group(2)}", rng("Nominal", "FSRpl"))
    rec("C19", "hidden FSRpl range", f"{m.group(3)}–{m.group(4)}", rng("Hidden", "FSRpl"))
    m = re.search(r"accepted at ([\d.]+)–([\d.]+)%, FSR ([\d.]+)–([\d.]+)%", claims["C20"])
    rec("C20", "Vis. OOD AR range", f"{m.group(1)}–{m.group(2)}", rng("Vis. OOD", "AR"))
    rec("C20", "Vis. OOD FSRpl range", f"{m.group(3)}–{m.group(4)}", rng("Vis. OOD", "FSRpl"))
    m = re.search(r"succeeds on (\d+), (\d+), (\d+) of (\d+); verdict accepts all (\d+)", claims["C21"])
    L = {r["level"]: r for r in levels if (r["model"], r["task"]) == ("LingBot-VA", "T0")}
    for lv, paper in zip(("A2", "A3", "A4"), m.group(1, 2, 3)):
        rec("C21", f"LingBot-VA/T0 {lv} replay successes", paper, int(L[lv]["n"]) - int(L[lv]["replay_fail"]))
        rec("C21", f"LingBot-VA/T0 {lv} accepted", m.group(5), L[lv]["accepted"])

    # ---------------------------------------------------------------- Fig 3
    say("\n== Figure 3 ==")
    b = f3["fig3b"]
    rec("C22 (Fig 3b)", "Spearman rho (2 dp)", re.search(r"ρ = ([\d.]+)", claims["C22"]).group(1), b["spearman_rho_2dp"],
        f"(rho = {b['spearman_rho']})")
    m = re.search(r"(\d+) levels on all five tasks.*deployment exceeds plan on (\d+) levels", claims["C23"])
    rec("C23 (Fig 3b)", "levels", m.group(1), b["levels"])
    rec("C23 (Fig 3b)", "deployment > plan", m.group(2), b["deployment_gt_plan"])
    m = re.search(r"(\d+) levels \(all Gaussian blur or fog\) where plan succeeds on ≥(\d+)% of starts and deployment on ≤(\d+)%",
                  claims["C24"])
    rec("C24 (Fig 3b)", f"levels with plan >= {m.group(2)}% and deployment <= {m.group(3)}%", m.group(1), b["plan_ge70_dep_le30_count"],
        "[" + ", ".join(f"{h['model']}/{h['task']}/{h['level']}" for h in b["plan_ge70_dep_le30"]) + "]")
    rec("C24 (Fig 3b)", "all of them Gaussian blur (Bg) or fog (Bf)", True, b["plan_ge70_dep_le30_all_blur_or_fog"])
    m = re.search(r"plan success ranges (\d+) to ([\d.]+)", claims["C29"])
    for v in ("113_levels", "106_levels"):
        a = f3["fig3a"][v]
        rec("C29 (Fig 3a)", f"plan success range, {v.replace('_', ' ')}", f"{num(m.group(1)):g}–{num(m.group(2)):g}",
            f"{a['plan_success_min']:g}–{a['plan_success_max']:g}",
            f"(acceptance {a['acceptance_min']:g}–{a['acceptance_max']:g}; {a['levels_acceptance_1']}/{a['levels']} levels at 1.0)")

    # ---------------------------------------------------------------- Fig 4
    say("\n== Figure 4 ==")
    words = {"twelve": 12}
    rec("C01 (Fig 4)", "shift families", words[re.search(r"\"(\w+) families", claims["C01"]).group(1)], f4["families"])
    m = re.search(r"(\w+) shift families with (\d+) levels", claims["C07"])
    rec("C07 (Fig 4)", "shift families", words[m.group(1)], f4["families"])
    lv = f4["levels"]
    rec("C07 (Fig 4)", "levels = registry non-sham levels (main-table 57 + NLw 5)", m.group(2), lv["registry_nonsham"],
        f"[{f4['derivation_of_62']}]")
    rec("C07 (Fig 4)", "levels = main-table non-baseline levels (for reference)", m.group(2), lv["main_nonbase"],
        "(main table 2026-11-22 incl. A0/P0/B0; NLw not in the main table)")
    m = re.search(r"Visible (\d+) families, Hidden (\d+), After Verdict (\d+)", claims["C08"])
    for k, g in (("Visible", 1), ("Hidden", 2), ("After verdict", 3)):
        rec("C08 (Fig 4)", f"{k} families", m.group(g), f4["visibility"][k], f"({', '.join(f4['visibility_families'][k])})")

    # ---------------------------------------------------------------- summary
    say("\n== Summary ==")
    for v, (k, n) in summary.items():
        say(f"Table 3 variant {v}: {k}/{n} MATCH")
    other = [r for r in RES if not r["section"].startswith("T3/")]
    say(f"Other claims: {sum(r['status'] == 'MATCH' for r in other)}/{len(other)} MATCH")
    for r in other:
        if r["status"] == "DIFF":
            say(f"  DIFF {r['section']} {r['item']}: paper {r['paper']} vs ours {r['ours']} {r['note']}")
    open(out("check_log.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    with open(out("check_results.tsv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(RES[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(RES)
    write_manifest("check_against_paper")
    return 0 if summary["113"][0] == summary["113"][1] else 1


if __name__ == "__main__":
    sys.exit(main())
