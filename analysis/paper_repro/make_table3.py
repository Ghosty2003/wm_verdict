"""Table 3 (verdict reliability) of the VerdictBench draft, in two variants.

Inputs (read-only, see inputs.md):
  libero/out/cells/compare_lingbot_openwam_H400.json  per-unit verdict / replay / closed-loop rows, base + A + P (compare_models.py)
  libero/out/cells/compare_new_H400.json              same for E / G / N / O / B (compare_new.py; carries gate_pending for B)
  libero/out/closed_main/主表_长表_20261122.tsv        closed-loop main table, 50 deployment episodes per level

Per unit: verdict = the model's own imagined rollout accepted by the frozen per-task judge; real = the imagined plan
replayed open-loop in the simulator succeeds (H = 400); one unit = one verdict.
  AR     = accepted / N
  FRpl   = replay failures / N
  FSRpl  = (accepted and replay failure) / accepted
  MRpl   = (accepted and replay failure) / replay failures
  FRdep  = deployment failures / deployment episodes (main table)
  FSRdep = deployment episodes are not paired with verdict units, so the share of accepted verdicts whose deployment
           fails is only bounded (Frechet): with F = deployment failures scaled to N units and R = rejected verdicts,
           [max(0, F - R), min(F, accepted)] / accepted, pooled over the levels of the row. Printed as one number when
           the bounds coincide. (The open-loop records also carry a closed-loop run of the same unit; that paired value
           is given in the tsv as FSRdep_paired for reference.)
Condition groups:
  Nominal       = base (no shift)
  Vis. in-range = A1, P1, P2, O1-O5, and the B levels whose judge passed the trajectory-level corruption gate
                  (gate_pending False in compare_new_H400.json; B0 is the OpenWAM 128-px render control, not a shift)
  Vis. OOD      = A2-A4, P3, P4
  Hidden        = Em1-4, Ef1-4, G1-4, N1_1, NR1-5
  Sham          = A0, P0 (zero-displacement sham levels) and OpenWAM B0 (render control)   [113 variant only]
  (K: closed-loop only; NL / NLw: open-loop is reference only; F: closed-loop only -> not in Table 3)
Variants:
  106 = the four groups as printed in the draft (36 + 35 + 35 levels); "All" pools those 106 levels.
  113 = the 106 plus a separate Sham row per pair (2 + 2 + 3 levels); "All" pools all 113 levels (= the draft's All row).
Outputs (out/): table3.tsv, table3.tex (both variants), table3_106.tex, table3_113.tex, table3_wilson.tsv
(Appendix B.6, Wilson 95% intervals), table3_levels.tsv (per-level counts).
"""
import csv
import json

from repro_common import (CELLS_A, CELLS_B, PAIRS, load_json, load_main_table, out, pct, wilson, write_manifest)

OUTS = ["table3.tsv", "table3.tex", "table3_106.tex", "table3_113.tex", "table3_wilson.tsv", "table3_levels.tsv",
        ]
GROUPS = ["Nominal", "Vis. in-range", "Vis. OOD", "Hidden"]
SHAM = "Sham"
# B levels whose judge passed the trajectory gate, as recorded in perturbation_registry.md (2026-11-01 entry);
# used only as a consistency check of the gate_pending flags stored with the units.
B_GATE_PASS = {("lingbot", "t5"): {"Bg2", "Bg5", "Bg9", "Bf5"}, ("lingbot", "t3"): {"Bg2", "Bf2", "Bz2"},
               ("openwam", "t3"): {"Bg2", "Bf2", "Bf5"}}


def group_of(level, gate_pending):
    if level == "none":
        return "Nominal"
    if level in ("A0", "P0", "B0"):
        return SHAM
    if level in ("A1", "P1", "P2") or level.startswith("O"):
        return "Vis. in-range"
    if level.startswith("B"):
        return None if gate_pending else "Vis. in-range"
    if level in ("A2", "A3", "A4", "P3", "P4"):
        return "Vis. OOD"
    if level[:2] in ("Em", "Ef") or level.startswith("G") or level == "N1_1" or level.startswith("NR"):
        return "Hidden"
    return None


def units(a, b, m, t):
    U = []
    for k, v in a["units"].items():
        if k.startswith(f"{m}/{t}/"):
            U += v
    U += b["units"][f"{m}/{t}"]
    return U


def level_rows(outputs=OUTS):
    a = load_json(CELLS_A, outputs)
    b = load_json(CELLS_B, outputs)
    main = load_main_table(outputs)
    rows = []
    for M, T, m, t in PAIRS:
        by_level = {}
        for u in units(a, b, m, t):
            by_level.setdefault(u["level"], []).append(u)
        passed_b = set()
        for L, V in sorted(by_level.items()):
            g = group_of(L, any(u.get("gate_pending") for u in V))
            if g is None:
                continue
            if L.startswith("B") and L != "B0":
                passed_b.add(L)
            assert all(u["verdict"] is not None and u["real"] is not None for u in V), (m, t, L)
            n = len(V)
            acc = sum(bool(u["verdict"]) for u in V)
            fpl = sum(not u["real"] for u in V)
            afpl = sum(bool(u["verdict"]) and not u["real"] for u in V)
            D = [u for u in V if u["closed"] is not None]
            mr = main[(M, T, L)]
            nd, sd = int(mr["n"]), int(mr["n_success"])
            rows.append(dict(model=M, task=T, level=L, group=g, n=n, accepted=acc, replay_fail=fpl,
                             accepted_replay_fail=afpl, n_dep=nd, dep_fail=nd - sd,
                             n_paired=len(D), accepted_paired=sum(bool(u["verdict"]) for u in D),
                             accepted_dep_fail_paired=sum(bool(u["verdict"]) and not u["closed"] for u in D)))
        assert passed_b == B_GATE_PASS[(m, t)], (m, t, passed_b)
    return rows


def agg(R):
    s = lambda k: sum(r[k] for r in R)
    N, acc, fpl, afpl, nd, fd = s("n"), s("accepted"), s("replay_fail"), s("accepted_replay_fail"), s("n_dep"), s("dep_fail")
    F = round(fd * N / nd)                       # deployment failures scaled to the number of verdict units (N == nd here)
    lo, hi = max(0, F - (N - acc)), min(F, acc)
    fsr_lo, fsr_hi = pct(lo, acc), pct(hi, acc)
    return dict(levels=len(R), N=N, accepted=acc, AR=pct(acc, N), FRpl=pct(fpl, N), FSRpl=pct(afpl, acc),
                MRpl=pct(afpl, fpl), FRdep=pct(fd, nd),
                FSRdep=f"{fsr_lo:.1f}" if fsr_lo == fsr_hi else f"{fsr_lo:.1f}–{fsr_hi:.1f}",
                FSRdep_lo=fsr_lo, FSRdep_hi=fsr_hi,
                FSRdep_paired=pct(s("accepted_dep_fail_paired"), s("accepted_paired")),
                replay_fail=fpl, accepted_replay_fail=afpl, n_dep=nd, dep_fail=fd, fsr_lo_count=lo, fsr_hi_count=hi,
                level_list=",".join(r["level"] for r in R))


def build(rows, variant):
    groups = GROUPS + ([SHAM] if variant == "113" else [])
    table = []
    for M, T, _, _ in PAIRS:
        for g in groups:
            R = [r for r in rows if (r["model"], r["task"], r["group"]) == (M, T, g)]
            table.append(dict(variant=variant, model_task=f"{M} / {T}", condition=g, **agg(R)))
    R = [r for r in rows if r["group"] in groups]
    table.append(dict(variant=variant, model_task="All three pairs", condition="All levels", **agg(R)))
    return table


COLS = ["variant", "model_task", "condition", "levels", "N", "accepted", "AR", "FRpl", "FSRpl", "MRpl", "FRdep", "FSRdep",
        "FSRdep_lo", "FSRdep_hi", "FSRdep_paired", "replay_fail", "accepted_replay_fail", "n_dep", "dep_fail",
        "fsr_lo_count", "fsr_hi_count", "level_list"]


def tex(table, variant):
    f1 = lambda v: f"{v:.1f}"
    L = [f"% Table 3, variant {variant} levels -- generated by paper_repro/make_table3.py; do not edit by hand.",
         "\\begin{tabular}{llrrrrrrr}", "\\toprule",
         "Model / task & Condition & Levels & AR & FR$_{\\mathrm{pl}}$ & FSR$_{\\mathrm{pl}}$ & MR$_{\\mathrm{pl}}$ "
         "& FR$_{\\mathrm{dep}}$ & FSR$_{\\mathrm{dep}}$ \\\\", "\\midrule"]
    prev = None
    for r in table:
        name = r["model_task"].replace("OpenWAM-α", "OpenWAM-$\\alpha$")
        if prev is not None and r["model_task"] != prev:
            L.append("\\midrule")
        first = r["model_task"] != prev
        prev = r["model_task"]
        L.append(" & ".join([name if first else "", r["condition"], str(r["levels"]), f1(r["AR"]), f1(r["FRpl"]),
                             f1(r["FSRpl"]), f1(r["MRpl"]), f1(r["FRdep"]), r["FSRdep"].replace("–", "--")]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L) + "\n"


def wilson_rows(table):
    W = []
    for r in table:
        for metric, x, n in (("AR", r["accepted"], r["N"]), ("FRpl", r["replay_fail"], r["N"]),
                             ("FSRpl", r["accepted_replay_fail"], r["accepted"]),
                             ("MRpl", r["accepted_replay_fail"], r["replay_fail"]), ("FRdep", r["dep_fail"], r["n_dep"]),
                             ("FSRdep_lo", r["fsr_lo_count"], r["accepted"]), ("FSRdep_hi", r["fsr_hi_count"], r["accepted"])):
            lo, hi = wilson(x, n)
            W.append(dict(variant=r["variant"], model_task=r["model_task"], condition=r["condition"], metric=metric,
                          x=x, n=n, pct=pct(x, n), ci95_lo=round(100 * lo, 1) if n else "", ci95_hi=round(100 * hi, 1) if n else ""))
    return W


def write_tsv(path, rows, cols):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    rows = level_rows()
    t106, t113 = build(rows, "106"), build(rows, "113")
    assert t106[-1]["levels"] == 106 and t113[-1]["levels"] == 113, (t106[-1]["levels"], t113[-1]["levels"])
    write_tsv(out("table3.tsv"), t106 + t113, COLS)
    write_tsv(out("table3_wilson.tsv"), wilson_rows(t106 + t113),
              ["variant", "model_task", "condition", "metric", "x", "n", "pct", "ci95_lo", "ci95_hi"])
    write_tsv(out("table3_levels.tsv"), rows, list(rows[0].keys()))
    s106, s113 = tex(t106, "106"), tex(t113, "113")
    open(out("table3_106.tex"), "w", encoding="utf-8").write(s106)
    open(out("table3_113.tex"), "w", encoding="utf-8").write(s113)
    open(out("table3.tex"), "w", encoding="utf-8").write(
        "% Both variants of Table 3. Pick one; see paper_repro/make_table3.py for the definitions.\n\n" + s106 + "\n" + s113)
    for t in (t106, t113):
        a = t[-1]
        print(f"variant {a['variant']}: levels={a['levels']} verdicts={a['N']} accepted={a['accepted']} AR={a['AR']} "
              f"FRpl={a['FRpl']} FSRpl={a['FSRpl']} MRpl={a['MRpl']} FRdep={a['FRdep']} FSRdep={a['FSRdep']}")
    print("levels per pair:", json.dumps({f"{M}/{T}": {g: sum(1 for r in rows if (r['model'], r['task'], r['group']) == (M, T, g))
                                                      for g in GROUPS + [SHAM]} for M, T, _, _ in PAIRS}, ensure_ascii=False))
    write_manifest("make_table3")


if __name__ == "__main__":
    main()
