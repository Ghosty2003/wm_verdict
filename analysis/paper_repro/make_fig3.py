"""Figure 3 of the VerdictBench draft.

Fig. 3a  acceptance rate vs replayed-plan success per shift level, for the three judge-qualified pairs
         (LingBot-VA/T0, LingBot-VA/T1, OpenWAM-α/T1). Same per-level units as Table 3 (make_table3.level_rows):
         113 levels = the 106 grouped levels + 7 sham/control levels (A0, P0, OpenWAM B0; drawn hollow).
Fig. 3b  replayed-plan success vs closed-loop deployment success per level, LingBot-VA and OpenWAM-α on all five tasks.
         Plan success comes from the ORIGINAL 50-unit open-loop records in libero/out/runs/ (pre-2026-11-19; NOT the
         5-unit openloop5_20261119 reduced study, which gives rho = 0.37). Indexing follows the audit script
         s04_old_openloop_index.py: glob patterns in the order */*/truth*.jsonl, */truth*.jsonl, */records*.jsonl,
         */*/records*.jsonl (sorted within a pattern), skip */rerun_closed/* and */x86_rerun*, first record per unit_id wins
         (truth files first: the LingBot T2-T4 records.*.jsonl carry a stale `real` field; truth.*.jsonl holds the replay);
         success = real_first_success_step <= 400, else real_verdict; level = setting_id / level / 3rd uid field;
         records with perturbation.sham are sham levels and dropped.
         Deployment success = n_success / n of the main closed-loop table (主表_长表_20261122.tsv, 50 episodes/level).
         Excluded: latency (NL*, NLw*), pushes (K*), blackout (F*), and levels without a main-table row (sham level-0
         rows of Em/Ef/G/NR/N1). Spearman rho = Pearson on average ranks (ties averaged).
Outputs (out/): fig3a.pdf/png, fig3b.pdf/png, fig3_stats.json, fig3a_points.tsv, fig3b_points.tsv.
"""
import collections
import csv
import glob
import json
import os
import re

import plotlib
from make_table3 import level_rows
from repro_common import (MAIN_TABLE, MODEL_NAME, PAIRS, ROOT, RUNS, TASK_NAME, load_main_table, out, spearman, track, write_manifest)

H = 400
OUT3A = ["fig3a.pdf", "fig3a.png", "fig3_stats.json", "fig3a_points.tsv"]
OUT3B = ["fig3b.pdf", "fig3b.png", "fig3_stats.json", "fig3b_points.tsv"]
GLOBS = ["*/*/truth*.jsonl", "*/truth*.jsonl", "*/records*.jsonl", "*/*/records*.jsonl"]
COL = {"LingBot-VA": "#2a78d6", "OpenWAM-α": "#eb6834"}
PAIR_STYLE = {("LingBot-VA", "T0"): ("#2a78d6", "o"), ("LingBot-VA", "T1"): ("#1baf7a", "s"), ("OpenWAM-α", "T1"): ("#eb6834", "^")}
INK2, INK3 = "#52514e", "#8a8984"


# ---------------------------------------------------------------------------------------------------------------- 3b data
def original_openloop_index():
    """-> {(model, task, level): [success, n, set(files)]}, list of scanned files (relative to RUNS)."""
    seen = {}
    files = []
    for pat in GLOBS:
        for f in sorted(glob.glob(os.path.join(RUNS, pat))):
            if "/rerun_closed/" in f or "/x86_rerun" in f:
                continue
            rel = os.path.relpath(f, RUNS)
            files.append(rel)
            track(f, OUT3B)
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    uid = r.get("unit_id")
                    if not uid or uid in seen:
                        continue
                    rs = r.get("real_first_success_step", "MISSING")
                    if rs == "MISSING":
                        if r.get("real_verdict") is None:
                            continue
                        real = bool(r["real_verdict"])
                    else:
                        real = rs is not None and int(rs) <= H
                    m = re.match(r"(lingbot|openwam)_libero10_(t\d)_", uid)
                    if not m:
                        continue
                    lv = r.get("setting_id") or r.get("level") or (uid.split("__")[2] if uid.count("__") >= 2 else "none")
                    sham = isinstance(r.get("perturbation"), dict) and bool(r["perturbation"].get("sham"))
                    seen[uid] = (MODEL_NAME[m.group(1)], TASK_NAME.get(m.group(2), m.group(2)), lv, sham, real, rel)
    idx = {}
    for uid, (M, T, lv, sham, real, rel) in seen.items():
        if sham:
            continue
        a = idx.setdefault((M, T, lv), [0, 0, collections.Counter()])
        a[0] += real
        a[1] += 1
        a[2][rel] += 1
    return idx, files, len(seen)


def excluded(level):
    return level.startswith(("NL", "K", "F"))


def fig3b_points():
    idx, files, n_units = original_openloop_index()
    main = load_main_table(OUT3B)
    pts, no_main = [], []
    for (M, T, L), (s, n, src) in sorted(idx.items()):
        if excluded(L):
            continue
        if (M, T, L) not in main:
            no_main.append(f"{M}/{T}/{L}")
            continue
        r = main[(M, T, L)]
        pts.append(dict(model=M, task=T, level=L, family=r["family"], plan_success=s, plan_n=n, plan=s / n,
                        dep_success=int(r["n_success"]), dep_n=int(r["n"]), dep=int(r["n_success"]) / int(r["n"]),
                        source_files=";".join(sorted(src))))
    return pts, files, n_units, no_main


# ---------------------------------------------------------------------------------------------------------------- plots
def plot_3a(rows):
    fig = plotlib.figure(3.4, 3.0)
    ax = fig.axes((0.16, 0.15, 0.80, 0.70), (-0.03, 1.03), (-0.03, 1.03), "Replayed-plan success (per level)",
                  "Verdict acceptance rate", "", [0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.line([0, 1], [0, 1], INK3, 0.8, dash=True, z=1)
    ax.text(0.62, 0.55, "calibrated verdict", 6.5, INK3, "left", "center")
    for (M, T), (c, mk) in PAIR_STYLE.items():
        R = [r for r in rows if (r["model"], r["task"]) == (M, T)]
        G = [r for r in R if r["group"] != "Sham"]
        S = [r for r in R if r["group"] == "Sham"]
        ax.scatter([1 - r["replay_fail"] / r["n"] for r in G], [r["accepted"] / r["n"] for r in G], c, 4.5, mk, True, 0.8)
        ax.scatter([1 - r["replay_fail"] / r["n"] for r in S], [r["accepted"] / r["n"] for r in S], c, 4.5, mk, False, 1.0)
    y = 0.955
    for i, ((M, T), (c, mk)) in enumerate(PAIR_STYLE.items()):
        x = 0.05 + 0.31 * i
        fig.axes((x, y - 0.02, 0.02, 0.04), (0, 1), (0, 1), frame=False).scatter([0.5], [0.5], c, 4.5, mk, True)
        fig.text(x + 0.03, y, f"{M} / {T}", 6.5, "#0b0b0b")
    fig.text(0.05, 0.895, "hollow = sham / control level (A0, P0, B0)", 6, INK2)
    fig.save(out("fig3a.pdf"), out("fig3a.png"))


def plot_3b(pts, hits, rho):
    fig = plotlib.figure(3.4, 3.0)
    ax = fig.axes((0.16, 0.15, 0.80, 0.70), (-0.03, 1.03), (-0.03, 1.03), "Replayed-plan success (per level)",
                  "Deployment success (per level)", "", [0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.rect(0.70, -0.03, 1.03, 0.30, "#fbe9e9", "#e34948", 0.6, dash=True, z=0)
    ax.line([0, 1], [0, 1], INK3, 0.8, dash=True, z=1)
    for M in ("LingBot-VA", "OpenWAM-α"):
        P = [p for p in pts if p["model"] == M]
        ax.scatter([p["plan"] for p in P], [p["dep"] for p in P], COL[M], 3.6, "o", True, 0.55, edgewidth=0.3)
    ax.scatter([p["plan"] for p in hits], [p["dep"] for p in hits], "#e34948", 5.5, "o", False, 1.0)
    ax.text(0.68, 0.15, f"{len(hits)} levels (blur / fog):", 6, "#0b0b0b", "right", "center")
    ax.text(0.68, 0.09, "plan ≥ 70%, deploy ≤ 30%", 6, "#0b0b0b", "right", "center")
    n_above = sum(p["dep"] > p["plan"] for p in pts)
    fig.text(0.16, 0.895, f"Spearman ρ = {rho:.2f} · {len(pts)} levels · {n_above} above the diagonal", 6.5, INK2)
    for i, M in enumerate(("LingBot-VA", "OpenWAM-α")):
        x = 0.16 + 0.34 * i
        fig.axes((x, 0.935, 0.02, 0.04), (0, 1), (0, 1), frame=False).scatter([0.5], [0.5], COL[M], 4.5, "o", True)
        fig.text(x + 0.03, 0.955, M, 6.5, "#0b0b0b")
    fig.save(out("fig3b.pdf"), out("fig3b.png"))


def write_tsv(name, rows):
    with open(out(name), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    # ---- 3a
    rows = level_rows(OUT3A)
    for r in rows:
        r["plan"] = round(1 - r["replay_fail"] / r["n"], 4)
        r["acceptance"] = round(r["accepted"] / r["n"], 4)
    write_tsv("fig3a_points.tsv", [{k: r[k] for k in ("model", "task", "level", "group", "n", "accepted", "replay_fail",
                                                       "plan", "acceptance")} for r in rows])
    core = [r for r in rows if r["group"] != "Sham"]
    s3a = {}
    for name, R in (("113_levels", rows), ("106_levels", core)):
        s3a[name] = dict(levels=len(R), plan_success_min=min(r["plan"] for r in R), plan_success_max=max(r["plan"] for r in R),
                         acceptance_min=min(r["acceptance"] for r in R), acceptance_max=max(r["acceptance"] for r in R),
                         levels_acceptance_1=sum(r["acceptance"] == 1 for r in R),
                         argmax_plan=[f"{r['model']}/{r['task']}/{r['level']}" for r in R if r["plan"] == max(x["plan"] for x in R)],
                         argmin_acceptance=[f"{r['model']}/{r['task']}/{r['level']}" for r in R
                                            if r["acceptance"] == min(x["acceptance"] for x in R)])
    s3a["per_pair"] = {f"{M}/{T}": dict(levels=sum(1 for r in rows if (r["model"], r["task"]) == (M, T)))
                       for M, T, _, _ in PAIRS}
    plot_3a(rows)

    # ---- 3b
    pts, files, n_units, no_main = fig3b_points()
    x, y = [p["plan"] for p in pts], [p["dep"] for p in pts]
    rho = spearman(x, y)
    hits = [p for p in pts if p["plan"] >= 0.70 and p["dep"] <= 0.30]
    used = collections.Counter()
    for p in pts:
        for f in p["source_files"].split(";"):
            used[f] += 1
    write_tsv("fig3b_points.tsv", pts)
    s3b = dict(
        data_source=dict(plan="ORIGINAL 50-unit open-loop records under libero/out/runs/ (see make_fig3.py docstring for the "
                              "exact indexing rule); the 2026-11-19 reduced 5-unit study is NOT used",
                         deployment=os.path.relpath(MAIN_TABLE, ROOT),
                         files_scanned=len(files), units_indexed=n_units,
                         files_contributing_to_plotted_levels={f: used[f] for f in sorted(used)},
                         files_scanned_list=files),
        levels=len(pts), per_model_task={f"{k[0]}/{k[1]}": v for k, v in sorted(collections.Counter((p["model"], p["task"]) for p in pts).items())},
        units_per_level=sorted({p["plan_n"] for p in pts}), episodes_per_level=sorted({p["dep_n"] for p in pts}),
        excluded_rule="levels NL*/NLw* (latency), K* (pushes), F* (blackout); open-loop levels without a main-table row",
        open_loop_levels_without_main_row=no_main,
        spearman_rho=round(rho, 4), spearman_rho_2dp=f"{rho:.2f}",
        deployment_gt_plan=sum(b > a for a, b in zip(x, y)), deployment_eq_plan=sum(b == a for a, b in zip(x, y)),
        deployment_lt_plan=sum(b < a for a, b in zip(x, y)),
        plan_ge70_dep_le30_count=len(hits),
        plan_ge70_dep_le30=[dict(model=p["model"], task=p["task"], level=p["level"], plan=f"{p['plan_success']}/{p['plan_n']}",
                                 deployment=f"{p['dep_success']}/{p['dep_n']}") for p in hits],
        plan_ge70_dep_le30_all_blur_or_fog=all(p["level"].startswith(("Bg", "Bf")) for p in hits))
    plot_3b(pts, hits, rho)

    stats = dict(backend=plotlib.backend_note(), fig3a=s3a, fig3b=s3b)
    with open(out("fig3_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print("fig3a:", json.dumps(s3a["113_levels"], ensure_ascii=False))
    print(f"fig3b: levels={len(pts)} rho={rho:.4f} dep>plan={s3b['deployment_gt_plan']} hits={len(hits)} "
          f"{[(h['model'], h['task'], h['level']) for h in hits]} backend={plotlib.backend_note()}")
    write_manifest("make_fig3")


if __name__ == "__main__":
    main()
