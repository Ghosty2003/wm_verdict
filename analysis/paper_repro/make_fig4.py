"""Figure 4 of the VerdictBench draft: shift families, their levels, and when each is visible to the verdict.

Inputs (read-only):
  libero/perturbation_registry.md                    family rows (category, model input, timing) + change log (NLw)
  libero/out/closed_main/主表_长表_20261122.tsv        which levels exist in the closed-loop main table (any model)

Family attributes are parsed from the registry table; the per-level parameters below are transcribed from the same
registry rows (level definitions are free text there). The script asserts that every family row it relies on is
present and that its timing text still starts as transcribed, so a registry edit shows up as an error.

Visibility rule (applied to the registry text):
  After verdict = the shift is a single event at t > 0 on the executed trajectory only (K: "t > 0 单次"; F: "仅闭环轨")
  Visible       = otherwise, the shifted input enters the model input ("进")
  Hidden        = otherwise ("不进")
Level counting (the draft says "62 levels"; the script derives every variant and prints the breakdown):
  main_nonbase          levels in the main table other than the nominal row
  main_nonsham          ... minus sham / control levels (A0, P0 = zero-dose shams; B0 = OpenWAM 128-px render control)
  registry_nonsham      non-sham levels defined in the registry = main_nonsham + NLw1-5 (registered 2026-10-28,
                        run, but not in the 2026-11-22 main table)
Outputs (out/): fig4.pdf/png, fig4_levels.tsv, fig4_counts.json
"""
import collections
import csv
import json
import re

import plotlib
from repro_common import REGISTRY, load_main_table, out, track, write_manifest

OUTS = ["fig4.pdf", "fig4.png", "fig4_levels.tsv", "fig4_counts.json"]

# family -> (registry row code, English name, expected start of the registry timing text, English timing)
FAMILIES = {
    "A": ("A", "Object displacement", "t = 0", "t = 0 (first frame)"),
    "P": ("P", "Initial robot pose", "t = 0", "t = 0 (first frame)"),
    "O": ("O", "Occlusion patch", "想象模型 t = 0 首帧", "t = 0 first frame (policies: t = 50)"),
    "B": ("B", "Image corruption", "渲染 128×128", "every real frame from t = 0"),
    "Em": ("E", "Object mass", "全程", "whole episode"),
    "Ef": ("E", "Object friction", "全程", "whole episode"),
    "G": ("G", "Actuator gain", "全程", "whole episode"),
    "N1": ("N1", "Single action drop", "t = 50", "single step, t = 50"),
    "NR": ("NR", "Random action drop", "全程", "whole episode"),
    "NL": ("NL", "Action latency", "全程", "whole episode"),
    "K": ("K", "Mid-episode push", "t > 0 单次", "single push, t ∈ {60, 160, 260}"),
    "F": ("F", "Full-frame blackout", "仅闭环轨", "first replanning after t ∈ {60, 160, 260}"),
}
ORDER = ["A", "P", "O", "B", "Em", "Ef", "G", "N1", "NR", "NL", "K", "F"]
# level -> short parameter (transcribed from the registry rows; K order from the closed-grid legend: K1-3 @60, K4-6 @160, K7-9 @260)
PARAM = {
    "A0": "0 cm (sham)", "A1": "boundary", "A2": "+2 cm", "A3": "+5 cm", "A4": "+10 cm",
    "P0": "0 rad (sham)", "P1": "0.02 rad", "P2": "0.06 rad", "P3": "0.12 rad", "P4": "0.24 rad",
    "O1": "8 px", "O2": "16 px", "O3": "24 px", "O4": "32 px", "O5": "40 px",
    "B0": "no corr.", "Bg2": "sev. 2", "Bg5": "sev. 5", "Bg9": "sev. 9", "Bf2": "sev. 2", "Bf5": "sev. 5", "Bf9": "sev. 9",
    "Bz2": "sev. 2", "Bz5": "sev. 5", "Bz9": "sev. 9",
    "Em1": "×1.5", "Em2": "×2", "Em3": "×4", "Em4": "×8", "Ef1": "×0.8", "Ef2": "×0.6", "Ef3": "×0.4", "Ef4": "×0.2",
    "G1": "×0.9", "G2": "×0.75", "G3": "×0.55", "G4": "×0.4", "N1_1": "step 50",
    "NR1": "5%", "NR2": "10%", "NR3": "20%", "NR4": "30%", "NR5": "40%",
    "NL1": "1 call", "NL2": "2 calls", "NL3": "3 calls", "NL4": "4 calls", "NL5": "5 calls",
    "NLw1": "4 steps", "NLw2": "8 steps", "NLw3": "16 steps", "NLw4": "32 steps", "NLw5": "64 steps",
    "K1": "2 cm@60", "K2": "5 cm@60", "K3": "10 cm@60", "K4": "2 cm@160", "K5": "5 cm@160", "K6": "10 cm@160",
    "K7": "2 cm@260", "K8": "5 cm@260", "K9": "10 cm@260", "F60": "t = 60", "F160": "t = 160", "F260": "t = 260",
}
SHAM = {"A0", "P0", "B0"}
REGISTRY_ONLY = ["NLw1", "NLw2", "NLw3", "NLw4", "NLw5"]
VIS_COLOR = {"Visible": ("#2a78d6", "#d9e7f8"), "Hidden": ("#eb6834", "#fbe3d8"), "After verdict": ("#1baf7a", "#d3f0e5")}
INK, INK2, GREY = "#0b0b0b", "#52514e", "#8a8984"


def family_of(level):
    for p in ("Em", "Ef", "NR", "NL", "N1", "Bg", "Bf", "Bz"):
        if level.startswith(p):
            return {"Bg": "B", "Bf": "B", "Bz": "B"}.get(p, p)
    return level[0]


def parse_registry():
    text = open(track(REGISTRY, OUTS), encoding="utf-8").read()
    rows = {}
    for line in text.splitlines():
        if not line.startswith("| ") or line.startswith(("| 代号", "|---")):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) >= 7 and re.fullmatch(r"[A-Z][A-Z0-9]*", c[0]):
            rows[c[0]] = dict(name=c[1], category=c[2], model_input=c[3], levels=c[4], timing=c[5])
    assert "N 族新增 NLw" in text, "registry change log no longer records NLw"
    return rows


def family_attrs(reg):
    A = {}
    for fam, (code, name, timing_prefix, timing_en) in FAMILIES.items():
        r = reg[code]
        assert r["timing"].startswith(timing_prefix), (fam, r["timing"][:30])
        cat, persist = [x.strip() for x in r["category"].split("×")]
        channel = {"环境": "environment", "物理": "physics", "信息": "sensing"}[cat]
        persistence = {"只动一次": "one-shot", "一直在": "persistent"}[persist]
        if "t > 0" in r["timing"] or r["timing"].startswith("仅闭环轨"):
            vis = "After verdict"
        elif r["model_input"].startswith("进"):
            vis = "Visible"
        else:
            vis = "Hidden"
        A[fam] = dict(family_name=name, registry_code=code, channel=channel, persistence=persistence, timing=timing_en,
                      model_input="yes" if r["model_input"].startswith("进") else "no", visibility=vis)
    return A


def level_key(L):
    m = re.match(r"([A-Za-z_]+?)(\d+)$", L)
    fam_rank = ORDER.index(family_of(L))
    sub = {"Bg": 1, "Bf": 2, "Bz": 3, "NLw": 1, "B": 0}.get(re.match(r"[A-Za-z]+", L).group(0), 0)
    return (fam_rank, sub, int(m.group(2)) if m else 0, L)


def main():
    reg = parse_registry()
    fam = family_attrs(reg)
    main_rows = load_main_table(OUTS)
    in_main = collections.defaultdict(set)
    for (M, T, L) in main_rows:
        if L != "none":
            in_main[L].add(M)
    levels = sorted(set(in_main) | set(REGISTRY_ONLY), key=level_key)
    rows = []
    for L in levels:
        f = family_of(L)
        a = fam[f]
        rows.append(dict(family=f, family_name=a["family_name"], level=L, parameter=PARAM[L], channel=a["channel"],
                         persistence=a["persistence"], timing=a["timing"], model_input=a["model_input"],
                         visibility=a["visibility"], sham_or_control="yes" if L in SHAM else "no",
                         in_main_table="yes" if L in in_main else "no",
                         main_table_models=",".join(sorted(in_main.get(L, []))),
                         counted_in_62="yes" if L not in SHAM else "no"))
    with open(out("fig4_levels.tsv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    main_nonbase = [r for r in rows if r["in_main_table"] == "yes"]
    main_nonsham = [r for r in main_nonbase if r["sham_or_control"] == "no"]
    reg_nonsham = [r for r in rows if r["sham_or_control"] == "no"]
    per_fam = {f: dict(main_table=sum(1 for r in main_nonbase if r["family"] == f),
                       sham_control=[r["level"] for r in rows if r["family"] == f and r["sham_or_control"] == "yes"],
                       registry_only=[r["level"] for r in rows if r["family"] == f and r["in_main_table"] == "no"],
                       counted=sum(1 for r in reg_nonsham if r["family"] == f)) for f in ORDER}
    vis = collections.Counter(fam[f]["visibility"] for f in ORDER)
    counts = dict(families=len(ORDER), visibility={k: vis[k] for k in ("Visible", "Hidden", "After verdict")},
                  visibility_families={k: [f for f in ORDER if fam[f]["visibility"] == k] for k in ("Visible", "Hidden", "After verdict")},
                  levels=dict(main_nonbase=len(main_nonbase), main_nonsham=len(main_nonsham), registry_nonsham=len(reg_nonsham),
                              main_nonbase_plus_registry_only=len(rows)),
                  sham_control_levels=sorted(SHAM), registry_only_levels=REGISTRY_ONLY, per_family=per_fam,
                  derivation_of_62=f"{len(main_nonsham)} non-sham main-table levels + {len(REGISTRY_ONLY)} NLw levels = "
                                   f"{len(reg_nonsham)}; the main table alone has {len(main_nonbase)} non-baseline levels "
                                   f"({len(main_nonsham)} after removing A0, P0, B0)",
                  backend=plotlib.backend_note())
    with open(out("fig4_counts.json"), "w", encoding="utf-8") as fh:
        json.dump(counts, fh, ensure_ascii=False, indent=1)
    plot(rows, fam, counts)
    print(json.dumps({k: counts[k] for k in ("families", "visibility", "levels", "derivation_of_62")}, ensure_ascii=False))
    for f in ORDER:
        print(f"  {f:3s} {fam[f]['visibility']:13s} counted={per_fam[f]['counted']:2d} main={per_fam[f]['main_table']:2d} "
              f"sham={per_fam[f]['sham_control']} registry_only={per_fam[f]['registry_only']}")
    write_manifest("make_fig4")


def plot(rows, fam, counts):
    W, Hh = 7.0, 4.6
    fig = plotlib.figure(W, Hh)
    n = len(ORDER)
    ax = fig.axes((0.0, 0.10, 1.0, 0.80), (0, 100), (n + 0.4, -0.2), frame=False)
    x_lvl, bw, gap = 31.0, 6.2, 0.3
    for i, f in enumerate(ORDER):
        a = fam[f]
        c, tint = VIS_COLOR[a["visibility"]]
        y = i + 0.5
        ax.text(5.5, y, f, 7.5, INK, "left", "center", bold=True)
        ax.text(9.5, y - 0.17, a["family_name"], 6.5, INK, "left", "center")
        ax.text(9.5, y + 0.2, a["timing"], 5, INK2, "left", "center")
        R = [r for r in rows if r["family"] == f]
        for j, r in enumerate(R):
            x0 = x_lvl + j * (bw + gap)
            sham, only = r["sham_or_control"] == "yes", r["in_main_table"] == "no"
            face = "#f0efec" if sham else ("#ffffff" if only else tint)
            edge = GREY if sham else c
            ax.rect(x0, i + 0.1, x0 + bw, i + 0.9, face, edge, 0.7, hatch=only, dash=only)
            ax.text(x0 + bw / 2, y - 0.14, r["level"], 6, INK if not sham else INK2, "center", "center", bold=not sham)
            ax.text(x0 + bw / 2, y + 0.2, r["parameter"].replace(" (sham)", ""), 4.6, INK2, "center", "center")
    for vis, (c, _) in VIS_COLOR.items():
        idx = [i for i, f in enumerate(ORDER) if fam[f]["visibility"] == vis]
        ax.rect(1.0, min(idx) + 0.08, 2.2, max(idx) + 0.92, c, c, 0.5)
    # header + legend
    fig.text(0.01, 0.965, "Shift families and levels", 9, INK, bold=True)
    lv = counts["levels"]
    fig.text(0.01, 0.925, f"{counts['families']} families · {lv['main_nonsham']} levels in the closed-loop main table "
             f"+ {len(counts['registry_only_levels'])} NLw levels (registered, hatched) = {lv['registry_nonsham']}; "
             f"grey = sham / control (not counted)", 6.3, INK2)
    lx = 0.01
    for vis, (c, tint) in VIS_COLOR.items():
        fax = fig.axes((lx, 0.025, 0.02, 0.04), (0, 1), (0, 1), frame=False)
        fax.rect(0.05, 0.05, 0.95, 0.95, tint, c, 0.8)
        names = ", ".join(counts["visibility_families"][vis])
        fig.text(lx + 0.026, 0.045, f"{vis} ({counts['visibility'][vis]} families): {names}", 6.5, INK)
        lx += {"Visible": 0.27, "Hidden": 0.36, "After verdict": 0.3}[vis]
    fig.save(out("fig4.pdf"), out("fig4.png"))


if __name__ == "__main__":
    main()
