"""The 20 cells (lingbot_famA A0-A4, lingbot_famP P0-P4; t5 and t3) under the final rules (2026-10-21).

Model verdict per unit, two rule sets:
  v1  single-frame reach-anytime with predicate v3 (the run's own record: records.model_verdict)
  v2  t5: predicate v3, k = 24 consecutive frames on the stored full per-frame model_pred (spec revision #4)
      t3: predicate v4 conjunction (Close AND NOT bowl-outside-visible), k = 1, on the scores from the same-seed
          recomputation (rescore/<unit>.npz; units that failed an equivalence gate are excluded and counted)
Truth: simulator check_success, reach-anytime, truth (a) open-loop replay (records.real_verdict); closed-loop (b)
success is the reference column. Intervals: Clopper-Pearson 95 %.
Cross-reference (2026-10-21 ruling): v2 false-safe units x final content labels (out/content_review/final_labels.json);
"clean prediction error" = v2 false-safe AND the imagined endpoint looks like success.
Layer 2 (predicate on REAL frames under the v2 rule) per task.

  python cells_final.py     -> out/cells/cells_final.json, out/cells/cells_final.md
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import numpy as np
from scipy.stats import beta

ROOT = _wmvp.P("$WMV_ROOT")
OUT = f"{ROOT}/libero/out/cells"
RUNS = (("famA", "lingbot_famA", ("A0", "A1", "A2", "A3", "A4")), ("famP", "lingbot_famP", ("P0", "P1", "P2", "P3", "P4")))
K_T5 = 24
K_T3 = 1


def fires(h, k):
    run = 0
    for x in h:
        run = run + 1 if x else 0
        if run >= k:
            return True
    return False


def cp(x, n):
    lo = 0.0 if x == 0 else float(beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(0.975, x + 1, n - x))
    return [round(lo, 3), round(hi, 3)]


def main(H=520):
    """H: horizon in env steps; every sequence is truncated to steps / frames 1..H (2026-10-23 ruling: H = 400 is the
    official horizon; H = 520 tables are kept, marked superseded)."""
    os.makedirs(OUT, exist_ok=True)
    sfx = "" if H == 520 else f"_H{H}"
    thr5 = float(json.load(open(f"{ROOT}/libero/out/predicate_v3/t5/calibration.json"))["threshold"])
    v4 = {d: json.load(open(f"{ROOT}/libero/out/predicate_v4_t3/calibration_{d}.json"))["threshold"] for d in ("close", "bowl_out")}
    rescore = {json.loads(l)["unit"]: json.loads(l) for l in open(f"{ROOT}/libero/out/predicate_v4_t3/rescore/rescore_log.jsonl")}
    labels = {u["unit_id"]: u for u in json.load(open(f"{ROOT}/libero/out/content_review/final_labels.json"))["units"]}
    cells, l2 = {}, {"t5": [], "t3": []}
    excluded = []
    for fam, run, settings in RUNS:
        for t in ("t5", "t3"):
            recs = [json.loads(l) for l in open(f"{ROOT}/libero/out/runs/{run}/records_{t}.jsonl")]
            for s in settings:
                U = []
                for r in (x for x in recs if x["setting_id"] == s):
                    uid = r["unit_id"]; tr = np.load(f"{ROOT}/{r['trajectory_file']}")
                    if t == "t5":
                        v2 = fires(tr["model_pred"][1:H + 1] >= thr5, K_T5)
                        real_v2 = fires(tr["real_pred"][1:H + 1] >= thr5, K_T5)
                    else:
                        if not rescore.get(uid, {}).get("pass"):
                            excluded.append(uid); continue
                        z = np.load(f"{ROOT}/libero/out/predicate_v4_t3/rescore/{uid}.npz")
                        v2 = fires(((z["p_close"] >= v4["close"]) & (z["p_bowl"] < v4["bowl_out"]))[1:H + 1], K_T3)
                        zr = np.load(f"{ROOT}/libero/out/predicate_v4_t3/traj/{uid}.npz")
                        real_v2 = fires(((zr["p_close"] >= v4["close"]) & (zr["p_bowl"] < v4["bowl_out"]))[1:H + 1], K_T3)
                    real = bool(tr["real_success"][:H].any())
                    cl = r["closed_loop"]; closed = bool(cl["success"] and cl["first_success_step"] <= H)
                    v1 = fires(tr["model_pred"][1:H + 1] >= float(r["threshold"]), 1)
                    U.append(dict(unit=uid, real=real, closed_loop=closed, v1=v1, v2=v2, real_v2=real_v2,
                                  real_first=r["real_first_success_step"],
                                  content=labels.get(uid, {}).get("label")))
                    l2[t].append((real_v2, real))
                n = len(U); c = {}
                for key in ("real", "closed_loop", "v1", "v2"):
                    x = sum(u[key] for u in U); c[key] = dict(x=x, n=n, rate=round(x / n, 3) if n else None, cp95=cp(x, n) if n else None)
                for v in ("v1", "v2"):
                    c[f"false_safe_{v}"] = sum(u[v] and not u["real"] for u in U)
                    c[f"false_unsafe_{v}"] = sum((not u[v]) and u["real"] for u in U)
                    c[f"transfer_gap_{v}"] = round((c[v]["x"] - c["real"]["x"]) / n, 3) if n else None
                fs2 = [u for u in U if u["v2"] and not u["real"]]
                c["v2_false_safe_content"] = {lab: sum(u["content"] == lab for u in fs2)
                                              for lab in ("looks_success", "borderline", "not_success_but_triggered", None)}
                c["clean_prediction_errors_v2"] = c["v2_false_safe_content"]["looks_success"]
                c["units_real_success_after_H"] = [u["unit"] for u in U if not u["real"] and u["real_first"]]
                c["v2_false_safe_without_content_label"] = c["v2_false_safe_content"][None]
                cells[f"{t}/{s}"] = c
    layer2 = {}
    for t, rows in l2.items():
        pos = [p for p, y in rows if y]; neg = [p for p, y in rows if not y]
        layer2[t] = dict(n=len(rows), tpr=round(float(np.mean(pos)), 3), tnr=round(1 - float(np.mean(neg)), 3) if neg else None)
    res = dict(H=H, rules=dict(t5=f"v3 predicate, k = {K_T5}", t3=f"v4 conjunction, k = {K_T3}", v1="v3 predicate, k = 1 (records)"),
               cells=cells, layer2_predicate_on_real_v2=layer2, t3_excluded_units=excluded)
    json.dump(res, open(f"{OUT}/cells_final{sfx}.json", "w"), indent=1)
    head = [f"# 20 格结果，H = {H}" + ("（正式）" if H == 400 else "（superseded：正式口径为 H = 400，见 cells_final_H400.md）" if H == 520 else ""), ""]
    lines = head + ["| cell | real (a) | closed loop (b) | model v1 | model v2 | FS v1 | FS v2 | clean pred. err. v2 | FS v2 borderline | FS v2 trig. |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k_, c in cells.items():
        f = lambda d: f"{d['x']}/{d['n']} {d['cp95']}"
        lines.append(f"| {k_} | {f(c['real'])} | {f(c['closed_loop'])} | {f(c['v1'])} | {f(c['v2'])} | {c['false_safe_v1']} | "
                     f"{c['false_safe_v2']} | {c['clean_prediction_errors_v2']} | {c['v2_false_safe_content']['borderline']} | "
                     f"{c['v2_false_safe_content']['not_success_but_triggered']} |")
    open(f"{OUT}/cells_final{sfx}.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("layer2", layer2, "t3 excluded", len(excluded))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 520)
