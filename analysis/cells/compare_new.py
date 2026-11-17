"""New families + B (rulings 2026-10-26 / 2026-11-01) for LingBot-VA and OpenWAM-alpha, H = 400: per-unit rows in the compare_models schema.

  E (Em mass, Ef friction), G (controller gain), N (N1 one zero action, NR random zero actions, NL delay)
      the model never sees these: open-loop verdict and imagined completion step are the BASELINE unit's (no new
      imagination); truth (a) = replay of the baseline's imagined actions under the perturbation (phys_truth.py);
      closed loop (b) = phys_closed_*.py (reference column; first run's first success step).
  K (push) closed loop only: no open-loop verdict and no truth (a) row (verdict / real = None).
  O (occlusion of the model's first agentview frame) new imaginations (lingbot_famO.py / ow_famO.py):
      LingBot-VA scored here with the final rules (t5 v3 k = 24 on model_pred; t3 v4 conjunction k = 1 on p_close /
      p_bowl, both computed in the run); OpenWAM-alpha t3 from ow_score --run famO, t5 pending.
Sham levels are not rows (they equal the baseline bit for bit; checked in the runs). Missing closed-loop records
(still running) give closed = None and are reported as counts.

  python compare_new.py   -> out/cells/compare_new_H400.json
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import glob
import json
import numpy as np
from cells_final import K_T5, K_T3
from compare_models import lingbot_base_units, openwam_units, onset, frac, med_iqr

ROOT = _wmvp.P("$WMV_ROOT")
RUNS = f"{ROOT}/libero/out/runs"
OUT = f"{ROOT}/libero/out/cells"
H = 400
GROUP = {"Em": "E", "Ef": "E", "G": "G", "N1": "N", "NR": "N", "NL": "N", "NLw": "N", "K": "K", "O": "O"}
# delay families: the open-loop replay of a fixed action sequence under a delay is a pure time shift (hold d steps, then
# the same actions), so their open-loop cells are a reference only and do not enter danger / miss (ruling 2026-10-28)
OPEN_LOOP_REFERENCE_ONLY = ("NL", "NLw")
# B (image corruption, unified protocol v1.0): trajectory-level predicate gates (ruling 2026-11-01); a failed cell
# keeps its data and its truth / closed columns, but the verdict column is pending (predicate unreliable at the level)
import glob as _g
GATE_PENDING = set()
for _m in ("lingbot", "openwam"):
    _gp = f"{ROOT}/libero/out/b_family/gate/gate_traj_{_m}.json"
    if os.path.exists(_gp):
        for _cell in json.load(open(_gp))["failed"]:
            _t, _L = _cell.split("/")
            GATE_PENDING.add((_m, _t, _L))


def jl(pattern):
    return [json.loads(l) for f in sorted(glob.glob(pattern)) for l in open(f)]


def closed_of(rec):
    cl = rec["closed_loop"]
    s = cl["first_success_step"] if cl["success"] and cl["first_success_step"] <= H else None
    return s is not None, s


def phys_units(model, t):
    base = {u["uid"]: u for u in (lingbot_base_units(t)[0] if model == "lingbot" else openwam_units("base", t))}
    cl = {r["unit_id"]: r for r in jl(f"{RUNS}/{model}_phys/closed_{t}*.jsonl")}
    U = []
    for r in jl(f"{RUNS}/{model}_phys/truth_{t}.jsonl"):
        if r["perturbation"].get("sham"):
            continue
        b = base[r["base_unit"]]; rs = r["real_first_success_step"]; real = rs is not None and rs <= H
        c = cl.get(r["unit_id"]); closed, cs = closed_of(c) if c else (None, None)
        U.append(dict(uid=r["unit_id"], base=r["base_unit"], level=r["setting_id"], group=GROUP[r["perturbation"]["family"]],
                      verdict=b["verdict"], imag_step=b["imag_step"], real=real, real_step=rs if real else None,
                      closed=closed, closed_step=cs, closed_consistent=c["closed_loop"]["conclusion_consistent"] if c else None))
    for uid, c in cl.items():                                    # K: closed loop only
        if c["setting_id"].startswith("K"):
            closed, cs = closed_of(c)
            U.append(dict(uid=uid, base=c["base_unit"], level=c["setting_id"], group="K", verdict=None, imag_step=None,
                          real=None, real_step=None, closed=closed, closed_step=cs,
                          closed_consistent=c["closed_loop"]["conclusion_consistent"]))
    return U


def famB_units(model, t):
    U = []
    thr5 = float(json.load(open(f"{ROOT}/libero/out/predicate_v3/t5/calibration.json"))["threshold"])
    v4 = {d: json.load(open(f"{ROOT}/libero/out/predicate_v4_t3/calibration_{d}.json"))["threshold"] for d in ("close", "bowl_out")}
    for r in jl(f"{RUNS}/{model}_famB/records_{t}*.jsonl"):
        L = r["setting_id"]; closed, cs = closed_of(r)
        if model == "lingbot":
            tr = np.load(f"{ROOT}/{r['trajectory_file']}")
            if t == "t5":
                on = onset(tr["model_pred"][1:H + 1] >= thr5, K_T5)
            else:
                on = onset(((tr["p_close"] >= v4["close"]) & (tr["p_bowl"] < v4["bowl_out"]))[1:H + 1], K_T3)
            verdict, imag = on is not None, (on + 1 if on is not None else None)
        else:
            verdict = None if r["model_verdict"] is None else bool(r["model_verdict"])
            imag = r.get("model_first_success_step")
        rs = r["real_first_success_step"]; real = rs is not None and rs <= H
        U.append(dict(uid=r["unit_id"], level=L, group="B", verdict=verdict, imag_step=imag,
                      real=real, real_step=rs if real else None, closed=closed, closed_step=cs,
                      closed_consistent=r["closed_loop"].get("conclusion_consistent"),
                      replay_identical=r["replay_check_identical"],
                      gate_pending=(model, t, L) in GATE_PENDING))
    return U


def famO_units(model, t):
    U = []
    if model == "lingbot":
        thr5 = float(json.load(open(f"{ROOT}/libero/out/predicate_v3/t5/calibration.json"))["threshold"])
        v4 = {d: json.load(open(f"{ROOT}/libero/out/predicate_v4_t3/calibration_{d}.json"))["threshold"] for d in ("close", "bowl_out")}
        for r in jl(f"{RUNS}/lingbot_famO/records_{t}.*.jsonl"):
            tr = np.load(f"{ROOT}/{r['trajectory_file']}")
            if t == "t5":
                on = onset(tr["model_pred"][1:H + 1] >= thr5, K_T5)
            else:
                on = onset(((tr["p_close"] >= v4["close"]) & (tr["p_bowl"] < v4["bowl_out"]))[1:H + 1], K_T3)
            rs = r["real_first_success_step"]; real = rs is not None and rs <= H; closed, cs = closed_of(r)
            U.append(dict(uid=r["unit_id"], level=r["setting_id"], group="O", verdict=on is not None,
                          imag_step=on + 1 if on is not None else None, real=real, real_step=rs if real else None,
                          closed=closed, closed_step=cs, closed_consistent=r["closed_loop"]["conclusion_consistent"],
                          replay_identical=r["replay_check_identical"]))
    else:
        for r in jl(f"{RUNS}/openwam_famO/records_{t}.jsonl"):
            closed, cs = closed_of(r)
            U.append(dict(uid=r["unit_id"], level=r["setting_id"], group="O",
                          verdict=None if r["model_verdict"] is None else bool(r["model_verdict"]),
                          imag_step=r.get("model_first_success_step"), real=bool(r["real_verdict"]), real_step=r["real_first_success_step"],
                          closed=closed, closed_step=cs, closed_consistent=r["closed_loop"]["conclusion_consistent"],
                          replay_identical=r["replay_check_identical"]))
    return U


def level_cell(U):
    """compare_models.cell, tolerant of rows still missing (closed None) and of K (no open loop)."""
    n = len(U); c = dict(n=n)
    C = [u for u in U if u["closed"] is not None]
    c["closed"] = frac(sum(u["closed"] for u in C), len(C)); c["closed_missing"] = n - len(C)
    c["closed_inconsistent"] = sum(u["closed_consistent"] is False for u in U)
    if all(u["real"] is None for u in U):
        c["verdict"] = "closed_loop_only"; return c
    fail = [u for u in U if not u["real"]]; succ = [u for u in U if u["real"]]; yes = [u for u in U if u["verdict"]]
    c["real"] = frac(len(succ), n)
    if any(u["verdict"] is None for u in U):
        c["verdict"] = "pending"; return c
    if any(u.get("gate_pending") for u in U):
        c["verdict"] = "pending (predicate unreliable at this corruption level; trajectory gate 2026-11-01)"; return c
    if U and U[0]["level"].rstrip("0123456789") in OPEN_LOOP_REFERENCE_ONLY:
        c["verdict"] = "open_loop_reference_only (delay = time shift of the replayed actions)"; return c
    c.update(model=frac(len(yes), n), danger=frac(sum(not u["real"] for u in yes), len(yes)),
             miss=frac(sum(u["verdict"] for u in fail), len(fail)), false_danger=frac(sum(not u["verdict"] for u in succ), len(succ)))
    return c


def main():
    res = dict(H=H, units={}, cells={}, times={})
    for model in ("lingbot", "openwam"):
        for t in ("t5", "t3"):
            U = phys_units(model, t) + famO_units(model, t) + famB_units(model, t)
            res["units"][f"{model}/{t}"] = U
            for L in sorted({u["level"] for u in U}):
                V = [u for u in U if u["level"] == L]
                res["cells"][f"{model}/{t}/{L}"] = level_cell(V)
                res["times"][f"{model}/{t}/{L}"] = dict(
                    real=med_iqr([u["real_step"] for u in V if u["real"]]),
                    closed=med_iqr([u["closed_step"] for u in V if u["closed"]]),
                    imag_minus_real=None if any(u["verdict"] is None for u in V) else
                    med_iqr([u["imag_step"] - u["real_step"] for u in V if u["verdict"] and u["real"]]))
    json.dump(res, open(f"{OUT}/compare_new_H400.json", "w"), indent=1)
    for k, c in res["cells"].items():
        d = c.get("danger"); print(k, c["n"], "real", c.get("real", {}).get("x"), "closed", c["closed"]["x"], "/", c["closed"]["n"],
                                   "danger", f"{d['x']}/{d['n']}" if d else c.get("verdict"))


if __name__ == "__main__":
    main()
