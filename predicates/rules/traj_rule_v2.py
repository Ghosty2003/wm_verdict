"""Trajectory-level decision rule v2 (2026-10-20 ruling 2; spec revision #4).

Rule v1 = reach-anytime on single frames: a trajectory is a success iff any frame 1..H has p >= threshold. The
frame-level gate (85 %) does not bound the trajectory-level false-alarm rate over 520 frames. Rule v2 = success iff
p >= threshold on k CONSECUTIVE frames somewhere in 1..H (same threshold, same predicate v3).

  eval     REAL trajectories only (simulator labels known): v3 calibration episodes (out-of-fold predictions) and the
           truth (a) replays of lingbot_famA / lingbot_famP (deployed predicate; these episodes were never in the
           calibration fit). Per task and k in K: trajectory-level TPR / TNR, pooled and per source. Selected k = the
           smallest k with pooled TPR >= 0.85 and TNR >= 0.85; none -> STOP. Imagined data are not read.
           -> out/rule_v2/rule_v2_eval.json
  rescore  after k is frozen in protocol.yaml (libero_lingbot_v3.trajectory_rule_v2): recompute the model verdict of
           every famA / famP unit under v1 and v2 from the stored per-frame model predictions.
           -> out/rule_v2/rescore.json

  eval_ext t5 only: extended grid (16, 24, 36, 48), k chosen on half A of the real episodes, confirmed on half B
           -> out/rule_v2/rule_v2_eval_t5_ext.json

  python traj_rule_v2.py eval | eval_ext | rescore
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import numpy as np

ROOT = _wmvp.P("$WMV_ROOT")
OUT = f"{ROOT}/libero/out/rule_v2"
K = (1, 2, 3, 5, 8)
GATE = 0.85
RUNS = {"famA": "lingbot_famA", "famP": "lingbot_famP"}
TASKS = {"t5": 5, "t3": 3}


def fires(p, thr, k):
    """True iff p >= thr on at least k consecutive entries."""
    run = 0
    for x in p >= thr:
        run = run + 1 if x else 0
        if run >= k:
            return True
    return False


def first_fire(p, thr, k):
    """index (into p) of the frame that completes the first run of k consecutive hits; None if none."""
    run = 0
    for i, x in enumerate(p >= thr):
        run = run + 1 if x else 0
        if run >= k:
            return i
    return None


def threshold(t):
    return float(json.load(open(f"{ROOT}/libero/out/predicate_v3/{t}/calibration.json"))["threshold"])


def records(run, t):
    return [json.loads(l) for l in open(f"{ROOT}/libero/out/runs/{run}/records_{t}.jsonl") if l.strip()]


def real_trajectories(t):
    """(source, id, p over frames 1..end, label) for every real trajectory with a simulator label."""
    out = []
    z = np.load(f"{ROOT}/libero/out/predicate_v3/{t}/oof.npz")
    ep, p, y = z["episode"], z["oof"], z["labels"]
    order = list(dict.fromkeys(ep.tolist()))
    for e in order:
        idx = np.where(ep == e)[0]
        assert np.all(np.diff(idx) == 1), f"calibration episode {e} is not contiguous"
        out.append(("calib", e, p[idx][1:], bool(y[idx][1:].any())))     # drop frame 0 (initial observation)
    for src, run in RUNS.items():
        for r in records(run, t):
            tr = np.load(f"{ROOT}/{r['trajectory_file']}")
            assert bool(tr["real_success"].any()) == bool(r["real_verdict"]), r["unit_id"]
            out.append((src, r["unit_id"], tr["real_pred"][1:], bool(r["real_verdict"])))
    return out


def rates(rows):
    pos = [f for f, y in rows if y]; neg = [f for f, y in rows if not y]
    return dict(n=len(rows), n_pos=len(pos), n_neg=len(neg),
                tpr=(sum(pos) / len(pos)) if pos else None, tnr=(sum(not f for f in neg) / len(neg)) if neg else None)


def eval_mode():
    os.makedirs(OUT, exist_ok=True)
    res = dict(rule="success iff p >= threshold on k consecutive frames in 1..H (k = 1 is rule v1)", K=list(K),
               gate=dict(min_tpr=GATE, min_tnr=GATE, on="pooled real trajectories, per task"), tasks={})
    stop = []
    for t in TASKS:
        thr = threshold(t)
        T = real_trajectories(t)
        per_k = {}
        for k in K:
            f = [(fires(p, thr, k), y) for _, _, p, y in T]
            per_k[k] = dict(pooled=rates(f), by_source={s: rates([x for x, (src, *_) in zip(f, T) if src == s])
                                                         for s in ("calib", "famA", "famP")})
        ok = [k for k in K if per_k[k]["pooled"]["tpr"] >= GATE and per_k[k]["pooled"]["tnr"] >= GATE]
        sel = min(ok) if ok else None
        if sel is None:
            stop.append(t)
        res["tasks"][t] = dict(threshold=thr, n_trajectories=len(T), feasible_k=ok, selected_k=sel, per_k=per_k)
        for k in K:
            pk = per_k[k]
            print(t, f"k={k}", "pooled TPR %.3f TNR %.3f" % (pk["pooled"]["tpr"], pk["pooled"]["tnr"]),
                  " | ".join(f"{s} TPR {v['tpr']:.3f} TNR {v['tnr']:.3f} (n+ {v['n_pos']}, n- {v['n_neg']})"
                             for s, v in pk["by_source"].items()), flush=True)
        print(t, "selected k =", sel)
    res["stop"] = stop
    json.dump(res, open(f"{OUT}/rule_v2_eval.json", "w"), indent=1)
    print("STOP: no feasible k for " + ", ".join(stop) if stop else "feasible for every task")


K_EXT = (16, 24, 36, 48)
SPLIT_SEED = 20261020


def eval_ext_mode():
    """t5 only (2026-10-20 ruling 2 on the stop): grid extension guided by the false-alarm run-length diagnostic, with a
    held-out confirmation. Real trajectories are split in half by episode (fixed seed, stratified by source x label);
    k is chosen on half A (smallest k in K_EXT with TPR and TNR >= 0.85), then confirmed on half B. A k that fails the
    confirmation is void and the next A-feasible k in the grid is tried; all void -> STOP. Imagined data are not read."""
    t = "t5"; thr = threshold(t); T = real_trajectories(t)
    rs = np.random.RandomState(SPLIT_SEED); half = np.zeros(len(T), int)
    for s in ("calib", "famA", "famP"):
        for y in (True, False):
            idx = [i for i, (src, _, _, lab) in enumerate(T) if src == s and lab == y]
            perm = rs.permutation(idx); half[perm[len(perm) // 2:]] = 1
    res = dict(task=t, threshold=thr, grid=list(K_EXT), split_seed=SPLIT_SEED, split="by episode, stratified by source x label",
               n_A=int((half == 0).sum()), n_B=int((half == 1).sum()),
               split_ids={"A": [T[i][1] for i in range(len(T)) if half[i] == 0], "B": [T[i][1] for i in range(len(T)) if half[i] == 1]},
               per_k={}, tried=[], selected_k=None)
    for k in K_EXT:
        f = [fires(p, thr, k) for _, _, p, _ in T]
        res["per_k"][k] = {h: rates([(f[i], T[i][3]) for i in range(len(T)) if half[i] == hi]) for h, hi in (("A", 0), ("B", 1))}
        res["per_k"][k]["B_by_source"] = {s: rates([(f[i], T[i][3]) for i in range(len(T)) if half[i] == 1 and T[i][0] == s])
                                          for s in ("calib", "famA", "famP")}
        a, b = res["per_k"][k]["A"], res["per_k"][k]["B"]
        print(t, f"k={k}", "A TPR %.3f TNR %.3f | B TPR %.3f TNR %.3f" % (a["tpr"], a["tnr"], b["tpr"], b["tnr"]), flush=True)
    for k in K_EXT:
        a, b = res["per_k"][k]["A"], res["per_k"][k]["B"]
        if not (a["tpr"] >= GATE and a["tnr"] >= GATE):
            continue
        ok = b["tpr"] >= GATE and b["tnr"] >= GATE
        res["tried"].append(dict(k=k, A_feasible=True, B_confirmed=ok))
        if ok:
            res["selected_k"] = k; break
    res["stop"] = res["selected_k"] is None
    json.dump(res, open(f"{OUT}/rule_v2_eval_t5_ext.json", "w"), indent=1)
    print("tried", res["tried"], "selected k =", res["selected_k"], "STOP" if res["stop"] else "")


def rescore_mode():
    import yaml
    rule = yaml.safe_load(open(f"{ROOT}/protocol.yaml"))["libero_lingbot_v3"]["trajectory_rule_v2"]
    out = dict(rule=rule, cells={}, units={})
    for t in TASKS:
        k2 = int(rule["k"][t]); thr = threshold(t)
        for fam, run in RUNS.items():
            for r in records(run, t):
                tr = np.load(f"{ROOT}/{r['trajectory_file']}")
                mp = tr["model_pred"][1:]
                v1 = fires(mp, thr, 1)
                assert v1 == bool(r["model_verdict"]), r["unit_id"]        # v1 must reproduce the record
                v2 = fires(mp, thr, k2)
                f2 = first_fire(mp, thr, k2)
                out["units"][r["unit_id"]] = dict(task=t, family=fam, setting=r["setting_id"], real=bool(r["real_verdict"]),
                                                  model_v1=v1, model_v2=v2,
                                                  model_v2_fire_frame=None if f2 is None else f2 + 1,
                                                  real_pred_v2=fires(tr["real_pred"][1:], thr, k2))
        for s in sorted({u["setting"] for u in out["units"].values() if u["task"] == t}):
            U = [u for u in out["units"].values() if u["task"] == t and u["setting"] == s]
            c = dict(n=len(U), real_success=sum(u["real"] for u in U))
            for v in ("v1", "v2"):
                c[f"model_success_{v}"] = sum(u[f"model_{v}"] for u in U)
                c[f"false_safe_{v}"] = sum(u[f"model_{v}"] and not u["real"] for u in U)
                c[f"false_unsafe_{v}"] = sum((not u[f"model_{v}"]) and u["real"] for u in U)
            out["cells"][f"{t}/{s}"] = c
            print(t, s, c, flush=True)
    json.dump(out, open(f"{OUT}/rescore.json", "w"), indent=1)


if __name__ == "__main__":
    {"eval": eval_mode, "eval_ext": eval_ext_mode, "rescore": rescore_mode}[sys.argv[1]]()
