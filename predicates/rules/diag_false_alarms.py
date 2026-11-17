"""read-only diagnostic for the rule-v2 stop: shape of trajectory-level false alarms on REAL failure trajectories."""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import sys, json, numpy as np
_wmvp.bootstrap()
from traj_rule_v2 import real_trajectories, threshold
out = {}
for t in ("t5", "t3"):
    thr = threshold(t); rows = []
    for src, uid, p, y in real_trajectories(t):
        if y:
            continue
        h = p >= thr; runs, r = [], 0
        for x in h:
            r = r + 1 if x else 0
            if x: runs.append(r)
        longest = max(runs) if runs else 0
        rows.append(dict(src=src, uid=uid, longest=longest, frac=float(h.mean()), n=len(p),
                         fires_at_end=bool(h[-10:].all()), first=int(np.argmax(h)) + 1 if h.any() else None))
    fa = [r for r in rows if r["longest"] > 0]
    L = np.array([r["longest"] for r in fa])
    out[t] = dict(n_neg=len(rows), n_fire_k1=len(fa), longest_run_quantiles={q: float(np.quantile(L, q)) for q in (.1, .25, .5, .75, .9)},
                  n_longest_ge_50=int((L >= 50).sum()), n_longest_ge_100=int((L >= 100).sum()),
                  n_fire_on_last_10=sum(r["fires_at_end"] for r in fa),
                  by_source={s: dict(neg=sum(r["src"] == s for r in rows), fire=sum(r["src"] == s for r in fa),
                                     fire_on_last_10=sum(r["src"] == s and r["fires_at_end"] for r in fa)) for s in ("calib", "famA", "famP")},
                  examples=sorted(fa, key=lambda r: -r["longest"])[:8])
    print(t, json.dumps({k: v for k, v in out[t].items() if k != "examples"}))
    for e in out[t]["examples"][:5]: print("  ", e)
json.dump(out, open(_wmvp.P("$WMV_ROOT/libero/out/rule_v2/diag_false_alarms.json"), "w"), indent=1)
