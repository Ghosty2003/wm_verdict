"""F 失败案例补充：同格成功 vs 失败单元，置黑调用的想象亮度与动作（只读存档，不重放）。→ out/f_family/cases/success_vs_fail.json"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import glob, json, numpy as np
ROOT = _wmvp.P("$WMV_ROOT"); RUNS = f"{ROOT}/libero/out/runs"
out = {}
for T, tid, lv in (("T1", 3, "F160"), ("T3", 4, "F60")):
    rows = [json.loads(l) for f in sorted(glob.glob(f"{RUNS}/openwam_famF/closed_t{tid}.*.jsonl")) for l in open(f) if l.strip()]
    rows = [r for r in rows if r["setting_id"] == lv]
    R = []
    for r in rows:
        z = np.load(f"{ROOT}/{r['closed_loop_file']}")
        b = r["perturbation"]["blackout_call"]; cl = z["imagined_call_lens"]; off = np.concatenate([[0], np.cumsum(cl)])
        imB = z["imagined"][off[b] + 1:off[b + 1], :256]; imN = z["imagined"][off[b + 1] + 1:off[b + 2], :256]
        a = z["commanded"]; s = 32 * b
        seg = lambda i, j: a[i:j]
        R.append(dict(uid=r["unit_id"], success=bool(r["closed_loop"]["success"]), fs=r["closed_loop"]["first_success_step"],
                      imag_mean_blackout=float(imB.mean()), imag_mean_next=float(imN.mean()),
                      grip_prev=float(seg(s - 32, s)[:, 6].mean()), grip_blackout=float(seg(s, s + 32)[:, 6].mean()), grip_next=float(seg(s + 32, s + 64)[:, 6].mean()),
                      trans_prev=float(np.linalg.norm(seg(s - 32, s)[:, :3], axis=1).mean()), trans_blackout=float(np.linalg.norm(seg(s, s + 32)[:, :3], axis=1).mean()),
                      trans_next=float(np.linalg.norm(seg(s + 32, s + 64)[:, :3], axis=1).mean()),
                      z_blackout=float(seg(s, s + 32)[:, 2].sum()), z_prev=float(seg(s - 32, s)[:, 2].sum())))
    out[f"{T}_{lv}"] = R
    for k in (True, False):
        g = [x for x in R if x["success"] == k]
        f = lambda key: round(float(np.mean([x[key] for x in g])), 3)
        print(T, lv, "success" if k else "FAIL", len(g), {key: f(key) for key in ("imag_mean_blackout", "imag_mean_next", "grip_prev", "grip_blackout", "grip_next",
                                                                                 "trans_prev", "trans_blackout", "trans_next", "z_prev", "z_blackout")})
json.dump(out, open(f"{ROOT}/libero/out/f_family/cases/success_vs_fail.json", "w"), indent=1)
