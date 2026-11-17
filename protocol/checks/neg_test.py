import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import json, sys, os, numpy as np
_wmvp.bootstrap()
import min_artifacts as MA
row = json.loads(open(sorted(__import__("glob").glob(_wmvp.P("$WMV_ROOT/libero/out/runs/lingbot_famF/closed_t*.jsonl")))[0]).readline())
src = MA.npz_of(row); z = dict(np.load(src))
tmp = _wmvp.P("$WMV_ROOT/libero/out/artifact_verify/neg_tmp.npz")
cases = {"no gpu field": (dict(row, gpu=None), z),
         "imagined latents dropped": (row, {k: v for k, v in z.items() if "imag_latents" not in k}),
         "executed truncated": (row, dict(z, executed=z["executed"][:-5])),
         "intact (should pass)": (row, z)}
for name, (r, d) in cases.items():
    np.savez(tmp, **d)
    try:
        MA.assert_unit(tmp, r); print("PASS ", name)
    except MA.MinArtifactsError as e:
        print("RAISE", name, "->", str(e)[:160])
os.remove(tmp)
