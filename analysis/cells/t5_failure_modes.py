"""t5 real outcomes by level, from the simulator state of the truth (a) replays (check-6 interpretation; read-only).

Per unit (lingbot_famA / lingbot_famP, t5): bit-exact replay of the stored actions; book origin vs the back-compartment
box (t5_geometry.Geo). Classes:
  success              the simulator reported success at some step
  never_grasped        the book never rose more than 3 cm above its start height
  lifted_never_inside  lifted, but its origin never entered the box
  inside_then_lost     -- impossible for a failure (inside = success); kept as a consistency check
plus, for failures, where the book ends: final origin distance to the box centre (xy) and whether it is on the caddy.

  python t5_failure_modes.py   -> out/content_review/t5_failure_modes.json
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import numpy as np
from collections import Counter

ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap(); _wmvp.bootstrap()
from libero_env_adapter import _make_env, _anchor
from t5_geometry import Geo


def main():
    rows = []
    for run, bank in (("lingbot_famA", "familyA"), ("lingbot_famP", "familyP")):
        for l in open(f"{ROOT}/libero/out/runs/{run}/records_t5.jsonl"):
            r = json.loads(l)
            a = _anchor(f"{ROOT}/libero/out/bank/{bank}/{bank}_anchors.json", r["anchor_id"])
            acts = np.load(f"{ROOT}/{r['action_file']}")["env"]
            env = _make_env(a, H=len(acts)); u = env.envs[0]; geo = Geo(u); s0 = geo.state()
            z0 = s0["centre"][2]; zmax = z0; inside_any = False
            for x in acts:
                u.step(np.asarray(x, np.float32)); s = geo.state()
                zmax = max(zmax, s["centre"][2]); inside_any |= s["inside"]
            box_c = (s["lb"] + s["ub"]) / 2
            env.close()
            if r["real_verdict"]:
                cls = "success"
            elif zmax - z0 < 0.03:
                cls = "never_grasped"
            elif not inside_any:
                cls = "lifted_never_inside"
            else:
                cls = "inside_then_lost"
            rows.append(dict(unit=r["unit_id"], setting=r["setting_id"], cls=cls, lift_m=round(float(zmax - z0), 3),
                             final_xy_dist_to_box_m=round(float(np.linalg.norm(s["centre"][:2] - box_c[:2])), 3),
                             final_z_m=round(float(s["centre"][2]), 3), box_top_z_m=round(float(s["ub"][2]), 3)))
            print(rows[-1], flush=True)
    by = {}
    for st in sorted({x["setting"] for x in rows}):
        R = [x for x in rows if x["setting"] == st]
        by[st] = dict(Counter(x["cls"] for x in R))
    json.dump(dict(by_setting=by, units=rows), open(f"{ROOT}/libero/out/content_review/t5_failure_modes.json", "w"), indent=1)
    print(json.dumps(by))


if __name__ == "__main__":
    main()
