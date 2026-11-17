"""t3 sub-goal labels from simulator state (spec revision #5): Close / In via LIBERO's own predicates, and the bowl's
visibility in the two policy cameras via a MuJoCo geom-segmentation render. The segmentation renderer only reads
model/data (mjv_updateScene), so it does not touch the physics state or the env's own observation rendering.

Per frame: close, in_, bowl_px_agent, bowl_px_wrist (bowl pixels at the camera's native 128 x 128).
  label close              = Close(cabinet bottom)
  label bowl_outside_vis   = (not In) and bowl visible (pixels >= VIS_MIN in agentview or wrist)
  undecidable              = (not In) and not visible   (occluded / out of view: excluded from detector training;
                                                          never a success frame)

  python t3_labels.py smoke <unit_id>     # replay one stored unit and print the per-step label summary
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import numpy as np
import mujoco

VIS_MIN = 15                       # bowl pixels (128 x 128) for "visible"; provisional, set with the detailed plan
CAMS = ("agentview", "robot0_eye_in_hand")
BOWL = "akita_black_bowl_1"


class BowlVis:
    def __init__(self, sim, res=128):
        self.m, self.d = sim.model._model, sim.data._data
        self.r = mujoco.Renderer(self.m, res, res)
        self.r.enable_segmentation_rendering()
        self.geoms = np.array([g for g in range(self.m.ngeom)
                               if mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[g]).startswith(BOWL)])

    def pixels(self):
        out = []
        for cam in CAMS:
            self.r.update_scene(self.d, camera=cam)
            seg = self.r.render()[..., 0]              # geom id per pixel (objtype in [..., 1])
            out.append(int(np.isin(seg, self.geoms).sum()))
        return out

    def close(self):
        self.r.close()


def frame_labels(u, vis):
    prob = u._env.env
    goals = {(" ".join(g)).split()[0]: g for g in prob.parsed_problem["goal_state"]}
    close = bool(prob._eval_predicate(goals["close"])); in_ = bool(prob._eval_predicate(goals["in"]))
    pa, pw = vis.pixels()
    return dict(close=close, in_=in_, bowl_px_agent=pa, bowl_px_wrist=pw)


def smoke(uid):
    _wmvp.bootstrap()
    _wmvp.bootstrap()
    from libero_env_adapter import _make_env, _anchor
    ROOT = _wmvp.P("$WMV_ROOT")
    fam = "familyA" if "_fa" in uid else "familyP"
    run = "lingbot_famA" if fam == "familyA" else "lingbot_famP"
    r = next(json.loads(l) for l in open(f"{ROOT}/libero/out/runs/{run}/records_t3.jsonl") if uid in l)
    a = _anchor(f"{ROOT}/libero/out/bank/{fam}/{fam}_anchors.json", r["anchor_id"])
    acts = np.load(f"{ROOT}/{r['action_file']}")["env"]
    env = _make_env(a, H=len(acts)); u = env.envs[0]
    vis = BowlVis(u._env.sim)
    L = [frame_labels(u, vis)]
    for x in acts:
        u.step(np.asarray(x, np.float32)); L.append(frame_labels(u, vis))
    vis.close(); env.close()
    for t in range(0, len(L), 40):
        print(t, L[t])
    print("final", L[-1])


if __name__ == "__main__":
    if sys.argv[1] == "smoke":
        smoke(sys.argv[2])
