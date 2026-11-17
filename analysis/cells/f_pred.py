"""F 失败单元逐步子目标（LIBERO 自带 _eval_predicate；只用存档 commanded 动作重放，不调模型）。
→ out/f_family/cases/predicates.json：每单元每个子目标 是否曾为真 / 首次为真步 / 第 400 步真值，以及逐 8 步真值串；重放 state 与存档逐位相同才计入。"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import json, sys, numpy as np
_wmvp.bootstrap()
import f_cases as F
_wmvp.bootstrap()
import mujoco
from libero_env_adapter import _make_env
from ow_famF import base_units
out = {}
for u in F.failures():
    anc = {r["unit_id"]: a for r, a, _ in base_units(u["tid"])}[u["base_unit"]]
    z = np.load(u["npz"]); acts, S0 = z["commanded"], z["states"]
    env = _make_env(dict(anc, resolution=256), H=F.H + 100); e = env.envs[0]; prob = e._env.env
    m, d = e._env.sim.model._model, e._env.sim.data._data; st = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    goals = prob.parsed_problem["goal_state"]; sub = [[bool(prob._eval_predicate(g)) for g in goals]]; ok = True
    for k, a in enumerate(acts):
        e.step(np.asarray(a, np.float32)); mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        ok = ok and bool(np.array_equal(st, S0[k])); sub.append([bool(prob._eval_predicate(g)) for g in goals])
    env.close(); sub = np.array(sub)
    G = {}
    for gi, g in enumerate(goals):
        col = sub[:, gi]; G[" ".join(g)] = dict(ever=bool(col.any()), first=int(np.argmax(col)) if col.any() else None,
                                              at400=bool(col[-1]), every8="".join("1" if x else "." for x in col[::8]))
    out[u["uid"]] = dict(T=u["T"], replay_state_identical=ok, all_true_ever=bool(sub.all(1).any()), goals=G)
    print("PRED", u["uid"], ok, {k: (v["ever"], v["at400"]) for k, v in G.items()}, flush=True)
json.dump(out, open(f"{F.OUT}/predicates.json", "w"), indent=1)
print("PRED_DONE", len(out))
