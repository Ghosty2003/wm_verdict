"""F 失败案例：从存档 full-physics states 直接读物体量（mj_setState + mj_forward，不重放、不调模型）。
T1 F160：底层抽屉关节开度（逐 8 步）+ close 子目标；T3 F60：白瓷杯离桌高度、杯-夹爪距离（逐 4 步，0–160）。同格成功与失败单元都算。
→ out/f_family/cases/state_probe.json"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import glob, json, sys, numpy as np
_wmvp.bootstrap()
import f_cases as F
_wmvp.bootstrap()
import mujoco
from libero_env_adapter import _make_env
from ow_famF import base_units
out = {}
for T, tid, lv in F.CELLS:
    A = {r["unit_id"]: a for r, a, _ in base_units(tid)}
    rows = sorted([json.loads(l) for f in glob.glob(f"{F.RUNS}/openwam_famF/closed_t{tid}.*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["unit_id"])
    rows = [r for r in rows if r["setting_id"] == lv]
    envs = {}
    for r in rows:
        aid = r["anchor_id"]
        if aid not in envs:
            envs[aid] = _make_env(dict(A[r["base_unit"]], resolution=256), H=500)
        e = envs[aid].envs[0]; prob = e._env.env; m, d = e._env.sim.model._model, e._env.sim.data._data
        S = np.load(f"{F.ROOT}/{r['closed_loop_file']}")["states"]
        rec = dict(success=bool(r["closed_loop"]["success"]), first_success=r["closed_loop"]["first_success_step"])
        if T == "T1":
            jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
            jj = [j for j, n in enumerate(jn) if n and "cabinet" in n and "bottom" in n]
            assert len(jj) == 1, [n for n in jn if n and "cabinet" in n]
            adr = m.jnt_qposadr[jj[0]]; rec["joint"] = jn[jj[0]]
            goal = [g for g in prob.parsed_problem["goal_state"] if g[0] == "close"][0]
            q, c = [], []
            for k in range(7, 400, 8):
                mujoco.mj_setState(m, d, S[k], mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(m, d)
                q.append(round(float(d.qpos[adr]), 4)); c.append(bool(prob._eval_predicate(goal)))
            rec.update(drawer_q_every8=q, close_every8="".join("1" if x else "." for x in c), q400=q[-1])
        else:
            bid = prob.obj_body_id["porcelain_mug_1"]; eid = prob.robots[0].eef_site_id; eid = eid["right"] if isinstance(eid, dict) else eid
            mujoco.mj_setState(m, d, S[0], mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(m, d); z0 = None
            hz, gd, up = [], [], []
            for k in range(0, 164, 4):
                mujoco.mj_setState(m, d, S[k], mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(m, d)
                p = d.xpos[bid].copy(); z0 = p[2] if z0 is None else z0
                hz.append(round(float(p[2] - z0), 4)); gd.append(round(float(np.linalg.norm(d.site_xpos[eid] - p)), 4))
                up.append(round(float(d.xmat[bid].reshape(3, 3)[2, 2]), 3))          # 杯身 z 轴与竖直方向夹角余弦：1 = 直立
            rec.update(steps=list(range(0, 164, 4)), mug_dz=hz, grip_mug_dist=gd, mug_upright_cos=up)
        out[r["unit_id"]] = dict(T=T, **rec)
        print("PROBE", r["unit_id"], rec["success"], rec.get("q400", ""), flush=True)
    for v in envs.values():
        v.close()
json.dump(out, open(f"{F.OUT}/state_probe.json", "w"), indent=1)
print("PROBE_DONE", len(out))
