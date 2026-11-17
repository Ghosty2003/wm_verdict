"""Family A graspability margin (ruling 3: 'distance to the base must leave a graspable margin'): kinematic check.

For each family-A anchor: restore (native recipe), then on a COPY of the MuJoCo data solve damped-least-squares IK over
the 7 arm joints for the grip site at (object position + dz) with the CURRENT (top-down) gripper orientation, joint
limits enforced. Reported per anchor and height: final position error (cm), orientation error (deg), and the
minimum joint-limit margin (rad). A target is 'reachable' if position error < 0.5 cm and orientation error < 2 deg.
The scripted OSC reach in familyA_bank.py is NOT used as evidence (it fails even at native A0 positions).

  python familyA_ik.py
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

_wmvp.bootstrap()
from libero.libero import benchmark
from familyA_bank import make, OUT, SUITE


def ik(m, d0, site, qidx, didx, target, R_target, iters=400, lam=0.05):
    d = mujoco.MjData(m); mujoco.mj_copyData(d, m, d0)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
    lo = np.array([m.jnt_range[m.dof_jntid[i]][0] for i in didx]); hi = np.array([m.jnt_range[m.dof_jntid[i]][1] for i in didx])
    jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    for _ in range(iters):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        p = d.site_xpos[sid]; R = d.site_xmat[sid].reshape(3, 3)
        ep = target - p
        Rerr = R_target @ R.T
        er = 0.5 * np.array([Rerr[2, 1] - Rerr[1, 2], Rerr[0, 2] - Rerr[2, 0], Rerr[1, 0] - Rerr[0, 1]])
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        J = np.vstack([jp[:, didx], jr[:, didx]]); e = np.concatenate([ep, er])
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
        d.qpos[qidx] = np.clip(d.qpos[qidx] + dq, lo, hi)
    mujoco.mj_kinematics(m, d)
    p = d.site_xpos[sid]; R = d.site_xmat[sid].reshape(3, 3)
    ang = np.degrees(np.arccos(np.clip((np.trace(R_target @ R.T) - 1) / 2, -1, 1)))
    margin = float(np.min(np.minimum(d.qpos[qidx] - lo, hi - d.qpos[qidx])))
    return float(np.linalg.norm(target - p) * 100), float(ang), margin


def main():
    suite = benchmark.get_benchmark_dict()[SUITE]()
    bank = json.load(open(f"{OUT}/familyA_anchors.json"))
    rows = []
    for a in bank["anchors"]:
        env = make(suite, a); env.reset(seed=a["seed"]); u = env._env
        m, d = u.sim.model._model, u.sim.data._data
        rob = u.robots[0]; qidx = np.array(rob._ref_joint_pos_indexes); didx = np.array(rob._ref_joint_vel_indexes)
        site = rob.gripper.important_sites["grip_site"]
        obj = a["displacement"]["object"]
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{obj}_joint0"); pos = d.qpos[m.jnt_qposadr[jid]:m.jnt_qposadr[jid] + 3].copy()
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site); R0 = d.site_xmat[sid].reshape(3, 3).copy()
        base = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")]
        r = dict(anchor_id=a["anchor_id"], level=a["level"], task_id=a["task_id"], object_xyz=np.round(pos, 4).tolist(),
                 planar_dist_to_base_m=round(float(np.linalg.norm(pos[:2] - base[:2])), 3))
        for name, dz in (("pre_grasp_6cm", 0.06), ("grasp_2cm", 0.02)):
            pe, oe, mg = ik(m, d, site, qidx, didx, pos + np.array([0, 0, dz]), R0)
            r[name] = dict(pos_err_cm=round(pe, 2), ori_err_deg=round(oe, 2), joint_limit_margin_rad=round(mg, 3),
                           reachable=bool(pe < 0.5 and oe < 2.0))
        env.close(); rows.append(r)
        print(json.dumps(r), flush=True)
    json.dump(rows, open(f"{OUT}/familyA_ik.json", "w"), indent=1)


if __name__ == "__main__":
    main()
