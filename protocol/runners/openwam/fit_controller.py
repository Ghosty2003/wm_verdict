"""Calibrated OSC controller model for OpenWAM-alpha chaining (2026-10-23 ruling: option 'calibrated'; an INJECTED
component -- declared in table footnotes like WorldVLA's constrained retry).

Data: REAL truth replays only -- the 100 lingbot_base units (main bank anchors, stored post-processed LIBERO 7-D actions),
bit-exact replays, per step the achieved end-effector state (robot0_eef_pos, robot0_eef_quat, gripper qpos).
Split by unit (seed 20261036): 70 fit / 30 held out.
Model (per axis, least squares on the fit units):
  position   dp_t = a * u_t + b * u_{t-1} + c * dp_{t-1}          (u = commanded xyz in [-1, 1])
  rotation   w_t  = a * r_t + b * r_{t-1} + c * w_{t-1}           (w = world-frame rotvec of R_{t+1} R_t^T, r = commanded rotvec)
  gripper    g_{t+1} = g_t + k * (cmd_open_t - g_t)                (g = open-scale from qpos, cmd_open = -LIBERO gripper cmd)
Held-out validation: from the TRUE state at every 32-step boundary, roll 32 steps with the TRUE commands; error at the
block end, compared with naive OSC integration (0.05 m / 0.5 rad per unit command) and with 'reuse' (no update).

  python fit_controller.py collect      (lerobot_libero; GPU for rendering)  -> controller/data.npz
  python fit_controller.py fit          (any numpy/scipy)                    -> controller/params.json, validation.json
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import hashlib
import numpy as np

ROOT = _wmvp.P("$WMV_ROOT")
OUT = f"{os.path.dirname(os.path.abspath(__file__))}/controller"
SPLIT_SEED = 20261036
BLOCK = 32
OSC_POS, OSC_ROT, GRIP_W = 0.05, 0.5, 0.08


def collect():
    _wmvp.bootstrap()
    from libero_env_adapter import _make_env, _anchor
    os.makedirs(OUT, exist_ok=True)
    U, P, Q, G, ID = [], [], [], [], []
    for t in ("t5", "t3"):
        for l in open(f"{ROOT}/libero/out/runs/lingbot_base/records_{t}.jsonl"):
            r = json.loads(l)
            a = _anchor(f"{ROOT}/libero/out/bank/libero_anchors.json", r.get("anchor_id") or r["s0_id"])   # lingbot_base records use s0_id
            acts = np.load(f"{ROOT}/{r['action_file']}")["env"]
            env = _make_env(a, H=len(acts)); u = env.envs[0]

            def st():
                o = u._env.env._get_observations(force_update=True)
                return o["robot0_eef_pos"].copy(), o["robot0_eef_quat"].copy(), o["robot0_gripper_qpos"].copy()
            p, q, g = zip(*([st()] + [(u.step(np.asarray(x, np.float32)), st())[1] for x in acts]))
            env.close()
            U.append(acts); P.append(np.stack(p)); Q.append(np.stack(q)); G.append(np.stack(g)); ID.append(r["unit_id"])
            print(r["unit_id"], flush=True)
    np.savez_compressed(f"{OUT}/data.npz", U=np.stack(U), P=np.stack(P), Q=np.stack(Q), G=np.stack(G), unit=np.array(ID))


def _rot(q):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_quat(q)                               # robosuite eef quat is xyzw


def _features(U, P, Q, G):
    from scipy.spatial.transform import Rotation as Rot
    dp = np.diff(P, axis=1)                               # (N, T, 3)
    R = [_rot(q) for q in Q]
    w = np.stack([(r[1:] * r[:-1].inv()).as_rotvec() for r in R])          # world-frame increments
    g = np.clip(2 * (G[..., 0] - G[..., 1]) / GRIP_W - 1, -1, 1)
    return dp, w, g


def _fit_arx(y, u):
    """y_t = a u_t + b u_{t-1} + c y_{t-1}, per axis; y, u: (N, T, 3)."""
    out = []
    for k in range(3):
        X = np.stack([u[:, 1:, k], u[:, :-1, k], y[:, :-1, k]], -1).reshape(-1, 3)
        coef, *_ = np.linalg.lstsq(X, y[:, 1:, k].reshape(-1), rcond=None)
        out.append(coef.tolist())
    return out


def _roll(params, state, u_seq, prev):
    """roll the model 32 steps: state = (p, R, g); prev = (u_xyz_prev, dp_prev, u_rot_prev, w_prev)."""
    from scipy.spatial.transform import Rotation as Rot
    p, R, g = state[0].copy(), state[1], state[2]
    ux0, dp0, ur0, w0 = prev
    for u in u_seq:
        ux, ur, cg = u[:3], u[3:6], -u[6]
        dp = np.array([params["pos"][k][0] * ux[k] + params["pos"][k][1] * ux0[k] + params["pos"][k][2] * dp0[k] for k in range(3)])
        w = np.array([params["rot"][k][0] * ur[k] + params["rot"][k][1] * ur0[k] + params["rot"][k][2] * w0[k] for k in range(3)])
        p = p + dp; R = Rot.from_rotvec(w) * R; g = g + params["grip_k"] * (cg - g)
        ux0, dp0, ur0, w0 = ux, dp, ur, w
    return p, R, g


def fit():
    from scipy.spatial.transform import Rotation as Rot
    z = np.load(f"{OUT}/data.npz"); U, P, Q, G = z["U"], z["P"], z["Q"], z["G"]
    dp, w, g = _features(U, P, Q, G)
    perm = np.random.RandomState(SPLIT_SEED).permutation(len(U)); tr, va = perm[:70], perm[70:]
    pos = _fit_arx(dp[tr], U[tr][..., :3]); rot = _fit_arx(w[tr], U[tr][..., 3:6])
    cg = -U[tr][..., 6]; x = (cg - g[tr][:, :-1]).reshape(-1); y = (g[tr][:, 1:] - g[tr][:, :-1]).reshape(-1)
    k = float(np.dot(x, y) / np.dot(x, x))
    params = dict(model="ARX(1,1) per axis for position and world-frame rotation increments; first-order gripper",
                  pos=pos, rot=rot, grip_k=k, fit_units=int(len(tr)), split_seed=SPLIT_SEED,
                  data_sha256=hashlib.sha256(open(f"{OUT}/data.npz", "rb").read()).hexdigest())
    json.dump(params, open(f"{OUT}/params.json", "w"), indent=1)
    errs = {m: dict(pos=[], rot=[], grip=[]) for m in ("calibrated", "naive", "reuse")}
    for i in va:
        R_true = _rot(Q[i])
        for s in range(0, U.shape[1] - BLOCK + 1, BLOCK):
            st = (P[i, s], R_true[s], g[i, s]); us = U[i, s:s + BLOCK]
            prev = (U[i, s - 1, :3] if s else np.zeros(3), dp[i, s - 1] if s else np.zeros(3),
                    U[i, s - 1, 3:6] if s else np.zeros(3), w[i, s - 1] if s else np.zeros(3))
            preds = dict(calibrated=_roll(params, st, us, prev), reuse=st)
            p_n, R_n, g_n = P[i, s].copy(), R_true[s], g[i, s]
            for u in us:
                p_n = p_n + OSC_POS * np.clip(u[:3], -1, 1); R_n = Rot.from_rotvec(OSC_ROT * np.clip(u[3:6], -1, 1)) * R_n; g_n = -u[6]
            preds["naive"] = (p_n, R_n, g_n)
            for m, (pp, RR, gg) in preds.items():
                errs[m]["pos"].append(float(np.linalg.norm(pp - P[i, s + BLOCK])))
                errs[m]["rot"].append(float(np.degrees((RR * R_true[s + BLOCK].inv()).magnitude())))
                errs[m]["grip"].append(float(abs(gg - g[i, s + BLOCK])))
    val = {m: {k_: dict(median=round(float(np.median(v)), 4), p90=round(float(np.quantile(v, 0.9)), 4)) for k_, v in e.items()}
           for m, e in errs.items()}
    json.dump(dict(held_out_units=int(len(va)), block=BLOCK, error_at_block_end=val, units="pos m, rot deg, grip open-scale"),
              open(f"{OUT}/validation.json", "w"), indent=1)
    print(json.dumps(params)); print(json.dumps(val, indent=1))


if __name__ == "__main__":
    {"collect": collect, "fit": fit}[sys.argv[1]]()
