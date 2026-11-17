"""OpenWAM T2 open-loop failure triage (ruling 2026-11-01, before the calibration-layer downgrade): is the imagined
plan wrong from step 0, or reasonable early and drifting later?

For 3 smoke units of init state 1 (where the closed loop succeeded 10/10), replay BOTH recorded action sequences
(open-loop imagined actions; closed-loop executed actions) in a fresh env, recording per step: end-effector position,
the stove knob joint qpos, and the moka-pot position. Reads only the smoke npz files; no model, no GPU generation.

  python diag_ow_t2.py   (lerobot_libero interpreter)  -> out/task2/diag_ow/{report.json, diag.png}
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
_wmvp.bootstrap()
OUT = f"{ROOT}/libero/out/task2/diag_ow"
SMOKE = f"{ROOT}/libero/out/task2/smoke"
ANCHOR = dict(anchor_id="libero10_t2_s1_smoke", suite="libero_10", task_id=2, init_index=1,
              seed=20261102, settle_steps=10, resolution=128)
UNITS = [f"openwam_t2_smoke_i1__k{k}" for k in range(3)]


def trace_job(args):
    """subprocess worker: one env per process (a second EGL context in one process crashes at teardown)."""
    uid, key = args
    acts = np.load(f"{SMOKE}/actions/{uid}.npz")[key]
    return uid, key, trace(acts)


def trace(acts):
    import mujoco
    from libero_env_adapter import _make_env
    env = _make_env(ANCHOR, H=len(acts)); u = env.envs[0]
    sim = u._env.sim; m = sim.model._model; d = sim.data._data
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
    assert sid >= 0
    p = u._env.env
    stove = p.get_object("flat_stove_1")
    knob_adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in stove.joints]
    pot_bid = p.obj_body_id["moka_pot_1"]
    eef, knob, pot, succ = [], [], [], []
    for a in np.asarray(acts, np.float32):
        u.step(a)
        eef.append(d.site_xpos[sid].copy()); knob.append(float(max(d.qpos[q] for q in knob_adr)))
        pot.append(d.xpos[pot_bid].copy()); succ.append(bool(u._env.check_success()))
    try:
        env.close()
    except Exception as e:                                       # EGL context teardown noise (robosuite egl_context.free)
        print("env.close:", type(e).__name__, flush=True)
    return np.array(eef), np.array(knob), np.array(pot), np.array(succ)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True)
    with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        T = {(u, k): t for u, k, t in pool.imap(trace_job, [(u, k) for u in UNITS for k in ("env", "closed_loop")])}
    fig, axes = plt.subplots(len(UNITS), 3, figsize=(15, 3.2 * len(UNITS)))
    rep = {}
    for i, uid in enumerate(UNITS):
        z = np.load(f"{SMOKE}/actions/{uid}.npz")
        A_open, A_cl = z["env"], z["closed_loop"]
        eo, ko, po, so = T[(uid, "env")]
        ec, kc, pc, sc = T[(uid, "closed_loop")]
        n = min(len(eo), len(ec)); dist = np.linalg.norm(eo[:n] - ec[:n], axis=1)
        over = np.where(dist > 0.05)[0]
        rep[uid] = dict(
            closed_replay_success=bool(sc.any()), open_replay_success=bool(so.any()),
            eef_dist_mean_first50=float(dist[:50].mean()), eef_dist_max_first50=float(dist[:50].max()),
            first_step_dist_gt_5cm=int(over[0]) + 1 if len(over) else None,
            knob_max_open=float(ko.max()), knob_max_closed=float(kc.max()),
            action_l2_first50=float(np.linalg.norm(A_open[:50] - A_cl[:50], axis=1).mean()))
        ax = axes[i]
        ax[0].plot(dist); ax[0].axhline(0.05, color="r", ls="--"); ax[0].set_title(f"{uid}\n|eef open - eef closed| (m)")
        ax[1].plot(ko, label="open"); ax[1].plot(kc, label="closed"); ax[1].axhline(0.5, color="r", ls="--")
        ax[1].set_title("stove knob qpos (>= 0.5 = on)"); ax[1].legend()
        ax[2].plot(po[:, 0], po[:, 1], label="pot open"); ax[2].plot(pc[:, 0], pc[:, 1], label="pot closed")
        ax[2].plot(*pc[0, :2], "ko"); ax[2].set_title("moka pot xy"); ax[2].legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/diag.png", dpi=110)
    json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1)
    early_ok = [u for u, r in rep.items() if r["eef_dist_mean_first50"] < 0.03]
    print(json.dumps(rep, indent=1))
    print("DIAG_DONE early_close_first50:", len(early_ok), "/", len(UNITS),
          "-> plan-offline-drift" if len(early_ok) == len(UNITS) else "-> check chaining/controller on T2")


if __name__ == "__main__":
    main()
