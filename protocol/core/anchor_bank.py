"""LIBERO batch: freeze the shared anchor bank and verify bit-exact restore + replay.

Anchor = (suite, task_id, init_state index, seed, settle steps). Restore = a NEW LeRobot LiberoEnv (the same wrapper the
checkpoints are evaluated with) -> reset(seed): hard reset with `seed` -> set_init_state(init[idx]) -> `num_steps_wait`
settle steps with the LIBERO dummy action.
A fresh env per restore is REQUIRED: LIBERO appends to `object_property_initializers` on every hard reset and never
clears it (bddl_base_domain.py:653/674), so each further reset of the same env draws one more np.random.uniform before
placement and the fixture poses (model.body_pos, not part of the init-state vector) drift. Tasks with an Open/Close or
Turnon/Turnoff init predicate are affected (e.g. KITCHEN_SCENE4); a fresh env always has the same list length. The flattened LIBERO state is not enough for bit-exact replay (fixture poses are
model-level and seed-sampled, solver warm start, gripper integrator, controller state, global RNG), so an anchor is a
recipe, not a stored state vector.

Verification per anchor (all must be bit-identical, tolerance 0):
  A  fresh env, restore -> R steps of a fixed action sequence
  B  second fresh env, same
  C  env A reused ("dirtied": it already ran an episode from another anchor), restore again -> same sequence
Compared at every step: full MuJoCo physics state (mj_getState mjSTATE_FULLPHYSICS), fixture/body poses
(model.body_pos / body_quat), gripper integrator, controller goal, success flag, and both camera images.

  python anchor_bank.py            # freeze + verify -> out/bank/libero_anchors.json, out/bank/anchor_verify.json
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import json
import hashlib
import numpy as np
import mujoco

from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv

OUT = _wmvp.P("$WMV_ROOT/libero/out/bank")
SUITE = "libero_10"
TASKS = {3: "multi-stage", 5: "single-stage"}   # KITCHEN_SCENE4 bowl->drawer->close ; STUDY_SCENE1 book->caddy
N_ANCHORS = 5
MASTER_SEED = 20261025
SETTLE = 10
RES = 128                    # LingBot-VA LIBERO eval resolution (docs: --env.observation_height/width=128)
R_STEPS = 120


def make_env(suite, tid, init_idx):
    return LiberoEnv(task_suite=suite, task_id=tid, task_suite_name=SUITE, observation_width=RES,
                     observation_height=RES, init_states=True, episode_index=init_idx, n_envs=1,
                     num_steps_wait=SETTLE, hard_reset=True)


def snapshot(env, obs):
    sim = env._env.sim
    m, d = sim.model._model, sim.data._data
    st = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    rob = env._env.robots[0]
    ctrl = rob.controller
    return dict(
        phys=st.copy(), body_pos=m.body_pos.copy(), body_quat=m.body_quat.copy(),
        warm=d.qacc_warmstart.copy(),
        grip=np.asarray(getattr(rob.gripper, "current_action", np.zeros(1)), float).copy(),
        goal_pos=np.asarray(getattr(ctrl, "goal_pos", np.zeros(3)), float).copy(),
        succ=np.array([env._env.check_success()], float),
        img1=np.asarray(obs["pixels"]["image"]).copy(), img2=np.asarray(obs["pixels"]["image2"]).copy())


def restore(suite, a):
    """fresh env -> reset(seed); returns (env, obs)"""
    env = make_env(suite, a["task_id"], a["init_index"])
    obs, _ = env.reset(seed=a["seed"])
    return env, obs


def run(suite, a, actions, keep_open=False):
    env, obs = restore(suite, a)
    snaps = [snapshot(env, obs)]
    for x in actions:
        obs, _, _, _, _ = env.step(x)
        snaps.append(snapshot(env, obs))
    if keep_open:
        return snaps, env
    env.close()
    return snaps


def maxdiff(S1, S2):
    out = {}
    for k in S1[0]:
        out[k] = float(max(np.max(np.abs(a[k].astype(float) - b[k].astype(float))) if a[k].size else 0.0
                           for a, b in zip(S1, S2)))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[SUITE]()
    rs = np.random.RandomState(MASTER_SEED)
    anchors = []
    for tid, kind in TASKS.items():
        idx = rs.choice(50, N_ANCHORS, replace=False)
        seeds = rs.randint(0, 2 ** 31 - 1, size=N_ANCHORS)
        for j in range(N_ANCHORS):
            anchors.append(dict(anchor_id=f"libero10_t{tid}_a{j}", suite=SUITE, task_id=int(tid),
                                task=suite.get_task(tid).name, stage_kind=kind, init_index=int(idx[j]),
                                seed=int(seeds[j]), settle_steps=SETTLE, resolution=RES))
    # fixed replay action sequence: smooth bounded random actions (independent of any policy)
    ars = np.random.RandomState(MASTER_SEED + 1)
    actions = np.clip(np.cumsum(ars.uniform(-0.15, 0.15, size=(R_STEPS, 7)), axis=0), -0.6, 0.6).astype(np.float32)
    actions[:, 6] = np.where(np.arange(R_STEPS) % 60 < 30, -1.0, 1.0)
    report, hashes = [], {}
    for tid in TASKS:
        A = [a for a in anchors if a["task_id"] == tid]
        prev = None
        for a in A:
            SA = run(suite, a, actions)                        # fresh env
            SB = run(suite, a, actions)                        # second fresh env
            dAB = maxdiff(SA, SB)
            ok = all(v == 0.0 for v in dAB.values())
            dD = None
            if prev is not None:                               # restore right after another anchor's episode
                _, envP = run(suite, prev, actions, keep_open=True)
                SD = run(suite, a, actions)
                envP.close()
                dD = maxdiff(SA, SD)
                ok = ok and all(v == 0.0 for v in dD.values())
            hashes[a["anchor_id"]] = hashlib.sha256(b"".join(np.ascontiguousarray(s[k]).tobytes()
                                                             for s in SA for k in sorted(s))).hexdigest()
            report.append(dict(anchor_id=a["anchor_id"], pass_=ok, fresh_vs_fresh=dAB, after_other_anchor=dD,
                               steps=R_STEPS, init_success=bool(SA[0]["succ"][0]), trajectory_sha256=hashes[a["anchor_id"]]))
            print(a["anchor_id"], "PASS" if ok else "FAIL", {k: v for k, v in dAB.items() if v},
                  None if dD is None else {k: v for k, v in dD.items() if v}, hashes[a["anchor_id"]][:12], flush=True)
            prev = a
    bank = dict(suite=SUITE, master_seed=MASTER_SEED, anchors_per_task=N_ANCHORS, settle_steps=SETTLE, resolution=RES,
                restore_recipe="LeRobot LiberoEnv(hard_reset=True, init_states=True): seed -> hard reset -> "
                               "set_init_state(init[init_index]) -> settle_steps x dummy action",
                seed_rule="rs = RandomState(master); per task in order (3, 5): idx = rs.choice(50, 5, replace=False); "
                          "seeds = rs.randint(0, 2**31-1, 5)",
                anchors=anchors)
    p = f"{OUT}/libero_anchors.json"
    json.dump(bank, open(p, "w"), indent=1)
    vpath = os.environ.get("VERIFY_OUT", f"{OUT}/anchor_verify.json")
    json.dump(dict(all_pass=all(r["pass_"] for r in report), tolerance=0.0, replay_steps=R_STEPS, results=report,
                   bank_sha256=hashlib.sha256(open(p, "rb").read()).hexdigest()),
              open(vpath, "w"), indent=1)
    print("ALL PASS" if all(r["pass_"] for r in report) else "SOME FAIL", hashlib.sha256(open(p, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
