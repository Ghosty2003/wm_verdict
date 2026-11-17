"""T2 truth-replay CPU pool ($NODE_CLOUD, 2026-11-01): replays each unit's imagined actions TWICE in fresh envs (bit
compare on full success traces + final physics state) and writes truth.{shard}.jsonl rows that the merge joins to
the imagination records (truth_pending=True) by unit_id. One env per subprocess; workers spread over the EGL devices.

  WMV_ROOT=... WMV_RESULTS=... python t2_replay.py <nworkers>
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import time
import numpy as np

ROOT = os.environ.get("WMV_ROOT", _wmvp.P("$WMV_ROOT"))
RES = os.environ.get("WMV_RESULTS", f"{ROOT}/libero/out/runs/lingbot_t2")
_wmvp.bootstrap()
FAMS = os.environ.get("WMV_FAMS", "base,famO,famP,famA").split(",")
if os.environ.get("WMV_BANKS"):
    BANKS = json.loads(os.environ["WMV_BANKS"])                  # generic tasks (T3/T4 onboarding)
else:
    from lingbot_t2_cloud import BANKS  # noqa: E402


def one(job):
    fam, uid, anchor, n = job
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(n % int(os.environ.get("WMV_EGL_N", "1")))   # $NODE_CLOUD used 4 GPUs; cluster nodes have 1
    import mujoco
    from libero_env_adapter import _make_env
    acts = np.load(f"{RES}/{fam}/actions/{uid}.npz")["env"]

    def replay():
        env = _make_env(anchor, H=len(acts)); u = env.envs[0]
        succ, first = [], None
        for t, a in enumerate(acts):
            u.step(np.asarray(a, np.float32))
            s = bool(u._env.check_success()); succ.append(s)
            if s and first is None:
                first = t + 1
        sim = u._env.sim; m, d = sim.model._model, sim.data._data
        st = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        try:
            env.close()
        except Exception:
            pass
        return np.array(succ), first, st

    t0 = time.time()
    s1, f1, st1 = replay(); s2, f2, st2 = replay()
    same = bool(np.array_equal(s1, s2) and np.array_equal(st1, st2))
    return fam, dict(unit_id=uid, real_verdict=bool(s1.any()), real_first_success_step=f1,
                     replay_check_identical=same, replays=2, seconds=round(time.time() - t0, 1),
                     machine=os.uname().nodename)


def anchors_by_uid(fam):
    bank = json.load(open(f"{ROOT}/{BANKS[fam]}"))
    for a in bank["anchors"]:
        if a.get("bddl_file"):
            a["bddl_file"] = os.path.join(ROOT, a["bddl_file"])
    return {a["anchor_id"]: a for a in bank["anchors"]}


def main():
    import multiprocessing as mp
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    n_done = 0
    with mp.get_context("spawn").Pool(nw, maxtasksperchild=1) as pool:
        while True:
            jobs = []
            for fam in FAMS:
                out = f"{RES}/{fam}"
                if not os.path.isdir(out):
                    continue
                done = set()
                for f in os.listdir(out):
                    if f.startswith("truth."):
                        done |= {json.loads(l)["unit_id"] for l in open(f"{out}/{f}")}
                try:
                    A = anchors_by_uid(fam)
                except FileNotFoundError:
                    continue
                for f in os.listdir(out):
                    if not f.startswith("records."):
                        continue
                    for l in open(f"{out}/{f}"):
                        r = json.loads(l)
                        if r["unit_id"] in done or not os.path.exists(f"{out}/actions/{r['unit_id']}.npz"):
                            continue
                        jobs.append((fam, r["unit_id"], A[r["anchor_id"]], len(jobs)))
                        done.add(r["unit_id"])
            if not jobs:
                drivers = [f for f in os.listdir(RES) if f.startswith("driver.")]
                if len(drivers) >= 4:
                    break
                time.sleep(60); continue
            for fam, rec in pool.imap_unordered(one, jobs):
                open(f"{RES}/{fam}/truth.0.jsonl", "a").write(json.dumps(rec) + "\n")
                n_done += 1
                if n_done % 20 == 0:
                    print("truth done", n_done, time.strftime("%H:%M"), flush=True)
    print("REPLAY_POOL_DONE", n_done)


if __name__ == "__main__":
    main()
