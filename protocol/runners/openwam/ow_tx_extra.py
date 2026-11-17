"""OpenWAM-alpha follow-ups (2026-11-02): (1) the T2 closed-loop REFERENCE column kept by the downgrade ruling
(5 frozen T2 anchors x 10 seeds, closed loop x2, conclusion consistency); (2) T3 (#4) and T4 (#6) onboarding smokes
(1 anchor x 10 seeds, open-loop imagination + truth replay AND closed loop) to decide per task whether OpenWAM enters
(the T2 downgrade was per-task).

  python ow_tx_extra.py t2closed | smoke4 | smoke6   (openwam interpreter, GPU)
    -> out/runs/openwam_t2/closed.jsonl ; out/task2/smoke/openwam_t{4,6}.jsonl
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

ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
PROMPT = {2: "turn on the stove and put the moka pot on it",
          4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
          6: "put the white mug on the plate and put the chocolate pudding to the right of the plate"}
SEED_MASTER = {2: 20261104, 4: 20261111, 6: 20261112}            # same masters as the LingBot runs (shared ruler)


def main():
    mode = sys.argv[1]
    import ow_runner as W
    W._setup()
    eng, pre, arch, t_load, ckname = W._engine()
    sim = W.Sim(f"/dev/shm/ow_tx_{os.getpid()}")
    if mode == "t2closed":
        bank = json.load(open(f"{ROOT}/libero/out/bank/libero_anchors_t2.json"))["anchors"]
        out = f"{ROOT}/libero/out/runs/openwam_t2"; os.makedirs(out, exist_ok=True)
        path = f"{out}/closed.jsonl"
        done = {json.loads(l)["unit_id"] for l in open(path)} if os.path.exists(path) else set()
        seeds = np.random.RandomState(SEED_MASTER[2]).randint(0, 2 ** 31 - 1, size=(16, 10))
        for ai, a in enumerate(bank):
            a = dict(a, resolution=256)
            for k in range(10):
                uid = f"openwam_{a['anchor_id']}__k{k}"
                if uid in done:
                    continue
                t0 = time.time(); seed = int(seeds[ai, k])
                ca, cs, cS = W.closed_loop(eng, sim, a, PROMPT[2], seed)
                ca2, cs2, _ = W.closed_loop(eng, sim, a, PROMPT[2], seed)
                s1, s2 = bool(np.asarray(cs, bool).any()), bool(np.asarray(cs2, bool).any())
                row = dict(unit_id=uid, model="openwam_alpha_sim_libero", task_id=2, anchor_id=a["anchor_id"],
                           seed_index=k, imagination_seed=seed, H_env=W.H,
                           closed_loop=dict(success=s1, first_success_step=int(np.argmax(cs)) + 1 if s1 else None,
                                            rerun=dict(success=s2), conclusion_consistent=bool(s1 == s2)),
                           note="reference column only (T2 downgrade ruling 2026-11-01)",
                           seconds=round(time.time() - t0, 1), machine=os.uname().nodename)
                open(path, "a").write(json.dumps(row) + "\n")
                print(uid, s1, s2, row["seconds"], flush=True)
        print("OW_T2_CLOSED_DONE")
    else:
        tid = int(mode[-1])
        out = f"{ROOT}/libero/out/task2/smoke"; os.makedirs(out, exist_ok=True)
        path = f"{out}/openwam_t{tid}.jsonl"
        done = {json.loads(l)["unit_id"] for l in open(path)} if os.path.exists(path) else set()
        anchor = dict(anchor_id=f"libero10_t{tid}_s0_smoke", suite="libero_10", task_id=tid, init_index=0,
                      seed=20261102, settle_steps=10, resolution=256)
        seeds = np.random.RandomState(20261102).randint(0, 2 ** 31 - 1, 10)
        for k in range(10):
            uid = f"openwam_t{tid}_smoke__k{k}"
            if uid in done:
                continue
            t0 = time.time()
            o = sim.reset(anchor)
            frames, env7, t_calls = W.imagine(eng, o, PROMPT[tid], int(seeds[k]))
            r = sim.replay(anchor, env7)
            succ = np.asarray(r["success"], bool)
            ca, cs, _ = W.closed_loop(eng, sim, anchor, PROMPT[tid], int(seeds[k]))
            cs = np.asarray(cs, bool)
            rec = dict(unit_id=uid, task_id=tid, anchor=anchor, prompt=PROMPT[tid], imagination_seed=int(seeds[k]),
                       H=W.H, real_success=bool(succ.any()),
                       real_first_success_step=int(np.argmax(succ)) + 1 if succ.any() else None,
                       closed_success=bool(cs.any()),
                       closed_first_success_step=int(np.argmax(cs)) + 1 if cs.any() else None,
                       seconds=round(time.time() - t0, 1), machine=os.uname().nodename)
            open(path, "a").write(json.dumps(rec) + "\n")
            print(uid, "open", rec["real_success"], "closed", rec["closed_success"], rec["seconds"], flush=True)
        R = [json.loads(l) for l in open(path)]
        print(f"OW_SMOKE_DONE t{tid} open", sum(r["real_success"] for r in R), "/", len(R),
              "closed", sum(r["closed_success"] for r in R), "/", len(R))
    sim.close()


if __name__ == "__main__":
    main()
