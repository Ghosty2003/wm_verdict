"""T2 (LIBERO-10 #2, stove + moka pot) base-success smoke, OpenWAM-alpha (2026-11-01): 1 start x 10 imaginations
(13 chained calls to H = 400), open-loop truth replay; success = simulator reach-anytime. Same anchor convention as
the LingBot smoke; prompt = the task's official language.

  python smoke_openwam.py [init_index]   (openwam interpreter, GPU)  -> out/task2/smoke/openwam_t2.jsonl

Round 2 (user, 2026-11-01, after 0/10 open-loop on init 0): a second official init state, and ALSO one closed-loop
run per unit (the model's normal deployment: real frame every 32 steps), to separate "cannot do T2" from
"open-loop brittleness". Unit ids carry the init index (i<idx>); init 0 rows keep their original ids.
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
OUT = f"{ROOT}/libero/out/task2/smoke"
PROMPT = "turn on the stove and put the moka pot on it"
ANCHOR = dict(anchor_id="libero10_t2_s0_smoke", suite="libero_10", task_id=2, init_index=0,
              seed=20261102, settle_steps=10, resolution=256)
K = 10


def main():
    import ow_runner as W
    init = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    anchor = dict(ANCHOR, init_index=init, anchor_id=f"libero10_t2_s{init}_smoke")
    tag = "" if init == 0 else f"_i{init}"
    os.makedirs(f"{OUT}/actions", exist_ok=True)
    path = f"{OUT}/openwam_t2.jsonl"
    done = {json.loads(l)["unit_id"] for l in open(path)} if os.path.exists(path) else set()
    W._setup()
    eng, pre, arch, t_load, ckname = W._engine()
    sim = W.Sim(f"/dev/shm/ow_t2smoke_{os.getpid()}")
    seeds = np.random.RandomState(20261102).randint(0, 2 ** 31 - 1, K)
    for k in range(K):
        uid = f"openwam_t2_smoke{tag}__k{k}"
        if uid in done:
            continue
        t0 = time.time()
        o = sim.reset(anchor)
        frames, env7, t_calls = W.imagine(eng, o, PROMPT, int(seeds[k]))
        r = sim.replay(anchor, env7)
        succ = np.asarray(r["success"], bool)
        ca, cs, cS = W.closed_loop(eng, sim, anchor, PROMPT, int(seeds[k]))
        cs = np.asarray(cs, bool)
        np.savez_compressed(f"{OUT}/actions/{uid}.npz", env=np.asarray(env7, np.float32), closed_loop=np.asarray(ca, np.float32))
        rec = dict(unit_id=uid, model="openwam_alpha_sim_libero", task_id=2, anchor=anchor, prompt=PROMPT,
                   imagination_seed=int(seeds[k]), H=W.H, checkpoint=ckname,
                   real_success=bool(succ.any()), real_first_success_step=int(np.argmax(succ)) + 1 if succ.any() else None,
                   closed_success=bool(cs.any()), closed_first_success_step=int(np.argmax(cs)) + 1 if cs.any() else None,
                   seconds=round(time.time() - t0, 1), machine=os.uname().nodename)
        open(path, "a").write(json.dumps(rec) + "\n")
        print(uid, "open", rec["real_success"], rec["real_first_success_step"], "closed", rec["closed_success"],
              rec["closed_first_success_step"], rec["seconds"], flush=True)
    sim.close()
    R = [json.loads(l) for l in open(path)]
    for t_, G in (("i0", [r for r in R if "_i" not in r["unit_id"]]), (f"i{init}", [r for r in R if tag and tag in r["unit_id"]])):
        if G:
            print(f"SMOKE_DONE openwam t2 [{t_}] open", sum(r["real_success"] for r in G), "/", len(G),
                  "closed", sum(bool(r.get("closed_success")) for r in G), "/", sum("closed_success" in r for r in G))


if __name__ == "__main__":
    main()
