"""T2 (LIBERO-10 #2, stove + moka pot) base-success smoke, LingBot-VA (2026-11-01): 1 start x 10 imaginations,
open-loop truth replay of the imagined actions; success = simulator reach-anytime. Gate: < 20 % -> calibration layer
only (task-expansion proposal). Not a frozen-bank run: the anchor here is init state 0 with a fixed seed, recorded in
every row; the formal bank is frozen later.

  python smoke_lingbot.py   (lerobot_libero interpreter, GPU)  -> out/task2/smoke/lingbot_t2.jsonl
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
ANCHOR = dict(anchor_id="libero10_t2_s0_smoke", suite="libero_10", task_id=2, init_index=0,
              seed=20261102, settle_steps=10, resolution=128)
K = 10


def main():
    from lingbot_unit import LingBotRig, H
    os.makedirs(f"{OUT}/actions", exist_ok=True)
    path = f"{OUT}/lingbot_t2.jsonl"
    done = {json.loads(l)["unit_id"] for l in open(path)} if os.path.exists(path) else set()
    rig = LingBotRig(2)
    seeds = np.random.RandomState(20261102).randint(0, 2 ** 31 - 1, K)
    for k in range(K):
        uid = f"lingbot_t2_smoke__k{k}"
        if uid in done:
            continue
        t0 = time.time()
        img = rig.imagine(ANCHOR, int(seeds[k]))
        r = rig.replay(ANCHOR, img["env_actions"])
        succ = r["succ"]
        np.savez_compressed(f"{OUT}/actions/{uid}.npz", env=img["env_actions"])
        rec = dict(unit_id=uid, model="lingbot_va_libero_long", task_id=2, anchor=ANCHOR, imagination_seed=int(seeds[k]),
                   H=H, real_success_520=bool(succ.any()), real_success_400=bool(succ[:400].any()),
                   real_first_success_step=int(np.argmax(succ)) + 1 if succ.any() else None,
                   seconds=round(time.time() - t0, 1), machine=os.uname().nodename)
        open(path, "a").write(json.dumps(rec) + "\n")
        print(uid, rec["real_success_400"], rec["real_first_success_step"], rec["seconds"], flush=True)
    R = [json.loads(l) for l in open(path)]
    print("SMOKE_DONE lingbot t2", sum(r["real_success_400"] for r in R), "/", len(R), "@400 |",
          sum(r["real_success_520"] for r in R), "/", len(R), "@520")


if __name__ == "__main__":
    main()
