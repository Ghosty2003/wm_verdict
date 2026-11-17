"""Generic main anchor bank for a new LIBERO-10 task (T3/T4 onboarding, ruling 2026-11-01): the anchor_bank_t2.py
recipe parameterized by task id and master seed. Checks identical (fresh x2, dirty restore, 120-step fixed actions,
tolerance 0, no anchor initially successful).

  python anchor_bank_tx.py <tid> <master>   -> out/bank/libero_anchors_t<tid>.json + anchor_verify_t<tid>.json
Masters (recorded here): t4 = 20261107, t6 = 20261108.
"""
import os
import sys
import json
import hashlib
import numpy as np

import anchor_bank as B
from libero.libero import benchmark


def main(tid, master):
    OUT = B.OUT; os.makedirs(OUT, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[B.SUITE]()
    rs = np.random.RandomState(master)
    idx = rs.choice(50, 5, replace=False); seeds = rs.randint(0, 2 ** 31 - 1, size=5)
    anchors = [dict(anchor_id=f"libero10_t{tid}_a{j}", suite=B.SUITE, task_id=tid, task=suite.get_task(tid).name,
                    stage_kind="multi-stage", init_index=int(idx[j]), seed=int(seeds[j]), settle_steps=B.SETTLE,
                    resolution=B.RES) for j in range(5)]
    ars = np.random.RandomState(master + 1)
    actions = np.clip(np.cumsum(ars.uniform(-0.15, 0.15, size=(B.R_STEPS, 7)), axis=0), -0.6, 0.6).astype(np.float32)
    actions[:, 6] = np.where(np.arange(B.R_STEPS) % 60 < 30, -1.0, 1.0)
    report, hashes, prev = [], {}, None
    for a in anchors:
        SA = B.run(suite, a, actions); SB = B.run(suite, a, actions)
        dAB = B.maxdiff(SA, SB); ok = all(v == 0.0 for v in dAB.values()); dD = None
        if prev is not None:
            _, envP = B.run(suite, prev, actions, keep_open=True)
            SD = B.run(suite, a, actions); envP.close()
            dD = B.maxdiff(SA, SD); ok = ok and all(v == 0.0 for v in dD.values())
        ok = ok and not bool(SA[0]["succ"][0])
        hashes[a["anchor_id"]] = hashlib.sha256(b"".join(np.ascontiguousarray(s[k]).tobytes()
                                                         for s in SA for k in sorted(s))).hexdigest()
        report.append(dict(anchor_id=a["anchor_id"], pass_=ok, fresh_vs_fresh=dAB, after_other_anchor=dD,
                           steps=B.R_STEPS, init_success=bool(SA[0]["succ"][0]), trajectory_sha256=hashes[a["anchor_id"]]))
        print(a["anchor_id"], "PASS" if ok else "FAIL", flush=True)
        prev = a
    bank = dict(suite=B.SUITE, master_seed=master, anchors_per_task=5, settle_steps=B.SETTLE, resolution=B.RES,
                restore_recipe="LeRobot LiberoEnv(hard_reset=True, init_states=True): seed -> hard reset -> "
                               "set_init_state(init[init_index]) -> settle_steps x dummy action",
                seed_rule=f"rs = RandomState({master}); idx = rs.choice(50, 5, replace=False); seeds = rs.randint(0, 2**31-1, 5)",
                anchors=anchors)
    p = f"{OUT}/libero_anchors_t{tid}.json"; json.dump(bank, open(p, "w"), indent=1)
    allp = all(r["pass_"] for r in report)
    json.dump(dict(all_pass=allp, tolerance=0.0, replay_steps=B.R_STEPS, results=report,
                   bank_sha256=hashlib.sha256(open(p, "rb").read()).hexdigest()),
              open(f"{OUT}/anchor_verify_t{tid}.json", "w"), indent=1)
    print("ALL PASS" if allp else "SOME FAIL", f"t{tid}", hashlib.sha256(open(p, "rb").read()).hexdigest())
    return allp


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]), int(sys.argv[2])) else 1)
