"""T2 (LIBERO-10 #2, stove + moka pot) anchor bank: freeze 5 anchors and verify bit-exact restore + replay, the exact
recipe and checks of anchor_bank.py (A fresh-vs-fresh, C after another anchor's episode; tolerance 0; 120-step fixed
action sequence; anchors must not start successful). Separate bank file; master seed 20261103 (2026-11-01 prep).

  python anchor_bank_t2.py   -> out/bank/libero_anchors_t2.json, out/bank/anchor_verify_t2.json
"""
import os
import json
import hashlib
import numpy as np

import anchor_bank as B
from libero.libero import benchmark

OUT = B.OUT
TID, KIND = 2, "multi-stage"      # KITCHEN_SCENE3 turn on the stove + put the moka pot on it
N = 5
MASTER = 20261103


def main():
    os.makedirs(OUT, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[B.SUITE]()
    rs = np.random.RandomState(MASTER)
    idx = rs.choice(50, N, replace=False); seeds = rs.randint(0, 2 ** 31 - 1, size=N)
    anchors = [dict(anchor_id=f"libero10_t{TID}_a{j}", suite=B.SUITE, task_id=TID, task=suite.get_task(TID).name,
                    stage_kind=KIND, init_index=int(idx[j]), seed=int(seeds[j]), settle_steps=B.SETTLE,
                    resolution=B.RES) for j in range(N)]
    ars = np.random.RandomState(MASTER + 1)
    actions = np.clip(np.cumsum(ars.uniform(-0.15, 0.15, size=(B.R_STEPS, 7)), axis=0), -0.6, 0.6).astype(np.float32)
    actions[:, 6] = np.where(np.arange(B.R_STEPS) % 60 < 30, -1.0, 1.0)
    report, hashes, prev = [], {}, None
    for a in anchors:
        SA = B.run(suite, a, actions)
        SB = B.run(suite, a, actions)
        dAB = B.maxdiff(SA, SB); ok = all(v == 0.0 for v in dAB.values())
        dD = None
        if prev is not None:
            _, envP = B.run(suite, prev, actions, keep_open=True)
            SD = B.run(suite, a, actions); envP.close()
            dD = B.maxdiff(SA, SD); ok = ok and all(v == 0.0 for v in dD.values())
        ok = ok and not bool(SA[0]["succ"][0])
        hashes[a["anchor_id"]] = hashlib.sha256(b"".join(np.ascontiguousarray(s[k]).tobytes()
                                                         for s in SA for k in sorted(s))).hexdigest()
        report.append(dict(anchor_id=a["anchor_id"], pass_=ok, fresh_vs_fresh=dAB, after_other_anchor=dD,
                           steps=B.R_STEPS, init_success=bool(SA[0]["succ"][0]), trajectory_sha256=hashes[a["anchor_id"]]))
        print(a["anchor_id"], "PASS" if ok else "FAIL", {k: v for k, v in dAB.items() if v},
              None if dD is None else {k: v for k, v in dD.items() if v}, hashes[a["anchor_id"]][:12], flush=True)
        prev = a
    bank = dict(suite=B.SUITE, master_seed=MASTER, anchors_per_task=N, settle_steps=B.SETTLE, resolution=B.RES,
                restore_recipe="LeRobot LiberoEnv(hard_reset=True, init_states=True): seed -> hard reset -> "
                               "set_init_state(init[init_index]) -> settle_steps x dummy action",
                seed_rule="rs = RandomState(20261103); idx = rs.choice(50, 5, replace=False); "
                          "seeds = rs.randint(0, 2**31-1, 5)",
                anchors=anchors)
    p = f"{OUT}/libero_anchors_t2.json"
    json.dump(bank, open(p, "w"), indent=1)
    json.dump(dict(all_pass=all(r["pass_"] for r in report), tolerance=0.0, replay_steps=B.R_STEPS, results=report,
                   bank_sha256=hashlib.sha256(open(p, "rb").read()).hexdigest()),
              open(f"{OUT}/anchor_verify_t2.json", "w"), indent=1)
    print("ALL PASS" if all(r["pass_"] for r in report) else "SOME FAIL", hashlib.sha256(open(p, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
