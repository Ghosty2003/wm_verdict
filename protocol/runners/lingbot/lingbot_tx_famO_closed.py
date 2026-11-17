"""famO closed-loop FIX for LingBot T2/T3/T4 (defect found 2026-11-05): lingbot_tx_closed.py ran the famO closed
loops WITHOUT the first-frame occlusion, so all five O levels were bit-identical copies of the base closed run.
Per the 2026-10-26 O-family ruling the ratio-sized black square must be applied to the frame the model reads at
t = 0 of the closed loop as well. This reruns every famO unit's closed loop x2 with that occlusion and writes
famO/closed_occ.<shard>.jsonl; the old famO/closed.*.jsonl stay on disk as the (unoccluded) record of the defect.

  python lingbot_tx_famO_closed.py <tid> <shard> <nshards>   (lerobot_libero, GPU)
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import glob
import time
import zlib
import numpy as np

ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
from lingbot_t2_cloud import occlude, RATIOS  # noqa: E402

H = 400


def main():
    tid, shard, nsh = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    import torch
    import lingbot_unit as LU
    LU.H = H

    class Rig(LU.LingBotRig):
        occ = None                                   # ratio applied to the NEXT make_env observation only

        def make_env(self, anchor):
            env, obs = super().make_env(anchor)
            if self.occ is not None:
                obs = dict(obs); obs["pixels"] = dict(obs["pixels"])
                obs["pixels"]["image"] = occlude(obs["pixels"]["image"], self.occ)
                self.occ = None
            return env, obs

    rig = Rig(tid)
    RUN = f"{ROOT}/libero/out/runs/lingbot_t{tid}"
    bank = json.load(open(f"{ROOT}/libero/out/bank/libero_anchors_t{tid}.json"))
    A = {}
    for a in bank["anchors"]:
        if a.get("bddl_file") and not a["bddl_file"].startswith("/"):
            a["bddl_file"] = os.path.join(ROOT, a["bddl_file"])
        A[a["anchor_id"]] = a
    out = f"{RUN}/famO"
    done = set()
    for f in glob.glob(f"{out}/closed_occ.*.jsonl"):
        done |= {json.loads(l)["unit_id"] for l in open(f)}
    recs = [json.loads(l) for f in glob.glob(f"{out}/records.*.jsonl") for l in open(f)]
    for r in recs:
        uid = r["unit_id"]
        if uid in done or zlib.crc32(uid.encode()) % nsh != shard:
            continue
        L = uid.split("__")[-1]
        ratio = RATIOS[L] if isinstance(RATIOS, dict) else RATIOS[int(L[1]) - 1]
        acts = np.load(f"{out}/actions/{uid}.npz")["env"]
        a = A[r["anchor_id"]]; seed = r["imagination_seed"]; t0 = time.time()
        rig.occ = ratio; c1 = rig.closed_loop(a, seed, acts)
        rig.occ = ratio; c2 = rig.closed_loop(a, seed, acts)
        rig.occ = None
        row = dict(unit_id=uid, setting_id=L, family="famO", task=f"t{tid}", anchor_id=r["anchor_id"],
                   imagination_seed=seed, H_env=H, occlusion_ratio=ratio,
                   applied="closed-loop first frame occluded (fix of 2026-11-05 defect; ruling 2026-10-26)",
                   closed_loop=dict(success=c1["success"], first_success_step=c1["first_success_step"],
                                    steps=int(c1["steps"]), action_dev_l2_mean=float(np.mean(c1["action_dev_l2"])),
                                    rerun=dict(success=c2["success"], first_success_step=c2["first_success_step"]),
                                    conclusion_consistent=bool(c1["success"] == c2["success"])),
                   seconds=round(time.time() - t0, 1), machine=os.uname().nodename,
                   gpu=torch.cuda.get_device_name(0))
        open(f"{out}/closed_occ.{shard}.jsonl", "a").write(json.dumps(row) + "\n")
        print(uid, c1["success"], c2["success"], row["seconds"], flush=True)
    print("FAMO_CLOSED_OCC_DONE", f"t{tid}", shard)


if __name__ == "__main__":
    main()
