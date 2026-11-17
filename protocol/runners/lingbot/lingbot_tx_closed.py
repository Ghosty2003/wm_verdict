"""LingBot closed loops (reference column) for T3/T4 base/famO/famP/famA units (2026-11-03 queue ruling): x2 per
unit with the unit's imagination seed, H = 400, conclusion consistency, L2 deviation vs the imagined actions.

  python lingbot_tx_closed.py <tid> <shard> <nshards>   (lerobot_libero, GPU)
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
from lingbot_tx_run import banks_of  # noqa: E402

FAMS = ["base", "famO", "famP", "famA"]


def main():
    tid, shard, nsh = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    import torch
    import lingbot_unit as LU
    LU.H = 400
    rig = LU.LingBotRig(tid)
    RUN = f"{ROOT}/libero/out/runs/lingbot_t{tid}"
    for fam in FAMS:
        bank = json.load(open(f"{ROOT}/{banks_of(tid)[fam]}"))
        A = {}
        for a in bank["anchors"]:
            if a.get("bddl_file") and not a["bddl_file"].startswith("/"):
                a["bddl_file"] = os.path.join(ROOT, a["bddl_file"])
            A[a["anchor_id"]] = a
        out = f"{RUN}/{fam}"
        done = set()
        for f in os.listdir(out):
            if f.startswith("closed."):
                done |= {json.loads(l)["unit_id"] for l in open(f"{out}/{f}")}
        recs = [json.loads(l) for f in glob.glob(f"{out}/records.*.jsonl") for l in open(f)]
        for r in recs:
            uid = r["unit_id"]
            if uid in done or zlib.crc32(uid.encode()) % nsh != shard:
                continue
            acts = np.load(f"{out}/actions/{uid}.npz")["env"]
            a = A[r["anchor_id"]]; seed = r["imagination_seed"]; t0 = time.time()
            c1 = rig.closed_loop(a, seed, acts); c2 = rig.closed_loop(a, seed, acts)
            row = dict(unit_id=uid, setting_id=r["setting_id"], family=fam, task=f"t{tid}", anchor_id=r["anchor_id"],
                       imagination_seed=seed, H_env=400,
                       closed_loop=dict(success=c1["success"], first_success_step=c1["first_success_step"],
                                        steps=int(c1["steps"]), action_dev_l2_mean=float(np.mean(c1["action_dev_l2"])),
                                        rerun=dict(success=c2["success"], first_success_step=c2["first_success_step"]),
                                        conclusion_consistent=bool(c1["success"] == c2["success"])),
                       seconds=round(time.time() - t0, 1), machine=os.uname().nodename,
                       gpu=torch.cuda.get_device_name(0))
            open(f"{out}/closed.{shard}.jsonl", "a").write(json.dumps(row) + "\n")
            print(uid, c1["success"], c2["success"], row["seconds"], flush=True)
    print("TX_CLOSED_DONE", f"t{tid}", shard)


if __name__ == "__main__":
    main()
