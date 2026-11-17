"""T2 LingBot closed loops (reference column) on the cluster RTX 5090s, consuming units as the cloud auto-pull lands
them on the NAS (ruling 2026-11-01: open-loop pair on $NODE_CLOUD, closed loops here; machine recorded per row).

Per unit: closed loop x2 with the unit's imagination seed (conclusion consistency checked); H = 400; L2 deviation
against the pulled imagined actions. Scans all four families repeatedly; exits once every family's driver.done marker
is present and nothing is left to do.

  python lingbot_t2_closed.py <shard> <nshards>   (lerobot_libero, GPU)
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import time
import zlib
import numpy as np

ROOT = _wmvp.P("$WMV_ROOT")
PULL = _wmvp.P(f"{ROOT.rsplit('/wm_verdict', 1)[0]}/cloud_results/$NODE_CLOUD/wm_t2")
OUT = f"{ROOT}/libero/out/runs/lingbot_t2"
_wmvp.bootstrap()
FAMS = ["base", "famO", "famP", "famA"]
from lingbot_t2_cloud import BANKS  # noqa: E402


def anchors_of(fam):
    bank = json.load(open(f"{ROOT}/{BANKS[fam]}"))
    for a in bank["anchors"]:
        if a.get("bddl_file") and not a["bddl_file"].startswith("/"):
            a["bddl_file"] = os.path.join(ROOT, a["bddl_file"])
    return {a["anchor_id"]: a for a in bank["anchors"]}


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    import torch
    import lingbot_unit as LU
    LU.H = 400
    rig = LU.LingBotRig(2)
    quiet_since = None
    while True:
        did = 0
        for fam in FAMS:
            src = f"{PULL}/{fam}"
            if not os.path.isdir(src):
                continue
            out = f"{OUT}/{fam}"; os.makedirs(out, exist_ok=True)
            done = set()
            for f in os.listdir(out):
                if f.startswith("closed."):
                    done |= {json.loads(l)["unit_id"] for l in open(f"{out}/{f}")}
            A = anchors_of(fam)
            recs = [json.loads(l) for f in sorted(os.listdir(src)) if f.startswith("records.") for l in open(f"{src}/{f}")]
            for r in recs:
                uid = r["unit_id"]
                if uid in done or zlib.crc32(uid.encode()) % nsh != shard:
                    continue
                ap = f"{src}/actions/{uid}.npz"
                if not os.path.exists(ap):
                    continue
                acts = np.load(ap)["env"]
                a = A[r["anchor_id"]]; seed = r["imagination_seed"]; t0 = time.time()
                c1 = rig.closed_loop(a, seed, acts); c2 = rig.closed_loop(a, seed, acts)
                consistent = bool(c1["success"] == c2["success"])
                row = dict(unit_id=uid, setting_id=r["setting_id"], family=fam, anchor_id=r["anchor_id"],
                           imagination_seed=seed, H_env=400,
                           closed_loop=dict(success=c1["success"], first_success_step=c1["first_success_step"],
                                            steps=int(c1["steps"]), action_dev_l2_mean=float(np.mean(c1["action_dev_l2"])),
                                            rerun=dict(success=c2["success"], first_success_step=c2["first_success_step"],
                                                       steps=int(c2["steps"])),
                                            conclusion_consistent=consistent),
                           seconds=round(time.time() - t0, 1), machine=os.uname().nodename,
                           gpu=torch.cuda.get_device_name(0))
                open(f"{out}/closed.{shard}.jsonl", "a").write(json.dumps(row) + "\n")
                done.add(uid); did += 1
                print(uid, c1["success"], c2["success"], "consistent", consistent, row["seconds"], flush=True)
        n_drv = len([f for f in os.listdir(PULL) if f.startswith("driver.")]) if os.path.isdir(PULL) else 0
        if did == 0:
            if n_drv >= 4:
                break
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since > 2 * 3600:
                print("no new work for 2 h and drivers not done; exiting"); break
            time.sleep(300)
        else:
            quiet_since = None
    print("CLOSED_DONE shard", shard)


if __name__ == "__main__":
    main()
