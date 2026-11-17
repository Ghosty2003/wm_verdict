"""OpenWAM-alpha E/G/N/K closed loops for T2/T3/T4 (queue ruling 2026-11-04): phys_closed_openwam.py's loop (the
perturbation is applied inside sim_server at reset: physics change + action stream + the K push) on the new-task
banks and baselines. Two runs per unit, conclusion consistency recorded.

  WMV_TID=<2|4|6> python ow_phys_closed_tx.py <shard> <nshards>   (openwam interpreter, GPU)
  -> out/runs/openwam_t<tid>/phys/closed.<shard>.jsonl
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import glob
import json
import time
import zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
import ow_runner as W
from phys_perturb import FAMILIES, settings, spec_of
from phys_closed_openwam import closed

TID = int(os.environ.get("WMV_TID", "4"))
BANK = f"{ROOT}/libero/out/bank/libero_anchors_t{TID}.json"
RUN = f"{ROOT}/libero/out/runs/openwam_t{TID}"
PROMPT = {2: "turn on the stove and put the moka pot on it",
          4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
          6: "put the white mug on the plate and put the chocolate pudding to the right of the plate"}


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    W._setup()
    eng, pre, arch, _, ckname = W._engine()
    sim = W.Sim(f"/dev/shm/ow_phys_t{TID}_{os.getpid()}")
    A = {a["anchor_id"]: a for a in json.load(open(BANK))["anchors"]}
    os.makedirs(f"{RUN}/phys", exist_ok=True)
    path = f"{RUN}/phys/closed.{shard}.jsonl"
    done = {json.loads(l)["unit_id"] for f in glob.glob(f"{RUN}/phys/closed.*.jsonl") for l in open(f)}
    rows = [json.loads(l) for l in open(f"{RUN}/records_base.jsonl")]
    jobs = [(r, s) for r in rows for fam in FAMILIES for s in settings(fam)[1:]
            if f"{r['unit_id']}__{s}" not in done and zlib.crc32(f"{r['unit_id']}__{s}".encode()) % nsh == shard]
    print("jobs", len(jobs), flush=True)
    for r, s in jobs:
        uid = f"{r['unit_id']}__{s}"
        spec = spec_of(s, TID, r["unit_id"]); t0 = time.time(); seed = r["imagination_seed"]
        anc = dict(A[r["anchor_id"]], resolution=256)
        c1, e1, s1, i1 = closed(eng, sim, anc, PROMPT[TID], seed, spec)
        c2, e2, s2, i2 = closed(eng, sim, anc, PROMPT[TID], seed, spec)
        v1, v2 = bool(s1.any()), bool(s2.any())
        os.makedirs(f"{RUN}/closed", exist_ok=True)
        np.savez_compressed(f"{RUN}/closed/{uid}.npz", commanded=c1, executed=e1, success=s1,
                            commanded2=c2, executed2=e2, success2=s2,
                            imagined=np.concatenate(i1), imagined_call_lens=np.array([len(x) for x in i1]),
                            imagined2=np.concatenate(i2), imagined2_call_lens=np.array([len(x) for x in i2]))
        rec = dict(unit_id=uid, base_unit=r["unit_id"], setting_id=s, task=f"t{TID}", perturbation=spec,
                   imagination_seed=seed, H_env=W.H,
                   closed_loop=dict(success=v1, first_success_step=int(np.argmax(s1)) + 1 if v1 else None,
                                    steps=int(len(s1)), rerun=dict(success=v2),
                                    conclusion_consistent=bool(v1 == v2),
                                    executed_differs_from_commanded=int((np.abs(e1 - c1).sum(1) > 0).sum())),
                   seconds=round(time.time() - t0, 1), machine=os.uname().nodename)
        open(path, "a").write(json.dumps(rec) + "\n")
        print(uid, v1, v2, rec["seconds"], flush=True)
    sim.close()
    print("OW_PHYS_CLOSED_DONE", f"t{TID}", "shard", shard)


if __name__ == "__main__":
    main()
