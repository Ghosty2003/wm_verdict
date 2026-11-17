"""OpenWAM-alpha E/G/N truth replays for T2/T3/T4 (queue ruling 2026-11-04): open-loop replay of the OpenWAM
baseline's imagined actions under each perturbation, x2 bit-identical; sham levels checked semantically against the
baseline record (success + first step). CPU pool; model never loaded.

  WMV_TID=<2|4|6> NPROC=10 python ow_egn_truth.py <shard> <nshards>   (lerobot_libero)
  -> out/runs/openwam_t<tid>/phys/truth.<shard>.jsonl + trajectories/
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
from phys_perturb import TRUTH_FAMILIES, settings, spec_of
from phys_truth import replay

TID = int(os.environ.get("WMV_TID", "4"))
BANK = f"{ROOT}/libero/out/bank/libero_anchors_t{TID}.json"
RUN = f"{ROOT}/libero/out/runs/openwam_t{TID}"
H = 400


def base_rows():
    import glob
    rows = []
    for f in glob.glob(f"{RUN}/records_base.jsonl") + glob.glob(f"{RUN}/base/records.*.jsonl"):
        rows += [json.loads(l) for l in open(f)]
    return [r for r in rows if r.get("setting_id", "none") == "none"]


def _unit(args):
    r, setting, anchor = args
    spec = spec_of(setting, TID, r["unit_id"])
    acts = np.load(f"{RUN}/actions/{r['unit_id']}.npz")["env"][:H]
    S1, s1, ex, info, (ferr, nfc) = replay(anchor, acts, spec)
    S2, s2, _, _, _ = replay(anchor, acts, spec)
    same = bool(np.array_equal(S1, S2) and np.array_equal(s1, s2))
    sham_ok = None
    if spec["sham"]:
        first = int(np.argmax(s1)) + 1 if s1.any() else None
        sham_ok = bool(s1.any() == r["real_verdict"] and first == r["real_first_success_step"])
    uid = f"{r['unit_id']}__{setting}"
    out = f"{RUN}/phys/trajectories/{uid}.npz"
    np.savez_compressed(out, states=S1, success=s1, executed=ex)
    first = int(np.argmax(s1)) + 1 if s1.any() else None
    return dict(unit_id=uid, base_unit=r["unit_id"], model="openwam", task=f"t{TID}", setting_id=setting,
                perturbation=spec, physics_change=info, real_verdict=bool(s1.any()), real_first_success_step=first,
                real_verdict_final=bool(s1[-1]), replay_check_identical=same, sham_semantic_matches_baseline=sham_ok,
                friction_contact_max_abs_err=ferr if spec["kind"] == "friction" else None, friction_contacts_checked=nfc,
                executed_differs_from_commanded=int((np.abs(ex - acts).sum(1) > 0).sum()),
                trajectory_file=os.path.relpath(out, ROOT), H_env=H, machine=os.uname().nodename)


def main():
    import multiprocessing as mp
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    A = {a["anchor_id"]: a for a in json.load(open(BANK))["anchors"]}
    os.makedirs(f"{RUN}/phys/trajectories", exist_ok=True)
    done = set()
    for f in os.listdir(f"{RUN}/phys"):
        if f.startswith("truth."):
            done |= {json.loads(l)["unit_id"] for l in open(f"{RUN}/phys/{f}")}
    jobs = [(r, s, A[r["anchor_id"]]) for r in base_rows() for fam in TRUTH_FAMILIES for s in settings(fam)
            if f"{r['unit_id']}__{s}" not in done and zlib.crc32(f"{r['unit_id']}__{s}".encode()) % nsh == shard]
    print("jobs", len(jobs), flush=True); t0 = time.time(); bad = []
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", 10)), maxtasksperchild=20) as pool:
        for i, rec in enumerate(pool.imap_unordered(_unit, jobs, chunksize=1)):
            open(f"{RUN}/phys/truth.{shard}.jsonl", "a").write(json.dumps(rec) + "\n")
            if not rec["replay_check_identical"] or rec["sham_semantic_matches_baseline"] is False or \
               (rec["friction_contact_max_abs_err"] or 0) > 1e-9:
                bad.append(rec["unit_id"]); print("BAD", json.dumps(rec)[:300], flush=True)
            if i % 100 == 0:
                print(i, "done", round(time.time() - t0), "s", flush=True)
    print("OW_EGN_TRUTH_DONE", f"t{TID}", "shard", shard, "bad", len(bad), bad[:8])
    if bad:
        sys.exit("STOP")


if __name__ == "__main__":
    main()
