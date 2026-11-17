"""Machine probe (2026-10-20 ruling 1): replay the frozen anchors on this host and compare each trajectory SHA-256
with the one stored at freeze time. Read-only on the banks; writes only out/machine_probe/.

Anchors: the 10 main anchors (libero_anchors.json) and the 50 family-P anchors (familyP_anchors.json). Per anchor:
two fresh-env replays of the bank's own fixed action sequence (R steps), fresh vs fresh must be bit-identical, and the
SHA over the first replay (same byte layout as anchor_bank.py / familyP_bank.py) must equal the stored value.
Bank and verify-file SHAs are checked before and after.

  MUJOCO_GL=egl python machine_probe.py
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import time
import socket
import hashlib
import numpy as np

_wmvp.bootstrap()
from libero.libero import benchmark
from anchor_bank import run as run_main, maxdiff, SUITE, R_STEPS, MASTER_SEED
from familyP_bank import run as run_P, MASTER as MASTER_P

ROOT = _wmvp.P("$WMV_ROOT")
BANK = f"{ROOT}/libero/out/bank"
OUT = f"{ROOT}/libero/out/machine_probe"
FILES = {"main_bank": (f"{BANK}/libero_anchors.json", "4c4cc503e692b35bf710fe25b08d85ff916e9fee78a584094a0c1dfb7e9ac5d7"),
         "famP_bank": (f"{BANK}/familyP/familyP_anchors.json", "64fa63cc9c02fa9dadcb621de44d21661993b5157867abc5a948c2c911208e95")}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def fixed_actions(master):
    ars = np.random.RandomState(master + 1)
    actions = np.clip(np.cumsum(ars.uniform(-0.15, 0.15, size=(R_STEPS, 7)), axis=0), -0.6, 0.6).astype(np.float32)
    actions[:, 6] = np.where(np.arange(R_STEPS) % 60 < 30, -1.0, 1.0)
    return actions


def traj_sha(S):
    return hashlib.sha256(b"".join(np.ascontiguousarray(s[k]).tobytes() for s in S for k in sorted(s))).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    host = socket.gethostname()
    files_before = {k: dict(path=p, expected=e, actual=sha(p), ok=sha(p) == e) for k, (p, e) in FILES.items()}
    if not all(v["ok"] for v in files_before.values()):
        json.dump(dict(machine=host, files_before=files_before, verdict="STOP: bank SHA mismatch"),
                  open(f"{OUT}/probe_{host}.json", "w"), indent=1)
        print("STOP: bank SHA mismatch", files_before); return
    suite = benchmark.get_benchmark_dict()[SUITE]()
    jobs = []
    stored = {r["anchor_id"]: r["trajectory_sha256"] for r in json.load(open(f"{BANK}/anchor_verify.json"))["results"]}
    for a in json.load(open(FILES["main_bank"][0]))["anchors"]:
        jobs.append(("main", a, run_main, fixed_actions(MASTER_SEED), stored[a["anchor_id"]]))
    stored = {r["anchor_id"]: r["trajectory_sha256"] for r in json.load(open(f"{BANK}/familyP/familyP_verify.json"))["results"]}
    for a in json.load(open(FILES["famP_bank"][0]))["anchors"]:
        jobs.append(("famP", a, run_P, fixed_actions(MASTER_P), stored[a["anchor_id"]]))
    rows = []
    for bank, a, fn, actions, want in jobs:
        t0 = time.time()
        SA = fn(suite, a, actions); SB = fn(suite, a, actions)
        dAB = maxdiff(SA, SB); got = traj_sha(SA)
        row = dict(bank=bank, anchor_id=a["anchor_id"], fresh_vs_fresh_identical=all(v == 0.0 for v in dAB.values()),
                   fresh_vs_fresh=dAB, stored_sha256=want, probe_sha256=got, sha_match=got == want,
                   seconds=round(time.time() - t0, 1))
        if not row["sha_match"]:      # which channels moved is not recoverable from a single stored hash; keep a phys-only hash
            row["probe_sha256_phys_only"] = hashlib.sha256(b"".join(np.ascontiguousarray(s["phys"]).tobytes() for s in SA)).hexdigest()
        rows.append(row)
        print(bank, a["anchor_id"], "MATCH" if row["sha_match"] else "MISMATCH",
              "ff-ok" if row["fresh_vs_fresh_identical"] else f"ff-DIFF {dAB}", row["seconds"], flush=True)
    files_after = {k: dict(path=p, actual=sha(p), ok=sha(p) == e) for k, (p, e) in FILES.items()}
    ok = all(r["sha_match"] and r["fresh_vs_fresh_identical"] for r in rows) and all(v["ok"] for v in files_after.values())
    json.dump(dict(machine=host, date=time.strftime("%Y-%m-%d %H:%M:%S"), files_before=files_before, files_after=files_after,
                   n_anchors=len(rows), n_sha_match=sum(r["sha_match"] for r in rows),
                   n_fresh_identical=sum(r["fresh_vs_fresh_identical"] for r in rows),
                   all_pass=ok, results=rows), open(f"{OUT}/probe_{host}.json", "w"), indent=1)
    print("ALL PASS" if ok else "SOME FAIL", f"{sum(r['sha_match'] for r in rows)}/{len(rows)} SHA match", host)


if __name__ == "__main__":
    main()
