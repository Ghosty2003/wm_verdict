"""T2 truth (a) for the non-input families E / G / N (ruling 2026-11-02: queued on 585/627): open-loop replay of the
T2 LingBot baseline's imagined actions ($NODE_CLOUD run, H = 400) under each perturbation, x2 bit-identical, sham levels
checked SEMANTICALLY against the baseline truth records (success + first step) - the baseline replays ran on the
A100 node, so cross-machine bit identity is not expected and is not claimed; self-consistency stays bit-exact.

  python t2_phys_truth.py probe              stability probe at Em max / Ef min on the 5 T2 anchors
  python t2_phys_truth.py run <shard> <n>    -> out/runs/lingbot_t2/phys/truth.<shard>.jsonl + trajectories/
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
from phys_truth import replay, anchors as _anchors_t53  # noqa: F401  (replay reused verbatim)

TID = int(os.environ.get("WMV_TID", "2"))
BANK = f"{ROOT}/libero/out/bank/libero_anchors_t{TID}.json"
RUN = f"{ROOT}/libero/out/runs/lingbot_t{TID}"
H = 400


def anchors():
    return {a["anchor_id"]: a for a in json.load(open(BANK))["anchors"]}


def base_rows():
    import glob
    R = {json.loads(l)["unit_id"]: json.loads(l) for f in glob.glob(f"{RUN}/base/records.*.jsonl") for l in open(f)}
    T = {json.loads(l)["unit_id"]: json.loads(l) for f in glob.glob(f"{RUN}/base/truth.*.jsonl") for l in open(f)}
    return [dict(r, **{"truth": T[u]}) for u, r in R.items()]


def _unit(args):
    r, setting, anchor = args
    spec = spec_of(setting, TID, r["unit_id"])
    acts = np.load(f"{RUN}/base/actions/{r['unit_id']}.npz")["env"][:H]
    S1, s1, ex, info, (ferr, nfc) = replay(anchor, acts, spec)
    S2, s2, _, _, _ = replay(anchor, acts, spec)
    same = bool(np.array_equal(S1, S2) and np.array_equal(s1, s2))
    sham_ok = None
    if spec["sham"]:
        first = int(np.argmax(s1)) + 1 if s1.any() else None
        sham_ok = bool(s1.any() == r["truth"]["real_verdict"] and first == r["truth"]["real_first_success_step"])
    uid = f"{r['unit_id']}__{setting}"
    out = f"{RUN}/phys/trajectories/{uid}.npz"
    np.savez_compressed(out, states=S1, success=s1, executed=ex)
    first = int(np.argmax(s1)) + 1 if s1.any() else None
    return dict(unit_id=uid, base_unit=r["unit_id"], model="lingbot", task=f"t{TID}", setting_id=setting, perturbation=spec,
                physics_change=info, real_verdict=bool(s1.any()), real_first_success_step=first,
                real_verdict_final=bool(s1[-1]), replay_check_identical=same,
                sham_semantic_matches_baseline=sham_ok,
                sham_note=None if sham_ok is None else _wmvp.P("baseline replay ran on $NODE_CLOUD (A100): semantic check only, self-consistency bit-exact"),
                friction_contact_max_abs_err=ferr if spec["kind"] == "friction" else None, friction_contacts_checked=nfc,
                executed_differs_from_commanded=int((np.abs(ex - acts).sum(1) > 0).sum()),
                trajectory_file=os.path.relpath(out, ROOT), H_env=H, machine=os.uname().nodename)


def run(shard, nsh):
    import multiprocessing as mp
    A = anchors(); os.makedirs(f"{RUN}/phys/trajectories", exist_ok=True)
    path = f"{RUN}/phys/truth.{shard}.jsonl"
    done = set()
    for f in os.listdir(f"{RUN}/phys"):
        if f.startswith("truth."):
            done |= {json.loads(l)["unit_id"] for l in open(f"{RUN}/phys/{f}")}
    jobs = [(r, s, A[r["anchor_id"]]) for r in base_rows() for fam in TRUTH_FAMILIES for s in settings(fam)
            if f"{r['unit_id']}__{s}" not in done and zlib.crc32(f"{r['unit_id']}__{s}".encode()) % nsh == shard]
    print("jobs", len(jobs), flush=True); t0 = time.time(); bad = []
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", 10)), maxtasksperchild=20) as pool:
        for i, rec in enumerate(pool.imap_unordered(_unit, jobs, chunksize=1)):
            open(path, "a").write(json.dumps(rec) + "\n")
            if not rec["replay_check_identical"] or rec["sham_semantic_matches_baseline"] is False or \
               (rec["friction_contact_max_abs_err"] or 0) > 1e-9:
                bad.append(rec["unit_id"]); print("BAD", json.dumps(rec)[:300], flush=True)
            if i % 100 == 0:
                print(i, "done", round(time.time() - t0), "s", flush=True)
    print("T2_TRUTH_DONE shard", shard, "bad", len(bad), bad[:8])
    if bad:
        sys.exit("t2 truth replay check failed -- STOP")


def probe():
    from libero_env_adapter import _make_env
    from phys_perturb import apply_physics
    A = anchors()
    for setting in ("Em4", "Ef4"):
        for aid, a in A.items():
            spec = spec_of(setting, TID)
            env = _make_env(a, H=H); u = env.envs[0]
            apply_physics(u, spec); p = u._env.env
            bid = p.obj_body_id[spec["object"]]; x0 = u._env.sim.data.body_xpos[bid].copy()
            for _ in range(100):
                u.step(np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
            dx = float(np.linalg.norm(u._env.sim.data.body_xpos[bid] - x0))
            env.close()
            print("probe", setting, aid, "drift_m", round(dx, 4), "OK" if dx < 0.01 else "UNSTABLE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "probe":
        probe()
    else:
        run(int(sys.argv[2]), int(sys.argv[3]))
