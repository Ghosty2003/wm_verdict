"""Closed loop (reference column) of LingBot-VA under the non-input families E, G, N (ruling 2026-10-26). Same loop as
lingbot_unit.LingBotRig.closed_loop (policy reads the real frame every step, same anchor + torch seed as the baseline
unit), with the perturbation applied right after reset and the action stream applied to every executed command;
H = 400. Two runs per unit (closed-loop protocol); hard requirement = same success verdict (revision 2026-10-25),
first-success-step difference recorded. Sham levels are not rerun (their truth replay is bit-identical to the
baseline).

  python phys_closed_lingbot.py <shard> <nshards> [fam,...]   -> out/runs/lingbot_phys/closed_t<k>.<shard>.jsonl
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
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
from phys_perturb import FAMILIES, settings, spec_of, apply_physics, ActionStream, maybe_push

H = 400
OUT = f"{ROOT}/libero/out/runs/lingbot_phys"


def closed(rig, anchor, seed, spec):
    env, obs = rig.make_env(anchor)
    apply_physics(env.envs[0], spec)
    stream = ActionStream(spec, H)
    rig.reset_policy(seed)
    cmd, ex, first = [], [], None
    for t in range(H):
        a = rig.to_env_action(rig.policy.select_action(rig.obs_to_batch(obs)))
        e = stream(t, a[0]); cmd.append(a[0]); ex.append(e)
        maybe_push(env.envs[0], spec, t)                  # K: push once `push_after_step` steps are done (before step t + 1)
        obs, r, term, trunc, info = env.step(e[None])
        s = bool(np.asarray(info.get("is_success", [False])).reshape(-1)[0]) if "is_success" in info else False
        if "final_info" in info and isinstance(info["final_info"], dict) and "is_success" in info["final_info"]:
            s = s or bool(np.asarray(info["final_info"]["is_success"]).reshape(-1)[0])
        if s and first is None:
            first = t + 1
        if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
            break
    env.close()
    return np.stack(cmd), np.stack(ex), first


def main():
    import torch
    from lingbot_unit import LingBotRig
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    fams = sys.argv[3].split(",") if len(sys.argv) > 3 else list(FAMILIES)
    A = {a["anchor_id"]: a for a in json.load(open(f"{ROOT}/libero/out/bank/libero_anchors.json"))["anchors"]}
    os.makedirs(f"{OUT}/closed", exist_ok=True)
    for t in ("t5", "t3"):
        path = f"{OUT}/closed_{t}.{shard}.jsonl"
        done = {json.loads(l)["unit_id"] for f in glob.glob(f"{OUT}/closed_{t}.*.jsonl") for l in open(f)}   # any shard layout
        jobs = []
        for l in open(f"{ROOT}/libero/out/runs/lingbot_base/records_{t}.jsonl"):
            r = json.loads(l)
            for fam in fams:
                for s in settings(fam)[1:]:                     # sham levels excluded
                    jobs.append((r, s))
        jobs = [j for i, j in enumerate(jobs) if i % nsh == shard and f"{j[0]['unit_id']}__{j[1]}" not in done]
        if not jobs:
            continue
        rig = LingBotRig(int(t[1:]))
        for r, s in jobs:
            uid = f"{r['unit_id']}__{s}"; spec = spec_of(s, int(t[1:]), r["unit_id"]); t0 = time.time()
            c1, e1, f1 = closed(rig, A[r["s0_id"]], r["imagination_seed"], spec)
            c2, e2, f2 = closed(rig, A[r["s0_id"]], r["imagination_seed"], spec)
            n = min(len(e1), len(e2)); diff = np.where(np.any(e1[:n] != e2[:n], axis=1))[0]
            div = int(diff[0]) + 1 if len(diff) else (None if len(e1) == len(e2) else n + 1)
            np.savez_compressed(f"{OUT}/closed/{uid}.npz", commanded=c1, executed=e1, commanded2=c2, executed2=e2)
            rec = dict(unit_id=uid, base_unit=r["unit_id"], setting_id=s, perturbation=spec, machine=os.uname().nodename,
                       closed_loop=dict(success=f1 is not None, first_success_step=f1, steps=len(e1),
                                        deterministic_rerun_identical=div is None,
                                        rerun=dict(success=f2 is not None, first_success_step=f2, steps=len(e2)),
                                        first_divergence_step=div, conclusion_consistent=(f1 is None) == (f2 is None)),
                       closed_loop_file=os.path.relpath(f"{OUT}/closed/{uid}.npz", ROOT), seconds=round(time.time() - t0, 1))
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(dict(unit=uid, s1=f1, s2=f2, div=div, sec=rec["seconds"])), flush=True)
            torch.cuda.empty_cache()
        del rig; torch.cuda.empty_cache()
    print("DONE")


_RIGS = {}


def SHAM_RUN(r, t):
    from lingbot_unit import LingBotRig
    if t not in _RIGS:
        _RIGS.clear(); _RIGS[t] = LingBotRig(int(t[1:]))
    A = {a["anchor_id"]: a for a in json.load(open(f"{ROOT}/libero/out/bank/libero_anchors.json"))["anchors"]}
    c, e, f = closed(_RIGS[t], A[r["s0_id"]], r["imagination_seed"], spec_of("Em0", int(t[1:])))
    return c


def sham_check(n=5):
    """closed-loop code path check: the sham level (Em0) must reproduce the baseline's stored closed-loop actions
    bit for bit on the first n units of each task."""
    import json as _j
    ok = []
    for t in ("t5", "t3"):
        recs = [_j.loads(l) for l in open(f"{ROOT}/libero/out/runs/lingbot_base/records_{t}.jsonl")][:n]
        for r in recs:
            ref = np.load(f"{ROOT}/{r['action_file']}")["closed_loop"][:H]     # baseline closed loop ran to 520; this one stops at 400
            got = SHAM_RUN(r, t)
            same = bool(got.shape == ref.shape and np.array_equal(got, ref))
            ok.append(same); print("sham", r["unit_id"], "identical", same, got.shape, ref.shape, flush=True)
    print("SHAM_CHECK", "PASS" if all(ok) else "FAIL", sum(ok), "/", len(ok))
    if not all(ok):
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sham":
        sham_check()
    else:
        main()
