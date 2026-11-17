"""LingBot production units for a new task (T3 = LIBERO-10 #4 / T4 = #6; onboarding rulings 2026-11-01/02):
imagination to H = 400 (T2 precedent), truth replays delegated to the companion CPU pool (t2_replay.py with WMV_FAMS /
WMV_BANKS), base -> famO -> famP (family A waits for the object ruling). Smoke gate: after the base family the run
STOPS if the open-loop truth success of the first 10 base units is < 20 % (proposal rule), printing SMOKE_GATE.

Imagination seeds: RandomState(master)[anchor_index, k] as everywhere; masters recorded here: t4 = 20261111,
t6 = 20261112.

  python lingbot_tx_run.py <tid> <fam> <shard> <nshards>
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
import hashlib
import numpy as np

ROOT = os.environ.get("WMV_ROOT", _wmvp.P("$WMV_ROOT"))
_wmvp.bootstrap()
from lingbot_t2_cloud import occlude, imagine_with_batch, RATIOS  # noqa: E402

CFG = {4: dict(master=20261111), 6: dict(master=20261112)}
K = 10
H2 = 400


def banks_of(tid):
    return {"base": f"libero/out/bank/libero_anchors_t{tid}.json",
            "famO": f"libero/out/bank/libero_anchors_t{tid}.json",
            "famP": f"libero/out/bank/familyP_t{tid}/familyP_t{tid}_anchors.json",
            "famA": f"libero/out/bank/familyA_t{tid}/familyA_t{tid}_anchors.json"}


def units_of(fam, bank):
    A = bank["anchors"]
    if fam == "base":
        return [(a, a["anchor_id"], None, i) for i, a in enumerate(A)]
    if fam == "famO":
        return [(a, a["anchor_id"], L, i) for i, a in enumerate(A) for L in ("O1", "O2", "O3", "O4", "O5")]
    bases = sorted({a["base_id"] for a in A})
    return [(a, a["base_id"], a["level"], bases.index(a["base_id"])) for a in A]


def main():
    tid, fam, shard, nsh = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    import torch
    import lingbot_unit as LU
    LU.H = H2
    RES = os.environ.get("WMV_RESULTS", f"{ROOT}/libero/out/runs/lingbot_t{tid}")
    bank = json.load(open(f"{ROOT}/{banks_of(tid)[fam]}"))
    for a in bank["anchors"]:
        if a.get("bddl_file"):
            a["bddl_file"] = os.path.join(ROOT, a["bddl_file"])
    rig = LU.LingBotRig(tid)
    out = f"{RES}/{fam}"; os.makedirs(f"{out}/actions", exist_ok=True); os.makedirs(f"{out}/latents", exist_ok=True)
    seeds = np.random.RandomState(CFG[tid]["master"]).randint(0, 2 ** 31 - 1, size=(16, K))
    path = f"{out}/records.{shard}.jsonl"
    done = set()
    for f in os.listdir(out):
        if f.startswith("records."):
            done |= {json.loads(l)["unit_id"] for l in open(f"{out}/{f}")}
    jobs = [(a, s0, L, ai, k) for a, s0, L, ai in units_of(fam, bank) for k in range(K)]
    jobs = [j for n, j in enumerate(jobs) if n % nsh == shard]
    for a, s0, L, ai, k in jobs:
        uid = f"lingbot_{s0}__k{k}" + (f"__{L}" if L else "")
        if uid in done:
            continue
        t0 = time.time(); seed = int(seeds[ai, k])
        if fam == "famO":
            env, obs = rig.make_env(a)
            obs = dict(obs); obs["pixels"] = dict(obs["pixels"])
            obs["pixels"]["image"] = occlude(obs["pixels"]["image"], RATIOS[L])
            batch = rig.obs_to_batch(obs); env.close()
            rig.reset_policy(seed)
            img = imagine_with_batch(rig, batch)
        else:
            img = rig.imagine(a, seed)
        np.savez_compressed(f"{out}/actions/{uid}.npz", env=img["env_actions"])
        torch.save(img["latents"], f"{out}/latents/{uid}.pt")
        rec = dict(unit_id=uid, model="lingbot_va_libero_long", task_id=tid, s0_id=s0, anchor_id=a["anchor_id"],
                   seed_index=k, imagination_seed=seed, seed_master=CFG[tid]["master"], setting_id=L or "none",
                   family=fam, H_env=H2, criterion="reach_anytime", n_chunks=img["n_chunks"],
                   real_verdict=None, real_first_success_step=None, replay_check_identical=None, truth_pending=True,
                   closed_loop=None, closed_loop_note="runs later (reference column)",
                   action_sha256=hashlib.sha256(np.ascontiguousarray(img["env_actions"]).tobytes()).hexdigest(),
                   seconds=round(time.time() - t0, 1), seconds_imagine=round(img["seconds_imagine"], 1),
                   machine=os.uname().nodename, gpu=torch.cuda.get_device_name(0))
        open(path, "a").write(json.dumps(rec) + "\n")
        print(uid, "imagined", rec["seconds"], flush=True)
        if fam == "base":                                        # smoke gate on the truth rows the CPU pool writes
            T = [json.loads(l) for f_ in glob.glob(f"{out}/truth.*.jsonl") for l in open(f_)]
            if len(T) >= 10 and sum(t["real_verdict"] for t in T[:20]) / min(len(T), 20) < 0.2 and len(T) >= 20:
                print("SMOKE_GATE base truth success < 20% after", len(T), "units -- stopping this task", flush=True)
                return 3
    print("SHARD_DONE", f"t{tid}", fam, shard, "/", nsh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
