"""T2 family K (mid-episode push; closed loop only, ruling 2026-10-26 definitions): LingBot closed loop x2 per
(base unit x K setting), push = move the moka pot's free joint by d m along +x right after env step t
(t in {60, 160, 260}, d in {0.02, 0.05, 0.10}), same imagination seed as the base unit; H = 400; conclusion
consistency across the two runs; L2 deviation against the base unit's imagined actions.

  python t2_k_closed.py <shard> <nshards>   (lerobot_libero, GPU)  -> out/runs/lingbot_t2/phys/closed_K.<shard>.jsonl
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import time
import glob
import zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
from phys_perturb import settings, spec_of, maybe_push

TID = int(os.environ.get("WMV_TID", "2"))
BANK = f"{ROOT}/libero/out/bank/libero_anchors_t{TID}.json"
RUN = f"{ROOT}/libero/out/runs/lingbot_t{TID}"
H = 400


def closed_push(rig, anchor, seed, spec, imagined):
    import torch
    env, obs = rig.make_env(anchor)
    rig.reset_policy(seed)
    u = env.envs[0]
    acts, succ, first = [], [], None
    with torch.inference_mode():
        for t in range(H):
            batch = rig.obs_to_batch(obs)
            a = rig.to_env_action(rig.policy.select_action(batch))
            acts.append(a[0])
            obs, rwd, term, trunc, info = env.step(a)
            maybe_push(u, spec, t + 1)                           # steps_done is 1-based after this step
            s = bool(np.asarray(info.get("is_success", [False])).reshape(-1)[0]) if "is_success" in info else False
            succ.append(s)
            if s and first is None:
                first = t + 1
            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                break
    env.close()
    acts = np.stack(acts); n = len(acts)
    dev = float(np.mean(np.linalg.norm(acts - imagined[:n], axis=1)))
    return dict(success=first is not None, first_success_step=first, steps=n, action_dev_l2_mean=dev)


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    import torch
    import lingbot_unit as LU
    LU.H = H
    rig = LU.LingBotRig(TID)
    A = {a["anchor_id"]: a for a in json.load(open(BANK))["anchors"]}
    R = {json.loads(l)["unit_id"]: json.loads(l) for f in glob.glob(f"{RUN}/base/records.*.jsonl") for l in open(f)}
    os.makedirs(f"{RUN}/phys", exist_ok=True)
    done = set()
    for f in os.listdir(f"{RUN}/phys"):
        if f.startswith("closed_K."):
            done |= {json.loads(l)["unit_id"] for l in open(f"{RUN}/phys/{f}")}
    path = f"{RUN}/phys/closed_K.{shard}.jsonl"
    jobs = [(r, s) for r in R.values() for s in settings("K")[1:]
            if f"{r['unit_id']}__{s}" not in done and zlib.crc32(f"{r['unit_id']}__{s}".encode()) % nsh == shard]
    print("jobs", len(jobs), flush=True)
    for r, s in jobs:
        uid = f"{r['unit_id']}__{s}"; spec = spec_of(s, TID, r["unit_id"]); t0 = time.time()
        imagined = np.load(f"{RUN}/base/actions/{r['unit_id']}.npz")["env"][:H]
        c1 = closed_push(rig, A[r["anchor_id"]], r["imagination_seed"], spec, imagined)
        c2 = closed_push(rig, A[r["anchor_id"]], r["imagination_seed"], spec, imagined)
        row = dict(unit_id=uid, base_unit=r["unit_id"], setting_id=s, family="K", task=f"t{TID}",
                   perturbation=spec, imagination_seed=r["imagination_seed"], H_env=H,
                   closed_loop=dict(**c1, rerun=c2, conclusion_consistent=bool(c1["success"] == c2["success"])),
                   seconds=round(time.time() - t0, 1), machine=os.uname().nodename)
        open(path, "a").write(json.dumps(row) + "\n")
        print(uid, c1["success"], c2["success"], row["closed_loop"]["conclusion_consistent"], row["seconds"], flush=True)
    print("T2_K_DONE shard", shard)


if __name__ == "__main__":
    main()
