"""T2 E/G/N closed loops (2026-11-03 queue ruling): the model deployed normally while the physics / action-stream
perturbation is active (apply_physics at reset; ActionStream between the policy and the env), x2 per unit with the
base unit's imagination seed; H = 400. Settings = every non-sham E/G/N level (shams equal the plain base closed runs
already recorded).

  python t2_egn_closed.py <shard> <nshards>   (lerobot_libero, GPU)
  -> out/runs/lingbot_t2/phys/closed_egn.<shard>.jsonl
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
from phys_perturb import TRUTH_FAMILIES, settings, spec_of, apply_physics, ActionStream

TID = int(os.environ.get("WMV_TID", "2"))
BANK = f"{ROOT}/libero/out/bank/libero_anchors_t{TID}.json"
RUN = f"{ROOT}/libero/out/runs/lingbot_t{TID}"
H = 400


def closed_pert(rig, anchor, seed, spec, imagined):
    import torch
    env, obs = rig.make_env(anchor)
    u = env.envs[0]
    info = apply_physics(u, spec)
    stream = ActionStream(spec, H)
    rig.reset_policy(seed)
    acts, succ, first = [], [], None
    with torch.inference_mode():
        for t in range(H):
            batch = rig.obs_to_batch(obs)
            a = rig.to_env_action(rig.policy.select_action(batch))
            acts.append(a[0])
            e = stream(t, a[0].astype(np.float32))
            obs, rwd, term, trunc, inf = env.step(e[None])
            s = bool(np.asarray(inf.get("is_success", [False])).reshape(-1)[0]) if "is_success" in inf else False
            succ.append(s)
            if s and first is None:
                first = t + 1
            if bool(np.asarray(term).reshape(-1)[0]) or bool(np.asarray(trunc).reshape(-1)[0]):
                break
    env.close()
    acts = np.stack(acts); n = len(acts)
    dev = float(np.mean(np.linalg.norm(acts - imagined[:n], axis=1)))
    return dict(success=first is not None, first_success_step=first, steps=n, action_dev_l2_mean=dev), info


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    import lingbot_unit as LU
    LU.H = H
    rig = LU.LingBotRig(TID)
    A = {a["anchor_id"]: a for a in json.load(open(BANK))["anchors"]}
    R = {json.loads(l)["unit_id"]: json.loads(l) for f in glob.glob(f"{RUN}/base/records.*.jsonl") for l in open(f)}
    done = set()
    for f in os.listdir(f"{RUN}/phys"):
        if f.startswith("closed_egn."):
            done |= {json.loads(l)["unit_id"] for l in open(f"{RUN}/phys/{f}")}
    path = f"{RUN}/phys/closed_egn.{shard}.jsonl"
    S = [s for fam in TRUTH_FAMILIES for s in settings(fam) if not spec_of(s, TID)["sham"]]
    jobs = [(r, s) for r in R.values() for s in S
            if f"{r['unit_id']}__{s}" not in done and zlib.crc32(f"{r['unit_id']}__{s}".encode()) % nsh == shard]
    print("jobs", len(jobs), flush=True)
    for r, s in jobs:
        uid = f"{r['unit_id']}__{s}"; spec = spec_of(s, TID, r["unit_id"]); t0 = time.time()
        imagined = np.load(f"{RUN}/base/actions/{r['unit_id']}.npz")["env"][:H]
        c1, info = closed_pert(rig, A[r["anchor_id"]], r["imagination_seed"], spec, imagined)
        c2, _ = closed_pert(rig, A[r["anchor_id"]], r["imagination_seed"], spec, imagined)
        row = dict(unit_id=uid, base_unit=r["unit_id"], setting_id=s, task=f"t{TID}", perturbation=spec,
                   physics_change=info, imagination_seed=r["imagination_seed"], H_env=H,
                   closed_loop=dict(**c1, rerun=c2, conclusion_consistent=bool(c1["success"] == c2["success"])),
                   seconds=round(time.time() - t0, 1), machine=os.uname().nodename)
        open(path, "a").write(json.dumps(row) + "\n")
        print(uid, c1["success"], c2["success"], row["seconds"], flush=True)
    print("T2_EGN_CLOSED_DONE shard", shard)


if __name__ == "__main__":
    main()
