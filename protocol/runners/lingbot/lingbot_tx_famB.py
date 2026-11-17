"""LingBot-VA family B for T2/T3/T4 (queue ruling 2026-11-04): gaussian/fog/zoom x severity 2/5/9 on the raw 128
agentview (native resolution, so the no-corruption level IS the task's baseline). Open loop corrupts only the real
t = 0 frame the model reads; closed loop corrupts every real frame (x2, divergence recorded). Imagination stops at
H = 400 (new-task ruling). No predicate scores yet (T2/T4/T6 judges uncalibrated): latents + sampled decoded frames
are stored for later scoring, verdict columns stay pending.
Sham: one unit per anchor re-imagined with no corruption must reproduce the baseline actions - bit-exact for t4/t6
(baselines ran on these same 5090 nodes), semantic (success + first step after replay) for t2 (baseline imagined on
the A100 node).

  WMV_TID=<2|4|6> python lingbot_tx_famB.py sham | <shard> <nshards>   (lerobot_libero, GPU)
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import time
import hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
from bcorrupt import SETTINGS, spec, Corruptor
from lingbot_famB import make_rig_class

TID = int(os.environ.get("WMV_TID", "4"))
MASTER = {2: 20261104, 4: 20261111, 6: 20261112}[TID]
K, H2 = 10, 400
BANK = f"{ROOT}/libero/out/bank/libero_anchors_t{TID}.json"
BASE = f"{ROOT}/libero/out/runs/lingbot_t{TID}/base"
OUT = f"{ROOT}/libero/out/runs/lingbot_t{TID}/famB"


def base_recs():
    import glob
    return {(r["anchor_id"], r["seed_index"]): r for f in glob.glob(f"{BASE}/records.*.jsonl")
            for r in [json.loads(l) for l in open(f)]}


def main():
    import torch
    import lingbot_unit as LU
    LU.H = H2
    Rig = make_rig_class(LU.LingBotRig)
    bank = json.load(open(BANK))
    seeds = np.random.RandomState(MASTER).randint(0, 2 ** 31 - 1, size=(16, K))
    anchors = list(enumerate(bank["anchors"]))
    if sys.argv[1] == "sham":
        R = base_recs(); ok = []
        rig = Rig(TID)
        for ai, a in anchors[:2]:
            ref = R[(a["anchor_id"], 0)]
            I = rig.imagine(a, int(seeds[ai, 0]))
            h = hashlib.sha256(np.ascontiguousarray(I["env_actions"]).tobytes()).hexdigest()
            if TID == 2:
                r1 = rig.replay(a, I["env_actions"]); first = int(np.argmax(r1["succ"])) + 1 if r1["succ"].any() else None
                tr = {json.loads(l)["unit_id"]: json.loads(l) for f_ in __import__("glob").glob(f"{BASE}/truth.*.jsonl") for l in open(f_)}
                t = tr[ref["unit_id"]]
                same = bool(r1["succ"].any() == t["real_verdict"] and first == t["real_first_success_step"])
                print("B sham (semantic, A100 baseline)", a["anchor_id"], same, flush=True)
            else:
                same = (h == ref["action_sha256"])
                print("B sham (bit)", a["anchor_id"], same, flush=True)
            ok.append(same)
        print("B_SHAM", "PASS" if all(ok) else "FAIL"); sys.exit(0 if all(ok) else 1)
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    for sub in ("actions", "trajectories", "latents", "samples"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    corr = Corruptor()
    rig = Rig(TID); rig.corr = corr
    path = f"{OUT}/records.{shard}.jsonl"
    done = {json.loads(l)["unit_id"] for f in os.listdir(OUT) if f.startswith("records.") for l in open(f"{OUT}/{f}")}
    jobs = [(ai, a, k, s) for ai, a in anchors for s in SETTINGS[1:] for k in range(K)]
    jobs = [j for n_, j in enumerate(jobs) if n_ % nsh == shard]
    for ai, a, k, s in jobs:
        uid = f"lingbot_{a['anchor_id']}__k{k}__{s}"
        if uid in done:
            continue
        seed = int(seeds[ai, k]); t0 = time.time()
        rig.b, rig.uid, rig.log = s, uid, []
        I = rig.imagine(a, seed); first_pair = rig.log[0]
        rig.b = None
        frames = rig.decode(I["latents"]).numpy()[: H2 + 1]
        R1, R2 = rig.replay(a, I["env_actions"]), rig.replay(a, I["env_actions"])
        rig.b, rig.log = s, []
        C1 = rig.closed_loop(a, seed, I["env_actions"]); n1 = len(rig.log)
        cl_samples = [(t, y) for t, _, y in rig.log if t % 50 == 0]
        rig.log = []
        C2 = rig.closed_loop(a, seed, I["env_actions"]); n2 = len(rig.log)
        rig.b, rig.log = None, None
        assert n1 == C1["steps"] and n2 == C2["steps"], (uid, n1, C1["steps"], n2, C2["steps"])
        same = bool(np.array_equal(R1["states"], R2["states"]) and np.array_equal(R1["succ"], R2["succ"]))
        import torch as _t
        np.savez_compressed(f"{OUT}/actions/{uid}.npz", env=I["env_actions"], closed_loop=C1["actions"], closed_loop2=C2["actions"])
        np.savez_compressed(f"{OUT}/trajectories/{uid}.npz", replay1=R1["states"], replay2=R2["states"], real_success=R1["succ"],
                            decoded_frames_sampled=frames[::20], decoded_frames_idx=np.arange(0, H2 + 1, 20))
        np.savez_compressed(f"{OUT}/latents/{uid}.npz", latents=I["latents"].to(_t.float16).cpu().numpy())
        np.savez_compressed(f"{OUT}/samples/{uid}.npz", t0_clean=first_pair[1], t0_corrupted=first_pair[2],
                            closed_loop_corrupted=np.stack([y for _, y in cl_samples]),
                            closed_loop_steps=np.array([t for t, _ in cl_samples]))
        s1 = R1["succ"]
        rec = dict(unit_id=uid, model="lingbot_va_libero_long", task_id=TID, s0_id=a["anchor_id"], anchor_id=a["anchor_id"],
                   seed_index=k, imagination_seed=seed, seed_master=MASTER, setting_id=s, family="B",
                   perturbation_spec=spec(s), corruption_versions=corr.versions, H_env=H2, criterion="reach_anytime",
                   model_verdict=None, model_verdict_note="judge uncalibrated for this task: pending; latents stored",
                   real_verdict=bool(s1.any()), real_first_success_step=int(np.argmax(s1)) + 1 if s1.any() else None,
                   replay_check_identical=same,
                   closed_loop=dict(success=C1["success"], first_success_step=C1["first_success_step"], steps=C1["steps"],
                                    rerun=dict(success=C2["success"], first_success_step=C2["first_success_step"]),
                                    conclusion_consistent=bool(C1["success"] == C2["success"]),
                                    corrupted_frames=[n1, n2]),
                   applied=dict(open_loop="real t = 0 frame only", closed_loop="every real frame read"),
                   seconds=round(time.time() - t0, 1), machine=os.environ.get("WMV_MACHINE_TAG", os.uname().nodename),
                   gpu=_t.cuda.get_device_name(0))
        open(path, "a").write(json.dumps(rec) + "\n")
        print(uid, rec["real_verdict"], C1["success"], C2["success"], "same", same, rec["seconds"], flush=True)
    print("LB_FAMB_DONE", f"t{TID}", shard)


if __name__ == "__main__":
    main()
