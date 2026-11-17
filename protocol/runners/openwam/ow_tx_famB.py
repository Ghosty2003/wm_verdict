"""OpenWAM-alpha family B for T2/T3/T4 (queue ruling 2026-11-04): protocol render path (128 x 128 cameras, raw
agentview corrupted, then OpenWAM's own canvas preprocessing). Per task: B0 control (128 render, no corruption,
50 units) + 9 corrupted levels x 50. Open loop corrupts only the real t = 0 frame; closed loop corrupts the real
frame of every one of the 13 calls (x2). Verdict columns pending (no task judge yet).

  WMV_TID=<2|4|6> python ow_tx_famB.py <shard> <nshards>   (openwam interpreter, GPU)
  -> out/runs/openwam_t<tid>/famB/records.<shard>.jsonl
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
from bcorrupt import SETTINGS, spec, Corruptor
import ow_runner as W
from ow_famB import reset128, corrupt, closed_b

TID = int(os.environ.get("WMV_TID", "4"))
MASTER = {2: 20261104, 4: 20261111, 6: 20261112}[TID]
K = 10
BANK = f"{ROOT}/libero/out/bank/libero_anchors_t{TID}.json"
OUT = f"{ROOT}/libero/out/runs/openwam_t{TID}/famB"
PROMPT = {2: "turn on the stove and put the moka pot on it",
          4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
          6: "put the white mug on the plate and put the chocolate pudding to the right of the plate"}


def main():
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    W._setup()
    eng, pre, arch, _, ckname = W._engine()
    sim = W.Sim(f"/dev/shm/ow_famB_t{TID}_{os.getpid()}")
    corr = Corruptor()
    bank = json.load(open(BANK))
    seeds = np.random.RandomState(MASTER).randint(0, 2 ** 31 - 1, size=(16, K))
    for sub in ("actions", "samples"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    path = f"{OUT}/records.{shard}.jsonl"
    done = {json.loads(l)["unit_id"] for f in os.listdir(OUT) if f.startswith("records.") for l in open(f"{OUT}/{f}")}
    levels = ["B0"] + SETTINGS[1:]
    jobs = [(ai, a, k, s) for ai, a in enumerate(bank["anchors"]) for s in levels for k in range(K)]
    jobs = [j for n_, j in enumerate(jobs) if n_ % nsh == shard]
    for ai, a, k, s in jobs:
        uid = f"openwam_{a['anchor_id']}__k{k}__{s}"
        if uid in done:
            continue
        seed = int(seeds[ai, k]); t0 = time.time(); log = []
        o = reset128(sim, a)
        oc = o if s == "B0" else corrupt(o, corr, s, uid, 0, log)
        frames, env7, t_calls = W.imagine(eng, oc, PROMPT[TID], seed)
        r1 = sim.replay(a, env7); r2 = sim.replay(a, env7)
        same = bool(np.array_equal(r1["states"], r2["states"]) and np.array_equal(r1["success"], r2["success"]))
        i1, i2 = [], []
        if s == "B0":
            ca, cs, cS = W.closed_loop(eng, sim, dict(a, resolution=128), PROMPT[TID], seed, imag_out=i1)
            ca2, cs2, _ = W.closed_loop(eng, sim, dict(a, resolution=128), PROMPT[TID], seed, imag_out=i2)
        else:
            ca, cs, cS = closed_b(eng, sim, corr, a, PROMPT[TID], seed, s, uid, None, imag_out=i1)
            ca2, cs2, _ = closed_b(eng, sim, corr, a, PROMPT[TID], seed, s, uid, None, imag_out=i2)
        v1, v2 = bool(np.asarray(cs, bool).any()), bool(np.asarray(cs2, bool).any())
        succ = np.asarray(r1["success"], bool)
        os.makedirs(f"{OUT}/closed_imagined", exist_ok=True)
        np.savez_compressed(f"{OUT}/closed_imagined/{uid}.npz",
                            imagined=np.concatenate(i1), imagined_call_lens=np.array([len(x) for x in i1]),
                            imagined2=np.concatenate(i2), imagined2_call_lens=np.array([len(x) for x in i2]),
                            open_loop_canvas=np.asarray(frames, np.uint8))
        np.savez_compressed(f"{OUT}/actions/{uid}.npz", env=np.asarray(env7, np.float32), closed_loop=np.asarray(ca, np.float32))
        if log:
            np.savez_compressed(f"{OUT}/samples/{uid}.npz", t0_clean=log[0][1], t0_corrupted=log[0][2])
        rec = dict(unit_id=uid, model="openwam_alpha_sim_libero", task_id=TID, s0_id=a["anchor_id"], anchor_id=a["anchor_id"],
                   seed_index=k, imagination_seed=seed, seed_master=MASTER, setting_id=s, family="B",
                   perturbation_spec=None if s == "B0" else spec(s), corruption_versions=corr.versions,
                   render_resolution=128, H_env=W.H, criterion="reach_anytime", checkpoint=ckname,
                   model_verdict=None, model_verdict_note="judge uncalibrated for this task: pending",
                   real_verdict=bool(succ.any()), real_first_success_step=int(np.argmax(succ)) + 1 if succ.any() else None,
                   replay_check_identical=same,
                   closed_loop=dict(success=v1, first_success_step=int(np.argmax(np.asarray(cs, bool))) + 1 if v1 else None,
                                    rerun=dict(success=v2), conclusion_consistent=bool(v1 == v2)),
                   applied=dict(open_loop="real t = 0 frame only (B0: none)", closed_loop="every real frame read (B0: none)"),
                   seconds=round(time.time() - t0, 1), machine=os.uname().nodename)
        open(path, "a").write(json.dumps(rec) + "\n")
        print(uid, rec["real_verdict"], v1, v2, "same", same, rec["seconds"], flush=True)
    sim.close()
    print("OW_FAMB_DONE", f"t{TID}", shard)


if __name__ == "__main__":
    main()
