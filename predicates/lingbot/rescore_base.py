"""Main-bank baseline (lingbot_base, both tasks): same-seed deterministic recomputation of the stored LingBot-VA
imaginations, to score every decoded frame with the FINAL predicates (t5 v3, k = 24 in cells; t3 v4 conjunction, k = 1).
Needed to fill the "no perturbation" verdict column of the result tables (ruling 2026-10-26). Same procedure and gates as
rescore_t3_v4.py; NOT a new imagination:
  action gate  recomputed env actions == stored actions/<unit>.npz['env'] (bit-exact)
  frame gate   recomputed decoded frames at the stored sample indices == trajectories/<unit>.npz['decoded_frames_sampled']
Units failing a gate are not scored and are counted; 3 failures -> STOP. Cost stop: 1.5 x (100 units x 85 s).
Stored per unit (out/rescore_base/<unit>.npz): v3 scores (all frames), p_close / p_bowl (t3), latents (float16).
Must run on $NODE_D.

  python rescore_base.py
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
import numpy as np
import torch

_wmvp.bootstrap()
from lingbot_unit import LingBotRig, H
from lingbot_predicate import Net, predict
import lingbot_famA as R

ROOT = _wmvp.P("$WMV_ROOT")
OUT = f"{ROOT}/libero/out/rescore_base"
V4 = f"{ROOT}/libero/out/predicate_v4_t3"
BANK = f"{ROOT}/libero/out/bank/libero_anchors.json"
MASTER, K = 20261026, 10                       # = lingbot_baseline.py
MACHINE = "$NODE_D"
BUDGET_S = 1.5 * 100 * 85
MAX_FAIL = 3


def main():
    assert socket.gethostname() == MACHINE, f"must run on {MACHINE}"
    os.makedirs(OUT, exist_ok=True)
    bank = json.load(open(BANK))
    seeds = np.random.RandomState(MASTER).randint(0, 2 ** 31 - 1, size=(len(bank["anchors"]), K))
    cal4 = {d: json.load(open(f"{V4}/calibration_{d}.json")) for d in ("close", "bowl_out")}
    nets4 = {}
    for d, c in cal4.items():
        n = Net().cuda(); n.load_state_dict(torch.load(f"{ROOT}/{c['artifact']['path']}")); n.eval(); nets4[d] = n
    log_path = f"{OUT}/rescore_log.jsonl"
    rows = [json.loads(l) for l in open(log_path)] if os.path.exists(log_path) else []
    done = {r["unit"] for r in rows}; fails = sum(not r["pass"] for r in rows)
    t_start = time.time() - sum(r["seconds"] for r in rows)
    for tid in (5, 3):
        rig, judge = LingBotRig(tid), R.Judge(tid)                  # Judge = predicate v3 of the task
        for l in open(f"{ROOT}/libero/out/runs/lingbot_base/records_t{tid}.jsonl"):
            rec = json.loads(l); uid = rec["unit_id"]
            if uid in done:
                continue
            ai, a = next((i, x) for i, x in enumerate(bank["anchors"]) if x["anchor_id"] == rec["s0_id"])
            seed = int(seeds[ai, rec["seed_index"]]); assert seed == rec["imagination_seed"], uid
            t0 = time.time()
            I = rig.imagine(a, seed)
            frames = rig.decode(I["latents"]).numpy()[: H + 1]
            tr = np.load(f"{ROOT}/{rec['trajectory_file']}"); idx = tr["decoded_frames_idx"]
            row = dict(unit=uid, task=tid, machine=MACHINE, seed=seed,
                       gate_action=bool(np.array_equal(I["env_actions"], np.load(f"{ROOT}/{rec['action_file']}")["env"])),
                       gate_frame=bool(np.array_equal(frames[idx], tr["decoded_frames_sampled"])))
            row["pass"] = row["gate_action"] and row["gate_frame"]
            if row["pass"]:
                extra = dict(p_close=predict(nets4["close"], frames), p_bowl=predict(nets4["bowl_out"], frames)) if tid == 3 else {}
                np.savez_compressed(f"{OUT}/{uid}.npz", v3=judge.probs(frames), **extra,
                                    latents=I["latents"].to(torch.float16).cpu().numpy())
            row["seconds"] = round(time.time() - t0, 1)
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            fails += not row["pass"]
            if fails >= MAX_FAIL:
                sys.exit(f"STOP: {fails} units failed the equivalence gates")
            if time.time() - t_start > BUDGET_S:
                sys.exit("STOP: cost exceeded 1.5 x the estimate")
            torch.cuda.empty_cache()
        del rig, judge; torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
