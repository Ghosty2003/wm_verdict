"""t3: same-seed deterministic recomputation of the stored LingBot-VA imaginations, to score all 521 frames with the
frozen predicate-v4 detectors (2026-10-21 ruling). NOT a new imagination: every unit must pass two equivalence gates
against the stored record, otherwise it is discarded (not scored) and counted; 3 failed units -> STOP.
  action gate  recomputed env actions == stored actions/<unit>.npz['env'] (bit-exact; these are the actions the truth
               branch replayed, so this gate is the pairing check)
  frame gate   recomputed decoded frames at the stored sample indices == trajectories/<unit>.npz['decoded_frames_sampled']
Also reported (not a gate): v3 per-frame predictions vs the stored model_pred.
Stored per unit (out/predicate_v4_t3/rescore/<unit>.npz): p_close, p_bowl (521 frames each), v3 recomputed scores,
key frames (the stored sample indices + up to 12 frames from the first v4 conjunction hit), and the latents
(float16) so that any later predicate change needs a re-decode only. Must run on $NODE_D
(same machine / driver as the original runs). Cost stop: > 1.5 x 11.5 h (500 units), counting the time already logged.

  python rescore_t3_v4.py
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
from regen_check import seed_of

ROOT = _wmvp.P("$WMV_ROOT")
V4 = f"{ROOT}/libero/out/predicate_v4_t3"
OUT = f"{V4}/rescore"
MACHINE = "$NODE_D"
BUDGET_S = 1.5 * 11.5 * 3600      # 2026-10-22: estimate corrected to 500 units (famA 250 + famP 250), ~83 s each; time already logged counts
MAX_FAIL = 3


def main():
    host = socket.gethostname()
    assert host == MACHINE, f"must run on {MACHINE}, got {host}"
    os.makedirs(OUT, exist_ok=True)
    cal = {d: json.load(open(f"{V4}/calibration_{d}.json")) for d in ("close", "bowl_out")}
    assert all(c["gate_pass"] for c in cal.values()), "v4 detectors not frozen (frame-level gate not passed)"
    nets = {}
    for d in cal:
        n = Net().cuda(); n.load_state_dict(torch.load(f"{ROOT}/{cal[d]['artifact']['path']}")); n.eval(); nets[d] = n
    rig, judge = LingBotRig(3), R.Judge(3)
    log_path = f"{OUT}/rescore_log.jsonl"
    done = {json.loads(l)["unit"] for l in open(log_path)} if os.path.exists(log_path) else set()
    fails = sum(1 for l in open(log_path) if not json.loads(l)["pass"]) if os.path.exists(log_path) else 0
    used = sum(json.loads(l)["seconds"] for l in open(log_path)) if os.path.exists(log_path) else 0.0
    t_start = time.time() - used
    for fam, run in (("A", "lingbot_famA"), ("P", "lingbot_famP")):
        for l in open(f"{ROOT}/libero/out/runs/{run}/records_t3.jsonl"):
            rec = json.loads(l); uid = rec["unit_id"]
            if uid in done:
                continue
            base, k, L = uid[len("lingbot_"):].split("__"); k = int(k[1:])
            seed, bank = seed_of(fam, base, k)
            assert seed == rec["imagination_seed"], uid
            a = next(x for x in bank["anchors"] if x["base_id"] == base and x["level"] == L)
            t0 = time.time()
            I = rig.imagine(a, seed)
            frames = rig.decode(I["latents"]).numpy()[: H + 1]
            st_act = np.load(f"{ROOT}/{rec['action_file']}")["env"]
            tr = np.load(f"{ROOT}/{rec['trajectory_file']}")
            idx = tr["decoded_frames_idx"]
            gate_action = bool(np.array_equal(I["env_actions"], st_act))
            gate_frame = bool(np.array_equal(frames[idx], tr["decoded_frames_sampled"]))
            row = dict(unit=uid, machine=host, seed=seed, gate_action=gate_action, gate_frame=gate_frame,
                       pass_=None, seconds=None)
            if gate_action and gate_frame:
                mp = judge.probs(frames)
                pc, pb = predict(nets["close"], frames), predict(nets["bowl_out"], frames)
                conj = (pc >= cal["close"]["threshold"]) & (pb < cal["bowl_out"]["threshold"])
                hit = np.where(conj[1:])[0]
                keep = sorted(set(idx.tolist()) | (set(range(int(hit[0]) + 1, min(int(hit[0]) + 13, H + 1))) if len(hit) else set()))
                np.savez_compressed(f"{OUT}/{uid}.npz", p_close=pc, p_bowl=pb, v3_recomputed=mp,
                                    key_frames=frames[keep], key_frames_idx=np.array(keep),
                                    latents=I["latents"].to(torch.float16).cpu().numpy())   # protocol lesson: re-decodable without re-imagining
                row["v3_scores_identical"] = bool(np.array_equal(mp, tr["model_pred"]))
            row["pass"] = gate_action and gate_frame
            row["seconds"] = round(time.time() - t0, 1)
            del row["pass_"]
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if not row["pass"]:
                fails += 1
                if fails >= MAX_FAIL:
                    print(f"STOP: {fails} units failed the equivalence gates"); return
            if time.time() - t_start > BUDGET_S:
                print("STOP: cost exceeded 1.5 x the 6 h estimate"); return
            torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
