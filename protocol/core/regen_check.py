"""Can the stored LingBot-VA imaginations be regenerated bit-exactly on this machine? (prerequisite for re-reading the
t3 imagined frames with predicate v4: only 38 decoded frames per unit were stored.)

For the given units: same anchor, same imagination seed (runner seed rule), imagine + full decode, then compare with
the stored record: env actions (actions/<unit>.npz['env']), the stored decoded frame sample, and the v3 per-frame
model predictions (trajectories/<unit>.npz['model_pred']). Read-only; prints a JSON line per unit.

  python regen_check.py <unit_id> [<unit_id> ...]
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import numpy as np
import torch

_wmvp.bootstrap()
from lingbot_unit import LingBotRig, H
import lingbot_famA as R


def seed_of(fam, base_id, k):
    F = R.FAMILY[fam]
    bank = json.load(open(F["bank"]))
    bases = sorted({a["base_id"] for a in bank["anchors"]}, key=lambda s: (-int(s.split("_t")[1].split("_")[0]), int(s[-1])))
    return int(np.random.RandomState(F["master"]).randint(0, 2 ** 31 - 1, size=(len(bases), R.K))[bases.index(base_id), k]), bank


def main():
    rigs, judges = {}, {}
    for uid in sys.argv[1:]:
        fam = "A" if "_fa" in uid else "P"
        base, k, L = uid[len("lingbot_"):].split("__"); k = int(k[1:])
        tid = int(base.split("_t")[1].split("_")[0])
        seed, bank = seed_of(fam, base, k)
        a = next(x for x in bank["anchors"] if x["base_id"] == base and x["level"] == L)
        out = R.FAMILY[fam]["out"]
        rec = next(json.loads(l) for l in open(f"{out}/records_t{tid}.jsonl") if json.loads(l)["unit_id"] == uid)
        assert rec["imagination_seed"] == seed, (rec["imagination_seed"], seed)
        if tid not in rigs:
            rigs[tid], judges[tid] = LingBotRig(tid), R.Judge(tid)
        rig, judge = rigs[tid], judges[tid]
        I = rig.imagine(a, seed)
        frames = rig.decode(I["latents"]).numpy()[: H + 1]
        mp = judge.probs(frames)
        st_act = np.load(f"{R.ROOT}/{rec['action_file']}")["env"]
        tr = np.load(f"{R.ROOT}/{rec['trajectory_file']}")
        idx = tr["decoded_frames_idx"]
        res = dict(unit=uid, seed=seed,
                   env_actions_identical=bool(np.array_equal(I["env_actions"], st_act)),
                   env_actions_maxdiff=float(np.abs(I["env_actions"] - st_act).max()),
                   sampled_frames_identical=bool(np.array_equal(frames[idx], tr["decoded_frames_sampled"])),
                   sampled_frames_maxdiff=int(np.abs(frames[idx].astype(int) - tr["decoded_frames_sampled"].astype(int)).max()),
                   model_pred_identical=bool(np.array_equal(mp, tr["model_pred"])),
                   model_pred_maxdiff=float(np.abs(mp - tr["model_pred"]).max()),
                   seconds=round(I["seconds_imagine"], 1))
        print(json.dumps(res), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
