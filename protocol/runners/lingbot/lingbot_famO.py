"""Family O (occlusion; enters the model input) for LingBot-VA, main bank (ruling 2026-10-26). A black square at the
image centre on the AGENTVIEW input only (wrist untouched), side = ratio x the model's input edge (128 px):
O1..O5 = 1/16, 1/8, 3/16, 1/4, 5/16 (8 / 16 / 24 / 32 / 40 px). O0 (sham) = the main-bank baseline (lingbot_base).
Applied to the frame the model reads at t = 0: the first frame of the open-loop imagination and the first frame of the
closed loop; the truth branch and the predicate's real frames are never occluded.

Per unit (same anchors and imagination seeds as lingbot_base, MASTER 20261026): imagination to 520 (as the baseline;
tables truncate to 400), full decoding scored with the FINAL predicates (t5 v3 per frame; t3 v4 close / bowl_out),
latents kept (float16); truth (a) replay twice; closed loop twice (protocol; FAIL only on a different success verdict).

  python lingbot_famO.py <shard> <nshards>   -> out/runs/lingbot_famO/records_t<k>.<shard>.jsonl
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

RATIOS = [None, 1 / 16, 1 / 8, 3 / 16, 1 / 4, 5 / 16]
MASTER, K = 20261026, 10
OUT = f"{ROOT}/libero/out/runs/lingbot_famO"


def occlude(img, ratio):
    """img [..., H, W, C]: black square of side round(ratio * H) at the centre (copy)."""
    img = np.array(img, copy=True); h = img.shape[-3]; s = int(round(ratio * h)); o = (h - s) // 2
    img[..., o:o + s, o:o + s, :] = 0
    return img


def main():
    import torch
    import mujoco
    from lingbot_unit import LingBotRig, H
    from lingbot_predicate import Net, predict
    import lingbot_famA as R
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    for sub in ("actions", "trajectories", "latents"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    bank = json.load(open(f"{ROOT}/libero/out/bank/libero_anchors.json"))
    seeds = np.random.RandomState(MASTER).randint(0, 2 ** 31 - 1, size=(len(bank["anchors"]), K))
    V4 = f"{ROOT}/libero/out/predicate_v4_t3"
    cal4 = {d: json.load(open(f"{V4}/calibration_{d}.json")) for d in ("close", "bowl_out")}
    nets4 = {}
    for d, c in cal4.items():
        n = Net().cuda(); n.load_state_dict(torch.load(f"{ROOT}/{c['artifact']['path']}")); n.eval(); nets4[d] = n

    class Rig(LingBotRig):
        occ = None                                         # ratio applied to the NEXT make_env observation only

        def make_env(self, anchor):
            env, obs = super().make_env(anchor)
            if self.occ is not None:
                obs = dict(obs); obs["pixels"] = dict(obs["pixels"]); obs["pixels"]["image"] = occlude(obs["pixels"]["image"], self.occ)
                self.occ = None
            return env, obs

    if shard < 0:                                          # sham check: occ = None reproduces the baseline's actions
        ok = []
        for tid in (5, 3):
            rig = Rig(tid)
            for ai, a in [(ai, a) for ai, a in enumerate(bank["anchors"]) if a["task_id"] == tid][:2]:
                ref = next(json.loads(l) for l in open(f"{ROOT}/libero/out/runs/lingbot_base/records_t{tid}.jsonl")
                           if json.loads(l)["s0_id"] == a["anchor_id"] and json.loads(l)["seed_index"] == 0)
                I = rig.imagine(a, int(seeds[ai, 0]))
                same = bool(np.array_equal(I["env_actions"], np.load(f"{ROOT}/{ref['action_file']}")["env"])); ok.append(same)
                print("O sham", a["anchor_id"], same, flush=True)
            del rig; torch.cuda.empty_cache()
        print("O_SHAM", "PASS" if all(ok) else "FAIL"); sys.exit(0 if all(ok) else 1)
    jobs = [(ai, a, k, i) for ai, a in enumerate(bank["anchors"]) for i in range(1, len(RATIOS)) for k in range(K)]
    jobs = [j for n_, j in enumerate(jobs) if n_ % nsh == shard]
    for tid in (5, 3):
        path = f"{OUT}/records_t{tid}.{shard}.jsonl"
        done = {json.loads(l)["unit_id"] for l in open(path)} if os.path.exists(path) else set()
        todo = [j for j in jobs if j[1]["task_id"] == tid and f"lingbot_{j[1]['anchor_id']}__k{j[2]}__O{j[3]}" not in done]
        if not todo:
            continue
        rig, judge = Rig(tid), R.Judge(tid)
        for ai, a, k, i in todo:
            uid = f"lingbot_{a['anchor_id']}__k{k}__O{i}"; seed = int(seeds[ai, k]); t0 = time.time()
            rig.occ = RATIOS[i]; I = rig.imagine(a, seed)
            frames = rig.decode(I["latents"]).numpy()[: H + 1]
            mp = judge.probs(frames)
            extra = dict(p_close=predict(nets4["close"], frames), p_bowl=predict(nets4["bowl_out"], frames)) if tid == 3 else {}
            t_model = time.time() - t0
            R1, R2 = rig.replay(a, I["env_actions"]), rig.replay(a, I["env_actions"])
            rig.occ = RATIOS[i]; C1 = rig.closed_loop(a, seed, I["env_actions"])
            rig.occ = RATIOS[i]; C2 = rig.closed_loop(a, seed, I["env_actions"])
            n = min(len(C1["actions"]), len(C2["actions"])); dd = np.where(np.any(C1["actions"][:n] != C2["actions"][:n], axis=1))[0]
            div = int(dd[0]) + 1 if len(dd) else (None if len(C1["actions"]) == len(C2["actions"]) else n + 1)
            np.savez_compressed(f"{OUT}/actions/{uid}.npz", env=I["env_actions"], imagined_norm=I["actions_norm"],
                                closed_loop=C1["actions"], closed_loop2=C2["actions"])
            np.savez_compressed(f"{OUT}/trajectories/{uid}.npz", replay1=R1["states"], replay2=R2["states"], real_success=R1["succ"],
                                model_pred=mp, **extra, decoded_frames_sampled=frames[::20], decoded_frames_idx=np.arange(0, H + 1, 20))
            np.savez_compressed(f"{OUT}/latents/{uid}.npz", latents=I["latents"].to(torch.float16).cpu().numpy())
            s1 = R1["succ"]
            rec = dict(unit_id=uid, model="lingbot_va_libero_long", task_id=tid, s0_id=a["anchor_id"], anchor_id=a["anchor_id"],
                       seed_index=k, imagination_seed=seed, setting_id=f"O{i}", perturbation=f"O{i}",
                       perturbation_spec=dict(family="O", ratio=RATIOS[i], side_px=int(round(RATIOS[i] * 128)), camera="agentview",
                                              applied="model input frame at t = 0 (imagination first frame, closed-loop first frame)"),
                       family="O", H_env=H, model_verdict=None,                   # scored by compare_models with the final rules
                       real_verdict=bool(s1.any()), real_first_success_step=int(np.argmax(s1)) + 1 if s1.any() else None,
                       real_verdict_final=bool(s1[-1]), replay_check_identical=bool(np.array_equal(R1["states"], R2["states"])),
                       closed_loop=dict(success=C1["success"], first_success_step=C1["first_success_step"], steps=C1["steps"],
                                        deterministic_rerun_identical=div is None,
                                        rerun=dict(success=C2["success"], first_success_step=C2["first_success_step"], steps=C2["steps"]),
                                        first_divergence_step=div, conclusion_consistent=C1["success"] == C2["success"]),
                       action_file=os.path.relpath(f"{OUT}/actions/{uid}.npz", ROOT),
                       trajectory_file=os.path.relpath(f"{OUT}/trajectories/{uid}.npz", ROOT),
                       latents_file=os.path.relpath(f"{OUT}/latents/{uid}.npz", ROOT),
                       machine=os.uname().nodename, seconds_model=round(t_model, 1), seconds_total=round(time.time() - t0, 1))
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(dict(unit=uid, real=rec["real_verdict"], cl=[C1["success"], C2["success"]], sec=rec["seconds_total"])), flush=True)
            torch.cuda.empty_cache()
        del rig, judge; torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
