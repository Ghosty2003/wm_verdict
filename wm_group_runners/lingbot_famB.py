"""Family B (image degradation, unified protocol v1.0; rulings 2026-10-27) for LingBot-VA, main bank.

Settings (bcorrupt.SETTINGS): gaussian_blur / fog / zoom_blur x severity 2 / 5 / 9 on the raw 128 x 128 agentview
(LingBot-VA renders at 128 natively, so the no-corruption level is the main-bank baseline lingbot_base, checked by the
sham mode). Corruption is applied to obs["pixels"]["image"] - the raw LIBERO agentview, before LeRobot's env
processor and the policy preprocessor - every time the policy reads a real observation: open loop = the real t = 0
frame only (imagined frames are never corrupted, ruling 2); closed loop = every step's real frame, step index = env
steps executed so far. Truth (a) replays the imagined actions (images play no role). Same anchors and imagination seeds
as lingbot_base (MASTER 20261026); imagination to 520, tables truncate to 400; final predicates (t5 v3 per frame,
t3 v4 close / bowl_out) on the decoded frames; latents kept; replay x2; closed loop x2.

  python lingbot_famB.py <shard> <nshards>   -> out/runs/lingbot_famB/records_t<k>.<shard>.jsonl
  python lingbot_famB.py sham                 -> no-corruption path reproduces lingbot_base's imagined actions
"""
import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/mnt/qnap/chenmg93/wm_verdict"
sys.path[:0] = [HERE, ROOT, f"{ROOT}/libero", f"{ROOT}/libero/phys"]
from bcorrupt import SETTINGS, spec, Corruptor

MASTER, K = 20261026, 10
OUT = f"{ROOT}/libero/out/runs/lingbot_famB"


def make_rig_class(LingBotRig):
    class Rig(LingBotRig):
        """corrupts the raw agentview of every observation the policy reads; counter = env steps since make_env."""
        b = None; uid = None; corr = None; log = None

        def make_env(self, anchor):
            env, obs = super().make_env(anchor)
            self._t = 0
            return env, obs

        def obs_to_batch(self, obs):
            if self.b is not None:
                img = obs["pixels"]["image"]; wrist = obs["pixels"]["image2"]
                y = self.corr(img[0], self.b, self.uid, self._t)
                obs = dict(obs); obs["pixels"] = dict(obs["pixels"]); obs["pixels"]["image"] = y[None]
                assert obs["pixels"]["image2"] is wrist                       # wrist camera untouched
                if self.log is not None:
                    self.log.append((self._t, img[0].copy(), y))
            self._t += 1
            return super().obs_to_batch(obs)
    return Rig


def main():
    import torch
    from lingbot_unit import LingBotRig, H
    from lingbot_predicate import Net, predict
    import lingbot_famA as R
    Rig = make_rig_class(LingBotRig)
    bank = json.load(open(f"{ROOT}/libero/out/bank/libero_anchors.json"))
    seeds = np.random.RandomState(MASTER).randint(0, 2 ** 31 - 1, size=(len(bank["anchors"]), K))
    if sys.argv[1] == "sham":                              # b = None reproduces the baseline's imagined actions
        ok = []
        for tid in (5, 3):
            rig = Rig(tid)
            for ai, a in [(ai, a) for ai, a in enumerate(bank["anchors"]) if a["task_id"] == tid][:2]:
                ref = next(json.loads(l) for l in open(f"{ROOT}/libero/out/runs/lingbot_base/records_t{tid}.jsonl")
                           if json.loads(l)["s0_id"] == a["anchor_id"] and json.loads(l)["seed_index"] == 0)
                I = rig.imagine(a, int(seeds[ai, 0]))
                same = bool(np.array_equal(I["env_actions"], np.load(f"{ROOT}/{ref['action_file']}")["env"])); ok.append(same)
                print("B sham", a["anchor_id"], same, flush=True)
            del rig; torch.cuda.empty_cache()
        print("B_SHAM", "PASS" if all(ok) else "FAIL"); sys.exit(0 if all(ok) else 1)
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    for sub in ("actions", "trajectories", "latents", "samples"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    V4 = f"{ROOT}/libero/out/predicate_v4_t3"
    cal4 = {d: json.load(open(f"{V4}/calibration_{d}.json")) for d in ("close", "bowl_out")}
    nets4 = {}
    for d, c in cal4.items():
        n = Net().cuda(); n.load_state_dict(torch.load(f"{ROOT}/{c['artifact']['path']}")); n.eval(); nets4[d] = n
    corr = Corruptor()
    jobs = [(ai, a, k, s) for ai, a in enumerate(bank["anchors"]) for s in SETTINGS[1:] for k in range(K)]
    jobs = [j for n_, j in enumerate(jobs) if n_ % nsh == shard]
    for tid in (5, 3):
        path = f"{OUT}/records_t{tid}.{shard}.jsonl"
        done = {json.loads(l)["unit_id"] for f in os.listdir(OUT) if f.startswith(f"records_t{tid}.") for l in open(f"{OUT}/{f}")}
        todo = [j for j in jobs if j[1]["task_id"] == tid and f"lingbot_{j[1]['anchor_id']}__k{j[2]}__{j[3]}" not in done]
        if not todo:
            continue
        rig, judge = Rig(tid), R.Judge(tid); rig.corr = corr
        for ai, a, k, s in todo:
            uid = f"lingbot_{a['anchor_id']}__k{k}__{s}"; seed = int(seeds[ai, k]); t0 = time.time()
            rig.b, rig.uid, rig.log = s, uid, []
            I = rig.imagine(a, seed); first_pair = rig.log[0]
            rig.b = None
            frames = rig.decode(I["latents"]).numpy()[: H + 1]
            mp = judge.probs(frames)
            extra = dict(p_close=predict(nets4["close"], frames), p_bowl=predict(nets4["bowl_out"], frames)) if tid == 3 else {}
            t_model = time.time() - t0
            R1, R2 = rig.replay(a, I["env_actions"]), rig.replay(a, I["env_actions"])
            rig.b, rig.log = s, []; C1 = rig.closed_loop(a, seed, I["env_actions"]); n_corr1 = len(rig.log)
            cl_samples = [(t, y) for t, _, y in rig.log if t % 50 == 0]
            rig.log = []; C2 = rig.closed_loop(a, seed, I["env_actions"]); n_corr2 = len(rig.log)
            rig.b, rig.log = None, None
            assert n_corr1 == C1["steps"] and n_corr2 == C2["steps"], (n_corr1, C1["steps"], n_corr2, C2["steps"])
            n = min(len(C1["actions"]), len(C2["actions"])); dd = np.where(np.any(C1["actions"][:n] != C2["actions"][:n], axis=1))[0]
            div = int(dd[0]) + 1 if len(dd) else (None if len(C1["actions"]) == len(C2["actions"]) else n + 1)
            np.savez_compressed(f"{OUT}/actions/{uid}.npz", env=I["env_actions"], imagined_norm=I["actions_norm"],
                                closed_loop=C1["actions"], closed_loop2=C2["actions"])
            np.savez_compressed(f"{OUT}/trajectories/{uid}.npz", replay1=R1["states"], replay2=R2["states"], real_success=R1["succ"],
                                model_pred=mp, **extra, decoded_frames_sampled=frames[::20], decoded_frames_idx=np.arange(0, H + 1, 20))
            np.savez_compressed(f"{OUT}/latents/{uid}.npz", latents=I["latents"].to(torch.float16).cpu().numpy())
            np.savez_compressed(f"{OUT}/samples/{uid}.npz", t0_clean=first_pair[1], t0_corrupted=first_pair[2],
                                closed_loop_corrupted=np.stack([y for _, y in cl_samples]), closed_loop_steps=np.array([t for t, _ in cl_samples]))
            s1 = R1["succ"]
            rec = dict(unit_id=uid, model="lingbot_va_libero_long", checkpoint="lerobot/lingbot_va_libero_long", task_id=tid,
                       episode_id=uid, s0_id=a["anchor_id"], anchor_id=a["anchor_id"], initial_state_id=a["anchor_id"],
                       seed_index=k, imagination_seed=seed, setting_id=s, perturbation=s, perturbation_spec=spec(s), family="B",
                       corruption_versions=corr.versions,
                       policy_preprocessing="LeRobot env processor (180 deg rotation) + LingBot-VA preprocessor, after corruption",
                       applied=dict(open_loop="real t = 0 frame only", closed_loop=f"every real frame read ({n_corr1} / {n_corr2} steps)"),
                       H_env=H, model_verdict=None,                     # scored by compare_new with the final rules
                       real_verdict=bool(s1.any()), real_first_success_step=int(np.argmax(s1)) + 1 if s1.any() else None,
                       real_verdict_final=bool(s1[-1]), replay_check_identical=bool(np.array_equal(R1["states"], R2["states"])),
                       closed_loop=dict(success=C1["success"], first_success_step=C1["first_success_step"], steps=C1["steps"],
                                        deterministic_rerun_identical=div is None,
                                        rerun=dict(success=C2["success"], first_success_step=C2["first_success_step"], steps=C2["steps"]),
                                        first_divergence_step=div, conclusion_consistent=C1["success"] == C2["success"]),
                       action_file=os.path.relpath(f"{OUT}/actions/{uid}.npz", ROOT),
                       trajectory_file=os.path.relpath(f"{OUT}/trajectories/{uid}.npz", ROOT),
                       latents_file=os.path.relpath(f"{OUT}/latents/{uid}.npz", ROOT),
                       samples_file=os.path.relpath(f"{OUT}/samples/{uid}.npz", ROOT),
                       machine=os.uname().nodename, seconds_model=round(t_model, 1), seconds_total=round(time.time() - t0, 1))
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(dict(unit=uid, real=rec["real_verdict"], cl=[C1["success"], C2["success"]], sec=rec["seconds_total"])), flush=True)
            torch.cuda.empty_cache()
        del rig, judge; torch.cuda.empty_cache()
    corr.close(); print("DONE")


if __name__ == "__main__":
    main()
