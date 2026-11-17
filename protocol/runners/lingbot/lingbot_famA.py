"""LIBERO family A (object displacement, batch 1), LingBot-VA: 2 tasks x 5 levels (A0..A4) x 5 native anchors x 10
imaginations. Unit = (base anchor, level, imagination seed k); identical to the baseline unit (lingbot_baseline.py:
pure imagination to H = 520, full decoding, predicate on every decoded frame; truth (a) open-loop replay x2 with the
predicate on the real frames; truth (b) closed loop) except:
  - anchors: native family-A anchors (out/bank/familyA/familyA_anchors.json; native_env.py), s0_id = base anchor id,
    setting_id = level, perturbation_spec = the protocol spec of that level
  - predicate: v3 (protocol libero_lingbot_v3)
  - seeds: imagination seed = RandomState(20261031).randint(0, 2**31-1, (10 base anchors, 10))[base, k], SHARED by all
    levels, so the levels of one (base anchor, k) differ only by the start
Output: out/runs/lingbot_famA/records_t<task>.jsonl (+ actions/, trajectories/). Resumable; one level at a time.

  python lingbot_famA.py --levels A0 [--units N]
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import mujoco
_wmvp.bootstrap()
from lingbot_unit import LingBotRig, H
from lingbot_predicate import Net, predict, model_view, batched

ROOT = _wmvp.P("$WMV_ROOT")
OUT = f"{ROOT}/libero/out/runs/lingbot_famA"
BANK = f"{ROOT}/libero/out/bank/familyA/familyA_anchors.json"
K = 10
MASTER = 20261031


def level_spec(a):
    """the perturbation_spec written into records; must equal protocol libero_lingbot_v3.perturbations.<task>.<level>.spec"""
    if a.get("family") == "P_pose":
        return dict(familyP_level=a["level"], initialization_noise=a["initialization_noise"])
    d = a["displacement"]
    return dict(familyA_level=a["level"], object=d["object"], direction=d["direction"],
                beyond_boundary_m=d["beyond_boundary_m"], bddl_sha256=a["bddl_sha256"])


# family P (batch 2, ruling of 2026-10-15): same unit, own bank / output / seed master
FAMILY = {"A": dict(bank=BANK, out=OUT, master=MASTER, tag="A_obj"),
          "P": dict(bank=f"{ROOT}/libero/out/bank/familyP/familyP_anchors.json", out=f"{ROOT}/libero/out/runs/lingbot_famP",
                    master=20261033, tag="P_pose")}


class Judge:
    def __init__(self, task_id):
        cal = json.load(open(f"{ROOT}/libero/out/predicate_v3/t{task_id}/calibration.json"))
        self.thr = float(cal["threshold"]); self.sha = cal["artifact"]["sha256"]
        self.net = Net().cuda(); self.net.load_state_dict(torch.load(f"{ROOT}/{cal['artifact']['path']}")); self.net.eval()

    def probs(self, frames):
        return predict(self.net, frames)


def run_unit(rig, judge, a, k, seed):
    t0 = time.time()
    I = rig.imagine(a, seed)
    frames = rig.decode(I["latents"]).numpy()[: H + 1]
    mp = judge.probs(frames)
    t_model = time.time() - t0
    env, obs = rig.make_env(a); u = env.envs[0]
    real_frames = [model_view(rig, rig.obs_to_batch(obs))]

    def snap():
        m, d = u._env.sim.model._model, u._env.sim.data._data
        st = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_FULLPHYSICS); return st
    S1, succ = [snap()], []
    for act in I["env_actions"]:
        o, r, term, trunc, info = u.step(act.astype(np.float32))
        succ.append(bool(u._env.check_success())); S1.append(snap())
        real_frames.append(model_view(rig, rig.obs_to_batch(batched(o))))
    env.close()
    S1 = np.stack(S1); succ = np.array(succ)
    R2 = rig.replay(a, I["env_actions"])
    rp = judge.probs(np.stack(real_frames))
    t_truth = time.time() - t0 - t_model
    CL = rig.closed_loop(a, seed, I["env_actions"])
    L = a["level"]; base = a["base_id"]
    uid = f"lingbot_{base}__k{k}__{L}"
    np.savez_compressed(f"{OUT}/actions/{uid}.npz", env=I["env_actions"], imagined_norm=I["actions_norm"],
                        closed_loop=CL["actions"], closed_loop_dev_l2=CL["action_dev_l2"])
    keep = sorted(set(range(0, H + 1, 20)) | set(np.where(mp >= judge.thr)[0][:12].tolist()))
    np.savez_compressed(f"{OUT}/trajectories/{uid}.npz", replay1=S1, replay2=R2["states"], real_success=succ,
                        real_pred=rp, model_pred=mp, decoded_frames_sampled=frames[keep], decoded_frames_idx=np.array(keep))
    mreach = bool(mp[1:].max() >= judge.thr)
    return dict(
        unit_id=uid, model="lingbot_va_libero_long", task=a["task"], task_id=a["task_id"], s0_id=base, anchor_id=a["anchor_id"],
        seed_index=k, imagination_seed=int(seed), setting_id=L, perturbation=L, perturbation_spec=level_spec(a),
        family=a.get("family", "A_obj"), H_env=H, threshold=judge.thr, predicate_sha256=judge.sha, criterion="reach_anytime",
        model_verdict=mreach, model_feat_dist=float(1.0 - mp[1:].max()),
        model_verdict_final=bool(mp[H] >= judge.thr),
        model_first_success_frame=(int(np.argmax(mp[1:] >= judge.thr)) + 1 if mreach else None),
        real_verdict=bool(succ.any()), real_verdict_final=bool(succ[-1]),
        real_first_success_step=(int(np.argmax(succ)) + 1 if succ.any() else None),
        real_final_dist=0.0, real_final_state_full=S1[-1].tolist(),
        real_final_feat_dist=float(1.0 - rp[1:].max()), predicate_on_real_final=bool(rp[1:].max() >= judge.thr),
        predicate_on_real_final_step=bool(rp[-1] >= judge.thr),
        replay_check_identical=bool(np.array_equal(S1, R2["states"])),
        replay2=dict(real_verdict=bool(R2["succ"].any()), real_final_dist=0.0, real_final_state_full=R2["states"][-1].tolist()),
        closed_loop=dict(success=CL["success"], first_success_step=CL["first_success_step"], steps=CL["steps"],
                         action_dev_l2_mean=float(CL["action_dev_l2"].mean()),
                         action_dev_l2_first_chunk=float(CL["action_dev_l2"][:12].mean())),
        action_file=os.path.relpath(f"{OUT}/actions/{uid}.npz", ROOT),
        trajectory_file=os.path.relpath(f"{OUT}/trajectories/{uid}.npz", ROOT),
        n_chunks=I["n_chunks"], seconds_model=round(t_model, 1), seconds_truth_a=round(t_truth, 1),
        seconds_truth_b=round(CL["seconds"], 1), seconds_total=round(time.time() - t0, 1))


def main():
    global OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--levels", required=True); ap.add_argument("--units", type=int, default=None)
    ap.add_argument("--family", choices=["A", "P"], default="A")
    A = ap.parse_args(); levels = A.levels.split(",")
    F = FAMILY[A.family]; OUT = F["out"]
    for sub in ("actions", "trajectories"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    bank = json.load(open(F["bank"]))
    bases = sorted({a["base_id"] for a in bank["anchors"]}, key=lambda s: (-int(s.split("_t")[1].split("_")[0]), int(s[-1])))
    seeds = np.random.RandomState(F["master"]).randint(0, 2 ** 31 - 1, size=(len(bases), K))
    done_new = 0
    for tid in (5, 3):
        rig = LingBotRig(tid); judge = Judge(tid)
        path = f"{OUT}/records_t{tid}.jsonl"
        for L in levels:
            done = {json.loads(l)["unit_id"] for l in open(path)} if os.path.exists(path) else set()
            for a in [x for x in bank["anchors"] if x["task_id"] == tid and x["level"] == L]:
                bi = bases.index(a["base_id"])
                for k in range(K):
                    uid = f"lingbot_{a['base_id']}__k{k}__{L}"
                    if uid in done:
                        continue
                    if A.units is not None and done_new >= A.units:
                        return
                    rec = run_unit(rig, judge, a, k, int(seeds[bi, k]))
                    with open(path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    done_new += 1
                    print(json.dumps({x: rec[x] for x in ("unit_id", "model_verdict", "real_verdict", "replay_check_identical",
                                                          "seconds_total")}), flush=True)
        del rig; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
