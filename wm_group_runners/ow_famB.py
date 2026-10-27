"""Family B (image degradation, unified protocol v1.0; rulings 2026-10-27) for OpenWAM-alpha, main bank.

Protocol render path: cameras rendered at 128 x 128 (sim_server reset resolution=128), the raw agentview corrupted, then
OpenWAM's own preprocessing (canvas_from_raw: 180 deg rotation + LANCZOS resize into the 320 x 256 canvas). This path
differs from the main-bank baseline (rendered at 256), so B has its own no-corruption control B0 (ruling 1: 100 units,
same anchors and seeds; table note: different render path from the main baseline). Corruption only on real camera
frames (ruling 2): open loop = the real t = 0 frame; the 12 chained calls start from OpenWAM's own imagined frames,
which are never corrupted; closed loop = the real frame read at every call (step index = env steps executed so far).
Open loop 13 calls (calibrated chaining), truth replay x2, closed loop x2.

  python ow_famB.py [shard nshards]   (openwam interpreter)   -> out/runs/openwam_famB/records_t<k>[.<shard>].jsonl
"""
import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/mnt/qnap/chenmg93/wm_verdict"
sys.path[:0] = [HERE, f"{ROOT}/libero/openwam"]
import ow_runner as W
from bcorrupt import SETTINGS, spec, Corruptor

OUT = f"{ROOT}/libero/out/runs/openwam_famB"
RES = 128
LEVELS = ["B0"] + SETTINGS[1:]


def reset128(sim, anchor):
    f = f"{sim.scratch}/obs.npz"; sim.call(cmd="reset", anchor=anchor, H=W.H + 100, out=f, resolution=RES)
    o = dict(np.load(f)); assert o["agentview_image"].shape == (RES, RES, 3), o["agentview_image"].shape
    return o


def corrupt(o, corr, s, uid, step, log=None):
    if s == "B0":
        return o
    o = dict(o); y = corr(o["agentview_image"], s, uid, step)
    if log is not None:
        log.append((step, o["agentview_image"].copy(), y))
    o["agentview_image"] = y
    return o


def closed_b(eng, sim, corr, anchor, prompt, seed, s, uid, log):
    from PIL import Image
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
    o = reset128(sim, anchor); acts, succ, S = [], [], []
    for c in range(W.N_CALLS):
        oc = corrupt(o, corr, s, uid, len(acts), log)
        img = Image.fromarray(W.canvas_from_raw(oc["agentview_image"], oc["robot0_eye_in_hand_image"]))
        out = eng.generate({"first_frame_image": [img], "prompt": prompt, "proprio": W._eef10(o), "seed": seed + c})
        A = [native_eef10_to_libero7d(x) for x in np.asarray(out["actions"], np.float32)[:, :10]][: W.H - len(acts)]
        r = sim.step(A); acts += A; succ += r["success"].tolist(); S.append(r["states"]); o = r
    return np.stack(acts), np.array(succ), np.concatenate(S)


def main():
    shard, nsh = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    W._setup()
    eng, pre, arch, _, ckname = W._engine()
    sim = W.Sim(f"/dev/shm/ow_famB_{os.getpid()}")
    corr = Corruptor()
    from libero_lang import LANG
    anchors = [(a, a["anchor_id"], None, s) for a, _, _, s in W.anchors_for("base", [])]
    for sub in ("actions", "trajectories", "samples"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    n_ = -1
    for tid in (5, 3):
        path = f"{OUT}/records_t{tid}.jsonl" if nsh == 1 else f"{OUT}/records_t{tid}.{shard}.jsonl"
        done = {json.loads(l)["unit_id"] for f in os.listdir(OUT) if f.startswith(f"records_t{tid}") for l in open(f"{OUT}/{f}")}
        for a, s0, _, seeds in anchors:
            if a["task_id"] != tid:
                continue
            for s in LEVELS:
                for k in range(W.K):
                    n_ += 1; uid = f"openwam_{s0}__k{k}__{s}"
                    if n_ % nsh != shard or uid in done:
                        continue
                    seed = int(seeds[k]); t0 = time.time(); prompt = LANG[tid]; log = []
                    o = corrupt(reset128(sim, a), corr, s, uid, 0, log)
                    frames, env7, t_calls = W.imagine(eng, o, prompt, seed); t_model = time.time() - t0
                    r1 = sim.replay(a, env7); r2 = sim.replay(a, env7)
                    same = bool(np.array_equal(r1["states"], r2["states"]) and np.array_equal(r1["success"], r2["success"]))
                    cl_log = []
                    ca, cs, cS = closed_b(eng, sim, corr, a, prompt, seed, s, uid, cl_log)
                    ca2, cs2, cS2 = closed_b(eng, sim, corr, a, prompt, seed, s, uid, [])
                    div, f2, concl = W.cl_compare(ca, cs, cS, ca2, cs2, cS2)
                    S1 = np.concatenate([o["phys"][None], r1["states"]]); S2 = np.concatenate([o["phys"][None], r2["states"]])
                    np.savez_compressed(f"{OUT}/actions/{uid}.npz", env=env7, closed_loop=ca, closed_loop2=ca2)
                    np.savez_compressed(f"{OUT}/trajectories/{uid}.npz", replay1=S1, replay2=S2, real_success=r1["success"],
                                        imagined_canvas=frames, closed_loop_success=cs, closed_loop_states=cS,
                                        closed_loop2_success=cs2, closed_loop2_states=cS2)
                    if s != "B0":
                        np.savez_compressed(f"{OUT}/samples/{uid}.npz", t0_clean=log[0][1], t0_corrupted=log[0][2],
                                            closed_loop_corrupted=np.stack([y for _, _, y in cl_log]),
                                            closed_loop_steps=np.array([t for t, _, _ in cl_log]))
                    succ = r1["success"]
                    psp = dict(family="B", level="B0", corruption=None, render_resolution=RES, camera="agentview_image") if s == "B0" else spec(s)
                    rec = dict(unit_id=uid, model="openwam_alpha_sim_libero", checkpoint=ckname, task_id=tid, episode_id=uid,
                               s0_id=s0, anchor_id=s0, initial_state_id=s0, seed_index=k, imagination_seed=seed,
                               setting_id=s, perturbation=s, perturbation_spec=psp, family="B",
                               corruption_versions=corr.versions,
                               policy_preprocessing="OpenWAM canvas_from_raw (180 deg rotation + LANCZOS resize 128 -> canvas), after corruption",
                               applied=dict(open_loop="real t = 0 frame only; chained imagined frames untouched",
                                            closed_loop=f"real frame at each of {len(cl_log)} calls" if s != "B0" else "none"),
                               H_env=W.H, criterion="reach_anytime", chaining="calibrated controller (injected component)",
                               model_verdict=None, real_verdict=bool(succ.any()), real_verdict_final=bool(succ[-1]),
                               real_first_success_step=int(np.argmax(succ)) + 1 if succ.any() else None, real_final_dist=0.0,
                               real_final_state_full=S1[-1].tolist(), replay_check_identical=same,
                               replay2=dict(real_verdict=bool(r2["success"].any()), real_final_dist=0.0, real_final_state_full=S2[-1].tolist()),
                               closed_loop=dict(success=bool(cs.any()), first_success_step=int(np.argmax(cs)) + 1 if cs.any() else None,
                                                steps=int(len(cs)), deterministic_rerun_identical=div is None,
                                                rerun=dict(success=bool(cs2.any()), first_success_step=f2, steps=int(len(cs2))),
                                                first_divergence_step=div, conclusion_consistent=concl),
                               action_file=os.path.relpath(f"{OUT}/actions/{uid}.npz", ROOT),
                               trajectory_file=os.path.relpath(f"{OUT}/trajectories/{uid}.npz", ROOT),
                               samples_file=os.path.relpath(f"{OUT}/samples/{uid}.npz", ROOT) if s != "B0" else None,
                               n_calls=W.N_CALLS, seconds_model=round(t_model, 1), seconds_total=round(time.time() - t0, 1),
                               machine=os.uname().nodename)
                    with open(path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(json.dumps(dict(unit=uid, real=rec["real_verdict"], cl=[bool(cs.any()), bool(cs2.any())], sec=rec["seconds_total"])), flush=True)
    corr.close(); sim.close(); print("DONE")


if __name__ == "__main__":
    main()
