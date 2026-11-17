"""Family O (occlusion) for OpenWAM-alpha, main bank (ruling 2026-10-26). Black square at the centre of the AGENTVIEW
input only (wrist untouched), side = ratio x the raw agentview edge (256 px rendered for the OpenWAM canvas; the
canvas then resizes it with the image, i.e. the same fraction of each edge of the model's 320 x 256 agentview slot):
O1..O5 = 1/16, 1/8, 3/16, 1/4, 5/16. O0 (sham) = openwam_base. Applied to the frame the model reads at t = 0 (open-loop
first frame; closed-loop first call). Same anchors and seeds as openwam_base; open loop 13 calls (calibrated
chaining), truth replay x2, closed loop x2 (FAIL only on a different success verdict).

  python ow_famO.py [sham]   (openwam interpreter)   -> out/runs/openwam_famO/records_t<k>.jsonl
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
import ow_runner as W
from lingbot_famO import RATIOS, occlude

OUT = f"{ROOT}/libero/out/runs/openwam_famO"


def closed_occ(eng, sim, anchor, prompt, seed, ratio):
    from PIL import Image
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
    o = sim.reset(anchor); acts, succ, S = [], [], []
    for c in range(W.N_CALLS):
        img_src = occlude(o["agentview_image"], ratio) if (c == 0 and ratio is not None) else o["agentview_image"]
        img = Image.fromarray(W.canvas_from_raw(img_src, o["robot0_eye_in_hand_image"]))
        out = eng.generate({"first_frame_image": [img], "prompt": prompt, "proprio": W._eef10(o), "seed": seed + c})
        A = [native_eef10_to_libero7d(x) for x in np.asarray(out["actions"], np.float32)[:, :10]][: W.H - len(acts)]
        r = sim.step(A); acts += A; succ += r["success"].tolist(); S.append(r["states"]); o = r
    return np.stack(acts), np.array(succ), np.concatenate(S)


def main():
    W._setup()
    eng, pre, arch, _, ckname = W._engine()
    sim = W.Sim(f"/dev/shm/ow_famO_{os.getpid()}")
    from libero_lang import LANG
    anchors = [(a, a["anchor_id"], None, s) for a, _, _, s in W.anchors_for("base", [])]
    if len(sys.argv) > 1 and sys.argv[1] == "sham":                # ratio None reproduces openwam_base's actions
        ok = []
        for t in ("t5", "t3"):
            r = json.loads(open(f"{ROOT}/libero/out/runs/openwam_base/records_{t}.jsonl").readline())
            a = next(x for x, *_ in anchors if x["anchor_id"] == r["s0_id"])
            o = sim.reset(a); _, env7, _ = W.imagine(eng, o, LANG[int(t[1:])], r["imagination_seed"])
            same = bool(np.array_equal(env7, np.load(f"{ROOT}/{r['action_file']}")["env"])); ok.append(same); print("O sham", r["unit_id"], same, flush=True)
        sim.close(); print("O_SHAM", "PASS" if all(ok) else "FAIL"); sys.exit(0 if all(ok) else 1)
    for sub in ("actions", "trajectories"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    for tid in (5, 3):
        path = f"{OUT}/records_t{tid}.jsonl"
        done = {json.loads(l)["unit_id"] for l in open(path)} if os.path.exists(path) else set()
        for a, s0, _, seeds in anchors:
            if a["task_id"] != tid:
                continue
            for i in range(1, len(RATIOS)):
                for k in range(W.K):
                    uid = f"openwam_{s0}__k{k}__O{i}"
                    if uid in done:
                        continue
                    seed = int(seeds[k]); t0 = time.time(); prompt = LANG[tid]
                    o = sim.reset(a); o["agentview_image"] = occlude(o["agentview_image"], RATIOS[i])
                    frames, env7, t_calls = W.imagine(eng, o, prompt, seed); t_model = time.time() - t0
                    r1 = sim.replay(a, env7); r2 = sim.replay(a, env7)
                    same = bool(np.array_equal(r1["states"], r2["states"]) and np.array_equal(r1["success"], r2["success"]))
                    ca, cs, cS = closed_occ(eng, sim, a, prompt, seed, RATIOS[i]); ca2, cs2, cS2 = closed_occ(eng, sim, a, prompt, seed, RATIOS[i])
                    div, f2, concl = W.cl_compare(ca, cs, cS, ca2, cs2, cS2)
                    S1 = np.concatenate([o["phys"][None], r1["states"]]); S2 = np.concatenate([o["phys"][None], r2["states"]])
                    np.savez_compressed(f"{OUT}/actions/{uid}.npz", env=env7, closed_loop=ca, closed_loop2=ca2)
                    np.savez_compressed(f"{OUT}/trajectories/{uid}.npz", replay1=S1, replay2=S2, real_success=r1["success"],
                                        imagined_canvas=frames, closed_loop_success=cs, closed_loop_states=cS,
                                        closed_loop2_success=cs2, closed_loop2_states=cS2)
                    succ = r1["success"]
                    rec = dict(unit_id=uid, model="openwam_alpha_sim_libero", task_id=tid, s0_id=s0, anchor_id=s0, seed_index=k,
                               imagination_seed=seed, setting_id=f"O{i}", perturbation=f"O{i}",
                               perturbation_spec=dict(family="O", ratio=RATIOS[i], side_px_raw256=int(round(RATIOS[i] * 256)), camera="agentview",
                                                      applied="model input frame at t = 0 (open-loop first frame, closed-loop first call)"),
                               family="O", H_env=W.H, criterion="reach_anytime", chaining="calibrated controller (injected component)",
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
                               n_calls=W.N_CALLS, seconds_model=round(t_model, 1), seconds_total=round(time.time() - t0, 1),
                               machine=os.uname().nodename, checkpoint=ckname)
                    with open(path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(json.dumps(dict(unit=uid, real=rec["real_verdict"], cl=[bool(cs.any()), bool(cs2.any())], sec=rec["seconds_total"])), flush=True)
    sim.close(); print("DONE")


if __name__ == "__main__":
    main()
