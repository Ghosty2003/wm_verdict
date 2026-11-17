"""OpenWAM-alpha formal runs (2026-10-23 ruling, step 1.2 / 1.3): open-loop main track + closed-loop reference, H = 400,
same anchors and the same seed rule as the LingBot-VA runs, pairing protocol.

Per unit (anchor, seed index k, setting):
  model branch   pure imagination: 13 chained calls (calibrated-controller chaining, protocol libero.openwam_alpha),
                 per-call seed = unit seed + call index; all imagined canvases stored (re-scorable by any predicate)
  truth (a)      open-loop replay of the first 400 imagined env actions, twice, bit-identical
  closed (b)     per block of 32: the model reads the REAL canvas + REAL proprio, generates, the simulator executes the 32
                 actions; to H = 400; run twice (same machine, same seed) -> bit-identical check; no verdict column
Model verdicts are added later by the scoring step once the OpenWAM-view predicate is frozen (records field
model_verdict = null until then).

The simulator runs in a lerobot_libero subprocess (sim_server.py); this process runs in the openwam interpreter.

  python ow_runner.py --run base|famA|famP [--levels A0,A1] [--units N]
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
import argparse
import subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
_wmvp.bootstrap()
from ow_view import canvas_from_raw
from openwam_unit import _engine, Chainer, _setup

H = 400
BLOCK = 32
N_CALLS = (H + BLOCK - 1) // BLOCK          # 13
K = 10
LIBERO_PY = os.path.expanduser(_wmvp.P("$CONDA_ROOT/envs/lerobot_libero/bin/python"))
RUNS = {
    "base": dict(bank=f"{ROOT}/libero/out/bank/libero_anchors.json", master=20261026, out=f"{ROOT}/libero/out/runs/openwam_base", family=None),
    "famA": dict(bank=f"{ROOT}/libero/out/bank/familyA/familyA_anchors.json", master=20261031, out=f"{ROOT}/libero/out/runs/openwam_famA", family="A_obj"),
    "famP": dict(bank=f"{ROOT}/libero/out/bank/familyP/familyP_anchors.json", master=20261033, out=f"{ROOT}/libero/out/runs/openwam_famP", family="P_pose"),
}


class Sim:
    def __init__(self, scratch):
        self.scratch = scratch; os.makedirs(scratch, exist_ok=True)
        self.p = subprocess.Popen([LIBERO_PY, "-u", f"{HERE}/sim_server.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(f"{scratch}/sim_server.err", "a"), text=True, bufsize=1,
                                  env=dict(os.environ, MUJOCO_GL="egl"))

    def call(self, **c):
        self.p.stdin.write(json.dumps(c) + "\n"); self.p.stdin.flush()
        for line in self.p.stdout:                    # robosuite / LIBERO print to stdout: skip non-JSON lines
            line = line.strip()
            if line.startswith("{"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("ok"):
                    return r
        raise RuntimeError("sim_server died")

    def reset(self, anchor):
        f = f"{self.scratch}/obs.npz"; self.call(cmd="reset", anchor=anchor, H=H + 100, out=f); return dict(np.load(f))

    def step(self, acts):
        a, f = f"{self.scratch}/acts.npz", f"{self.scratch}/step.npz"
        np.savez(a, actions=np.asarray(acts, np.float32)); self.call(cmd="step", actions=a, out=f); return dict(np.load(f))

    def replay(self, anchor, acts):
        a, f = f"{self.scratch}/acts.npz", f"{self.scratch}/replay.npz"
        np.savez(a, actions=np.asarray(acts, np.float32)); self.call(cmd="replay", anchor=anchor, actions=a, H=H + 100, out=f)
        return dict(np.load(f))

    def close(self):
        try:
            self.p.stdin.write(json.dumps(dict(cmd="close")) + "\n"); self.p.stdin.flush(); self.p.wait(30)
        except Exception:
            self.p.kill()


def _eef10(o):
    from benchmarks.utils.action_conversion import libero_obs_to_eef10
    return libero_obs_to_eef10(o["robot0_eef_pos"], o["robot0_eef_quat"], o["robot0_gripper_qpos"])


def imagine(eng, o, prompt, seed):
    from PIL import Image
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
    img = Image.fromarray(canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"]))
    state = _eef10(o); ch = Chainer("calibrated", state)
    frames, env7, t_calls = [np.asarray(img)], [], []
    for c in range(N_CALLS):
        t0 = time.time()
        out = eng.generate({"first_frame_image": [img], "prompt": prompt, "proprio": state, "seed": seed + c})
        t_calls.append(time.time() - t0)
        A = np.asarray(out["actions"], np.float32)[:, :10]
        frames += [np.asarray(v) for v in out["video"][1:]]
        env7 += [native_eef10_to_libero7d(x) for x in A]
        img = out["video"][-1]; state = ch.next(state, A)
    return np.stack(frames)[: H // 4 + 1], np.stack(env7)[:H], t_calls


def closed_loop(eng, sim, anchor, prompt, seed, imag_out=None):
    from PIL import Image
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
    o = sim.reset(anchor); acts, succ, S = [], [], []
    for c in range(N_CALLS):
        img = Image.fromarray(canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"]))
        out = eng.generate({"first_frame_image": [img], "prompt": prompt, "proprio": _eef10(o), "seed": seed + c})
        if imag_out is not None:   # store-all-data ruling 2026-11-08
            imag_out.append(np.stack([np.asarray(v, dtype=np.uint8) for v in out["video"]]))
        A = [native_eef10_to_libero7d(x) for x in np.asarray(out["actions"], np.float32)[:, :10]][: H - len(acts)]
        r = sim.step(A); acts += A; succ += r["success"].tolist(); S.append(r["states"])
        o = r
    succ = np.array(succ)
    return np.stack(acts), succ, np.concatenate(S)


CL_STEP_TOL = None    # revision 2026-10-25: no hard step tolerance (the +-10 of 2026-10-24 is superseded)


def cl_compare(ca, cs, cS, ca2, cs2, cS2):
    """two closed-loop runs: first step (1-based) where actions or states differ (None = bit-identical), the second
    run's first success step, and conclusion-level consistency = same success verdict (revision 2026-10-25: the step
    difference is recorded as measurement noise, no tolerance)."""
    n = min(len(ca), len(ca2), len(cS), len(cS2))
    diff = np.where(np.any(ca[:n] != ca2[:n], axis=1) | np.any(cS[:n] != cS2[:n], axis=1))[0]
    div = int(diff[0]) + 1 if len(diff) else (None if len(ca) == len(ca2) and len(cS) == len(cS2) else n + 1)
    f1 = int(np.argmax(cs)) + 1 if cs.any() else None; f2 = int(np.argmax(cs2)) + 1 if cs2.any() else None
    same_concl = bool(cs.any() == cs2.any())
    return div, f2, same_concl


def anchors_for(run, levels):
    R = RUNS[run]; bank = json.load(open(R["bank"]))
    if run == "base":
        A = bank["anchors"]
        seeds = np.random.RandomState(R["master"]).randint(0, 2 ** 31 - 1, size=(len(A), K))
        return [(a, a["anchor_id"], None, seeds[i]) for i, a in enumerate(A)]
    bases = sorted({a["base_id"] for a in bank["anchors"]}, key=lambda s: (-int(s.split("_t")[1].split("_")[0]), int(s[-1])))
    seeds = np.random.RandomState(R["master"]).randint(0, 2 ** 31 - 1, size=(len(bases), K))
    out = []
    for tid in (5, 3):
        for L in levels:
            for a in [x for x in bank["anchors"] if x["task_id"] == tid and x["level"] == L]:
                out.append((a, a["base_id"], L, seeds[bases.index(a["base_id"])]))
    return out


def level_spec(a):
    if a.get("family") == "P_pose":
        return dict(familyP_level=a["level"], initialization_noise=a["initialization_noise"])
    if a.get("family"):
        d = a["displacement"]
        return dict(familyA_level=a["level"], object=d["object"], direction=d["direction"], beyond_boundary_m=d["beyond_boundary_m"], bddl_sha256=a["bddl_sha256"])
    return {}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--levels", default="")
    ap.add_argument("--units", type=int, default=None); a_ = ap.parse_args()
    _setup()
    R = RUNS[a_.run]; out = R["out"]
    for sub in ("actions", "trajectories"):
        os.makedirs(f"{out}/{sub}", exist_ok=True)
    eng, pre, arch, t_load, ckname = _engine()
    sim = Sim(f"/dev/shm/ow_sim_{os.getpid()}"); host = socket.gethostname()
    from libero_lang import LANG
    done_new = 0
    for anc, s0, L, seeds in anchors_for(a_.run, [x for x in a_.levels.split(",") if x]):
        tid = anc["task_id"]; path = f"{out}/records_t{tid}.jsonl"
        done = {json.loads(l)["unit_id"] for l in open(path)} if os.path.exists(path) else set()
        for k in range(K):
            uid = f"openwam_{s0}__k{k}" + (f"__{L}" if L else "")
            if uid in done:
                continue
            if a_.units is not None and done_new >= a_.units:
                sim.close(); return
            t0 = time.time(); seed = int(seeds[k])
            o = sim.reset(anc); prompt = LANG[tid]
            frames, env7, t_calls = imagine(eng, o, prompt, seed); t_model = time.time() - t0
            r1 = sim.replay(anc, env7); r2 = sim.replay(anc, env7)
            same = bool(np.array_equal(r1["states"], r2["states"]) and np.array_equal(r1["success"], r2["success"]))
            t_truth = time.time() - t0 - t_model
            ca, cs, cS = closed_loop(eng, sim, anc, prompt, seed); ca2, cs2, cS2 = closed_loop(eng, sim, anc, prompt, seed)
            cl_same = bool(np.array_equal(ca, ca2) and np.array_equal(cS, cS2))
            cl_div, cl_f2, cl_concl = cl_compare(ca, cs, cS, ca2, cs2, cS2)        # 2026-10-24 hot fix: keep both runs
            t_cl = time.time() - t0 - t_model - t_truth
            S1 = np.concatenate([o["phys"][None], r1["states"]]); S2 = np.concatenate([o["phys"][None], r2["states"]])
            np.savez_compressed(f"{out}/actions/{uid}.npz", env=env7, closed_loop=ca, closed_loop2=ca2)
            np.savez_compressed(f"{out}/trajectories/{uid}.npz", replay1=S1, replay2=S2, real_success=r1["success"],
                                imagined_canvas=frames, closed_loop_success=cs, closed_loop_states=cS,
                                closed_loop2_success=cs2, closed_loop2_states=cS2)
            succ = r1["success"]
            rec = dict(unit_id=uid, model="openwam_alpha_sim_libero", task_id=tid, s0_id=s0, anchor_id=anc["anchor_id"],
                       seed_index=k, imagination_seed=seed, setting_id=L or "none", perturbation=L or "none",
                       perturbation_spec=level_spec(anc), family=R["family"], H_env=H, criterion="reach_anytime",
                       chaining="calibrated controller (injected component)", model_verdict=None,
                       real_verdict=bool(succ.any()), real_verdict_final=bool(succ[-1]),
                       real_first_success_step=int(np.argmax(succ)) + 1 if succ.any() else None, real_final_dist=0.0,
                       real_final_state_full=S1[-1].tolist(), replay_check_identical=same,
                       replay2=dict(real_verdict=bool(r2["success"].any()), real_final_dist=0.0, real_final_state_full=S2[-1].tolist()),
                       closed_loop=dict(success=bool(cs.any()), first_success_step=int(np.argmax(cs)) + 1 if cs.any() else None,
                                        steps=int(len(cs)), deterministic_rerun_identical=cl_same,
                                        rerun=dict(success=bool(cs2.any()), first_success_step=cl_f2, steps=int(len(cs2))),
                                        first_divergence_step=cl_div, conclusion_consistent=cl_concl, step_tolerance=CL_STEP_TOL),
                       action_file=os.path.relpath(f"{out}/actions/{uid}.npz", ROOT),
                       trajectory_file=os.path.relpath(f"{out}/trajectories/{uid}.npz", ROOT),
                       n_calls=N_CALLS, seconds_per_call=round(float(np.mean(t_calls)), 3), seconds_model=round(t_model, 1),
                       seconds_truth_a=round(t_truth, 1), seconds_truth_b=round(t_cl, 1), seconds_total=round(time.time() - t0, 1),
                       machine=host, checkpoint=ckname)
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done_new += 1
            print(json.dumps({x: rec[x] for x in ("unit_id", "real_verdict", "replay_check_identical", "seconds_total")}
                             | dict(closed=rec["closed_loop"]["success"], closed_rerun_identical=cl_same)), flush=True)
    sim.close()


if __name__ == "__main__":
    main()
