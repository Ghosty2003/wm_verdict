"""WorldVLA-view predicate: generated-frame row (appendix only; WorldVLA = exclusion E4; 2026-10-22 ruling).

Same construction as the LingBot one-chunk row: start from a REAL frame of a calibration episode (WorldVLA view,
256 px), let the WorldVLA world model generate the next 10 frames (one WorldVLA action chunk) driven by the episode's
REAL executed actions, and label generated frame k by the simulator state of that episode at the same step. Greedy WM
decoding, constrained retry on a malformed grid (the imagination loop's rule). Scored with the frozen WorldVLA-view
predicate; raw and timing-window rule (+-10 steps around the episode's first success) both reported; t5 / t3 separate.

Starts: every v3 calibration episode (recal_v3.registry), one start each -- success episodes at first success - 12
(the 10 frames straddle the success step), failed episodes at the end (failure-terminal). t3 additionally: the 40
predicate-v4 close-empty episodes (drawer closed, bowl outside), one start at the step the drawer closes -> the
"drawer closed" subset (the v3-recipe blind spot); their REAL WorldVLA-view frames are scored as well.
Budget: <= 4 h GPU in total (eval_design section 21); the gen steps stop themselves (t5 1.6 h, t3 2.0 h) and record what they have.

  prep  <task>   (lerobot_libero) start frames + actions + labels -> out/predicate_wv/genrow/starts_t<task>.npz
  gen   <task>   (worldvla)       WM generation -> out/predicate_wv/genrow/gen_t<task>.npz
  score <task>   (lerobot_libero) predicate rows -> out/predicate_wv/genrow/genrow_t<task>.json
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
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
WV = f"{ROOT}/libero/out/predicate_wv"
OUT = f"{WV}/genrow"
V4 = f"{ROOT}/libero/out/predicate_v4_t3"
N_GEN = 10
WINDOW = 10
BUDGET_S = {5: 1.6 * 3600, 3: 2.0 * 3600}     # gen steps together <= 3.6 h, + prep / load / scoring < 4 h


def _rot(raw):
    return np.ascontiguousarray(np.asarray(raw)[::-1, ::-1])


def prep(tid):
    _wmvp.bootstrap()
    from recal_v3 import registry, _displace_fixed
    os.makedirs(OUT, exist_ok=True)
    S = dict(episode=[], source=[], t0=[], frame256=[], actions=[], labels=[], close=[], failed=[], first_success=[])
    for e in registry(tid):
        z = np.load(e["path"]); acts = z["actions"]; lab = np.load(f"{WV}/t{tid}/frames/{e['id']}.npz")["labels"]
        rt = np.load(f"{WV}/t{tid}/frames_rt/{e['id']}.npz"); idx = list(rt["idx"])
        n = len(acts)
        want = int(np.argmax(lab)) - 12 if lab.any() else n - N_GEN - 1
        cand = [i for i in idx if 0 <= i <= n - N_GEN]
        t0 = min(cand, key=lambda i: abs(i - want))
        S["episode"].append(e["id"]); S["source"].append(e["source"]); S["t0"].append(t0)
        S["frame256"].append(rt["frames"][idx.index(t0)]); S["actions"].append(acts[t0:t0 + N_GEN].astype(np.float64))
        S["labels"].append(lab[t0 + 1:t0 + N_GEN + 1]); S["close"].append(np.full(N_GEN, -1, np.int8))
        S["failed"].append(not lab.any()); S["first_success"].append(int(np.argmax(lab)) if lab.any() else -1)
    real_closed = None
    if tid == 3:                                      # v4 close-empty episodes: render WorldVLA view (256) by replay
        from libero.libero import benchmark
        from lerobot.envs.libero import LiberoEnv
        suite = benchmark.get_benchmark_dict()["libero_10"]()
        RP, RL, RC = [], [], []
        for f in sorted(x for x in os.listdir(f"{V4}/episodes") if x.startswith("syn_close_empty")):
            y = np.load(f"{V4}/episodes/{f}"); meta = json.loads(str(y["meta"]))
            bank = json.load(open(f"{ROOT}/libero/out/bank/libero_anchors.json"))
            tpl = [a for a in bank["anchors"] if a["task_id"] == 3][0]
            env = LiberoEnv(task_suite=suite, task_id=3, task_suite_name="libero_10", observation_width=256, observation_height=256,
                            init_states=True, episode_index=meta["init_index"], n_envs=1, num_steps_wait=tpl["settle_steps"],
                            hard_reset=True, episode_length=600)
            obs, _ = env.reset(seed=meta["seed"])
            _displace_fixed(env, meta["displacement"]["object"], meta["displacement"]["offset_xy_m"])
            for _ in range(10):
                obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
            F, L = [_rot(obs["pixels"]["image"])], [bool(env._env.check_success())]
            for a in y["actions"]:
                obs, *_ = env.step(np.asarray(a, np.float32))
                F.append(_rot(obs["pixels"]["image"])); L.append(bool(env._env.check_success()))
            env.close()
            assert np.array_equal(np.array(L), y["success"]), f   # same trajectory as the v4 episode (render-size independent)
            cl = y["close"].astype(bool); out_ = ~y["in_"].astype(bool)
            ce = np.where(cl & out_)[0]
            if len(ce) and ce[0] <= len(y["actions"]) - N_GEN:
                t0 = int(ce[0])
                S["episode"].append(f[:-4]); S["source"].append("v4_close_empty"); S["t0"].append(t0)
                S["frame256"].append(F[t0]); S["actions"].append(y["actions"][t0:t0 + N_GEN].astype(np.float64))
                S["labels"].append(np.array(L[t0 + 1:t0 + N_GEN + 1])); S["close"].append(cl[t0 + 1:t0 + N_GEN + 1].astype(np.int8))
                S["failed"].append(True); S["first_success"].append(-1)
            for i in np.where(cl)[0]:                  # real frames with the drawer closed (all failures here)
                RP.append(np.asarray(Image.fromarray(F[i]).resize((128, 128), Image.BILINEAR))); RL.append(False); RC.append(bool(out_[i]))
            print(f, "closed frames", int(cl.sum()), flush=True)
        real_closed = dict(frames=np.stack(RP), labels=np.array(RL), bowl_out=np.array(RC))
    np.savez_compressed(f"{OUT}/starts_t{tid}.npz", **{k: np.array(v) for k, v in S.items()})
    if real_closed is not None:
        np.savez_compressed(f"{OUT}/real_closed_t{tid}.npz", **real_closed)
    print(f"t{tid} starts", len(S["t0"]), {s: S["source"].count(s) for s in sorted(set(S["source"]))})


def gen(tid):
    _wmvp.bootstrap()
    import worldvla_unit as W
    z = np.load(f"{OUT}/starts_t{tid}.npz")
    path = f"{OUT}/gen_t{tid}.npz"
    done = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else None
    rig = W.Rig(wm_prompt="helper")                     # frozen WM prompt (protocol libero.worldvla.wm_prompt)
    G = {k: list(done[k]) for k in ("frames", "start", "step", "malformed")} if done else dict(frames=[], start=[], step=[], malformed=[])
    start_done = set(int(x) for x in G["start"])
    t_start = time.time(); stopped = None
    for i in range(len(z["t0"])):
        if i in start_done:
            continue
        cur = Image.fromarray(z["frame256"][i])
        for k in range(N_GEN):
            img, _, _ = rig.wm(cur, z["actions"][i][k]); bad = img is None
            for _ in range(W.MAX_RETRY if bad else 0):
                img, _, _ = rig.wm(cur, z["actions"][i][k], constrain=True)
                if img is not None:
                    break
            if img is None:
                break
            G["frames"].append(np.asarray(img.resize((128, 128), Image.BILINEAR))); G["start"].append(i); G["step"].append(k)
            G["malformed"].append(bad); cur = img
        if (i + 1) % 10 == 0:
            np.savez_compressed(path, **{k: np.array(v) for k, v in G.items()})
            print(f"t{tid} start {i + 1}/{len(z['t0'])}  {time.time() - t_start:.0f}s", flush=True)
        if time.time() - t_start > BUDGET_S[tid]:
            stopped = f"budget: stopped after start {i + 1}/{len(z['t0'])}"; break
    np.savez_compressed(path, **{k: np.array(v) for k, v in G.items()})
    json.dump(dict(task=tid, seconds=time.time() - t_start, stopped=stopped, n_frames=len(G["frames"]),
                   malformed=int(np.sum(G["malformed"]))), open(f"{OUT}/gen_t{tid}_log.json", "w"), indent=1)
    print("GEN DONE", tid, stopped, len(G["frames"]))


def score(tid):
    import torch
    _wmvp.bootstrap()
    from wv_train import NetWV, predict, row, fail_terminal_mask
    from lingbot_predicate import metrics
    cal = json.load(open(f"{WV}/t{tid}/calibration.json")); thr = cal["threshold"]
    net = NetWV().cuda(); net.load_state_dict(torch.load(f"{ROOT}/{cal['artifact']['path']}")); net.eval()
    z = np.load(f"{OUT}/starts_t{tid}.npz"); g = np.load(f"{OUT}/gen_t{tid}.npz")
    st, k = g["start"].astype(int), g["step"].astype(int)
    p = predict(net, g["frames"])
    y = np.array([bool(z["labels"][i][j]) for i, j in zip(st, k)])
    ft = np.array([bool(z["failed"][i]) for i in st])
    absstep = np.array([z["t0"][i] + j + 1 for i, j in zip(st, k)]); fs = np.array([z["first_success"][i] for i in st])
    win = (~y) & (fs >= 0) & (np.abs(absstep - fs) <= WINDOW)
    src = np.array([z["source"][i] for i in st])
    main = src != "v4_close_empty"
    res = dict(task=tid, threshold=thr, n_frames=int(len(p)), starts_used=int(len(set(st.tolist()))),
               malformed_frames=int(g["malformed"].sum()),
               raw=row(p[main], y[main], ft[main], thr), timing_window_rule=row(p[main & ~win], y[main & ~win], ft[main & ~win], thr),
               window_steps=WINDOW, gen_log=json.load(open(f"{OUT}/gen_t{tid}_log.json")))
    gate = lambda r: bool(r["tpr"] is not None and r["tpr"] >= 0.85 and r["tnr"] >= 0.85 and (r["fail_terminal"] is None or r["fail_terminal"]["tnr"] >= 0.85))
    res["pass_85_timing_rule"] = gate(res["timing_window_rule"])
    if tid == 3:
        ce = src == "v4_close_empty"
        res["drawer_closed_generated"] = metrics(p[ce], y[ce], thr) if ce.any() else None       # all labels negative: TNR
        rc = np.load(f"{OUT}/real_closed_t{tid}.npz")
        pr = predict(net, rc["frames"])
        res["drawer_closed_real"] = dict(all=metrics(pr, rc["labels"], thr),
                                         bowl_outside=metrics(pr[rc["bowl_out"]], rc["labels"][rc["bowl_out"]], thr))
    json.dump(res, open(f"{OUT}/genrow_t{tid}.json", "w"), indent=1)
    show = {k_: res[k_] for k_ in ("n_frames", "starts_used", "malformed_frames", "pass_85_timing_rule")}
    show.update({r_: {x: res[r_][x] for x in ("n", "tpr", "tnr")} for r_ in ("raw", "timing_window_rule")})
    if tid == 3:
        show["drawer_closed_generated_tnr"] = res["drawer_closed_generated"] and res["drawer_closed_generated"]["tnr"]
        show["drawer_closed_real_tnr"] = res["drawer_closed_real"]["bowl_outside"]["tnr"]
    print("ROW", json.dumps(show))


if __name__ == "__main__":
    {"prep": prep, "gen": gen, "score": score}[sys.argv[1]](int(sys.argv[2]))
