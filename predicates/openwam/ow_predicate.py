"""OpenWAM-alpha-view success predicate (2026-10-23 ruling, step 1.1). Same recipe as LingBot-VA's: t5 = v3 (single
success detector), t3 = v4 conjunction (Close detector AND NOT bowl-outside-visible detector; VIS_MIN and labels of
predicate_v4_t3). Three gate rows + the drawer-closed subset (t3), and a trajectory-level k selected on held-out REAL
trajectories (halves, A selects / B confirms; grid fixed in advance). Not passing -> the verdict column stays pending.

Input view: ow_view.pred_view of the OpenWAM canvas (agentview | wrist, 128 x 256) -- same network as the LingBot-VA
predicates, weights trained on OpenWAM-view frames. Training data = the SAME episodes as the LingBot-VA predicates
(t5: v3 calibration set; t3: v4 training pool), re-rendered at 256 px through the OpenWAM canvas.

Rows (out-of-fold, folds = the source registry's episode folds):
  real        every frame of every episode
  roundtrip   every 6th frame through OpenWAM's own Wan VAE (encode -> decode)
  generated   one OpenWAM call (32 actions, 9 frames) from 3 real starts per episode, the model's OWN actions executed
              in the simulator from that start to label frame j (= step 4j); timing window = one OpenWAM block (32
              steps) around the episode's real first success, raw and rule both reported

  render <tid>      (lerobot_libero)  -> out/t<tid>/episodes/<id>.npz
  rt_gen <tid>      (openwam)         -> out/t<tid>/roundtrip.npz, gen_out.npz
  gen_label <tid>   (lerobot_libero)  -> out/t<tid>/gen_all.npz
  train <tid>       (lerobot_libero)  -> out/t<tid>/calibration*.json, *.pt
  traj_render       (lerobot_libero)  famA/famP truth (a) replays in the OpenWAM view -> out/traj/<unit>.npz
  traj <tid>        (lerobot_libero)  trajectory-level k (A selects, B confirms) -> out/t<tid>/trajectory_eval.json
"""
import os as _wmv_os, sys as _wmv_sys  # release path bootstrap: repository root = nearest parent holding wmv_paths.py
_wmv_d = _wmv_os.path.dirname(_wmv_os.path.abspath(__file__))
while not _wmv_os.path.exists(_wmv_os.path.join(_wmv_d, "wmv_paths.py")) and _wmv_d != _wmv_os.sep: _wmv_d = _wmv_os.path.dirname(_wmv_d)
_wmv_sys.path.insert(0, _wmv_d); import wmv_paths as _wmvp; _wmvp.bootstrap()  # noqa: E402
import os
import sys
import json
import hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _wmvp.P("$WMV_ROOT")
OUT = f"{HERE}/predicate"
_wmvp.bootstrap()
from ow_view import canvas_from_raw, pred_view, raw_obs

RT_EVERY = 6
N_GEN = 32
WINDOW = 32
GATE = 0.85
KGRID = (1, 2, 3, 5, 8, 16, 24, 36, 48)
SPLIT_SEED = {5: 20261037, 3: 20261038}
GEN_SEED = 20261039
H = 400


def _registry(tid):
    if tid == 5:
        from recal_v3 import registry
        return registry(5)
    from recal_v4_t3 import registry
    return registry()


def _env(anc, n):
    from libero_env_adapter import _make_env
    return _make_env(dict(anc, resolution=256), H=n + 20)


def _restore(anc, disp, n):
    from recal_v3 import _displace_fixed
    env = _env(anc, n); u = env.envs[0]
    if disp:
        _displace_fixed(u, disp["object"], disp["offset_xy_m"])
        for _ in range(10):
            u.step(np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
    return env, u


def _starts(lab, n):
    if lab.any():
        fs = int(np.argmax(lab)); s = np.clip(np.array([fs - 36, fs - 12, fs + 12]), 1, n - N_GEN - 1)
    else:
        s = np.linspace(n // 2, n - N_GEN - 1, 3).astype(int)
    return sorted(set(int(x) for x in s))


# ------------------------------------------------------------------------------------------------ render (env side)
def render(tid):
    from libero.libero import benchmark
    lang = benchmark.get_benchmark_dict()["libero_10"]().get_task(tid).language
    d = f"{OUT}/t{tid}/episodes"; os.makedirs(d, exist_ok=True)
    mism = 0
    for e in _registry(tid):
        path = f"{d}/{e['id']}.npz"
        if os.path.exists(path):
            continue
        z = np.load(e["path"]); acts = z["actions"]; lab = z["labels"] if "labels" in z.files else z["success"]
        env, u = _restore(e["anc"], e.get("disp"), len(acts))
        starts = _starts(lab.astype(bool), len(acts))
        F, L, rt, rt_idx, gc, gp = [], [], [], [], [], []
        for t in range(len(acts) + 1):
            if t:
                u.step(np.asarray(acts[t - 1], np.float32))
            o = raw_obs(u); cv = canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"])
            F.append(pred_view(cv)); L.append(bool(u._env.check_success()))
            if t % RT_EVERY == 0:
                rt.append(cv); rt_idx.append(t)
            if t in starts:
                gc.append(cv); gp.append(np.concatenate([o["robot0_eef_pos"], o["robot0_eef_quat"], o["robot0_gripper_qpos"]]))
        env.close()
        L = np.array(L); same = bool(np.array_equal(L, lab.astype(bool)))
        mism += not same
        extra = {k: z[k] for k in ("close", "in_", "px_agent", "px_wrist", "unocc_agent", "unocc_wrist") if k in z.files}
        np.savez_compressed(path, frames=np.stack(F), labels=L, labels_match=same, rt_canvas=np.stack(rt), rt_idx=np.array(rt_idx),
                            gen_t0=np.array(starts), gen_canvas=np.stack(gc), gen_proprio_raw=np.stack(gp), actions=acts,
                            language=np.array(lang), **extra)
        print(f"t{tid} {e['id']} T={len(L)} labels_match={same}", flush=True)
    print(f"t{tid} render done; label mismatches {mism}")


# --------------------------------------------------------------------------------------------- rt_gen (model side)
def rt_gen(tid):
    sys.path[:0] = [_wmvp.P("$OPENWAM_REPO"), _wmvp.P("$OPENWAM_REPO/third_party")]
    os.environ.setdefault("WAM_ATTENTION_IMPL", "sdpa")
    import torch
    from PIL import Image
    from openwam_unit import _engine
    from openwam.model.video_backbone.wan import encode as wan_encode
    from benchmarks.utils.action_conversion import libero_obs_to_eef10
    from benchmarks.libero.openwam2libero_interface import native_eef10_to_libero7d
    eng, pre, arch, _, _ = _engine()
    vb = arch.video_backbone
    d = f"{OUT}/t{tid}"
    rs = np.random.RandomState(GEN_SEED + tid)
    RT, RE, RI, GF, GA, GE, GT = [], [], [], [], [], [], []
    for f in sorted(os.listdir(f"{d}/episodes")):
        z = np.load(f"{d}/episodes/{f}"); eid = f[:-4]
        with torch.no_grad():
            for cv, i in zip(z["rt_canvas"], z["rt_idx"]):
                t = wan_encode.preprocess_video([Image.fromarray(cv)], encoder=vb.video_encoder, dtype=vb.dtype, device=vb.device)
                lat = wan_encode.encode_video(t.to(vb.device), vae=vb.vae, encoder=vb.video_encoder)
                RT.append(pred_view(np.asarray(vb.decode_video(lat, tiled=True)[0]))); RE.append(eid); RI.append(int(i))
        for cv, pr, t0 in zip(z["gen_canvas"], z["gen_proprio_raw"], z["gen_t0"]):
            st = libero_obs_to_eef10(pr[:3], pr[3:7], pr[7:9])
            out = eng.generate({"first_frame_image": [Image.fromarray(cv)], "prompt": str(z["language"]), "proprio": st,
                                "seed": int(rs.randint(2 ** 31 - 1))})
            A = np.asarray(out["actions"], np.float32)[:, :10]
            GA.append(np.stack([native_eef10_to_libero7d(x) for x in A])); GF.append(np.stack([pred_view(np.asarray(v)) for v in out["video"][1:]]))
            GE.append(eid); GT.append(int(t0))
        print(f"t{tid} {eid} rt {len(z['rt_idx'])} gen {len(z['gen_t0'])}", flush=True)
    np.savez_compressed(f"{d}/roundtrip.npz", frames=np.stack(RT), episode=np.array(RE), idx=np.array(RI))
    np.savez_compressed(f"{d}/gen_out.npz", frames=np.stack(GF), actions=np.stack(GA), episode=np.array(GE), t0=np.array(GT))


# ------------------------------------------------------------------------------------------- gen_label (env side)
def _label_start(args):
    """one generated start: replay the episode prefix, execute the model's own 32 actions, per-step labels.
    Runs in a short-lived worker process (simulator / renderer memory is not fully released by env.close())."""
    tid, eid, t0, gen_actions = args
    d = f"{OUT}/t{tid}"; e = {x["id"]: x for x in _registry(tid)}[eid]
    acts = np.load(f"{d}/episodes/{eid}.npz")["actions"]
    env, u = _restore(e["anc"], e.get("disp"), len(acts) + N_GEN)
    for a in acts[:t0]:
        u.step(np.asarray(a, np.float32))
    if tid == 3:
        from recal_v4_t3 import Labeler
        Lb = Labeler(u)
    labs = []
    for a in gen_actions:
        u.step(np.asarray(a, np.float32))
        labs.append(Lb() if tid == 3 else dict(success=bool(u._env.check_success())))
    if tid == 3:
        Lb.close()
    env.close()
    return labs


def gen_label(tid):
    import multiprocessing as mp
    d = f"{OUT}/t{tid}"; g = np.load(f"{d}/gen_out.npz")
    keys = ("success", "close", "in_", "px_agent", "px_wrist", "unocc_agent", "unocc_wrist") if tid == 3 else ("success",)
    LB = {k: [] for k in keys}; FR, EP, T0, STEP = [], [], [], []
    G_frames, G_actions, G_ep, G_t0 = g["frames"], g["actions"], g["episode"], g["t0"]   # read each array ONCE (NpzFile
    # re-decompresses the whole array on every key access; slicing those copies kept every copy alive -> OOM, 2026-10-24)
    jobs = [(tid, str(G_ep[i]), int(G_t0[i]), G_actions[i]) for i in range(len(G_ep))]
    with mp.get_context("spawn").Pool(1, maxtasksperchild=20) as pool:
        all_labs = pool.map(_label_start, jobs, chunksize=1)
    for i, labs in enumerate(all_labs):
        eid, t0 = jobs[i][1], jobs[i][2]
        for j in range(1, len(G_frames[i]) + 1):              # frame j = after 4j actions
            FR.append(G_frames[i][j - 1]); EP.append(eid); T0.append(t0); STEP.append(t0 + 4 * j)
            for k in keys:
                LB[k].append(labs[4 * j - 1][k])
    np.savez_compressed(f"{d}/gen_all.npz", frames=np.stack(FR), episode=np.array(EP), t0=np.array(T0), step=np.array(STEP),
                        **{k: np.array(v) for k, v in LB.items()})
    print(f"t{tid} gen_all {len(FR)}")


# ------------------------------------------------------------------------------------------------ train
def train(tid):
    import torch
    from lingbot_predicate import metrics
    from recal_v3 import _fit, _row
    from recal_predicate import fail_terminal_mask
    from recal_v4_t3 import _threshold, _fit_strat, targets
    d = f"{OUT}/t{tid}"
    E = _registry(tid); fold_of = {e["id"]: e["fold"] for e in E}
    X, LAB, M, EP = [], {}, [], []
    for e in E:
        z = np.load(f"{d}/episodes/{e['id']}.npz"); X.append(z["frames"]); M.append(fail_terminal_mask(z["labels"]))
        EP += [e["id"]] * len(z["labels"])
        for k in (("success",) if tid == 5 else ("success", "close", "in_", "px_agent", "px_wrist")):
            LAB.setdefault(k, []).append(z["labels"] if k == "success" else z[k])
    X = np.concatenate(X); M = np.concatenate(M); EP = np.array(EP); LAB = {k: np.concatenate(v) for k, v in LAB.items()}
    FO = np.array([fold_of[x] for x in EP])
    rt = dict(np.load(f"{d}/roundtrip.npz")); gn = dict(np.load(f"{d}/gen_all.npz"))        # materialise once
    lab_keys = ("labels",) + (() if tid == 5 else ("close", "in_", "px_agent", "px_wrist"))
    ep_lab = {e["id"]: {k: np.load(f"{d}/episodes/{e['id']}.npz")[k] for k in lab_keys} for e in E}
    first_succ = {k: (int(np.argmax(v["labels"])) if v["labels"].any() else None) for k, v in ep_lab.items()}
    failed = {k: v is None for k, v in first_succ.items()}
    rt_lab = {k: np.array([ep_lab[ep][("labels" if k == "success" else k)][i] for ep, i in zip(rt["episode"], rt["idx"])]) for k in LAB}
    rt_ft = np.array([failed[ep] for ep in rt["episode"]]); gn_ft = np.array([failed[ep] for ep in gn["episode"]])
    fo_rt = np.array([fold_of[x] for x in rt["episode"]]); fo_gn = np.array([fold_of[x] for x in gn["episode"]])
    # timing window = the established rule (libero/gen_timing_rule.py): s_ref = first simulated success step of the
    # chunk's OWN continuation, else the source episode's first success; negatives with |s - s_ref| <= one action chunk
    # (OpenWAM: 32 steps) are excluded from the generated row's TNR and reported separately.
    # (2026-10-24 fix: the first version used only the episode's success step.)
    chunk_ref = {}
    for ep, t0, s, y in zip(gn["episode"], gn["t0"], gn["step"], gn["success"]):
        if y and ((ep, t0) not in chunk_ref or s < chunk_ref[(ep, t0)]):
            chunk_ref[(ep, t0)] = int(s)
    win = np.array([(not y) and (chunk_ref.get((ep, t0), first_succ[ep]) is not None)
                    and abs(int(s) - chunk_ref.get((ep, t0), first_succ[ep])) <= WINDOW
                    for ep, t0, s, y in zip(gn["episode"], gn["t0"], gn["step"], gn["success"])])
    res_window_negatives = int(win.sum())
    if tid == 5:
        dets = {"success": (LAB["success"].astype(bool), np.ones(len(X), bool), rt_lab["success"].astype(bool), np.ones(len(rt_ft), bool),
                            gn["success"].astype(bool), np.ones(len(gn_ft), bool))}
        closed = rt_closed = gn_closed = None
    else:
        vis_min = json.load(open(f"{ROOT}/libero/out/predicate_v4_t3/vis_min.json"))["vis_min"]
        tz, trt, tgn = targets(LAB, vis_min), targets(rt_lab, vis_min), targets(gn, vis_min)
        dets = {k: (tz[k][0], tz[k][1], trt[k][0], trt[k][1], tgn[k][0], tgn[k][1]) for k in ("close", "bowl_out")}
        closed, rt_closed, gn_closed = LAB["close"].astype(bool), rt_lab["close"].astype(bool), gn["close"].astype(bool)
    summary = {}
    for name, (Y, V, yr, vr, yg, vg) in dets.items():
        strat = name == "bowl_out"
        oof = np.full(len(Y), np.nan); oof_rt = np.zeros(len(yr)); oof_gn = np.zeros(len(yg)); log = []
        for k in range(5):
            tr = (FO != k) & V
            net, loss = _fit_strat(X[tr], Y[tr], M[tr], closed[tr], seed=k) if strat else _fit(X[tr], Y[tr], M[tr], seed=k)
            from lingbot_predicate import predict
            oof[FO == k] = predict(net, X[FO == k]); oof_rt[fo_rt == k] = predict(net, rt["frames"][fo_rt == k])
            oof_gn[fo_gn == k] = predict(net, gn["frames"][fo_gn == k])
            log.append(dict(fold=k, final_loss=loss, oof_std=float(np.nanstd(oof[FO == k])))); print(name, log[-1], flush=True)
            del net; torch.cuda.empty_cache()
        if min(l["oof_std"] for l in log) < 0.05:
            raise SystemExit(f"{name}: training collapsed -- stop and report")
        thr, runs = _threshold(oof[V], Y[V], M[V], closed[V] if strat else None)
        t_ = thr if thr is not None else 0.5
        res = dict(detector=name, task=tid, view="OpenWAM canvas -> pred_view (agentview | wrist, 128 x 256)", threshold=thr,
                   feasible_runs=runs, fold_log=log,
                   holdout_real=_row(oof[V], Y[V], M[V], t_), holdout_vae_roundtrip=_row(oof_rt[vr], yr[vr], rt_ft[vr], t_),
                   generated_raw=_row(oof_gn[vg], yg[vg], gn_ft[vg], t_),
                   generated_timing_rule=_row(oof_gn[vg & ~win], yg[vg & ~win], gn_ft[vg & ~win], t_), window_steps=WINDOW,
                   window_negatives=dict(n=int((vg & win).sum()), read_as_success=int(((oof_gn >= t_) & vg & win).sum()),
                                         reference="chunk's own simulated success, else the episode's first success"))
        if strat:
            res["drawer_closed"] = {r_: metrics(o_[s_], y_[s_], t_) for r_, o_, y_, s_ in
                                    (("holdout_real", oof, Y, V & closed), ("vae_roundtrip", oof_rt, yr, vr & rt_closed),
                                     ("generated", oof_gn, yg, vg & gn_closed))}
        gate = {r_: bool(thr is not None and res[r_]["tpr"] is not None and res[r_]["tpr"] >= GATE and res[r_]["tnr"] >= GATE
                         and (res[r_]["fail_terminal"] is None or res[r_]["fail_terminal"]["tnr"] >= GATE))
                for r_ in ("holdout_real", "holdout_vae_roundtrip", "generated_timing_rule")}
        if strat:
            rc = res["drawer_closed"]["holdout_real"]
            gate["drawer_closed_real"] = bool(thr is not None and rc["tpr"] is not None and rc["tpr"] >= GATE and rc["tnr"] >= GATE)
        res["gate_85"] = gate; res["gate_pass"] = bool(thr is not None and all(gate.values()))
        net, _ = _fit_strat(X[V], Y[V], M[V], closed[V], seed=100) if strat else _fit(X[V], Y[V], M[V], seed=100)
        p = f"{d}/{name}.pt"; torch.save(net.state_dict(), p)
        res["artifact"] = dict(path=os.path.relpath(p, ROOT), sha256=hashlib.sha256(open(p, "rb").read()).hexdigest())
        json.dump(res, open(f"{d}/calibration_{name}.json", "w"), indent=1)
        summary[name] = dict(threshold=thr, gate_pass=res["gate_pass"], gate=gate,
                             rows={r_: {x: res[r_][x] for x in ("n", "tpr", "tnr")} for r_ in ("holdout_real", "holdout_vae_roundtrip", "generated_raw", "generated_timing_rule")},
                             drawer_closed=res.get("drawer_closed"))
        print("RESULT", name, json.dumps(summary[name]), flush=True)
    json.dump(summary, open(f"{d}/calibration_summary.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------------ trajectory level
def _traj_one(args):
    """render one truth (a) replay in the OpenWAM view (short-lived worker process, see _label_start)."""
    from libero_env_adapter import _make_env, _anchor
    run, bank, r = args; path = f"{OUT}/traj/{r['unit_id']}.npz"
    a = dict(_anchor(f"{ROOT}/libero/out/bank/{bank}/{bank}_anchors.json", r["anchor_id"]), resolution=256)
    acts = np.load(f"{ROOT}/{r['action_file']}")["env"][:H]
    env = _make_env(a, H=len(acts)); u = env.envs[0]
    o = raw_obs(u); F = [pred_view(canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"]))]; S = []
    for x in acts:
        u.step(np.asarray(x, np.float32)); S.append(bool(u._env.check_success()))
        o = raw_obs(u); F.append(pred_view(canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"])))
    env.close()
    tr = np.load(f"{ROOT}/{r['trajectory_file']}")
    same = bool(np.array_equal(np.array(S), tr["real_success"][:H]))
    np.savez_compressed(path, frames=np.stack(F), success=np.array(S), matches_record=same)
    return r["unit_id"], same


def traj_render():
    """famA / famP truth (a) replays (both tasks) in the OpenWAM view -> pred_view frames 0..H (validation only)."""
    import multiprocessing as mp
    od = f"{OUT}/traj"; os.makedirs(od, exist_ok=True)
    jobs = []
    for run, bank in (("lingbot_famA", "familyA"), ("lingbot_famP", "familyP")):
        for t in ("t5", "t3"):
            for l in open(f"{ROOT}/libero/out/runs/{run}/records_{t}.jsonl"):
                r = json.loads(l)
                if not os.path.exists(f"{od}/{r['unit_id']}.npz"):
                    jobs.append((run, bank, r))
    with mp.get_context("spawn").Pool(2, maxtasksperchild=20) as pool:
        bad = [u for u, ok in pool.imap_unordered(_traj_one, jobs, chunksize=1) if not ok]
    print("traj_render done", len(jobs), "rendered; success traces not matching the record:", bad)
    if bad:
        raise SystemExit("truth replay in the OpenWAM view does not reproduce the recorded success trace -- stop")


def traj(tid):
    import torch
    from lingbot_predicate import Net, predict
    d = f"{OUT}/t{tid}"
    names = ("success",) if tid == 5 else ("close", "bowl_out")
    cal = {n: json.load(open(f"{d}/calibration_{n}.json")) for n in names}
    assert all(c["gate_pass"] for c in cal.values()), "frame-level gate not passed"
    nets = {}
    for n in names:
        m = Net().cuda(); m.load_state_dict(torch.load(f"{ROOT}/{cal[n]['artifact']['path']}")); m.eval(); nets[n] = m
    diag = {u["unit"]: u for u in json.load(open(f"{ROOT}/libero/out/content_review/t3_subgoal_diag.json"))["units"]} if tid == 3 else {}
    T = []
    for run in ("lingbot_famA", "lingbot_famP"):
        for l in open(f"{ROOT}/libero/out/runs/{run}/records_t{tid}.jsonl"):
            r = json.loads(l); z = np.load(f"{OUT}/traj/{r['unit_id']}.npz")
            if tid == 5:
                ok = predict(nets["success"], z["frames"]) >= cal["success"]["threshold"]
            else:
                ok = (predict(nets["close"], z["frames"]) >= cal["close"]["threshold"]) & (predict(nets["bowl_out"], z["frames"]) < cal["bowl_out"]["threshold"])
            lab = bool(z["success"].any())
            if tid == 3 and not lab and r["unit_id"] not in diag:
                # t3_subgoal_diag types the H = 520 failures; a unit failing only at H = 400 (success in steps 401..520)
                # has no entry -> own stratification type (2026-10-24 fix; the labels themselves are the H = 400 ones)
                assert r["real_verdict"], f"{r['unit_id']}: fails at H = 520 but missing from t3_subgoal_diag"
                typ = "success_after_400"
            elif tid == 3 and not lab:
                typ = "close_empty_long" if diag[r["unit_id"]]["longest_false_alarm"] >= 50 else \
                      "in_open" if diag[r["unit_id"]]["final_in"] and not diag[r["unit_id"]]["final_close"] else "other_fail"
            else:
                typ = "success" if lab else "fail"
            T.append(dict(unit=r["unit_id"], label=lab, type=typ, ok=ok[1:]))
    rs = np.random.RandomState(SPLIT_SEED[tid]); half = {}
    for typ in sorted({t["type"] for t in T}):
        ids = [t["unit"] for t in T if t["type"] == typ]
        for i, u in enumerate(rs.permutation(ids)):
            half[u] = "A" if i < (len(ids) + 1) // 2 else "B"

    def fires(c, k):
        run = 0
        for x in c:
            run = run + 1 if x else 0
            if run >= k:
                return True
        return False
    res = dict(task=tid, H=H, grid=list(KGRID), split_seed=SPLIT_SEED[tid], per_k={}, tried=[], selected_k=None)
    for k in KGRID:
        f = {t["unit"]: fires(t["ok"], k) for t in T}; row = {}
        for h in "AB":
            S = [t for t in T if half[t["unit"]] == h]
            row[h] = dict(tpr=float(np.mean([f[t["unit"]] for t in S if t["label"]])), tnr=float(1 - np.mean([f[t["unit"]] for t in S if not t["label"]])),
                          by_type={ty: float(np.mean([f[t["unit"]] for t in S if t["type"] == ty])) for ty in sorted({t["type"] for t in S})})
        res["per_k"][k] = row; print(f"k={k}", json.dumps(row), flush=True)
    for k in KGRID:
        a_, b_ = res["per_k"][k]["A"], res["per_k"][k]["B"]
        if a_["tpr"] >= GATE and a_["tnr"] >= GATE:
            ok = b_["tpr"] >= GATE and b_["tnr"] >= GATE; res["tried"].append(dict(k=k, B_confirmed=ok))
            if ok:
                res["selected_k"] = k; break
    res["stop"] = res["selected_k"] is None
    json.dump(res, open(f"{d}/trajectory_eval.json", "w"), indent=1)
    print("tried", res["tried"], "selected k", res["selected_k"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "traj_render":
        traj_render()
    else:
        {"render": render, "rt_gen": rt_gen, "gen_label": gen_label, "train": train, "traj": traj}[cmd](int(sys.argv[2]))
