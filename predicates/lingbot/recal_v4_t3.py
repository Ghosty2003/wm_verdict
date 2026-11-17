"""t3 predicate v4 (spec revision #5; 2026-10-21 rulings): success frame = Close detector fires AND "bowl outside the
drawer, visible" detector does not fire; trajectory success = that conjunction on k consecutive frames.

Pools (ruling 1.2 / 1.4): TRAINING = the v3 calibration set's v1 closed-loop rollouts (60, all successes) + 30 new
episodes (10 per type). The v2/v3 perturbation episodes never enter training (reported only as a supplementary
frame-level holdout; not implemented yet). VALIDATION at trajectory level = lingbot_famA / lingbot_famP t3 truth (a) replays only
(500 episodes); not one of their frames enters training.

New episodes (master seed 20261034, non-anchor inits):
  syn_success      new LingBot closed-loop rollouts that succeed (attempts until 10)
  syn_close_empty  a v1 rollout's executed actions replayed from a displaced initial state (bowl moved 5-9 cm in a random
                   direction after reset + 10 settle steps); kept iff the end state is Close and not In (until 10)
  syn_in_open      a v1 rollout's actions replayed until In first holds, + 12 steps, then no-op (gripper open) to the
                   original length; kept iff the end state is In and not Close (until 10)
Derived episodes follow their base v1 rollout's fold.

Labels (simulator state, every frame; t3_labels.py): close = LIBERO Close(cabinet bottom); in = LIBERO In(bowl, ...);
bowl pixels per camera from a geom-segmentation render; per camera "unoccluded" = every sampled bowl-mesh point is in
the camera frustum and the first geom hit by the ray from the camera is a bowl geom.
  VIS_MIN (ruling 1.1): from the training pool, frames with not-In and the bowl unoccluded in a camera: the largest
  integer threshold that still counts >= 99 % of them visible (= floor of the 1 % quantile of their pixel counts).
  bowl_out_vis = not In and pixels >= VIS_MIN in agentview or wrist; undecidable = not In and below VIS_MIN in both
  (excluded from the bowl detector's training and gate rows; never a success frame).

Frame-level gate (v3 recipe + three rows): out-of-fold real holdout, VAE round trip (every 6th frame), one-chunk
generation (3 starts per episode, labelled by simulating). Each detector: TPR, TNR and failure-terminal TNR >= 0.85;
threshold = v3 rule (logit midpoint of the largest feasible run). A detector failing the gate -> STOP (ruling 1.3:
report before adding data). Bowl detector additionally reports the drawer-closed subset.

Trajectory level (ruling 1.5) on famA/famP t3: halves by episode (seed 20261021), stratified by type: close_empty_long
(the 61 long false alarms of t3_subgoal_diag), in_open (51), success, other_fail. Grid k in {1,2,3,5,8,16,24,36,48};
smallest k with TPR, TNR >= 0.85 on half A, confirmed on half B; per-type rates on both halves. None -> STOP.

  python recal_v4_t3.py data | train | traj
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
import torch
import mujoco

_wmvp.bootstrap()
from lingbot_predicate import predict, metrics, model_view, batched, OUT as V1_OUT
from recal_predicate import closed_episode, fail_terminal_mask, OUT as V2_OUT
from recal_v3 import restore, _anchor_tpl, _fit, _row, K, RT_EVERY, GATE
from lingbot_predicate import Net
import torch.nn.functional as F
from t3_labels import BowlVis, BOWL

ROOT = _wmvp.P("$WMV_ROOT")
OUT = f"{ROOT}/libero/out/predicate_v4_t3"
TID = 3
MASTER = 20261034
N_SYN = 10
N_CLOSE_EMPTY = 40                 # 2026-10-21 retrain ruling: close-empty type raised from 10 to 40
N_CLOSED_POS = 1000                # per-epoch draws from the drawer-closed bowl-outside stratum (of 3000 positives)
KGRID = (1, 2, 3, 5, 8, 16, 24, 36, 48)
SPLIT_SEED = 20261021
DETECTORS = ("close", "bowl_out")
LABEL_KEYS = ("success", "close", "in_", "px_agent", "px_wrist", "unocc_agent", "unocc_wrist")


# --------------------------------------------------------------------------------------------- labels
class Labeler:
    """per-frame simulator labels for t3 (sub-goals, bowl pixels and occlusion per camera)."""

    def __init__(self, u, n_pts=48):
        self.u = u; sim = u._env.sim
        self.m, self.d = sim.model._model, sim.data._data
        self.vis = BowlVis(sim)
        self.bowl_geoms = set(self.vis.geoms.tolist())
        self.cams = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in ("agentview", "robot0_eye_in_hand")]
        pts = []
        for g in self.vis.geoms:
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.m.geom_dataid[g]; a = self.m.mesh_vertadr[mid]; n = self.m.mesh_vertnum[mid]
                v = self.m.mesh_vert[a:a + n]
                sel = np.linspace(0, n - 1, min(n, n_pts)).astype(int)
                pts += [(g, v[i].copy()) for i in sel]
        assert pts, "bowl has no mesh geoms"
        self.pts = pts
        prob = u._env.env
        self.goals = {" ".join(g).split()[0]: g for g in prob.parsed_problem["goal_state"]}

    def _unoccluded(self, cid):
        d, m = self.d, self.m
        cp = d.cam_xpos[cid].copy(); R = d.cam_xmat[cid].reshape(3, 3); t = np.tan(np.deg2rad(m.cam_fovy[cid]) / 2)
        gid = np.array([-1], np.int32)
        for g, v in self.pts:
            p = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ v
            rel = R.T @ (p - cp)
            if rel[2] >= 0 or abs(rel[0] / -rel[2]) > t or abs(rel[1] / -rel[2]) > t:
                return False
            dist = mujoco.mj_ray(m, d, cp, p - cp, None, 1, -1, gid)
            if dist >= 0 and dist < 1 - 1e-6 and int(gid[0]) not in self.bowl_geoms:
                return False
        return True

    def __call__(self):
        prob = self.u._env.env
        pa, pw = self.vis.pixels()
        return dict(success=bool(self.u._env.check_success()), close=bool(prob._eval_predicate(self.goals["close"])),
                    in_=bool(prob._eval_predicate(self.goals["in"])), px_agent=pa, px_wrist=pw,
                    unocc_agent=self._unoccluded(self.cams[0]), unocc_wrist=self._unoccluded(self.cams[1]))

    def close(self):
        self.vis.close()


def labelled_replay(rig, anc, acts, disp=None, tail_noop=0):
    """restore (+ displacement) -> replay acts (+ tail_noop no-op steps); model-view frames and labels per frame."""
    env, u, obs = restore(rig, anc, disp)
    L = Labeler(u)
    frames, labs = [model_view(rig, rig.obs_to_batch(obs))], [L()]
    seq = [np.asarray(a, np.float32) for a in acts] + [np.array([0, 0, 0, 0, 0, 0, -1], np.float32)] * tail_noop
    for a in seq:
        o, *_ = u.step(a)
        frames.append(model_view(rig, rig.obs_to_batch(batched(o)))); labs.append(L())
    L.close(); env.close()
    return np.stack(frames), {k: np.array([x[k] for x in labs]) for k in LABEL_KEYS}, np.stack(seq)


# --------------------------------------------------------------------------------------------- registry
def registry():
    E = []
    _, tpl, _ = _anchor_tpl(TID)
    for f in sorted(os.listdir(f"{V1_OUT}/t{TID}/rollouts")):
        z = np.load(f"{V1_OUT}/t{TID}/rollouts/{f}")
        E.append(dict(id=f"v1_{f[:-4]}", source="v1_rollout", base=None, path=f"{OUT}/episodes/v1_{f[:-4]}.npz",
                      anc=dict(tpl, init_index=int(z["init_index"]), seed=int(z["seed"])), disp=None))
    d = f"{OUT}/episodes"
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.startswith("syn_"):
            meta = json.loads(str(np.load(f"{d}/{f}")["meta"]))
            E.append(dict(id=f[:-4], source=meta["source"], base=meta.get("base"), path=f"{d}/{f}",
                          anc=dict(tpl, init_index=meta["init_index"], seed=meta["seed"]), disp=meta.get("displacement"),
                          tail_noop=meta.get("tail_noop", 0)))
    rs = np.random.RandomState(MASTER)
    for src in ("v1_rollout", "syn_success"):
        S = [e for e in E if e["source"] == src]
        for rank, i in enumerate(rs.permutation(len(S))):
            S[i]["fold"] = int(rank % K)
    fold = {e["id"]: e.get("fold") for e in E}
    for e in E:
        if e.get("base"):
            e["fold"] = fold[e["base"]]
    return E


# --------------------------------------------------------------------------------------------- data
def _save(path, frames, labs, acts, meta):
    np.savez_compressed(path, frames=frames, actions=acts, meta=json.dumps(meta), **labs)


def data():
    from lingbot_unit import LingBotRig
    rig = LingBotRig(TID)
    os.makedirs(f"{OUT}/episodes", exist_ok=True)
    _, tpl, used = _anchor_tpl(TID)
    v1 = [e for e in registry() if e["source"] == "v1_rollout"]
    log = dict(v1_frames_identical=0, v1_success_identical=0)
    # 1. v1 rollouts: labelled bit-exact replay of their executed actions
    for e in v1:
        z = np.load(f"{V1_OUT}/t{TID}/rollouts/{e['id'][3:]}.npz")
        if not os.path.exists(e["path"]):
            fr, labs, acts = labelled_replay(rig, e["anc"], z["actions"])
            _save(e["path"], fr, labs, acts, dict(source="v1_rollout", init_index=e["anc"]["init_index"], seed=e["anc"]["seed"]))
        y = np.load(e["path"])
        log["v1_frames_identical"] += int(np.array_equal(y["frames"][:-1], z["frames"][:-1]))
        log["v1_success_identical"] += int(np.array_equal(y["success"], z["labels"]))
    print("v1", log, flush=True)
    rs = np.random.RandomState(MASTER)
    v1_inits = {e["anc"]["init_index"] for e in v1}
    pool = [i for i in range(50) if i not in used]
    # 2. syn_success: new closed-loop rollouts that succeed
    n, tries = len([f for f in os.listdir(f"{OUT}/episodes") if f.startswith("syn_success")]), 0
    while n < N_SYN and tries < 3 * N_SYN:
        tries += 1
        init = int(pool[rs.randint(len(pool))]); seed = int(rs.randint(2 ** 31 - 1)); pseed = int(rs.randint(2 ** 31 - 1))
        anc = dict(tpl, init_index=init, seed=seed)
        _, lab, acts, _ = closed_episode(rig, anc, pseed, np.random.RandomState(pseed))
        if not lab.any():
            print("syn_success attempt failed", init, flush=True); continue
        fr, labs, acts = labelled_replay(rig, anc, acts)
        assert labs["success"].any()
        _save(f"{OUT}/episodes/syn_success_{n:02d}.npz", fr, labs, acts,
              dict(source="syn_success", init_index=init, seed=seed, policy_seed=pseed, new_init=init not in v1_inits))
        print(f"syn_success_{n:02d} init {init} T {len(acts)}", flush=True); n += 1
    log["syn_success"] = dict(kept=n, attempts=tries)
    # 3. syn_close_empty: v1 actions from a displaced initial state; keep iff end state Close and not In
    n, tries = len([f for f in os.listdir(f"{OUT}/episodes") if f.startswith("syn_close_empty")]), 0
    while n < N_CLOSE_EMPTY and tries < 8 * N_CLOSE_EMPTY:
        tries += 1
        e = v1[rs.randint(len(v1))]
        ang = rs.uniform(0, 2 * np.pi); rad = rs.uniform(0.05, 0.09)
        disp = dict(object=BOWL, offset_xy_m=[round(rad * np.cos(ang), 4), round(rad * np.sin(ang), 4)])
        acts = np.load(f"{V1_OUT}/t{TID}/rollouts/{e['id'][3:]}.npz")["actions"]
        fr, labs, acts = labelled_replay(rig, e["anc"], acts, disp)
        ok = bool(labs["close"][-1] and not labs["in_"][-1])
        print(f"syn_close_empty try {tries} base {e['id']} disp {disp['offset_xy_m']} end close={labs['close'][-1]} "
              f"in={labs['in_'][-1]} px_agent_end={labs['px_agent'][-1]}", flush=True)
        if ok:
            _save(f"{OUT}/episodes/syn_close_empty_{n:02d}.npz", fr, labs, acts,
                  dict(source="syn_close_empty", base=e["id"], init_index=e["anc"]["init_index"], seed=e["anc"]["seed"],
                       displacement=disp))
            n += 1
    log["syn_close_empty"] = dict(kept=n, attempts=tries)
    # 4. syn_in_open: v1 actions until In first holds (+12), then no-op with the gripper open; keep iff In and not Close
    n, tries = len([f for f in os.listdir(f"{OUT}/episodes") if f.startswith("syn_in_open")]), 0
    order = list(rs.permutation(len(v1)))
    while n < N_SYN and order:
        tries += 1
        e = v1[order.pop(0)]
        y = np.load(e["path"]); acts = y["actions"]
        hit = np.where(y["in_"] & ~y["close"])[0]
        if not len(hit):
            continue
        cut = min(int(hit[0]) + 12, len(acts))
        fr, labs, acts2 = labelled_replay(rig, e["anc"], acts[:cut], tail_noop=len(acts) - cut)
        ok = bool(labs["in_"][-1] and not labs["close"][-1])
        print(f"syn_in_open try {tries} base {e['id']} cut {cut} end in={labs['in_'][-1]} close={labs['close'][-1]}", flush=True)
        if ok:
            _save(f"{OUT}/episodes/syn_in_open_{n:02d}.npz", fr, labs, acts2,
                  dict(source="syn_in_open", base=e["id"], init_index=e["anc"]["init_index"], seed=e["anc"]["seed"],
                       cut=cut, tail_noop=len(acts) - cut))
            n += 1
    log["syn_in_open"] = dict(kept=n, attempts=tries)
    # 5. VIS_MIN from the training pool (ruling 1.1)
    E = registry()
    px = []
    for e in E:
        y = np.load(e["path"])
        for cam in ("agent", "wrist"):
            sel = ~y["in_"] & y[f"unocc_{cam}"]
            px += y[f"px_{cam}"][sel].tolist()
    px = np.array(px)
    vis_min = int(np.floor(np.quantile(px, 0.01)))
    vis = dict(rule="largest integer threshold counting >= 99 % of not-In, unoccluded (frame, camera) pairs visible",
               n_pairs=int(len(px)), vis_min=vis_min, share_visible_at_vis_min=float((px >= vis_min).mean()),
               quantiles={q: float(np.quantile(px, q)) for q in (0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 1)},
               histogram=np.histogram(px, bins=[0, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 5000, 20000])[0].tolist())
    json.dump(vis, open(f"{OUT}/vis_min.json", "w"), indent=1)
    _vis_plot(px, vis_min)
    print("VIS_MIN", json.dumps(vis), flush=True)
    log["episodes"] = {s: sum(e["source"] == s for e in E) for s in sorted({e["source"] for e in E})}
    json.dump(log, open(f"{OUT}/data_log.json", "w"), indent=1)
    # 6. derived rows: VAE round trip and one-chunk generation, both carrying the full label set
    _derived(rig, E)


def _vis_plot(px, vis_min):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(np.clip(px, 1, None), bins=np.logspace(0, np.log10(max(px.max(), 10)), 60), color="#4a6fa5")
    ax.axvline(max(vis_min, 1), color="#c0392b", lw=1.5, label=f"VIS_MIN = {vis_min}")
    ax.set_xscale("log"); ax.set_xlabel("bowl pixels (128 x 128), not-In and unoccluded"); ax.set_ylabel("(frame, camera) pairs")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{OUT}/vis_min_hist.png", dpi=140)


def _derived(rig, E):
    if not os.path.exists(f"{OUT}/roundtrip_all.npz"):
        p = rig.policy; p._ensure_frozen_modules()
        RT, RL, RE = [], {k: [] for k in LABEL_KEYS}, []
        with torch.inference_mode():
            for e in E:
                y = np.load(e["path"])
                for i in range(0, len(y["frames"]), RT_EVERY):
                    fr = torch.from_numpy(y["frames"][i]).float().div(255)
                    halves = fr.permute(2, 0, 1).split(fr.shape[1] // 2, dim=-1)
                    vids = torch.stack([h * 2 - 1 for h in halves])[:, :, None].to(p.config.device).to(p.dtype)
                    p._frozen["streaming_vae"].clear_cache()
                    enc = p._streaming_vae.encode_chunk(vids.to(next(p._vae.parameters()).device))
                    lat = torch.cat(p._normalize_vae_latent(enc).split(1, dim=0), dim=-1)
                    RT.append(rig.decode(lat)[0].numpy()); RE.append(e["id"])
                    for k in LABEL_KEYS:
                        RL[k].append(y[k][i])
            p._frozen["streaming_vae"].clear_cache()
        np.savez_compressed(f"{OUT}/roundtrip_all.npz", frames=np.stack(RT), episode=np.array(RE),
                            **{k: np.array(v) for k, v in RL.items()})
        print("roundtrip_all", len(RT), flush=True)
    if not os.path.exists(f"{OUT}/gen_all.npz"):
        GF, GL, GE = [], {k: [] for k in LABEL_KEYS}, []
        grs = np.random.RandomState(MASTER + 100)
        for e in E:
            y = np.load(e["path"]); acts = y["actions"]; n = len(acts)
            if y["success"].any():
                fs = int(np.argmax(y["success"])); starts = np.clip(np.array([fs - 36, fs - 12, fs + 12]), 1, n - 13)
            else:
                starts = np.linspace(n // 2, n - 13, 3).astype(int)
            for t0 in sorted(set(int(x) for x in starts)):
                env, u, obs = restore(rig, e["anc"], e["disp"])
                o = None
                for a in acts[:t0]:
                    o, *_ = u.step(np.asarray(a, np.float32))
                batch = rig.obs_to_batch(batched(o) if o is not None else obs)
                rig.reset_policy(int(grs.randint(2 ** 31 - 1)))
                with torch.inference_mode():
                    p = rig.policy; p._ensure_frozen_modules(); p._maybe_init_prompt(batch)
                    ch = p.predict_action_chunk(batch); lat = p.last_predicted_latents
                frames = rig.decode(lat).numpy()
                ea = np.concatenate([rig.to_env_action(ch[:, k]) for k in range(ch.shape[1])], 0)
                L = Labeler(u); labs = []
                for a in ea:
                    u.step(a.astype(np.float32)); labs.append(L())
                L.close(); env.close()
                for k in range(1, min(len(frames), len(labs) + 1)):
                    GF.append(frames[k]); GE.append(e["id"])
                    for kk in LABEL_KEYS:
                        GL[kk].append(labs[k - 1][kk])
        np.savez_compressed(f"{OUT}/gen_all.npz", frames=np.stack(GF), episode=np.array(GE),
                            **{k: np.array(v) for k, v in GL.items()})
        print("gen_all", len(GF), flush=True)


# --------------------------------------------------------------------------------------------- targets
def targets(z, vis_min):
    """detector labels + validity mask from a label table (episode npz or derived row file)."""
    vis = (z["px_agent"] >= vis_min) | (z["px_wrist"] >= vis_min)
    bowl_y = ~z["in_"] & vis
    undecidable = ~z["in_"] & ~vis
    return dict(close=(z["close"].astype(bool), np.ones(len(z["close"]), bool)),
                bowl_out=(bowl_y.astype(bool), ~undecidable))


# --------------------------------------------------------------------------------------------- train
def train():
    vis_min = json.load(open(f"{OUT}/vis_min.json"))["vis_min"]
    E = registry(); fold_of = {e["id"]: e["fold"] for e in E}; src_of = {e["id"]: e["source"] for e in E}
    X, T, M, EP, CL = [], {d: ([], []) for d in DETECTORS}, [], [], []
    for e in E:
        z = np.load(e["path"]); X.append(z["frames"]); M.append(fail_terminal_mask(z["success"]))
        EP += [e["id"]] * len(z["success"]); CL.append(z["close"])
        for d, (y, v) in targets(z, vis_min).items():
            T[d][0].append(y); T[d][1].append(v)
    X = np.concatenate(X); M = np.concatenate(M); EP = np.array(EP); CL = np.concatenate(CL).astype(bool)
    FO = np.array([fold_of[x] for x in EP]); SRC = np.array([src_of[x] for x in EP])
    rt = np.load(f"{OUT}/roundtrip_all.npz"); gn = np.load(f"{OUT}/gen_all.npz")
    fo_rt = np.array([fold_of[x] for x in rt["episode"]]); fo_gn = np.array([fold_of[x] for x in gn["episode"]])
    ft_rt = _ft_rows(rt, E); ft_gn = _ft_rows(gn, E)
    t_rt, t_gn = targets(rt, vis_min), targets(gn, vis_min)
    summary = dict(vis_min=vis_min)
    for d in DETECTORS:
        Y = np.concatenate(T[d][0]); V = np.concatenate(T[d][1])
        oof = np.full(len(Y), np.nan); oof_rt = np.zeros(len(rt["episode"])); oof_gn = np.zeros(len(gn["episode"]))
        log = []
        for k in range(K):
            tr = (FO != k) & V
            net, loss = (_fit_strat(X[tr], Y[tr], M[tr], CL[tr], seed=k) if d == "bowl_out" else _fit(X[tr], Y[tr], M[tr], seed=k))
            oof[FO == k] = predict(net, X[FO == k])
            oof_rt[fo_rt == k] = predict(net, rt["frames"][fo_rt == k]); oof_gn[fo_gn == k] = predict(net, gn["frames"][fo_gn == k])
            log.append(dict(fold=k, n_train=int(tr.sum()), final_loss=loss, oof_prob_std=float(np.nanstd(oof[FO == k]))))
            print(d, log[-1], flush=True)
            del net; torch.cuda.empty_cache()
        if min(l["oof_prob_std"] for l in log) < 0.05:
            raise SystemExit(f"{d}: training collapsed in a fold -- stop and report")
        thr, runs = _threshold(oof[V], Y[V], M[V], CL[V] if d == "bowl_out" else None)
        res = dict(detector=d, fold_log=log, feasible_runs=runs, threshold=thr)
        yr, vr = t_rt[d]; yg, vg = t_gn[d]
        res["holdout_real"] = _row(oof[V], Y[V], M[V], thr if thr is not None else 0.5, SRC[V])
        res["holdout_vae_roundtrip"] = _row(oof_rt[vr], yr[vr], ft_rt[vr], thr if thr is not None else 0.5)
        res["model_generated_1chunk"] = _row(oof_gn[vg], yg[vg], ft_gn[vg], thr if thr is not None else 0.5)
        if d == "bowl_out":
            sel = V & CL
            res["holdout_real_drawer_closed"] = metrics(oof[sel], Y[sel], thr if thr is not None else 0.5)
            for name, o_, y_, v_, z_ in (("holdout_vae_roundtrip_drawer_closed", oof_rt, yr, vr, rt), ("model_generated_1chunk_drawer_closed", oof_gn, yg, vg, gn)):
                s_ = v_ & z_["close"].astype(bool)
                res[name] = metrics(o_[s_], y_[s_], thr if thr is not None else 0.5) if s_.any() else None   # reported, not gated
            res["undecidable_frames"] = int((~V).sum())
        gate = {}
        for r_ in ("holdout_real", "holdout_vae_roundtrip", "model_generated_1chunk"):
            r = res[r_]; ft = r["fail_terminal"]
            gate[r_] = bool(thr is not None and r["tpr"] is not None and r["tpr"] >= GATE and r["tnr"] >= GATE)
            gate[r_ + "/fail_terminal_tnr"] = bool(ft is None or ft["tnr"] >= GATE)
        if d == "bowl_out":                 # 2026-10-21 retrain ruling: drawer-closed subset is a gate condition (real holdout)
            rc = res["holdout_real_drawer_closed"]
            gate["holdout_real_drawer_closed"] = bool(thr is not None and rc["tpr"] is not None and rc["tpr"] >= GATE and rc["tnr"] >= GATE)
        res["gate_85"] = gate; res["gate_pass"] = bool(thr is not None and all(gate.values()))
        np.savez_compressed(f"{OUT}/oof_{d}.npz", oof=oof, labels=Y, valid=V, fail_terminal=M, episode=EP)
        if res["gate_pass"]:
            net, loss = (_fit_strat(X[V], Y[V], M[V], CL[V], seed=100) if d == "bowl_out" else _fit(X[V], Y[V], M[V], seed=100))
            path = f"{OUT}/{d}.pt"; torch.save(net.state_dict(), path)
            res["artifact"] = dict(path=os.path.relpath(path, ROOT), sha256=hashlib.sha256(open(path, "rb").read()).hexdigest())
        json.dump(res, open(f"{OUT}/calibration_{d}.json", "w"), indent=1)
        summary[d] = dict(threshold=thr, gate_pass=res["gate_pass"], gate_85=gate,
                          rows={r_: {x: res[r_][x] for x in ("n", "tpr", "tnr")} for r_ in ("holdout_real", "holdout_vae_roundtrip", "model_generated_1chunk")})
        print("RESULT", d, json.dumps(summary[d]), flush=True)
    summary["episodes"] = {s: sum(e["source"] == s for e in E) for s in sorted({e["source"] for e in E})}
    json.dump(summary, open(f"{OUT}/calibration_summary.json", "w"), indent=1)
    if not all(summary[d]["gate_pass"] for d in DETECTORS):
        print("STOP: a detector failed the frame-level gate (ruling 1.3: report before adding data)")


def _fit_strat(X, Y, M, S, seed):
    """v3 _fit with one extra positive stratum (2026-10-21 retrain ruling): S = drawer closed and bowl outside visible.
    Per epoch: N_CLOSED_POS from S, 3000 - N_CLOSED_POS from the other positives, 1500 failure-terminal and 1500 other
    negatives (v3 proportions otherwise unchanged)."""
    torch.manual_seed(seed)
    net = Net().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    ps = np.where(Y & S)[0]; po = np.where(Y & ~S)[0]; negf = np.where(~Y & M)[0]; nego = np.where(~Y & ~M)[0]
    rs = np.random.RandomState(seed)
    for ep in range(25):
        net.train()
        idx = np.concatenate([rs.choice(ps, N_CLOSED_POS), rs.choice(po, 3000 - N_CLOSED_POS), rs.choice(negf, 1500),
                              rs.choice(nego, 1500)]); rs.shuffle(idx)
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            loss = F.binary_cross_entropy_with_logits(net(torch.from_numpy(X[b]).permute(0, 3, 1, 2).float().div(255).cuda()),
                                                      torch.from_numpy(Y[b].astype(np.float32)).cuda())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval()
    return net, float(loss)


def _ft_rows(rows, E):
    """failure-terminal flag for derived rows: frame belongs to a never-successful episode (v3: second half; here the
    derived row index is not ordered, so the whole failed episode counts)."""
    failed = {e["id"]: not np.load(e["path"])["success"].any() for e in E}
    return np.array([failed[x] for x in rows["episode"]])


def _threshold(oof, Y, M, closed=None):
    """v3 rule; with `closed` (bowl detector) the drawer-closed subset's TPR and TNR must also be >= GATE."""
    zs = np.arange(-12, 12.0001, 0.05); ps = 1 / (1 + np.exp(-zs)); fneg = M & ~Y

    def ok(p_):
        f = (oof[Y] >= p_).mean() >= GATE and (oof[~Y] < p_).mean() >= GATE and (not fneg.any() or (oof[fneg] < p_).mean() >= GATE)
        if closed is not None:
            f = f and (oof[closed & Y] >= p_).mean() >= GATE and (oof[closed & ~Y] < p_).mean() >= GATE
        return bool(f)
    feas = np.array([ok(p_) for p_ in ps])
    runs, cur = [], None
    for i, f in enumerate(feas):
        if f and cur is None:
            cur = i
        if (not f or i == len(feas) - 1) and cur is not None:
            runs.append((cur, i if f else i - 1)); cur = None
    if not runs:
        return None, []
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return float(1 / (1 + np.exp(-(zs[a] + zs[b]) / 2))), [[float(ps[x]), float(ps[y])] for x, y in runs]


# --------------------------------------------------------------------------------------------- trajectory level
def traj():
    from lingbot_unit import LingBotRig
    from lingbot_predicate import Net
    cal = {d: json.load(open(f"{OUT}/calibration_{d}.json")) for d in DETECTORS}
    assert all(cal[d]["gate_pass"] for d in DETECTORS), "frame-level gate not passed"
    nets = {}
    for d in DETECTORS:
        n = Net().cuda(); n.load_state_dict(torch.load(f"{ROOT}/{cal[d]['artifact']['path']}")); n.eval(); nets[d] = n
    rig = LingBotRig(TID)
    diag = {u["unit"]: u for u in json.load(open(f"{ROOT}/libero/out/content_review/t3_subgoal_diag.json"))["units"]}
    os.makedirs(f"{OUT}/traj", exist_ok=True)
    T = []
    for run, bank in (("lingbot_famA", "familyA"), ("lingbot_famP", "familyP")):
        anchors = {a["anchor_id"]: a for a in json.load(open(f"{ROOT}/libero/out/bank/{bank}/{bank}_anchors.json"))["anchors"]}
        for l in open(f"{ROOT}/libero/out/runs/{run}/records_t3.jsonl"):
            r = json.loads(l); uid = r["unit_id"]; path = f"{OUT}/traj/{uid}.npz"
            if not os.path.exists(path):
                acts = np.load(f"{ROOT}/{r['action_file']}")["env"]
                env, obs = rig.make_env(anchors[r["anchor_id"]]); u = env.envs[0]; L = Labeler(u)
                fr, succ, labs = [model_view(rig, rig.obs_to_batch(obs))], [], [L()]
                for a in acts:
                    o, *_ = u.step(np.asarray(a, np.float32))
                    fr.append(model_view(rig, rig.obs_to_batch(batched(o)))); succ.append(bool(u._env.check_success())); labs.append(L())
                L.close(); env.close()
                succ = np.array(succ)
                assert np.array_equal(succ, np.load(f"{ROOT}/{r['trajectory_file']}")["real_success"]), uid
                fr = np.stack(fr)
                np.savez_compressed(path, p_close=predict(nets["close"], fr), p_bowl=predict(nets["bowl_out"], fr),
                                    **{k: np.array([x[k] for x in labs]) for k in LABEL_KEYS})
            z = np.load(path)
            conj = (z["p_close"] >= cal["close"]["threshold"]) & (z["p_bowl"] < cal["bowl_out"]["threshold"])
            if r["real_verdict"]:
                typ = "success"
            elif diag[uid]["longest_false_alarm"] >= 50:
                typ = "close_empty_long"
            elif diag[uid]["final_in"] and not diag[uid]["final_close"]:
                typ = "in_open"
            else:
                typ = "other_fail"
            vis_min = cal["bowl_out"].get("vis_min") or json.load(open(f"{OUT}/vis_min.json"))["vis_min"]
            undec_end = bool(z["close"][-1] and not z["in_"][-1] and max(z["px_agent"][-1], z["px_wrist"][-1]) < vis_min)
            T.append(dict(unit=uid, type=typ, label=bool(r["real_verdict"]), conj=conj[1:], undecidable_end=undec_end))
    rs = np.random.RandomState(SPLIT_SEED); half = {}
    for typ in ("close_empty_long", "in_open", "success", "other_fail"):
        ids = [t["unit"] for t in T if t["type"] == typ]
        for i, uid in enumerate(rs.permutation(ids)):
            half[uid] = "A" if i < (len(ids) + 1) // 2 else "B"

    def fires(c, k):
        run = 0
        for x in c:
            run = run + 1 if x else 0
            if run >= k:
                return True
        return False
    res = dict(grid=list(KGRID), split_seed=SPLIT_SEED, thresholds={d: cal[d]["threshold"] for d in DETECTORS},
               halves={h: {typ: sum(1 for t in T if half[t["unit"]] == h and t["type"] == typ)
                           for typ in ("close_empty_long", "in_open", "success", "other_fail")} for h in "AB"},
               per_k={}, tried=[], selected_k=None)
    for k in KGRID:
        f = {t["unit"]: fires(t["conj"], k) for t in T}
        row = {}
        for h in "AB":
            S = [t for t in T if half[t["unit"]] == h]
            pos = [f[t["unit"]] for t in S if t["label"]]; neg = [f[t["unit"]] for t in S if not t["label"]]
            row[h] = dict(tpr=float(np.mean(pos)), tnr=float(1 - np.mean(neg)),
                          by_type={typ: float(np.mean([f[t["unit"]] for t in S if t["type"] == typ]))
                                   for typ in ("close_empty_long", "in_open", "success", "other_fail")},
                          undecidable_end=dict(n=sum(t["undecidable_end"] for t in S),
                                               fired=sum(f[t["unit"]] for t in S if t["undecidable_end"])))
        res["per_k"][k] = row
        print(f"k={k}", json.dumps(row), flush=True)
    for k in KGRID:
        a, b = res["per_k"][k]["A"], res["per_k"][k]["B"]
        if a["tpr"] >= GATE and a["tnr"] >= GATE:
            ok = b["tpr"] >= GATE and b["tnr"] >= GATE
            res["tried"].append(dict(k=k, B_confirmed=ok))
            if ok:
                res["selected_k"] = k; break
    res["stop"] = res["selected_k"] is None
    res["undecidable_end"] = dict(rule="failure trajectories ending with drawer closed, bowl not In and below VIS_MIN in both cameras "
                                       "(counted separately; if validation fails and failures concentrate here -> temporal version)",
                                  units=[t["unit"] for t in T if t["undecidable_end"]])
    if res["stop"]:
        k_ = KGRID[-1]; fails = []
        for t in T:
            fz = fires(t["conj"], k_)
            if fz != t["label"]:
                fails.append(dict(unit=t["unit"], type=t["type"], undecidable_end=t["undecidable_end"], fired=fz))
        res["failure_composition_at_kmax"] = dict(n=len(fails), undecidable_end=sum(x["undecidable_end"] for x in fails), cases=fails)
    json.dump(res, open(f"{OUT}/trajectory_eval.json", "w"), indent=1)
    print("tried", res["tried"], "selected k", res["selected_k"], "STOP" if res["stop"] else "")


if __name__ == "__main__":
    {"data": data, "train": train, "traj": traj}[sys.argv[1]]()
