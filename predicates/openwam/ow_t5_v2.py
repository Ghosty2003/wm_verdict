"""OpenWAM-alpha t5: object-centric two-stage predicate (ruling 2026-10-26: redo authorized, ONE attempt, gate unchanged).

Why: the whole-image classifier read "gripper above the back compartment" (content check, 26/40 gate-row false
positives showed the book elsewhere); hard negatives fixed it on real frames but not on generated ones.

Stage 1  book-mask network: input = ow_view.pred_view frame (agentview | wrist, 128 x 256), output = the AMODAL book
         silhouette on the agentview half (128 x 128): the book mesh vertices projected with the T0 geometry
         (t5_geometry.Geo, OpenWAM orientation FLIP = 'h'), convex hull filled. Supervision from the simulator state.
Stage 2  gripper-blind decision: three features of the predicted mask relative to the projected back-compartment
         volume of the episode's scene (no image pixels enter this stage):
           f_in    share of mask mass inside the projection of {box footprint} x [box bottom, box top + upright half height]
           f_above share of mask mass inside the projection of the 15 cm above that volume
           area    mask mass / 128^2
         logistic regression -> p(success); threshold = v3 rule (logit midpoint of the largest feasible run).
Data, folds, rows: as ow_predicate.train for t5 (same 148 episodes, same folds); gate rows real / VAE round trip /
generated (simulator labels, timing window); the gate is decided by the OUT-OF-FOLD rows only (ruling 2026-10-24),
the deployed row is reference only; fold stability reported. Trajectory-level k: A selects / B confirms on the
famA / famP truth replays (ow_predicate KGRID). Scoring: from stored canvases only (no new imagination).

  masks     (lerobot_libero, GPU render)  -> predicate/t5/v2/masks/<episode>.npz, regions.json
  train                                   -> predicate/t5/v2/calibration_success.json, mask_net.pt, stage2.json
  traj                                    -> predicate/t5/v2/trajectory_eval.json
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
_wmvp.bootstrap()
import ow_predicate as P

D = f"{P.OUT}/t5"
V2 = f"{D}/v2"
ABOVE = 0.15
FLIP = "h"


# ------------------------------------------------------------------------------------------------ geometry
def _hull_mask(pts, res=128):
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull
    pts = np.asarray(pts, float)
    im = Image.new("L", (res, res), 0)
    if len(pts) >= 3:
        try:
            h = pts[ConvexHull(pts).vertices]
            ImageDraw.Draw(im).polygon([tuple(x) for x in h], fill=1)
        except Exception:
            pass
    return np.asarray(im, bool)


def _box_corners(lb, ub, z0, z1):
    return np.array([[x, y, z] for x in (lb[0], ub[0]) for y in (lb[1], ub[1]) for z in (z0, z1)])


def regions(geo, st, half_up):
    lb, ub = st["lb"], st["ub"]; top = ub[2] + half_up
    rin = _hull_mask(geo.project(_box_corners(lb, ub, lb[2], top)))
    rab = _hull_mask(geo.project(_box_corners(lb, ub, top, top + ABOVE))) & ~rin
    return rin, rab


def book_mask(geo):
    d = geo.d; pts = []
    for g in geo.book_geoms:
        mid = geo.m.geom_dataid[g]; a = geo.m.mesh_vertadr[mid]; n = geo.m.mesh_vertnum[mid]
        pts.append(d.geom_xpos[g][None] + geo.m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T)
    return _hull_mask(geo.project(np.concatenate(pts)))


def _geo(u):
    import t5_geometry
    t5_geometry.FLIP = FLIP
    return t5_geometry.Geo(u)


# ------------------------------------------------------------------------------------------------ stage: masks
def _mask_episode(eid):
    from ow_view import canvas_from_raw, pred_view, raw_obs
    E = {e["id"]: e for e in P._registry(5)}; e = E[eid]
    z = np.load(f"{D}/episodes/{eid}.npz"); acts, frames = z["actions"], z["frames"]
    env, u = P._restore(e["anc"], e.get("disp"), len(acts)); geo = _geo(u)
    s0 = geo.state(); half_up = s0["top_z"] - s0["centre"][2]; rin, rab = regions(geo, s0, half_up)
    M, same = [book_mask(geo)], []
    o = raw_obs(u); same.append(np.array_equal(pred_view(canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"])), frames[0]))
    for t, a in enumerate(acts, 1):
        u.step(np.asarray(a, np.float32)); M.append(book_mask(geo))
        if t % 25 == 0:
            o = raw_obs(u); same.append(np.array_equal(pred_view(canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"])), frames[t]))
    env.close()
    np.savez_compressed(f"{V2}/masks/{eid}.npz", masks=np.packbits(np.stack(M), axis=-1), region_in=rin, region_above=rab)
    return eid, bool(all(same))


def masks():
    import multiprocessing as mp
    os.makedirs(f"{V2}/masks", exist_ok=True)
    todo = [e["id"] for e in P._registry(5) if not os.path.exists(f"{V2}/masks/{e['id']}.npz")]
    with mp.get_context("spawn").Pool(6, maxtasksperchild=10) as pool:
        bad = [eid for eid, ok in pool.imap_unordered(_mask_episode, todo, chunksize=1) if not ok]
    print("masks", len(todo), "frame mismatches", bad)
    if bad:
        sys.exit("replay does not reproduce the stored episode frames -- stop")


# ------------------------------------------------------------------------------------------------ stage 1 network
def _unet():
    import torch.nn as nn
    import torch

    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))

    class U(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3, self.e4 = blk(6, 32), blk(32, 64), blk(64, 128), blk(128, 192)
            self.p = nn.MaxPool2d(2)
            self.u3, self.d3 = nn.ConvTranspose2d(192, 128, 2, 2), blk(256, 128)
            self.u2, self.d2 = nn.ConvTranspose2d(128, 64, 2, 2), blk(128, 64)
            self.u1, self.d1 = nn.ConvTranspose2d(64, 32, 2, 2), blk(64, 32)
            self.out = nn.Conv2d(32, 1, 1)

        def forward(self, x):                              # x: [B, 6, 128, 128] = agentview + wrist stacked
            a = self.e1(x); b = self.e2(self.p(a)); c = self.e3(self.p(b)); d = self.e4(self.p(c))
            c = self.d3(torch.cat([self.u3(d), c], 1)); b = self.d2(torch.cat([self.u2(c), b], 1)); a = self.d1(torch.cat([self.u1(b), a], 1))
            return self.out(a)[:, 0]
    return U()


def _inp(frames):
    import torch
    x = torch.from_numpy(np.ascontiguousarray(frames)).float().div(255)          # [B, 128, 256, 3]
    return torch.cat([x[:, :, :128], x[:, :, 128:]], -1).permute(0, 3, 1, 2)    # [B, 6, 128, 128]


def fit_mask_net(X, Mk, seed, epochs=20, per_epoch=6000):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed); rs = np.random.RandomState(seed)
    net = _unet().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        net.train(); idx = rs.choice(len(X), per_epoch)
        for i in range(0, per_epoch, 64):
            b = idx[i:i + 64]; y = torch.from_numpy(Mk[b].astype(np.float32)).cuda()
            lo = net(_inp(X[b]).cuda()); pr = torch.sigmoid(lo)
            dice = 1 - (2 * (pr * y).sum((1, 2)) + 1) / (pr.sum((1, 2)) + y.sum((1, 2)) + 1)
            loss = F.binary_cross_entropy_with_logits(lo, y) + dice.mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, float(loss)


def predict_masks(net, frames, bs=128):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), bs):
            out.append(torch.sigmoid(net(_inp(frames[i:i + bs]).cuda())).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


def features(pm, rin, rab):
    """pm: [N, 128, 128] predicted mask probabilities; rin / rab: [N, 128, 128] or broadcastable bool regions."""
    pm = pm.astype(np.float32); s = pm.sum((1, 2)) + 1e-6
    return np.stack([(pm * rin).sum((1, 2)) / s, (pm * rab).sum((1, 2)) / s, s / 128 ** 2], 1)


class Logit:
    """L2-regularised logistic regression (C = 1, as sklearn's default), Newton iterations on [features, 1]."""

    def __init__(self, Fe, Y, C=1.0, iters=50):
        X = np.c_[Fe, np.ones(len(Fe))].astype(np.float64); y = np.asarray(Y, np.float64); w = np.zeros(X.shape[1])
        R = np.eye(X.shape[1]) / C; R[-1, -1] = 0.0                     # intercept not penalised
        for _ in range(iters):
            p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) + R @ w
            Hs = (X * (p * (1 - p))[:, None]).T @ X + R
            step = np.linalg.solve(Hs, g); w -= step
            if np.abs(step).max() < 1e-10:
                break
        self.coef_, self.intercept_ = w[None, :-1], w[-1:]

    def predict_proba(self, Fe):
        z = Fe @ self.coef_[0] + self.intercept_[0]; p = 1 / (1 + np.exp(-z))
        return np.stack([1 - p, p], 1)


def fit_stage2(Fe, Y):
    return Logit(Fe, Y)


# ------------------------------------------------------------------------------------------------ stage: train + gate
def train():
    import torch
    from recal_v3 import _row
    from recal_predicate import fail_terminal_mask
    from recal_v4_t3 import _threshold
    E = P._registry(5); fold_of = {e["id"]: e["fold"] for e in E}
    X, Y, M, EP, MK, RI, RA = [], [], [], [], [], {}, {}
    for e in E:
        z = np.load(f"{D}/episodes/{e['id']}.npz"); mk = np.load(f"{V2}/masks/{e['id']}.npz")
        X.append(z["frames"]); Y.append(z["labels"]); M.append(fail_terminal_mask(z["labels"]))
        MK.append(np.unpackbits(mk["masks"], axis=-1)[:, :, :128].astype(bool)); EP += [e["id"]] * len(z["labels"])
        RI[e["id"]], RA[e["id"]] = mk["region_in"], mk["region_above"]
    X = np.concatenate(X); Y = np.concatenate(Y).astype(bool); M = np.concatenate(M); MK = np.concatenate(MK); EP = np.array(EP)
    FO = np.array([fold_of[x] for x in EP])
    rt = dict(np.load(f"{D}/roundtrip.npz")); gn = dict(np.load(f"{D}/gen_all.npz"))
    first = {e["id"]: (int(np.argmax(l)) if (l := np.load(f"{D}/episodes/{e['id']}.npz")["labels"]).any() else None) for e in E}
    rt_lab = np.array([np.load(f"{D}/episodes/{ep}.npz")["labels"][i] for ep, i in zip(rt["episode"], rt["idx"])]).astype(bool)
    rt_ft = np.array([first[x] is None for x in rt["episode"]]); gn_ft = np.array([first[x] is None for x in gn["episode"]])
    fo_rt = np.array([fold_of[x] for x in rt["episode"]]); fo_gn = np.array([fold_of[x] for x in gn["episode"]])
    chunk_ref = {}
    for ep, t0, s, y in zip(gn["episode"], gn["t0"], gn["step"], gn["success"]):
        if y and ((ep, t0) not in chunk_ref or s < chunk_ref[(ep, t0)]):
            chunk_ref[(ep, t0)] = int(s)
    win = np.array([(not y) and (chunk_ref.get((ep, t0), first[ep]) is not None) and abs(int(s) - chunk_ref.get((ep, t0), first[ep])) <= P.WINDOW
                    for ep, t0, s, y in zip(gn["episode"], gn["t0"], gn["step"], gn["success"])])
    yg = gn["success"].astype(bool)
    reg = lambda eps: (np.stack([RI[x] for x in eps]), np.stack([RA[x] for x in eps]))
    ri_x, ra_x = reg(EP); ri_rt, ra_rt = reg(rt["episode"]); ri_gn, ra_gn = reg(gn["episode"])
    oof, oof_rt, oof_gn = np.zeros(len(Y)), np.zeros(len(rt_lab)), np.zeros(len(yg)); P_all = np.zeros((6, len(yg))); log = []
    for k in range(5):
        tr = FO != k
        net, loss = fit_mask_net(X[tr], MK[tr], seed=k)
        pm_tr = predict_masks(net, X[tr]); lr = fit_stage2(features(pm_tr, ri_x[tr], ra_x[tr]), Y[tr])
        sc = lambda fr, ri, ra: lr.predict_proba(features(predict_masks(net, fr), ri, ra))[:, 1]
        oof[~tr] = sc(X[~tr], ri_x[~tr], ra_x[~tr])
        oof_rt[fo_rt == k] = sc(rt["frames"][fo_rt == k], ri_rt[fo_rt == k], ra_rt[fo_rt == k])
        P_all[k] = sc(gn["frames"], ri_gn, ra_gn); oof_gn[fo_gn == k] = P_all[k][fo_gn == k]
        pm_te = predict_masks(net, X[~tr][::10]); iou = float(((pm_te > .5) & MK[~tr][::10]).sum() / max(((pm_te > .5) | MK[~tr][::10]).sum(), 1))
        log.append(dict(fold=k, mask_loss=loss, mask_iou_heldout=iou, stage2_coef=lr.coef_.tolist(), stage2_intercept=float(lr.intercept_[0])))
        print(log[-1], flush=True); del net; torch.cuda.empty_cache()
    thr, runs = _threshold(oof, Y, M, None); t_ = thr if thr is not None else 0.5
    net, _ = fit_mask_net(X, MK, seed=100); lr = fit_stage2(features(predict_masks(net, X), ri_x, ra_x), Y)
    P_all[5] = lr.predict_proba(features(predict_masks(net, gn["frames"]), ri_gn, ra_gn))[:, 1]
    dec = P_all >= t_; neg = ~yg & ~win
    stab = dict(per_fold=[dict(fold=k, decision_disagree_vs_deployed=float((dec[k] != dec[5])[neg].mean())) for k in range(5)],
                conflicting_decisions_on_negatives=float((dec.any(0) & ~dec.all(0))[neg].mean()))
    res = dict(detector="success", task=5, predicate="object-centric two-stage (ow_t5_v2.py)", threshold=thr, feasible_runs=runs,
               fold_log=log, holdout_real=_row(oof, Y, M, t_), holdout_vae_roundtrip=_row(oof_rt, rt_lab, rt_ft, t_),
               generated_raw=_row(oof_gn, yg, gn_ft, t_), generated_timing_rule=_row(oof_gn[~win], yg[~win], gn_ft[~win], t_),
               generated_timing_rule_deployed=_row(P_all[5][~win], yg[~win], gn_ft[~win], t_), window_steps=P.WINDOW,
               fold_stability=stab, split_sensitive=stab["conflicting_decisions_on_negatives"] > 0.05,
               generated_row_for_gate="generated_timing_rule")
    rows = ("holdout_real", "holdout_vae_roundtrip", "generated_timing_rule")
    gate = {r_: bool(thr is not None and res[r_]["tpr"] >= P.GATE and res[r_]["tnr"] >= P.GATE
                     and (res[r_]["fail_terminal"] is None or res[r_]["fail_terminal"]["tnr"] >= P.GATE)) for r_ in rows}
    res["gate_85"] = gate; res["gate_pass"] = bool(thr is not None and all(gate.values()))
    dep = res["generated_timing_rule_deployed"]
    res["deployed_row_fails_while_oof_passes"] = bool(res["gate_pass"] and not (dep["tpr"] >= P.GATE and dep["tnr"] >= P.GATE))
    p = f"{V2}/mask_net.pt"; torch.save(net.state_dict(), p)
    json.dump(dict(coef=lr.coef_.tolist(), intercept=lr.intercept_.tolist(), features=["f_in", "f_above", "area"]), open(f"{V2}/stage2.json", "w"))
    res["artifact"] = dict(path=os.path.relpath(p, ROOT), sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(),
                           stage2=os.path.relpath(f"{V2}/stage2.json", ROOT), stage2_sha256=hashlib.sha256(open(f"{V2}/stage2.json", "rb").read()).hexdigest())
    json.dump(res, open(f"{V2}/calibration_success.json", "w"), indent=1)
    print("RESULT", json.dumps({r_: {x: res[r_][x] for x in ("n", "tpr", "tnr")} for r_ in rows + ("generated_timing_rule_deployed",)}),
          "gate", gate, "pass", res["gate_pass"], "stability", stab["conflicting_decisions_on_negatives"], flush=True)


if __name__ == "__main__":
    {"masks": masks, "train": train}[sys.argv[1]]()
