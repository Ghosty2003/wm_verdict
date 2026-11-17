"""t5 OpenWAM-view predicate: ONE hard-negative retrain (user ruling 2026-10-24, after the content check).

The content check showed the detector reads "gripper above the back compartment" instead of the book (26/40 gate-row false
positives: empty gripper over the compartment, book elsewhere). Same fix as t3's closed-empty-drawer blind spot: one extra
sampling stratum, here on the NEGATIVE side. Everything else = ow_predicate.train (t5): same episodes, folds, seeds,
threshold rule, three gate rows, generated row with simulator labels + timing window. Threshold not relaxed. One attempt:
pass -> t5 verdict column scored from the stored canvases; fail -> t5 pending, final.

Hard negative (fixed before looking at counts): frame of a never-successful episode, book NOT in the box (simulator), and
the gripper grip site within the back-compartment box footprint +- HN_XY_MARGIN in x and y and between the box top and
HN_Z_ABOVE above it. Per epoch: 3000 positives; negatives 1000 hard + 1000 failure-terminal + 1000 other (hard negatives
removed from the other two pools; v3 1:1 proportion kept for the rest).

Fold stability (ruling item 4): every generated frame is scored by all 5 fold models and the deployed model, for the old
(content-checked) and the new predicate; reported per fold. "Still large" (fixed now): > 5 % of generated negatives with
conflicting decisions across the six models -> recorded as "predicate sensitive to the training split" (fragility).
Correction (user, 2026-10-24): the gate is ALWAYS decided by the out-of-fold rows; the deployed weights saw the generated
frames' source episodes, so their generated row is optimistic and is listed only as a reference for the weights actually
used in scoring. Out-of-fold pass but deployed-row fail -> stop and report.

  hn_state   (lerobot_libero, GPU render for the frame check) -> predicate/t5/hn/hn_state.npz
  train      (lerobot_libero, GPU)                             -> predicate/t5/hn/{calibration_success.json, success.pt,
                                                                  calibration_summary.json, fold_stability.json}
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
from ow_view import canvas_from_raw, pred_view, raw_obs

D = f"{P.OUT}/t5"
HN = f"{D}/hn"
HN_XY_MARGIN = 0.03
HN_Z_ABOVE = 0.15
N_HN = 1000
UNSTABLE = 0.05


def _state_one(eid):
    from t5_geometry import Geo
    e = {x["id"]: x for x in P._registry(5)}[eid]
    z = np.load(f"{D}/episodes/{eid}.npz"); acts, frames = z["actions"], z["frames"]
    env, u = P._restore(e["anc"], e.get("disp"), len(acts)); geo = Geo(u)
    S = []
    for t in range(len(acts) + 1):
        if t:
            u.step(np.asarray(acts[t - 1], np.float32))
        st = geo.state(); o = raw_obs(u); eef = o["robot0_eef_pos"]          # grip site position
        same = bool(np.array_equal(pred_view(canvas_from_raw(o["agentview_image"], o["robot0_eye_in_hand_image"])), frames[t]))
        S.append((st["inside"], bool(u._env.check_success()), eef, st["lb"], st["ub"], same))
    env.close()
    inside, succ, eef, lb, ub, same = (np.array(x) for x in zip(*S))
    above = (np.all(eef[:, :2] >= lb[:, :2] - HN_XY_MARGIN, 1) & np.all(eef[:, :2] <= ub[:, :2] + HN_XY_MARGIN, 1)
             & (eef[:, 2] >= ub[:, 2]) & (eef[:, 2] <= ub[:, 2] + HN_Z_ABOVE))
    return eid, dict(hn=above & ~inside, inside=inside, success=succ, eef=eef, frames_match=same)


def hn_state():
    import multiprocessing as mp
    os.makedirs(HN, exist_ok=True)
    failed = [e["id"] for e in P._registry(5) if not np.load(f"{D}/episodes/{e['id']}.npz")["labels"].any()]
    with mp.get_context("spawn").Pool(2, maxtasksperchild=10) as pool:
        res = dict(pool.map(_state_one, failed, chunksize=1))
    bad = {k: dict(success=int(v["success"].sum()), frames_mismatch=int((~v["frames_match"]).sum())) for k, v in res.items()
           if v["success"].any() or not v["frames_match"].all()}
    out = {f"{k}__{f}": v[f] for k, v in res.items() for f in ("hn", "inside", "eef")}
    np.savez_compressed(f"{HN}/hn_state.npz", episodes=np.array(sorted(res)), **out)
    n = {k: int(v["hn"].sum()) for k, v in res.items()}
    print("failed episodes", len(res), "hard-negative frames", sum(n.values()), "episodes with any", sum(1 for x in n.values() if x),
          "per episode", n, flush=True)
    if bad:
        raise SystemExit(f"replay does not reproduce the stored episodes: {bad} -- stop")


def _fit_hn(X, Y, M, H, seed):
    """recal_v3._fit with one extra NEGATIVE stratum H (hard negatives): per epoch 3000 positives, N_HN hard negatives,
    (3000 - N_HN) / 2 failure-terminal and (3000 - N_HN) / 2 other negatives."""
    import torch
    import torch.nn.functional as F
    from lingbot_predicate import Net
    torch.manual_seed(seed)
    net = Net().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    pos = np.where(Y)[0]; hn = np.where(~Y & H)[0]; negf = np.where(~Y & M & ~H)[0]; nego = np.where(~Y & ~M & ~H)[0]
    rs = np.random.RandomState(seed); half = (3000 - N_HN) // 2
    for ep in range(25):
        net.train()
        idx = np.concatenate([rs.choice(pos, 3000), rs.choice(hn, N_HN), rs.choice(negf, half), rs.choice(nego, half)]); rs.shuffle(idx)
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            loss = F.binary_cross_entropy_with_logits(net(torch.from_numpy(X[b]).permute(0, 3, 1, 2).float().div(255).cuda()),
                                                      torch.from_numpy(Y[b].astype(np.float32)).cuda())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval()
    return net, float(loss)


def _stability(P_all, thr, neg):
    """P_all: (6, n) = 5 fold models + deployed on every generated frame. Per fold: mean |p_fold - p_deployed|; overall:
    share of generated negatives whose decision differs across the six models; mean per-frame std."""
    dec = P_all >= thr
    return dict(per_fold=[dict(fold=k, mean_abs_diff_vs_deployed=float(np.abs(P_all[k] - P_all[5]).mean()),
                               decision_disagree_vs_deployed=float((dec[k] != dec[5])[neg].mean())) for k in range(5)],
                conflicting_decisions_on_negatives=float((dec.any(0) & ~dec.all(0))[neg].mean()),
                mean_per_frame_std=float(P_all.std(0).mean()))


def train():
    import torch
    from recal_v3 import _fit, _row
    from recal_predicate import fail_terminal_mask
    from recal_v4_t3 import _threshold
    from lingbot_predicate import Net, predict
    E = P._registry(5); fold_of = {e["id"]: e["fold"] for e in E}
    st = np.load(f"{HN}/hn_state.npz")
    X, Y, M, H, EP = [], [], [], [], []
    for e in E:
        z = np.load(f"{D}/episodes/{e['id']}.npz"); X.append(z["frames"]); Y.append(z["labels"]); M.append(fail_terminal_mask(z["labels"]))
        H.append(st[f"{e['id']}__hn"] if f"{e['id']}__hn" in st.files else np.zeros(len(z["labels"]), bool)); EP += [e["id"]] * len(z["labels"])
    X = np.concatenate(X); Y = np.concatenate(Y).astype(bool); M = np.concatenate(M); H = np.concatenate(H); EP = np.array(EP)
    assert not (H & Y).any()
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
    stab = {}
    for name, fit in (("old", lambda Xa, Ya, Ma, Ha, seed: _fit(Xa, Ya, Ma, seed)), ("hn", _fit_hn)):
        oof = np.zeros(len(Y)); oof_rt = np.zeros(len(rt_lab)); oof_gn = np.zeros(len(yg)); P_all = np.zeros((6, len(yg))); log = []
        for k in range(5):
            tr = FO != k
            net, loss = fit(X[tr], Y[tr], M[tr], H[tr], k)
            oof[~tr] = predict(net, X[~tr]); oof_rt[fo_rt == k] = predict(net, rt["frames"][fo_rt == k])
            P_all[k] = predict(net, gn["frames"]); oof_gn[fo_gn == k] = P_all[k][fo_gn == k]
            log.append(dict(fold=k, final_loss=loss, oof_std=float(oof[~tr].std()))); print(name, log[-1], flush=True)
            del net; torch.cuda.empty_cache()
        net, _ = fit(X, Y, M, H, 100); P_all[5] = predict(net, gn["frames"])
        if name == "old":
            cal_old = json.load(open(f"{D}/calibration_success.json")); thr_old = cal_old["threshold"]
            same = json.dumps(_row(oof_gn[~win], yg[~win], gn_ft[~win], thr_old), sort_keys=True) == json.dumps(cal_old["generated_timing_rule"], sort_keys=True)
            stab["old"] = dict(reproduced_stored_generated_row=same, threshold=thr_old, **_stability(P_all, thr_old, ~yg & ~win))
            print("old", json.dumps(stab["old"]), flush=True)
            del net; torch.cuda.empty_cache()
            continue
        if min(l["oof_std"] for l in log) < 0.05:
            raise SystemExit("training collapsed -- stop and report")
        thr, runs = _threshold(oof, Y, M, None)
        t_ = thr if thr is not None else 0.5
        stab["hn"] = dict(threshold=thr, **_stability(P_all, t_, ~yg & ~win))
        unstable = stab["hn"]["conflicting_decisions_on_negatives"] > UNSTABLE
        hn_frames = H & ~Y
        res = dict(detector="success", task=5, retrain="hard-negative stratum (ruling 2026-10-24)", threshold=thr, feasible_runs=runs,
                   hard_negative=dict(definition=f"never-successful episode, book not in box, grip site in the box footprint +- {HN_XY_MARGIN} m "
                                                 f"and 0..{HN_Z_ABOVE} m above the box top", frames=int(hn_frames.sum()), per_epoch=N_HN,
                                      oof_tnr=float(1 - (oof[hn_frames] >= t_).mean())),
                   fold_log=log, holdout_real=_row(oof, Y, M, t_), holdout_vae_roundtrip=_row(oof_rt, rt_lab, rt_ft, t_),
                   generated_raw=_row(oof_gn, yg, gn_ft, t_), generated_timing_rule=_row(oof_gn[~win], yg[~win], gn_ft[~win], t_),
                   generated_timing_rule_deployed=_row(P_all[5][~win], yg[~win], gn_ft[~win], t_), window_steps=P.WINDOW,
                   window_negatives=dict(n=int(win.sum()), read_as_success=int(((oof_gn >= t_) & win).sum())),
                   fold_stability=stab, split_sensitive=unstable,
                   generated_row_for_gate="generated_timing_rule")      # always out-of-fold (ruling correction 2026-10-24)
        rows = ("holdout_real", "holdout_vae_roundtrip", "generated_timing_rule")
        gate = {r_: bool(thr is not None and res[r_]["tpr"] >= P.GATE and res[r_]["tnr"] >= P.GATE
                         and (res[r_]["fail_terminal"] is None or res[r_]["fail_terminal"]["tnr"] >= P.GATE)) for r_ in rows}
        res["gate_85"] = gate; res["gate_pass"] = bool(thr is not None and all(gate.values()))
        p = f"{HN}/success.pt"; torch.save(net.state_dict(), p)
        res["artifact"] = dict(path=os.path.relpath(p, ROOT), sha256=hashlib.sha256(open(p, "rb").read()).hexdigest())
        json.dump(res, open(f"{HN}/calibration_success.json", "w"), indent=1)
        json.dump(stab, open(f"{HN}/fold_stability.json", "w"), indent=1)
        summ = {"success": dict(threshold=thr, gate_pass=res["gate_pass"], gate=gate, split_sensitive=unstable,
                                rows={r_: {x: res[r_][x] for x in ("n", "tpr", "tnr")} for r_ in
                                      ("holdout_real", "holdout_vae_roundtrip", "generated_raw", "generated_timing_rule", "generated_timing_rule_deployed")})}
        json.dump(summ, open(f"{HN}/calibration_summary.json", "w"), indent=1)
        print("RESULT", json.dumps(summ), "hard negatives", res["hard_negative"], flush=True)


if __name__ == "__main__":
    {"hn_state": hn_state, "train": train}[sys.argv[1]]()
