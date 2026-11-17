"""LIBERO batch, LingBot-VA: predicate re-calibration, the ONE rerun (ruling of 2026-10-15, option 4 + 4 conditions).

Changes vs v2 (recal_predicate.py, failed the failure-terminal gate):
  1 threshold rule (fixed, condition 1): on out-of-fold REAL predictions, a threshold is feasible iff TPR >= 0.85,
    TNR >= 0.85 and failure-terminal TNR >= 0.85 all hold. Grid p = sigmoid(z), z in [-12, 12] step 0.05; the
    threshold is the logit-space midpoint of the LARGEST contiguous feasible run (never an endpoint). No feasible
    threshold -> predicate failure -> stop (no relaxing either side).
  2 near-miss negatives (condition 2): NEW LingBot closed-loop rollouts from non-anchor inits with new seeds (master
    20261029; isolated from the frozen anchor bank), each replayed open loop 3x with the task object displaced
    2-4 cm (random direction) after reset + 10 no-op settle steps. Provenance (init, env seed, policy seed,
    displacement, replay seed) is stored per episode and in calibration.json. Base rollouts are kept as episodes too.
  3 k-fold by episode (condition 3): K = 5, folds stratified by source (seeded shuffle). Every derived frame
    (VAE round trip, one-chunk generation) belongs to its source episode's fold. All gate numbers are out-of-fold.
    The deployed predicate is refit on all episodes with the same recipe; the threshold from step 1 is applied to it.
  4 one rerun only (condition 4).
Episodes: v1 closed-loop rollouts (out/predicate/t<task>/rollouts), v2 failure episodes (out/predicate_v2/t<task>/
fail_eps: fs_replay, pert_gain, pert_disp, rand_inject, pure_random) and the new near-miss episodes. Labels = simulator
check_success per frame. Failure-terminal subset = second half of episodes that never succeed.
Rows (gate 0.85 on TPR, TNR and failure-terminal TNR each): holdout_real (OOF), holdout_vae_roundtrip (every 6th frame
of every episode, OOF), model_generated_1chunk (3 starts per episode, 12 frames each, labelled by simulating; not for
pert_gain, whose dynamics are perturbed). Monitor: v1 full-rollout drift (final model).

  python recal_v3.py data <task_id>      # near-miss episodes + derived rows
  python recal_v3.py train <task_id>     # k-fold, threshold, refit, gate
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
import torch.nn.functional as F

_wmvp.bootstrap()
from lingbot_predicate import Net, predict, metrics, model_view, batched, OUT as V1_OUT, BANK
from recal_predicate import closed_episode, fail_terminal_mask, OUT as V2_OUT

ROOT = _wmvp.P("$WMV_ROOT")
OUT = f"{ROOT}/libero/out/predicate_v3"
MASTER = 20261029
K = 5
N_NM_BASE = 12
N_NM_DISP = 3
RT_EVERY = 6
GATE = 0.85


def _anchor_tpl(task_id):
    bank = json.load(open(BANK))
    return {a["anchor_id"]: a for a in bank["anchors"]}, [a for a in bank["anchors"] if a["task_id"] == task_id][0], \
        {a["init_index"] for a in bank["anchors"] if a["task_id"] == task_id}


def _displace_fixed(u, obj, off):
    sim = u._env.sim
    adr = sim.model.jnt_qposadr[sim.model.joint_name2id(f"{obj}_joint0")]
    sim.data.qpos[adr:adr + 2] += np.asarray(off)
    sim.forward()


def restore(rig, anc, disp=None):
    """fresh env (anchor-bank recipe) [+ fixed object displacement + 10 no-op settle steps]. Returns env, u, obs."""
    env, obs = rig.make_env(anc); u = env.envs[0]
    if disp:
        _displace_fixed(u, disp["object"], disp["offset_xy_m"])
        for _ in range(10):
            o, *_ = u.step(np.array([0, 0, 0, 0, 0, 0, -1], np.float32))
        obs = batched(o)
    return env, u, obs


# --------------------------------------------------------------------------------------------- episode registry
def registry(task_id):
    """all episodes with source, frames path, restore info; folds stratified by source."""
    anchors, tpl, _ = _anchor_tpl(task_id)
    E = []
    for f in sorted(os.listdir(f"{V1_OUT}/t{task_id}/rollouts")):
        z = np.load(f"{V1_OUT}/t{task_id}/rollouts/{f}")
        E.append(dict(id=f"v1_{f[:-4]}", source="v1_rollout", path=f"{V1_OUT}/t{task_id}/rollouts/{f}",
                      anc=dict(tpl, init_index=int(z["init_index"]), seed=int(z["seed"])), disp=None, gen_ok=True))
    for f in sorted(os.listdir(f"{V2_OUT}/t{task_id}/fail_eps")):
        z = np.load(f"{V2_OUT}/t{task_id}/fail_eps/{f}"); meta = json.loads(str(z["meta"]))
        src = meta["source"]
        anc = anchors[meta["anchor"]] if src == "fs_replay" else dict(tpl, init_index=meta["init_index"], seed=meta["seed"])
        disp = meta["spec"].get("displacement")
        E.append(dict(id=f"v2_{f[:-4]}", source=src, path=f"{V2_OUT}/t{task_id}/fail_eps/{f}", anc=anc, disp=disp,
                      gen_ok=(src != "pert_gain")))
    d = f"{OUT}/t{task_id}/nearmiss_eps"
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            z = np.load(f"{d}/{f}"); meta = json.loads(str(z["meta"]))
            E.append(dict(id=f"v3_{f[:-4]}", source=meta["source"], path=f"{d}/{f}",
                          anc=dict(tpl, init_index=meta["init_index"], seed=meta["seed"]), disp=meta.get("displacement"),
                          gen_ok=True))
    rs = np.random.RandomState(MASTER)
    for src in sorted({e["source"] for e in E}):
        S = [e for e in E if e["source"] == src]
        order = rs.permutation(len(S))
        for rank, i in enumerate(order):
            S[i]["fold"] = int(rank % K)
    return E


# --------------------------------------------------------------------------------------------- data
def gen_data(task_id):
    from lingbot_unit import LingBotRig
    rig = LingBotRig(task_id)
    anchors, tpl, used = _anchor_tpl(task_id)
    pool = [i for i in range(50) if i not in used]
    d = f"{OUT}/t{task_id}"; os.makedirs(f"{d}/nearmiss_eps", exist_ok=True)
    rs = np.random.RandomState(MASTER + task_id)
    obj = None
    for i in range(N_NM_BASE):
        init = int(pool[rs.randint(len(pool))]); seed = int(rs.randint(2 ** 31 - 1)); pseed = int(rs.randint(2 ** 31 - 1))
        rseeds = [int(rs.randint(2 ** 31 - 1)) for _ in range(N_NM_DISP)]
        anc = dict(tpl, init_index=init, seed=seed)
        bpath = f"{d}/nearmiss_eps/nm_base_{i:02d}.npz"
        if not os.path.exists(bpath):
            fr, lab, acts, _ = closed_episode(rig, anc, pseed, np.random.RandomState(pseed))
            np.savez_compressed(bpath, frames=fr, labels=lab, actions=acts, meta=json.dumps(
                dict(source="nm_base", init_index=init, seed=seed, policy_seed=pseed, displacement=None,
                     note="LingBot closed loop, unperturbed; base of the displaced open-loop replays")))
            print(f"t{task_id} nm_base_{i:02d}: T={len(acts)} success={bool(lab.any())}", flush=True)
        acts = np.load(bpath)["actions"]
        for j, rsd in enumerate(rseeds):
            path = f"{d}/nearmiss_eps/nm_disp_{i:02d}_{j}.npz"
            if os.path.exists(path):
                continue
            r_ = np.random.RandomState(rsd)
            ang = r_.uniform(0, 2 * np.pi); rad = r_.uniform(0.02, 0.04)
            env, u, obs = restore(rig, anc)
            obj = obj or u._env.obj_of_interest[0]
            env.close()
            disp = dict(object=obj, offset_xy_m=[round(rad * np.cos(ang), 4), round(rad * np.sin(ang), 4)])
            env, u, obs = restore(rig, anc, disp)
            frames, labels = [model_view(rig, rig.obs_to_batch(obs))], [bool(u._env.check_success())]
            for a in acts:
                o, *_ = u.step(np.asarray(a, np.float32))
                frames.append(model_view(rig, rig.obs_to_batch(batched(o)))); labels.append(bool(u._env.check_success()))
            env.close()
            lab = np.array(labels)
            np.savez_compressed(path, frames=np.stack(frames), labels=lab, actions=acts, meta=json.dumps(
                dict(source="nm_disp", init_index=init, seed=seed, policy_seed=pseed, replay_seed=rsd,
                     base=f"nm_base_{i:02d}", displacement=disp,
                     note="open-loop replay of the base rollout's executed actions, object displaced after reset")))
            print(f"t{task_id} nm_disp_{i:02d}_{j}: disp={disp['offset_xy_m']} success={bool(lab.any())} "
                  f"fail_terminal={int(fail_terminal_mask(lab).sum())}", flush=True)
    E = registry(task_id)
    # VAE round trip: every RT_EVERY-th frame of every episode
    if not os.path.exists(f"{d}/roundtrip_all.npz"):
        p = rig.policy; p._ensure_frozen_modules()
        RT, RL, RF, RE = [], [], [], []
        _im = torch.inference_mode(); _im.__enter__()
        for e in E:
            z = np.load(e["path"]); m = fail_terminal_mask(z["labels"])
            for i in range(0, len(z["frames"]), RT_EVERY):
                fr = torch.from_numpy(z["frames"][i]).float().div(255)
                halves = fr.permute(2, 0, 1).split(fr.shape[1] // 2, dim=-1)
                vids = torch.stack([h * 2 - 1 for h in halves])[:, :, None].to(p.config.device).to(p.dtype)
                p._frozen["streaming_vae"].clear_cache()
                enc = p._streaming_vae.encode_chunk(vids.to(next(p._vae.parameters()).device))
                lat = torch.cat(p._normalize_vae_latent(enc).split(1, dim=0), dim=-1)
                RT.append(rig.decode(lat)[0].numpy()); RL.append(bool(z["labels"][i])); RF.append(bool(m[i])); RE.append(e["id"])
        p._frozen["streaming_vae"].clear_cache(); _im.__exit__(None, None, None)
        np.savez_compressed(f"{d}/roundtrip_all.npz", frames=np.stack(RT), labels=np.array(RL), fail_terminal=np.array(RF),
                            episode=np.array(RE))
        print(f"t{task_id} roundtrip_all {len(RT)} fail_terminal {int(np.sum(RF))}", flush=True)
    # one-chunk generation: 3 starts per episode
    if not os.path.exists(f"{d}/gen_all.npz"):
        GF, GL, GM, GE = [], [], [], []
        grs = np.random.RandomState(MASTER + 100 + task_id)
        for e in E:
            if not e["gen_ok"]:
                continue
            z = np.load(e["path"]); acts = z["actions"]; lab = z["labels"]; n = len(acts); failed = not lab.any()
            if failed:
                starts = np.linspace(n // 2, n - 13, 3).astype(int)
            else:
                fs = int(np.argmax(lab))
                starts = np.clip(np.array([fs - 36, fs - 12, fs + 12]), 1, n - 13)
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
                labs = []
                for a in ea:
                    u.step(a.astype(np.float32)); labs.append(bool(u._env.check_success()))
                env.close()
                for k in range(1, min(len(frames), len(labs) + 1)):
                    GF.append(frames[k]); GL.append(labs[k - 1]); GM.append(failed and t0 >= n // 2); GE.append(e["id"])
        np.savez_compressed(f"{d}/gen_all.npz", frames=np.stack(GF), labels=np.array(GL), fail_terminal=np.array(GM),
                            episode=np.array(GE))
        print(f"t{task_id} gen_all {len(GF)} positives {int(np.sum(GL))} fail_terminal {int(np.sum(GM))}", flush=True)


# --------------------------------------------------------------------------------------------- training
def _fit(X, Y, M, seed):
    torch.manual_seed(seed)
    net = Net().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    pos = np.where(Y)[0]; negf = np.where(~Y & M)[0]; nego = np.where(~Y & ~M)[0]
    rs = np.random.RandomState(seed)
    for ep in range(25):
        net.train()
        idx = np.concatenate([rs.choice(pos, 3000), rs.choice(negf, 1500), rs.choice(nego, 1500)]); rs.shuffle(idx)
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            loss = F.binary_cross_entropy_with_logits(net(torch.from_numpy(X[b]).permute(0, 3, 1, 2).float().div(255).cuda()),
                                                      torch.from_numpy(Y[b].astype(np.float32)).cuda())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval()
    return net, float(loss)


def _row(p, y, ft, thr, src=None):
    out = metrics(p, y, thr)
    out["fail_terminal"] = metrics(p[ft & ~y], y[ft & ~y], thr) if (ft & ~y).any() else None
    if src is not None:
        out["fail_terminal_by_source"] = {s: metrics(p[(src == s) & ft & ~y], y[(src == s) & ft & ~y], thr)
                                          for s in sorted(set(src[ft & ~y]))}
    return out


def train(task_id):
    d = f"{OUT}/t{task_id}"
    E = registry(task_id); fold_of = {e["id"]: e["fold"] for e in E}; src_of = {e["id"]: e["source"] for e in E}
    X, Y, M, EP = [], [], [], []
    for e in E:
        z = np.load(e["path"]); X.append(z["frames"]); Y.append(z["labels"]); M.append(fail_terminal_mask(z["labels"]))
        EP += [e["id"]] * len(z["labels"])
    X = np.concatenate(X); Y = np.concatenate(Y).astype(bool); M = np.concatenate(M); EP = np.array(EP)
    FO = np.array([fold_of[x] for x in EP]); SRC = np.array([src_of[x] for x in EP])
    rt = np.load(f"{d}/roundtrip_all.npz"); gn = np.load(f"{d}/gen_all.npz")
    oof = np.zeros(len(Y)); oof_rt = np.zeros(len(rt["labels"])); oof_gn = np.zeros(len(gn["labels"]))
    fo_rt = np.array([fold_of[x] for x in rt["episode"]]); fo_gn = np.array([fold_of[x] for x in gn["episode"]])
    log = []
    for k in range(K):
        tr = FO != k
        net, loss = _fit(X[tr], Y[tr], M[tr], seed=k)
        oof[~tr] = predict(net, X[~tr])
        oof_rt[fo_rt == k] = predict(net, rt["frames"][fo_rt == k])
        oof_gn[fo_gn == k] = predict(net, gn["frames"][fo_gn == k])
        log.append(dict(fold=k, n_train=int(tr.sum()), n_test=int((~tr).sum()), final_loss=loss,
                        oof_prob_std=float(oof[~tr].std())))
        print(log[-1], flush=True)
        del net; torch.cuda.empty_cache()
    if min(l["oof_prob_std"] for l in log) < 0.05:
        raise SystemExit("predicate training collapsed in a fold -- stop and report")
    # threshold rule (condition 1): feasible = TPR, TNR, failure-terminal TNR all >= 0.85 on OOF real predictions
    zs = np.arange(-12, 12.0001, 0.05); ps = 1 / (1 + np.exp(-zs))
    fneg = M & ~Y
    feas = []
    for p_ in ps:
        yh = oof >= p_
        feas.append(bool(yh[Y].mean() >= GATE and (~yh[~Y]).mean() >= GATE and (~yh[fneg]).mean() >= GATE))
    feas = np.array(feas)
    runs, cur = [], None
    for i, f in enumerate(feas):
        if f and cur is None:
            cur = i
        if (not f or i == len(feas) - 1) and cur is not None:
            runs.append((cur, i if f else i - 1)); cur = None
    res = dict(task_id=task_id, version="v3 (k-fold by episode, near-miss negatives, fixed threshold rule)",
               k_folds=K, fold_log=log, threshold_rule="logit midpoint of the largest contiguous run of p = sigmoid(z), "
               "z in [-12, 12] step 0.05, where OOF real TPR, TNR and failure-terminal TNR are all >= 0.85",
               feasible_runs=[[float(ps[a]), float(ps[b])] for a, b in runs])
    if not runs:
        res["threshold"] = None; res["gate_pass"] = False
        res["stop"] = "no threshold satisfies TPR, TNR and failure-terminal TNR >= 0.85 simultaneously (predicate failure)"
        thr_report = float(ps[int(np.argmax([0.5 * ((oof[Y] >= p_).mean() + (oof[~Y] < p_).mean()) for p_ in ps]))])
        res["diagnostic_threshold_max_balanced_acc"] = thr_report
        thr = thr_report
    else:
        a, b = max(runs, key=lambda r: r[1] - r[0])
        thr = float(1 / (1 + np.exp(-(zs[a] + zs[b]) / 2)))
        res["threshold"] = thr; res["feasible_run_used"] = [float(ps[a]), float(ps[b])]
    res["holdout_real"] = _row(oof, Y, M, thr, SRC)
    res["holdout_vae_roundtrip"] = _row(oof_rt, rt["labels"].astype(bool), rt["fail_terminal"],
                                        thr, np.array([src_of[x] for x in rt["episode"]]))
    res["model_generated_1chunk"] = _row(oof_gn, gn["labels"].astype(bool), gn["fail_terminal"],
                                         thr, np.array([src_of[x] for x in gn["episode"]]))
    # hardest failure-terminal negatives (OOF), for the stop report
    hard = np.argsort(-np.where(fneg, oof, -1))[:12]
    res["hardest_fail_terminal"] = [dict(episode=str(EP[i]), source=str(SRC[i]),
                                         frame=int(i - np.where(EP == EP[i])[0][0]), p=float(oof[i])) for i in hard]
    np.savez_compressed(f"{d}/oof.npz", oof=oof, labels=Y, fail_terminal=M, episode=EP, oof_rt=oof_rt, oof_gn=oof_gn)
    gate = {}
    for k_ in ("holdout_real", "holdout_vae_roundtrip", "model_generated_1chunk"):
        r = res[k_]; ft = r["fail_terminal"]
        gate[k_] = bool(r["tpr"] is not None and r["tpr"] >= GATE and r["tnr"] >= GATE)
        gate[k_ + "/fail_terminal_tnr"] = bool(ft is not None and ft["tnr"] >= GATE)
    res["gate_85"] = gate
    res["gate_pass"] = bool(res["threshold"] is not None and all(gate.values()))
    # deployed predicate: refit on all episodes (same recipe), threshold from the OOF rule
    net, loss = _fit(X, Y, M, seed=100)
    path = f"{d}/predicate.pt"; torch.save(net.state_dict(), path)
    res["artifact"] = dict(path=os.path.relpath(path, ROOT), sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
                           note="refit on all episodes; gate numbers are out-of-fold (k-fold by episode)")
    dz = np.load(f"{V1_OUT}/t{task_id}/drift.npz")
    res["model_generated_full_rollout_monitor"] = metrics(predict(net, dz["frames"]), dz["labels"].astype(bool), thr)
    res["composition"] = dict(
        episodes={s: sum(e["source"] == s for e in E) for s in sorted({e["source"] for e in E})},
        positive_frames=int(Y.sum()), fail_terminal_frames=int(fneg.sum()), other_negative_frames=int((~Y & ~M).sum()),
        folds={e["id"]: e["fold"] for e in E},
        provenance_nearmiss=[json.loads(str(np.load(e["path"])["meta"])) for e in E if e["id"].startswith("v3_")])
    json.dump(res, open(f"{d}/calibration.json", "w"), indent=1)
    print("RESULT", json.dumps({k_: res.get(k_) for k_ in ("threshold", "feasible_runs", "gate_85", "gate_pass")}))
    for k_ in ("holdout_real", "holdout_vae_roundtrip", "model_generated_1chunk"):
        r = res[k_]
        print(k_, {x: r[x] for x in ("n", "tpr", "tnr")}, "FT", r["fail_terminal"] and r["fail_terminal"]["tnr"],
              {s: round(v["tnr"], 3) for s, v in (r.get("fail_terminal_by_source") or {}).items()}, flush=True)


if __name__ == "__main__":
    (gen_data if sys.argv[1] == "data" else train)(int(sys.argv[2]))
